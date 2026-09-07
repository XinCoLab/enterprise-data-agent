import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime import agent_runtime
from agent_runtime.agent_run_context import AgentRunContext
from api.schemas import ChatRequest
from memory import conversation_history_database
from security.workspace_access import resolve_current_user


def _sql_call(call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "execute_readonly_sql",
                "args": {"sql": "SELECT COUNT(*) AS n FROM records"},
                "id": call_id,
            }
        ],
    )


def _sql_result(call_id):
    return ToolMessage(
        name="execute_readonly_sql",
        tool_call_id=call_id,
        content=(
            '{"columns":["n"],"rows":[{"n":12}],'
            '"returned_rows":1,"truncated":false}'
        ),
    )


class _LifecycleGraph:
    def __init__(self, messages=(), next_nodes=(), *, repair_succeeds=True):
        self.messages = list(messages)
        self.next_nodes = next_nodes
        self.repair_succeeds = repair_succeeds
        self.inputs = []
        self.configs = []
        self.resume_calls = []
        self.completed_rounds = []

    async def aget_state(self, _config):
        return SimpleNamespace(
            values={"messages": list(self.messages)},
            next=self.next_nodes,
        )

    async def ainvoke(self, payload, *, config):
        self.resume_calls.append((payload, config))
        if self.repair_succeeds:
            self.messages.append(_sql_result("previous-sql"))
            self.next_nodes = ()

    async def astream(self, payload, *, config, **_kwargs):
        self.inputs.append(payload)
        self.configs.append(config)
        self.messages.extend(payload.get("messages", []))
        round_number = len(self.inputs)
        yield {
            "type": "tasks",
            "data": {"name": "Main Agent LLM", "input": {}},
        }

        if round_number == 1:
            call = _sql_call("current-sql")
            self.messages.append(call)
            yield {
                "type": "updates",
                "data": {"Main Agent LLM": {"messages": [call]}},
            }
            yield {
                "type": "tasks",
                "data": {"name": "Tool Safety", "input": {}},
            }
            yield {
                "type": "updates",
                "data": {"Tool Safety": {"messages": [call]}},
            }
            yield {
                "type": "tasks",
                "data": {"name": "Tool Execution", "input": {}},
            }
            result = _sql_result("current-sql")
            self.messages.append(result)
            yield {
                "type": "updates",
                "data": {"Tool Execution": {"messages": [result]}},
            }
        else:
            answer = AIMessage(content="There are 12 records.")
            self.messages.append(answer)
            yield {
                "type": "updates",
                "data": {"Main Agent LLM": {"messages": [answer]}},
            }
        self.completed_rounds.append(round_number)


@pytest.fixture
def runtime_request(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", tmp_path / "settings.env")
    monkeypatch.setattr(
        agent_runtime, "ACTIVE_PROFILE_PATH", tmp_path / ".active_profile"
    )
    request = ChatRequest(
        question="Count records",
        thread_id="request-lifecycle",
        model="deepseek-v4-pro",
    )
    current_user = resolve_current_user(None)
    thread_id, run_config = agent_runtime.build_agent_config(request, current_user)
    return request, current_user, thread_id, run_config


def _execute_request(runtime_request, graph):
    request, current_user, thread_id, run_config = runtime_request
    return agent_runtime.execute_agent_request(
        request,
        current_user,
        single_round_graph=graph,
        thread_id=thread_id,
        run_config=run_config,
    )


async def _collect(events):
    return [event async for event in events]


def test_two_round_request_saves_history_and_run_once_before_final(runtime_request):
    request, current_user, thread_id, run_config = runtime_request
    run_context = run_config["configurable"]["agent_run_context"]
    assert isinstance(run_context, AgentRunContext)
    graph = _LifecycleGraph()

    async def consume():
        events = []
        async for event in _execute_request(runtime_request, graph):
            events.append(event)
            if event["type"] == "final":
                history = conversation_history_database.read_conversation_info(
                    thread_id, current_user.workspace_id
                )
                assert history is not None
                assert [message["role"] for message in history["messages"]] == [
                    "user",
                    "assistant",
                ]
                assert history["messages"][0]["content"] == request.question
                assert history["messages"][1]["content"] == "There are 12 records."
                assert history["messages"][1]["details"] == event["response"]
                recorded = [
                    run
                    for run in agent_runtime.list_recent_runs(current_user)
                    if run["request_id"] == run_context.request_id
                ]
                assert len(recorded) == 1
                assert recorded[0]["status"] == "success"
                assert recorded[0]["thread_id"] == thread_id
        return events

    events = asyncio.run(consume())

    assert events[0]["type"] == "started"
    assert all(event["request_id"] == run_context.request_id for event in events)
    assert [event["round"] for event in events if event["type"] == "round"] == [1, 2]
    assert [event["type"] for event in events].count("final") == 1
    assert events[-1]["response"]["status"] == "success"
    assert events[-1]["response"]["result_preview"]["rows"] == [{"n": 12}]
    assert len(graph.inputs) == 2
    assert len(graph.configs) == 2
    assert all(config is run_config for config in graph.configs)
    assert graph.inputs[1] == {}
    assert len(graph.inputs[0]["messages"]) == 1
    assert isinstance(graph.inputs[0]["messages"][0], HumanMessage)
    assert graph.inputs[0]["messages"][0].content == request.question
    assert sum(isinstance(message, HumanMessage) for message in graph.messages) == 1
    assert graph.completed_rounds == [1, 2]
    assert run_context.request_id not in agent_runtime.ACTIVE_RUNS


def test_cancel_immediately_after_started_completes_first_tool_round(runtime_request):
    _, current_user, _, run_config = runtime_request
    graph = _LifecycleGraph()

    async def consume_with_cancel():
        events = _execute_request(runtime_request, graph)
        try:
            started = await anext(events)
            assert started["type"] == "started"
            assert graph.inputs == []
            cancellation = agent_runtime.request_run_cancellation(
                started["request_id"], current_user
            )
            assert cancellation["status"] == "cancel_requested"
            return [started, *[event async for event in events]]
        finally:
            await events.aclose()

    events = asyncio.run(consume_with_cancel())

    assert len(graph.inputs) == 1
    assert graph.completed_rounds == [1]
    assert isinstance(graph.messages[-1], ToolMessage)
    assert graph.messages[-1].tool_call_id == "current-sql"
    assert agent_runtime.classify_tool_message_sequence(graph.messages) == "complete"
    assert [event["round"] for event in events if event["type"] == "round"] == [1]
    assert any(event.get("tool") == "execute_readonly_sql" for event in events[:-1])
    assert events[-1]["type"] == "final"
    assert events[-1]["response"]["status"] == "canceled"
    assert events[-1]["response"]["result_preview"]["rows"] == [{"n": 12}]
    request_id = run_config["configurable"]["agent_run_context"].request_id
    assert request_id not in agent_runtime.ACTIVE_RUNS


def test_complete_checkpoint_does_not_reexecute_previous_tools(runtime_request):
    graph = _LifecycleGraph(
        [HumanMessage(content="Previous question"), _sql_call("previous-sql"),
         _sql_result("previous-sql")],
        next_nodes=("Tool Execution",),
    )

    events = asyncio.run(_collect(_execute_request(runtime_request, graph)))

    assert events[-1]["response"]["status"] == "success"
    assert graph.resume_calls == []
    assert sum(
        isinstance(message, ToolMessage) and message.tool_call_id == "previous-sql"
        for message in graph.messages
    ) == 1
    assert graph.completed_rounds == [1, 2]


@pytest.mark.parametrize("pending_node", ["Tool Safety", "Tool Execution"])
def test_pending_checkpoint_resumes_tools_before_new_question(
    runtime_request, pending_node
):
    request, _, _, run_config = runtime_request
    graph = _LifecycleGraph(
        [HumanMessage(content="Previous question"), _sql_call("previous-sql")],
        next_nodes=(pending_node,),
    )

    events = asyncio.run(_collect(_execute_request(runtime_request, graph)))

    assert graph.resume_calls == [(None, run_config)]
    assert events[-1]["response"]["status"] == "success"
    assert isinstance(graph.messages[2], ToolMessage)
    assert graph.messages[2].tool_call_id == "previous-sql"
    assert isinstance(graph.messages[3], HumanMessage)
    assert graph.messages[3].content == request.question
    assert agent_runtime.classify_tool_message_sequence(graph.messages) == "complete"


def test_unrepaired_checkpoint_rejects_new_question(runtime_request):
    _, current_user, thread_id, run_config = runtime_request
    graph = _LifecycleGraph(
        [HumanMessage(content="Previous question"), _sql_call("previous-sql")],
        next_nodes=("Tool Execution",),
        repair_succeeds=False,
    )

    with pytest.raises(RuntimeError, match="上一批工具消息不完整"):
        asyncio.run(_collect(_execute_request(runtime_request, graph)))

    assert graph.resume_calls == [(None, run_config)]
    assert graph.inputs == []
    assert len(graph.messages) == 2
    assert conversation_history_database.read_conversation_info(
        thread_id, current_user.workspace_id
    ) is None
    request_id = run_config["configurable"]["agent_run_context"].request_id
    assert request_id not in agent_runtime.ACTIVE_RUNS
