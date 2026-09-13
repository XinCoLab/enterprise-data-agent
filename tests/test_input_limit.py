import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agent_runtime import agent_runtime
from agent_runtime.input_limit import (
    MAX_INPUT_TOKENS,
    InputBudgetExceeded,
    estimate_input_tokens,
)
from api.schemas import ChatRequest
from graph.graph_state import GraphState
from graph.nodes import main_agent_llm_node as node_module
from memory import conversation_history_database
from security.workspace_access import resolve_current_user


MODEL = "deepseek-v4-flash"
TOOL_DEFINITIONS = [{
    "type": "function",
    "function": {
        "name": "execute_readonly_sql",
        "description": "Run a read-only query.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
    },
}]


def tool_call():
    return AIMessage(
        content="",
        tool_calls=[{
            "id": "call-budget",
            "name": "execute_readonly_sql",
            "args": {"sql": "SELECT 1"},
        }],
        additional_kwargs={
            "reasoning_content": "Need a query result.",
            "tool_safety_decisions": [{"decision": "ALLOW"}],
        },
    )


@pytest.fixture
def fake_model(monkeypatch):
    class FakeModel:
        kwargs = {"tools": deepcopy(TOOL_DEFINITIONS)}

        def __init__(self):
            self.inputs = []
            self.output = AIMessage(content="Answer")

        async def ainvoke(self, messages):
            self.inputs.append(messages)
            return self.output

    model = FakeModel()
    monkeypatch.setattr(node_module, "_model_with_tools", lambda _name: model)
    monkeypatch.setattr(node_module, "get_main_llm", Mock(
        side_effect=AssertionError("Budget estimation must not obtain a model client")
    ))
    return model


def test_estimate_includes_all_context_sources_but_not_local_metadata():
    messages = [
        SystemMessage(content="System prompt and runtime context"),
        HumanMessage(content="Question"),
        tool_call(),
        ToolMessage(content="Result", tool_call_id="call-budget", name="execute_readonly_sql"),
    ]

    def estimate(items, definitions):
        return estimate_input_tokens(
            messages=items, tool_definitions=definitions,
        )

    baseline = estimate(messages, TOOL_DEFINITIONS)
    for index in (0, 1, 3):
        changed = deepcopy(messages)
        changed[index].content += "x" * 4000
        assert estimate(changed, TOOL_DEFINITIONS) > baseline

    changed = deepcopy(messages)
    changed[2].tool_calls[0]["args"]["sql"] += "x" * 4000
    assert estimate(changed, TOOL_DEFINITIONS) > baseline

    changed = deepcopy(messages)
    changed[2].additional_kwargs["reasoning_content"] += "x" * 4000
    assert estimate(changed, TOOL_DEFINITIONS) > baseline

    definitions = deepcopy(TOOL_DEFINITIONS)
    definitions[0]["function"]["description"] += "x" * 4000
    assert estimate(messages, definitions) > baseline

    changed = deepcopy(messages)
    changed[2].additional_kwargs["tool_safety_decisions"] = [{"internal": "x" * 4000}]
    changed[2].response_metadata = {"internal": "x" * 4000}
    assert estimate(changed, TOOL_DEFINITIONS) == baseline


@pytest.mark.parametrize("estimated_tokens", [MAX_INPUT_TOKENS - 1, MAX_INPUT_TOKENS, MAX_INPUT_TOKENS + 1])
def test_budget_boundary_runs_after_assembly_and_before_model_call(
    monkeypatch, fake_model, estimated_tokens,
):
    captured = {}

    def fake_estimate(*, messages, tool_definitions):
        captured.update(messages=messages, tools=tool_definitions)
        return estimated_tokens

    monkeypatch.setattr(node_module, "estimate_input_tokens", fake_estimate)
    state = {"messages": [
        HumanMessage(content="Question"),
        tool_call(),
        ToolMessage(content="Result", tool_call_id="call-budget"),
    ]}
    before = deepcopy(state)
    invocation = node_module.main_agent_llm_node(
        state, {"configurable": {"model": MODEL}}
    )
    if estimated_tokens > MAX_INPUT_TOKENS:
        with pytest.raises(InputBudgetExceeded) as raised:
            asyncio.run(invocation)
        assert raised.value.code == "LOCAL_INPUT_BUDGET_EXCEEDED"
        assert raised.value.estimated_input_tokens == estimated_tokens
        assert raised.value.max_input_tokens == MAX_INPUT_TOKENS
        assert fake_model.inputs == []
    else:
        result = asyncio.run(invocation)
        assert result["messages"][-1].content == "Answer"
        assert len(fake_model.inputs) == 1

    assert state == before
    assert all(isinstance(message, SystemMessage) for message in captured["messages"][:4])
    assert captured["tools"] == TOOL_DEFINITIONS
    assert captured["messages"][-2].additional_kwargs["reasoning_content"] == "Need a query result."
    assert captured["messages"][-1].tool_call_id == "call-budget"
    if fake_model.inputs:
        assert captured["messages"] is fake_model.inputs[0]


def test_large_tool_result_fails_next_call_and_remains_in_sqlite(
    tmp_path, monkeypatch, fake_model,
):
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", tmp_path / "settings.env")
    monkeypatch.setattr(agent_runtime, "ACTIVE_PROFILE_PATH", tmp_path / ".active_profile")
    summary = AsyncMock(side_effect=AssertionError("Budget failure must not run a summary"))
    monkeypatch.setattr(agent_runtime, "generate_round_limit_summary", summary)
    fake_model.output = tool_call()
    tool_executions = []
    large_tool_result = "x" * (MAX_INPUT_TOKENS * 4 + 4)

    async def fake_tool_node(state):
        call = state["messages"][-1].tool_calls[0]
        tool_executions.append(call["id"])
        return {"messages": [ToolMessage(
            content=large_tool_result, name=call["name"], tool_call_id=call["id"],
        )]}

    builder = StateGraph(GraphState)
    builder.add_node("Main Agent LLM", node_module.main_agent_llm_node)
    builder.add_node("Tool Execution", fake_tool_node)
    builder.add_edge(START, "Main Agent LLM")
    builder.add_edge("Main Agent LLM", "Tool Execution")
    builder.add_edge("Tool Execution", END)
    request = ChatRequest(question="Count records", thread_id="input-budget", model=MODEL)
    current_user = resolve_current_user(None)
    thread_id, run_config = agent_runtime.build_agent_config(request, current_user)
    database_path = str(tmp_path / "budget-checkpoints.sqlite")

    async def run():
        async with AsyncSqliteSaver.from_conn_string(database_path) as saver:
            graph = builder.compile(checkpointer=saver)
            events = [json.loads(event) async for event in agent_runtime.stream_chat_response(
                request, current_user, single_round_graph=graph,
                thread_id=thread_id, run_config=run_config,
            )]
        # Reopen SQLite to check that the completed tool result survives the failure.
        async with AsyncSqliteSaver.from_conn_string(database_path) as saver:
            snapshot = await builder.compile(checkpointer=saver).aget_state(run_config)
        return events, snapshot.values["messages"]

    events, messages = asyncio.run(run())
    assert events[-1]["type"] == "error"
    assert "LOCAL_INPUT_BUDGET_EXCEEDED" in events[-1]["message"]
    assert not any(event["type"] == "final" for event in events)
    assert len(fake_model.inputs) == 1
    assert tool_executions == ["call-budget"]
    summary.assert_not_called()
    assert [type(message) for message in messages] == [HumanMessage, AIMessage, ToolMessage]
    assert messages[-1].content == large_tool_result
    assert agent_runtime.classify_tool_message_sequence(messages) == "complete"
    request_id = run_config["configurable"]["agent_run_context"].request_id
    assert request_id not in agent_runtime.ACTIVE_RUNS
    history = conversation_history_database.read_conversation_info(thread_id, current_user.workspace_id)
    assert [message["role"] for message in history["messages"]] == ["user"]
