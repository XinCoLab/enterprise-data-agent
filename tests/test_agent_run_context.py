import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from api.schemas import ChatRequest
from graph.nodes.main_agent_llm_node import selected_model_name
from graph.nodes.tool_execution_node import tool_execution_node
from agent_runtime import agent_runtime
from agent_runtime.agent_run_context import AgentRunContext, read_agent_run_context
from tools.tool_safety import ALLOW
from security.workspace_access import resolve_current_user
from tools.tool_registry import TOOLS_BY_NAME


def make_context(
    request_id: str,
    data_source_id: str,
) -> AgentRunContext:
    return AgentRunContext(
        request_id=request_id,
        thread_id=f"thread-{request_id}",
        workspace_id=f"workspace-{request_id}",
        user_id=f"user-{request_id}",
        model="deepseek-v4-pro",
        permissions=frozenset({"chat:run"}),
        allowed_data_source_ids=(data_source_id,),
        selected_data_source_ids=(data_source_id,),
    )


def test_agent_run_context_is_immutable():
    context = make_context("run-a", "database-a")

    with pytest.raises(FrozenInstanceError):
        context.request_id = "changed-request"

    assert isinstance(context.permissions, frozenset)
    assert isinstance(context.allowed_data_source_ids, tuple)
    assert isinstance(context.selected_data_source_ids, tuple)


def test_build_agent_config_snapshots_each_run_resources(tmp_path, monkeypatch):
    active_profile_path = tmp_path / ".active_profile"
    monkeypatch.setattr(agent_runtime, "ACTIVE_PROFILE_PATH", active_profile_path)
    request = ChatRequest(question="测试 Run Context", thread_id="context-thread")
    current_user = resolve_current_user("admin-a")

    active_profile_path.write_text("database-a\n", encoding="utf-8")
    _thread_a, config_a = agent_runtime.build_agent_config(request, current_user)
    context_a = read_agent_run_context(config_a)

    active_profile_path.write_text("database-b\n", encoding="utf-8")
    _thread_b, config_b = agent_runtime.build_agent_config(request, current_user)
    context_b = read_agent_run_context(config_b)

    assert context_a is not None
    assert context_b is not None
    assert context_a.request_id != context_b.request_id
    assert context_a.selected_data_source_ids == ("database-a",)
    assert context_b.selected_data_source_ids == ("database-b",)
    assert context_a.selected_data_source_ids == ("database-a",)

    # Keep the old LangGraph keys until every caller has migrated.
    assert config_a["configurable"]["thread_id"] == "context-thread"
    assert config_a["configurable"]["workspace_id"] == "workspace-a"
    assert config_a["configurable"]["user_id"] == "user-admin-a"


def test_run_refuses_a_resource_that_changed_after_context_creation(
    tmp_path,
    monkeypatch,
):
    active_profile_path = tmp_path / ".active_profile"
    active_profile_path.write_text("database-a\n", encoding="utf-8")
    monkeypatch.setattr(agent_runtime, "ACTIVE_PROFILE_PATH", active_profile_path)
    current_user = resolve_current_user("admin-a")
    _thread_id, config = agent_runtime.build_agent_config(
        ChatRequest(question="资源快照", thread_id="snapshot-thread"),
        current_user,
    )
    context = read_agent_run_context(config)
    assert context is not None

    active_profile_path.write_text("database-b\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="data source changed"):
        agent_runtime.confirm_run_resources_are_unchanged(context, current_user)


def test_main_agent_node_reads_model_from_run_context():
    context = make_context("run-model", "database-model")
    config = {
        "configurable": {
            "model": "deepseek-v4-flash",
            "agent_run_context": context,
        }
    }

    assert selected_model_name(context, config) == "deepseek-v4-pro"


def test_tool_receives_the_correct_context_for_parallel_runs(monkeypatch):
    @tool("context_probe")
    def context_probe(config: RunnableConfig) -> str:
        """Return the immutable Run Context received by this test tool."""

        context = read_agent_run_context(config)
        assert context is not None
        return json.dumps(
            {
                "request_id": context.request_id,
                "selected_data_source_ids": context.selected_data_source_ids,
            }
        )

    monkeypatch.setitem(TOOLS_BY_NAME, "context_probe", context_probe)

    def invoke_probe(context: AgentRunContext) -> dict:
        tool_call_id = f"call-{context.request_id}"
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "context_probe",
                    "args": {},
                    "id": tool_call_id,
                }
            ],
            additional_kwargs={
                "tool_safety_decisions": [
                    {
                        "tool_call_id": tool_call_id,
                        "decision": ALLOW,
                    }
                ]
            },
        )
        update = asyncio.run(
            tool_execution_node(
                {"messages": [message]},
                config={"configurable": {"agent_run_context": context}},
            )
        )
        return json.loads(update["messages"][0].content)

    context_a = make_context("run-a", "database-a")
    context_b = make_context("run-b", "database-b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        result_a, result_b = executor.map(invoke_probe, (context_a, context_b))

    assert result_a == {
        "request_id": "run-a",
        "selected_data_source_ids": ["database-a"],
    }
    assert result_b == {
        "request_id": "run-b",
        "selected_data_source_ids": ["database-b"],
    }
