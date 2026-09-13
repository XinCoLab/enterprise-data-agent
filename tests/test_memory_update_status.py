import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_runtime.agent_runtime import build_turn_result
from agent_runtime.translate_graph_events import (
    memory_update_for_result,
    translate_graph_progress_events,
)
from graph.nodes.tool_safety_node import tool_safety_node
from memory import mem0_client


@pytest.mark.parametrize("status, expected", [
    ("PENDING", "pending"), ("RUNNING", "pending"),
    ("SUCCEEDED", "succeeded"), ("FAILED", "failed"),
    ("NO_CHANGE", "unchanged"),
    ("error", "failed"), ("denied", "failed"), ("unexpected", "unknown"),
])
def test_only_confirmed_success_displays_updated(status, expected):
    result = memory_update_for_result("memory-1", {"status": status})
    assert result == {"tool_call_id": "memory-1", "status": expected}


def test_empty_extraction_does_not_claim_updated(monkeypatch):
    monkeypatch.setattr(mem0_client, "get_mem0_client", lambda: SimpleNamespace(add=lambda **kwargs: {"results": []}))
    result = mem0_client.save_memory("重复约定", user_id="u", workspace_id="w", thread_id="t")
    assert result["status"] == "NO_CHANGE"
    assert memory_update_for_result("memory-1", result)["status"] == "unchanged"


def test_local_storage_failure_is_reported_to_the_ui(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Storage unavailable")
    monkeypatch.setattr(mem0_client, "get_mem0_client", lambda: SimpleNamespace(add=fail))
    result = mem0_client.save_memory("约定", user_id="u", workspace_id="w", thread_id="t")
    assert result["status"] == "FAILED"
    assert memory_update_for_result("memory-1", result)["status"] == "failed"


def test_allowed_tool_emits_live_status_and_result_survives_in_history_details():
    call = {"name": "add_memory", "args": {"content": "后续分析使用中文表格。"}, "id": "memory-1"}
    output = AIMessage(content="", tool_calls=[call])
    safe_output = tool_safety_node({"messages": [output]})
    start_events = list(translate_graph_progress_events(
        {"type": "updates", "data": {"Tool Safety": safe_output}}, 1, "request-1"
    ))
    assert start_events == [{
        "type": "memory_update", "request_id": "request-1",
        "update": {"tool_call_id": "memory-1", "status": "updating"},
    }]
    result = ToolMessage(name="add_memory", tool_call_id="memory-1", content=json.dumps({"status": "SUCCEEDED"}))
    end_events = list(translate_graph_progress_events(
        {"type": "updates", "data": {"Tool Execution": {"messages": [result]}}}, 1, "request-1"
    ))
    assert end_events[0]["update"]["status"] == "succeeded"
    details = build_turn_result([HumanMessage(content="记住我的表格偏好"), output, result, AIMessage(content="已记住。")])
    assert details["memory_updates"] == [end_events[0]["update"]]
    assert build_turn_result([HumanMessage(content="新一轮普通问题")])["memory_updates"] == []


def test_denied_tool_does_not_display_updating():
    output = AIMessage(content="", tool_calls=[{"name": "add_memory", "args": {}, "id": "denied-1"}])
    safe_output = tool_safety_node({"messages": [output]})
    assert list(translate_graph_progress_events(
        {"type": "updates", "data": {"Tool Safety": safe_output}}, 1, "request-1"
    )) == []
