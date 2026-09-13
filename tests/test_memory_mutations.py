import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage

from agent_runtime.agent_run_context import AgentRunContext
from graph.nodes.tool_execution_node import tool_execution_node
from graph.nodes.tool_safety_node import tool_safety_node
from memory import mem0_client
from tools.tool_registry import TOOLS_BY_NAME


SCOPE = {"user_id": "user-a", "workspace_id": "workspace-a", "thread_id": "thread-a"}
OLD = "虚拟偶像候选最多7人，金额显示USD两位小数。"
NEW = "虚拟偶像候选最多5人，金额显示USD两位小数。"


class MemoryStore:
    def __init__(self):
        self.records = {"entry-a": {
            "id": "entry-a", "memory": OLD,
            "user_id": SCOPE["user_id"], "run_id": SCOPE["thread_id"],
            "metadata": {"workspace_id": SCOPE["workspace_id"], "source_thread_id": SCOPE["thread_id"]},
        }}
        self.writes = []

    def get(self, memory_id):
        return deepcopy(self.records.get(memory_id))

    def update(self, *, memory_id, text):
        self.writes.append(("update", memory_id))
        self.records[memory_id]["memory"] = text

    def delete(self, *, memory_id):
        self.writes.append(("delete", memory_id))
        del self.records[memory_id]


@pytest.fixture
def store(monkeypatch):
    client = MemoryStore()
    monkeypatch.setattr(mem0_client, "get_mem0_client", lambda: client)
    return client


def config():
    return {"configurable": {"agent_run_context": AgentRunContext(
        request_id="request-a", **SCOPE, model="deepseek-v4-flash",
        permissions=frozenset({"chat:run"}), allowed_data_source_ids=(), selected_data_source_ids=(),
    )}}


def mutate(operation, memory_id="entry-a", scope=None):
    if operation == "update":
        return mem0_client.update_memory_record(memory_id, NEW, **(scope or SCOPE))
    return mem0_client.delete_memory_record(memory_id, **(scope or SCOPE))


@pytest.mark.parametrize("operation", ["update", "delete"])
@pytest.mark.parametrize("field", ["user_id", "workspace_id", "thread_id"])
def test_known_foreign_id_cannot_be_modified_or_read_back(store, operation, field):
    result = mutate(operation, scope={**SCOPE, field: "other"})
    assert result["status"] == "NOT_FOUND"
    assert OLD not in json.dumps(result, ensure_ascii=False)
    assert not store.writes
    assert store.records["entry-a"]["memory"] == OLD


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_missing_id_or_missing_scope_metadata_cannot_be_mutated(store, operation):
    assert mutate(operation, memory_id="absent")["status"] == "NOT_FOUND"
    store.records["entry-a"]["metadata"] = {}
    assert mutate(operation)["status"] == "NOT_FOUND"
    assert not store.writes


def test_update_preserves_id_other_details_and_scope_and_skips_identical_write(store):
    before = deepcopy(store.records["entry-a"])
    result = mutate("update")
    assert result["status"] == "SUCCEEDED"
    assert result["results"] == [{"id": "entry-a", "event": "UPDATE", "memory": NEW}]
    assert store.records == {"entry-a": {**before, "memory": NEW}}
    assert mutate("update")["status"] == "NO_CHANGE"
    assert store.writes == [("update", "entry-a")]


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_storage_failure_is_not_reported_as_success(store, monkeypatch, operation):
    def fail(**kwargs):
        raise RuntimeError("storage unavailable")
    monkeypatch.setattr(store, operation, fail)
    assert mutate(operation)["status"] == "FAILED"
    assert store.records["entry-a"]["memory"] == OLD


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_registered_tool_executes_with_trusted_scope_and_invalidates_stale_retrieval(store, operation):
    name = operation + "_memory"
    args = {"memory_id": "entry-a"}
    if operation == "update":
        args["content"] = NEW
    output = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-a"}])
    safe = tool_safety_node({"messages": [output]})
    original = [{"id": "entry-a", "memory": OLD}, {"id": "entry-other", "memory": "无关约定"}]
    result = asyncio.run(tool_execution_node({**safe, "retrieved_memories": original}, config()))
    assert json.loads(result["messages"][0].content)["status"] == "SUCCEEDED"
    assert result["retrieved_memories"] == [original[1]]
    assert original[0]["memory"] == OLD  # 输入状态仍由图统一更新。
    assert store.writes == [(operation, "entry-a")]
    if operation == "delete":
        assert store.get("entry-a") is None
    else:
        assert store.get("entry-a")["memory"] == NEW


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_failed_mutation_keeps_retrieved_context(store, operation):
    args = {"memory_id": "absent"}
    if operation == "update":
        args["content"] = NEW
    output = AIMessage(content="", tool_calls=[{"name": operation + "_memory", "args": args, "id": "call-a"}])
    safe = tool_safety_node({"messages": [output]})
    result = asyncio.run(tool_execution_node({**safe, "retrieved_memories": [store.get("entry-a")]}, config()))
    assert json.loads(result["messages"][0].content)["status"] == "NOT_FOUND"
    assert "retrieved_memories" not in result


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_model_cannot_supply_scope_or_run_without_trusted_context(store, operation):
    name = operation + "_memory"
    args = {"memory_id": "entry-a"}
    if operation == "update":
        args["content"] = NEW
    assert not {"user_id", "workspace_id", "thread_id", "config"}.intersection(TOOLS_BY_NAME[name].args)
    output = AIMessage(content="", tool_calls=[{"name": name, "args": {**args, "thread_id": "other"}, "id": "call-a"}])
    safe = tool_safety_node({"messages": [output]})
    assert safe["messages"][0].additional_kwargs["tool_safety_decisions"][0]["decision"] == "DENY"
    with pytest.raises(RuntimeError, match="用户信息"):
        TOOLS_BY_NAME[name].invoke(args, config={})
    assert not store.writes
