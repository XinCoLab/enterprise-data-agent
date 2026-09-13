"""真实模型选择记忆工具的三轮联调，仅使用独立测试目录和会话。"""
import asyncio
import json
import os
from pathlib import Path
import sys
from uuid import uuid4


async def main():
    root = Path(__file__).resolve().parents[2]
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.path[:0] = [str(root / "src"), str(root)]
    store = root / "runtime" / "mem0-crud-smoke" / uuid4().hex
    os.environ["DATA_AGENT_MEMORY_DIR"] = str(store)
    os.environ["DATA_AGENT_MEMORY_ENABLED"] = "1"
    os.environ["DATA_AGENT_MEMORY_EMBEDDING_LOCAL_ONLY"] = "1"
    os.environ["MEM0_TELEMETRY"] = "false"

    from langchain_core.messages import HumanMessage, SystemMessage
    from agent_runtime.agent_run_context import AgentRunContext
    from graph.nodes.tool_execution_node import tool_execution_node
    from graph.nodes.tool_safety_node import tool_safety_node
    from memory.mem0_client import (
        close_memory_client, delete_memory_record, get_mem0_client,
        memory_filters, search_memories, update_memory_record,
    )
    from model_clients.llm_api_clients import get_main_llm
    from prompts.runtime_memory_context import build_memory_context_message
    from tools.tool_registry import TOOLS_BY_NAME

    scope = {"user_id": "crud-smoke-user", "workspace_id": "crud-smoke-workspace", "thread_id": store.name}
    context = AgentRunContext(
        request_id="crud-smoke-request", **scope, model="deepseek-v4-flash",
        permissions=frozenset({"chat:run"}), allowed_data_source_ids=(), selected_data_source_ids=(),
    )
    config = {"configurable": {"agent_run_context": context}}
    filters = memory_filters(**scope)
    model = get_main_llm(context.model).bind_tools([
        TOOLS_BY_NAME[name] for name in ("add_memory", "update_memory", "delete_memory")
    ])
    system = SystemMessage(content=(
        "你是数据分析Agent，现在处理用户的记忆管理请求。选择适当工具，每轮只执行一个记忆操作。\n"
        + (root / "src/prompts/memory_instruction.md").read_text(encoding="utf-8")
    ))
    turns = [
        ("add_memory", "请记住：虚拟偶像回访候选名单最多7人，作为本会话的长期约定。"),
        ("update_memory", "刚才的长期人数约定改成5人，替换掉7人的旧约定。"),
        ("delete_memory", "撤销这条候选人数约定，把它从记忆中删除。"),
    ]
    messages = []
    report = {"scope": scope, "storage": str(store), "turns": [], "status": "running"}
    memory_id = None
    try:
        for expected, question in turns:
            recalled = search_memories(question, **scope)
            memory_context = build_memory_context_message(recalled)
            messages.append(HumanMessage(content=question))
            output = await model.ainvoke([system, *([memory_context] if memory_context else []), *messages], max_tokens=2048)
            row = {"question": question, "expected_tool": expected, "tool_calls": output.tool_calls,
                   "usage": output.usage_metadata, "recalled_ids": [m["id"] for m in recalled]}
            report["turns"].append(row)
            assert len(output.tool_calls) == 1 and output.tool_calls[0]["name"] == expected, row
            if memory_id is not None:
                assert output.tool_calls[0]["args"]["memory_id"] == memory_id, row
            safe = tool_safety_node({"messages": [output]})
            result = await tool_execution_node({**safe, "retrieved_memories": recalled}, config)
            payload = json.loads(result["messages"][0].content)
            row["result"] = payload
            assert payload["status"] == "SUCCEEDED", payload
            messages.extend([output, *result["messages"]])
            stored = get_mem0_client().get_all(filters=filters, top_k=100)["results"]
            row["stored_after"] = stored
            if expected == "add_memory":
                assert len(stored) == 1 and "7" in stored[0]["memory"], stored
                memory_id = stored[0]["id"]
                for field in scope:
                    other = {**scope, field: scope[field] + "-other"}
                    assert update_memory_record(memory_id, "不可写入的跨会话内容", **other)["status"] == "NOT_FOUND"
                    assert delete_memory_record(memory_id, **other)["status"] == "NOT_FOUND"
                row["isolation"] = "passed: known ID cannot update/delete across user/workspace/thread"
            elif expected == "update_memory":
                assert len(stored) == 1 and stored[0]["id"] == memory_id
                assert "5" in stored[0]["memory"] and "7" not in stored[0]["memory"], stored
                assert not any(m["id"] == memory_id for m in result["retrieved_memories"])
                # 重新打开本地存储，确认修改并非只发生在进程缓存里。
                close_memory_client()
                assert "5" in get_mem0_client().get(memory_id)["memory"]
                row["reopen_check"] = "passed"
            else:
                assert stored == [] and get_mem0_client().get(memory_id) is None
                assert search_memories("虚拟偶像回访候选名单人数约定", **scope) == []
                assert not any(m["id"] == memory_id for m in result["retrieved_memories"])
            print(json.dumps({"tool": expected, "status": payload["status"], "stored_after": len(stored)}, ensure_ascii=False), flush=True)
        report["status"] = "passed"
    finally:
        close_memory_client()
        store.mkdir(parents=True, exist_ok=True)
        (store / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"联调记录：{store / 'result.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
