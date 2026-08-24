"""Convert LangGraph stream parts into the small public NDJSON event protocol."""

from __future__ import annotations

import json
from typing import Any


NODE_ACTIVITY = {
    "Main Agent LLM": "正在分析现有信息并决定下一步…",
    "Tool Safety": "正在校验工具请求…",
    "Tool Execution": "正在执行已通过校验的工具…",
}


def encode_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def task_progress_event(part: dict) -> dict | None:
    data = part.get("data")
    if not isinstance(data, dict) or "input" not in data:
        return None
    node_name = str(data.get("name", ""))
    message = NODE_ACTIVITY.get(node_name)
    if not message:
        return None
    return {"type": "progress", "stage": node_name, "message": message}


def tool_result_progress(message: Any) -> dict | None:
    from langchain_core.messages import ToolMessage

    if not isinstance(message, ToolMessage):
        return None
    tool_name = str(message.name or "")
    if tool_name != "execute_readonly_sql":
        return {
            "type": "progress",
            "stage": "Tool Execution",
            "tool": tool_name,
            "message": f"{tool_name} 已返回结果。",
        }
    try:
        payload = json.loads(str(message.content))
    except (TypeError, json.JSONDecodeError):
        payload = {}
    returned_rows = payload.get("returned_rows")
    if isinstance(returned_rows, int):
        suffix = "，结果已截断" if payload.get("truncated") else ""
        text = f"SQL 执行完成，返回 {returned_rows} 行{suffix}。"
    elif payload.get("status") in {"error", "denied"}:
        text = "SQL 未成功执行，Agent 将根据错误信息调整。"
    else:
        text = "SQL 执行已返回。"
    return {
        "type": "progress",
        "stage": "Tool Execution",
        "tool": tool_name,
        "message": text,
    }


def visible_ai_content(content: Any) -> str:
    """Return only assistant content intended for the public message."""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "\n".join(part.strip() for part in text_parts if part.strip())


def llm_round_event(part: dict, round_number: int) -> dict | None:
    from langchain_core.messages import AIMessage

    data = part.get("data")
    if not isinstance(data, dict):
        return None
    main_update = data.get("Main Agent LLM")
    if not isinstance(main_update, dict):
        return None
    messages = main_update.get("messages", [])
    reply = messages[-1] if messages else None
    if not isinstance(reply, AIMessage):
        return None

    tool_calls = []
    for call in reply.tool_calls:
        arguments = call.get("args", {})
        if not isinstance(arguments, dict):
            arguments = {"value": str(arguments)}
        tool_calls.append(
            {
                "name": str(call.get("name", "")),
                "arguments": arguments,
            }
        )

    return {
        "type": "round",
        "stage": "Main Agent LLM",
        "round": round_number,
        "content": visible_ai_content(reply.content),
        "tool_calls": tool_calls,
        "message": (
            "本轮模型输出已生成，正在执行工具。"
            if tool_calls
            else "本轮模型已生成最终回答。"
        ),
    }


def update_progress_events(part: dict) -> list[dict]:
    data = part.get("data")
    if not isinstance(data, dict):
        return []
    events: list[dict] = []
    execution_update = data.get("Tool Execution")
    if isinstance(execution_update, dict):
        for message in execution_update.get("messages", []):
            event = tool_result_progress(message)
            if event is not None:
                events.append(event)
    return events


def safe_cancel_boundary(part: dict) -> bool:
    """Stop only after a complete Tool round or terminal Assistant reply."""

    from langchain_core.messages import AIMessage

    data = part.get("data")
    if not isinstance(data, dict):
        return False
    if "Tool Execution" in data:
        return True
    main_update = data.get("Main Agent LLM")
    if not isinstance(main_update, dict):
        return False
    messages = main_update.get("messages", [])
    reply = messages[-1] if messages else None
    return isinstance(reply, AIMessage) and not reply.tool_calls
