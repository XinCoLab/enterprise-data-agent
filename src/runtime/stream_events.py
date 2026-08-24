"""Convert LangGraph stream parts into the small public NDJSON event protocol."""

from __future__ import annotations

import json
from typing import Any


NODE_ACTIVITY = {
    "Main Agent LLM": "正在分析现有信息并决定下一步…",
    "Tool Safety": "正在校验工具请求…",
    "Tool Execution": "正在执行已通过校验的工具…",
}

KNOWLEDGE_TOOLS = {
    "browse_knowledge",
    "search_knowledge",
    "read_knowledge",
}

KNOWLEDGE_STAGE = {
    "browse_knowledge": ("browsing", "正在浏览 Knowledge 目录"),
    "search_knowledge": ("searching", "正在搜索 Knowledge"),
    "read_knowledge": ("reading", "正在读取 KnowledgeCard"),
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
    model_output = messages[-1] if messages else None
    if not isinstance(model_output, AIMessage):
        return None

    tool_calls = []
    for call in model_output.tool_calls:
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
        "content": visible_ai_content(model_output.content),
        "tool_calls": tool_calls,
        "message": (
            "本轮模型输出已生成，正在执行工具。"
            if tool_calls
            else "本轮模型已生成最终回答。"
        ),
    }


def knowledge_trace_events(part: dict) -> list[dict]:
    """Expose real Knowledge navigation activity without exposing reasoning.

    The graph opens only after Tool Safety has allowed a Knowledge tool. It
    stays open while the next model call consumes those results, then closes
    when the model moves to SQL or produces its final answer.
    """

    from langchain_core.messages import AIMessage

    data = part.get("data")
    if not isinstance(data, dict):
        return []

    safety_update = data.get("Tool Safety")
    if isinstance(safety_update, dict):
        messages = safety_update.get("messages", [])
        safe_message = messages[-1] if messages else None
        if not isinstance(safe_message, AIMessage):
            return []

        decisions = safe_message.additional_kwargs.get(
            "tool_safety_decisions",
            [],
        )
        allowed_call_ids = {
            str(decision.get("tool_call_id", ""))
            for decision in decisions
            if isinstance(decision, dict)
            and decision.get("decision") == "ALLOW"
            and decision.get("tool_name") in KNOWLEDGE_TOOLS
        }
        allowed_calls = [
            call
            for call in safe_message.tool_calls
            if str(call.get("id", "")) in allowed_call_ids
            and call.get("name") in KNOWLEDGE_TOOLS
        ]
        if not allowed_calls:
            return []

        trace = (safe_message.response_metadata or {}).get("knowledge_view")
        active_ids = []
        if isinstance(trace, dict):
            active_ids.extend(trace.get("subglobal_knowledge_ids") or [])
        for call in allowed_calls:
            arguments = call.get("args") or {}
            requested_ids = arguments.get("knowledge_ids", [])
            if call.get("name") == "read_knowledge" and isinstance(
                requested_ids,
                list,
            ):
                active_ids.extend(requested_ids)

        ordered_ids = list(
            dict.fromkeys(
                str(item)
                for item in active_ids
                if isinstance(item, str) and item.strip()
            )
        )
        tool_names = [str(call.get("name", "")) for call in allowed_calls]
        primary_tool = next(
            (
                name
                for name in (
                    "read_knowledge",
                    "search_knowledge",
                    "browse_knowledge",
                )
                if name in tool_names
            ),
            "browse_knowledge",
        )
        stage, message = KNOWLEDGE_STAGE[primary_tool]
        return [
            {
                "type": "knowledge_trace",
                "action": "open",
                "stage": stage,
                "message": message,
                "mode": (
                    trace.get("knowledge_view_mode")
                    if isinstance(trace, dict)
                    else None
                ),
                "active_ids": ordered_ids,
                "tool_names": tool_names,
            }
        ]

    main_update = data.get("Main Agent LLM")
    if not isinstance(main_update, dict):
        return []
    messages = main_update.get("messages", [])
    model_output = messages[-1] if messages else None
    if not isinstance(model_output, AIMessage):
        return []
    if any(
        call.get("name") in KNOWLEDGE_TOOLS
        for call in model_output.tool_calls
    ):
        return []
    return [
        {
            "type": "knowledge_trace",
            "action": "close",
            "stage": "complete",
            "message": "Knowledge 导航完成",
            "active_ids": [],
            "tool_names": [],
        }
    ]


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
    """Stop only after Tool execution or a terminal Assistant message."""

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
    latest_message = messages[-1] if messages else None
    return (
        isinstance(latest_message, AIMessage)
        and not latest_message.tool_calls
    )
