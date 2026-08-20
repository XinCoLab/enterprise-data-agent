"""Execute only tool calls explicitly allowed by Tool Safety."""

import json
import re

from langchain_core.messages import AIMessage, ToolMessage

from graph.text2sql_state import Text2SQLState
from safety.tool_safety import ALLOW
from tools.tool_registry import TOOLS_BY_NAME


_SENSITIVE_ERROR_VALUE = re.compile(
    r"(?i)\b(password|token|api[_-]?key|secret)\s*[:=]\s*([^\s,;]+)"
)


def _safe_error_details(error: Exception) -> str:
    """Keep concise repair evidence without returning secrets or a traceback."""

    details = " ".join(str(error).split())
    details = _SENSITIVE_ERROR_VALUE.sub(r"\1=[REDACTED]", details)
    return details[:1000]


def tool_execution_node(state: Text2SQLState) -> dict:
    """Execute ALLOW decisions and return DENY decisions without execution."""

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        raise TypeError("Tool Execution requires the latest message to be an AIMessage.")

    decisions = {
        decision["tool_call_id"]: decision
        for decision in message.additional_kwargs.get("tool_safety_decisions", [])
    }
    results = []
    for tool_call in message.tool_calls:
        decision = decisions.get(tool_call["id"])
        if decision is None or decision["decision"] != ALLOW:
            reason = (
                decision.get("reason")
                if decision is not None
                else "Tool call has no Tool Safety decision."
            )
            content = json.dumps(
                {
                    "status": "denied",
                    "error_type": (
                        decision.get("error_code")
                        if decision is not None
                        else "POLICY_DENIED"
                    ),
                    "reason": reason,
                },
                ensure_ascii=False,
            )
        else:
            registered_tool = TOOLS_BY_NAME[tool_call["name"]]
            try:
                content = registered_tool.invoke(tool_call["args"])
            except Exception as error:
                content = json.dumps(
                    {
                        "status": "error",
                        "error_type": "EXECUTION_ERROR",
                        "message": "The Tool failed during execution.",
                        "details": _safe_error_details(error),
                    },
                    ensure_ascii=False,
                )

        results.append(
            ToolMessage(
                content=str(content),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}
