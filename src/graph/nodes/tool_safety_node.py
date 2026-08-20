"""LangGraph adapter for the agent-independent tool safety boundary."""

from uuid import uuid4

from langchain_core.messages import AIMessage

from graph.text2sql_state import Text2SQLState
from safety.tool_safety import StandardToolCall, check_tool_calls
from tools.tool_registry import TOOLS_BY_NAME


def tool_safety_node(state: Text2SQLState) -> dict:
    """Translate AI tool calls, attach ALLOW/DENY decisions, and do nothing else."""

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        raise TypeError("Tool Safety requires the latest message to be an AIMessage.")

    standard_calls = [
        StandardToolCall(
            tool_name=tool_call["name"],
            arguments=tool_call["args"],
            tool_call_id=tool_call["id"],
        )
        for tool_call in message.tool_calls
    ]
    decisions = check_tool_calls(
        standard_calls,
        registered_tools=TOOLS_BY_NAME,
    )

    # Reuse the existing messages channel instead of introducing Tool State.
    # add_messages replaces a message with the same ID.
    safe_message = message.model_copy(
        update={
            "id": message.id or f"tool-safety-{uuid4()}",
            "additional_kwargs": {
                **message.additional_kwargs,
                "tool_safety_decisions": [
                    decision.to_dict() for decision in decisions
                ],
            },
        }
    )
    return {"messages": [safe_message]}
