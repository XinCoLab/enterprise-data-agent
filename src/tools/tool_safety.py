"""Framework-independent safety decisions for standard tool calls."""

from dataclasses import asdict, dataclass
from importlib import import_module
import os
import re
from typing import Any, Mapping, Sequence

ALLOW = "ALLOW"
DENY = "DENY"

UNKNOWN_TOOL = "UNKNOWN_TOOL"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
POLICY_DENIED = "POLICY_DENIED"

MAX_DIRECTORY_PATH_LENGTH = 256
MAX_SEARCH_QUERY_LENGTH = 500
MAX_KNOWLEDGE_IDS_PER_CALL = 100
MAX_SQL_LENGTH = 100_000

_EXTERNAL_SQL_SOURCE = re.compile(
    r"(?:\b(?:read_[a-z0-9_]+|[a-z0-9_]+_scan|glob)\s*\(|"
    r"\bfrom\s+['\"])",
    re.IGNORECASE,
)


def _validate_configured_database_sql(sql: str) -> None:
    backend = os.getenv("DATA_AGENT_DATABASE_BACKEND", "postgresql").strip().lower()
    if backend not in {"postgresql", "mysql", "duckdb"}:
        raise ValueError(f"Unsupported database backend: {backend!r}")
    module = import_module(f"databases.{backend}_executor")
    module.validate_readonly_sql(sql)


@dataclass(frozen=True)
class StandardToolCall:
    """The only input shape understood by the safety layer."""

    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class ToolSafetyDecision:
    """A minimal allow-or-deny answer for one standard tool call."""

    decision: str
    tool_name: str
    tool_call_id: str
    error_code: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_tool_calls(
    tool_calls: Sequence[StandardToolCall],
    *,
    registered_tools: Mapping[str, Any],
) -> list[ToolSafetyDecision]:
    """Return one ALLOW or DENY decision for every proposed tool call.

    The function knows nothing about LangGraph, messages, prompts, planning,
    retries, budgets, recovery, or tool execution.
    """

    return [
        _check_one_tool_call(
            tool_call,
            registered_tools=registered_tools,
        )
        for tool_call in tool_calls
    ]


def _check_one_tool_call(
    tool_call: StandardToolCall,
    *,
    registered_tools: Mapping[str, Any],
) -> ToolSafetyDecision:
    registered_tool = registered_tools.get(tool_call.tool_name)
    if registered_tool is None:
        return _deny(
            tool_call,
            UNKNOWN_TOOL,
            "Tool is not present in the active registry.",
        )

    if not isinstance(tool_call.arguments, dict):
        return _deny(
            tool_call,
            INVALID_ARGUMENTS,
            "Tool arguments must be a JSON object.",
        )
    if not str(tool_call.tool_call_id or "").strip():
        return _deny(
            tool_call,
            INVALID_ARGUMENTS,
            "Tool call ID must be a non-empty string.",
        )

    try:
        unexpected_arguments = set(tool_call.arguments) - set(registered_tool.args)
        if unexpected_arguments:
            return _deny(
                tool_call,
                INVALID_ARGUMENTS,
                "Tool arguments contain unknown fields: "
                + ", ".join(sorted(unexpected_arguments)),
            )
        registered_tool.get_input_schema().model_validate(tool_call.arguments)
    except Exception as error:
        details = " ".join(str(error).split())[:500]
        return _deny(
            tool_call,
            INVALID_ARGUMENTS,
            f"Tool arguments do not match the schema: {details}",
        )

    argument_check = _TOOL_ARGUMENT_CHECKS.get(tool_call.tool_name)
    if argument_check is not None:
        reason = argument_check(tool_call.arguments)
        if reason is not None:
            return _deny(tool_call, INVALID_ARGUMENTS, reason)

    safety_check = _TOOL_SAFETY_CHECKS.get(tool_call.tool_name)
    if safety_check is not None:
        reason = safety_check(
            tool_call.arguments,
        )
        if reason is not None:
            return _deny(tool_call, POLICY_DENIED, reason)

    return ToolSafetyDecision(
        decision=ALLOW,
        tool_name=tool_call.tool_name,
        tool_call_id=tool_call.tool_call_id,
    )


def _deny(
    tool_call: StandardToolCall,
    error_code: str,
    reason: str,
) -> ToolSafetyDecision:
    return ToolSafetyDecision(
        decision=DENY,
        tool_name=tool_call.tool_name,
        tool_call_id=tool_call.tool_call_id,
        error_code=error_code,
        reason=reason,
    )


def _check_browse_knowledge(
    arguments: dict[str, Any],
    **_: Any,
) -> str | None:
    directory_path = arguments.get("directory_path", "/")
    if not isinstance(directory_path, str) or not directory_path.startswith("/"):
        return "Knowledge directory path must be an absolute virtual path."
    if len(directory_path) > MAX_DIRECTORY_PATH_LENGTH:
        return "Knowledge directory path exceeds the allowed length."
    if "\\" in directory_path or ".." in directory_path or "\x00" in directory_path:
        return "Knowledge directory path is outside the virtual catalog boundary."
    return None


def _check_search_knowledge(
    arguments: dict[str, Any],
    **_: Any,
) -> str | None:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return "Knowledge search query must not be empty."
    if len(query) > MAX_SEARCH_QUERY_LENGTH:
        return "Knowledge search query exceeds the allowed length."
    return None


def _check_read_knowledge(
    arguments: dict[str, Any],
    **_: Any,
) -> str | None:
    knowledge_ids = arguments.get("knowledge_ids")
    if not isinstance(knowledge_ids, list) or not knowledge_ids:
        return "knowledge_ids must be a non-empty list."
    if len(knowledge_ids) > MAX_KNOWLEDGE_IDS_PER_CALL:
        return "Too many knowledge IDs were requested in one Tool call."
    if any(not isinstance(item, str) or not item.strip() for item in knowledge_ids):
        return "Every knowledge_id must be a non-empty string."
    return None


def _check_execute_readonly_sql(
    arguments: dict[str, Any],
    **_: Any,
) -> str | None:
    sql = arguments["sql"]
    if len(sql) > MAX_SQL_LENGTH:
        return "SQL exceeds the allowed statement length."
    try:
        _validate_configured_database_sql(sql)
    except ValueError as error:
        return f"SQL is outside the read-only database boundary: {error}"
    if _EXTERNAL_SQL_SOURCE.search(sql):
        return "SQL external file, URL, object-store, or database access is not allowed."
    return None


_TOOL_ARGUMENT_CHECKS = {
    "search_knowledge": _check_search_knowledge,
    "read_knowledge": _check_read_knowledge,
}

_TOOL_SAFETY_CHECKS = {
    "browse_knowledge": _check_browse_knowledge,
    "execute_readonly_sql": _check_execute_readonly_sql,
}
