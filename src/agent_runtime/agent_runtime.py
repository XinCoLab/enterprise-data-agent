"""Run one Agent request from the HTTP boundary to the saved final answer.

Main path:
    api/routers/chat.py
    -> build_agent_config()
    -> stream_agent() or run_agent()
    -> run_agent_events()                 # the only Agent execution loop
    -> single_round_graph.astream()       # LLM -> Safety -> Tool -> END
    -> save_assistant_message()

``stream_agent`` only converts event dictionaries to NDJSON. ``run_agent`` only
collects the final event for the non-streaming endpoint. Agent execution belongs
in ``run_agent_events`` so the two HTTP endpoints cannot drift apart.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatRequest

from memory.conversation_history_database import (
    DEFAULT_WORKSPACE_ID,
    delete_conversation as delete_conversation_history,
    read_conversation_workspace_id,
    save_assistant_message,
    save_user_message,
)
from config.project_paths import CONFIG_ROOT
from security.workspace_access import CurrentUser
from agent_runtime.agent_run_context import AgentRunContext
from agent_runtime.translate_graph_events import (
    encode_event,
    extract_visible_ai_content,
    is_safe_cancel_boundary,
    translate_knowledge_trace_events,
    translate_llm_round_event,
    translate_task_progress_event,
    translate_update_progress_events,
)

SETTINGS_PATH = CONFIG_ROOT / "settings.env"
SECRETS_PATH = CONFIG_ROOT / "secrets.env"
ACTIVE_PROFILE_PATH = CONFIG_ROOT / ".active_profile"
RECENT_RUNS: deque[dict] = deque(maxlen=50)
ACTIVE_RUNS_LOCK = threading.Lock()
RECENT_RUNS_LOCK = threading.Lock()
RESOURCE_CONFIG_LOCK = threading.Lock()
CONVERSATION_LOCKS: dict[str, asyncio.Lock] = {}
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass
class ActiveRun:
    """Cooperative cancellation state for one Web Agent request."""

    request_id: str
    thread_id: str
    workspace_id: str
    user_id: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)


ACTIVE_RUNS: dict[str, ActiveRun] = {}


def conversation_lock(checkpoint_thread_id: str) -> asyncio.Lock:
    """Serialize writes to one conversation while other conversations run."""

    lock = CONVERSATION_LOCKS.get(checkpoint_thread_id)
    if lock is None:
        lock = asyncio.Lock()
        CONVERSATION_LOCKS[checkpoint_thread_id] = lock
    return lock


def active_runs_exist() -> bool:
    with ACTIVE_RUNS_LOCK:
        return bool(ACTIVE_RUNS)


def register_active_run(
    run_control: ActiveRun,
    run_context: AgentRunContext,
    current_user: CurrentUser,
) -> None:
    """Start a run without racing a global database/knowledge configuration change."""

    with RESOURCE_CONFIG_LOCK:
        confirm_run_resources_are_unchanged(run_context, current_user)
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS[run_control.request_id] = run_control


def list_recent_runs(current_user: CurrentUser) -> list[dict]:
    """Return runs from the current workspace only."""

    with RECENT_RUNS_LOCK:
        return [
            run
            for run in RECENT_RUNS
            if run["workspace_id"] == current_user.workspace_id
        ]


def request_cancel(
    request_id: str,
    current_user: CurrentUser,
) -> dict[str, str]:
    """Ask an active run to stop at its next complete message boundary."""

    if not REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id 格式不合法。")
    with ACTIVE_RUNS_LOCK:
        control = ACTIVE_RUNS.get(request_id)
        if control is None or control.workspace_id != current_user.workspace_id:
            return {
                "status": "not_running",
                "message": "任务已结束或不存在。",
            }
        if control.user_id != current_user.user_id and current_user.role != "admin":
            return {
                "status": "not_running",
                "message": "任务已结束或不存在。",
            }
        control.cancel_requested.set()
    return {
        "status": "cancel_requested",
        "message": "已请求停止，将在当前模型或工具步骤结束后安全停止。",
    }


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def safe_error_text(error: Exception) -> str:
    text = " ".join(str(error).split())
    text = re.sub(
        r"(?i)\b(password|token|api[_-]?key|secret)\s*[:=]\s*([^\s,;]+)",
        r"\1=[REDACTED]",
        text,
    )
    return text[:800]


def model_api_key_is_configured() -> bool:
    return bool(
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or read_env_file(SECRETS_PATH).get("DEEPSEEK_API_KEY", "").strip()
    )


def current_turn_messages(messages: list) -> list:
    from langchain_core.messages import HumanMessage

    last_human = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    return messages[last_human:]


def build_turn_result(messages: list) -> dict:
    from langchain_core.messages import AIMessage, ToolMessage

    current = current_turn_messages(messages)
    tool_calls: list[dict] = []
    tool_results: dict[str, dict] = {}
    final_answer = ""
    navigation_trace = None
    artifacts: list[dict[str, str]] = []

    for message in current:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "args": call.get("args", {}),
                    }
                )
            if message.content and not message.tool_calls:
                final_answer = str(message.content)
            trace = (message.response_metadata or {}).get("knowledge_view")
            if trace:
                navigation_trace = trace
        elif isinstance(message, ToolMessage):
            # Tool Execution always serializes ToolMessage.content as JSON.
            # Invalid content is an internal protocol error and must fail fast.
            payload = json.loads(str(message.content))
            tool_results[message.tool_call_id] = payload
            artifact = payload.get("artifact")
            if (
                isinstance(artifact, dict)
                and artifact.get("kind") == "report"
                and isinstance(artifact.get("preview_url"), str)
            ):
                artifacts.append(
                    {
                        "id": str(artifact.get("id", "")),
                        "kind": "report",
                        "title": str(artifact.get("title", "报告")),
                        "preview_url": artifact["preview_url"],
                    }
                )

    sql_queries = [
        {
            "tool_call_id": call["id"],
            "sql": str(call["args"].get("sql", "")),
            "result": tool_results.get(call["id"]),
        }
        for call in tool_calls
        if call["name"] == "execute_readonly_sql"
    ]
    successful_results = [
        item["result"]
        for item in sql_queries
        if isinstance(item.get("result"), dict)
        and isinstance(item["result"].get("rows"), list)
    ]
    tool_counts = Counter(call["name"] for call in tool_calls)
    if artifacts:
        final_answer = re.sub(
            r"(?:Dashboard\s*已生成[，,。]?\s*)?"
            r"预览地址\s*[:：]\s*`?"
            r"/api/artifacts/[A-Za-z0-9_-]+/view`?[。.]?",
            "Dashboard 已生成。",
            final_answer,
            flags=re.IGNORECASE,
        )
        final_answer = re.sub(
            r"`?/api/artifacts/[A-Za-z0-9_-]+/view`?",
            "",
            final_answer,
        )
        final_answer = re.sub(r"\n{3,}", "\n\n", final_answer).strip()
    if artifacts and any(
        marker in final_answer.lower()
        for marker in ("<!doctype html", "<html", "<style", "<script")
    ):
        final_answer = "报告已生成。"
    return {
        "answer": final_answer,
        "tool_counts": dict(sorted(tool_counts.items())),
        "sql_queries": sql_queries,
        "result_preview": successful_results[-1] if successful_results else None,
        "knowledge_view": navigation_trace,
        "artifacts": artifacts,
    }


def empty_turn_result() -> dict:
    return {
        "answer": "",
        "tool_counts": {},
        "sql_queries": [],
        "result_preview": None,
        "knowledge_view": None,
        "artifacts": [],
    }


def data_source_ids_for_run(current_user: CurrentUser) -> tuple[str, ...]:
    """Snapshot the data source selected by the current compatibility UI."""

    if not current_user.resources_ready:
        return ()
    if not ACTIVE_PROFILE_PATH.is_file():
        return ("current",)
    data_source_id = ACTIVE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    return (data_source_id or "current",)


def confirm_run_resources_are_unchanged(
    run_context: AgentRunContext,
    current_user: CurrentUser,
) -> None:
    """Fail closed if global compatibility settings changed before execution."""

    current_data_source_ids = data_source_ids_for_run(current_user)
    if current_data_source_ids != run_context.selected_data_source_ids:
        raise RuntimeError(
            "The selected data source changed before this Agent run started."
        )


def build_agent_config(
    request: ChatRequest,
    current_user: CurrentUser,
) -> tuple[str, dict]:
    thread_id = request.thread_id.strip() or str(uuid4())
    if not THREAD_ID.fullmatch(thread_id):
        raise HTTPException(status_code=400, detail="thread_id 格式不合法。")
    settings = read_env_file(SETTINGS_PATH)
    max_recursions = max(
        1,
        int(settings.get("DATA_AGENT_MAX_RECURSIONS", "10")),
    )
    data_source_ids = data_source_ids_for_run(current_user)
    run_context = AgentRunContext(
        request_id=str(uuid4()),
        thread_id=thread_id,
        workspace_id=current_user.workspace_id,
        user_id=current_user.user_id,
        model=request.model,
        permissions=current_user.permissions,
        allowed_data_source_ids=data_source_ids,
        selected_data_source_ids=data_source_ids,
    )
    return thread_id, {
        "configurable": {
            # workspace-a keeps the old raw ID so existing local checkpoints
            # remain readable. Every additional workspace receives its own
            # internal ID, so an orphan checkpoint can never leak across users.
            "thread_id": checkpoint_thread_id(
                thread_id,
                current_user.workspace_id,
            ),
            "conversation_thread_id": thread_id,
            "workspace_id": current_user.workspace_id,
            "user_id": current_user.user_id,
            "model": request.model,
            "max_recursions": max_recursions,
            "agent_run_context": run_context,
        },
        # One round graph contains only LLM, Safety, and Tool Execution. This
        # remains an internal emergency guard, not the product recursion budget.
        "recursion_limit": max(
            4,
            int(settings.get("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "6")),
        ),
    }


def checkpoint_thread_id(thread_id: str, workspace_id: str) -> str:
    """Return the private LangGraph thread ID for one workspace."""

    if workspace_id == DEFAULT_WORKSPACE_ID:
        return thread_id
    return f"{workspace_id}--{thread_id}"


async def delete_saved_conversation(
    thread_id: str,
    current_user: CurrentUser,
    checkpointer: Any,
) -> None:
    """Delete one conversation's checkpoint and visible history together."""

    private_thread_id = checkpoint_thread_id(
        thread_id,
        current_user.workspace_id,
    )
    async with conversation_lock(private_thread_id):
        await run_in_threadpool(ensure_thread_workspace, thread_id, current_user)
        await checkpointer.adelete_thread(private_thread_id)
        await run_in_threadpool(
            delete_conversation_history,
            thread_id,
            current_user.workspace_id,
        )


def ensure_thread_workspace(
    thread_id: str,
    current_user: CurrentUser,
) -> None:
    """Block a foreign thread before reading its LangGraph checkpoint."""

    owner_workspace = read_conversation_workspace_id(thread_id)
    if (
        owner_workspace is not None
        and owner_workspace != current_user.workspace_id
    ):
        raise HTTPException(status_code=404, detail="会话不存在。")


def max_agent_rounds(config: dict) -> int:
    return int(config["configurable"]["max_recursions"])


def tool_protocol_state(messages: list[Any]) -> str:
    """Return complete, pending, or invalid for contiguous Tool batches."""

    from langchain_core.messages import AIMessage, ToolMessage

    pending: list[str] = []
    for message in messages:
        if pending:
            if not isinstance(message, ToolMessage):
                return "invalid"
            tool_call_id = str(message.tool_call_id)
            if tool_call_id not in pending:
                return "invalid"
            pending.remove(tool_call_id)
            continue

        if isinstance(message, AIMessage) and message.tool_calls:
            pending = [str(call.get("id", "")) for call in message.tool_calls]
            if not all(pending) or len(set(pending)) != len(pending):
                return "invalid"
        elif isinstance(message, ToolMessage):
            return "invalid"
    return "pending" if pending else "complete"


async def prepare_checkpoint_for_new_input(graph: Any, config: dict) -> bool:
    """Finish a recoverable half-round before accepting a new HumanMessage."""

    snapshot = await graph.aget_state(config)
    messages = list(snapshot.values.get("messages", []))
    protocol_state = tool_protocol_state(messages)
    if protocol_state == "complete":
        return True
    if protocol_state == "invalid":
        return False

    pending_nodes = set(snapshot.next)
    if not pending_nodes.intersection({"Tool Safety", "Tool Execution"}):
        return False

    await graph.ainvoke(None, config=config)
    repaired = await graph.aget_state(config)
    repaired_messages = list(repaired.values.get("messages", []))
    return tool_protocol_state(repaired_messages) == "complete"


async def read_graph_messages(graph: Any, config: dict) -> list[Any]:
    """Read the messages persisted by LangGraph for this conversation."""

    saved_state = await graph.aget_state(config)
    return list(saved_state.values.get("messages", []))


def final_answer_is_ready(messages: list[Any]) -> bool:
    """Return whether the graph ended with a tool-free Assistant answer."""

    from langchain_core.messages import AIMessage

    latest_message = messages[-1]
    return isinstance(latest_message, AIMessage) and not latest_message.tool_calls


def record_run(
    *,
    request_id: str,
    thread_id: str,
    workspace_id: str,
    user_id: str,
    model: str,
    status: str,
    latency_ms: int,
    details: dict,
    created_at: str | None = None,
) -> None:
    with RECENT_RUNS_LOCK:
        RECENT_RUNS.appendleft(
            {
                "request_id": request_id,
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                "thread_id": thread_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "model": model,
                "status": status,
                "latency_ms": latency_ms,
                "tool_counts": details["tool_counts"],
                "sql_count": len(details["sql_queries"]),
            }
        )


def build_run_response(
    *,
    request_id: str,
    thread_id: str,
    model: str,
    status: str,
    latency_ms: int,
    details: dict,
) -> dict:
    return {
        "request_id": request_id,
        "status": status,
        "thread_id": thread_id,
        "model": model,
        "latency_ms": latency_ms,
        **details,
    }


async def generate_recursion_limit_summary(
    messages: list[Any],
    *,
    model_name: str,
):
    from langchain_core.messages import AIMessage
    from prompts.recursion_limit_summary import (
        generate_recursion_limit_summary as create_summary,
    )

    fallback = (
        "本次运行已达到最大循环次数，当前进度和工具结果已经保存。"
        "你可以继续原任务，或在当前会话中调整要求。"
    )
    try:
        model_output = await create_summary(
            messages,
            model_name=model_name,
        )
        content = extract_visible_ai_content(model_output.content)
    except Exception:
        # This is a second provider call made only after the round budget is
        # exhausted. A network/provider failure must not discard completed
        # tool results, so the saved conversation receives the fixed summary.
        content = ""
    tool_markup = ("<|DSML|", "tool_calls>", "invoke name=")
    if any(marker in content for marker in tool_markup):
        content = ""
    return AIMessage(content=content or fallback)


async def run_agent_events(
    request: ChatRequest,
    current_user: CurrentUser,
    *,
    single_round_graph: Any,
    thread_id: str,
    run_config: dict,
) -> AsyncIterator[dict]:
    from langchain_core.messages import HumanMessage

    run_context: AgentRunContext = run_config["configurable"]["agent_run_context"]
    request_id = run_context.request_id
    created_at = datetime.now(timezone.utc).isoformat()
    run_control = ActiveRun(
        request_id=request_id,
        thread_id=thread_id,
        workspace_id=current_user.workspace_id,
        user_id=current_user.user_id,
    )

    start_time = perf_counter()
    turn_result = empty_turn_result()
    status = "error"
    try:
        await run_in_threadpool(
            register_active_run,
            run_control,
            run_context,
            current_user,
        )
        yield {
            "type": "started",
            "request_id": request_id,
            "thread_id": thread_id,
            "model": request.model,
        }
        graph_input = {
            "messages": [HumanMessage(content=request.question.strip())]
        }
        has_final_answer = False
        canceled_at_safe_boundary = False
        checkpoint_id = str(run_config["configurable"]["thread_id"])

        async with conversation_lock(checkpoint_id):
            await run_in_threadpool(
                ensure_thread_workspace,
                thread_id,
                current_user,
            )
            if not await prepare_checkpoint_for_new_input(
                single_round_graph,
                run_config,
            ):
                raise RuntimeError(
                    "当前会话的上一批工具消息不完整，Runtime 已拒绝写入"
                    "这次新问题。"
                )

            await run_in_threadpool(
                save_user_message,
                thread_id=thread_id,
                content=request.question,
                workspace_id=current_user.workspace_id,
                created_by_user_id=current_user.user_id,
            )

            for recursion_index in range(max_agent_rounds(run_config)):
                if (
                    recursion_index > 0
                    and run_control.cancel_requested.is_set()
                ):
                    canceled_at_safe_boundary = True
                    break

                graph_stream = single_round_graph.astream(
                    graph_input,
                    config=run_config,
                    stream_mode=["tasks", "updates"],
                    version="v2",
                )
                graph_input = {}

                try:
                    async for stream_part in graph_stream:
                        if stream_part.get("type") == "tasks":
                            event = translate_task_progress_event(stream_part)
                            if event is not None:
                                event["request_id"] = request_id
                                yield event
                        elif stream_part.get("type") == "updates":
                            round_event = translate_llm_round_event(
                                stream_part,
                                recursion_index + 1,
                            )
                            if round_event is not None:
                                round_event["request_id"] = request_id
                                yield round_event
                            for event in translate_knowledge_trace_events(stream_part):
                                event["request_id"] = request_id
                                yield event
                            for event in translate_update_progress_events(stream_part):
                                event["request_id"] = request_id
                                yield event
                            if (
                                run_control.cancel_requested.is_set()
                                and is_safe_cancel_boundary(stream_part)
                            ):
                                canceled_at_safe_boundary = True
                finally:
                    await graph_stream.aclose()

                conversation_messages = await read_graph_messages(
                    single_round_graph,
                    run_config,
                )
                if tool_protocol_state(conversation_messages) != "complete":
                    raise RuntimeError(
                        "单轮 Agent Loop 结束后仍存在未闭合的工具消息。"
                    )
                if canceled_at_safe_boundary:
                    break

                if final_answer_is_ready(conversation_messages):
                    has_final_answer = True
                    break

            conversation_messages = await read_graph_messages(
                single_round_graph,
                run_config,
            )

            if canceled_at_safe_boundary:
                status = "canceled"
                turn_result = build_turn_result(conversation_messages)
            elif has_final_answer:
                status = "success"
                turn_result = build_turn_result(conversation_messages)
            else:
                yield {
                    "type": "progress",
                    "request_id": request_id,
                    "stage": "Runtime Pause Summary",
                    "message": "本次循环次数已用完，正在整理当前进度…",
                }
                if run_control.cancel_requested.is_set():
                    status = "canceled"
                    turn_result = build_turn_result(conversation_messages)
                else:
                    recursion_limit_summary = await generate_recursion_limit_summary(
                        conversation_messages,
                        model_name=request.model,
                    )
                    await single_round_graph.aupdate_state(
                        run_config,
                        {"messages": [recursion_limit_summary]},
                    )
                    turn_result = build_turn_result(
                        await read_graph_messages(single_round_graph, run_config)
                    )
                    status = (
                        "canceled"
                        if run_control.cancel_requested.is_set()
                        else "paused"
                    )

            if status == "canceled":
                turn_result["answer"] = (
                    "本次分析已在消息完整的安全位置停止。已完成的 Knowledge "
                    "与 SQL 结果仍保留在当前会话中。"
                )

            elapsed_ms = round((perf_counter() - start_time) * 1000)
            final_response = build_run_response(
                request_id=request_id,
                thread_id=thread_id,
                model=request.model,
                status=status,
                latency_ms=elapsed_ms,
                details=turn_result,
            )
            await run_in_threadpool(
                save_assistant_message,
                thread_id=thread_id,
                content=final_response["answer"],
                details=final_response,
                workspace_id=current_user.workspace_id,
            )
            record_run(
                request_id=request_id,
                thread_id=thread_id,
                workspace_id=current_user.workspace_id,
                user_id=current_user.user_id,
                model=request.model,
                status=status,
                latency_ms=elapsed_ms,
                details=turn_result,
                created_at=created_at,
            )

        yield {
            "type": "final",
            "request_id": request_id,
            "response": final_response,
        }
    except Exception:
        elapsed_ms = round((perf_counter() - start_time) * 1000)
        record_run(
            request_id=request_id,
            thread_id=thread_id,
            workspace_id=current_user.workspace_id,
            user_id=current_user.user_id,
            model=request.model,
            status="error",
            latency_ms=elapsed_ms,
            details=turn_result,
            created_at=created_at,
        )
        raise
    finally:
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS.pop(request_id, None)


async def stream_agent(
    request: ChatRequest,
    current_user: CurrentUser,
    *,
    single_round_graph: Any,
    thread_id: str,
    run_config: dict,
) -> AsyncIterator[str]:
    """Encode the shared Agent event stream as newline-delimited JSON."""

    run_context: AgentRunContext = run_config["configurable"]["agent_run_context"]

    event_stream = run_agent_events(
        request,
        current_user,
        single_round_graph=single_round_graph,
        thread_id=thread_id,
        run_config=run_config,
    )
    try:
        async for event in event_stream:
            yield encode_event(event)
    except Exception as error:
        # StreamingResponse may already have sent HTTP 200, so a later runtime
        # failure must be reported as the stream's terminal error event.
        yield encode_event(
            {
                "type": "error",
                "request_id": run_context.request_id,
                "message": f"分析执行失败：{safe_error_text(error)}",
            }
        )
    finally:
        await event_stream.aclose()


async def run_agent(
    request: ChatRequest,
    current_user: CurrentUser,
    *,
    single_round_graph: Any,
) -> dict:
    """Run the same event-producing path and return its final response."""

    thread_id, run_config = await run_in_threadpool(
        build_agent_config,
        request,
        current_user,
    )
    final_response = None

    async for event in run_agent_events(
        request,
        current_user,
        single_round_graph=single_round_graph,
        thread_id=thread_id,
        run_config=run_config,
    ):
        if event["type"] == "final":
            final_response = event["response"]

    if final_response is None:
        raise RuntimeError("Agent run ended without a final response.")
    return final_response
