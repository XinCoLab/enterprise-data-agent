"""Agent execution loop: one user turn from input to final answer or safe pause."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from fastapi import HTTPException

from api.schemas import ChatRequest
from config.project_paths import CONFIG_ROOT
from runtime.stream_events import (
    encode_event,
    llm_round_event,
    safe_cancel_boundary,
    task_progress_event,
    update_progress_events,
    visible_ai_content,
)

SETTINGS_PATH = CONFIG_ROOT / "settings.env"
SECRETS_PATH = CONFIG_ROOT / "secrets.env"
RECENT_RUNS: deque[dict] = deque(maxlen=50)
GRAPH_RUN_LOCK = threading.RLock()
ACTIVE_RUNS_LOCK = threading.Lock()
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass
class ActiveRun:
    """Cooperative cancellation state for one Web Agent request."""

    run_id: str
    thread_id: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)


ACTIVE_RUNS: dict[str, ActiveRun] = {}


def list_recent_runs() -> list[dict]:
    """Return a stable snapshot for the HTTP layer."""

    return list(RECENT_RUNS)


def request_cancel(run_id: str) -> dict[str, str]:
    """Ask an active run to stop at its next complete message boundary."""

    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id 格式不合法。")
    with ACTIVE_RUNS_LOCK:
        control = ACTIVE_RUNS.get(run_id)
        if control is None:
            return {
                "status": "not_running",
                "message": "任务已结束或不存在。",
            }
        control.cancel_requested.set()
    return {
        "status": "cancel_requested",
        "message": "已请求停止，将在当前模型或工具步骤结束后安全停止。",
    }


def _read_env(path: Path) -> dict[str, str]:
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
        or _read_env(SECRETS_PATH).get("DEEPSEEK_API_KEY", "").strip()
    )


def _turn_messages(messages: list) -> list:
    from langchain_core.messages import HumanMessage

    last_human = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    return messages[last_human:]


def _turn_result(messages: list) -> dict:
    from langchain_core.messages import AIMessage, ToolMessage

    current = _turn_messages(messages)
    tool_calls: list[dict] = []
    tool_results: dict[str, dict] = {}
    final_answer = ""
    navigation_trace = None

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
            try:
                payload = json.loads(str(message.content))
            except (TypeError, json.JSONDecodeError):
                payload = {"raw": str(message.content)[:1000]}
            tool_results[message.tool_call_id] = payload

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
    return {
        "answer": final_answer,
        "tool_counts": dict(sorted(tool_counts.items())),
        "sql_queries": sql_queries,
        "result_preview": successful_results[-1] if successful_results else None,
        "knowledge_view": navigation_trace,
    }


def _empty_turn_result() -> dict:
    return {
        "answer": "",
        "tool_counts": {},
        "sql_queries": [],
        "result_preview": None,
        "knowledge_view": None,
    }


def build_agent_config(request: ChatRequest) -> tuple[str, dict]:
    thread_id = request.thread_id.strip() or str(uuid4())
    if not THREAD_ID.fullmatch(thread_id):
        raise HTTPException(status_code=400, detail="thread_id 格式不合法。")
    settings = _read_env(SETTINGS_PATH)
    max_recursions = max(
        1,
        int(settings.get("DATA_AGENT_MAX_RECURSIONS", "10")),
    )
    return thread_id, {
        "configurable": {
            "thread_id": thread_id,
            "model": request.model,
            "max_recursions": max_recursions,
        },
        # One round graph contains only LLM, Safety, and Tool Execution. This
        # remains an internal emergency guard, not the product recursion budget.
        "recursion_limit": max(
            4,
            int(settings.get("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "6")),
        ),
    }


def _max_recursions(config: dict) -> int:
    return int(config.get("configurable", {}).get("max_recursions", 10))


def _tool_protocol_state(messages: list[Any]) -> str:
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


def _prepare_checkpoint_for_new_input(graph: Any, config: dict) -> bool:
    """Finish a recoverable half-round before accepting a new HumanMessage."""

    snapshot = graph.get_state(config)
    messages = list(snapshot.values.get("messages", []))
    protocol_state = _tool_protocol_state(messages)
    if protocol_state == "complete":
        return True
    if protocol_state == "invalid":
        return False

    pending_nodes = set(getattr(snapshot, "next", ()) or ())
    if not pending_nodes.intersection({"Tool Safety", "Tool Execution"}):
        return False

    graph.invoke(None, config=config)
    repaired = graph.get_state(config)
    repaired_messages = list(repaired.values.get("messages", []))
    return _tool_protocol_state(repaired_messages) == "complete"


def _record_run(
    *,
    run_id: str,
    thread_id: str,
    model: str,
    status: str,
    latency_ms: int,
    details: dict,
    created_at: str | None = None,
) -> None:
    RECENT_RUNS.appendleft(
        {
            "run_id": run_id,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "thread_id": thread_id,
            "model": model,
            "status": status,
            "latency_ms": latency_ms,
            "tool_counts": details["tool_counts"],
            "sql_count": len(details["sql_queries"]),
        }
    )


def _run_response(
    *,
    run_id: str,
    thread_id: str,
    model: str,
    status: str,
    latency_ms: int,
    details: dict,
) -> dict:
    return {
        "run_id": run_id,
        "status": status,
        "thread_id": thread_id,
        "model": model,
        "latency_ms": latency_ms,
        **details,
    }


def _agent_graph():
    from graph.round_graph import round_graph

    return round_graph


def _generate_pause_summary(messages: list[Any], *, model_name: str):
    from langchain_core.messages import AIMessage
    from prompts.runtime_pause_summary import generate_runtime_pause_summary

    fallback = (
        "本次运行已达到最大循环次数，当前进度和工具结果已经保存。"
        "你可以继续原任务，或在当前会话中调整要求。"
    )
    try:
        raw_reply = generate_runtime_pause_summary(messages, model_name=model_name)
        content = visible_ai_content(raw_reply.content)
    except Exception:
        content = ""
    tool_markup = ("<|DSML|", "tool_calls>", "invoke name=")
    if any(marker in content for marker in tool_markup):
        content = ""
    return AIMessage(content=content or fallback)


def stream_agent(request: ChatRequest) -> Iterator[str]:
    from langchain_core.messages import AIMessage, HumanMessage

    thread_id, config = build_agent_config(request)
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    control = ActiveRun(run_id=run_id, thread_id=thread_id)
    with ACTIVE_RUNS_LOCK:
        ACTIVE_RUNS[run_id] = control

    started = perf_counter()
    details = _empty_turn_result()
    status = "error"
    recorded = False
    graph = _agent_graph()
    stream = None
    try:
        yield encode_event(
            {
                "type": "started",
                "run_id": run_id,
                "thread_id": thread_id,
                "model": request.model,
            }
        )
        input_payload = {
            "messages": [HumanMessage(content=request.question.strip())]
        }
        completed = False
        canceled_at_safe_boundary = control.cancel_requested.is_set()

        with GRAPH_RUN_LOCK:
            if not _prepare_checkpoint_for_new_input(graph, config):
                raise RuntimeError(
                    "当前会话的上一批工具消息不完整，Runtime 已拒绝写入"
                    "这次新问题。"
                )
            for recursion_index in range(_max_recursions(config)):
                if control.cancel_requested.is_set():
                    canceled_at_safe_boundary = True
                    break

                stream = graph.stream(
                    input_payload,
                    config=config,
                    stream_mode=["tasks", "updates"],
                    version="v2",
                )
                input_payload = {}

                for part in stream:
                    if part.get("type") == "tasks":
                        event = task_progress_event(part)
                        if event is not None:
                            event["run_id"] = run_id
                            yield encode_event(event)
                    elif part.get("type") == "updates":
                        round_event = llm_round_event(part, recursion_index + 1)
                        if round_event is not None:
                            round_event["run_id"] = run_id
                            yield encode_event(round_event)
                        for event in update_progress_events(part):
                            event["run_id"] = run_id
                            yield encode_event(event)
                        if (
                            control.cancel_requested.is_set()
                            and safe_cancel_boundary(part)
                        ):
                            canceled_at_safe_boundary = True
                            break

                if hasattr(stream, "close"):
                    stream.close()
                stream = None

                snapshot = graph.get_state(config)
                messages = list(snapshot.values.get("messages", []))
                if _tool_protocol_state(messages) != "complete":
                    raise RuntimeError(
                        "单轮 Agent Loop 结束后仍存在未闭合的工具消息。"
                    )
                if canceled_at_safe_boundary:
                    break

                reply = messages[-1] if messages else None
                if isinstance(reply, AIMessage) and not reply.tool_calls:
                    completed = True
                    break

            snapshot = graph.get_state(config)
            messages = list(snapshot.values.get("messages", []))

            if canceled_at_safe_boundary:
                status = "canceled"
                details = _turn_result(messages)
            elif completed:
                status = "success"
                details = _turn_result(messages)
            else:
                yield encode_event(
                    {
                        "type": "progress",
                        "run_id": run_id,
                        "stage": "Runtime Pause Summary",
                        "message": "本次循环次数已用完，正在整理当前进度…",
                    }
                )
                pause_summary = _generate_pause_summary(
                    messages,
                    model_name=request.model,
                )
                graph.update_state(config, {"messages": [pause_summary]})
                snapshot = graph.get_state(config)
                details = _turn_result(
                    list(snapshot.values.get("messages", []))
                )
                status = (
                    "canceled"
                    if control.cancel_requested.is_set()
                    else "paused"
                )

        if status == "canceled":
            details["answer"] = (
                "本次分析已在消息完整的安全位置停止。已完成的 Knowledge "
                "与 SQL 结果仍保留在当前会话中。"
            )

        elapsed_ms = round((perf_counter() - started) * 1000)
        _record_run(
            run_id=run_id,
            thread_id=thread_id,
            model=request.model,
            status=status,
            latency_ms=elapsed_ms,
            details=details,
            created_at=created_at,
        )
        recorded = True
        response = _run_response(
            run_id=run_id,
            thread_id=thread_id,
            model=request.model,
            status=status,
            latency_ms=elapsed_ms,
            details=details,
        )
        yield encode_event({"type": "final", "run_id": run_id, "response": response})
    except GeneratorExit:
        control.cancel_requested.set()
        raise
    except Exception as error:
        elapsed_ms = round((perf_counter() - started) * 1000)
        if not recorded:
            _record_run(
                run_id=run_id,
                thread_id=thread_id,
                model=request.model,
                status="error",
                latency_ms=elapsed_ms,
                details=details,
                created_at=created_at,
            )
            recorded = True
        yield encode_event(
            {
                "type": "error",
                "run_id": run_id,
                "message": f"分析执行失败：{safe_error_text(error)}",
            }
        )
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS.pop(run_id, None)


def run_agent(request: ChatRequest) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    thread_id, config = build_agent_config(request)
    run_id = str(uuid4())
    started = perf_counter()
    status = "success"
    graph = _agent_graph()
    input_payload = {
        "messages": [HumanMessage(content=request.question.strip())]
    }
    completed = False

    with GRAPH_RUN_LOCK:
        if not _prepare_checkpoint_for_new_input(graph, config):
            raise RuntimeError(
                "当前会话的上一批工具消息不完整，Runtime 已拒绝写入"
                "这次新问题。"
            )
        for _recursion_index in range(_max_recursions(config)):
            result = graph.invoke(input_payload, config=config)
            input_payload = {}
            messages = list(result.get("messages", []))
            if _tool_protocol_state(messages) != "complete":
                raise RuntimeError(
                    "单轮 Agent Loop 结束后仍存在未闭合的工具消息。"
                )
            reply = messages[-1] if messages else None
            if isinstance(reply, AIMessage) and not reply.tool_calls:
                completed = True
                break

        if completed:
            status = "success"
            details = _turn_result(messages)
        else:
            pause_summary = _generate_pause_summary(
                messages,
                model_name=request.model,
            )
            graph.update_state(config, {"messages": [pause_summary]})
            snapshot = graph.get_state(config)
            details = _turn_result(list(snapshot.values.get("messages", [])))
            status = "paused"

    elapsed_ms = round((perf_counter() - started) * 1000)
    _record_run(
        run_id=run_id,
        thread_id=thread_id,
        model=request.model,
        status=status,
        latency_ms=elapsed_ms,
        details=details,
    )
    return _run_response(
        run_id=run_id,
        thread_id=thread_id,
        model=request.model,
        status=status,
        latency_ms=elapsed_ms,
        details=details,
    )
