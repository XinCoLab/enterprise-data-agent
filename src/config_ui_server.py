"""Local-only configuration UI for database and Knowledge profiles."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
from time import perf_counter
from uuid import uuid4
import webbrowser
from typing import Any, Iterator, Literal
from urllib.parse import unquote
from zipfile import BadZipFile, ZipFile

import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import psycopg2
import pymysql
from pydantic import BaseModel, Field

from knowledge_runtime.catalog import load_knowledge_cards
from config.project_paths import CONFIG_ROOT, KNOWLEDGE_IMPORT_ROOT, PROJECT_ROOT


SETTINGS_PATH = CONFIG_ROOT / "settings.env"
SECRETS_PATH = CONFIG_ROOT / "secrets.env"
PROFILES_ROOT = CONFIG_ROOT / "profiles"
PROFILE_SECRETS_ROOT = CONFIG_ROOT / "profile_secrets"
ACTIVE_PROFILE_PATH = CONFIG_ROOT / ".active_profile"
FRONTEND_BUILD = PROJECT_ROOT / "frontend" / "dist" / "client"
ALLOWED_MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")
MAX_KNOWLEDGE_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_KNOWLEDGE_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_KNOWLEDGE_FILES = 2000
RECENT_RUNS: deque[dict] = deque(maxlen=50)
GRAPH_RUN_LOCK = threading.RLock()
ACTIVE_RUNS_LOCK = threading.Lock()
ACTIVE_RUNS: dict[str, ActiveRun] = {}

PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ENV_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ProfilePayload(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    backend: Literal["postgresql", "mysql", "duckdb"]
    host: str = Field(default="", max_length=255)
    port: int = Field(default=0, ge=0, le=65535)
    username: str = Field(default="", max_length=255)
    database: str = Field(default="", max_length=255)
    password: str = Field(default="", max_length=2048)
    duckdb_path: str = Field(default="", max_length=2048)
    knowledge_root: str = Field(default="", max_length=2048)


class ProfileReference(BaseModel):
    profile_id: str


class KnowledgeRequest(BaseModel):
    knowledge_root: str = Field(min_length=1, max_length=2048)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    thread_id: str = Field(default="", max_length=128)
    model: Literal["deepseek-v4-pro", "deepseek-v4-flash"] = "deepseek-v4-pro"


@dataclass
class ActiveRun:
    """Cooperative cancellation state for one Web Agent request."""

    run_id: str
    thread_id: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)


class ModelSettingsPayload(BaseModel):
    model: Literal["deepseek-v4-pro", "deepseek-v4-flash"]
    api_key: str = Field(default="", max_length=4096)


app = FastAPI(title="DataAgent", docs_url=None, redoc_url=None)


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _quoted_env_value(value: object) -> str:
    text = str(value)
    if not text:
        return ""
    if re.search(r"\s|#|=|\"", text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _update_env(path: Path, updates: dict[str, object]) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        match = ENV_ASSIGNMENT.match(line)
        if match and match.group(1) in updates:
            key = match.group(1)
            output.append(f"{key}={_quoted_env_value(updates[key])}")
            seen.add(key)
        else:
            output.append(line)
    if output and output[-1]:
        output.append("")
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={_quoted_env_value(value)}")
    _atomic_text(path, "\n".join(output).rstrip() + "\n")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _resolve_local_path(raw_path: str, default: Path | None = None) -> Path:
    text = raw_path.strip()
    if not text:
        if default is None:
            raise ValueError("路径不能为空。")
        return default.resolve()
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _knowledge_summary(root: Path) -> dict:
    cards = load_knowledge_cards(root)
    counts = Counter(card.knowledge_type for card in cards.values())
    return {
        "path": str(root),
        "card_count": len(cards),
        "types": dict(sorted(counts.items())),
    }


def _safe_error_text(error: Exception) -> str:
    text = " ".join(str(error).split())
    text = re.sub(
        r"(?i)\b(password|token|api[_-]?key|secret)\s*[:=]\s*([^\s,;]+)",
        r"\1=[REDACTED]",
        text,
    )
    return text[:800]


def _model_api_key() -> str:
    return (
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


def _agent_config(request: ChatRequest) -> tuple[str, dict]:
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


def _stream_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


_NODE_ACTIVITY = {
    "Main Agent LLM": "正在分析现有信息并决定下一步…",
    "Tool Safety": "正在校验工具请求…",
    "Tool Execution": "正在执行已通过校验的工具…",
}

_TOOL_ACTIVITY = {
    "browse_knowledge": "正在浏览 Knowledge 目录…",
    "search_knowledge": "正在搜索相关 Knowledge…",
    "read_knowledge": "正在读取 KnowledgeCard…",
    "execute_readonly_sql": "正在执行只读 SQL…",
}


def _task_progress_event(part: dict) -> dict | None:
    data = part.get("data")
    if not isinstance(data, dict) or "input" not in data:
        return None
    node_name = str(data.get("name", ""))
    message = _NODE_ACTIVITY.get(node_name)
    if not message:
        return None
    return {"type": "progress", "stage": node_name, "message": message}


def _tool_result_progress(message: Any) -> dict | None:
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


def _visible_ai_content(content: Any) -> str:
    """Return only the assistant content that is part of the public message."""

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


def _llm_round_event(part: dict, round_number: int) -> dict | None:
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
        "content": _visible_ai_content(reply.content),
        "tool_calls": tool_calls,
        "message": (
            "本轮模型输出已生成，正在执行工具。"
            if tool_calls
            else "本轮模型已生成最终回答。"
        ),
    }


def _update_progress_events(part: dict) -> list[dict]:
    data = part.get("data")
    if not isinstance(data, dict):
        return []
    events: list[dict] = []
    execution_update = data.get("Tool Execution")
    if isinstance(execution_update, dict):
        for message in execution_update.get("messages", []):
            event = _tool_result_progress(message)
            if event is not None:
                events.append(event)
    return events


def _safe_cancel_boundary(part: dict) -> bool:
    """Stop only after a complete Tool round or a terminal Assistant reply."""

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


def _agent_graph():
    from graph.data_agent_graph import round_graph

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
        content = _visible_ai_content(raw_reply.content)
    except Exception:
        content = ""
    tool_markup = ("<|DSML|", "tool_calls>", "invoke name=")
    if any(marker in content for marker in tool_markup):
        content = ""
    return AIMessage(content=content or fallback)


def _stream_agent(request: ChatRequest) -> Iterator[str]:
    from langchain_core.messages import AIMessage, HumanMessage

    thread_id, config = _agent_config(request)
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
        yield _stream_line(
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
                        event = _task_progress_event(part)
                        if event is not None:
                            event["run_id"] = run_id
                            yield _stream_line(event)
                    elif part.get("type") == "updates":
                        round_event = _llm_round_event(part, recursion_index + 1)
                        if round_event is not None:
                            round_event["run_id"] = run_id
                            yield _stream_line(round_event)
                        for event in _update_progress_events(part):
                            event["run_id"] = run_id
                            yield _stream_line(event)
                        if (
                            control.cancel_requested.is_set()
                            and _safe_cancel_boundary(part)
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
                yield _stream_line(
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
        yield _stream_line({"type": "final", "run_id": run_id, "response": response})
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
        yield _stream_line(
            {
                "type": "error",
                "run_id": run_id,
                "message": f"分析执行失败：{_safe_error_text(error)}",
            }
        )
    finally:
        if stream is not None and hasattr(stream, "close"):
            stream.close()
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS.pop(run_id, None)


def _run_agent(request: ChatRequest) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    thread_id, config = _agent_config(request)
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


def _load_profile(profile_id: str) -> dict:
    if not PROFILE_ID.fullmatch(profile_id):
        raise HTTPException(status_code=400, detail="配置方案 ID 不合法。")
    path = PROFILES_ROOT / f"{profile_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="配置方案不存在。")
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_secret_path(profile_id: str) -> Path:
    return PROFILE_SECRETS_ROOT / f"{profile_id}.env"


def _password_key(backend: str) -> str | None:
    if backend == "postgresql":
        return "DATA_AGENT_POSTGRES_PASSWORD"
    if backend == "mysql":
        return "DATA_AGENT_MYSQL_PASSWORD"
    return None


def _profile_password(profile_id: str, backend: str) -> str:
    key = _password_key(backend)
    if key is None:
        return ""
    return _read_env(_profile_secret_path(profile_id)).get(key, "")


def _active_payload() -> dict:
    settings = _read_env(SETTINGS_PATH)
    secrets = _read_env(SECRETS_PATH)
    backend = settings.get("DATA_AGENT_DATABASE_BACKEND", "postgresql").lower()
    if backend == "mysql":
        host_key, port_key, user_key, database_key = (
            "DATA_AGENT_MYSQL_HOST",
            "DATA_AGENT_MYSQL_PORT",
            "DATA_AGENT_MYSQL_USER",
            "DATA_AGENT_MYSQL_DATABASE",
        )
        default_port = 3306
    else:
        host_key, port_key, user_key, database_key = (
            "DATA_AGENT_POSTGRES_HOST",
            "DATA_AGENT_POSTGRES_PORT",
            "DATA_AGENT_POSTGRES_USER",
            "DATA_AGENT_POSTGRES_DATABASE",
        )
        default_port = 5432
    knowledge_root = settings.get("DATA_AGENT_KNOWLEDGE_ROOT", "").strip()
    resolved_knowledge = _resolve_local_path(
        knowledge_root,
        PROJECT_ROOT / "knowledge",
    )
    password_key = _password_key(backend)
    return {
        "id": ACTIVE_PROFILE_PATH.read_text(encoding="utf-8").strip()
        if ACTIVE_PROFILE_PATH.is_file()
        else "current",
        "label": "当前生效配置",
        "description": "当前 settings.env 与 secrets.env 的实际值。",
        "backend": backend,
        "host": settings.get(host_key, ""),
        "port": int(settings.get(port_key, str(default_port)) or default_port),
        "username": settings.get(user_key, ""),
        "database": settings.get(database_key, ""),
        "duckdb_path": settings.get("DATA_AGENT_DATABASE_PATH", ""),
        "knowledge_root": str(resolved_knowledge),
        "password_saved": bool(password_key and secrets.get(password_key)),
    }


def _public_profile(profile: dict) -> dict:
    backend = str(profile.get("backend", ""))
    profile_id = str(profile.get("id", ""))
    result = dict(profile)
    result["password_saved"] = bool(_profile_password(profile_id, backend))
    result["knowledge_root"] = str(
        _resolve_local_path(str(profile.get("knowledge_root", "")), PROJECT_ROOT / "knowledge")
    )
    return result


def _list_profiles() -> list[dict]:
    profiles = []
    for path in sorted(PROFILES_ROOT.glob("*.json")):
        try:
            profiles.append(_public_profile(json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return profiles


def _payload_password(payload: ProfilePayload) -> str:
    if payload.password:
        return payload.password
    profile_password = _profile_password(payload.id, payload.backend)
    if profile_password:
        return profile_password
    key = _password_key(payload.backend)
    return _read_env(SECRETS_PATH).get(key, "") if key else ""


def _validate_required_connection_fields(payload: ProfilePayload) -> None:
    if payload.backend in {"postgresql", "mysql"}:
        if not payload.host.strip() or not payload.username.strip() or not payload.database.strip():
            raise ValueError("Host、用户名和 Database 均不能为空。")
        if payload.port <= 0:
            raise ValueError("Port 必须大于 0。")
    elif not payload.duckdb_path.strip():
        raise ValueError("DuckDB 文件路径不能为空。")


def _test_postgresql(payload: ProfilePayload, password: str) -> dict:
    connection = psycopg2.connect(
        host=payload.host,
        port=payload.port,
        user=payload.username,
        password=password,
        dbname=payload.database,
        connect_timeout=5,
        application_name="data-agent-config-test",
    )
    try:
        connection.set_session(readonly=True)
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1, current_database(), current_user")
            ok, database_name, current_user = cursor.fetchone()
            cursor.execute("SHOW transaction_read_only")
            transaction_read_only = cursor.fetchone()[0]
        connection.rollback()
    finally:
        connection.close()
    return {
        "ok": ok == 1,
        "database": database_name,
        "current_user": current_user,
        "readonly_transaction": str(transaction_read_only).lower() == "on",
    }


def _test_mysql(payload: ProfilePayload, password: str) -> dict:
    connection = pymysql.connect(
        host=payload.host,
        port=payload.port,
        user=payload.username,
        password=password,
        database=payload.database,
        charset="utf8mb4",
        connect_timeout=5,
        read_timeout=7,
        write_timeout=5,
        autocommit=False,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT 1, DATABASE(), CURRENT_USER()")
            ok, database_name, current_user = cursor.fetchone()
            cursor.execute("SHOW GRANTS FOR CURRENT_USER")
            grants = [str(row[0]) for row in cursor.fetchall()]
        connection.rollback()
    finally:
        connection.close()
    dangerous = re.compile(
        r"\b(ALL PRIVILEGES|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRIGGER|EVENT|GRANT OPTION)\b",
        re.IGNORECASE,
    )
    risky_grants = [grant for grant in grants if dangerous.search(grant)]
    return {
        "ok": ok == 1,
        "database": database_name,
        "current_user": current_user,
        "readonly_transaction": True,
        "readonly_account_likely": not risky_grants,
        "grant_warning": "；".join(risky_grants) if risky_grants else "",
    }


def _test_duckdb(payload: ProfilePayload) -> dict:
    path = _resolve_local_path(payload.duckdb_path)
    if not path.is_file():
        raise ValueError(f"DuckDB 文件不存在：{path}")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        ok = connection.execute("SELECT 1").fetchone()[0]
    finally:
        connection.close()
    return {"ok": ok == 1, "database": str(path), "readonly_transaction": True}


def _profile_document(payload: ProfilePayload) -> dict:
    return {
        "id": payload.id,
        "label": payload.label.strip(),
        "description": payload.description.strip(),
        "backend": payload.backend,
        "host": payload.host.strip(),
        "port": payload.port,
        "username": payload.username.strip(),
        "database": payload.database.strip(),
        "duckdb_path": payload.duckdb_path.strip(),
        "knowledge_root": payload.knowledge_root.strip(),
    }


def _apply_payload(payload: ProfilePayload, password: str) -> None:
    knowledge_root = _resolve_local_path(payload.knowledge_root, PROJECT_ROOT / "knowledge")
    updates: dict[str, object] = {
        "DATA_AGENT_DATABASE_BACKEND": payload.backend,
        "DATA_AGENT_KNOWLEDGE_ROOT": str(knowledge_root),
    }
    if payload.backend == "postgresql":
        updates.update(
            {
                "DATA_AGENT_POSTGRES_HOST": payload.host.strip(),
                "DATA_AGENT_POSTGRES_PORT": payload.port,
                "DATA_AGENT_POSTGRES_USER": payload.username.strip(),
                "DATA_AGENT_POSTGRES_DATABASE": payload.database.strip(),
            }
        )
    elif payload.backend == "mysql":
        updates.update(
            {
                "DATA_AGENT_MYSQL_HOST": payload.host.strip(),
                "DATA_AGENT_MYSQL_PORT": payload.port,
                "DATA_AGENT_MYSQL_USER": payload.username.strip(),
                "DATA_AGENT_MYSQL_DATABASE": payload.database.strip(),
            }
        )
    else:
        updates["DATA_AGENT_DATABASE_PATH"] = str(_resolve_local_path(payload.duckdb_path))
    _update_env(SETTINGS_PATH, updates)
    for key, value in updates.items():
        os.environ[key] = str(value)
    password_key = _password_key(payload.backend)
    if password_key and password:
        _update_env(SECRETS_PATH, {password_key: password})
        os.environ[password_key] = password
    _atomic_text(ACTIVE_PROFILE_PATH, payload.id + "\n")


def _refresh_knowledge_runtime(root: Path) -> None:
    from knowledge_runtime.current_knowledge import reload_knowledge

    reload_knowledge(root)


def _refresh_model_runtime() -> None:
    from graph.nodes.main_agent_llm_node import refresh_model_runtime

    refresh_model_runtime()


@app.get("/api/state")
def get_state():
    active = _active_payload()
    settings = _read_env(SETTINGS_PATH)
    try:
        knowledge = _knowledge_summary(Path(active["knowledge_root"]))
    except Exception as error:
        knowledge = {
            "path": active["knowledge_root"],
            "error": _safe_error_text(error),
        }
    model = settings.get("DATA_AGENT_MODEL", "deepseek-v4-pro")
    if model not in ALLOWED_MODELS:
        model = "deepseek-v4-pro"
    return {
        "active": active,
        "profiles": _list_profiles(),
        "model": model,
        "models": list(ALLOWED_MODELS),
        "knowledge": knowledge,
        "model_configured": bool(_model_api_key()),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "data-agent"}


@app.get("/api/runs")
def recent_runs():
    return {"runs": list(RECENT_RUNS)}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not _model_api_key():
        raise HTTPException(
            status_code=400,
            detail="尚未配置模型 API Key，请先前往模型设置完成配置。",
        )
    try:
        return _run_agent(request)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"分析执行失败：{_safe_error_text(error)}",
        ) from error


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    if not _model_api_key():
        raise HTTPException(
            status_code=400,
            detail="尚未配置模型 API Key，请先前往模型设置完成配置。",
        )
    _agent_config(request)
    return StreamingResponse(
        _stream_agent(request),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="run_id 格式不合法。")
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


@app.post("/api/model-settings")
def save_model_settings(payload: ModelSettingsPayload):
    api_key = payload.api_key.strip()
    if not api_key and not _model_api_key():
        raise HTTPException(status_code=400, detail="请输入 DeepSeek API Key。")

    _update_env(SETTINGS_PATH, {"DATA_AGENT_MODEL": payload.model})
    os.environ["DATA_AGENT_MODEL"] = payload.model
    if api_key:
        _update_env(SECRETS_PATH, {"DEEPSEEK_API_KEY": api_key})
        os.environ["DEEPSEEK_API_KEY"] = api_key
    with GRAPH_RUN_LOCK:
        _refresh_model_runtime()

    return {
        "status": "success",
        "message": "模型配置已保存并立即生效。",
        "model": payload.model,
        "model_configured": True,
    }


@app.post("/api/test-database")
def test_database(payload: ProfilePayload):
    try:
        _validate_required_connection_fields(payload)
        password = _payload_password(payload)
        if payload.backend == "postgresql":
            details = _test_postgresql(payload, password)
        elif payload.backend == "mysql":
            details = _test_mysql(payload, password)
        else:
            details = _test_duckdb(payload)
        return {"status": "success", "message": "数据库连接和只读事务测试通过。", "details": details}
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"数据库测试失败：{error}") from error


@app.post("/api/validate-knowledge")
def validate_knowledge(request: KnowledgeRequest):
    try:
        root = _resolve_local_path(request.knowledge_root)
        return {
            "status": "success",
            "message": "Knowledge 校验通过。",
            "details": _knowledge_summary(root),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Knowledge 校验失败：{error}") from error


@app.post("/api/import-knowledge")
async def import_knowledge(request: Request):
    """Import one bounded ZIP archive into the local runtime area."""

    archive_bytes = await request.body()
    if not archive_bytes:
        raise HTTPException(status_code=400, detail="请选择一个 ZIP 文件。")
    if len(archive_bytes) > MAX_KNOWLEDGE_ARCHIVE_BYTES:
        raise HTTPException(status_code=413, detail="Knowledge ZIP 不能超过 20 MB。")

    raw_name = unquote(request.headers.get("x-knowledge-name", "knowledge"))
    slug = re.sub(r"[^a-z0-9-]+", "-", raw_name.lower()).strip("-")[:48]
    slug = slug or "knowledge"
    destination = (KNOWLEDGE_IMPORT_ROOT / f"{slug}-{uuid4().hex[:8]}").resolve()
    destination.mkdir(parents=True, exist_ok=False)

    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if not entries:
                raise ValueError("ZIP 中没有文件。")
            if len(entries) > MAX_KNOWLEDGE_FILES:
                raise ValueError("ZIP 文件数量超过 2000 个。")
            total_size = sum(entry.file_size for entry in entries)
            if total_size > MAX_KNOWLEDGE_EXTRACTED_BYTES:
                raise ValueError("ZIP 解压后不能超过 100 MB。")

            for entry in entries:
                relative = PurePosixPath(entry.filename.replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("ZIP 包含不安全路径。")
                file_type = (entry.external_attr >> 16) & 0o170000
                if file_type == 0o120000:
                    raise ValueError("ZIP 不允许包含符号链接。")
                target = (destination / Path(*relative.parts)).resolve()
                if destination != target and destination not in target.parents:
                    raise ValueError("ZIP 包含越界路径。")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

        summary = _knowledge_summary(destination)
        return {
            "status": "success",
            "message": f"Knowledge 已导入，共 {summary['card_count']} 张卡。",
            "details": summary,
        }
    except BadZipFile as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise HTTPException(status_code=400, detail="文件不是有效的 ZIP。") from error
    except Exception as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge 导入失败：{_safe_error_text(error)}",
        ) from error


@app.post("/api/save-and-apply")
def save_and_apply(payload: ProfilePayload):
    if not PROFILE_ID.fullmatch(payload.id):
        raise HTTPException(status_code=400, detail="方案 ID 只能包含小写字母、数字和连字符。")
    try:
        _validate_required_connection_fields(payload)
        knowledge_root = _resolve_local_path(payload.knowledge_root, PROJECT_ROOT / "knowledge")
        load_knowledge_cards(knowledge_root)
        password = _payload_password(payload)
        _atomic_json(PROFILES_ROOT / f"{payload.id}.json", _profile_document(payload))
        password_key = _password_key(payload.backend)
        if password_key and password:
            _update_env(_profile_secret_path(payload.id), {password_key: password})
        with GRAPH_RUN_LOCK:
            _apply_payload(payload, password)
            _refresh_knowledge_runtime(knowledge_root)
        return {
            "status": "success",
            "message": "数据库与 Knowledge 配置已保存并立即生效。",
            "restart_required": False,
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"保存失败：{error}") from error


@app.post("/api/apply-profile")
def apply_profile(reference: ProfileReference):
    profile = _load_profile(reference.profile_id)
    payload = ProfilePayload(**profile, password="")
    try:
        password = _profile_password(payload.id, payload.backend)
        knowledge_root = _resolve_local_path(
            payload.knowledge_root,
            PROJECT_ROOT / "knowledge",
        )
        load_knowledge_cards(knowledge_root)
        with GRAPH_RUN_LOCK:
            _apply_payload(payload, password)
            _refresh_knowledge_runtime(knowledge_root)
        return {
            "status": "success",
            "message": f"已切换到“{payload.label}”并立即生效。",
            "restart_required": False,
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"切换失败：{error}") from error


if not FRONTEND_BUILD.is_dir():
    raise RuntimeError(f"Configuration UI build is missing: {FRONTEND_BUILD}")

app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
