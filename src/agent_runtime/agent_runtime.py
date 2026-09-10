"""执行网页聊天请求：流式响应包装在前，唯一的 Agent 循环紧随其后。

阅读顺序：
1. chat.py 调用 build_agent_config()，为请求生成一次配置和 request_id。
2. StreamingResponse 迭代 stream_chat_response()，它驱动 execute_agent_request()。
3. execute_agent_request() 控制单轮图的重复执行，并保存页面聊天记录。

本文件用语：
- request：一次 HTTP 请求；turn：一次用户提问及其相关处理。
- round：一次模型决策，以及本轮可能发生的工具校验、工具执行。
- conversation / thread：可以包含多次用户提问的会话。

参数来源：
- request（ChatRequest）：浏览器提交的问题、会话 ID、模型等请求数据。
- current_user：服务端识别的用户、工作区和权限。
- run_context（AgentRunContext）：build_agent_config 固定的一次运行快照；
  request_id 从这里读取，Node / Tool 也从 run_config 读取这份快照。
- run_config：交给 LangGraph 的配置，包含快照、checkpoint 内部 ID 和执行限制。
  Runtime 的问题/模型仍取自 request，身份取自 current_user；调用者必须传入
  构建该配置时的同一组请求与用户，不在执行过程中重新构建配置。

ID 的区别：
- request_id 标识这一次执行。
- 对外 thread_id 标识页面会话，也用于聊天历史。
- run_config["configurable"]["thread_id"] 是 checkpoint 内部 ID，
  由工作区和对外 thread_id 按现有规则构造，不能拿它当页面会话 ID。
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
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
    bindings_match,
    read_conversation_metadata,
    delete_conversation as delete_conversation_history,
    get_workspaceID,
    save_assistant_message,
    save_user_message,
)
from memory.conversation_binding import read_active_conversation_binding
from config.project_paths import CONFIG_ROOT
from security.workspace_access import CurrentUser
from agent_runtime.agent_run_context import AgentRunContext
from agent_runtime.context_usage import context_usage_for_message
from agent_runtime.translate_graph_events import (
    encode_event,
    extract_visible_ai_content,
    is_safe_cancel_boundary,
    translate_graph_progress_events,
)

logger = logging.getLogger("Agent")

SETTINGS_PATH = CONFIG_ROOT / "settings.env"
SECRETS_PATH = CONFIG_ROOT / "secrets.env"
ACTIVE_PROFILE_PATH = CONFIG_ROOT / ".active_profile"
# 运行摘要只保留在当前进程的内存中；页面聊天记录由 history 数据库保存。
RECENT_RUNS: deque[dict] = deque(maxlen=50)
# 同步接口和线程池会访问这些共享数据，因此用 threading.Lock 短暂保护读写。
ACTIVE_RUNS_LOCK = threading.Lock()  # 运行登记表及取消标记
RECENT_RUNS_LOCK = threading.Lock()  # 最近运行摘要
RESOURCE_CONFIG_LOCK = threading.Lock()  # 协调配置修改与运行登记
# 同一 checkpoint 会话的请求串行；不同 ID 使用不同锁，可以并发等待 I/O。
# 锁表属于当前进程，不能协调多个服务进程。
CONVERSATION_LOCKS: dict[str, asyncio.Lock] = {}
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass
class ActiveRun:
    """记录一次正在执行的请求；cancel_requested 仅表示用户提出了停止请求。"""

    request_id: str
    thread_id: str
    workspace_id: str
    user_id: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)


ACTIVE_RUNS: dict[str, ActiveRun] = {}


# 聊天响应与执行主流程


async def stream_chat_response(
    request: ChatRequest,
    current_user: CurrentUser,
    *,
    single_round_graph: Any,
    thread_id: str,
    run_config: dict,
) -> AsyncIterator[str]:
    """供聊天路由使用：驱动 Agent 执行，将产出的事件编码为 NDJSON 文本。

    这是异步生成器：调用只创建生成器；StreamingResponse 开始 async for
    迭代后才执行，不能直接 await。single_round_graph 是应用启动时创建的图，
    thread_id 和 run_config 必须来自本请求唯一一次 build_agent_config 调用。
    每个 yield 把一行 JSON 交给 HTTP 层；调用者继续迭代时再向下执行。

    内层 execute_agent_request 负责 Agent 执行并产出字典事件；本层负责转成网页
    所需的 NDJSON 文本及 error 事件。两层逐条传递同一次执行的事件，不会重复运行 Agent。
    本函数会驱动模型/工具调用和存储写入，不是纯编码函数。
    普通运行异常转成 error 事件；结束、异常或调用者关闭响应流时关闭核心流。
    """

    run_context: AgentRunContext = run_config["configurable"]["agent_run_context"]

    # 创建内层异步生成器：它负责执行 Agent，逐条产出字典事件。
    # 这里只创建对象，还没有开始执行；下一步 async for 取值时才会启动它。
    request_events = execute_agent_request(
        request,
        current_user,
        single_round_graph=single_round_graph,
        thread_id=thread_id,
        run_config=run_config,
    )
    try:
        # 读取内层生成器：每次取值让 execute_agent_request 继续执行到下一个 yield。
        async for event in request_events:
            # 把这条字典事件编码成一行 NDJSON，再由外层生成器 yield 给 StreamingResponse。
            # StreamingResponse 发送后再次取值，本循环才继续读取内层的下一条事件。
            yield encode_event(event)
    except Exception as error:
        # HTTP 响应可能已经开始，无法再改为 HTTP 错误状态；沿用 error 事件告知前端。
        yield encode_event(
            {
                "type": "error",
                "request_id": run_context.request_id,
                "message": f"分析执行失败：{redact_error_text(error)}",
            }
        )
    finally:
        # 将响应流的结束/关闭传递给核心生成器，触发图流关闭和 ACTIVE_RUNS 清理。
        await request_events.aclose()


async def execute_agent_request(
    request: ChatRequest,
    current_user: CurrentUser,
    *,
    single_round_graph: Any,
    thread_id: str,
    run_config: dict,
) -> AsyncIterator[dict]:
    """执行一次用户请求的多轮 Agent 循环，保存聊天记录并产出字典事件。

    由 stream_chat_response 用 async for 驱动。调用仅创建异步生成器，
    开始迭代才登记运行、读取 checkpoint、调用模型/工具及写入聊天历史；
    不能当普通协程直接 await。yield 交给包装函数，包装继续取事件时恢复执行。

    request 提供问题和模型，current_user 提供身份；thread_id 是页面会话 ID，
    run_config 是入口构建的同一份配置，内部 thread_id 用于锁和 checkpoint。
    先产出 started，再产出各轮进度，保存最终聊天记录后产出 final。
    普通异常记录日志及内存运行摘要后上抛，由包装函数输出 error；
    关闭或结束时清除 ACTIVE_RUNS 登记，不删除 RECENT_RUNS 中的摘要。
    """

    from langchain_core.messages import HumanMessage

    # 1. 准备本次执行；配置和 request_id 已由路由创建，这里直接使用。
    run_context: AgentRunContext = run_config["configurable"]["agent_run_context"]
    request_id = run_context.request_id
    created_at = datetime.now(timezone.utc).isoformat()
    run_control = ActiveRun(
        request_id=request_id,
        thread_id=thread_id,
        workspace_id=current_user.workspace_id,
        user_id=current_user.user_id,
    )
    logger.info(f"请求开始 request_id={request_id} thread_id={thread_id}")

    start_time = perf_counter()
    turn_result = build_empty_turn_result()
    status = "error"
    try:
        # 先检查所选数据源 ID 是否变化；未变化才把本次请求登记到 ACTIVE_RUNS 内存表。
        # 取消接口按 request_id 从表中找到本次运行的取消标记；
        # 配置修改接口也会据此判断是否仍有请求未结束。
        # 内部会同步读取数据源配置文件，并可能等待线程锁，因此交给工作线程。
        # await 等检查和登记完成后，才向浏览器发送下面的 started 事件。
        await run_in_threadpool(
            register_run,
            run_control,
            run_context,
            current_user,
        )
        # 将已登记的 request_id 交给响应包装；前端随后可用它请求取消。
        # yield 暂停在这里，包装函数再次取事件时继续；不表示整个请求已完成。
        yield {
            "type": "started",
            "request_id": request_id,
            "thread_id": thread_id,
            "model": request.model,
        }
        graph_input = {
            "messages": [HumanMessage(content=request.question.strip())]
        }
        final_answer_ready = False
        canceled_at_safe_boundary = False
        checkpoint_id = str(run_config["configurable"]["thread_id"])

        # 2. 同会话从恢复到最终保存一直持锁，避免两个请求同时追加/覆盖状态。
        # 等待模型或工具时只占用这个会话的锁，其他会话可继续执行。
        async with get_or_create_conversation_lock(checkpoint_id):
            await run_in_threadpool(
                validate_conversation_binding,
                thread_id,
                current_user,
                run_context.binding,
            )
            if not await resume_pending_tools_if_needed(
                single_round_graph,
                run_config,
            ):
                raise RuntimeError(
                    "当前会话的上一批工具消息不完整，Runtime 已拒绝写入"
                    "这次新问题。"
                )

            # 3. 先完成旧工具，再保存这次用户问题的页面历史；全请求仅保存一次。
            # 页面历史和 LangGraph checkpoint 是两份存储，分别写入，不是原子事务。
            await run_in_threadpool(
                save_user_message,
                thread_id=thread_id,
                content=request.question,
                workspace_id=current_user.workspace_id,
                created_by_user_id=current_user.user_id,
                binding=run_context.binding,
            )

            # 4. 每轮从这里开始：需要记录轮次日志时，放在取消判断后的图调用前。
            # Runtime 数的是模型决策轮次；recursion_limit 限制图内部的节点执行步数。
            for round_index in range(read_max_agent_rounds(run_config)):
                # 首轮仍接收并处理新问题；之后才在轮前响应取消。
                # 请求取消只设标记，不中断正在等待的模型或执行中的工具。
                if (
                    round_index > 0
                    and run_control.cancel_requested.is_set()
                ):
                    canceled_at_safe_boundary = True
                    break

                round_stream = single_round_graph.astream(
                    graph_input,
                    config=run_config,
                    stream_mode=["tasks", "updates"],
                    version="v2",
                )
                # 新用户消息只传入第一轮，防止重复追加。同一 run_config 内部
                # thread_id 会让 LangGraph 从 checkpoint 读回历史，{} 只是不加新消息，
                # 不是清空上下文；每轮产生的图状态由 LangGraph/checkpointer 保存。
                graph_input = {}

                try:
                    # 把图的事件整理成网页展示的进度；这里还没有保存最终回复。
                    async for graph_event in round_stream:
                        for chat_event in translate_graph_progress_events(
                            graph_event,
                            round_index + 1,
                            request_id,
                        ):
                            yield chat_event
                        if (
                            graph_event.get("type") == "updates"
                            and run_control.cancel_requested.is_set()
                            and is_safe_cancel_boundary(graph_event)
                        ):
                            # 此处只记下停止意图，不 break：单轮流自然结束后，
                            # LangGraph 才能完成本轮 checkpoint 的收尾写入。
                            canceled_at_safe_boundary = True
                finally:
                    # 关闭本轮事件流；下一轮是否继续，由下面的判断决定。
                    # 即使异常或外层响应流被关闭，也要让本轮生成器执行其清理。
                    await round_stream.aclose()

                checkpoint_messages = await read_checkpoint_messages(
                    single_round_graph,
                    run_config,
                )
                # 5. 工具消息完整 = AI 发出的每个工具调用都有对应结果。
                # 先检查完整性，再响应取消或结束；否则下次 LLM 会读到悬空调用。
                if classify_tool_message_sequence(checkpoint_messages) != "complete":
                    raise RuntimeError(
                        "单轮 Agent Loop 结束后仍存在未闭合的工具消息。"
                    )
                if canceled_at_safe_boundary:
                    break

                if has_final_answer(checkpoint_messages):
                    final_answer_ready = True
                    break

            checkpoint_messages = await read_checkpoint_messages(
                single_round_graph,
                run_config,
            )

            # 6. 整理本次用户轮次的结果；预算用尽时再生成一次不调用工具的摘要。
            if canceled_at_safe_boundary:
                status = "canceled"
                turn_result = build_turn_result(checkpoint_messages, model_name=request.model)
            elif final_answer_ready:
                status = "success"
                turn_result = build_turn_result(checkpoint_messages, model_name=request.model)
            else:
                yield {
                    "type": "progress",
                    "request_id": request_id,
                    "stage": "Runtime Pause Summary",
                    "message": "本次循环次数已用完，正在整理当前进度…",
                }
                if run_control.cancel_requested.is_set():
                    status = "canceled"
                    turn_result = build_turn_result(checkpoint_messages, model_name=request.model)
                else:
                    round_limit_summary = await generate_round_limit_summary(
                        checkpoint_messages,
                        model_name=request.model,
                    )
                    await single_round_graph.aupdate_state(
                        run_config,
                        {"messages": [round_limit_summary]},
                    )
                    turn_result = build_turn_result(
                        await read_checkpoint_messages(single_round_graph, run_config),
                        model_name=request.model,
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
            final_response = build_chat_response(
                request_id=request_id,
                thread_id=thread_id,
                model=request.model,
                status=status,
                latency_ms=elapsed_ms,
                details=turn_result,
            )
            # 7. 保存最终 AI 回复到聊天历史数据库；await 返回后表示保存成功。
            # 下方“AI回复已保存”日志就是最终回答落库完成的位置；final 还没有发送。
            await run_in_threadpool(
                save_assistant_message,
                thread_id=thread_id,
                content=final_response["answer"],
                details=final_response,
                workspace_id=current_user.workspace_id,
            )
            logger.info(
                f"AI回复已保存 request_id={request_id} "
                f"thread_id={thread_id} status={status}"
            )
            record_recent_run(
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

        # 已保存聊天历史、登记运行摘要并释放会话锁，才把最终结果交给 HTTP 包装。
        yield {
            "type": "final",
            "request_id": request_id,
            "response": final_response,
        }
    except Exception:
        logger.exception(
            f"请求失败 request_id={request_id} thread_id={thread_id}"
        )
        elapsed_ms = round((perf_counter() - start_time) * 1000)
        record_recent_run(
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
        # 8. 正常结束、异常、取消协程或关闭生成器都撤销正在运行的登记。
        # 否则配置接口会一直误以为仍有任务；这不删除会话或最近运行摘要。
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS.pop(request_id, None)


# 请求配置：构造一次，整次执行共用


def build_agent_config(
    request: ChatRequest,
    current_user: CurrentUser,
) -> tuple[str, dict]:
    """由聊天路由调用一次，构造本请求共用的运行配置，返回页面 thread_id 和 run_config。

    读取本地设置，为没有会话 ID 的请求生成会话 ID，并生成唯一的 request_id；
    将用户身份、模型和数据源选择固定为 AgentRunContext，供 Node / Tool 读取。
    不执行图、不保存消息。会话 ID 格式错误抛 HTTP 400，数值配置错误直接抛出。
    """

    thread_id = request.thread_id.strip() or str(uuid4())
    if not THREAD_ID.fullmatch(thread_id):
        raise HTTPException(status_code=400, detail="thread_id 格式不合法。")
    with RESOURCE_CONFIG_LOCK:
        settings = read_env_file(SETTINGS_PATH)
        data_source_ids = read_selected_data_source_ids(current_user)
        binding = read_current_conversation_binding(current_user)
        validate_request_binding(request, binding)
        validate_conversation_binding(thread_id, current_user, binding)
    max_recursions = max(
        1,
        int(settings.get("DATA_AGENT_MAX_RECURSIONS", "10")),
    )
    run_context = AgentRunContext(
        request_id=str(uuid4()),
        thread_id=thread_id,
        workspace_id=current_user.workspace_id,
        user_id=current_user.user_id,
        model=request.model,
        permissions=current_user.permissions,
        allowed_data_source_ids=data_source_ids,
        selected_data_source_ids=data_source_ids,
        binding=binding,
    )
    return thread_id, {
        "configurable": {
            # 默认工作区沿用原始 ID 以兼容已有存档；其他工作区按当前规则加前缀。
            # 这是 checkpoint 的索引，页面及聊天历史继续使用对外 thread_id。
            "thread_id": build_checkpoint_thread_id(
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
        # 单轮图只经过 LLM、Safety、Tool Execution；此项限制图内部节点步数，
        # 外层模型轮次预算仍使用上面的 max_recursions（保留已有配置键）。
        "recursion_limit": max(
            4,
            int(settings.get("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "6")),
        ),
    }


def build_checkpoint_thread_id(thread_id: str, workspace_id: str) -> str:
    """按现有工作区规则构造 checkpoint 内部 ID；默认工作区保留原 ID，以读取旧存档。"""

    if workspace_id == DEFAULT_WORKSPACE_ID:
        return thread_id
    return f"{workspace_id}--{thread_id}"


def read_selected_data_source_ids(current_user: CurrentUser) -> tuple[str, ...]:
    """读取当前界面所选的数据源 ID，作为运行快照的来源。

    读取全局 .active_profile 文件；未配置资源时返回空元组，
    没有配置文件或 ID 为空时沿用 current。只读取选择，不创建数据库连接。
    """

    if not current_user.resources_ready:
        return ()
    if not ACTIVE_PROFILE_PATH.is_file():
        return ("current",)
    data_source_id = ACTIVE_PROFILE_PATH.read_text(encoding="utf-8").strip()
    return (data_source_id or "current",)


def read_current_conversation_binding(current_user: CurrentUser) -> dict | None:
    """Read the active resource identity only; never connect to the database."""

    if not current_user.resources_ready:
        return None
    return read_active_conversation_binding(
        SETTINGS_PATH,
        ACTIVE_PROFILE_PATH,
        SETTINGS_PATH.parent / "profiles",
    )


def validate_request_binding(request: ChatRequest, active_binding: dict | None) -> None:
    """Reject a stale browser tab before the response stream starts."""

    for field in ("data_source_id", "knowledge_base_id"):
        requested_id = getattr(request, field)
        if requested_id is not None and (
            active_binding is None or requested_id != active_binding.get(field)
        ):
            raise HTTPException(
                status_code=409,
                detail="数据源或知识库已切换，请刷新后开始新分析。",
            )


def validate_data_source_ids_unchanged(
    run_context: AgentRunContext,
    current_user: CurrentUser,
) -> None:
    """比较当前所选数据源 ID 与运行快照；不同则抛 RuntimeError，相同返回 None。

    由运行登记函数在配置锁内调用；先比较配置方案 ID，随后由登记函数
    核对固定的数据源与知识库身份。
    """

    current_data_source_ids = read_selected_data_source_ids(current_user)
    if current_data_source_ids != run_context.selected_data_source_ids:
        raise RuntimeError(
            "The selected data source changed before this Agent run started."
        )


# 会话互斥与运行登记


def get_or_create_conversation_lock(checkpoint_id: str) -> asyncio.Lock:
    """取得或创建当前进程内某个 checkpoint 会话的锁对象，不在这里获取锁。

    由执行/删除会话的协程调用；此函数没有 await，锁表读写在事件循环中
    连续完成。调用者用 async with 真正等待并持有返回的锁。
    """

    lock = CONVERSATION_LOCKS.get(checkpoint_id)
    if lock is None:
        lock = asyncio.Lock()
        CONVERSATION_LOCKS[checkpoint_id] = lock
    return lock


def register_run(
    run_control: ActiveRun,
    run_context: AgentRunContext,
    current_user: CurrentUser,
) -> None:
    """检查数据源 ID 未变化后，登记本次运行，供取消和配置修改接口查询。

    由执行核心通过线程池调用。run_control 保存请求身份及取消标记；
    run_context 提供本次运行已选定的数据源 ID，current_user 用于读取当前选择。
    检查通过后将 run_control 存入 ACTIVE_RUNS 内存字典，返回 None。
    不写入聊天数据库，也不执行模型或工具。

    与配置修改接口共用 RESOURCE_CONFIG_LOCK，避免检查 ID 与登记之间插入
    一次配置修改；同时核对数据库和知识库身份，不读取数据库凭据内容。
    绑定改变时拒绝登记。登记表另由 ACTIVE_RUNS_LOCK 保护。
    """

    with RESOURCE_CONFIG_LOCK:
        validate_data_source_ids_unchanged(run_context, current_user)
        if run_context.binding is not None and not bindings_match(
            run_context.binding, read_current_conversation_binding(current_user),
        ):
            raise HTTPException(
                status_code=409,
                detail="数据源或知识库已切换，请开始新分析。",
            )
        with ACTIVE_RUNS_LOCK:
            ACTIVE_RUNS[run_control.request_id] = run_control


def has_active_runs() -> bool:
    """读取内存运行登记表，返回当前进程是否仍有已登记的请求。"""

    with ACTIVE_RUNS_LOCK:
        return bool(ACTIVE_RUNS)


def request_run_cancellation(
    request_id: str,
    current_user: CurrentUser,
) -> dict[str, str]:
    """由取消接口设置运行的停止标记，返回请求结果，不等待任务真正停止。

    request_id 是运行 ID；仅同工作区的本人或管理员可以请求取消。
    ID 格式错误抛 ValueError；不存在或无权操作返回 not_running。
    已运行的模型/工具不会被这里强行打断，执行核心在完整消息边界停止。
    """

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


def record_recent_run(
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
    """向进程内存写入最近运行摘要，最多保留 50 条；不写入数据库。

    由执行核心在最终聊天记录保存后，或普通异常时调用。
    details 只提取工具次数与 SQL 数量；RECENT_RUNS_LOCK 保护摘要列表的读写。
    """

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


def list_recent_runs(current_user: CurrentUser) -> list[dict]:
    """读取当前工作区的最近运行摘要；来源是进程内存，不查询聊天历史数据库。"""

    with RECENT_RUNS_LOCK:
        return [
            run
            for run in RECENT_RUNS
            if run["workspace_id"] == current_user.workspace_id
        ]


# 会话归属、checkpoint 读取与旧工具恢复


def validate_conversation_workspace(
    thread_id: str,
    current_user: CurrentUser,
) -> None:
    """读取页面会话的所属工作区；属于其他工作区时抛 HTTP 404。

    没有历史记录时允许作为新会话继续；成功返回 None。
    只读取归属，不读取图消息、不执行工具；本检查不区分同工作区的用户。
    """

    owner_workspaceID = get_workspaceID(thread_id)
    if (
        owner_workspaceID is not None
        and owner_workspaceID != current_user.workspace_id
    ):
        raise HTTPException(status_code=404, detail="会话不存在。")


def validate_conversation_binding(
    thread_id: str,
    current_user: CurrentUser,
    binding: dict | None,
) -> None:
    """Reject legacy or differently bound history before any checkpoint/tool read."""

    validate_conversation_workspace(thread_id, current_user)
    conversation = read_conversation_metadata(thread_id, current_user.workspace_id)
    if conversation is None:
        return
    stored_binding = conversation["binding"]
    if stored_binding is None:
        raise HTTPException(
            status_code=409,
            detail="该会话尚未确认数据源，只能查看。请开始新分析。",
        )
    if not bindings_match(stored_binding, binding):
        raise HTTPException(
            status_code=409,
            detail="该会话绑定的数据源或知识库与当前配置不同，请切换回对应配置或开始新分析。",
        )


async def resume_pending_tools_if_needed(graph: Any, config: dict) -> bool:
    """读取 checkpoint，必要时执行上次未完成的工具，再判断能否接收新问题。

    由执行核心在持有会话锁、保存新问题之前 await 调用。
    complete 直接返回 True，不执行旧工具；invalid 直接返回 False。
    pending 且下一节点是 Tool Safety/Tool Execution 时，ainvoke(None)
    会真实恢复旧节点、可能执行工具并写 checkpoint，再读消息检查是否 complete；
    其他 pending 情况返回 False。返回值表示能否接收新问题，不表示是否执行过恢复。
    图调用失败直接上抛；此函数本身不保存页面聊天记录。
    """

    snapshot = await graph.aget_state(config)
    messages = list(snapshot.values.get("messages", []))
    protocol_state = classify_tool_message_sequence(messages)
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
    return classify_tool_message_sequence(repaired_messages) == "complete"


async def read_checkpoint_messages(graph: Any, config: dict) -> list[Any]:
    """异步读取指定 checkpoint 的图消息；不读取页面聊天历史，也不执行模型或工具。"""

    saved_state = await graph.aget_state(config)
    return list(saved_state.values.get("messages", []))


def classify_tool_message_sequence(messages: list[Any]) -> str:
    """检查已有消息中工具调用与结果的配对，返回 complete、pending 或 invalid。

    每批 AI 工具调用必须具有非空且唯一的 ID，后面紧接各 ID 对应的 ToolMessage；
    批内结果顺序可以不同，但不能重复、插入别的消息或出现无调用的结果。
    complete 表示全部配齐，pending 表示末批尚缺结果，invalid 表示序列不合法。
    只检查传入消息，不读取存储、不执行工具。
    """

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


async def delete_saved_conversation(
    thread_id: str,
    current_user: CurrentUser,
    checkpointer: Any,
) -> None:
    """供会话删除接口调用，校验工作区后依次删除 checkpoint 和页面聊天历史。

    与聊天执行共用会话锁，防止一边执行一边删除。两个存储依次写入，
    不是原子事务；任一步失败会直接上抛，已完成的删除不会自动回滚。
    """

    private_thread_id = build_checkpoint_thread_id(
        thread_id,
        current_user.workspace_id,
    )
    async with get_or_create_conversation_lock(private_thread_id):
        await run_in_threadpool(validate_conversation_workspace, thread_id, current_user)
        await checkpointer.adelete_thread(private_thread_id)
        await run_in_threadpool(
            delete_conversation_history,
            thread_id,
            current_user.workspace_id,
        )


# 轮次判断、结果整理与响应拼装


def read_max_agent_rounds(config: dict) -> int:
    """读取 Runtime 的模型轮次预算，沿用配置键 max_recursions；不是图内部的节点步数限制。"""

    return int(config["configurable"]["max_recursions"])


def has_final_answer(messages: list[Any]) -> bool:
    """判断非空图消息列表的最后一条是否为不带工具调用的 AI 消息。

    由执行核心在单轮结束且工具消息配齐后调用；只判断消息，不执行或保存。
    """

    from langchain_core.messages import AIMessage

    latest_message = messages[-1]
    return isinstance(latest_message, AIMessage) and not latest_message.tool_calls


def select_current_turn_messages(messages: list) -> list:
    """从已有消息中选取最后一条用户消息及其后续消息；无用户消息时返回全部。"""

    from langchain_core.messages import HumanMessage

    last_human = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    return messages[last_human:]


def build_turn_result(messages: list, *, model_name: str = "") -> dict:
    """从图消息提取当前用户轮次的回答、SQL 结果、工具次数及报表信息。

    只整理传入的消息，不执行工具、不访问存储。返回字典的 answer 用于显示，
    sql_queries/result_preview 用于结果展示，knowledge_view/artifacts 用于导航和报表。
    Tool Execution 约定工具内容为 JSON；内容损坏时 json.loads 直接抛异常。
    """

    from langchain_core.messages import AIMessage, ToolMessage

    current = select_current_turn_messages(messages)
    tool_calls: list[dict] = []
    tool_results: dict[str, dict] = {}
    final_answer = ""
    context_usage = context_usage_for_message(None, model_name=model_name)
    navigation_trace = None
    artifacts: list[dict[str, str]] = []

    for message in current:
        if isinstance(message, AIMessage):
            # Every AI response replaces the snapshot, including missing usage.
            context_usage = context_usage_for_message(message, model_name=model_name)
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
            # 工具节点约定 ToolMessage.content 是 JSON；内部协议损坏就直接报错。
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
        "context_usage": context_usage,
        "artifacts": artifacts,
    }


def build_empty_turn_result() -> dict:
    """构造空的回答与工具结果字典，供尚未完成或异常时的运行记录使用。"""

    return {
        "answer": "",
        "tool_counts": {},
        "sql_queries": [],
        "result_preview": None,
        "knowledge_view": None,
        "artifacts": [],
    }


def build_chat_response(
    *,
    request_id: str,
    thread_id: str,
    model: str,
    status: str,
    latency_ms: int,
    details: dict,
) -> dict:
    """把本次请求结果、运行 ID、页面会话 ID、状态和耗时合成响应字典；不保存或发送。"""

    return {
        "request_id": request_id,
        "status": status,
        "thread_id": thread_id,
        "model": model,
        "latency_ms": latency_ms,
        **details,
    }


async def generate_round_limit_summary(
    messages: list[Any],
    *,
    model_name: str,
):
    """轮次预算用完后额外调用模型生成进度摘要，返回无工具调用的 AIMessage。

    由执行核心 await 调用；这里只生成，调用者随后写入 checkpoint。
    这次补充模型调用不占用外层循环轮次，也不再执行工具。
    模型调用/内容提取失败，或文本含列出的工具标记时使用固定摘要，保留已完成进度。
    """

    from langchain_core.messages import AIMessage
    from prompts.recursion_limit_summary import (
        generate_recursion_limit_summary as create_summary,
    )

    fallback = (
        "本次运行已达到最大循环次数，当前进度和工具结果已经保存。"
        "你可以继续原任务，或在当前会话中调整要求。"
    )
    model_output = None
    try:
        model_output = await create_summary(
            messages,
            model_name=model_name,
        )
        content = extract_visible_ai_content(model_output.content)
    except Exception:
        # 这是轮次用完后的补充模型调用；网络/服务失败时仍用固定摘要收尾，
        # 不因摘要失败丢弃前面已完成的工具结果。
        content = ""
    tool_markup = ("<|DSML|", "tool_calls>", "invoke name=")
    if any(marker in content for marker in tool_markup):
        content = ""
    return AIMessage(
        content=content or fallback,
        usage_metadata=getattr(model_output, "usage_metadata", None),
        response_metadata={
            **(getattr(model_output, "response_metadata", None) or {}),
            "context_usage": context_usage_for_message(
                model_output, model_name=model_name,
            ),
        },
    )


# 配置文件与错误展示


def read_env_file(path: Path) -> dict[str, str]:
    """读取本地配置文件中的简单 KEY=VALUE 项；文件不存在返回空字典。"""

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


def is_model_api_key_configured() -> bool:
    """检查环境变量或 secrets.env 是否填写模型密钥；不验证密钥有效性。"""

    return bool(
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or read_env_file(SECRETS_PATH).get("DEEPSEEK_API_KEY", "").strip()
    )


def redact_error_text(error: Exception) -> str:
    """整理异常文字，遮盖 password/token/api_key/secret 的部分键值写法，截取前 800 字符。

    供流式错误响应使用；只做此处正则能匹配的有限脱敏，不保证清除所有敏感内容。
    """

    text = " ".join(str(error).split())
    text = re.sub(
        r"(?i)\b(password|token|api[_-]?key|secret)\s*[:=]\s*([^\s,;]+)",
        r"\1=[REDACTED]",
        text,
    )
    return text[:800]
