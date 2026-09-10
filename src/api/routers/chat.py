"""聊天接口：检查是否允许运行，再使用后端已创建的图处理问题。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatRequest
from agent_runtime import agent_runtime
from security.workspace_access import (
    CurrentUser,
    read_current_user,
    require_permission,
    require_workspace_resources,
)


# 本文件中的路径都会自动带上 /api 前缀。
router = APIRouter(prefix="/api", tags=["agent"])


@router.get("/runs")
def recent_runs(
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    """返回当前工作区最近的运行摘要；读取的是进程内存，不是聊天历史数据库。"""

    require_permission(current_user, "conversation:read")
    return {"runs": agent_runtime.list_recent_runs(current_user)}


@router.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,  # FastAPI 解析请求体后得到的聊天数据：问题、thread_id、模型
    http_request: Request,  # 同一次 HTTP 请求的框架对象；这里用它访问已创建的 Graph
    # FastAPI 调用 read_current_user()：根据请求头选择模拟账号，身份和权限从服务端记录读取。
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    """接收一次聊天请求，检查通过后返回逐行输出进度和答案的响应流。

    三个参数由 FastAPI 准备并传入，对应同一次 HTTP 请求。
    current_user 由服务端构造，不直接使用客户端提交的用户对象。
    两项检查均在响应流开始前完成；校验失败直接返回 HTTP 错误，不进入 Agent 执行。
    本函数取用已有 Graph；模型/工具执行和聊天记录保存交给 Agent Runtime。
    """

    # 1. 检查是否允许运行：用户权限、工作区是否已配置、旧会话是否属于当前工作区。
    # 检查含同步数据库操作，交给框架管理的工作线程，避免阻塞事件循环。
    # run_in_threadpool 使用框架的线程执行机制，不是每次创建一个线程池。
    # await 等这项检查通过后才进入第 2 步；两项检查按顺序执行。
    await run_in_threadpool(check_run_allowed, chat_request, current_user)

    # 2. 检查模型密钥是否已填写，可能同步读配置文件，所以也使用工作线程。
    # 未填写就直接返回 HTTP 400，避免等响应流开始后才发现缺少密钥。
    await run_in_threadpool(check_model_key)

    # 3. 从后端 app 取出 lifespan 在启动时创建的 Graph，这里不重新创建。
    # http_request.app 指向处理本次请求的后端应用；Graph 不来自浏览器或请求体。
    single_round_graph = http_request.app.state.single_round_graph

    # 4. 读取本地设置，为这次聊天准备运行配置，其中包含固定身份和资源的 AgentRunContext。
    # 在响应流开始前只构造一次配置，也只生成一个 request_id。
    thread_id, run_config = await run_in_threadpool(
        agent_runtime.build_agent_config,
        chat_request,
        current_user,
    )

    # 5. 创建异步生成器；此时还没开始执行 Agent，取值时才会执行到 yield。
    reply_stream = agent_runtime.stream_chat_response(
        chat_request,
        current_user,
        single_round_graph=single_round_graph,
        thread_id=thread_id,
        run_config=run_config,
    )

    # 创建响应对象，让它负责读取生成器产出的 NDJSON 文本并发送给浏览器。
    response = StreamingResponse(
        reply_stream,
        media_type="application/x-ndjson",
        headers={
            # 禁止浏览器或反向代理缓存、攒批，尽量让每条进度及时到达前端。
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )

    # 返回这个响应对象；框架随后开始读取并发送进度和回答，Agent 才随之执行。
    return response


@router.post("/runs/{request_id}/cancel")
def cancel_run(
    request_id: str,
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    """请求停止指定运行；只设置取消标记，由 Runtime 在完整消息边界停止。"""

    require_permission(current_user, "chat:run")
    try:
        return agent_runtime.request_run_cancellation(request_id, current_user)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def check_model_key() -> None:
    """检查模型密钥是否已填写；不验证密钥有效性，也不调用模型。

    已填写返回 None，未填写抛 HTTP 400，让聊天在响应流开始前被拒绝。
    """

    if not agent_runtime.is_model_api_key_configured():
        raise HTTPException(
            status_code=400,
            detail="尚未配置模型 API Key，请先前往模型设置完成配置。",
        )


def check_run_allowed(
    request: ChatRequest,
    current_user: CurrentUser,
) -> None:
    """检查当前用户是否允许开始这次聊天，由 chat_stream 通过工作线程调用。

    request 是已解析的聊天数据，current_user 是服务端识别的用户。
    依次检查聊天权限、工作区的 resources_ready 标记、已有会话的所属工作区。
    检查通过返回 None，不返回 True/False；校验不通过抛 HTTPException，阻止继续执行。
    本函数可能查询聊天历史数据库，但不写入消息、不调用模型，也不执行工具。
    """

    # 无聊天权限或工作区尚未配置时，在这里报错，不再检查后续步骤。
    require_permission(current_user, "chat:run")
    require_workspace_resources(current_user)

    with agent_runtime.RESOURCE_CONFIG_LOCK:
        binding = agent_runtime.read_current_conversation_binding(current_user)
        agent_runtime.validate_request_binding(request, binding)
        thread_id = request.thread_id.strip()
        if thread_id:
            agent_runtime.validate_conversation_binding(
                thread_id, current_user, binding,
            )
