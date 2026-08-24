"""Chat 接口：接收 HTTP 请求，然后把实际执行交给 Agent Runtime。"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from runtime import agent_runtime


# 本文件中的路径都会自动带上 /api 前缀。
router = APIRouter(prefix="/api", tags=["agent"])


@router.get("/runs")
def recent_runs():
    return {"runs": agent_runtime.list_recent_runs()}


@router.post("/chat")
def chat(request: ChatRequest):
    _require_model_key()
    try:
        return agent_runtime.run_agent(request)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"分析执行失败：{agent_runtime.safe_error_text(error)}",
        ) from error


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    # 在创建流式响应前先检查关键配置，让配置错误直接作为 HTTP 错误返回。
    _require_model_key()

    # 提前校验 thread_id 和 Runtime 设置。真正执行时 stream_agent() 会再次
    # 构造并使用这份配置；这里提前调用是为了避免错误发生在流已经开始之后。
    agent_runtime.build_agent_config(request)

    # StreamingResponse 会持续读取 stream_agent() 产生的 NDJSON 事件，
    # 将“模型思考、工具执行、最终回答”等进度逐条发送给浏览器。
    return StreamingResponse(
        agent_runtime.stream_agent(request),
        media_type="application/x-ndjson",
        headers={
            # 禁止浏览器或反向代理缓存、攒批，尽量让每条进度及时到达前端。
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    try:
        return agent_runtime.request_cancel(run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _require_model_key() -> None:
    """没有模型 API Key 时，在进入 Agent Runtime 前直接拒绝请求。"""

    if not agent_runtime.model_api_key_is_configured():
        raise HTTPException(
            status_code=400,
            detail="尚未配置模型 API Key，请先前往模型设置完成配置。",
        )
