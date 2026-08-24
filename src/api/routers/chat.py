"""Chat endpoints: translate HTTP requests into Runtime calls."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from runtime import agent_runtime


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
    _require_model_key()
    agent_runtime.build_agent_config(request)
    return StreamingResponse(
        agent_runtime.stream_agent(request),
        media_type="application/x-ndjson",
        headers={
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
    if not agent_runtime.model_api_key_is_configured():
        raise HTTPException(
            status_code=400,
            detail="尚未配置模型 API Key，请先前往模型设置完成配置。",
        )
