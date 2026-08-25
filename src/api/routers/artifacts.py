"""Preview generated report artifacts without exposing server file paths."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from visualization.artifacts import load_artifact


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("/{artifact_id}/view", response_class=HTMLResponse)
def view_artifact(artifact_id: str) -> HTMLResponse:
    try:
        artifact = load_artifact(artifact_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="报告不存在。") from None
    if artifact["kind"] != "report":
        raise HTTPException(status_code=400, detail="该产物不是可预览报告。")
    return HTMLResponse(
        artifact["html"],
        headers={"Cache-Control": "no-store"},
    )
