"""Small artifact store used by visualization tools and the preview route."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from config.project_paths import PROJECT_ROOT


ARTIFACT_ROOT = PROJECT_ROOT / "runtime" / "artifacts"
ARTIFACT_ID = re.compile(
    r"^(metric_cards|chart|dashboard|report)_[0-9a-f]{32}$"
)


def save_artifact(kind: str, payload: dict[str, Any]) -> str:
    artifact_id = f"{kind}_{uuid4().hex}"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / f"{artifact_id}.json"
    path.write_text(
        json.dumps(
            {"artifact_id": artifact_id, "kind": kind, **payload},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return artifact_id


def load_artifact(artifact_id: str) -> dict[str, Any]:
    if not ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("Invalid artifact ID.")
    path = ARTIFACT_ROOT / f"{artifact_id}.json"
    if not path.is_file():
        raise FileNotFoundError(artifact_id)
    return json.loads(path.read_text(encoding="utf-8"))


def tool_result(artifact_id: str, *, title: str) -> str:
    artifact = load_artifact(artifact_id)
    result: dict[str, Any] = {
        "status": "success",
        "artifact": {
            "id": artifact_id,
            "kind": artifact["kind"],
            "title": title,
        },
    }
    if artifact["kind"] == "report":
        result["artifact"]["preview_url"] = (
            f"/api/artifacts/{artifact_id}/view"
        )
    return json.dumps(result, ensure_ascii=False)
