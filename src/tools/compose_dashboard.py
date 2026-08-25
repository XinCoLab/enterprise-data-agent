"""Compose metric-card and chart artifacts into one dashboard."""

from typing import Annotated

from langchain_core.tools import tool

from visualization.artifacts import load_artifact, save_artifact, tool_result


@tool("compose_dashboard")
def compose_dashboard(
    title: Annotated[str, "Dashboard title."],
    artifact_ids: Annotated[
        list[str],
        "IDs returned by create_metric_cards or create_chart, in display order.",
    ],
) -> str:
    """Combine existing metric cards and charts into one dashboard."""

    if not artifact_ids or len(artifact_ids) > 12:
        raise ValueError("artifact_ids must contain between 1 and 12 items.")
    for artifact_id in artifact_ids:
        artifact = load_artifact(artifact_id)
        if artifact["kind"] not in {"metric_cards", "chart"}:
            raise ValueError("Dashboard sections must be cards or charts.")
    dashboard_id = save_artifact(
        "dashboard",
        {"title": title[:120], "artifact_ids": artifact_ids},
    )
    return tool_result(dashboard_id, title=title[:120])
