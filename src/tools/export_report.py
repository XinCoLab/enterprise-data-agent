"""Export a dashboard as a browser-viewable report artifact."""

from typing import Annotated

from langchain_core.tools import tool

from visualization.artifacts import load_artifact, save_artifact, tool_result
from visualization.report_renderer import render_dashboard


@tool("export_report")
def export_report(
    dashboard_id: Annotated[
        str,
        "Dashboard ID returned by compose_dashboard.",
    ],
) -> str:
    """Export one dashboard and return a preview URL, never raw HTML."""

    dashboard = load_artifact(dashboard_id)
    if dashboard["kind"] != "dashboard":
        raise ValueError("dashboard_id must refer to a dashboard artifact.")
    html = render_dashboard(dashboard_id)
    report_id = save_artifact(
        "report",
        {
            "title": dashboard["title"],
            "dashboard_id": dashboard_id,
            "html": html,
        },
    )
    return tool_result(report_id, title=dashboard["title"])
