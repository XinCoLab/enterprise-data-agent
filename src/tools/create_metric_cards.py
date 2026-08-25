"""Create metric-card specs from verified SQL results."""

from typing import Annotated, Any

from langchain_core.tools import tool

from visualization.artifacts import save_artifact, tool_result


@tool("create_metric_cards")
def create_metric_cards(
    title: Annotated[str, "Short section title."],
    metrics: Annotated[
        list[dict[str, Any]],
        "Metric objects with label and value copied from successful SQL results.",
    ],
) -> str:
    """Create dashboard metric cards from actual query results."""

    if not metrics or len(metrics) > 12:
        raise ValueError("metrics must contain between 1 and 12 items.")
    normalized = []
    for metric in metrics:
        if "label" not in metric or "value" not in metric:
            raise ValueError("Every metric needs label and value.")
        normalized.append(
            {
                "label": str(metric["label"])[:100],
                "value": str(metric["value"])[:100],
            }
        )
    artifact_id = save_artifact(
        "metric_cards",
        {"title": title[:120], "metrics": normalized},
    )
    return tool_result(artifact_id, title=title[:120])
