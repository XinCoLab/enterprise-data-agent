"""Create a reusable chart spec from verified SQL results."""

from collections import defaultdict
from typing import Annotated, Any, Literal

from langchain_core.tools import tool

from visualization.artifacts import save_artifact, tool_result


ChartType = Literal["bar", "line", "area", "pie", "donut"]


def _cartesian_option(
    *,
    chart_type: ChartType,
    rows: list[dict[str, Any]],
    category_field: str,
    value_field: str,
    series_field: str,
) -> dict[str, Any]:
    categories = list(dict.fromkeys(str(row[category_field]) for row in rows))
    if series_field:
        groups: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in rows:
            groups[str(row[series_field])][str(row[category_field])] = row[
                value_field
            ]
        series = [
            {
                "name": name,
                "type": "line" if chart_type in {"line", "area"} else "bar",
                "areaStyle": {} if chart_type == "area" else None,
                "data": [values.get(category) for category in categories],
            }
            for name, values in groups.items()
        ]
    else:
        series = [
            {
                "type": "line" if chart_type in {"line", "area"} else "bar",
                "areaStyle": {} if chart_type == "area" else None,
                "data": [row[value_field] for row in rows],
            }
        ]
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0},
        "grid": {"left": 44, "right": 24, "top": 46, "bottom": 52},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"interval": 0, "rotate": 20},
        },
        "yAxis": {"type": "value"},
        "series": series,
    }


@tool("create_chart")
def create_chart(
    title: Annotated[str, "Short chart title."],
    chart_type: Annotated[
        ChartType,
        "One of bar, line, area, pie, or donut.",
    ],
    rows: Annotated[
        list[dict[str, Any]],
        "Rows copied from a successful SQL result.",
    ],
    category_field: Annotated[str, "Column used for category labels."],
    value_field: Annotated[str, "Numeric column used for values."],
    series_field: Annotated[
        str,
        "Optional column used to split multiple series; use an empty string when unnecessary.",
    ] = "",
) -> str:
    """Create one chart from actual read-only SQL result rows."""

    if not rows or len(rows) > 500:
        raise ValueError("rows must contain between 1 and 500 items.")
    required = {category_field, value_field}
    if series_field:
        required.add(series_field)
    if any(not required.issubset(row) for row in rows):
        raise ValueError("Chart fields must exist in every row.")

    if chart_type in {"pie", "donut"}:
        option: dict[str, Any] = {
            "tooltip": {"trigger": "item"},
            "legend": {"bottom": 0},
            "series": [
                {
                    "type": "pie",
                    "radius": ["45%", "70%"] if chart_type == "donut" else "70%",
                    "data": [
                        {
                            "name": str(row[category_field]),
                            "value": row[value_field],
                        }
                        for row in rows
                    ],
                    "label": {"formatter": "{b}: {c}"},
                }
            ],
        }
    else:
        option = _cartesian_option(
            chart_type=chart_type,
            rows=rows,
            category_field=category_field,
            value_field=value_field,
            series_field=series_field,
        )

    artifact_id = save_artifact(
        "chart",
        {"title": title[:120], "chart_type": chart_type, "option": option},
    )
    return tool_result(artifact_id, title=title[:120])
