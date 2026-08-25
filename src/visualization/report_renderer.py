"""Render trusted artifact specs into a dashboard page."""

from __future__ import annotations

from html import escape
import json
from typing import Any

from visualization.artifacts import load_artifact


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _metric_cards_html(artifact: dict[str, Any]) -> str:
    cards = []
    for metric in artifact["metrics"]:
        cards.append(
            '<div class="metric-card">'
            f'<span>{escape(str(metric["label"]))}</span>'
            f'<strong>{escape(str(metric["value"]))}</strong>'
            "</div>"
        )
    return '<section class="metric-grid">' + "".join(cards) + "</section>"


def _chart_html(artifact: dict[str, Any], index: int) -> str:
    element_id = f"chart-{index}"
    option = _json_for_script(artifact["option"])
    return (
        '<section class="chart-card">'
        f'<h2>{escape(str(artifact["title"]))}</h2>'
        f'<div class="chart" id="{element_id}"></div>'
        "</section>"
        "<script>"
        f"echarts.init(document.getElementById('{element_id}')).setOption({option});"
        "</script>"
    )


def render_dashboard(dashboard_id: str) -> str:
    dashboard = load_artifact(dashboard_id)
    if dashboard["kind"] != "dashboard":
        raise ValueError("The requested artifact is not a dashboard.")

    sections: list[str] = []
    chart_index = 0
    for child_id in dashboard["artifact_ids"]:
        child = load_artifact(child_id)
        if child["kind"] == "metric_cards":
            sections.append(_metric_cards_html(child))
        elif child["kind"] == "chart":
            chart_index += 1
            sections.append(_chart_html(child, chart_index))

    title = escape(str(dashboard["title"]))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f7fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}}
h1{{margin:0 0 22px;font-size:26px}}
h2{{margin:0 0 10px;font-size:16px}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:18px}}
.metric-card,.chart-card{{border:1px solid #e0e6ef;border-radius:12px;background:#fff;box-shadow:0 8px 24px rgba(27,42,70,.06)}}
.metric-card{{display:grid;gap:10px;padding:18px}}
.metric-card span{{color:#6b778b;font-size:13px}}
.metric-card strong{{font-size:28px}}
.chart-card{{margin-bottom:18px;padding:18px}}
.chart{{height:380px}}
@media(max-width:640px){{main{{padding:14px}}.chart{{height:300px}}}}
</style>
</head>
<body><main><h1>{title}</h1>{"".join(sections)}</main></body>
</html>"""
