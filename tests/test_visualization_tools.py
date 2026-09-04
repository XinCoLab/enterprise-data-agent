import json

from visualization import artifacts
from tools.compose_dashboard import compose_dashboard
from tools.create_chart import create_chart
from tools.create_metric_cards import create_metric_cards
from tools.export_report import export_report


def _artifact_id(result: str) -> str:
    return json.loads(result)["artifact"]["id"]


def test_visualization_tools_return_a_viewable_report(tmp_path, monkeypatch, client):
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path)
    cards_id = _artifact_id(
        create_metric_cards.invoke(
            {
                "title": "概览",
                "metrics": [{"label": "承运商", "value": 798}],
            }
        )
    )
    chart_id = _artifact_id(
        create_chart.invoke(
            {
                "title": "认证覆盖率",
                "chart_type": "bar",
                "rows": [
                    {"认证": "GDP", "覆盖率": 23.6},
                    {"认证": "Both", "覆盖率": 22.29},
                ],
                "category_field": "认证",
                "value_field": "覆盖率",
                "series_field": "",
            }
        )
    )
    dashboard_id = _artifact_id(
        compose_dashboard.invoke(
            {
                "title": "冷链承运能力概览",
                "artifact_ids": [cards_id, chart_id],
            }
        )
    )
    report = json.loads(
        export_report.invoke({"dashboard_id": dashboard_id})
    )["artifact"]

    response = client.get(report["preview_url"])
    assert response.status_code == 200
    assert "echarts.init" in response.text
