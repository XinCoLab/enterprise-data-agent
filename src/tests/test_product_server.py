import io
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config_ui_server as server


client = TestClient(server.app)


def test_state_exposes_only_supported_models():
    response = client.get("/api/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["deepseek-v4-pro", "deepseek-v4-flash"]
    assert payload["active"]["backend"] in {"postgresql", "mysql", "duckdb"}


def test_chat_rejects_unknown_model_before_agent_execution():
    response = client.post(
        "/api/chat",
        json={"question": "count records", "model": "unknown-model"},
    )
    assert response.status_code == 422


def test_chat_reports_missing_model_key_without_exposing_internal_name(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(server, "SECRETS_PATH", tmp_path / "secrets.env")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = client.post(
        "/api/chat",
        json={"question": "count records", "model": "deepseek-v4-pro"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "模型设置" in detail
    assert "DEEPSEEK_API_KEY" not in detail


def test_model_settings_store_key_without_returning_it(tmp_path: Path, monkeypatch):
    settings_path = tmp_path / "settings.env"
    secrets_path = tmp_path / "secrets.env"
    monkeypatch.setattr(server, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(server, "SECRETS_PATH", secrets_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    secret = "test-secret-not-for-output"

    response = client.post(
        "/api/model-settings",
        json={"model": "deepseek-v4-flash", "api_key": secret},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "deepseek-v4-flash"
    assert secret not in response.text
    assert "DATA_AGENT_MODEL=deepseek-v4-flash" in settings_path.read_text(
        encoding="utf-8"
    )
    assert f"DEEPSEEK_API_KEY={secret}" in secrets_path.read_text(encoding="utf-8")


def test_turn_result_links_sql_to_its_real_tool_result():
    messages = [
        HumanMessage(content="Count rows"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_readonly_sql",
                    "args": {"sql": "SELECT COUNT(*) AS n FROM records"},
                    "id": "sql-1",
                }
            ],
        ),
        ToolMessage(
            name="execute_readonly_sql",
            tool_call_id="sql-1",
            content='{"columns":["n"],"rows":[{"n":12}],"returned_rows":1,"truncated":false}',
        ),
        AIMessage(content="There are 12 records."),
    ]

    result = server._turn_result(messages)

    assert result["answer"] == "There are 12 records."
    assert result["sql_queries"][0]["sql"].startswith("SELECT COUNT")
    assert result["result_preview"]["rows"] == [{"n": 12}]


def _archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_knowledge_import_accepts_valid_cards(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "KNOWLEDGE_IMPORT_ROOT", tmp_path)
    card = """
knowledge_id: table.demo.records
knowledge_type: table
title: Records
summary: One row per record.
payload:
  physical_name: records
"""
    response = client.post(
        "/api/import-knowledge",
        content=_archive({"tables/records.yaml": card}),
        headers={"content-type": "application/zip", "x-knowledge-name": "demo"},
    )

    assert response.status_code == 200
    assert response.json()["details"]["card_count"] == 1


def test_knowledge_import_rejects_parent_traversal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "KNOWLEDGE_IMPORT_ROOT", tmp_path)
    response = client.post(
        "/api/import-knowledge",
        content=_archive({"../outside.yaml": "not allowed"}),
        headers={"content-type": "application/zip"},
    )

    assert response.status_code == 400
    assert "不安全路径" in response.json()["detail"]
