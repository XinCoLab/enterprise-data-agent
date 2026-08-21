import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
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
    monkeypatch.setattr(server, "_refresh_model_runtime", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DATA_AGENT_MODEL", "deepseek-v4-pro")
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


def test_apply_payload_updates_database_environment_immediately(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(server, "SETTINGS_PATH", tmp_path / "settings.env")
    monkeypatch.setattr(server, "SECRETS_PATH", tmp_path / "secrets.env")
    monkeypatch.setattr(server, "ACTIVE_PROFILE_PATH", tmp_path / ".active_profile")
    for key in (
        "DATA_AGENT_DATABASE_BACKEND",
        "DATA_AGENT_POSTGRES_HOST",
        "DATA_AGENT_POSTGRES_PORT",
        "DATA_AGENT_POSTGRES_USER",
        "DATA_AGENT_POSTGRES_PASSWORD",
        "DATA_AGENT_POSTGRES_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)
    payload = server.ProfilePayload(
        id="test-profile",
        label="Test",
        backend="postgresql",
        host="db.internal",
        port=5433,
        username="readonly",
        database="analytics",
        password="",
        knowledge_root=str(server.PROJECT_ROOT / "knowledge"),
    )

    server._apply_payload(payload, "local-password")

    assert os.environ["DATA_AGENT_DATABASE_BACKEND"] == "postgresql"
    assert os.environ["DATA_AGENT_POSTGRES_HOST"] == "db.internal"
    assert os.environ["DATA_AGENT_POSTGRES_PORT"] == "5433"
    assert os.environ["DATA_AGENT_POSTGRES_USER"] == "readonly"
    assert os.environ["DATA_AGENT_POSTGRES_PASSWORD"] == "local-password"
    assert os.environ["DATA_AGENT_POSTGRES_DATABASE"] == "analytics"


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


class _StreamingGraph:
    def __init__(self):
        self.messages = []
        self.emitted_terminal_answer = False

    def stream(self, payload, **_kwargs):
        human = payload["messages"][0]
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_readonly_sql",
                    "args": {"sql": "SELECT COUNT(*) AS n FROM records"},
                    "id": "sql-stream-1",
                }
            ],
        )
        tool_result = ToolMessage(
            name="execute_readonly_sql",
            tool_call_id="sql-stream-1",
            content='{"columns":["n"],"rows":[{"n":12}],"returned_rows":1,"truncated":false}',
        )
        self.messages = [human]
        yield {
            "type": "tasks",
            "ns": (),
            "data": {"name": "Main Agent LLM", "input": {}},
        }
        self.messages.append(tool_call)
        yield {
            "type": "updates",
            "ns": (),
            "data": {"Main Agent LLM": {"messages": [tool_call]}},
        }
        yield {
            "type": "tasks",
            "ns": (),
            "data": {"name": "Tool Safety", "input": {}},
        }
        yield {
            "type": "updates",
            "ns": (),
            "data": {"Tool Safety": {"messages": [tool_call]}},
        }
        yield {
            "type": "tasks",
            "ns": (),
            "data": {"name": "Tool Execution", "input": {}},
        }
        self.messages.append(tool_result)
        yield {
            "type": "updates",
            "ns": (),
            "data": {"Tool Execution": {"messages": [tool_result]}},
        }
        self.emitted_terminal_answer = True
        final = AIMessage(content="There are 12 records.")
        self.messages.append(final)
        yield {
            "type": "updates",
            "ns": (),
            "data": {"Main Agent LLM": {"messages": [final]}},
        }

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": list(self.messages)})


def test_streaming_run_cancels_only_after_tool_results_complete_protocol(
    tmp_path: Path,
    monkeypatch,
):
    fake_graph = _StreamingGraph()
    monkeypatch.setattr(server, "_agent_graph", lambda: fake_graph)
    monkeypatch.setattr(server, "SETTINGS_PATH", tmp_path / "settings.env")
    request = server.ChatRequest(
        question="Count records",
        thread_id="stream-thread",
        model="deepseek-v4-pro",
    )

    events = server._stream_agent(request)
    started = json.loads(next(events))
    assert started["type"] == "started"
    first_progress = json.loads(next(events))
    assert first_progress["type"] == "progress"

    cancel_response = client.post(f"/api/runs/{started['run_id']}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancel_requested"

    remaining = [json.loads(line) for line in events]
    round_event = next(event for event in remaining if event["type"] == "round")
    assert round_event["round"] == 1
    assert round_event["content"] == ""
    assert round_event["tool_calls"] == [
        {
            "name": "execute_readonly_sql",
            "arguments": {"sql": "SELECT COUNT(*) AS n FROM records"},
        }
    ]
    final = next(event for event in remaining if event["type"] == "final")
    assert final["response"]["status"] == "canceled"
    assert final["response"]["result_preview"]["rows"] == [{"n": 12}]
    assert fake_graph.emitted_terminal_answer is False
    assert started["run_id"] not in server.ACTIVE_RUNS


def test_cancel_endpoint_is_idempotent_for_finished_or_unknown_run():
    response = client.post("/api/runs/not-running/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "not_running"


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
