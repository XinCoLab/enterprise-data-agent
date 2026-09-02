import io
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from api import configuration_app as config_server
from api.app import app
from api.schemas import ChatRequest, ProfilePayload
from runtime import agent_runtime
from security.workspace_access import resolve_current_user


client = TestClient(app)


def test_state_exposes_only_supported_models():
    response = client.get("/api/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["deepseek-v4-pro", "deepseek-v4-flash"]
    assert payload["active"]["backend"] in {"postgresql", "mysql", "duckdb"}


def test_knowledge_graph_exposes_runtime_nodes_and_edges():
    response = client.get("/api/knowledge-graph")
    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"]
    assert payload["edges"]
    assert {"ref", "knowledge_id", "knowledge_type", "title"} <= payload["nodes"][0].keys()
    assert {"source", "relation", "target"} <= payload["edges"][0].keys()


def test_database_schema_endpoint_does_not_expose_secrets(monkeypatch):
    monkeypatch.setattr(
        config_server,
        "_database_schema",
        lambda: {
            "backend": "postgresql",
            "database": "demo",
            "host": "127.0.0.1",
            "port": 5432,
            "username": "reader",
            "table_count": 1,
            "column_count": 2,
            "tables": [{"schema": "public", "name": "events", "kind": "BASE TABLE", "columns": []}],
        },
    )
    response = client.get("/api/database-schema")
    assert response.status_code == 200
    payload = response.json()
    assert payload["table_count"] == 1
    assert "password" not in json.dumps(payload).lower()


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
    monkeypatch.setattr(agent_runtime, "SECRETS_PATH", tmp_path / "secrets.env")
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
    monkeypatch.setattr(config_server, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config_server, "SECRETS_PATH", secrets_path)
    monkeypatch.setattr(config_server, "_refresh_model_runtime", lambda: None)
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


def test_delete_profile_removes_document_and_saved_secret(tmp_path: Path, monkeypatch):
    profiles_root = tmp_path / "profiles"
    secrets_root = tmp_path / "profile_secrets"
    profiles_root.mkdir()
    secrets_root.mkdir()
    profile_path = profiles_root / "old-profile.json"
    secret_path = secrets_root / "old-profile.env"
    profile_path.write_text("{}", encoding="utf-8")
    secret_path.write_text("DATA_AGENT_MYSQL_PASSWORD=secret", encoding="utf-8")
    active_path = tmp_path / ".active_profile"
    active_path.write_text("other-profile", encoding="utf-8")
    monkeypatch.setattr(config_server, "PROFILES_ROOT", profiles_root)
    monkeypatch.setattr(config_server, "PROFILE_SECRETS_ROOT", secrets_root)
    monkeypatch.setattr(config_server, "ACTIVE_PROFILE_PATH", active_path)

    response = client.delete("/api/profiles/old-profile")

    assert response.status_code == 200
    assert not profile_path.exists()
    assert not secret_path.exists()


def test_delete_profile_rejects_active_profile(tmp_path: Path, monkeypatch):
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    profile_path = profiles_root / "active-profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    active_path = tmp_path / ".active_profile"
    active_path.write_text("active-profile", encoding="utf-8")
    monkeypatch.setattr(config_server, "PROFILES_ROOT", profiles_root)
    monkeypatch.setattr(config_server, "PROFILE_SECRETS_ROOT", tmp_path / "profile_secrets")
    monkeypatch.setattr(config_server, "ACTIVE_PROFILE_PATH", active_path)

    response = client.delete("/api/profiles/active-profile")

    assert response.status_code == 409
    assert profile_path.exists()


def test_apply_payload_updates_database_environment_immediately(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(config_server, "SETTINGS_PATH", tmp_path / "settings.env")
    monkeypatch.setattr(config_server, "SECRETS_PATH", tmp_path / "secrets.env")
    monkeypatch.setattr(
        config_server,
        "ACTIVE_PROFILE_PATH",
        tmp_path / ".active_profile",
    )
    for key in (
        "DATA_AGENT_DATABASE_BACKEND",
        "DATA_AGENT_POSTGRES_HOST",
        "DATA_AGENT_POSTGRES_PORT",
        "DATA_AGENT_POSTGRES_USER",
        "DATA_AGENT_POSTGRES_PASSWORD",
        "DATA_AGENT_POSTGRES_DATABASE",
    ):
        monkeypatch.delenv(key, raising=False)
    payload = ProfilePayload(
        id="test-profile",
        label="Test",
        backend="postgresql",
        host="db.internal",
        port=5433,
        username="readonly",
        database="analytics",
        password="",
        knowledge_root=str(config_server.PROJECT_ROOT / "knowledge"),
    )

    config_server._apply_payload(payload, "local-password")

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

    result = agent_runtime._turn_result(messages)

    assert result["answer"] == "There are 12 records."
    assert result["sql_queries"][0]["sql"].startswith("SELECT COUNT")
    assert result["result_preview"]["rows"] == [{"n": 12}]


def test_round_graph_returns_to_runtime_after_one_complete_tool_cycle():
    from graph.round_graph import round_graph

    edges = {
        (edge.source, edge.target)
        for edge in round_graph.get_graph().edges
    }

    assert ("Tool Safety", "Tool Execution") in edges
    assert ("Tool Execution", "__end__") in edges
    assert ("Tool Execution", "Main Agent LLM") not in edges


class _StreamingGraph:
    def __init__(self):
        self.messages = []
        self.round_number = 0
        self.emitted_terminal_answer = False

    def stream(self, payload, **_kwargs):
        if payload.get("messages"):
            self.messages.extend(payload["messages"])
        self.round_number += 1

        if self.round_number > 1:
            self.emitted_terminal_answer = True
            final = AIMessage(content="There are 12 records.")
            self.messages.append(final)
            yield {
                "type": "updates",
                "ns": (),
                "data": {"Main Agent LLM": {"messages": [final]}},
            }
            return

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

    def update_state(self, _config, values):
        self.messages.extend(values.get("messages", []))

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": list(self.messages)})


def test_streaming_run_cancels_only_after_tool_results_complete_protocol(
    tmp_path: Path,
    monkeypatch,
):
    fake_graph = _StreamingGraph()
    monkeypatch.setattr(agent_runtime, "_agent_graph", lambda: fake_graph)
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", tmp_path / "settings.env")
    request = ChatRequest(
        question="Count records",
        thread_id="stream-thread",
        model="deepseek-v4-pro",
    )

    events = agent_runtime.stream_agent(request, resolve_current_user(None))
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
    assert started["run_id"] not in agent_runtime.ACTIVE_RUNS


def test_streaming_runtime_counts_one_complete_tool_cycle_as_one_recursion(
    tmp_path: Path,
    monkeypatch,
):
    fake_graph = _StreamingGraph()
    settings_path = tmp_path / "settings.env"
    settings_path.write_text(
        "DATA_AGENT_MAX_RECURSIONS=2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_runtime, "_agent_graph", lambda: fake_graph)
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", settings_path)
    request = ChatRequest(
        question="Count records",
        thread_id="two-round-thread",
        model="deepseek-v4-pro",
    )

    events = [
        json.loads(line)
        for line in agent_runtime.stream_agent(request, resolve_current_user(None))
    ]

    rounds = [event["round"] for event in events if event["type"] == "round"]
    final = next(event for event in events if event["type"] == "final")
    assert rounds == [1, 2]
    assert final["response"]["status"] == "success"
    assert final["response"]["answer"] == "There are 12 records."
    assert fake_graph.emitted_terminal_answer is True


def test_max_recursions_generates_tool_free_fallback_summary(
    tmp_path: Path,
    monkeypatch,
):
    fake_graph = _StreamingGraph()
    settings_path = tmp_path / "settings.env"
    settings_path.write_text(
        "DATA_AGENT_MAX_RECURSIONS=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_runtime, "_agent_graph", lambda: fake_graph)
    monkeypatch.setattr(
        agent_runtime,
        "_generate_recursion_limit_summary",
        lambda _messages, **_kwargs: AIMessage(content="Pause report"),
    )
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", settings_path)
    request = ChatRequest(
        question="Count records",
        thread_id="budget-thread",
        model="deepseek-v4-pro",
    )

    events = [
        json.loads(line)
        for line in agent_runtime.stream_agent(request, resolve_current_user(None))
    ]

    final = next(event for event in events if event["type"] == "final")
    response = final["response"]
    assert response["status"] == "paused"
    assert response["answer"] == "Pause report"
    assert isinstance(fake_graph.messages[-2], ToolMessage)
    assert isinstance(fake_graph.messages[-1], AIMessage)
    assert fake_graph.emitted_terminal_answer is False


def test_recursion_limit_summary_never_persists_new_tool_calls(monkeypatch):
    from prompts import recursion_limit_summary

    monkeypatch.setattr(
        recursion_limit_summary,
        "generate_recursion_limit_summary",
        lambda _messages, **_kwargs: AIMessage(
            content='<|DSML|tool_calls><|DSML|invoke name="read_knowledge">',
            tool_calls=[
                {
                    "id": "unexpected-summary-tool",
                    "name": "read_knowledge",
                    "args": {"knowledge_ids": ["table.example"]},
                }
            ],
        ),
    )

    model_output = agent_runtime._generate_recursion_limit_summary(
        [HumanMessage(content="复杂分析任务")],
        model_name="deepseek-v4-pro",
    )

    assert model_output.content
    assert "DSML" not in model_output.content
    assert model_output.tool_calls == []


def test_agent_config_separates_product_recursions_from_langgraph_guard(
    tmp_path: Path,
    monkeypatch,
):
    settings_path = tmp_path / "settings.env"
    settings_path.write_text(
        "DATA_AGENT_MAX_RECURSIONS=10\n"
        "LANGGRAPH_DEFAULT_RECURSION_LIMIT=6\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", settings_path)

    _thread_id, config = agent_runtime.build_agent_config(
        ChatRequest(question="test", thread_id="budget-config"),
        resolve_current_user(None),
    )

    assert config["configurable"]["max_recursions"] == 10
    assert config["recursion_limit"] == 6


def test_cancel_endpoint_is_idempotent_for_finished_or_unknown_run():
    response = client.post("/api/runs/not-running/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "not_running"


def test_graph_run_lock_supports_streaming_worker_handoffs():
    """The stream lock must not require release by its acquiring thread."""

    lock = agent_runtime.GRAPH_RUN_LOCK
    assert type(lock) is type(threading.Lock())

    errors: list[Exception] = []

    def acquire_on_first_worker():
        lock.acquire()

    def release_on_second_worker():
        try:
            lock.release()
        except Exception as error:  # pragma: no cover - regression evidence
            errors.append(error)

    first_worker = threading.Thread(target=acquire_on_first_worker)
    first_worker.start()
    first_worker.join()
    second_worker = threading.Thread(target=release_on_second_worker)
    second_worker.start()
    second_worker.join()

    assert errors == []


def _archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_knowledge_import_accepts_valid_cards(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config_server, "KNOWLEDGE_IMPORT_ROOT", tmp_path)
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
    monkeypatch.setattr(config_server, "KNOWLEDGE_IMPORT_ROOT", tmp_path)
    response = client.post(
        "/api/import-knowledge",
        content=_archive({"../outside.yaml": "not allowed"}),
        headers={"content-type": "application/zip"},
    )

    assert response.status_code == 400
    assert "不安全路径" in response.json()["detail"]
