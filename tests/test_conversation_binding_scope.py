"""Resource-scoped history and preflight checks; only isolated test files/SQLite."""
import asyncio
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent_runtime import agent_runtime
from api.schemas import ChatRequest
from memory import conversation_history_database as history
from memory import workspace_database
from memory.conversation_binding import RESOURCE_ENV_KEYS, build_conversation_binding
from security.workspace_access import resolve_current_user


@pytest.fixture
def source_settings(tmp_path, monkeypatch):
    for key in RESOURCE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings_path = tmp_path / "settings.env"
    active_path = tmp_path / ".active_profile"
    active_path.write_text("test-profile", encoding="utf-8")
    monkeypatch.setattr(agent_runtime, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(agent_runtime, "ACTIVE_PROFILE_PATH", active_path)

    def activate(database="cold", knowledge="cold-knowledge"):
        settings_path.write_text(
            "DATA_AGENT_DATABASE_BACKEND=postgresql\n"
            "DATA_AGENT_POSTGRES_HOST=127.0.0.1\n"
            "DATA_AGENT_POSTGRES_PORT=5432\n"
            f"DATA_AGENT_POSTGRES_DATABASE={database}\n"
            f"DATA_AGENT_KNOWLEDGE_ROOT={tmp_path / knowledge}\n",
            encoding="utf-8",
        )
        return agent_runtime.read_current_conversation_binding(resolve_current_user("admin-a"))

    return activate


def save_thread(thread_id, binding, *, workspace_id="workspace-a", user_id="user-admin-a"):
    history.save_user_message(
        thread_id, "测试问题", workspace_id, user_id, binding=binding,
    )


def test_binding_migration_keeps_old_messages_unassigned(tmp_path, monkeypatch):
    database_path = tmp_path / "old-history.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.executescript("""
            CREATE TABLE conversations (
                thread_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                custom_title INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, details_json TEXT,
                created_at TEXT NOT NULL
            );
            INSERT INTO conversations VALUES ('old', '原始标题', 0, '2026-01-01', '2026-01-02');
            INSERT INTO messages(thread_id, role, content, created_at)
            VALUES ('old', 'user', '原始问题', '2026-01-01');
        """)
    monkeypatch.setattr(history, "CHAT_HISTORY_DATABASE_PATH", database_path)
    monkeypatch.setattr(workspace_database, "CHAT_HISTORY_DATABASE_PATH", database_path)
    history.create_chat_history_tables()
    history.create_chat_history_tables()
    original = history.read_conversation_info("old")
    assert original["binding"] is None
    assert original["title"] == "原始标题"
    assert original["messages"][0]["content"] == "原始问题"
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT data_source_id, knowledge_base_id, binding_json FROM conversations"
        ).fetchone() == (None, None, None)
    assert [row["thread_id"] for row in history.list_conversation_rows(legacy=True)] == ["old"]


def test_storage_filters_source_and_keeps_original_binding(source_settings):
    cold = source_settings()
    idol = source_settings("idol", "idol-knowledge")
    save_thread("cold", cold)
    save_thread("idol", idol)
    save_thread("legacy", None)
    assert {row["thread_id"] for row in history.list_conversation_rows()} == {"cold", "idol", "legacy"}
    assert [row["thread_id"] for row in history.list_conversation_rows(data_source_id=cold["data_source_id"])] == ["cold"]
    assert [row["thread_id"] for row in history.list_conversation_rows(legacy=True)] == ["legacy"]
    history.save_user_message("cold", "继续", binding={**cold, "data_source_name": "重命名"})
    assert history.read_conversation_info("cold")["binding"] == cold
    assert len(history.read_conversation_info("cold")["messages"]) == 2


@pytest.mark.parametrize("existing_kind,new_kind", [
    ("legacy", "cold"), ("cold", "legacy"), ("cold", "idol"), ("cold", "other-kb"),
])
def test_storage_does_not_claim_or_rebind_history(source_settings, existing_kind, new_kind):
    bindings = {
        "legacy": None,
        "cold": source_settings(),
        "idol": source_settings("idol", "idol-knowledge"),
        "other-kb": source_settings("cold", "another-knowledge"),
    }
    save_thread("fixed", bindings[existing_kind])
    before = history.read_conversation_info("fixed")
    with pytest.raises(history.ConversationBindingError):
        history.save_user_message("fixed", "不应写入", binding=bindings[new_kind])
    assert history.read_conversation_info("fixed") == before


def test_api_defaults_to_active_source_and_exposes_workspace_groups(client, source_settings):
    cold = source_settings()
    save_thread("cold", cold)
    idol = source_settings("idol", "idol-knowledge")
    save_thread("idol", idol)
    save_thread("legacy", None)
    save_thread("foreign", idol, workspace_id="workspace-b", user_id="user-analyst-b")
    payload = client.get("/api/conversations").json()
    assert [row["thread_id"] for row in payload["conversations"]] == ["idol"]
    assert payload["conversations"][0]["binding"] == idol
    assert payload["conversations"][0]["can_continue"] is True
    assert payload["legacy_count"] == 1
    assert payload["active_binding"] == idol
    assert {row["data_source_id"] for row in payload["data_sources"]} == {cold["data_source_id"], idol["data_source_id"]}
    selected = client.get("/api/conversations", params={"data_source_id": cold["data_source_id"]}).json()
    assert [row["thread_id"] for row in selected["conversations"]] == ["cold"]
    assert selected["conversations"][0]["can_continue"] is False
    legacy = client.get("/api/conversations?legacy=true").json()
    assert [row["thread_id"] for row in legacy["conversations"]] == ["legacy"]
    assert legacy["conversations"][0]["can_continue"] is False
    assert client.get("/api/conversations", params={"legacy": "true", "data_source_id": cold["data_source_id"]}).status_code == 400
    foreign = client.get("/api/conversations?legacy=true", headers={"X-Dev-User": "analyst-b"}).json()
    assert foreign["conversations"] == []
    assert foreign["legacy_count"] == 0
    assert len(foreign["data_sources"]) == 1
    assert client.get("/api/conversations/cold", headers={"X-Dev-User": "analyst-b"}).status_code == 404


def test_detail_and_rename_keep_thread_knowledge_binding(client, source_settings):
    original = source_settings()
    save_thread("thread", original)
    save_thread("legacy", None)
    assert client.get("/api/conversations/thread").json()["can_continue"] is True
    source_settings("cold", "different-knowledge")
    detail = client.get("/api/conversations/thread").json()
    assert detail["binding"] == original
    assert detail["can_continue"] is False
    renamed = client.patch("/api/conversations/thread", json={"title": "新的标题"}).json()["conversation"]
    assert renamed["binding"] == original
    assert renamed["can_continue"] is False
    source_settings()
    assert client.get("/api/conversations/thread").json()["can_continue"] is True
    assert client.get("/api/conversations/legacy").json()["can_continue"] is False


@pytest.mark.parametrize("field", ["data_source_id", "knowledge_base_id"])
def test_stale_tab_ids_are_rejected_before_model_check(client, monkeypatch, source_settings, field):
    source_settings()
    def unexpected_model_check():
        raise AssertionError("stale request reached model-key checking")
    monkeypatch.setattr(agent_runtime, "is_model_api_key_configured", unexpected_model_check)
    response = client.post("/api/chat/stream", json={
        "question": "查询", "thread_id": "new-thread", field: "stale-id",
    })
    assert response.status_code == 409
    assert history.read_conversation_info("new-thread") is None


@pytest.mark.parametrize("binding_kind", ["legacy", "different-source", "different-kb"])
def test_old_thread_cannot_resume_under_wrong_binding(client, monkeypatch, source_settings, binding_kind):
    current = source_settings()
    old = {
        "legacy": None,
        "different-source": build_conversation_binding({
            "backend": "postgresql", "database": "idol", "knowledge_root": current["knowledge_root"],
        }),
        "different-kb": build_conversation_binding({
            "backend": "postgresql", "database": "cold", "knowledge_root": current["knowledge_root"] + "-other",
        }),
    }[binding_kind]
    save_thread("old-thread", old)
    def unexpected_model_check():
        raise AssertionError("blocked thread reached model-key checking")
    monkeypatch.setattr(agent_runtime, "is_model_api_key_configured", unexpected_model_check)
    response = client.post("/api/chat/stream", json={"question": "继续", "thread_id": "old-thread"})
    assert response.status_code == 409
    assert len(history.read_conversation_info("old-thread")["messages"]) == 1


def test_build_and_register_detect_same_profile_resource_change(source_settings):
    original = source_settings()
    current_user = resolve_current_user("admin-a")
    request = ChatRequest(question="测试", thread_id="snapshot", **{
        key: original[key] for key in ("data_source_id", "knowledge_base_id")
    })
    thread_id, config = agent_runtime.build_agent_config(request, current_user)
    context = config["configurable"]["agent_run_context"]
    assert context.binding == original
    source_settings("cold", "changed-knowledge")
    assert context.binding == original
    control = agent_runtime.ActiveRun(context.request_id, thread_id, current_user.workspace_id, current_user.user_id)
    with pytest.raises(HTTPException) as error:
        agent_runtime.register_run(control, context, current_user)
    assert error.value.status_code == 409
    assert context.request_id not in agent_runtime.ACTIVE_RUNS
    with pytest.raises(HTTPException) as error:
        agent_runtime.build_agent_config(request, current_user)
    assert error.value.status_code == 409


def test_runtime_rechecks_thread_after_acquiring_conversation_lock(source_settings):
    binding = source_settings()
    current_user = resolve_current_user("admin-a")
    request = ChatRequest(question="新问题", thread_id="race-thread")
    thread_id, config = agent_runtime.build_agent_config(request, current_user)
    other = source_settings("idol", "idol-knowledge")
    source_settings()

    class ForbiddenGraph:
        async def aget_state(self, _config):
            raise AssertionError("wrong binding reached checkpoint/pending tools")

    async def execute():
        checkpoint_id = config["configurable"]["thread_id"]
        lock = agent_runtime.get_or_create_conversation_lock(checkpoint_id)
        async with lock:
            stream = agent_runtime.execute_agent_request(
                request, current_user, single_round_graph=ForbiddenGraph(),
                thread_id=thread_id, run_config=config,
            )
            started = await anext(stream)
            assert started["type"] == "started"
            waiting = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            save_thread(thread_id, other)
        try:
            with pytest.raises(HTTPException) as error:
                await waiting
            assert error.value.status_code == 409
        finally:
            await stream.aclose()
        assert config["configurable"]["agent_run_context"].request_id not in agent_runtime.ACTIVE_RUNS

    asyncio.run(execute())
    assert history.read_conversation_info(thread_id)["binding"] == other
    assert len(history.read_conversation_info(thread_id)["messages"]) == 1
    assert binding != other


def test_unconfigured_workspace_has_no_current_binding(source_settings):
    source_settings()
    user = replace(resolve_current_user("admin-a"), resources_ready=False)
    assert agent_runtime.read_current_conversation_binding(user) is None


@pytest.mark.parametrize("override_key", [
    "DATA_AGENT_POSTGRES_DATABASE", "DATA_AGENT_KNOWLEDGE_ROOT",
])
def test_removing_startup_override_blocks_old_thread(
    client, monkeypatch, source_settings, tmp_path, override_key,
):
    file_binding = source_settings()
    override_value = "idol" if override_key.endswith("DATABASE") else str(tmp_path / "env-knowledge")
    monkeypatch.setenv(override_key, override_value)
    overridden = agent_runtime.read_current_conversation_binding(resolve_current_user("admin-a"))
    assert overridden != file_binding
    save_thread("env-thread", overridden)

    # Simulate restarting without the launch override: settings.env wins again.
    monkeypatch.delenv(override_key)
    assert agent_runtime.read_current_conversation_binding(resolve_current_user("admin-a")) == file_binding
    def unexpected_model_check():
        raise AssertionError("mismatched restart reached model-key checking")
    monkeypatch.setattr(agent_runtime, "is_model_api_key_configured", unexpected_model_check)
    response = client.post("/api/chat/stream", json={
        "question": "继续", "thread_id": "env-thread",
    })
    assert response.status_code == 409
    assert history.read_conversation_info("env-thread")["binding"] == overridden
    assert len(history.read_conversation_info("env-thread")["messages"]) == 1


def test_page_configuration_and_runtime_use_same_effective_binding(
    client, monkeypatch, source_settings, tmp_path,
):
    from api import configuration_app as config_server

    source_settings()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "test-profile.json").write_text(
        '{"id": "test-profile", "label": "Readable source", "backend": "postgresql"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_server, "SETTINGS_PATH", agent_runtime.SETTINGS_PATH)
    monkeypatch.setattr(config_server, "ACTIVE_PROFILE_PATH", agent_runtime.ACTIVE_PROFILE_PATH)
    monkeypatch.setattr(config_server, "PROFILES_ROOT", profiles)
    monkeypatch.setattr(config_server, "SECRETS_PATH", tmp_path / "absent-secrets.env")
    monkeypatch.setattr(config_server, "PROFILE_SECRETS_ROOT", tmp_path / "absent-profile-secrets")
    monkeypatch.setattr(config_server, "_model_api_key", lambda: "")
    monkeypatch.setattr(config_server, "_knowledge_summary", lambda root: {"path": str(root)})
    monkeypatch.setenv("DATA_AGENT_POSTGRES_DATABASE", "idol")
    monkeypatch.setenv("DATA_AGENT_POSTGRES_USER", "env-reader")
    monkeypatch.setenv("DATA_AGENT_KNOWLEDGE_ROOT", str(tmp_path / "env-knowledge"))

    page_response = client.get("/api/page_configuration")
    assert page_response.status_code == 200
    active = page_response.json()["active"]
    runtime = agent_runtime.read_current_conversation_binding(resolve_current_user("admin-a"))
    assert active["binding"] == runtime
    assert client.get("/api/conversations").json()["active_binding"] == runtime
    assert active["database"] == "idol"
    assert active["username"] == "env-reader"
    assert active["knowledge_root"] == str((tmp_path / "env-knowledge").resolve())
