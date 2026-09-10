from memory.conversation_binding import (
    build_conversation_binding,
    effective_resource_settings,
    read_active_conversation_binding,
)


def test_resource_identity_uses_database_not_profile_name(tmp_path):
    profile = {"id": "cold", "label": "Cold", "backend": "postgresql", "host": "LOCALHOST", "port": 5432, "database": "cold", "knowledge_root": str(tmp_path)}
    original = build_conversation_binding(profile)
    renamed = build_conversation_binding({**profile, "id": "another", "label": "Renamed", "host": "localhost", "password": "secret", "username": "another-reader"})
    assert original["data_source_id"] == renamed["data_source_id"]
    assert original["knowledge_base_id"] == renamed["knowledge_base_id"]
    assert "secret" not in str(renamed)
    assert build_conversation_binding({**profile, "database": "idol"})["data_source_id"] != original["data_source_id"]
    assert build_conversation_binding({**profile, "port": 5433})["data_source_id"] != original["data_source_id"]
    other_knowledge = build_conversation_binding({**profile, "knowledge_root": str(tmp_path / "other")})
    assert other_knowledge["data_source_id"] == original["data_source_id"]
    assert other_knowledge["knowledge_base_id"] != original["knowledge_base_id"]


def test_effective_settings_win_over_saved_profile(tmp_path):
    settings = tmp_path / "settings.env"
    active = tmp_path / ".active_profile"
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    active.write_text("shared", encoding="utf-8")
    (profiles / "shared.json").write_text('{"label": "Readable name", "database": "old"}', encoding="utf-8")
    settings.write_text('DATA_AGENT_POSTGRES_DATABASE="new"\nDATA_AGENT_POSTGRES_PORT=5432\n', encoding="utf-8")
    result = read_active_conversation_binding(settings, active, environment={})
    assert result["data_source_name"] == "Readable name"
    assert result["database"] == "new"
    assert result["profile_id"] == "shared"


def test_duckdb_path_and_knowledge_path_are_normalized(tmp_path):
    profile = {"backend": "duckdb", "duckdb_path": str(tmp_path / "db.duckdb"), "knowledge_root": str(tmp_path / "kb")}
    other = {**profile, "duckdb_path": str(tmp_path / "nested" / ".." / "db.duckdb"), "knowledge_root": str(tmp_path / "nested" / ".." / "kb")}
    assert build_conversation_binding(profile) == build_conversation_binding(other)


def test_effective_resources_whitelist_environment_and_preserve_empty_overrides():
    settings = {
        "DATA_AGENT_POSTGRES_HOST": "file-host",
        "DATA_AGENT_POSTGRES_DATABASE": "file-db",
        "DATA_AGENT_POSTGRES_PASSWORD": "file-secret",
    }
    environment = {
        "DATA_AGENT_POSTGRES_HOST": "",
        "DATA_AGENT_POSTGRES_DATABASE": "env-db",
        "DATA_AGENT_POSTGRES_PASSWORD": "env-secret",
        "DEEPSEEK_API_KEY": "model-secret",
    }
    assert effective_resource_settings(settings, environment) == {
        "DATA_AGENT_POSTGRES_HOST": "",
        "DATA_AGENT_POSTGRES_DATABASE": "env-db",
    }
    assert effective_resource_settings(settings, {}) == {
        "DATA_AGENT_POSTGRES_HOST": "file-host",
        "DATA_AGENT_POSTGRES_DATABASE": "file-db",
    }


def test_environment_database_and_knowledge_override_file_binding(tmp_path):
    settings = tmp_path / "settings.env"
    active = tmp_path / ".active_profile"
    file_knowledge = tmp_path / "file-knowledge"
    env_knowledge = tmp_path / "env-knowledge"
    settings.write_text(
        "DATA_AGENT_DATABASE_BACKEND=postgresql\n"
        "DATA_AGENT_POSTGRES_DATABASE=cold\n"
        f"DATA_AGENT_KNOWLEDGE_ROOT={file_knowledge}\n",
        encoding="utf-8",
    )
    original = read_active_conversation_binding(settings, active, environment={})
    overridden = read_active_conversation_binding(settings, active, environment={
        "DATA_AGENT_POSTGRES_DATABASE": "idol",
        "DATA_AGENT_KNOWLEDGE_ROOT": str(env_knowledge),
    })
    assert overridden["database"] == "idol"
    assert overridden["knowledge_root"] == str(env_knowledge.resolve())
    assert overridden["data_source_id"] != original["data_source_id"]
    assert overridden["knowledge_base_id"] != original["knowledge_base_id"]
    assert read_active_conversation_binding(settings, active, environment={}) == original
