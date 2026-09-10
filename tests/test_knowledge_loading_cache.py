"""Fast cache regressions against tiny temporary knowledge roots only."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def write_cards(root, count=1):
    root.mkdir(parents=True, exist_ok=True)
    cards = [
        {
            "knowledge_id": f"glossary_term.example.item_{index}",
            "knowledge_type": "glossary_term",
            "title": f"Item {index}",
            "summary": "Tiny fixture",
            "payload": {},
        }
        for index in range(count)
    ]
    path = root / "cards.yaml"
    path.write_text(yaml.safe_dump(cards), encoding="utf-8")
    return path


def profile_for(root, profile_id="next"):
    return {
        "id": profile_id,
        "label": profile_id,
        "backend": "duckdb",
        "duckdb_path": str(root.parent / f"{profile_id}.duckdb"),
        "knowledge_root": str(root),
    }


@pytest.fixture
def knowledge_cache(tmp_path, monkeypatch):
    from api import configuration_app as config
    from api.schemas import ProfilePayload
    from knowledge_runtime import current_knowledge as runtime
    from memory.conversation_binding import RESOURCE_ENV_KEYS

    for field in (
        "KNOWLEDGE_CARDS", "KNOWLEDGE_CATALOG",
        "KNOWLEDGE_NAVIGATION_GRAPH", "KNOWLEDGE_NAVIGATION_GRAPH_TEXT",
        "_ACTIVE_KNOWLEDGE_SUMMARY",
    ):
        monkeypatch.setattr(runtime, field, getattr(runtime, field))
    for key in RESOURCE_ENV_KEYS:
        # Record even absent keys: _apply_payload writes os.environ directly.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)
    for field, relative in (
        ("SETTINGS_PATH", "settings.env"),
        ("SECRETS_PATH", "absent-secrets.env"),
        ("ACTIVE_PROFILE_PATH", ".active_profile"),
        ("PROFILES_ROOT", "profiles"),
        ("PROFILE_SECRETS_ROOT", "profile_secrets"),
    ):
        monkeypatch.setattr(config, field, tmp_path / relative)
    config.PROFILES_ROOT.mkdir()
    monkeypatch.setattr(config, "_model_api_key", lambda: "")
    monkeypatch.setattr(config, "has_active_runs", lambda: False)

    def forbidden_connection(*args, **kwargs):
        raise AssertionError("cache tests must not connect to a business database")

    monkeypatch.setattr(config.duckdb, "connect", forbidden_connection)
    monkeypatch.setattr(config.psycopg2, "connect", forbidden_connection)
    monkeypatch.setattr(config.pymysql, "connect", forbidden_connection)
    root = tmp_path / "old-knowledge"
    source = write_cards(root)
    runtime.reload_knowledge(root)
    config._apply_payload(ProfilePayload(**profile_for(root, "old")), "")
    return SimpleNamespace(root=root, source=source, config=config, runtime=runtime)


def test_repeated_page_configuration_does_not_parse_yaml(client, knowledge_cache, monkeypatch):
    cache = knowledge_cache

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("page configuration reparsed knowledge YAML")

    monkeypatch.setattr(yaml, "load", forbidden_parse)
    for _ in range(3):
        response = client.get("/api/page_configuration")
        assert response.status_code == 200
        assert response.json()["knowledge"] == {
            "path": str(cache.root.resolve()),
            "card_count": 1,
            "types": {"glossary_term": 1},
        }


@pytest.mark.parametrize("endpoint", ["/api/apply-profile", "/api/save-and-apply"])
def test_apply_prepares_once_and_following_gets_do_not_parse(
    client, knowledge_cache, tmp_path, monkeypatch, endpoint,
):
    cache = knowledge_cache
    next_root = tmp_path / "next-knowledge"
    source = write_cards(next_root, 2)
    profile = profile_for(next_root)
    (cache.config.PROFILES_ROOT / "next.json").write_text(json.dumps(profile), encoding="utf-8")

    prepared, parsed = [], []
    original_prepare = cache.runtime.prepare_knowledge
    original_load = yaml.load

    def counted_prepare(root):
        prepared.append(Path(root).resolve())
        return original_prepare(root)

    def counted_load(stream, *args, **kwargs):
        parsed.append(Path(stream.name).resolve())
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(cache.runtime, "prepare_knowledge", counted_prepare)
    if hasattr(cache.config, "prepare_knowledge"):
        monkeypatch.setattr(cache.config, "prepare_knowledge", counted_prepare)
    monkeypatch.setattr(yaml, "load", counted_load)
    body = {"profile_id": "next"} if endpoint.endswith("apply-profile") else profile
    response = client.post(endpoint, json=body)
    assert response.status_code == 200, response.text
    assert prepared == [next_root.resolve()]
    assert parsed == [source.resolve()]

    for _ in range(2):
        response = client.get("/api/page_configuration")
        assert response.status_code == 200
        assert response.json()["knowledge"]["card_count"] == 2
        assert response.json()["knowledge"]["path"] == str(next_root.resolve())
    assert prepared == [next_root.resolve()]
    assert parsed == [source.resolve()]


def test_failed_prepare_keeps_configuration_and_old_runtime(
    client, knowledge_cache, tmp_path, monkeypatch,
):
    cache = knowledge_cache
    bad_root = tmp_path / "bad-knowledge"
    source = write_cards(bad_root)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document[0]["payload"]["required_knowledge_ids"] = ["glossary_term.missing"]
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    profile = profile_for(bad_root, "bad")
    profile_path = cache.config.PROFILES_ROOT / "bad.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    tracked = (cache.config.SETTINGS_PATH, cache.config.ACTIVE_PROFILE_PATH, profile_path)
    original_files = [path.read_bytes() for path in tracked]
    old_cards = cache.runtime.KNOWLEDGE_CARDS
    old_summary = cache.runtime.get_loaded_knowledge_summary(cache.root)
    applied = []
    monkeypatch.setattr(cache.config, "_apply_payload", lambda *args: applied.append(args))

    for endpoint, body in (
        ("/api/apply-profile", {"profile_id": "bad"}),
        ("/api/save-and-apply", {**profile, "label": "Must not be saved"}),
    ):
        response = client.post(endpoint, json=body)
        assert response.status_code == 400
        assert applied == []
        assert [path.read_bytes() for path in tracked] == original_files
        assert cache.runtime.KNOWLEDGE_CARDS is old_cards
        assert cache.runtime.get_loaded_knowledge_summary(cache.root) == old_summary


def test_reload_updates_detached_summary_and_rejects_another_root(
    knowledge_cache, tmp_path, monkeypatch,
):
    cache = knowledge_cache
    write_cards(cache.root, 3)
    assert cache.runtime.get_loaded_knowledge_summary(cache.root)["card_count"] == 1
    cache.config._refresh_knowledge_runtime(cache.root)
    summary = cache.runtime.get_loaded_knowledge_summary(cache.root)
    assert summary["card_count"] == 3
    summary["card_count"] = 999
    summary["types"]["glossary_term"] = 999
    assert cache.runtime.get_loaded_knowledge_summary(cache.root)["card_count"] == 3
    assert cache.runtime.get_loaded_knowledge_summary(cache.root)["types"] == {"glossary_term": 3}

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("summary lookup must not parse YAML")

    monkeypatch.setattr(yaml, "load", forbidden_parse)
    with pytest.raises(ValueError):
        cache.runtime.get_loaded_knowledge_summary(tmp_path / "not-the-loaded-root")


def test_explicit_validation_reads_changed_file_without_replacing_active_summary(
    client, knowledge_cache,
):
    cache = knowledge_cache
    first = client.post("/api/validate-knowledge", json={"knowledge_root": str(cache.root)})
    assert first.status_code == 200
    assert first.json()["details"]["card_count"] == 1
    write_cards(cache.root, 2)
    second = client.post("/api/validate-knowledge", json={"knowledge_root": str(cache.root)})
    assert second.status_code == 200
    assert second.json()["details"]["card_count"] == 2
    assert cache.runtime.get_loaded_knowledge_summary(cache.root)["card_count"] == 1


def test_fast_yaml_loader_rejects_unsafe_python_tags(tmp_path):
    from knowledge_runtime.catalog import load_knowledge_cards

    root = tmp_path / "unsafe"
    root.mkdir()
    (root / "unsafe.yaml").write_text("!!python/tuple [1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_knowledge_cards(root)
