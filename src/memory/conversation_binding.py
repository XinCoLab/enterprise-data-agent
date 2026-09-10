"""Stable conversation resource identities derived from non-secret configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping
from uuid import NAMESPACE_URL, uuid5

from config.project_paths import PROJECT_ROOT


RESOURCE_ENV_KEYS = (
    "DATA_AGENT_DATABASE_BACKEND",
    "DATA_AGENT_POSTGRES_HOST", "DATA_AGENT_POSTGRES_PORT",
    "DATA_AGENT_POSTGRES_USER", "DATA_AGENT_POSTGRES_DATABASE",
    "DATA_AGENT_MYSQL_HOST", "DATA_AGENT_MYSQL_PORT",
    "DATA_AGENT_MYSQL_USER", "DATA_AGENT_MYSQL_DATABASE",
    "DATA_AGENT_DATABASE_PATH", "DATA_AGENT_KNOWLEDGE_ROOT",
)


def effective_resource_settings(
    settings: Mapping[str, str],
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Match runtime environment precedence without copying credentials.

    Startup environment values override settings.env, as load_dotenv(override=False)
    and the SQL adapters do. Explicit empty values are overrides, not fallbacks.
    """
    environment = os.environ if environment is None else environment
    return {
        key: environment[key] if key in environment else settings[key]
        for key in RESOURCE_ENV_KEYS
        if key in environment or key in settings
    }


def _local_path(value: str, default: Path) -> Path:
    path = Path(value.strip()).expanduser() if value.strip() else default
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def build_conversation_binding(profile: Mapping[str, object]) -> dict[str, str]:
    """One physical database groups histories; each thread also pins a KB root.

    Profile names, passwords and usernames are not database identifiers. Editing
    a profile to point elsewhere creates a different source ID. Editing knowledge
    cards in the same root does not create a new knowledge collection.
    """
    backend = str(profile.get("backend", "postgresql")).strip().lower()
    database = str(profile.get("database", "")).strip()
    if backend == "duckdb":
        database_path = _local_path(
            str(profile.get("duckdb_path", "")), PROJECT_ROOT / "databases" / "data_agent.duckdb"
        )
        identity = {"backend": backend, "path": os.path.normcase(str(database_path))}
        database = database_path.name
    else:
        identity = {
            "backend": backend,
            "host": str(profile.get("host", "")).strip().lower() or "127.0.0.1",
            "port": int(profile.get("port", 0) or (3306 if backend == "mysql" else 5432)),
            "database": database,
        }
    root = _local_path(str(profile.get("knowledge_root", "")), PROJECT_ROOT / "knowledge")
    source_key = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    label = str(profile.get("label", "")).strip()
    return {
        "data_source_id": "ds_" + uuid5(NAMESPACE_URL, "data-agent:source:" + source_key).hex,
        "data_source_name": label or database or backend,
        "knowledge_base_id": "kb_" + uuid5(NAMESPACE_URL, "data-agent:knowledge:" + os.path.normcase(str(root))).hex,
        "knowledge_base_name": root.name,
        "knowledge_root": str(root),
        "profile_id": str(profile.get("id", "current")),
        "backend": backend,
        "database": database,
    }


def read_active_conversation_binding(
    settings_path: Path,
    active_profile_path: Path,
    profiles_root: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read effective settings, not possibly stale saved profile connection values."""
    settings: dict[str, str] = {}
    if settings_path.is_file():
        for line in settings_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            settings[key.strip()] = value
    settings = effective_resource_settings(settings, environment)
    backend = settings.get("DATA_AGENT_DATABASE_BACKEND", "postgresql").strip().lower()
    prefix = "DATA_AGENT_MYSQL_" if backend == "mysql" else "DATA_AGENT_POSTGRES_"
    profile_id = active_profile_path.read_text(encoding="utf-8").strip() if active_profile_path.is_file() else "current"
    label = ""
    profiles_root = profiles_root or active_profile_path.parent / "profiles"
    # The profile pointer is a local filename, never a path supplied to open().
    if profile_id and all(character.isascii() and (character.isalnum() or character == "-") for character in profile_id):
        document_path = profiles_root / (profile_id + ".json")
        if document_path.is_file():
            try:
                document = json.loads(document_path.read_text(encoding="utf-8"))
                if isinstance(document, dict):
                    label = str(document.get("label", ""))
            except (OSError, ValueError):
                pass
    return build_conversation_binding({
        "id": profile_id or "current", "label": label, "backend": backend,
        "host": settings.get(prefix + "HOST", "127.0.0.1"),
        "port": settings.get(prefix + "PORT", "3306" if backend == "mysql" else "5432"),
        "database": settings.get(prefix + "DATABASE", ""),
        "duckdb_path": settings.get("DATA_AGENT_DATABASE_PATH", ""),
        "knowledge_root": settings.get("DATA_AGENT_KNOWLEDGE_ROOT", ""),
    })
