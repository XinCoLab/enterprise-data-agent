import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_APP_LAB_ROOT = PROJECT_ROOT


def _project_relative_path(raw_path: str, default: Path) -> Path:
    path = Path(raw_path).expanduser() if raw_path.strip() else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# Docker and packaged deployments may keep mutable configuration outside the
# application image. Local development continues to use ``./config``.
CONFIG_ROOT = _project_relative_path(
    os.getenv("DATA_AGENT_CONFIG_ROOT", ""),
    PROJECT_ROOT / "config",
)


def _shared_env_path() -> Path:
    """Locate the private local secrets file for this product checkout."""

    configured = os.getenv("DATA_AGENT_ENV_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    return CONFIG_ROOT / "secrets.env"


BACKEND_ENV_PATH = _shared_env_path()
load_dotenv(CONFIG_ROOT / "settings.env", override=False)
load_dotenv(BACKEND_ENV_PATH, override=False)


def _configured_path(environment_name: str, default: Path) -> Path:
    """Resolve an optional runner-supplied path without changing defaults."""

    configured = os.getenv(environment_name, "").strip()
    if not configured:
        return default
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "databases" / "data_agent.duckdb"
DEFAULT_KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"
DEFAULT_KNOWLEDGE_IMPORT_ROOT = PROJECT_ROOT / "runtime" / "knowledge"
DEFAULT_MAIN_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "system_prompt.md"

# Benchmark runners may explicitly override these deployment inputs before
# importing any Agent module. Product defaults remain domain-neutral.
DATABASE_PATH = _configured_path("DATA_AGENT_DATABASE_PATH", DEFAULT_DATABASE_PATH)
DATABASE_BACKEND = os.getenv("DATA_AGENT_DATABASE_BACKEND", "postgresql").strip().lower()
POSTGRES_HOST = os.getenv("DATA_AGENT_POSTGRES_HOST", "127.0.0.1").strip()
POSTGRES_PORT = int(os.getenv("DATA_AGENT_POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("DATA_AGENT_POSTGRES_USER", "").strip()
POSTGRES_PASSWORD = os.getenv("DATA_AGENT_POSTGRES_PASSWORD", "")
POSTGRES_DATABASE = os.getenv("DATA_AGENT_POSTGRES_DATABASE", "").strip()
MYSQL_HOST = os.getenv("DATA_AGENT_MYSQL_HOST", "127.0.0.1").strip()
MYSQL_PORT = int(os.getenv("DATA_AGENT_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("DATA_AGENT_MYSQL_USER", "").strip()
MYSQL_PASSWORD = os.getenv("DATA_AGENT_MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("DATA_AGENT_MYSQL_DATABASE", "").strip()
KNOWLEDGE_ROOT = _configured_path("DATA_AGENT_KNOWLEDGE_ROOT", DEFAULT_KNOWLEDGE_ROOT)
KNOWLEDGE_IMPORT_ROOT = _configured_path(
    "DATA_AGENT_KNOWLEDGE_IMPORT_ROOT",
    DEFAULT_KNOWLEDGE_IMPORT_ROOT,
)
MAIN_PROMPT_PATH = _configured_path(
    "DATA_AGENT_MAIN_PROMPT_PATH",
    DEFAULT_MAIN_PROMPT_PATH,
)
CHECKPOINT_DATABASE_PATH = _configured_path(
    "DATA_AGENT_CHECKPOINT_PATH",
    PROJECT_ROOT / "databases" / "langgraph_conversation_memory.sqlite",
)
