"""Seed mutable runtime files, then start the single product Web service."""

from pathlib import Path
import os
import shutil


APP_ROOT = Path("/app")
CONFIG_ROOT = Path(os.environ.get("DATA_AGENT_CONFIG_ROOT", "/app/runtime/config"))


def copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and source.is_file():
        shutil.copyfile(source, destination)


CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "profiles").mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "profile_secrets").mkdir(parents=True, exist_ok=True)
Path(os.environ["DATA_AGENT_KNOWLEDGE_IMPORT_ROOT"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["DATA_AGENT_CHECKPOINT_PATH"]).parent.mkdir(parents=True, exist_ok=True)

copy_once(APP_ROOT / "config" / "settings.env.example", CONFIG_ROOT / "settings.env")
copy_once(APP_ROOT / "config" / "secrets.env.example", CONFIG_ROOT / "secrets.env")
copy_once(
    APP_ROOT / "config" / "profiles" / "livesqlbench-cold-chain.json",
    CONFIG_ROOT / "profiles" / "livesqlbench-cold-chain.json",
)

os.execvp(
    "uvicorn",
    [
        "uvicorn",
        "config_ui_server:app",
        "--app-dir",
        "/app/src",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--no-access-log",
    ],
)
