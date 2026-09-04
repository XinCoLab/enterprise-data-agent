import os
from pathlib import Path
import tempfile

import pytest


_test_history_directory = tempfile.TemporaryDirectory(
    prefix="data-agent-chat-history-tests-"
)
os.environ["DATA_AGENT_CHAT_HISTORY_PATH"] = str(
    Path(_test_history_directory.name) / "chat_history.sqlite"
)
os.environ["DATA_AGENT_CHECKPOINT_PATH"] = str(
    Path(_test_history_directory.name) / "langgraph_checkpoints.sqlite"
)


@pytest.fixture(autouse=True)
def isolate_chat_history_database(tmp_path, monkeypatch):
    from memory import conversation_history_database, workspace_database

    database_path = tmp_path / "chat_history.sqlite"
    monkeypatch.setattr(
        conversation_history_database,
        "CHAT_HISTORY_DATABASE_PATH",
        database_path,
    )
    monkeypatch.setattr(
        workspace_database,
        "CHAT_HISTORY_DATABASE_PATH",
        database_path,
    )
    conversation_history_database.create_chat_history_tables()


@pytest.fixture
def client(isolate_chat_history_database):
    from fastapi.testclient import TestClient

    from api.app import app

    with TestClient(app) as test_client:
        yield test_client
