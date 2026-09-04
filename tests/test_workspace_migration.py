import sqlite3

from memory import conversation_history_database, workspace_database


def create_legacy_chat_database(database_path) -> None:
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE conversations (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            custom_title INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            details_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (thread_id)
                REFERENCES conversations(thread_id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        INSERT INTO conversations(
            thread_id,
            title,
            custom_title,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "legacy-thread",
            "旧会话",
            1,
            "2026-08-31T10:00:00+00:00",
            "2026-08-31T10:01:00+00:00",
        ),
    )
    connection.execute(
        """
        INSERT INTO messages(
            thread_id,
            role,
            content,
            details_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "legacy-thread",
            "assistant",
            "旧回答仍然存在",
            '{"answer":"旧回答仍然存在"}',
            "2026-08-31T10:01:00+00:00",
        ),
    )
    connection.commit()
    connection.close()


def test_legacy_database_migration_is_lossless_and_idempotent(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "legacy_chat_history.sqlite"
    create_legacy_chat_database(database_path)
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
    conversation_history_database.create_chat_history_tables()

    connection = sqlite3.connect(database_path)
    conversation_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(conversations)")
    }
    migrated_conversation = connection.execute(
        """
        SELECT
            thread_id,
            workspace_id,
            created_by_user_id,
            title,
            custom_title,
            created_at,
            updated_at
        FROM conversations
        WHERE thread_id = ?
        """,
        ("legacy-thread",),
    ).fetchone()
    migrated_message = connection.execute(
        """
        SELECT role, content, details_json, created_at
        FROM messages
        WHERE thread_id = ?
        """,
        ("legacy-thread",),
    ).fetchone()
    table_counts = {
        table_name: connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        for table_name in ("users", "workspaces", "workspace_members")
    }
    index_names = {
        row[1] for row in connection.execute("PRAGMA index_list(conversations)")
    }
    foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    connection.close()

    assert {"workspace_id", "created_by_user_id"} <= conversation_columns
    assert migrated_conversation == (
        "legacy-thread",
        "workspace-a",
        "user-admin-a",
        "旧会话",
        1,
        "2026-08-31T10:00:00+00:00",
        "2026-08-31T10:01:00+00:00",
    )
    assert migrated_message == (
        "assistant",
        "旧回答仍然存在",
        '{"answer":"旧回答仍然存在"}',
        "2026-08-31T10:01:00+00:00",
    )
    assert table_counts == {
        "users": 3,
        "workspaces": 3,
        "workspace_members": 3,
    }
    assert "conversations_workspace_updated" in index_names
    assert foreign_key_errors == []
