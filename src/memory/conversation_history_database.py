"""Persist conversation history and keep every workspace isolated."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from config.project_paths import CHAT_HISTORY_DATABASE_PATH
from memory import workspace_database


DEFAULT_ACCOUNT = workspace_database.default_account()
DEFAULT_WORKSPACE_ID = str(DEFAULT_ACCOUNT["workspace_id"])
DEFAULT_USER_ID = str(DEFAULT_ACCOUNT["user_id"])


class ConversationAccessError(PermissionError):
    """Raised when a thread ID already belongs to another workspace."""


def connect_chat_history_database() -> sqlite3.Connection:
    connection = sqlite3.connect(CHAT_HISTORY_DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_chat_history_tables() -> None:
    """Create new tables and attach existing conversations to workspace A."""

    CHAT_HISTORY_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    workspace_database.create_workspace_tables(CHAT_HISTORY_DATABASE_PATH)
    connection = connect_chat_history_database()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            created_by_user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            custom_title INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id),
            FOREIGN KEY (created_by_user_id) REFERENCES users(user_id)
        )
        """
    )

    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(conversations)")
    }
    if "workspace_id" not in existing_columns:
        connection.execute(
            """
            ALTER TABLE conversations
            ADD COLUMN workspace_id TEXT REFERENCES workspaces(workspace_id)
            """
        )
    if "created_by_user_id" not in existing_columns:
        connection.execute(
            """
            ALTER TABLE conversations
            ADD COLUMN created_by_user_id TEXT REFERENCES users(user_id)
            """
        )

    connection.execute(
        """
        UPDATE conversations
        SET workspace_id = ?
        WHERE workspace_id IS NULL OR workspace_id = ''
        """,
        (DEFAULT_WORKSPACE_ID,),
    )
    connection.execute(
        """
        UPDATE conversations
        SET created_by_user_id = ?
        WHERE created_by_user_id IS NULL OR created_by_user_id = ''
        """,
        (DEFAULT_USER_ID,),
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS conversations_workspace_updated
        ON conversations(workspace_id, updated_at DESC)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
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
    connection.commit()
    connection.close()


def read_conversation_workspace_id(thread_id: str) -> str | None:
    connection = connect_chat_history_database()
    row = connection.execute(
        "SELECT workspace_id FROM conversations WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    connection.close()
    return None if row is None else str(row[0])


def save_user_message(
    thread_id: str,
    content: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    created_by_user_id: str = DEFAULT_USER_ID,
) -> None:
    """Create or continue one conversation inside the selected workspace."""

    if not workspace_database.user_is_workspace_member(
        created_by_user_id,
        workspace_id,
        CHAT_HISTORY_DATABASE_PATH,
    ):
        raise ConversationAccessError("当前用户不属于这个工作空间。")

    message_created_at = datetime.now(timezone.utc).isoformat()
    user_content = content.strip()
    conversation_title = " ".join(user_content.split())
    if len(conversation_title) > 18:
        conversation_title = conversation_title[:18] + "…"

    connection = connect_chat_history_database()
    existing_conversation = connection.execute(
        """
        SELECT workspace_id
        FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if (
        existing_conversation is not None
        and existing_conversation[0] != workspace_id
    ):
        connection.close()
        raise ConversationAccessError("该会话属于另一个工作空间。")

    if existing_conversation is None:
        connection.execute(
            """
            INSERT INTO conversations (
                thread_id,
                workspace_id,
                created_by_user_id,
                title,
                custom_title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                workspace_id,
                created_by_user_id,
                conversation_title,
                0,
                message_created_at,
                message_created_at,
            ),
        )
    else:
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE thread_id = ? AND workspace_id = ?
            """,
            (message_created_at, thread_id, workspace_id),
        )

    connection.execute(
        """
        INSERT INTO messages(thread_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (thread_id, "user", user_content, message_created_at),
    )
    connection.commit()
    connection.close()


def save_assistant_message(
    thread_id: str,
    content: str,
    details: dict | None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> None:
    """Save the final assistant message only inside its owning workspace."""

    message_created_at = datetime.now(timezone.utc).isoformat()
    assistant_content = content.strip()
    details_json = (
        None
        if details is None
        else json.dumps(details, ensure_ascii=False)
    )

    connection = connect_chat_history_database()
    update_result = connection.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE thread_id = ? AND workspace_id = ?
        """,
        (message_created_at, thread_id, workspace_id),
    )
    if update_result.rowcount == 0:
        connection.close()
        raise ConversationAccessError("该会话不属于当前工作空间。")

    connection.execute(
        """
        INSERT INTO messages(thread_id, role, content, details_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            "assistant",
            assistant_content,
            details_json,
            message_created_at,
        ),
    )
    connection.commit()
    connection.close()


def list_conversation_rows(
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> list[dict]:
    """List only conversations owned by one workspace."""

    connection = connect_chat_history_database()
    rows = connection.execute(
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
        WHERE workspace_id = ?
        ORDER BY updated_at DESC, thread_id DESC
        """,
        (workspace_id,),
    ).fetchall()
    connection.close()
    return [
        {
            "thread_id": row[0],
            "workspace_id": row[1],
            "created_by_user_id": row[2],
            "title": row[3],
            "custom_title": bool(row[4]),
            "created_at": row[5],
            "updated_at": row[6],
        }
        for row in rows
    ]


def read_conversation_info(
    thread_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict | None:
    """Read one conversation only when it belongs to the workspace."""

    connection = connect_chat_history_database()
    conversation_row = connection.execute(
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
        WHERE thread_id = ? AND workspace_id = ?
        """,
        (thread_id, workspace_id),
    ).fetchone()
    if conversation_row is None:
        connection.close()
        return None

    message_rows = connection.execute(
        """
        SELECT id, role, content, details_json, created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()
    connection.close()

    messages = []
    for row in message_rows:
        details = None if row[3] is None else json.loads(row[3])
        # 旧会话不改 SQLite 原始数据；读取时统一成现在的 request_id 契约。
        if (
            isinstance(details, dict)
            and "request_id" not in details
            and "run_id" in details
        ):
            details["request_id"] = details.pop("run_id")
        messages.append(
            {
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "details": details,
                "created_at": row[4],
            }
        )

    return {
        "thread_id": conversation_row[0],
        "workspace_id": conversation_row[1],
        "created_by_user_id": conversation_row[2],
        "title": conversation_row[3],
        "custom_title": bool(conversation_row[4]),
        "created_at": conversation_row[5],
        "updated_at": conversation_row[6],
        "messages": messages,
    }


def rename_conversation(
    thread_id: str,
    new_title: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> bool:
    clean_title = " ".join(new_title.split())
    if clean_title == "":
        raise ValueError("会话标题不能为空。")

    connection = connect_chat_history_database()
    result = connection.execute(
        """
        UPDATE conversations
        SET title = ?, custom_title = 1
        WHERE thread_id = ? AND workspace_id = ?
        """,
        (clean_title, thread_id, workspace_id),
    )
    connection.commit()
    connection.close()
    return result.rowcount > 0


def delete_conversation(
    thread_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> bool:
    connection = connect_chat_history_database()
    result = connection.execute(
        """
        DELETE FROM conversations
        WHERE thread_id = ? AND workspace_id = ?
        """,
        (thread_id, workspace_id),
    )
    connection.commit()
    connection.close()
    return result.rowcount > 0
