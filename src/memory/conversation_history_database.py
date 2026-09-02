"""保存和查询不同会话的聊天记录。"""
import json
import sqlite3
from datetime import datetime, timezone

from config.project_paths import CHAT_HISTORY_DATABASE_PATH




def create_chat_history_tables() -> None:
    CHAT_HISTORY_DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    database_connection: sqlite3.Connection = sqlite3.connect(
            CHAT_HISTORY_DATABASE_PATH
    )
    database_connection.execute("PRAGMA foreign_keys = ON")
    database_connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations(
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        custom_title    INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL,
        updated_at      TEXT NOT NULL
        );
        """
    )

    database_connection.execute(
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
    database_connection.commit()
    database_connection.close()

def save_user_message(
    thread_id: str,
    content: str,
) -> None:
    message_created_at = datetime.now(timezone.utc).isoformat()  # 当前 UTC 时间
    user_content = content.strip()
    conversation_title = " ".join(user_content.split())
    if len(conversation_title) > 18:
        conversation_title = conversation_title[:18] + "…"

    CHAT_HISTORY_DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    database_connection: sqlite3.Connection = sqlite3.connect(
            CHAT_HISTORY_DATABASE_PATH
    )
    database_connection.execute("PRAGMA foreign_keys = ON")

    database_connection.execute(
        """
        INSERT OR IGNORE INTO conversations(thread_id,title,custom_title,created_at,updated_at)
        VALUES(?,?,?,?,?)
        """,
        (
            thread_id,
            conversation_title,
            0,
            message_created_at,
            message_created_at,
        ),
    )

    database_connection.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE thread_id = ?
        """,
        (
            message_created_at,
            thread_id,
        ),
    )

    database_connection.execute(
        """
        INSERT INTO messages (
            thread_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            thread_id,
            "user",
            user_content,
            message_created_at,
        ),
    )

    database_connection.commit()
    database_connection.close()


def save_assistant_message(
    thread_id: str,
    content: str,
    details: dict | None,
) -> None:
    message_created_at = datetime.now(timezone.utc).isoformat()
    assistant_content = content.strip()
    details_json = (
        None
        if details is None
        else json.dumps(details, ensure_ascii=False)
    )
    CHAT_HISTORY_DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    database_connection: sqlite3.Connection = sqlite3.connect(
            CHAT_HISTORY_DATABASE_PATH
    )
    database_connection.execute("PRAGMA foreign_keys = ON")

    database_connection.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE thread_id = ?
        """,
        (
            message_created_at,
            thread_id,
        ),
    )

    database_connection.execute(
        """
        INSERT INTO messages (
            thread_id,
            role,
            content,
            details_json,
            created_at
        )
        VALUES(?,?,?,?,?)
        """,
        (
            thread_id,
            "assistant",
            assistant_content,
            details_json,
            message_created_at,
        ),
    )
    database_connection.commit()
    database_connection.close()


def list_conversation_rows() -> list[dict]:
    """查询全部会话，最近更新的会话排在最前面。"""
    database_connection = sqlite3.connect(CHAT_HISTORY_DATABASE_PATH)
    conversation_rows = database_connection.execute(
        """
        SELECT
            thread_id,
            title,
            custom_title,
            created_at,
            updated_at
        FROM conversations
        ORDER BY updated_at DESC, thread_id DESC
        """
    ).fetchall()
    database_connection.close()

    conversation_list = []
    for conversation_row in conversation_rows:
        conversation_list.append(
            {
                "thread_id": conversation_row[0],
                "title": conversation_row[1],
                "custom_title": bool(conversation_row[2]),
                "created_at": conversation_row[3],
                "updated_at": conversation_row[4],
            }
        )

    return conversation_list


def read_conversation_info(thread_id: str) -> dict | None:
    """查询一个会话及其全部消息。"""
    database_connection = sqlite3.connect(CHAT_HISTORY_DATABASE_PATH)
    conversation_row = database_connection.execute(
        """
        SELECT
            thread_id,
            title,
            custom_title,
            created_at,
            updated_at
        FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()

    if conversation_row is None:
        database_connection.close()
        return None

    message_rows = database_connection.execute(
        """
        SELECT
            id,
            role,
            content,
            details_json,
            created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()
    database_connection.close()

    message_list = []
    for message_row in message_rows:
        if message_row[3] is None:
            message_details = None
        else:
            message_details = json.loads(message_row[3])

        message_list.append(
            {
                "id": message_row[0],
                "role": message_row[1],
                "content": message_row[2],
                "details": message_details,
                "created_at": message_row[4],
            }
        )

    return {
        "thread_id": conversation_row[0],
        "title": conversation_row[1],
        "custom_title": bool(conversation_row[2]),
        "created_at": conversation_row[3],
        "updated_at": conversation_row[4],
        "messages": message_list,
    }


def rename_conversation(thread_id: str, new_title: str) -> bool:
    """重命名会话；找到会话时返回 True。"""
    clean_title = " ".join(new_title.split())
    if clean_title == "":
        raise ValueError("会话标题不能为空。")

    database_connection = sqlite3.connect(CHAT_HISTORY_DATABASE_PATH)
    update_result = database_connection.execute(
        """
        UPDATE conversations
        SET
            title = ?,
            custom_title = 1
        WHERE thread_id = ?
        """,
        (
            clean_title,
            thread_id,
        ),
    )
    database_connection.commit()
    database_connection.close()

    return update_result.rowcount > 0


def delete_conversation(thread_id: str) -> bool:
    """删除会话及其全部消息；找到会话时返回 True。"""
    database_connection = sqlite3.connect(CHAT_HISTORY_DATABASE_PATH)
    database_connection.execute("PRAGMA foreign_keys = ON")
    delete_result = database_connection.execute(
        """
        DELETE FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,),
    )
    database_connection.commit()
    database_connection.close()

    return delete_result.rowcount > 0
