import sqlite3

import pytest

from memory import conversation_history_database


def prepare_temporary_database(tmp_path, monkeypatch):
    database_path = tmp_path / "chat_history.sqlite"
    monkeypatch.setattr(
        conversation_history_database,
        "CHAT_HISTORY_DATABASE_PATH",
        database_path,
    )
    conversation_history_database.create_chat_history_tables()
    return database_path


def test_list_rows_and_read_conversation_info(tmp_path, monkeypatch):
    database_path = prepare_temporary_database(tmp_path, monkeypatch)

    conversation_history_database.save_user_message(
        "thread-1",
        "第一行\n第二行",
    )
    conversation_history_database.save_assistant_message(
        "thread-1",
        "分析完成",
        {"status": "success", "answer": "分析完成"},
    )
    conversation_history_database.save_user_message(
        "thread-2",
        "第二个会话",
    )

    database_connection = sqlite3.connect(database_path)
    database_connection.execute(
        "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
        ("2026-08-31T10:00:00+00:00", "thread-1"),
    )
    database_connection.execute(
        "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
        ("2026-08-31T11:00:00+00:00", "thread-2"),
    )
    database_connection.commit()
    database_connection.close()

    conversation_list = conversation_history_database.list_conversation_rows()
    assert [item["thread_id"] for item in conversation_list] == [
        "thread-2",
        "thread-1",
    ]

    conversation = conversation_history_database.read_conversation_info("thread-1")
    assert conversation is not None
    assert conversation["title"] == "第一行 第二行"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert conversation["messages"][0]["content"] == "第一行\n第二行"
    assert conversation["messages"][1]["details"] == {
        "status": "success",
        "answer": "分析完成",
    }


def test_rename_conversation(tmp_path, monkeypatch):
    prepare_temporary_database(tmp_path, monkeypatch)
    conversation_history_database.save_user_message("thread-1", "原始标题")
    original_updated_at = conversation_history_database.read_conversation_info(
        "thread-1"
    )["updated_at"]

    renamed = conversation_history_database.rename_conversation(
        "thread-1",
        "  新的   标题  ",
    )

    conversation = conversation_history_database.read_conversation_info("thread-1")
    assert renamed is True
    assert conversation is not None
    assert conversation["title"] == "新的 标题"
    assert conversation["custom_title"] is True
    assert conversation["updated_at"] == original_updated_at
    with pytest.raises(ValueError, match="会话标题不能为空"):
        conversation_history_database.rename_conversation(
            "thread-1",
            "   ",
        )
    assert conversation_history_database.rename_conversation(
        "missing-thread",
        "不存在",
    ) is False


def test_old_run_id_is_returned_as_request_id(tmp_path, monkeypatch):
    prepare_temporary_database(tmp_path, monkeypatch)
    conversation_history_database.save_user_message("thread-1", "兼容旧记录")
    conversation_history_database.save_assistant_message(
        "thread-1",
        "旧回答",
        {"run_id": "old-request"},
    )

    conversation = conversation_history_database.read_conversation_info("thread-1")

    assert conversation is not None
    details = conversation["messages"][1]["details"]
    assert details == {"request_id": "old-request"}


def test_delete_conversation_also_deletes_messages(tmp_path, monkeypatch):
    database_path = prepare_temporary_database(tmp_path, monkeypatch)
    conversation_history_database.save_user_message("thread-1", "测试问题")
    conversation_history_database.save_assistant_message(
        "thread-1",
        "测试回答",
        None,
    )

    deleted = conversation_history_database.delete_conversation("thread-1")

    database_connection = sqlite3.connect(database_path)
    remaining_messages = database_connection.execute(
        "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
        ("thread-1",),
    ).fetchone()[0]
    database_connection.close()

    assert deleted is True
    assert remaining_messages == 0
    assert conversation_history_database.read_conversation_info("thread-1") is None
    assert conversation_history_database.delete_conversation("thread-1") is False
