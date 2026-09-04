import sqlite3

import pytest

from memory import conversation_history_database, workspace_database
from security.workspace_access import resolve_current_user


def test_default_simulation_accounts_are_created_with_database_roles():
    accounts = workspace_database.list_dev_accounts()

    assert [account["login_id"] for account in accounts] == [
        "admin-a",
        "analyst-b",
        "viewer-c",
    ]
    assert [account["workspace_id"] for account in accounts] == [
        "workspace-a",
        "workspace-b",
        "workspace-c",
    ]
    assert [account["role"] for account in accounts] == [
        "admin",
        "analyst",
        "viewer",
    ]
    assert resolve_current_user(None).login_id == "admin-a"


def test_simulation_header_selects_identity_but_not_its_role():
    connection = sqlite3.connect(workspace_database.CHAT_HISTORY_DATABASE_PATH)
    connection.execute(
        """
        UPDATE workspace_members
        SET role = 'viewer'
        WHERE workspace_id = 'workspace-a' AND user_id = 'user-admin-a'
        """
    )
    connection.commit()
    connection.close()

    selected_user = resolve_current_user("admin-a")

    assert selected_user.login_id == "admin-a"
    assert selected_user.role == "viewer"
    assert selected_user.permissions == frozenset(
        {"conversation:read", "config:read"}
    )


def test_conversation_queries_are_scoped_to_workspace():
    conversation_history_database.save_user_message(
        "thread-a",
        "A 的问题",
        workspace_id="workspace-a",
        created_by_user_id="user-admin-a",
    )
    conversation_history_database.save_user_message(
        "thread-b",
        "B 的问题",
        workspace_id="workspace-b",
        created_by_user_id="user-analyst-b",
    )

    assert [
        row["thread_id"]
        for row in conversation_history_database.list_conversation_rows(
            "workspace-a"
        )
    ] == ["thread-a"]
    assert [
        row["thread_id"]
        for row in conversation_history_database.list_conversation_rows(
            "workspace-b"
        )
    ] == ["thread-b"]
    assert (
        conversation_history_database.read_conversation_info(
            "thread-a",
            "workspace-b",
        )
        is None
    )
    assert conversation_history_database.rename_conversation(
        "thread-a",
        "越权改名",
        "workspace-b",
    ) is False
    assert conversation_history_database.delete_conversation(
        "thread-a",
        "workspace-b",
    ) is False


def test_other_workspace_cannot_claim_an_existing_thread_id():
    conversation_history_database.save_user_message(
        "guessed-thread",
        "A 的秘密问题",
        workspace_id="workspace-a",
        created_by_user_id="user-admin-a",
    )

    with pytest.raises(
        conversation_history_database.ConversationAccessError,
        match="另一个工作空间",
    ):
        conversation_history_database.save_user_message(
            "guessed-thread",
            "B 猜中了 thread_id",
            workspace_id="workspace-b",
            created_by_user_id="user-analyst-b",
        )
    with pytest.raises(conversation_history_database.ConversationAccessError):
        conversation_history_database.save_assistant_message(
            "guessed-thread",
            "不应写入",
            None,
            workspace_id="workspace-b",
        )

    owner_conversation = conversation_history_database.read_conversation_info(
        "guessed-thread",
        "workspace-a",
    )
    assert owner_conversation is not None
    assert [message["content"] for message in owner_conversation["messages"]] == [
        "A 的秘密问题"
    ]
    assert (
        conversation_history_database.read_conversation_info(
            "guessed-thread",
            "workspace-b",
        )
        is None
    )
