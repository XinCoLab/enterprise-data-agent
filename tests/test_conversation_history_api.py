from fastapi.testclient import TestClient

from api.app import app
from memory import conversation_history_database


client = TestClient(app)


def seed_conversation() -> None:
    conversation_history_database.save_user_message(
        "thread-1",
        "第一行\n第二行",
    )
    conversation_history_database.save_assistant_message(
        "thread-1",
        "分析完成",
        {
            "status": "success",
            "answer": "分析完成",
        },
    )


def test_list_and_open_saved_conversation():
    seed_conversation()

    list_response = client.get("/api/conversations")
    assert list_response.status_code == 200
    conversation_list = list_response.json()["conversations"]
    assert conversation_list == [
        {
            "thread_id": "thread-1",
            "workspace_id": "workspace-a",
            "created_by_user_id": "user-admin-a",
            "title": "第一行 第二行",
            "custom_title": False,
            "created_at": conversation_list[0]["created_at"],
            "updated_at": conversation_list[0]["updated_at"],
        }
    ]

    detail_response = client.get("/api/conversations/thread-1")
    assert detail_response.status_code == 200
    conversation = detail_response.json()
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
    ]
    assert conversation["messages"][0]["content"] == "第一行\n第二行"
    assert conversation["messages"][1]["details"]["answer"] == "分析完成"


def test_rename_and_delete_saved_conversation():
    seed_conversation()

    rename_response = client.patch(
        "/api/conversations/thread-1",
        json={"title": "  新的   标题  "},
    )
    assert rename_response.status_code == 200
    renamed = rename_response.json()["conversation"]
    assert renamed["title"] == "新的 标题"
    assert renamed["custom_title"] is True

    delete_response = client.delete("/api/conversations/thread-1")
    assert delete_response.status_code == 200
    assert delete_response.json()["thread_id"] == "thread-1"
    assert client.get("/api/conversations/thread-1").status_code == 404
    assert client.delete("/api/conversations/thread-1").status_code == 404


def test_conversation_api_validates_title_and_missing_conversation():
    assert client.get("/api/conversations/missing").status_code == 404
    assert client.patch(
        "/api/conversations/missing",
        json={"title": "新标题"},
    ).status_code == 404
    assert client.patch(
        "/api/conversations/missing",
        json={"title": "   "},
    ).status_code == 422
