from fastapi.testclient import TestClient

from api.app import app
from memory import conversation_history_database


client = TestClient(app)


def save_conversation(
    thread_id: str,
    *,
    workspace_id: str,
    user_id: str,
) -> None:
    conversation_history_database.save_user_message(
        thread_id,
        f"{workspace_id} 的问题",
        workspace_id=workspace_id,
        created_by_user_id=user_id,
    )


def test_account_endpoint_exposes_selected_simulation_identity_and_permissions():
    default_response = client.get("/api/accounts")
    analyst_response = client.get(
        "/api/accounts",
        headers={"X-Dev-User": "analyst-b"},
    )
    unknown_response = client.get(
        "/api/accounts",
        headers={"X-Dev-User": "missing-user"},
    )

    assert default_response.status_code == 200
    default_payload = default_response.json()
    assert default_payload["demo_mode"] is True
    assert default_payload["current"]["login_id"] == "admin-a"
    assert default_payload["current"]["role"] == "admin"
    assert "config:write" in default_payload["current"]["permissions"]
    assert [account["login_id"] for account in default_payload["accounts"]] == [
        "admin-a",
        "analyst-b",
        "viewer-c",
    ]

    assert analyst_response.status_code == 200
    analyst = analyst_response.json()["current"]
    assert analyst["workspace_id"] == "workspace-b"
    assert analyst["role"] == "analyst"
    assert "chat:run" in analyst["permissions"]
    assert "config:write" not in analyst["permissions"]
    assert unknown_response.status_code == 401


def test_conversation_api_hides_other_workspaces():
    save_conversation(
        "thread-a",
        workspace_id="workspace-a",
        user_id="user-admin-a",
    )
    save_conversation(
        "thread-b",
        workspace_id="workspace-b",
        user_id="user-analyst-b",
    )

    list_a = client.get("/api/conversations")
    list_b = client.get(
        "/api/conversations",
        headers={"X-Dev-User": "analyst-b"},
    )
    guessed_read = client.get(
        "/api/conversations/thread-a",
        headers={"X-Dev-User": "analyst-b"},
    )
    guessed_rename = client.patch(
        "/api/conversations/thread-a",
        headers={"X-Dev-User": "analyst-b"},
        json={"title": "越权改名"},
    )
    guessed_delete = client.delete(
        "/api/conversations/thread-a",
        headers={"X-Dev-User": "analyst-b"},
    )

    assert [row["thread_id"] for row in list_a.json()["conversations"]] == [
        "thread-a"
    ]
    assert [row["thread_id"] for row in list_b.json()["conversations"]] == [
        "thread-b"
    ]
    assert guessed_read.status_code == 404
    assert guessed_rename.status_code == 404
    assert guessed_delete.status_code == 404


def test_role_permissions_apply_to_conversation_and_configuration_routes():
    save_conversation(
        "analyst-thread",
        workspace_id="workspace-b",
        user_id="user-analyst-b",
    )
    save_conversation(
        "viewer-thread",
        workspace_id="workspace-c",
        user_id="user-viewer-c",
    )

    analyst_rename = client.patch(
        "/api/conversations/analyst-thread",
        headers={"X-Dev-User": "analyst-b"},
        json={"title": "分析员自己的会话"},
    )
    viewer_read = client.get(
        "/api/conversations/viewer-thread",
        headers={"X-Dev-User": "viewer-c"},
    )
    viewer_rename = client.patch(
        "/api/conversations/viewer-thread",
        headers={"X-Dev-User": "viewer-c"},
        json={"title": "只读账号不能改"},
    )
    viewer_delete = client.delete(
        "/api/conversations/viewer-thread",
        headers={"X-Dev-User": "viewer-c"},
    )
    analyst_config_write = client.post(
        "/api/model-settings",
        headers={"X-Dev-User": "analyst-b"},
        json={},
    )
    admin_reaches_config_validation = client.post(
        "/api/model-settings",
        json={},
    )

    assert analyst_rename.status_code == 200
    assert viewer_read.status_code == 200
    assert viewer_rename.status_code == 403
    assert viewer_delete.status_code == 403
    assert analyst_config_write.status_code == 403
    assert admin_reaches_config_validation.status_code == 422
