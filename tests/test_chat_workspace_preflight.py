import sqlite3

from fastapi.testclient import TestClient

from api.app import app
from memory import conversation_history_database, workspace_database
from runtime import agent_runtime
from security.workspace_access import resolve_current_user


client = TestClient(app)


def mark_workspace_resources_ready(workspace_id: str) -> None:
    connection = sqlite3.connect(workspace_database.CHAT_HISTORY_DATABASE_PATH)
    connection.execute(
        """
        UPDATE workspaces
        SET resources_ready = 1
        WHERE workspace_id = ?
        """,
        (workspace_id,),
    )
    connection.commit()
    connection.close()


def test_checkpoint_thread_ids_keep_legacy_a_and_namespace_other_workspaces():
    request = agent_runtime.ChatRequest(
        question="测试私有 Checkpoint",
        thread_id="same-visible-thread",
    )

    visible_a, config_a = agent_runtime.build_agent_config(
        request,
        resolve_current_user("admin-a"),
    )
    visible_b, config_b = agent_runtime.build_agent_config(
        request,
        resolve_current_user("analyst-b"),
    )
    visible_c, config_c = agent_runtime.build_agent_config(
        request,
        resolve_current_user("viewer-c"),
    )

    assert visible_a == visible_b == visible_c == "same-visible-thread"
    assert config_a["configurable"]["thread_id"] == "same-visible-thread"
    assert (
        config_b["configurable"]["thread_id"]
        == "workspace-b--same-visible-thread"
    )
    assert (
        config_c["configurable"]["thread_id"]
        == "workspace-c--same-visible-thread"
    )
    assert config_b["configurable"]["conversation_thread_id"] == (
        "same-visible-thread"
    )


def test_delete_conversation_removes_its_private_checkpoint():
    from memory.conversation_checkpointer import CHECKPOINTER

    thread_id = "delete-private-checkpoint"
    private_thread_id = f"workspace-b--{thread_id}"
    conversation_history_database.save_user_message(
        thread_id,
        "稍后删除",
        workspace_id="workspace-b",
        created_by_user_id="user-analyst-b",
    )
    CHECKPOINTER.setup()
    CHECKPOINTER.conn.execute(
        """
        INSERT INTO checkpoints(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            type,
            checkpoint,
            metadata
        )
        VALUES (?, '', 'checkpoint-1', NULL, 'json', ?, ?)
        """,
        (private_thread_id, b"{}", b"{}"),
    )
    CHECKPOINTER.conn.commit()

    response = client.delete(
        f"/api/conversations/{thread_id}",
        headers={"X-Dev-User": "analyst-b"},
    )

    remaining_checkpoints = CHECKPOINTER.conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
        (private_thread_id,),
    ).fetchone()[0]
    assert response.status_code == 200
    assert remaining_checkpoints == 0
    assert (
        conversation_history_database.read_conversation_info(
            thread_id,
            "workspace-b",
        )
        is None
    )


def test_cross_workspace_thread_is_rejected_before_model_or_runtime(monkeypatch):
    conversation_history_database.save_user_message(
        "private-thread-a",
        "A 的问题",
        workspace_id="workspace-a",
        created_by_user_id="user-admin-a",
    )
    mark_workspace_resources_ready("workspace-b")
    calls = {"model_check": 0, "runtime": 0}

    def forbidden_model_check():
        calls["model_check"] += 1
        raise AssertionError("跨空间请求不应进入模型配置检查")

    def forbidden_runtime(_request, _current_user):
        calls["runtime"] += 1
        raise AssertionError("跨空间请求不应进入 Runtime")

    monkeypatch.setattr(
        agent_runtime,
        "model_api_key_is_configured",
        forbidden_model_check,
    )
    monkeypatch.setattr(agent_runtime, "run_agent", forbidden_runtime)

    response = client.post(
        "/api/chat",
        headers={"X-Dev-User": "analyst-b"},
        json={
            "question": "尝试继续 A 的会话",
            "thread_id": "private-thread-a",
            "model": "deepseek-v4-pro",
        },
    )

    assert response.status_code == 404
    assert calls == {"model_check": 0, "runtime": 0}


def test_viewer_is_rejected_before_model_or_runtime(monkeypatch):
    calls = {"model_check": 0, "runtime": 0}

    def forbidden_model_check():
        calls["model_check"] += 1
        raise AssertionError("只读账号不应进入模型配置检查")

    def forbidden_runtime(_request, _current_user):
        calls["runtime"] += 1
        raise AssertionError("只读账号不应进入 Runtime")

    monkeypatch.setattr(
        agent_runtime,
        "model_api_key_is_configured",
        forbidden_model_check,
    )
    monkeypatch.setattr(agent_runtime, "run_agent", forbidden_runtime)

    response = client.post(
        "/api/chat",
        headers={"X-Dev-User": "viewer-c"},
        json={"question": "只读账号尝试运行 Agent"},
    )

    assert response.status_code == 403
    assert calls == {"model_check": 0, "runtime": 0}


def test_authorized_chat_passes_server_selected_workspace_to_fake_runtime(
    monkeypatch,
):
    captured = {}

    def fake_run_agent(request, current_user):
        captured["question"] = request.question
        captured["login_id"] = current_user.login_id
        captured["workspace_id"] = current_user.workspace_id
        captured["role"] = current_user.role
        return {
            "status": "success",
            "answer": "fake answer",
            "workspace_id": current_user.workspace_id,
        }

    monkeypatch.setattr(
        agent_runtime,
        "model_api_key_is_configured",
        lambda: True,
    )
    monkeypatch.setattr(agent_runtime, "run_agent", fake_run_agent)

    response = client.post(
        "/api/chat",
        json={
            "question": "不调用真实 LLM",
            "model": "deepseek-v4-pro",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "answer": "fake answer",
        "workspace_id": "workspace-a",
    }
    assert captured == {
        "question": "不调用真实 LLM",
        "login_id": "admin-a",
        "workspace_id": "workspace-a",
        "role": "admin",
    }
