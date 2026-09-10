"""Workspace-scoped conversation history endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from api.schemas import ConversationRenameRequest
from memory import conversation_history_database as conversation_history
from agent_runtime import agent_runtime
from security.workspace_access import (
    CurrentUser,
    read_current_user,
    require_permission,
)


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def require_conversation_change(
    current_user: CurrentUser,
    conversation: dict,
) -> None:
    require_permission(current_user, "conversation:write")
    if (
        current_user.role != "admin"
        and conversation["created_by_user_id"] != current_user.user_id
    ):
        raise HTTPException(
            status_code=403,
            detail="只能修改自己创建的会话。",
        )


def with_continuation_status(conversation: dict, current_user: CurrentUser) -> dict:
    with agent_runtime.RESOURCE_CONFIG_LOCK:
        active_binding = agent_runtime.read_current_conversation_binding(current_user)
    return {
        **conversation,
        "can_continue": conversation_history.bindings_match(
            conversation.get("binding"), active_binding,
        ),
    }


@router.get("")
def list_conversation_rows(
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
    data_source_id: str | None = None,
    legacy: bool = False,
):
    require_permission(current_user, "conversation:read")
    if legacy and data_source_id is not None:
        raise HTTPException(status_code=400, detail="不能同时选择数据源和未归属历史。")
    with agent_runtime.RESOURCE_CONFIG_LOCK:
        active_binding = agent_runtime.read_current_conversation_binding(current_user)
    all_rows = conversation_history.list_conversation_rows(current_user.workspace_id)
    source_id = data_source_id if data_source_id is not None else (
        active_binding["data_source_id"] if active_binding is not None else None
    )
    if legacy:
        selected = [row for row in all_rows if row["binding"] is None]
    elif source_id is None:
        selected = []
    else:
        selected = [
            row for row in all_rows
            if row["binding"] is not None
            and row["binding"]["data_source_id"] == source_id
        ]
    data_sources = {}
    legacy_count = 0
    for row in all_rows:
        binding = row["binding"]
        if binding is None:
            legacy_count += 1
            continue
        data_sources.setdefault(binding["data_source_id"], {
            key: binding.get(key, "")
            for key in ("data_source_id", "data_source_name", "backend", "database")
        })
    return {
        "conversations": [
            {**row, "can_continue": conversation_history.bindings_match(row["binding"], active_binding)}
            for row in selected
        ],
        "data_sources": list(data_sources.values()),
        "legacy_count": legacy_count,
        "active_binding": active_binding,
    }


@router.get("/{thread_id}")
def read_conversation_info(
    thread_id: str,
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    require_permission(current_user, "conversation:read")
    conversation = conversation_history.read_conversation_info(
        thread_id,
        current_user.workspace_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return with_continuation_status(conversation, current_user)


@router.patch("/{thread_id}")
def rename_conversation_title(
    thread_id: str,
    request: ConversationRenameRequest,
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    conversation = conversation_history.read_conversation_info(
        thread_id,
        current_user.workspace_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    require_conversation_change(current_user, conversation)

    conversation_history.rename_conversation(
        thread_id,
        request.title,
        current_user.workspace_id,
    )
    return {
        "status": "success",
        "message": "会话已重命名。",
        "conversation": with_continuation_status(
            conversation_history.read_conversation_info(
                thread_id, current_user.workspace_id,
            ),
            current_user,
        ),
    }


@router.delete("/{thread_id}")
async def delete_conversation(
    thread_id: str,
    http_request: Request,
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    conversation = await run_in_threadpool(
        conversation_history.read_conversation_info,
        thread_id,
        current_user.workspace_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    require_conversation_change(current_user, conversation)

    await agent_runtime.delete_saved_conversation(
        thread_id,
        current_user,
        http_request.app.state.checkpointer,
    )
    return {
        "status": "success",
        "message": "会话已删除。",
        "thread_id": thread_id,
    }
