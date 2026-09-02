"""Workspace-scoped conversation history endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import ConversationRenameRequest
from memory import conversation_history_database as conversation_history
from runtime import agent_runtime
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


@router.get("")
def list_conversation_rows(
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    require_permission(current_user, "conversation:read")
    return {
        "conversations": conversation_history.list_conversation_rows(
            current_user.workspace_id
        ),
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
    return conversation


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
        "conversation": conversation_history.read_conversation_info(
            thread_id,
            current_user.workspace_id,
        ),
    }


@router.delete("/{thread_id}")
def delete_conversation(
    thread_id: str,
    current_user: Annotated[CurrentUser, Depends(read_current_user)],
):
    conversation = conversation_history.read_conversation_info(
        thread_id,
        current_user.workspace_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    require_conversation_change(current_user, conversation)

    agent_runtime.delete_saved_agent_state(thread_id, current_user)
    conversation_history.delete_conversation(
        thread_id,
        current_user.workspace_id,
    )
    return {
        "status": "success",
        "message": "会话已删除。",
        "thread_id": thread_id,
    }
