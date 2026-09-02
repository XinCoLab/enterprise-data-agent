"""Conversation history HTTP endpoints."""

from fastapi import APIRouter, HTTPException

from api.schemas import ConversationRenameRequest
from memory import conversation_history_database as conversation_history


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def list_conversation_rows():
    return {
        "conversations": conversation_history.list_conversation_rows(),
    }


@router.get("/{thread_id}")
def read_conversation_info(thread_id: str):
    conversation = conversation_history.read_conversation_info(thread_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return conversation


@router.patch("/{thread_id}")
def rename_conversation_title(
    thread_id: str,
    request: ConversationRenameRequest,
):
    renamed = conversation_history.rename_conversation(
        thread_id,
        request.title,
    )
    if not renamed:
        raise HTTPException(status_code=404, detail="会话不存在。")

    return {
        "status": "success",
        "message": "会话已重命名。",
        "conversation": conversation_history.read_conversation_info(thread_id),
    }


@router.delete("/{thread_id}")
def delete_conversation(thread_id: str):
    deleted = conversation_history.delete_conversation(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在。")

    return {
        "status": "success",
        "message": "会话已删除。",
        "thread_id": thread_id,
    }
