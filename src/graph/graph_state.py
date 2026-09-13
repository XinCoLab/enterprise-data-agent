"""Mutable data passed between nodes in the Agent graph."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class CompactionState(TypedDict):
    """最近一次成功压缩；边界 ID 指向仍保留原文的第一条消息。"""

    summary: str
    first_kept_message_id: str
    tokens_before: int
    tokens_after: int


class GraphState(TypedDict):
    """Conversation and execution messages for the current Agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    # 原始 messages 不删除。摘要和边界与它们一起写入现有 SQLite checkpoint。
    compaction: NotRequired[CompactionState]
    retrieved_memories: NotRequired[list[dict]]
