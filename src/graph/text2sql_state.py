"""Shared state contract for the Text-to-SQL graph."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Text2SQLState(TypedDict):
    """Conversation and execution messages for the current Agent."""

    messages: Annotated[list[AnyMessage], add_messages]
