"""Mutable data passed between nodes in the Agent graph."""

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Conversation and execution messages for the current Agent."""

    messages: Annotated[list[AnyMessage], add_messages]
