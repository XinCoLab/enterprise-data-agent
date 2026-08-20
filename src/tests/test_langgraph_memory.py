import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MemoryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def reply_node(state: MemoryState):
    return {"messages": [AIMessage(content="ok")]}


class LangGraphMemoryTest(unittest.TestCase):
    def test_same_thread_restores_previous_messages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "memory.sqlite"
            connection = sqlite3.connect(database_path, check_same_thread=False)
            try:
                builder = StateGraph(MemoryState)
                builder.add_node("reply", reply_node)
                builder.add_edge(START, "reply")
                builder.add_edge("reply", END)
                graph = builder.compile(checkpointer=SqliteSaver(connection))

                config = {"configurable": {"thread_id": "same-thread"}}
                graph.invoke(
                    {"messages": [HumanMessage(content="第一轮")]},
                    config=config,
                )
                result = graph.invoke(
                    {"messages": [HumanMessage(content="第二轮")]},
                    config=config,
                )

                self.assertEqual(
                    [message.content for message in result["messages"]],
                    ["第一轮", "ok", "第二轮", "ok"],
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
