import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
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

    def test_single_round_graph_keeps_messages_across_explicit_runtime_loop(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "rounds.sqlite"
            connection = sqlite3.connect(database_path, check_same_thread=False)
            try:
                def llm_node(state: MemoryState):
                    last = state["messages"][-1]
                    if isinstance(last, HumanMessage) and last.content == "进度呢":
                        return {"messages": [AIMessage(content="进度回答")]}
                    tool_round = sum(
                        isinstance(message, AIMessage) and bool(message.tool_calls)
                        for message in state["messages"]
                    )
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "demo_tool",
                                        "args": {},
                                        "id": f"call-{tool_round + 1}",
                                    }
                                ],
                            )
                        ]
                    }

                def tool_node(state: MemoryState):
                    call = state["messages"][-1].tool_calls[0]
                    return {
                        "messages": [
                            ToolMessage(
                                content="工具完成",
                                name="demo_tool",
                                tool_call_id=call["id"],
                            )
                        ]
                    }

                builder = StateGraph(MemoryState)
                builder.add_node("llm", llm_node)
                builder.add_node("tool", tool_node)
                builder.add_edge(START, "llm")
                builder.add_conditional_edges(
                    "llm",
                    lambda state: (
                        "tool" if state["messages"][-1].tool_calls else END
                    ),
                    {"tool": "tool", END: END},
                )
                builder.add_edge("tool", END)
                graph = builder.compile(checkpointer=SqliteSaver(connection))
                config = {"configurable": {"thread_id": "runtime-loop"}}

                graph.invoke(
                    {"messages": [HumanMessage(content="复杂任务")]},
                    config=config,
                )
                graph.invoke({}, config=config)
                graph.update_state(
                    config,
                    {"messages": [AIMessage(content="暂停总结")]},
                )
                result = graph.invoke(
                    {"messages": [HumanMessage(content="进度呢")]},
                    config=config,
                )

                self.assertEqual(
                    [type(message).__name__ for message in result["messages"]],
                    [
                        "HumanMessage",
                        "AIMessage",
                        "ToolMessage",
                        "AIMessage",
                        "ToolMessage",
                        "AIMessage",
                        "HumanMessage",
                        "AIMessage",
                    ],
                )
                self.assertEqual(result["messages"][-1].content, "进度回答")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
