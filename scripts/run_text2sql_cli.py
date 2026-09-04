"""命令行运行入口；相同 thread_id 会恢复此前的 messages。"""

import argparse
import asyncio

from langchain_core.messages import HumanMessage

from graph.data_agent_graph import create_conversation_graph
from memory.conversation_checkpointer import open_conversation_checkpoint_database


def print_update(node_name: str, update: dict) -> str:
    print(f"\n--- 节点完成：{node_name} ---")
    final_answer = ""
    for message in update.get("messages", []):
        print(f"{type(message).__name__}: {message.content}")
        if getattr(message, "tool_calls", None):
            print("tool_calls:", message.tool_calls)
        elif type(message).__name__ == "AIMessage" and isinstance(message.content, str):
            final_answer = message.content
    return final_answer


async def run_one_turn(graph, user_text: str, config: dict):
    initial_state = {"messages": [HumanMessage(content=user_text)]}

    async for graph_update in graph.astream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in graph_update.items():
            print_update(node_name, update)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default="demo")
    parser.add_argument("--message")
    args = parser.parse_args()

    config = {
        "configurable": {
            "thread_id": args.thread_id,
        }
    }

    database_connection, checkpointer = (
        await open_conversation_checkpoint_database()
    )
    graph = create_conversation_graph(checkpointer)
    try:
        if args.message:
            await run_one_turn(graph, args.message, config)
        else:
            print(f"当前会话：{args.thread_id}（输入‘退出’结束）")
            while True:
                user_text = (await asyncio.to_thread(input, "你：")).strip()
                if user_text == "退出":
                    break
                if user_text:
                    await run_one_turn(graph, user_text, config)
    finally:
        await database_connection.close()


if __name__ == "__main__":
    asyncio.run(main())
