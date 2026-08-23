"""The current Data Agent graph."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from graph.nodes.main_agent_llm_node import main_agent_llm_node
from graph.nodes.tool_execution_node import tool_execution_node
from graph.nodes.tool_safety_node import tool_safety_node
from graph.text2sql_state import Text2SQLState
from memory.conversation_checkpointer import CHECKPOINTER


def _build_graph(*, loop_after_tools: bool) -> StateGraph:
    builder = StateGraph(Text2SQLState)
    builder.add_node("Main Agent LLM", main_agent_llm_node)
    builder.add_node("Tool Safety", tool_safety_node)
    builder.add_node("Tool Execution", tool_execution_node)

    builder.add_edge(START, "Main Agent LLM")
    builder.add_conditional_edges(
        "Main Agent LLM",
        tools_condition,
        {
            "tools": "Tool Safety",
            END: END,
        },
    )
    builder.add_edge("Tool Safety", "Tool Execution")
    builder.add_edge(
        "Tool Execution",
        "Main Agent LLM" if loop_after_tools else END,
    )
    return builder


builder = _build_graph(loop_after_tools=True)
studio_graph = builder.compile()
graph = builder.compile(checkpointer=CHECKPOINTER)

# The Web Runtime calls this graph once per explicit Agent recursion. One call
# always ends after either a final Assistant answer or a complete Tool round.
round_graph = _build_graph(loop_after_tools=False).compile(checkpointer=CHECKPOINTER)
