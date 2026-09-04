"""The current Data Agent graph."""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from graph.nodes.main_agent_llm_node import main_agent_llm_node
from graph.nodes.tool_execution_node import tool_execution_node
from graph.nodes.tool_safety_node import tool_safety_node
from graph.graph_state import GraphState



def build_data_agent_graph(*, loop_after_tools: bool) -> StateGraph:
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("Main Agent LLM", main_agent_llm_node)
    graph_builder.add_node("Tool Safety", tool_safety_node)
    graph_builder.add_node("Tool Execution", tool_execution_node)

    graph_builder.add_edge(START, "Main Agent LLM")
    graph_builder.add_conditional_edges(
        "Main Agent LLM",
        tools_condition,
        {
            "tools": "Tool Safety",
            END: END,
        },
    )
    graph_builder.add_edge("Tool Safety", "Tool Execution")
    graph_builder.add_edge(
        "Tool Execution",
        "Main Agent LLM" if loop_after_tools else END,
    )
    return graph_builder


def create_conversation_graph(checkpointer):
    graph_builder = build_data_agent_graph(loop_after_tools=True)
    return graph_builder.compile(checkpointer=checkpointer)

# LangGraph 的一次 stream()/invoke() 会沿着图中的边持续执行，直到遇到 END。
# 如果 Tool Execution 连接回 Main Agent LLM，LangGraph 会在一次调用内部继续循环，
# Runtime 无法在两轮之间检查取消请求、循环上限和工具消息完整性。
# 因此使用自定义参数 loop_after_tools 控制工具节点之后的边：
# True 返回 LLM，由 LangGraph 内部循环；False 连接 END，把控制权交还 Runtime。
def create_single_round_graph(checkpointer):
    graph_builder = build_data_agent_graph(loop_after_tools=False)
    return graph_builder.compile(checkpointer=checkpointer)


studio_graph = build_data_agent_graph(loop_after_tools=True).compile()
