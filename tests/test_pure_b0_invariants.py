from graph.data_agent_graph import create_single_round_graph, studio_graph
from graph.graph_state import GraphState

def test_graph_topology_and_state_have_no_audit_or_remaining_steps():
    studio_topology = studio_graph.get_graph()
    single_round_graph = create_single_round_graph(None)
    node_names = set(studio_topology.nodes)
    assert node_names == {
        "__start__",
        "Main Agent LLM",
        "Tool Safety",
        "Tool Execution",
        "__end__",
    }
    assert set(GraphState.__annotations__) == {"messages"}

    # Studio 图在工具执行完成后回到 Main Agent，形成图内循环。
    studio_edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in studio_topology.edges
    }
    assert studio_edges == {
        ("__start__", "Main Agent LLM", None, False),
        ("Main Agent LLM", "Tool Safety", "tools", True),
        ("Main Agent LLM", "__end__", None, True),
        ("Tool Safety", "Tool Execution", None, False),
        ("Tool Execution", "Main Agent LLM", None, False),
    }

    # Web Runtime 每次只执行一个完整工具轮次，因此工具执行后结束本轮。
    round_edges = {
        (edge.source, edge.target, edge.data, edge.conditional)
        for edge in single_round_graph.get_graph().edges
    }
    assert round_edges == {
        ("__start__", "Main Agent LLM", None, False),
        ("Main Agent LLM", "Tool Safety", "tools", True),
        ("Main Agent LLM", "__end__", None, True),
        ("Tool Safety", "Tool Execution", None, False),
        ("Tool Execution", "__end__", None, False),
    }
