import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from knowledge_runtime.knowledge_view import (
    build_subglobal_knowledge_graph,
    render_subglobal_knowledge_graph,
    select_knowledge_view,
    successful_read_knowledge_ids,
)


GLOBAL_GRAPH = {
    "nodes": [
        {"ref": "N001", "knowledge_id": "card.a", "knowledge_type": "metric", "title": "A"},
        {"ref": "N002", "knowledge_id": "card.b", "knowledge_type": "column", "title": "B"},
        {"ref": "N003", "knowledge_id": "card.c", "knowledge_type": "table", "title": "C"},
        {"ref": "N004", "knowledge_id": "card.d", "knowledge_type": "column", "title": "D"},
    ],
    "edges": [
        {"source": "N001", "relation": "related_to", "target": "N002"},
        {"source": "N003", "relation": "sourced_from", "target": "N001"},
        {"source": "N002", "relation": "belongs_to", "target": "N004"},
    ],
}


def _tool_call(name: str, call_id: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _successful_read(call_id: str, *knowledge_ids: str) -> ToolMessage:
    return ToolMessage(
        name="read_knowledge",
        tool_call_id=call_id,
        content=json.dumps(
            {"cards": [{"knowledge_id": knowledge_id} for knowledge_id in knowledge_ids]}
        ),
    )


def _discovery_result(name: str, call_id: str) -> ToolMessage:
    return ToolMessage(
        name=name,
        tool_call_id=call_id,
        content=json.dumps({"results": []}),
    )


class KnowledgeViewSelectionTest(unittest.TestCase):
    def test_first_main_agent_call_uses_global(self):
        state = {"messages": [HumanMessage(content="Question")]}
        self.assertEqual(select_knowledge_view(state), "GLOBAL")

    def test_successful_read_switches_to_subglobal(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "SUBGLOBAL")
        self.assertEqual(successful_read_knowledge_ids(state), ["card.a"])

    def test_search_after_subglobal_read_switches_to_reglobal(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
                _tool_call("search_knowledge", "search-1", {"query": "missing"}),
                _discovery_result("search_knowledge", "search-1"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "REGLOBAL")

    def test_browse_after_subglobal_read_switches_to_reglobal(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
                _tool_call("browse_knowledge", "browse-1", {"directory_path": "/table"}),
                _discovery_result("browse_knowledge", "browse-1"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "REGLOBAL")

    def test_new_successful_read_after_reglobal_returns_to_subglobal(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
                _tool_call("search_knowledge", "search-1", {"query": "B"}),
                _discovery_result("search_knowledge", "search-1"),
                _tool_call("read_knowledge", "read-2", {"knowledge_ids": ["card.b"]}),
                _successful_read("read-2", "card.b"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "SUBGLOBAL")
        self.assertEqual(successful_read_knowledge_ids(state), ["card.a", "card.b"])

    def test_failed_read_does_not_create_subglobal_nodes(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["missing"]}),
                ToolMessage(
                    name="read_knowledge",
                    tool_call_id="read-1",
                    content=json.dumps({"error": "KNOWLEDGE_ID_NOT_FOUND"}),
                ),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "GLOBAL")
        self.assertEqual(successful_read_knowledge_ids(state), [])

    def test_unrequested_card_in_tool_result_does_not_become_active(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.b"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "GLOBAL")
        self.assertEqual(successful_read_knowledge_ids(state), [])

    def test_new_human_turn_resets_to_global(self):
        state = {
            "messages": [
                HumanMessage(content="Old question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
                HumanMessage(content="New question"),
            ]
        }
        self.assertEqual(select_knowledge_view(state), "GLOBAL")


class SubglobalKnowledgeGraphTest(unittest.TestCase):
    def test_read_nodes_real_edges_and_one_hop_frontier(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
            ]
        }
        subglobal = build_subglobal_knowledge_graph(state, GLOBAL_GRAPH)

        self.assertEqual(
            [node["knowledge_id"] for node in subglobal["read_nodes"]],
            ["card.a"],
        )
        self.assertEqual(
            {
                (edge["source"], edge["relation"], edge["target"])
                for edge in subglobal["relations"]
            },
            {
                ("N001", "related_to", "N002"),
                ("N003", "sourced_from", "N001"),
            },
        )
        self.assertEqual(
            {node["knowledge_id"] for node in subglobal["frontier_nodes"]},
            {"card.b", "card.c"},
        )

    def test_no_relation_is_inferred_between_frontier_nodes(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a"]}),
                _successful_read("read-1", "card.a"),
            ]
        }
        subglobal = build_subglobal_knowledge_graph(state, GLOBAL_GRAPH)
        self.assertNotIn(
            {"source": "N002", "relation": "belongs_to", "target": "N004"},
            subglobal["relations"],
        )

    def test_multiple_reads_expand_subglobal_set_and_frontier(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["card.a", "card.b"]}),
                _successful_read("read-1", "card.a", "card.b"),
            ]
        }
        subglobal = build_subglobal_knowledge_graph(state, GLOBAL_GRAPH)
        self.assertEqual(
            [node["knowledge_id"] for node in subglobal["read_nodes"]],
            ["card.a", "card.b"],
        )
        self.assertEqual(
            {node["knowledge_id"] for node in subglobal["frontier_nodes"]},
            {"card.c", "card.d"},
        )
        rendered = render_subglobal_knowledge_graph(subglobal)
        self.assertIn("[READ]", rendered)
        self.assertIn("[RELATIONS]", rendered)
        self.assertIn("[FRONTIER]", rendered)

    def test_unknown_successfully_returned_id_fails_fast(self):
        state = {
            "messages": [
                HumanMessage(content="Question"),
                _tool_call("read_knowledge", "read-1", {"knowledge_ids": ["missing"]}),
                _successful_read("read-1", "missing"),
            ]
        }
        with self.assertRaises(ValueError):
            build_subglobal_knowledge_graph(state, GLOBAL_GRAPH)


if __name__ == "__main__":
    unittest.main()
