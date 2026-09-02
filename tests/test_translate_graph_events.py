import unittest

from langchain_core.messages import AIMessage

from graph.nodes.tool_safety_node import tool_safety_node
from runtime.translate_graph_events import translate_knowledge_trace_events


class KnowledgeTraceEventTests(unittest.TestCase):
    def test_allowed_read_opens_graph_with_requested_ids(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "read_knowledge",
                    "args": {
                        "knowledge_ids": [
                            "table.example.orders",
                            "column.example.orders.amount",
                        ]
                    },
                    "id": "call-1",
                }
            ],
            response_metadata={
                "knowledge_view": {
                    "knowledge_view_mode": "SUBGLOBAL",
                    "subglobal_knowledge_ids": ["table.example.customers"],
                }
            },
        )
        safe_message = tool_safety_node({"messages": [message]})["messages"][0]

        events = translate_knowledge_trace_events(
            {"data": {"Tool Safety": {"messages": [safe_message]}}}
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "open")
        self.assertEqual(events[0]["stage"], "reading")
        self.assertEqual(events[0]["mode"], "SUBGLOBAL")
        self.assertEqual(
            events[0]["active_ids"],
            [
                "table.example.customers",
                "table.example.orders",
                "column.example.orders.amount",
            ],
        )

    def test_denied_knowledge_call_does_not_open_graph(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "read_knowledge",
                    "args": {"knowledge_ids": []},
                    "id": "call-1",
                }
            ],
        )
        safe_message = tool_safety_node({"messages": [message]})["messages"][0]

        events = translate_knowledge_trace_events(
            {"data": {"Tool Safety": {"messages": [safe_message]}}}
        )

        self.assertEqual(events, [])

    def test_sql_decision_closes_knowledge_graph(self):
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_readonly_sql",
                    "args": {"sql": "SELECT 1"},
                    "id": "call-2",
                }
            ],
        )

        events = translate_knowledge_trace_events(
            {"data": {"Main Agent LLM": {"messages": [message]}}}
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "close")

    def test_another_knowledge_decision_keeps_graph_open(self):
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "revenue"},
                    "id": "call-2",
                }
            ],
        )

        events = translate_knowledge_trace_events(
            {"data": {"Main Agent LLM": {"messages": [message]}}}
        )

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
