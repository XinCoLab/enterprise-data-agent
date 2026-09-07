import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime import translate_graph_events
from graph.nodes.tool_safety_node import tool_safety_node
from agent_runtime.translate_graph_events import translate_knowledge_trace_events


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


class GraphProgressEventTests(unittest.TestCase):
    def test_updates_keep_round_knowledge_and_tool_progress_order(self):
        model_output = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_readonly_sql",
                    "args": {"sql": "SELECT 1"},
                    "id": "sql-1",
                }
            ],
        )
        tool_result = ToolMessage(
            name="execute_readonly_sql",
            tool_call_id="sql-1",
            content='{"returned_rows":1,"truncated":false}',
        )
        events = list(
            translate_graph_events.translate_graph_progress_events(
                {
                    "type": "updates",
                    "data": {
                        "Main Agent LLM": {"messages": [model_output]},
                        "Tool Execution": {"messages": [tool_result]},
                    },
                },
                round_number=2,
                request_id="request-order",
            )
        )

        self.assertEqual(
            [event["type"] for event in events],
            ["round", "knowledge_trace", "progress"],
        )
        self.assertTrue(
            all(event["request_id"] == "request-order" for event in events)
        )
        self.assertEqual(events[0]["round"], 2)
        self.assertEqual(
            events[0]["tool_calls"],
            [{"name": "execute_readonly_sql", "arguments": {"sql": "SELECT 1"}}],
        )
        self.assertEqual(events[1]["action"], "close")
        self.assertEqual(events[2]["tool"], "execute_readonly_sql")
        self.assertEqual(events[2]["message"], "SQL 执行完成，返回 1 行。")

    def test_paused_iteration_does_not_translate_later_event_categories(self):
        calls = []

        def translate_round(_part, _round_number):
            calls.append("round")
            return {"type": "round"}

        def translate_knowledge(_part):
            calls.append("knowledge")
            return [
                {"type": "knowledge_trace", "action": "open"},
                {"type": "knowledge_trace", "action": "close"},
            ]

        def translate_progress(_part):
            calls.append("progress")
            return [{"type": "progress"}]

        with (
            patch.object(
                translate_graph_events,
                "translate_llm_round_event",
                side_effect=translate_round,
            ),
            patch.object(
                translate_graph_events,
                "translate_knowledge_trace_events",
                side_effect=translate_knowledge,
            ),
            patch.object(
                translate_graph_events,
                "translate_update_progress_events",
                side_effect=translate_progress,
            ),
        ):
            events = translate_graph_events.translate_graph_progress_events(
                {"type": "updates"}, 1, "request-paused"
            )
            self.assertEqual(calls, [])
            self.assertEqual(
                next(events), {"type": "round", "request_id": "request-paused"}
            )
            self.assertEqual(calls, ["round"])
            self.assertEqual(next(events)["action"], "open")
            self.assertEqual(calls, ["round", "knowledge"])
            self.assertEqual(next(events)["action"], "close")
            self.assertEqual(calls, ["round", "knowledge"])
            self.assertEqual(
                next(events), {"type": "progress", "request_id": "request-paused"}
            )
            self.assertEqual(calls, ["round", "knowledge", "progress"])
            with self.assertRaises(StopIteration):
                next(events)

    def test_unknown_or_non_displayable_parts_emit_no_progress(self):
        for part in (
            {"type": "unknown", "data": {}},
            {"type": "tasks", "data": {"name": "Unknown Node", "input": {}}},
            {"type": "tasks", "data": {"name": "Main Agent LLM"}},
            {"type": "updates", "data": {"Main Agent LLM": {"messages": []}}},
        ):
            with self.subTest(part=part):
                self.assertEqual(
                    list(
                        translate_graph_events.translate_graph_progress_events(
                            part, 1, "request-empty"
                        )
                    ),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
