import asyncio
import json
import importlib
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from graph.data_agent_graph import studio_graph
from graph.nodes.main_agent_llm_node import _invalid_tool_json_fallback
from graph.nodes.tool_execution_node import tool_execution_node
from graph.nodes.tool_safety_node import tool_safety_node
from knowledge_runtime.current_knowledge import KNOWLEDGE_CARDS
from tools.tool_safety import (
    ALLOW,
    DENY,
    INVALID_ARGUMENTS,
    POLICY_DENIED,
    UNKNOWN_TOOL,
    StandardToolCall,
    check_tool_calls,
)
from tools.tool_registry import TOOLS_BY_NAME


class ToolSafetyBoundaryTest(unittest.TestCase):
    def check(self, *tool_calls):
        return check_tool_calls(
            tool_calls,
            registered_tools=TOOLS_BY_NAME,
        )

    def test_registered_tool_and_valid_arguments_are_allowed(self):
        decision = self.check(
            StandardToolCall("search_knowledge", {"query": "revenue"}, "call-1")
        )[0]
        self.assertEqual(decision.decision, ALLOW)

    def test_unknown_tool_and_invalid_arguments_are_denied(self):
        decisions = self.check(
            StandardToolCall("read_local_file", {"path": ".env"}, "call-1"),
            StandardToolCall("search_knowledge", {}, "call-2"),
        )
        self.assertEqual([item.decision for item in decisions], [DENY, DENY])
        self.assertEqual(decisions[0].error_code, UNKNOWN_TOOL)
        self.assertEqual(decisions[1].error_code, INVALID_ARGUMENTS)

    def test_empty_and_oversized_business_arguments_are_denied(self):
        decisions = self.check(
            StandardToolCall("search_knowledge", {"query": "   "}, "call-1"),
            StandardToolCall("read_knowledge", {"knowledge_ids": []}, "call-2"),
            StandardToolCall(
                "read_knowledge",
                {"knowledge_ids": [f"card-{index}" for index in range(101)]},
                "call-3",
            ),
        )
        self.assertEqual(
            [item.error_code for item in decisions],
            [INVALID_ARGUMENTS, INVALID_ARGUMENTS, INVALID_ARGUMENTS],
        )

    def test_knowledge_path_boundary_is_enforced(self):
        valid_id = next(iter(KNOWLEDGE_CARDS))
        decisions = self.check(
            StandardToolCall("browse_knowledge", {"directory_path": "/../"}, "call-1"),
            StandardToolCall("read_knowledge", {"knowledge_ids": [valid_id]}, "call-2"),
        )
        self.assertEqual(
            [item.decision for item in decisions],
            [DENY, ALLOW],
        )
        self.assertEqual(decisions[0].error_code, POLICY_DENIED)

    def test_invalid_knowledge_id_is_a_tool_error_not_a_safety_denial(self):
        decision = self.check(
            StandardToolCall(
                "read_knowledge",
                {"knowledge_ids": ["metric.not_in_this_catalog"]},
                "call-1",
            )
        )[0]
        self.assertEqual(decision.decision, ALLOW)

    def test_missing_catalog_path_is_a_tool_error_not_a_safety_denial(self):
        decision = self.check(
            StandardToolCall(
                "browse_knowledge",
                {"directory_path": "/not-present"},
                "call-1",
            )
        )[0]
        self.assertEqual(decision.decision, ALLOW)

    def test_sql_write_and_external_access_are_denied(self):
        decisions = self.check(
            StandardToolCall(
                "execute_readonly_sql",
                {"sql": "DELETE FROM shipments"},
                "call-1",
            ),
            StandardToolCall(
                "execute_readonly_sql",
                {"sql": "SELECT * FROM read_csv_auto('secret.csv')"},
                "call-2",
            ),
            StandardToolCall(
                "execute_readonly_sql",
                {"sql": "SELECT count(*) FROM shipments"},
                "call-3",
            ),
        )
        self.assertEqual(
            [item.decision for item in decisions],
            [DENY, DENY, ALLOW],
        )
        self.assertEqual(
            [item.error_code for item in decisions],
            [POLICY_DENIED, POLICY_DENIED, None],
        )

    def test_batch_decisions_keep_each_tool_call_id(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {"name": "search_knowledge", "args": {"query": "revenue"}, "id": "call-1"},
                {"name": "unknown_tool", "args": {}, "id": "call-2"},
            ],
        )
        update = tool_safety_node({"messages": [message]})
        decisions = update["messages"][0].additional_kwargs["tool_safety_decisions"]

        self.assertEqual(
            [(item["tool_call_id"], item["decision"]) for item in decisions],
            [("call-1", ALLOW), ("call-2", DENY)],
        )

    def test_denied_call_is_not_executed(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "execute_readonly_sql",
                    "args": {"sql": "DROP TABLE shipments"},
                    "id": "call-1",
                }
            ],
        )
        safety_update = tool_safety_node({"messages": [message]})

        class MustNotExecute:
            async def ainvoke(self, arguments):
                raise AssertionError("denied tool must not execute")

        real_tool = TOOLS_BY_NAME["execute_readonly_sql"]
        try:
            TOOLS_BY_NAME["execute_readonly_sql"] = MustNotExecute()
            execution_update = asyncio.run(
                tool_execution_node({"messages": safety_update["messages"]})
            )
        finally:
            TOOLS_BY_NAME["execute_readonly_sql"] = real_tool

        result = json.loads(execution_update["messages"][0].content)
        self.assertEqual(result["error_type"], "POLICY_DENIED")

    def test_allowed_call_is_executed(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "revenue"},
                    "id": "call-1",
                }
            ],
        )
        safety_update = tool_safety_node({"messages": [message]})
        execution_update = asyncio.run(
            tool_execution_node({"messages": safety_update["messages"]})
        )

        result = json.loads(execution_update["messages"][0].content)
        self.assertEqual(result["query"], "revenue")
        self.assertIn("results", result)

    def test_execution_exception_is_normalized_and_keeps_tool_call_id(self):
        message = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "revenue"},
                    "id": "call-1",
                }
            ],
        )
        safety_update = tool_safety_node({"messages": [message]})

        class BrokenTool:
            async def ainvoke(self, arguments):
                raise RuntimeError("database unavailable; password=do-not-return")

        real_tool = TOOLS_BY_NAME["search_knowledge"]
        try:
            TOOLS_BY_NAME["search_knowledge"] = BrokenTool()
            execution_update = asyncio.run(
                tool_execution_node({"messages": safety_update["messages"]})
            )
        finally:
            TOOLS_BY_NAME["search_knowledge"] = real_tool

        tool_message = execution_update["messages"][0]
        result = json.loads(tool_message.content)
        self.assertEqual(tool_message.tool_call_id, "call-1")
        self.assertEqual(result["error_type"], "EXECUTION_ERROR")
        self.assertNotIn("do-not-return", result["details"])

    def test_sql_timeout_has_a_stable_error_contract(self):
        sql_tool_module = importlib.import_module("tools.execute_readonly_sql")

        class TimeoutErrorForTest(Exception):
            pass

        def fail_with_timeout(_sql):
            raise TimeoutErrorForTest("too slow")

        with patch.object(
            sql_tool_module,
            "_configured_executor",
            return_value=(fail_with_timeout, ValueError, TimeoutErrorForTest),
        ):
            result = json.loads(
                sql_tool_module.execute_readonly_sql.invoke({"sql": "SELECT 1"})
            )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_type"], "TIMEOUT")
        self.assertFalse(result["retryable"])

    def test_sql_safety_uses_the_current_database_backend(self):
        call = StandardToolCall(
            tool_name="execute_readonly_sql",
            arguments={"sql": "SELECT * FROM another_database.records"},
            tool_call_id="sql-backend",
        )
        with patch.dict(
            "os.environ",
            {
                "DATA_AGENT_DATABASE_BACKEND": "mysql",
                "DATA_AGENT_MYSQL_DATABASE": "configured_database",
            },
        ):
            decision = check_tool_calls([call], registered_tools=TOOLS_BY_NAME)[0]

        self.assertEqual(decision.decision, DENY)
        self.assertEqual(decision.error_code, POLICY_DENIED)

    def test_invalid_tool_json_becomes_a_terminal_non_tool_response(self):
        malformed = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": "{not-json",
                    "id": "call-bad",
                    "error": "invalid json",
                    "type": "invalid_tool_call",
                }
            ],
        )
        fallback = _invalid_tool_json_fallback(malformed)

        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.tool_calls, [])
        self.assertEqual(fallback.response_metadata["error_code"], "INVALID_JSON")
        self.assertTrue(fallback.response_metadata["graceful"])

    def test_invalid_tool_json_ends_graph_without_tool_execution(self):
        class MalformedToolModel:
            async def ainvoke(self, messages):
                return AIMessage(
                    content="",
                    invalid_tool_calls=[
                        {
                            "name": "execute_readonly_sql",
                            "args": "{bad-json",
                            "id": "call-bad",
                            "error": "invalid json",
                            "type": "invalid_tool_call",
                        }
                    ],
                )

        with patch(
            "graph.nodes.main_agent_llm_node.MAIN_LLM_WITH_TOOLS",
            MalformedToolModel(),
        ):
            result = asyncio.run(
                studio_graph.ainvoke(
                    {"messages": [HumanMessage(content="Question")]},
                    config={"recursion_limit": 8},
                )
            )

        final = result["messages"][-1]
        self.assertEqual(final.tool_calls, [])
        self.assertEqual(final.response_metadata["error_code"], "INVALID_JSON")


if __name__ == "__main__":
    unittest.main()
