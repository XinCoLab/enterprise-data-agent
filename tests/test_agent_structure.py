import inspect
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from config.project_paths import AI_APP_LAB_ROOT
from graph.data_agent_graph import studio_graph
from knowledge_runtime.catalog import browse_catalog
from knowledge_runtime.current_knowledge import (
    KNOWLEDGE_CATALOG,
    KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
)
from prompts.prompt_loader import build_model_input
from tools.tool_registry import TOOLS


class GenericPromptTest(unittest.TestCase):
    def test_prompt_contains_no_profile_specific_schema_or_evaluation_terms(self):
        prompt_path = (
            AI_APP_LAB_ROOT / "src" / "prompts" / "system_prompt.md"
        )
        prompt = prompt_path.read_text(encoding="utf-8").lower()
        forbidden_terms = {
            "cold_chain_pharma_compliance",
            "q1",
            "q22",
            "shipments",
            "environmentalmonitoring",
            "reckeylink",
            "excursion_count",
            "gold sql",
            "benchmark",
        }

        self.assertEqual(
            {term for term in forbidden_terms if term in prompt},
            set(),
        )

    def test_runtime_root_directory_is_inserted_as_a_system_message(self):
        runtime_directory = browse_catalog(KNOWLEDGE_CATALOG, "/")
        model_input = build_model_input(
            [HumanMessage(content="Count the relevant records.")],
            runtime_directory=runtime_directory,
            runtime_navigation_graph=KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
        )

        self.assertIsInstance(model_input[0], SystemMessage)
        self.assertIsInstance(model_input[1], SystemMessage)
        self.assertIn("[Runtime knowledge directory]", model_input[1].content)
        self.assertIn('"path": "/"', model_input[1].content)

        self.assertIsInstance(model_input[2], SystemMessage)
        self.assertIn(
            "[Runtime knowledge navigation graph]",
            model_input[2].content,
        )
        self.assertIn("NODES (ref|type|title|knowledge_id)", model_input[2].content)
        self.assertIn("EDGES (source_ref|relation|target_ref)", model_input[2].content)

    def test_subglobal_view_replaces_the_full_graph_system_message(self):
        runtime_directory = browse_catalog(KNOWLEDGE_CATALOG, "/")
        model_input = build_model_input(
            [HumanMessage(content="Count the relevant records.")],
            runtime_directory=runtime_directory,
            runtime_navigation_graph=KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
            knowledge_view_mode="SUBGLOBAL",
            runtime_subglobal_graph="SUBGLOBAL KNOWLEDGE GRAPH\n[READ]\nN001",
        )
        system_text = "\n".join(
            str(message.content)
            for message in model_input
            if isinstance(message, SystemMessage)
        )
        self.assertIn("[Runtime subglobal knowledge graph]", system_text)
        self.assertNotIn("[Runtime knowledge navigation graph]", system_text)

    def test_reglobal_view_contains_global_and_subglobal_system_messages(self):
        runtime_directory = browse_catalog(KNOWLEDGE_CATALOG, "/")
        model_input = build_model_input(
            [HumanMessage(content="Count the relevant records.")],
            runtime_directory=runtime_directory,
            runtime_navigation_graph=KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
            knowledge_view_mode="REGLOBAL",
            runtime_subglobal_graph="SUBGLOBAL KNOWLEDGE GRAPH\n[READ]\nN001",
        )
        system_text = "\n".join(
            str(message.content)
            for message in model_input
            if isinstance(message, SystemMessage)
        )
        self.assertIn("[Runtime subglobal knowledge graph]", system_text)
        self.assertIn("[Runtime knowledge navigation graph]", system_text)


class GenericToolProfileTest(unittest.TestCase):
    def test_agent_exposes_data_and_visualization_tools(self):
        self.assertEqual(
            [tool.name for tool in TOOLS],
            [
                "browse_knowledge",
                "search_knowledge",
                "read_knowledge",
                "execute_readonly_sql",
                "create_metric_cards",
                "create_chart",
                "compose_dashboard",
                "export_report",
            ],
        )

    def test_visible_tool_descriptions_contain_no_profile_specific_names(self):
        exposed_text = "\n".join(
            "\n".join(
                [
                    tool.name,
                    tool.description,
                    str(tool.args),
                ]
            )
            for tool in TOOLS
        ).lower()

        forbidden_terms = {
            "cold_chain_pharma_compliance",
            "shipments",
            "environmentalmonitoring",
            "reckeylink",
            "excursion_count",
        }
        self.assertEqual(
            {term for term in forbidden_terms if term in exposed_text},
            set(),
        )

    def test_each_llm_tool_name_matches_its_source_filename(self):
        for registered_tool in TOOLS:
            module_name = inspect.getmodule(registered_tool.func).__name__
            self.assertEqual(module_name.rsplit(".", 1)[-1], registered_tool.name)

    def test_tools_package_matches_the_active_registry(self):
        tool_files = {
            path.name
            for path in (AI_APP_LAB_ROOT / "src" / "tools").glob("*.py")
        }
        self.assertEqual(
            tool_files,
            {
                "__init__.py",
                "tool_registry.py",
                "tool_safety.py",
                "browse_knowledge.py",
                "search_knowledge.py",
                "read_knowledge.py",
                "execute_readonly_sql.py",
                "create_metric_cards.py",
                "create_chart.py",
                "compose_dashboard.py",
                "export_report.py",
            },
        )


class AgentProfileFilesTest(unittest.TestCase):
    def test_current_agent_has_one_root_langgraph_config(self):
        self.assertTrue((AI_APP_LAB_ROOT / "langgraph.json").exists())
        self.assertFalse((AI_APP_LAB_ROOT / "langgraph_v2.json").exists())

    def test_graph_exposes_the_tool_safety_boundary(self):
        graph = studio_graph.get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn("Tool Safety", graph.nodes)
        self.assertIn(("Main Agent LLM", "Tool Safety"), edges)
        self.assertIn(("Tool Safety", "Tool Execution"), edges)
        self.assertNotIn(("Main Agent LLM", "Tool Execution"), edges)

    def test_graph_starts_directly_at_main_agent(self):
        graph = studio_graph.get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertNotIn("Task Draft LLM", graph.nodes)
        self.assertNotIn("Task Draft Check", graph.nodes)
        self.assertIn(("__start__", "Main Agent LLM"), edges)


if __name__ == "__main__":
    unittest.main()
