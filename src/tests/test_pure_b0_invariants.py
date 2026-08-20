import hashlib
import json
from pathlib import Path

from graph.data_agent_graph import studio_graph
from graph.text2sql_state import Text2SQLState
from tools.tool_registry import TOOLS


ROOT = Path(__file__).resolve().parents[1]

FROZEN_HASHES = {
    "prompts/system_prompt.md": "05e12e9de9d376f5785c41bb012e2bebe06d0b7648c6968d14d3c7fa5b5d5735",
    "prompts/prompt_loader.py": "25e0ef21b89d5d2b5a13e309ab037964d263a365daa8b7974dc872b2789950cc",
    "knowledge_runtime/knowledge_view.py": "7e85965acb66e2c56d1a3efab7b21136414eb6df5c66f3350bcf72ce8748dbcf",
    "graph/data_agent_graph.py": "b872dc635ca9e45fcd50b009ec4655ea2e36b8dbea4ae1298fe376763605bff4",
    "graph/text2sql_state.py": "1ea29c4aa9f58a34d956693c835eddcf67b8d10f6813f97fdaf5318cca76a545",
}


def test_prompt_navigation_graph_and_state_are_byte_identical_to_pure_b0():
    for relative_path, expected_hash in FROZEN_HASHES.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash, relative_path


def test_tool_schemas_are_identical_to_pure_b0():
    schemas = {
        tool.name: tool.get_input_schema().model_json_schema()
        for tool in TOOLS
    }
    digest = hashlib.sha256(
        json.dumps(schemas, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "cc5878f0bb2773ea6aa8b61b02c6d15740792db7d934d78aecd88713bf9d53f1"


def test_graph_topology_and_state_have_no_audit_or_remaining_steps():
    node_names = set(studio_graph.get_graph().nodes)
    assert node_names == {
        "__start__",
        "Main Agent LLM",
        "Tool Safety",
        "Tool Execution",
        "__end__",
    }
    assert set(Text2SQLState.__annotations__) == {"messages"}
