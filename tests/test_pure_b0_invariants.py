import hashlib
import json
from pathlib import Path

from graph.round_graph import round_graph, studio_graph
from graph.text2sql_state import Text2SQLState
from tools.tool_registry import TOOLS


ROOT = Path(__file__).resolve().parents[1] / "src"

FROZEN_SYSTEM_PROMPT_HASH = (
    "05e12e9de9d376f5785c41bb012e2bebe06d0b7648c6968d14d3c7fa5b5d5735"
)


def test_system_prompt_is_byte_identical_to_pure_b0():
    """Prompt文字会直接影响模型行为，因此保留精确的版本指纹。"""

    prompt_path = ROOT / "prompts" / "system_prompt.md"
    actual_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    assert actual_hash == FROZEN_SYSTEM_PROMPT_HASH


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
    studio_topology = studio_graph.get_graph()
    node_names = set(studio_topology.nodes)
    assert node_names == {
        "__start__",
        "Main Agent LLM",
        "Tool Safety",
        "Tool Execution",
        "__end__",
    }
    assert set(Text2SQLState.__annotations__) == {"messages"}

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
        for edge in round_graph.get_graph().edges
    }
    assert round_edges == {
        ("__start__", "Main Agent LLM", None, False),
        ("Main Agent LLM", "Tool Safety", "tools", True),
        ("Main Agent LLM", "__end__", None, True),
        ("Tool Safety", "Tool Execution", None, False),
        ("Tool Execution", "__end__", None, False),
    }
