import tempfile
import unittest
from pathlib import Path

import yaml

from config.project_paths import KNOWLEDGE_ROOT
from knowledge_runtime.catalog import load_knowledge_cards
from knowledge_runtime.navigation_graph import (
    build_navigation_graph,
    render_navigation_graph,
)


def _expected_explicit_edges(card) -> set[tuple[str, str, str]]:
    """Independent inventory of the contract's explicit knowledge-ID fields."""

    source = card.knowledge_id
    content = card.content
    payload = content.get("payload", {})
    discovery = content.get("discovery", {})
    result: set[tuple[str, str, str]] = set()

    def add(relation, value):
        values = value if isinstance(value, list) else [value]
        for target in values:
            if isinstance(target, str) and target:
                result.add((source, relation, target))

    add("related_to", discovery.get("related_knowledge_ids"))
    add("sourced_from", payload.get("source_knowledge_ids"))
    add("requires", payload.get("required_knowledge_ids"))
    add("contains", payload.get("table_ids"))
    add("belongs_to", payload.get("table_id"))
    add("describes", payload.get("column_id"))
    add("default_time_field", payload.get("default_time_field_id"))
    add("applies_to", payload.get("applies_to"))
    if isinstance(payload.get("from"), dict):
        add("from_table", payload["from"].get("table_id"))
    if isinstance(payload.get("to"), dict):
        add("to_table", payload["to"].get("table_id"))
    return result


class KnowledgeNavigationGraphTest(unittest.TestCase):
    def test_graph_contains_every_card_and_every_explicit_edge(self):
        cards = load_knowledge_cards(KNOWLEDGE_ROOT)
        graph = build_navigation_graph(cards)
        refs = {node["ref"]: node["knowledge_id"] for node in graph["nodes"]}

        self.assertEqual(
            {node["knowledge_id"] for node in graph["nodes"]},
            set(cards),
        )
        actual_edges = {
            (refs[edge["source"]], edge["relation"], refs[edge["target"]])
            for edge in graph["edges"]
        }
        expected_edges = {
            edge
            for card in cards.values()
            for edge in _expected_explicit_edges(card)
        }
        self.assertEqual(actual_edges, expected_edges)

    def test_rendering_is_stable_and_each_full_id_occurs_once(self):
        cards = load_knowledge_cards(KNOWLEDGE_ROOT)
        first = render_navigation_graph(build_navigation_graph(cards))
        second = render_navigation_graph(build_navigation_graph(cards))

        self.assertEqual(first, second)
        node_lines = first.split("EDGES (source_ref|relation|target_ref)")[0]
        rendered_ids = [
            line.split("|", 3)[3]
            for line in node_lines.splitlines()[1:]
        ]
        self.assertEqual(len(rendered_ids), len(set(rendered_ids)))
        self.assertEqual(set(rendered_ids), set(cards))

    def test_map_excludes_payload_bodies_and_sql(self):
        cards = load_knowledge_cards(KNOWLEDGE_ROOT)
        rendered = render_navigation_graph(build_navigation_graph(cards))

        self.assertNotIn("formula_sql", rendered)
        self.assertNotIn("join_sql", rendered)
        self.assertNotIn("COUNT(*)", rendered)
        self.assertNotIn("SELECT ", rendered.upper())
        self.assertNotIn("payload", rendered.lower())

    def test_different_root_builds_a_different_graph_without_code_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            document = {
                "cards": [
                    {
                        "knowledge_id": "table.private.events",
                        "knowledge_type": "table",
                        "title": "Events",
                        "summary": "Event records.",
                        "payload": {},
                        "discovery": {
                            "related_knowledge_ids": [
                                "column.private.events.event_id"
                            ]
                        },
                    },
                    {
                        "knowledge_id": "column.private.events.event_id",
                        "knowledge_type": "column",
                        "title": "Event ID",
                        "summary": "Event identifier.",
                        "payload": {"table_id": "table.private.events"},
                        "discovery": {"related_knowledge_ids": []},
                    },
                ]
            }
            (root / "cards.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            cards = load_knowledge_cards(root)
            graph = build_navigation_graph(cards)
            rendered = render_navigation_graph(graph)

            self.assertEqual(len(graph["nodes"]), 2)
            self.assertEqual(len(graph["edges"]), 2)
            self.assertIn("table.private.events", rendered)
            self.assertIn("column.private.events.event_id", rendered)
            self.assertNotIn("cold_chain_pharma_compliance", rendered)


if __name__ == "__main__":
    unittest.main()
