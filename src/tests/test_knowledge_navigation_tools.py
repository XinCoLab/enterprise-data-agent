import json
import tempfile
import unittest
from pathlib import Path

import yaml

from knowledge_runtime.catalog import (
    browse_catalog,
    build_catalog,
    load_knowledge_cards,
    read_knowledge_cards,
    search_knowledge_cards,
)
from tools.browse_knowledge import browse_knowledge
from tools.read_knowledge import read_knowledge
from tools.search_knowledge import search_knowledge


BUNDLED_DEMO_ROOT = Path(__file__).resolve().parents[2] / "knowledge"


class BundledColdChainDemoCatalogTest(unittest.TestCase):
    """Regression checks for the repository's public LiveSQLBench demo."""

    def test_bundled_demo_loads_all_cards(self):
        cards = load_knowledge_cards(BUNDLED_DEMO_ROOT)

        self.assertEqual(len(cards), 157)
        self.assertEqual(
            {card.knowledge_type for card in cards.values()},
            {
                "database",
                "table",
                "column",
                "relationship",
                "metric",
                "glossary_term",
            },
        )

    def test_root_and_table_directory_are_browsable(self):
        cards = load_knowledge_cards(BUNDLED_DEMO_ROOT)
        catalog = build_catalog(cards)

        root = browse_catalog(catalog, "/")
        root_paths = {entry["path"] for entry in root["entries"]}
        self.assertEqual(
            root_paths,
            {
                "/database",
                "/table",
                "/column",
                "/relationship",
                "/metric",
                "/glossary_term",
            },
        )

        table_directory = browse_catalog(catalog, "/table")
        self.assertEqual(len(table_directory["entries"]), 12)
        self.assertTrue(
            all(entry["entry_type"] == "knowledge" for entry in table_directory["entries"])
        )

    def test_read_uses_an_exact_id_returned_by_browse(self):
        cards = load_knowledge_cards(BUNDLED_DEMO_ROOT)
        catalog = build_catalog(cards)
        first_table = browse_catalog(catalog, "/table")["entries"][0]

        opened = read_knowledge_cards(cards, [first_table["knowledge_id"]])

        self.assertEqual(opened[0]["knowledge_id"], first_table["knowledge_id"])
        self.assertIn("payload", opened[0])

    def test_search_returns_summaries_then_read_opens_the_card(self):
        cards = load_knowledge_cards(BUNDLED_DEMO_ROOT)

        matches = search_knowledge_cards(cards, "Shipments", "table")
        self.assertTrue(matches)
        self.assertNotIn("payload", matches[0])

        opened = read_knowledge_cards(cards, [matches[0]["knowledge_id"]])
        self.assertIn("payload", opened[0])

    def test_search_ranks_exact_titles_first_and_keeps_results_compact(self):
        cards = load_knowledge_cards(BUNDLED_DEMO_ROOT)

        shipment_matches = search_knowledge_cards(cards, "Shipments", "table")
        duration_matches = search_knowledge_cards(cards, "actual duration")

        self.assertEqual(shipment_matches[0]["title"], "Shipments")
        self.assertEqual(
            duration_matches[0]["title"],
            "Shipments: Shipment Overview",
        )
        self.assertLessEqual(len(shipment_matches), 8)
        self.assertLessEqual(len(duration_matches), 8)

    def test_same_loader_builds_a_different_root_without_code_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            other_root = Path(temporary_directory)
            nested_directory = other_root / "any" / "folder" / "layout"
            nested_directory.mkdir(parents=True)
            document = {
                "items": [
                    {
                        "knowledge_id": "glossary.private.example",
                        "knowledge_type": "glossary",
                        "title": "Private Example",
                        "summary": "A card in a different Knowledge Root.",
                        "payload": {"definition": "Example"},
                    }
                ]
            }
            (nested_directory / "cards.yaml").write_text(
                yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            cards = load_knowledge_cards(other_root)
            catalog = build_catalog(cards)
            root = browse_catalog(catalog, "/")

            self.assertEqual(list(cards), ["glossary.private.example"])
            self.assertEqual(
                root["entries"],
                [
                    {
                        "entry_type": "directory",
                        "name": "glossary",
                        "path": "/glossary",
                        "item_count": 1,
                    }
                ],
            )


class KnowledgeToolSchemaTest(unittest.TestCase):
    def test_browse_schema_explains_how_to_fill_the_argument(self):
        argument = browse_knowledge.args["directory_path"]

        self.assertIn("Use '/' for the first call", argument["description"])
        self.assertIn("not a filesystem path", argument["description"])

    def test_active_profile_tool_wrappers_complete_browse_read_and_search(self):
        root_result = json.loads(
            browse_knowledge.invoke({"directory_path": "/"})
        )
        self.assertIn("entries", root_result)

        table_result = json.loads(
            browse_knowledge.invoke({"directory_path": "/table"})
        )
        selected_id = table_result["entries"][0]["knowledge_id"]

        read_result = json.loads(
            read_knowledge.invoke({"knowledge_ids": [selected_id]})
        )
        self.assertEqual(read_result["cards"][0]["knowledge_id"], selected_id)
        selected_title = read_result["cards"][0]["title"]

        search_result = json.loads(
            search_knowledge.invoke(
                {"query": selected_title, "knowledge_type": "table"}
            )
        )
        self.assertTrue(search_result["results"])
        self.assertEqual(search_result["results"][0]["knowledge_id"], selected_id)


if __name__ == "__main__":
    unittest.main()
