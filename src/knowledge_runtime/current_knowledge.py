"""Load the knowledge cards and virtual catalog selected by configuration."""

from collections import Counter
from pathlib import Path

from config.project_paths import KNOWLEDGE_ROOT
from knowledge_runtime.catalog import build_catalog, load_knowledge_cards
from knowledge_runtime.navigation_graph import (
    build_navigation_graph,
    render_navigation_graph,
)


def _load_knowledge_bundle(root: Path):
    cards = load_knowledge_cards(root)
    catalog = build_catalog(cards)
    navigation_graph = build_navigation_graph(cards)
    navigation_text = render_navigation_graph(navigation_graph)
    return cards, catalog, navigation_graph, navigation_text


def prepare_knowledge(root: Path) -> dict:
    """Parse and validate one bundle before any active configuration is changed."""
    root = root.expanduser().resolve()
    cards, catalog, graph, text = _load_knowledge_bundle(root)
    counts = Counter(card.knowledge_type for card in cards.values())
    return {
        "root": root, "cards": cards, "catalog": catalog,
        "navigation_graph": graph, "navigation_text": text,
        "summary": {"path": str(root), "card_count": len(cards), "types": dict(sorted(counts.items()))},
    }


def activate_knowledge(bundle: dict) -> None:
    """Publish the already prepared bundle without reading the YAML files again."""

    global KNOWLEDGE_CARDS
    global KNOWLEDGE_CATALOG
    global KNOWLEDGE_NAVIGATION_GRAPH
    global KNOWLEDGE_NAVIGATION_GRAPH_TEXT
    global _ACTIVE_KNOWLEDGE_SUMMARY

    (
        KNOWLEDGE_CARDS,
        KNOWLEDGE_CATALOG,
        KNOWLEDGE_NAVIGATION_GRAPH,
        KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
    ) = (bundle["cards"], bundle["catalog"], bundle["navigation_graph"], bundle["navigation_text"])
    _ACTIVE_KNOWLEDGE_SUMMARY = bundle["summary"]


def get_loaded_knowledge_summary(root: Path) -> dict:
    """Return the active root's summary; never scan files during page refresh."""
    summary = _ACTIVE_KNOWLEDGE_SUMMARY
    if str(root.expanduser().resolve()) != summary["path"]:
        raise ValueError("当前知识库尚未加载，请重新应用配置。")
    return {**summary, "types": dict(summary["types"])}


def reload_knowledge(root: Path) -> None:
    """Explicit reloads and knowledge edits also replace the cached summary."""
    activate_knowledge(prepare_knowledge(root))


reload_knowledge(KNOWLEDGE_ROOT)
