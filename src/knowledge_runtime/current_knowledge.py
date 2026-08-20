"""Load the knowledge cards and virtual catalog selected by configuration."""

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


(
    KNOWLEDGE_CARDS,
    KNOWLEDGE_CATALOG,
    KNOWLEDGE_NAVIGATION_GRAPH,
    KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
) = _load_knowledge_bundle(KNOWLEDGE_ROOT)


def reload_knowledge(root: Path) -> None:
    """Replace the active Knowledge bundle after explicit validation."""

    global KNOWLEDGE_CARDS
    global KNOWLEDGE_CATALOG
    global KNOWLEDGE_NAVIGATION_GRAPH
    global KNOWLEDGE_NAVIGATION_GRAPH_TEXT

    bundle = _load_knowledge_bundle(root)
    (
        KNOWLEDGE_CARDS,
        KNOWLEDGE_CATALOG,
        KNOWLEDGE_NAVIGATION_GRAPH,
        KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
    ) = bundle
