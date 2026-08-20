"""Load the knowledge cards and virtual catalog selected by configuration."""

from config.project_paths import KNOWLEDGE_ROOT
from knowledge_runtime.catalog import build_catalog, load_knowledge_cards
from knowledge_runtime.navigation_graph import (
    build_navigation_graph,
    render_navigation_graph,
)


KNOWLEDGE_CARDS = load_knowledge_cards(KNOWLEDGE_ROOT)
KNOWLEDGE_CATALOG = build_catalog(KNOWLEDGE_CARDS)
KNOWLEDGE_NAVIGATION_GRAPH = build_navigation_graph(KNOWLEDGE_CARDS)
KNOWLEDGE_NAVIGATION_GRAPH_TEXT = render_navigation_graph(
    KNOWLEDGE_NAVIGATION_GRAPH
)
