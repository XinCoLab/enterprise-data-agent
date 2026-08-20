"""Build a complete navigation graph from explicit KnowledgeCard links."""

from dataclasses import dataclass

from knowledge_runtime.catalog import KnowledgeCard


@dataclass(frozen=True, order=True)
class KnowledgeEdge:
    """One explicit, directed reference between two knowledge cards."""

    source_id: str
    relation: str
    target_id: str


def _as_id_list(value: object) -> list[str]:
    """Normalize a schema field that explicitly contains knowledge IDs."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def explicit_edges(card: KnowledgeCard) -> set[KnowledgeEdge]:
    """Extract only relationships proven by explicit ID fields in a card."""

    content = card.content
    payload = content.get("payload", {})
    discovery = content.get("discovery", {})
    source_id = card.knowledge_id
    edges: set[KnowledgeEdge] = set()

    def add(relation: str, values: object) -> None:
        for target_id in _as_id_list(values):
            edges.add(KnowledgeEdge(source_id, relation, target_id))

    add("related_to", discovery.get("related_knowledge_ids"))
    add("sourced_from", payload.get("source_knowledge_ids"))
    add("requires", payload.get("required_knowledge_ids"))
    add("contains", payload.get("table_ids"))
    add("belongs_to", payload.get("table_id"))
    add("describes", payload.get("column_id"))
    add("default_time_field", payload.get("default_time_field_id"))
    add("applies_to", payload.get("applies_to"))

    from_endpoint = payload.get("from")
    if isinstance(from_endpoint, dict):
        add("from_table", from_endpoint.get("table_id"))

    to_endpoint = payload.get("to")
    if isinstance(to_endpoint, dict):
        add("to_table", to_endpoint.get("table_id"))

    return edges


def build_navigation_graph(cards: dict[str, KnowledgeCard]) -> dict:
    """Build the full node and explicit-edge tables for a Knowledge Root."""

    ordered_cards = sorted(cards.values(), key=lambda card: card.knowledge_id)
    node_refs = {
        card.knowledge_id: f"N{index:03d}"
        for index, card in enumerate(ordered_cards, start=1)
    }

    edges = sorted(
        {
            edge
            for card in ordered_cards
            for edge in explicit_edges(card)
        }
    )
    dangling_targets = sorted(
        {edge.target_id for edge in edges if edge.target_id not in node_refs}
    )
    if dangling_targets:
        raise ValueError(
            "Navigation graph contains unknown knowledge IDs: "
            + ", ".join(dangling_targets)
        )

    return {
        "nodes": [
            {
                "ref": node_refs[card.knowledge_id],
                "knowledge_id": card.knowledge_id,
                "knowledge_type": card.knowledge_type,
                "title": str(card.content["title"]),
            }
            for card in ordered_cards
        ],
        "edges": [
            {
                "source": node_refs[edge.source_id],
                "relation": edge.relation,
                "target": node_refs[edge.target_id],
            }
            for edge in edges
        ],
    }


def _compact(value: object) -> str:
    """Keep the line-oriented graph format stable and unambiguous."""

    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def render_navigation_graph(graph: dict) -> str:
    """Render a complete graph without card payloads or duplicated full IDs."""

    lines = ["NODES (ref|type|title|knowledge_id)"]
    lines.extend(
        "|".join(
            [
                node["ref"],
                _compact(node["knowledge_type"]),
                _compact(node["title"]),
                _compact(node["knowledge_id"]),
            ]
        )
        for node in graph["nodes"]
    )
    lines.append("EDGES (source_ref|relation|target_ref)")
    lines.extend(
        "|".join([edge["source"], edge["relation"], edge["target"]])
        for edge in graph["edges"]
    )
    return "\n".join(lines)
