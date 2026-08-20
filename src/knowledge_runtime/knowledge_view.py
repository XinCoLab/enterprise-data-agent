"""Select the runtime Knowledge view shown to the Main Agent."""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


KnowledgeViewMode = Literal["GLOBAL", "SUBGLOBAL", "REGLOBAL"]


def _current_turn_messages(state: dict) -> list:
    """Return messages from the latest HumanMessage onward."""

    messages = list(state.get("messages", []))
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _tool_calls_by_id(messages: list) -> dict[str, dict]:
    """Index real AI tool calls by their tool_call_id."""

    calls = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "")
            if call_id:
                calls[call_id] = call
    return calls


def _successful_read_events(messages: list) -> list[tuple[int, list[str]]]:
    """Return successful read positions and the IDs actually returned."""

    calls = _tool_calls_by_id(messages)
    events: list[tuple[int, list[str]]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(str(message.tool_call_id))
        if not call or call.get("name") != "read_knowledge":
            continue
        arguments = call.get("args") or {}
        requested_ids = arguments.get("knowledge_ids", [])
        if not isinstance(requested_ids, list):
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, json.JSONDecodeError):
            continue
        cards = payload.get("cards") if isinstance(payload, dict) else None
        if not isinstance(cards, list):
            continue
        returned_ids = {
            str(card["knowledge_id"])
            for card in cards
            if isinstance(card, dict) and card.get("knowledge_id")
        }
        confirmed_ids = [
            str(knowledge_id)
            for knowledge_id in requested_ids
            if str(knowledge_id) in returned_ids
        ]
        if confirmed_ids:
            events.append((index, confirmed_ids))
    return events


def _discovery_events(messages: list) -> list[int]:
    """Return completed search/browse positions used as REGLOBAL signals."""

    calls = _tool_calls_by_id(messages)
    positions = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(str(message.tool_call_id))
        if call and call.get("name") in {"search_knowledge", "browse_knowledge"}:
            positions.append(index)
    return positions


def successful_read_knowledge_ids(state: dict) -> list[str]:
    """Return unique IDs confirmed by successful reads in the current turn."""

    messages = _current_turn_messages(state)
    seen = set()
    ordered_ids = []
    for _, knowledge_ids in _successful_read_events(messages):
        for knowledge_id in knowledge_ids:
            if knowledge_id not in seen:
                seen.add(knowledge_id)
                ordered_ids.append(knowledge_id)
    return ordered_ids


def build_subglobal_knowledge_graph(state: dict, global_graph: dict) -> dict:
    """Build the read-node working set and its real one-hop frontier."""

    nodes_by_id = {
        node["knowledge_id"]: node
        for node in global_graph["nodes"]
    }
    nodes_by_ref = {
        node["ref"]: node
        for node in global_graph["nodes"]
    }
    subglobal_ids = successful_read_knowledge_ids(state)
    unknown_ids = sorted(set(subglobal_ids) - set(nodes_by_id))
    if unknown_ids:
        raise ValueError(
            "Subglobal Knowledge Graph contains unknown knowledge IDs: "
            + ", ".join(unknown_ids)
        )

    subglobal_refs = {
        nodes_by_id[knowledge_id]["ref"] for knowledge_id in subglobal_ids
    }
    selected_edges = []
    frontier_refs = set()
    for edge in global_graph["edges"]:
        source_subglobal = edge["source"] in subglobal_refs
        target_subglobal = edge["target"] in subglobal_refs
        if not source_subglobal and not target_subglobal:
            continue
        selected_edges.append(dict(edge))
        if source_subglobal and not target_subglobal:
            frontier_refs.add(edge["target"])
        if target_subglobal and not source_subglobal:
            frontier_refs.add(edge["source"])

    return {
        "read_nodes": [
            nodes_by_id[knowledge_id] for knowledge_id in subglobal_ids
        ],
        "relations": selected_edges,
        "frontier_nodes": [
            nodes_by_ref[ref]
            for ref in sorted(frontier_refs)
        ],
    }


def render_subglobal_knowledge_graph(subglobal_graph: dict) -> str:
    """Render the subglobal working set without KnowledgeCard bodies."""

    lines = ["SUBGLOBAL KNOWLEDGE GRAPH", "[READ] (ref|type|title|knowledge_id)"]
    lines.extend(
        "|".join(
            [
                node["ref"],
                str(node["knowledge_type"]),
                str(node["title"]).replace("|", "/"),
                str(node["knowledge_id"]),
            ]
        )
        for node in subglobal_graph["read_nodes"]
    )
    lines.append("[RELATIONS] (source_ref|relation|target_ref)")
    lines.extend(
        "|".join([edge["source"], edge["relation"], edge["target"]])
        for edge in subglobal_graph["relations"]
    )
    lines.append("[FRONTIER] (ref|type|title|knowledge_id)")
    lines.extend(
        "|".join(
            [
                node["ref"],
                str(node["knowledge_type"]),
                str(node["title"]).replace("|", "/"),
                str(node["knowledge_id"]),
            ]
        )
        for node in subglobal_graph["frontier_nodes"]
    )
    return "\n".join(lines)


def select_knowledge_view(state: dict) -> KnowledgeViewMode:
    """Choose the deterministic GLOBAL, SUBGLOBAL, or REGLOBAL view."""

    messages = _current_turn_messages(state)
    read_events = _successful_read_events(messages)
    if not read_events:
        return "GLOBAL"

    latest_read_position = read_events[-1][0]
    discovery_positions = _discovery_events(messages)
    if discovery_positions and discovery_positions[-1] > latest_read_position:
        return "REGLOBAL"
    return "SUBGLOBAL"
