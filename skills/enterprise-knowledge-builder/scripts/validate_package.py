#!/usr/bin/env python3
"""Validate an evidence-backed KnowledgeCard package deterministically."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import jsonschema
import yaml


REQUIRED_CARD_FIELDS = frozenset(
    {"knowledge_id", "knowledge_type", "title", "summary", "payload"}
)
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"reference_sql", "formula_sql", "expression_sql", "join_sql", "query_recipe"}
)
ALLOWED_SOURCE_ROLES = frozenset(
    {
        "observed_evidence",
        "declared_evidence",
        "review_evidence",
        "standard_evidence",
        "validation_evidence",
        "no_evidence",
    }
)


def walk_cards(node: object) -> Iterator[dict]:
    if isinstance(node, dict):
        if REQUIRED_CARD_FIELDS.issubset(node):
            yield node
            return
        for value in node.values():
            yield from walk_cards(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_cards(value)


def load_cards(root: Path, errors: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    cards: dict[str, dict] = {}
    paths: dict[str, str] = {}
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # parse errors need their exact file
            errors.append(f"Invalid YAML {path.relative_to(root)}: {exc}")
            continue
        for card in walk_cards(document):
            knowledge_id = str(card.get("knowledge_id", "")).strip()
            if not knowledge_id:
                errors.append(f"Empty knowledge_id in {path.relative_to(root)}")
                continue
            if knowledge_id in cards:
                errors.append(
                    f"Duplicate knowledge_id {knowledge_id}: "
                    f"{paths[knowledge_id]} and {path.relative_to(root)}"
                )
                continue
            cards[knowledge_id] = card
            paths[knowledge_id] = str(path.relative_to(root))
    return cards, paths


def id_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def explicit_edges(card: dict) -> set[tuple[str, str, str]]:
    source = card["knowledge_id"]
    payload = card.get("payload", {})
    discovery = card.get("discovery", {})
    edges: set[tuple[str, str, str]] = set()

    def add(relation: str, value: object) -> None:
        edges.update((source, relation, target) for target in id_values(value))

    add("related_to", discovery.get("related_knowledge_ids"))
    add("sourced_from", payload.get("source_knowledge_ids"))
    add("requires", payload.get("required_knowledge_ids"))
    add("contains", payload.get("table_ids"))
    add("belongs_to", payload.get("table_id"))
    add("describes", payload.get("column_id"))
    add("default_time_field", payload.get("default_time_field_id"))
    add("applies_to", payload.get("applies_to"))
    for relation, endpoint_name in (("from_table", "from"), ("to_table", "to")):
        endpoint = payload.get(endpoint_name)
        if isinstance(endpoint, dict):
            add(relation, endpoint.get("table_id"))
    return edges


def nested_forbidden_keys(node: object, prefix: str = "payload") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}"
            if key in FORBIDDEN_PAYLOAD_KEYS:
                hits.append(path)
            hits.extend(nested_forbidden_keys(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            hits.extend(nested_forbidden_keys(value, f"{prefix}[{index}]"))
    return hits


def load_source_registry(root: Path, errors: list[str]) -> dict[str, dict]:
    path = root / "sources" / "source_registry.yaml"
    if not path.exists():
        errors.append("Missing sources/source_registry.yaml")
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        errors.append(f"Invalid source registry: {exc}")
        return {}
    sources: dict[str, dict] = {}
    for source in document.get("sources", []):
        source_id = str(source.get("source_id", "")).strip()
        if not source_id:
            errors.append("Source registry contains an empty source_id")
            continue
        if source_id in sources:
            errors.append(f"Duplicate source_id: {source_id}")
            continue
        role = source.get("trust_role")
        if role not in ALLOWED_SOURCE_ROLES:
            errors.append(f"{source_id}: unsupported trust_role {role!r}")
        sources[source_id] = source
    return sources


def validate_source_paths(root: Path, sources: dict[str, dict], errors: list[str]) -> None:
    for source_id, source in sources.items():
        candidates = []
        project_path = source.get("project_relative_path")
        if isinstance(project_path, str) and project_path and "://" not in project_path:
            candidates.append(project_path)
        candidates.extend(
            asset for asset in source.get("assets", []) if isinstance(asset, str)
        )
        for relative in candidates:
            if not (root / relative).resolve().exists():
                errors.append(f"{source_id}: registered path does not exist: {relative}")


def validate_manifest(root: Path, counts: Counter, database_id: str | None, errors: list[str]) -> dict:
    path = root / "manifest.yaml"
    if not path.exists():
        errors.append("Missing manifest.yaml")
        return {}
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        errors.append(f"Invalid manifest.yaml: {exc}")
        return {}

    package_id = manifest.get("package", {}).get("database_id")
    if database_id and package_id != database_id:
        errors.append(
            f"Manifest database_id {package_id!r} does not match card database_id {database_id!r}"
        )
    expected = manifest.get("agent_knowledge", {}).get("card_counts", {})
    if expected:
        for knowledge_type, actual in counts.items():
            if expected.get(knowledge_type) != actual:
                errors.append(
                    f"Manifest count mismatch for {knowledge_type}: "
                    f"expected {expected.get(knowledge_type)!r}, actual {actual}"
                )
        if expected.get("total") != sum(counts.values()):
            errors.append(
                f"Manifest total mismatch: expected {expected.get('total')!r}, "
                f"actual {sum(counts.values())}"
            )
    return manifest


def runtime_smoke(root: Path, runtime_root: Path) -> dict:
    source_root = runtime_root / "src"
    if not source_root.is_dir():
        source_root = runtime_root
    sys.path.insert(0, str(source_root.resolve()))
    catalog_module = importlib.import_module("knowledge_runtime.catalog")
    graph_module = importlib.import_module("knowledge_runtime.navigation_graph")

    runtime_cards = catalog_module.load_knowledge_cards(root)
    catalog = catalog_module.build_catalog(runtime_cards)
    root_listing = catalog_module.browse_catalog(catalog, "/")
    first_type = sorted(catalog)[0]
    type_listing = catalog_module.browse_catalog(catalog, f"/{first_type}")
    first_card = sorted(runtime_cards)[0]
    search_results = catalog_module.search_knowledge_cards(runtime_cards, first_card)
    read_result = catalog_module.read_knowledge_cards(runtime_cards, [first_card])
    graph = graph_module.build_navigation_graph(runtime_cards)
    if not root_listing["entries"] or not type_listing["entries"]:
        raise AssertionError("Runtime browse returned an empty directory")
    if not search_results or not read_result:
        raise AssertionError("Runtime search/read smoke failed")
    return {
        "result": "PASS",
        "card_count": len(runtime_cards),
        "catalog_directory_count": len(root_listing["entries"]),
        "navigation_node_count": len(graph["nodes"]),
        "navigation_edge_count": len(graph["edges"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("knowledge_root", type=Path)
    parser.add_argument("--runtime-root", type=Path, help="Optional DataAgent checkout")
    parser.add_argument("--schema", type=Path, help="Override JSON Schema path")
    parser.add_argument("--no-write", action="store_true", help="Do not write validation report")
    args = parser.parse_args()

    root = args.knowledge_root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        raise SystemExit(f"Knowledge root does not exist: {root}")

    cards, card_paths = load_cards(root, errors)
    if not cards:
        errors.append("No KnowledgeCards found")

    schema_path = args.schema
    if schema_path is None:
        package_schema = root / "contracts" / "knowledge_entry.schema.json"
        legacy_schema = root / "contracts" / "knowledge_entry.schema.v0.1.json"
        if package_schema.exists():
            schema_path = package_schema
        elif legacy_schema.exists():
            schema_path = legacy_schema
        else:
            schema_path = Path(__file__).resolve().parents[1] / "assets" / "knowledge_entry.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        for knowledge_id, card in cards.items():
            for error in validator.iter_errors(card):
                location = "/".join(str(part) for part in error.path) or "<card>"
                errors.append(f"{knowledge_id} [{location}]: {error.message}")
    except Exception as exc:
        errors.append(f"Unable to load or apply schema {schema_path}: {exc}")

    database_ids = {str(card.get("database_id", "")) for card in cards.values()}
    if "" in database_ids:
        errors.append("One or more cards have an empty database_id")
        database_ids.discard("")
    if len(database_ids) > 1:
        errors.append(f"Multiple logical database IDs in one package: {sorted(database_ids)}")
    database_id = next(iter(database_ids), None)

    for knowledge_id, card in cards.items():
        parts = knowledge_id.split(".")
        if database_id and (len(parts) < 3 or parts[1] != database_id):
            errors.append(
                f"{knowledge_id}: ID namespace does not match database_id {database_id!r}"
            )
        if card.get("knowledge_type") == "query_recipe":
            errors.append(f"{knowledge_id}: query_recipe cards are forbidden")
        for hit in nested_forbidden_keys(card.get("payload", {})):
            errors.append(f"{knowledge_id}: forbidden SQL/query-recipe key {hit}")

    edges = {edge for card in cards.values() for edge in explicit_edges(card)}
    for source, relation, target in sorted(edges):
        if target not in cards:
            errors.append(f"Dangling Knowledge edge: {source} --{relation}--> {target}")

    sources = load_source_registry(root, errors)
    validate_source_paths(root, sources, errors)
    source_roles = Counter(source.get("trust_role") for source in sources.values())
    for knowledge_id, card in cards.items():
        local_evidence_ids = {
            str(item.get("evidence_id", ""))
            for item in card.get("evidence_refs", [])
            if isinstance(item, dict)
        }
        for ref in card.get("evidence_refs", []):
            source_id = ref.get("source_id")
            if source_id not in sources:
                errors.append(f"{knowledge_id}: unknown source_id {source_id!r}")
            elif sources[source_id].get("trust_role") == "no_evidence":
                errors.append(f"{knowledge_id}: cites no_evidence source {source_id}")
        for assertion in card.get("assertions", []):
            for evidence_id in assertion.get("evidence_ids", []):
                if evidence_id not in local_evidence_ids:
                    errors.append(
                        f"{knowledge_id}: assertion cites unknown local evidence_id {evidence_id}"
                    )

    table_cards = [card for card in cards.values() if card.get("knowledge_type") == "table"]
    for card in table_cards:
        payload = card.get("payload", {})
        physical_database = str(payload.get("physical_database", ""))
        physical_name = str(payload.get("physical_name", ""))
        qualified_name = str(payload.get("qualified_name", ""))
        recommended = str(payload.get("recommended_sql_reference", ""))
        if physical_database and physical_name and physical_name not in qualified_name:
            errors.append(f"{card['knowledge_id']}: qualified_name omits physical table name")
        if database_id and physical_database and database_id != physical_database:
            forbidden = f"{database_id}.{physical_name}"
            if recommended == forbidden:
                errors.append(
                    f"{card['knowledge_id']}: recommended_sql_reference uses logical namespace"
                )
            identity_text = " ".join(
                [str(card.get("summary", "")), *map(str, payload.get("usage_warnings", []))]
            )
            if physical_database not in identity_text:
                warnings.append(
                    f"{card['knowledge_id']}: summary/warnings do not name physical database {physical_database}"
                )

    counts = Counter(str(card.get("knowledge_type")) for card in cards.values())
    statuses = Counter(str(card.get("status")) for card in cards.values())
    manifest = validate_manifest(root, counts, database_id, errors)

    runtime_result: dict = {"result": "NOT_RUN"}
    if args.runtime_root:
        try:
            runtime_result = runtime_smoke(root, args.runtime_root.expanduser().resolve())
        except Exception as exc:
            errors.append(f"Runtime compatibility smoke failed: {exc!r}")
            runtime_result = {"result": "FAIL", "error": repr(exc)}

    report = {
        "knowledge_root": str(root),
        "result": "PASS" if not errors else "FAIL",
        "logical_database_id": database_id,
        "physical_database": manifest.get("package", {}).get("physical_schema"),
        "schema_path": str(schema_path),
        "card_count": len(cards),
        "card_counts_by_type": dict(sorted(counts.items())),
        "status_counts": dict(sorted(statuses.items())),
        "knowledge_id_sha256": hashlib.sha256(
            "\n".join(sorted(cards)).encode("utf-8")
        ).hexdigest(),
        "navigation_node_count": len(cards),
        "navigation_edge_count": len(edges),
        "source_count": len(sources),
        "source_counts_by_trust_role": dict(sorted(source_roles.items(), key=lambda item: str(item[0]))),
        "runtime_smoke": runtime_result,
        "errors": errors,
        "warnings": warnings,
    }

    if not args.no_write:
        output = root / "validation" / "skill_validation_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
