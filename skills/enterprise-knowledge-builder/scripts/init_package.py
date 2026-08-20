#!/usr/bin/env python3
"""Create a non-destructive enterprise Knowledge package skeleton."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


DIRECTORIES = (
    "contracts",
    "database",
    "physical",
    "semantic/tables",
    "semantic/relationships",
    "semantic/enums",
    "semantic/glossary",
    "metrics",
    "warnings",
    "sources/reviews",
    "sources/evidence_assets",
    "validation",
)


def safe_database_id(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized or not normalized[0].isalpha():
        raise ValueError("database_id must start with a lowercase letter")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized):
        raise ValueError("database_id may contain only lowercase letters, digits, and underscores")
    return normalized


def write_yaml(path: Path, value: object) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New Knowledge root")
    parser.add_argument("--database-id", required=True, help="Logical ID used in Knowledge IDs")
    parser.add_argument("--physical-database", required=True, help="Physical database/schema/catalog")
    parser.add_argument("--engine", required=True, choices=("PostgreSQL", "MySQL", "DuckDB"))
    parser.add_argument("--language", default="zh-CN")
    args = parser.parse_args()

    database_id = safe_database_id(args.database_id)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {output}")

    output.mkdir(parents=True, exist_ok=True)
    for relative in DIRECTORIES:
        (output / relative).mkdir(parents=True, exist_ok=True)

    skill_root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        skill_root / "assets" / "knowledge_entry.schema.json",
        output / "contracts" / "knowledge_entry.schema.json",
    )

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "package": {
            "database_id": database_id,
            "physical_schema": args.physical_database,
            "language": args.language,
            "knowledge_version": "0.1.0-draft",
            "contract_version": "0.1.1",
            "lifecycle_status": "DRAFT",
            "updated_at": now,
        },
        "database_identity": {
            "engine": args.engine,
            "snapshot_id": None,
            "observed_object_count": 0,
            "observed_column_count": 0,
            "included_core_object_count": 0,
        },
        "agent_knowledge": {
            "included_types": [],
            "card_counts": {"total": 0},
            "excluded_artifacts": [
                "SQL",
                "query recipes",
                "benchmark questions",
                "Gold answers",
                "business data rows",
            ],
        },
        "review": {
            "status": "DRAFT",
            "reason": "The package skeleton contains no reviewed KnowledgeCards yet.",
        },
    }
    write_yaml(output / "manifest.yaml", manifest)
    write_yaml(
        output / "sources" / "source_registry.yaml",
        {
            "registry_version": "0.1.0",
            "database_id": database_id,
            "updated_at": now,
            "sources": [],
        },
    )

    print(f"Created Knowledge package skeleton: {output}")
    print("Next: register sources, capture metadata, create cards, then run validate_package.py.")


if __name__ == "__main__":
    main()
