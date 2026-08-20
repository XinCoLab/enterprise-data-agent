---
name: enterprise-knowledge-builder
description: Build, validate, and hand off evidence-backed, directory-based KnowledgeCard packages for enterprise data agents. Use when turning database metadata, data dictionaries, business definitions, reviewed decisions, and read-only validation evidence into a DataAgent-compatible Knowledge root with stable IDs and explicit navigation links. Do not use for answering business questions, writing production SQL, or modifying the Agent runtime.
---

# Enterprise Knowledge Builder

Build a reusable enterprise Knowledge package that separates physical database facts from reviewed business meaning. The package must be useful across future questions, not optimized for one benchmark case.

## Read first

Before creating or changing a package, read these files completely:

1. [references/workflow.md](references/workflow.md) — evidence intake, build phases, review gates, and update discipline.
2. [references/knowledge-contract.md](references/knowledge-contract.md) — directory layout, card envelope, payloads, IDs, and explicit graph edges.
3. [references/validation-and-handoff.md](references/validation-and-handoff.md) — deterministic checks and delivery report.

If a target runtime is available, inspect its real Knowledge loader and navigation-graph implementation before building. Do not assume that another project uses this contract unchanged.

## Required inputs

Identify and record:

- logical database ID used only in Knowledge IDs;
- database engine and physical database/schema/catalog names used in SQL;
- metadata snapshot for in-scope tables, columns, keys, views, and types;
- reviewed business documents, data dictionaries, metric definitions, and human decisions;
- optional read-only aggregate or relationship-validation evidence;
- target scope and explicitly excluded material.

If an input is missing, preserve the uncertainty. Do not fill it from model familiarity or field-name similarity.

## Build procedure

1. Create a new package skeleton with `scripts/init_package.py`, or reproduce the documented layout exactly.
2. Register every usable source before citing it. Mark rejected legacy material as `no_evidence` rather than silently mixing it with trusted sources.
3. Capture physical metadata without inventing business meaning. Database, table, and column cards may be generated mechanically from metadata.
4. Add business definitions only when supported by declared, reviewed, standard, validation, or observed evidence.
5. Add relationship cards only after documenting endpoints, cardinality, grain preservation, and fanout risk. Keep unresolved relationships `DRAFT` and add a warning.
6. Add measure and metric cards only when grain, source cards, formula semantics, NULL handling, unit, time field, and rounding are known. Store semantic expressions, not executable SQL.
7. Add glossary, enum, and warning cards for terminology, bounded values, ambiguity, unsafe substitutions, cache-versus-raw distinctions, and version boundaries.
8. Link cards only through explicit Knowledge ID fields. Never manufacture graph edges from semantic similarity.
9. Run `scripts/validate_package.py` until deterministic validation succeeds.
10. Perform human review for business meaning and report unresolved items honestly.

## Non-negotiable rules

- Keep logical identity separate from physical SQL identity. `table.acme.orders` does not imply an SQL schema named `acme`.
- Never infer business semantics, joins, cardinality, units, status meaning, or metric formulas from names alone.
- Never promote an inference to `VALIDATED` or `REVIEWED` without supporting evidence.
- Do not include production queries, query recipes, benchmark questions, Gold SQL/results, candidate answers, or business data rows.
- Do not encode one user's current question as permanent enterprise knowledge.
- Do not connect to a database or inspect data rows without explicit authorization. Prefer metadata and minimal read-only aggregate probes.
- Preserve unresolved ambiguity in `DRAFT` cards or warning cards instead of guessing.
- Keep every referenced Knowledge ID and source ID resolvable.

## Validation commands

From this skill directory:

```powershell
python scripts\validate_package.py "C:\path\to\knowledge-root"
```

To verify compatibility with an available DataAgent checkout:

```powershell
python scripts\validate_package.py "C:\path\to\knowledge-root" --runtime-root "C:\path\to\DataAgent"
```

Do not claim completion from schema validation alone. A completed handoff includes deterministic validation, runtime smoke when available, source/evidence review, and a list of remaining uncertainties.
