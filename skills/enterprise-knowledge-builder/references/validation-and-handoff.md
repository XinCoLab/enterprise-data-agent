# Validation and handoff

Completion requires four gates. Do not merge them into a single vague “validated” claim.

## Gate 1: Structural contract

Verify:

- YAML and JSON parse successfully;
- every card satisfies the contract schema;
- Knowledge IDs are non-empty and unique;
- all cards use one logical `database_id` for a single-database package;
- the ID namespace matches that logical ID;
- required payload fields exist for each card type;
- revision and timestamp fields are valid;
- forbidden SQL/query-recipe/benchmark artifacts are absent.

## Gate 2: Referential integrity

Verify:

- every explicit navigation target exists;
- every evidence `source_id` is registered;
- no card cites a source with `trust_role: no_evidence`;
- evidence assets and project-relative source paths exist where portability requires them;
- assertion evidence IDs resolve within the card;
- manifest card counts match actual cards.

Report node count, edge count, dangling references, card counts by type, and status counts.

## Gate 3: Runtime compatibility

When the target DataAgent is available, execute its real functions rather than a reimplementation:

1. load all cards;
2. build the catalog;
3. browse `/` and one type directory;
4. search by a known title, alias, or keyword;
5. read a returned ID;
6. build the complete explicit navigation graph.

The smoke test must not connect to a production database or call an LLM.

## Gate 4: Semantic review

Have a human reviewer inspect high-impact cards:

- core tables and grains;
- relationship cardinality and fanout;
- metric numerator, denominator, NULL policy, time field, unit, and rounding;
- status and enum semantics;
- cache/raw and current/history distinctions;
- warnings and unresolved ambiguities;
- logical versus physical database identity.

Schema-valid text can still be wrong. Deterministic validation does not certify business meaning.

## Delivery report

Return a compact report with:

```text
Knowledge root:
Logical database ID:
Physical database/schema:
Engine:
Snapshot/version:

Cards by type:
Navigation nodes/edges:
Sources by trust role:

Structural validation:
Referential validation:
Runtime smoke:
Human review:

Validated facts:
Reviewed business meanings:
Remaining DRAFT items:
Excluded artifacts:
Files created or changed:
```

Never report “complete” while remaining ambiguities could change a query result. List those ambiguities and the exact evidence or reviewer decision needed to resolve them.
