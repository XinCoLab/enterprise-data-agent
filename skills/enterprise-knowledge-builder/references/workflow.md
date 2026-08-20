# Evidence-first build workflow

This workflow converts heterogeneous enterprise material into a small, reviewable semantic layer for a data agent. It deliberately separates what the database physically contains from what the business says those objects mean.

## 1. Freeze scope and authority

Write down before generating cards:

- logical database ID;
- engine and physical database/schema/catalog;
- included tables or domains;
- excluded systems, documents, and legacy drafts;
- whether database access is allowed;
- whether access is metadata-only or permits minimal read-only aggregate probes;
- reviewers who can approve business definitions.

Do not let convenient access broaden the authorized scope.

## 2. Build a source registry

Give each source a stable ID and one trust role.

| Trust role | What it can support | Typical examples |
| --- | --- | --- |
| `observed_evidence` | Physical objects, types, constraints, observed distributions | metadata snapshot, read-only aggregate probe |
| `declared_evidence` | Intended business concepts and official definitions | approved product or business document |
| `review_evidence` | Human-confirmed interpretation or requirement | signed review note, confirmed decision |
| `standard_evidence` | External standard definition, not local field binding | standards excerpt |
| `validation_evidence` | A tested mapping or lineage conclusion | dashboard-to-table reconciliation |
| `no_evidence` | Discovery only; cannot support facts | obsolete drafts, unreviewed generated notes |

For every source, record title, type, path or locator, snapshot/version when applicable, access policy, trust role, and boundaries. A standard may define a formula but still not prove which local columns implement it.

## 3. Capture the physical layer

Use engine-native metadata only. Examples include `information_schema`, PostgreSQL catalogs, MySQL metadata, and DuckDB metadata/PRAGMA facilities.

Capture at least:

- physical database/schema/catalog names;
- object name and object type;
- columns, physical types, nullability, defaults;
- primary and unique keys;
- declared foreign keys;
- views and their observable metadata;
- snapshot timestamp and source ID.

Do not copy arbitrary business rows into the Knowledge package. If relationship cardinality or enum coverage requires a probe, use the smallest read-only aggregate that can answer the validation question and record the probe as evidence rather than embedding its result rows as knowledge.

Generate physical cards conservatively:

- physical existence and type may be `VALIDATED` from metadata;
- business definition, grain, and role remain `null`, `unknown`, or `DRAFT` until supported;
- a name such as `amount`, `status`, or `customer_id` is not enough to assert unit, state semantics, or a foreign-key relationship.

## 4. Add business meaning

Map reviewed sources onto physical cards. Keep each assertion attributable.

For tables, confirm:

- business entity or event;
- row grain and lifecycle;
- authoritative key and duplicate behavior;
- fact/dimension/bridge/cache role;
- retention or partial-history boundaries.

For columns, confirm:

- business meaning;
- unit and timezone;
- identifier versus measure versus state;
- default aggregation if one truly exists;
- NULL meaning and sentinel values;
- safe and unsafe uses.

For enums, distinguish observed values from a closed business domain. An observed list is not automatically exhaustive.

## 5. Validate relationships

A relationship card is not merely a plausible equality. Establish:

- exact endpoint table and column IDs;
- direction and cardinality;
- whether the relation is declared, observed, reviewed, or only a candidate;
- orphan rate and key uniqueness when authorized;
- whether joining preserves the source grain;
- fanout and duplicate-count risk;
- temporal or status conditions, if any.

If these are not known, keep cardinality `unknown`, status `DRAFT`, and create a warning card describing what must be verified.

## 6. Compile reusable business knowledge

Create cards for reusable concepts, not task-specific instructions.

### Measures

Record base grain, semantic expression, aggregation, unit, source Knowledge IDs, NULL handling, rounding, and warnings.

### Metrics

Record business definition, output grain, numerator, denominator, semantic formula, source Knowledge IDs, default time field, time-window semantics, NULL policy, unit, rounding, and warnings.

Do not store executable SQL. The Agent should generate SQL for the current dialect and task using the underlying facts.

### Glossary terms

State the approved meaning and explicitly excluded meanings. Link to physical or metric cards only when the binding is supported.

### Warnings

Use warnings for ambiguity and unsafe substitutions, including:

- logical ID mistaken for physical schema;
- cache metric mistaken for raw recomputation;
- current snapshot mistaken for historical coverage;
- alarm mistaken for failure;
- candidate join mistaken for validated relationship;
- standard formula mistaken for a local implementation;
- design-time object mistaken for a deployed table.

## 7. Create explicit navigation

Use only the explicit ID fields described in `knowledge-contract.md`. Prefer a connected graph around each core table:

- database → tables;
- table ↔ columns;
- relationship → endpoint tables;
- measure/metric → source cards and default time field;
- enum → described column;
- warning → affected cards;
- glossary and discovery links → closely related cards.

Do not add links merely because two embeddings or names are similar. Search and similarity may propose candidates, but a persisted edge must be reviewable and explicit.

## 8. Status promotion

Use status as an evidence claim, not a progress percentage.

| Status | Meaning |
| --- | --- |
| `DRAFT` | Candidate or incomplete; cannot be treated as settled business truth |
| `VALIDATED` | Supported by physical metadata or a recorded validation |
| `REVIEWED` | Reviewed business source or human decision supports the meaning |
| `CERTIFIED` | Organization-specific certification process completed |
| `STALE` | Previously supported but its source or snapshot is outdated |
| `DEPRECATED` | Retained for lineage but should not be used for new analysis |

A card can contain both validated physical facts and unresolved business meaning. In that case, keep the card `DRAFT` or separate the confirmed facts from the unresolved assertion.

## 9. Validate and review

Run deterministic validation, then review the package as a human data analyst:

1. Can every ID and source reference be resolved?
2. Can the logical ID ever be mistaken for a physical SQL name?
3. Are grain, cardinality, NULL rules, units, time fields, and warnings explicit where needed?
4. Does any text leak SQL, benchmark answers, or current-task hints?
5. Can the runtime browse, search, read, and construct its navigation graph?
6. Are unresolved items visible rather than silently filled?

## 10. Incremental maintenance

Never regenerate reviewed cards blindly from a new schema snapshot.

On update:

- diff the physical snapshot;
- preserve stable Knowledge IDs for unchanged concepts;
- mark removed or changed objects `STALE` before deletion;
- update evidence and revision numbers;
- rerun dangling-reference and manifest checks;
- request renewed business review when semantics, grain, or metric behavior may have changed.
