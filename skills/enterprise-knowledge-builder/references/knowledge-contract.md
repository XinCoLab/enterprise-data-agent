# Knowledge package contract

The runtime scans YAML recursively. File placement is for human maintainability; stable `knowledge_id` values, not paths, are the runtime identity.

## Recommended layout

```text
knowledge-root/
├── manifest.yaml
├── contracts/
│   └── knowledge_entry.schema.json
├── database/
│   └── database.yaml
├── physical/
│   └── schema_snapshot.json
├── semantic/
│   ├── tables/
│   ├── relationships/
│   ├── enums/
│   └── glossary/
├── metrics/
├── warnings/
├── sources/
│   ├── source_registry.yaml
│   ├── reviews/
│   └── evidence_assets/
└── validation/
```

Directories may be reorganized without changing runtime lookup, provided the YAML cards remain valid and IDs remain stable.

## Common card envelope

Every card must include:

```yaml
knowledge_id: table.acme.orders
knowledge_type: table
database_id: acme
title: Orders
summary: Short directory-facing description that distinguishes this card.
payload: {}
discovery:
  keywords: [orders, purchase]
  aliases: [customer orders]
  related_knowledge_ids: []
evidence_refs:
  - evidence_id: evidence.orders_table_metadata
    evidence_type: observed
    source_id: source.acme.schema_snapshot
    locator: TABLE_NAME=orders
    snapshot_id: snapshot.acme.20260820
    validation_id: null
    result_summary: Physical table and columns observed in the metadata snapshot.
status: VALIDATED
revision: 1
updated_at: '2026-08-20T00:00:00Z'
```

The runtime's minimal loader requires `knowledge_id`, `knowledge_type`, `title`, `summary`, and `payload`. This skill deliberately requires the richer evidence and lifecycle envelope as the preferred package format.

## ID rules

Use lowercase stable IDs:

```text
database.<logical_database_id>.main
table.<logical_database_id>.<object_name>
column.<logical_database_id>.<object_name>.<column_name>
relationship.<logical_database_id>.<from>__<to>
measure.<logical_database_id>.<name>
metric.<logical_database_id>.<name>
enum.<logical_database_id>.<name>
warning.<logical_database_id>.<name>
glossary_term.<logical_database_id>.<name>
```

Normalize unsafe characters to underscores. Never rename an ID merely because a file moved or a title changed.

The logical namespace is not an SQL identifier. For example:

```yaml
knowledge_id: table.acme.orders
database_id: acme
payload:
  physical_database: analytics_prod
  physical_name: orders
  qualified_name: analytics_prod.orders
  recommended_sql_reference: orders
```

The Agent must never infer `acme.orders` as the SQL object from the Knowledge ID.

## Card payloads

### Database

Required payload fields:

- `business_context`
- `schema_names`
- `table_ids`
- `physical_database`

### Table

Required payload fields:

- `physical_database`
- `physical_name`
- `qualified_name`
- `recommended_sql_reference`
- `business_definition`
- `grain`
- `table_role`
- `logical_primary_key`
- `usage_warnings`

Use `unknown` or `null` when business role or grain is unconfirmed.

### Column

Required payload fields:

- `table_id`
- `physical_name`
- `physical_type`
- `business_type`
- `column_role`
- `nullable`
- `business_definition`
- `unit`
- `default_aggregation`
- `usage_warnings`

Physical type and nullability can come from metadata. Business type, unit, meaning, and aggregation require stronger evidence.

### Relationship

```yaml
payload:
  from:
    table_id: table.acme.orders
    columns: [customer_id]
  to:
    table_id: table.acme.customers
    columns: [customer_id]
  cardinality: many_to_one
  join_condition: orders.customer_id equals customers.customer_id
  preserves_from_grain: true
  fanout_risks: []
```

`join_condition` is a semantic equality description, not a reusable SQL fragment.

### Enum

Required payload fields:

- `column_id`
- `closed_domain`
- `values`

Set `closed_domain: false` for sampled or currently observed values.

### Measure

Required payload fields:

- `business_definition`
- `base_grain`
- `expression`
- `aggregation`
- `unit`
- `rounding_rule`
- `source_knowledge_ids`
- `null_handling`
- `usage_warnings`

### Metric

Required payload fields:

- `business_definition`
- `grain`
- `formula_expression`
- `numerator`
- `denominator`
- `source_knowledge_ids`
- `default_time_field_id`
- `null_handling`
- `unit`
- `rounding_rule`
- `usage_warnings`

Formula fields describe business semantics. They must not contain executable SQL or a benchmark-specific recipe.

### Warning

Required payload fields:

- `applies_to`
- `severity`
- `rule`
- `safe_alternative`

### Glossary term

Required payload fields:

- `term`
- `definition`
- `excluded_meanings`

## Explicit navigation edges

The compatible navigation graph derives edges only from these fields:

| Source field | Relation |
| --- | --- |
| `discovery.related_knowledge_ids` | `related_to` |
| `payload.source_knowledge_ids` | `sourced_from` |
| `payload.required_knowledge_ids` | `requires` |
| `payload.table_ids` | `contains` |
| `payload.table_id` | `belongs_to` |
| `payload.column_id` | `describes` |
| `payload.default_time_field_id` | `default_time_field` |
| `payload.applies_to` | `applies_to` |
| `payload.from.table_id` | `from_table` |
| `payload.to.table_id` | `to_table` |

Every target must be another existing card. Navigation edges must be explicit and reviewable; titles, summaries, embeddings, and filename proximity do not create persisted edges.

## Source registry

Use a root-level source registry:

```yaml
registry_version: 0.1.0
database_id: acme
updated_at: '2026-08-20T00:00:00Z'
sources:
  - source_id: source.acme.schema_snapshot
    source_type: database_metadata_snapshot
    title: Production analytics metadata snapshot
    project_relative_path: physical/schema_snapshot.json
    snapshot_id: snapshot.acme.20260820
    access_policy: read_only
    trust_role: observed_evidence
    notes: Proves physical objects and types, not business meaning.
```

An evidence reference may cite only a registered source whose `trust_role` is not `no_evidence`.

## Manifest

Record at least:

- logical database ID;
- physical database/schema identity;
- engine;
- package and contract version;
- language and lifecycle status;
- snapshot identity;
- card counts by type and total;
- excluded artifacts and unresolved scope.

The manifest is a handoff summary. Card YAML and source evidence remain authoritative.
