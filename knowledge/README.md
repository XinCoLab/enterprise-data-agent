# Cold Chain Pharma Compliance Knowledge Base

This English directory-based package covers the complete public Schema, Column Meaning, and HKB for `cold_chain_pharma_compliance`.

## Contents

- `physical/database.yaml`: database card.
- `physical/tables/*.yaml`: 12 table cards and 74 physical column cards.
- `business/relationships.yaml`: 10 relationships directly declared by public foreign keys.
- `business/metrics.yaml`: 17 calculation records mapped from the complete public HKB.
- `business/glossary.yaml`: 43 public domain-rule and value-illustration records.
- `physical/schema.json`: parsed public physical schema inventory.
- `sources/source_registry.yaml` and `source_mapping.json`: source hashes and per-card provenance.

JSONB nested paths remain inside their real physical column cards; they are not represented as physical columns. Public HKB calculations preserve the official mathematical definitions. Their SQL implementations and physical-column bindings remain explicitly unresolved rather than inferred.
