# SDNF Payment Schema Upgrade Tracking Strategy v15

Generated: 2026-05-07T12:26:02+05:30

## Purpose

This package extends the v14 payment schema descriptors with explicit **future specification tracking** metadata. The goal is to handle cases such as: *Amex/PCI/card-data requirements change next month*, *Stripe adds a new PaymentIntent field*, *Razorpay changes currency subunit behavior*, *ISO 20022 publishes a new pacs.008 version*, or *NPCI releases a UPI circular affecting payload structure*.

## Core governance pattern

```text
watched provider/standard source
        ↓
change detected in docs/spec/changelog/circular
        ↓
candidate_schema_delta.json
        ↓
SDNF normal-form gates
        ↓
accepted / rejected / deferred / forked decision
        ↓
append-only SRS lineage
        ↓
payload regression + compliance report
        ↓
approved schema version promoted for runtime use
```

## Why this matters for the paper

This turns DBNF and lineage from an unused concept into a real production-like mechanism. The Master Payment SRS should not be a static JSON blob. It should be a versioned semantic geometry whose canonical nodes evolve only through justified deltas.

## Files in this package

- `*.schema.json`: SDNF schema descriptors with `spec_monitoring`, `upgrade_governance`, `upgrade_tracking_state`, and field-level `change_policy`.
- `SCHEMA_MANIFEST.json`: overall manifest and upgrade decision matrix.
- `UPGRADE_TRACKING_STRATEGY.md`: this explanation.
- `candidate_delta_template.json`: template for future changes discovered from provider docs.
- `srs_upgrade_workflow.json`: machine-readable workflow for the experiment code.

## How to use in the next experiment

1. Put approved schemas under `data/schemas_current/`.
2. Keep original payment examples under `payloads_regression/`.
3. When a source changes, create `candidate_delta_template.json` filled with provider, source, old field, new field, and evidence.
4. Run SDNF gates over the delta.
5. Append decision to `srs_lineage.csv/jsonl`.
6. Only promote a new schema to `schemas_current/` if the decision is accepted or explicitly reviewed.

## Recommended implementation outputs

```text
schema_change_report_v15.csv
candidate_schema_deltas_v15.jsonl
srs_upgrade_lineage_v15.jsonl
payload_regression_report_v15.csv
master_payment_srs_v15.audit.json
master_payment_srs_v15.compact.json
master_payment_srs_v15.graph.json
```

## Precision-safety rule

A future provider field must not be merged merely because it has a familiar suffix such as `id`, `status`, `date`, `amount`, or `method`. It must pass typed semantic-family compatibility and role/partition checks.
