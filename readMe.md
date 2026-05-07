# SDNF Unified Experiment — Schema-First Master Payment SRS Harness v14

This repository contains a single-file, reproducible experiment for validating the research framework:

**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**

The main experiment file for this version is:

```text
unified_sdnf_experiment_hybrid_v14.py
```

Version 14 upgrades the earlier v13 reviewer-grade audit harness from a primarily flat JSON / pairwise alias-matching experiment into a **schema-first, payload-evidenced, SDNF-governed Master Payment SRS evolution benchmark**.

The central v14 narrative is:

```text
explicit payment-type/provider schema descriptors
        +
production/sample payload evidence
        +
SDNF normal-form governance
        ↓
governed Master Payment SRS semantic geometry
        ↓
explainable pre-payment payload compliance decisions
```

In short, v14 treats provider/payment-type schema descriptors as the **intended semantic contracts**, while payload files are used as **empirical evidence** for validation, enrichment, drift/delta detection, and compliance audit.

---

## 1. What Changed in v14

v14 introduces the following major changes over v13:

- **Schema-first Master Payment SRS construction** from explicit payment-type/provider schema descriptors.
- **Payload-evidenced validation**, where payloads are used as observed evidence, not as the authoritative schema source.
- **Payment-domain semantic geometry**, with canonical SRS nodes spanning heterogeneous payment rails and providers.
- **Runtime-style payload compliance decisions** before payment initiation.
- **Candidate schema delta detection** for payload fields not declared in explicit schema descriptors.
- **Typed semantic-family safety** to prevent unsafe generic merges.
- **Role-sensitive RRNF gates** to prevent payer/payee, debtor/creditor, customer/merchant, and similar role mistakes.
- **Rail-aware CMNF checks**, where the Payment domain is modeled through rails/subdomains such as card, ISO20022 transfer, PSP gateway, open banking, and UPI.
- **Partition-aware PONF checks** to keep identifiers, methods, statuses, temporal fields, amounts, card fields, account fields, parties, merchants, and metadata separated.
- **Human-readable compact SRS output** in JSON and Markdown.
- **Graph/explainability output** in JSON and standalone HTML.
- **v14 artifact exports** for dataset summary, schema ingestion audit, payload profiling, field presence, unexpected fields, missing required fields, decisions, SRS mapping, lineage, conflicts, compliance, normal-form summary, and claim support.

The key design shift is that v14 no longer says:

```text
Walk all JSON files and infer a schema from observed fields.
```

Instead, v14 says:

```text
Use explicit schemas as contracts, use payloads as evidence, and use SDNF normal forms to govern SRS convergence and compliance.
```

---

## 2. Main Experiment File

Use this file as the v14 experiment harness:

```text
unified_sdnf_experiment_hybrid_v14.py
```

This script is intended to run as a single-file experiment with no mandatory external service dependencies.

If `sentence-transformers` is available, the configured embedding model is used. If it is not available, v14 uses a deterministic hashing-based embedding fallback so that the experiment remains executable offline.

---

## 3. Recommended Repository Structure

Recommended project layout:

```text
.
├── unified_sdnf_experiment_hybrid_v14.py
├── readMe.md
├── requirements.txt
├── data/
│   ├── INAmex.schema.json
│   ├── ISO20022.schema.json
│   ├── Mastercard.schema.json
│   ├── Plaid.schema.json
│   ├── PPVisa.schema.json
│   ├── Razorpay.schema.json
│   ├── Stripe.schema.json
│   └── UPI.schema.json
├── payloads/
│   └── payment/
│       ├── INAmex/
│       │   ├── INAmex_payload_01.json
│       │   ├── INAmex_payload_02.json
│       │   └── ...
│       ├── ISO20022/
│       │   ├── ISO20022_payload_01.json
│       │   ├── ISO20022_payload_02.json
│       │   └── ...
│       ├── Mastercard/
│       ├── Plaid/
│       ├── PPVisa/
│       ├── Razorpay/
│       ├── Stripe/
│       └── UPI/
├── ground_truth_aliases_closed_world_v12.json
└── controlled_drift_cases.json
```

The schema files under `data/` are the authoritative schema descriptors. The payload files under `payloads/payment/<PaymentType>/` are sample or production-like payload evidence.

---

## 4. Payment Types and Rails Modeled in v14

v14 models the Payment domain using payment types/providers and their rails/subdomains:

| Payment type | Default rail / subdomain |
|---|---|
| `INAmex` | `card_payment` |
| `PPVisa` | `card_payment` |
| `Mastercard` | `card_network_iso8583` |
| `ISO20022` | `account_to_account_credit_transfer` |
| `Plaid` | `open_banking` |
| `Razorpay` | `psp_gateway` |
| `Stripe` | `psp_gateway` |
| `UPI` | `upi` |

These rails help CMNF evaluate whether a merge is within the same rail or is a safe cross-rail convergence, such as amount or currency.

---

## 5. Schema Descriptor Expectations

Each schema file should be named:

```text
<PaymentType>.schema.json
```

Examples:

```text
INAmex.schema.json
Stripe.schema.json
ISO20022.schema.json
UPI.schema.json
```

A schema descriptor may contain:

```json
{
  "schema_id": "Stripe_PaymentIntent_v1",
  "schema_descriptor_version": "v1.0",
  "domain": "payments",
  "rail": "psp_gateway",
  "provider": "Stripe",
  "entity": "PaymentIntent",
  "version": "v1.0",
  "schema_source": "internal_contract",
  "review_status": "reviewed",
  "spec_monitoring": {},
  "upgrade_governance": {},
  "attributes": [
    {
      "name": "payment_intent_id",
      "type": "string",
      "required": true,
      "semantic_family": "identifier:payment_intent",
      "canonical_hint": "payment_intent_identifier",
      "role": "object_identifier",
      "aliases": ["id", "paymentIntentId", "pi_id"],
      "constraints": {
        "pattern": "^pi_.*"
      },
      "description": "Stripe PaymentIntent identifier",
      "merge_policy": {
        "do_not_merge_with_families": [
          "payment:method",
          "payment:status",
          "temporal"
        ]
      }
    }
  ]
}
```

v14 is robust when optional fields are missing, but the strongest experiment results come from including:

```text
name
type
required
semantic_family
canonical_hint
role
aliases
constraints
description
merge_policy.do_not_merge_with_families
```

---

## 6. Payload Evidence Expectations

Payload files should live under:

```text
payloads/payment/<PaymentType>/*.json
```

Example:

```text
payloads/payment/Stripe/Stripe_payload_01.json
payloads/payment/Stripe/Stripe_payload_02.json
payloads/payment/UPI/UPI_payload_01.json
payloads/payment/UPI/UPI_payload_02.json
```

Payloads are not treated as authoritative schemas. They are used to provide evidence such as:

- observed field presence,
- observed value type,
- value-shape pattern,
- regex-like pattern,
- missing required fields,
- unexpected fields,
- candidate schema deltas,
- payload compliance decisions.

v14 computes field presence per payment type. A typical interpretation is:

```text
6/6 fields  -> required candidate
4-5/6       -> conditional or strong optional candidate
2-3/6       -> optional or method-specific candidate
1/6         -> outlier or low-confidence candidate
```

---

## 7. SDNF Normal Forms in v14

v14 reports the following SDNF normal forms.

### 7.1 EENF — Embedding Existence / Embedding Stability Normal Form

EENF evaluates embedding availability and stability. v14 supports an embedding stability diagnostic using repeated deterministic regenerations or model encodings.

If `sentence-transformers` is unavailable, v14 uses deterministic hashing embeddings so the artifact can still run offline.

### 7.2 AANF — Attribute Alias Normal Form

AANF evaluates whether schema attributes are admissible aliases of the same canonical concept.

Signals include:

```text
same canonical_hint
same semantic_family
schema-declared alias
name similarity
embedding similarity
```

AANF is intentionally not allowed to override hard semantic-family, role, rail, or partition vetoes.

### 7.3 ECNF — Evidence Completeness Normal Form

ECNF checks whether there is enough independent evidence to support a merge or compliance decision.

Evidence includes:

```text
schema semantics
canonical_hint
semantic_family
role
aliases
name similarity
embedding similarity
payload field evidence
value-shape evidence
constraint evidence
```

Payload evidence supports ECNF, but payload evidence does not override hard semantic conflicts.

### 7.4 RRNF — Role-Respecting Normal Form

RRNF prevents unsafe role merges such as:

```text
payer != payee
debtor != creditor
customer != merchant
method != identifier
routing number != account number
```

Role conflicts are hard vetoes for automatic merge acceptance.

### 7.5 CMNF — Contextual / Rail Merge Normal Form

CMNF is active in v14 because the Payment domain is modeled through rails such as:

```text
card_payment
card_network_iso8583
account_to_account_credit_transfer
open_banking
psp_gateway
upi
```

Cross-rail convergence is allowed only for safe global concepts such as:

```text
payment:amount
payment:currency
```

when explicitly enabled through:

```text
--allow_cross_rail_amount_currency
```

### 7.6 DBNF — Drift / Delta Boundary Normal Form

DBNF in v14 focuses on schema/payload delta detection and future schema evolution.

It detects:

```text
payload fields not declared in schema descriptors
missing required fields
unexpected fields
schema-vs-payload mismatch
candidate schema deltas
```

v14 exports candidate schema deltas in JSONL and CSV formats.

### 7.7 PONF — Partition Orthogonality Normal Form

PONF protects semantic partitions. It prevents broad over-merging across partitions such as:

```text
identifier
payment:method
payment:status
temporal
payment:amount
payment:currency
payment_card
payment_account
party
merchant
metadata
```

This is one of the main v14 fixes for the v13 issue where unrelated identifiers and method-like fields could be grouped too broadly.

---

## 8. Critical v14 Semantic Safety Rules

v14 applies hard semantic vetoes before accepting merge decisions.

Important examples:

```text
identifier:* must not merge with payment:method
identifier:* must not merge with payment:status
identifier:* must not merge with temporal:*
temporal:* must not merge with payment:status
routing number must not merge with account number
card PAN must not merge with bank account fields
payer must not merge with payee
debtor must not merge with creditor
metadata must not merge with business attributes
different identifier subtypes must remain separate
```

Identifier examples that should remain separate unless explicitly and safely modeled:

```text
message_identifier
end_to_end_identifier
transaction_identifier
payment_intent_identifier
razorpay_payment_identifier
order_identifier
customer_identifier
plaid_account_identifier
card_acceptor_identifier
schema_identifier
```

This is central to the precision-first v14 design.

---

## 9. Setup

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Minimal Dependency Note

The script primarily uses Python standard library modules. It optionally uses:

```text
numpy
sentence-transformers
```

If `sentence-transformers` is unavailable, the script uses a deterministic hashing fallback for embeddings.

---

## 10. Recommended v14 Audit Run

Use this for the full reviewer/audit run:

```bash
python unified_sdnf_experiment_hybrid_v14.py \
  --profile audit \
  --schemas_dir data \
  --schema_glob "*.schema.json" \
  --payloads_root payloads/payment \
  --seed_srs_schema INAmex.schema.json \
  --schema_first \
  --payload_evidence \
  --build_master_srs \
  --validate_payloads \
  --strict_semantic_vetoes \
  --precision_first \
  --allow_cross_rail_amount_currency \
  --unknown_field_policy defer \
  --evidence_mode sdnf_hybrid \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy paper \
  --measure_timing \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20
```

This run produces the core v14 artifacts, including SRS, graph, compliance, and normal-form outputs.

---

## 11. Recommended v14 Paper-Facing Run

Use this for a concise paper-facing run:

```bash
python unified_sdnf_experiment_hybrid_v14.py \
  --profile paper \
  --schemas_dir data \
  --schema_glob "*.schema.json" \
  --payloads_root payloads/payment \
  --schema_first \
  --payload_evidence \
  --build_master_srs \
  --validate_payloads \
  --strict_semantic_vetoes \
  --precision_first \
  --allow_cross_rail_amount_currency \
  --unknown_field_policy defer \
  --evidence_mode sdnf_hybrid \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy paper \
  --measure_timing
```

---

## 12. Important v14 Outputs

v14 exports the following files by default in paper/audit/dev profiles.

### Dataset and ingestion outputs

```text
summary_v14.json
dataset_summary_v14.csv
schema_ingestion_audit_v14.csv
payload_profile_v14.csv
field_presence_report_v14.csv
```

### Payload validation and schema delta outputs

```text
unexpected_fields_v14.csv
missing_required_fields_v14.csv
candidate_schema_deltas_v14.jsonl
candidate_schema_deltas_v14.csv
payload_compliance_v14.csv
payload_compliance_v14.json
payload_compliance_summary_v14.csv
```

### Decision and evaluation outputs

```text
decisions_v14.csv
predicted_pairs_v14.json
false_positives_v14.csv
false_negatives_v14.csv
fp_root_causes_v14.csv
fn_root_causes_v14.csv
alias_confusion_v14.csv
```

### SRS outputs

```text
srs_evolved_schema_v14.audit.json
srs_evolved_schema_v14.compact.json
srs_evolved_schema_v14.md
srs_evolved_schema_v14.graph.json
srs_evolved_schema_v14.graph.html
srs_attribute_mapping_v14.csv
srs_lineage_v14.csv
srs_upgrade_lineage_v14.jsonl
srs_conflicts_v14.csv
```

### Normal-form, claim, and timing outputs

```text
normal_form_summary_v14.csv
claim_support_summary_v14.csv
timing_summary_v14.csv
dbnf_summary_v14.csv
```

---

## 13. Key v14 Output Meanings

### `srs_evolved_schema_v14.audit.json`

Full reproducibility and audit output. Includes:

```text
run configuration
dataset summary
embedding backend
schemas
canonical nodes
decisions
conflicts
payload compliance records
candidate schema deltas
```

### `srs_evolved_schema_v14.compact.json`

Reviewer-friendly SRS output. Each canonical node includes:

```text
node
meaning
semantic_family
role
domain
rails
providers
members
payload_evidence_summary
normal_forms
decision_summary
lineage_summary
accepted_aliases
rejected_near_misses
deferred_candidates
```

### `srs_evolved_schema_v14.md`

Human-readable explanation of the Master Payment SRS, organized for paper/reviewer inspection.

### `srs_evolved_schema_v14.graph.json`

Graph-format SRS representation with node and edge types.

Node types include:

```text
domain
rail
provider_schema
raw_attribute
canonical_srs_node
payload_file
compliance_decision
```

Edge types include:

```text
contains
defines
maps_to
compliant_with
non_compliant_with
```

### `srs_evolved_schema_v14.graph.html`

Standalone HTML explainability view. It does not require an external CDN.

### `payload_compliance_v14.csv` and `payload_compliance_v14.json`

Payload-level pre-payment compliance decisions.

Possible decisions:

```text
ALLOW
REJECT
DEFER_REVIEW
ROUTE_SCHEMA_ONBOARDING
```

### `candidate_schema_deltas_v14.jsonl`

Payload-observed fields that are not declared in explicit schema descriptors. These are treated as candidate deltas, not automatic SRS members.

---

## 14. Runtime-Style Payload Compliance Logic

For each payload, v14:

1. Identifies payment type from the folder name.
2. Loads the matching schema descriptor.
3. Maps payload fields to schema attributes using exact name, normalized name, aliases, and canonical hints.
4. Checks required fields.
5. Checks value constraints when available.
6. Detects unexpected fields.
7. Applies normal-form interpretation.
8. Produces a compliance decision.

Decision interpretation:

```text
ALLOW
  Required fields present, mapped fields safe, and no critical unexpected fields.

REJECT
  Required fields missing or hard constraint violation detected.

DEFER_REVIEW
  Payload is mostly understandable but includes unexpected fields needing review.

ROUTE_SCHEMA_ONBOARDING
  Payload includes many unknown fields, suggesting a new or changed schema contract.
```

---

## 15. Ground Truth Format

v14 supports the same general ground-truth style used in v12/v13.

Example:

```json
{
  "closed_world": true,
  "alias_groups": [
    {
      "canonical": "payment_amount",
      "aliases": ["amount", "transaction_amount", "txn_amount", "instd_amt"],
      "basis": "Payment amount fields across payment schemas."
    },
    {
      "canonical": "payment_currency",
      "aliases": ["currency", "transaction_currency", "iso_currency_code"],
      "basis": "Payment currency fields across schemas."
    }
  ],
  "true_pairs": [
    ["transaction_id", "txn_id"]
  ],
  "negative_pairs": [
    ["payment_intent_id", "payment_method"],
    ["transaction_status", "transaction_timestamp"],
    ["account_number", "routing_number"],
    ["debtor_account", "creditor_account"]
  ]
}
```

v14 predicted pairs are derived from accepted canonical-node co-membership.

---

## 16. Embeddings in v14

v14 uses embeddings for:

```text
schema attribute comparison
AANF semantic similarity evidence
EENF stability diagnostic
```

Embedding input typically combines:

```text
attribute name
semantic family
role/context
```

v14 intentionally does not let embedding similarity alone decide merges. This is important because payment identifiers, methods, statuses, and timestamps can be semantically close in generic embedding space but should remain separated by SDNF normal forms.

---

## 17. HNSW / ANN Scale Note

The current v14 experiment does not require HNSW for the small Payment benchmark. Candidate generation is deterministic and pruned using schema-aware signals such as:

```text
same canonical hint
same semantic family
name similarity
```

For larger enterprise-scale SRS construction, an optional HNSW or ANN candidate retrieval layer can be added in a future version:

```text
attribute embeddings
  -> HNSW top-k candidate retrieval
  -> SDNF hard vetoes
  -> AANF / ECNF / RRNF / CMNF / DBNF / PONF checks
  -> merge / reject / defer / fork
```

Important: HNSW should be used only for scalable candidate retrieval. It should not directly decide semantic merges.

---

## 18. v14 Integrity Rules

v14 follows these rules:

- Schema descriptors are authoritative semantic contracts.
- Payloads are empirical evidence, not the schema source of truth.
- Unknown payload fields become candidate schema deltas.
- Payload-inferred fields are not auto-merged into the Master SRS by default.
- Semantic-family hard vetoes run before merge acceptance.
- Role conflicts are hard safety blockers.
- Identifier subtypes remain separated unless explicitly and safely modeled.
- Metadata fields must not merge with business attributes.
- Payment method/status/temporal/identifier partitions remain separated.
- SRS outputs are generated in full audit, compact, Markdown, graph JSON, and graph HTML forms.
- Payload compliance decisions are exported for auditability.

---

## 19. Research Integrity Note

This artifact is intended for research, reproducibility, and reviewer validation.

For production use, avoid storing raw sensitive payment values in logs or outputs. Replace raw-value evidence with privacy-preserving summaries, hashed values, masked values, or aggregated statistics.

The provided synthetic payload examples should be treated as lab/test data only.

---

## 20. Summary

v14 upgrades the SDNF experiment into a production-close Payment-domain semantic governance benchmark.

The experiment now demonstrates:

```text
heterogeneous payment schema descriptors
  -> governed canonical Master Payment SRS
  -> payload evidence attachment
  -> candidate schema delta detection
  -> normal-form-governed convergence
  -> human-readable SRS outputs
  -> graph explainability
  -> pre-payment payload compliance decisions
```

This is the recommended version for demonstrating how SDNF can standardize heterogeneous payment schemas into a governed semantic geometry while preserving role safety, partition safety, schema lineage, and compliance explainability.
