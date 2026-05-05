# SDNF Unified Experiment (Reviewer-Grade Audit Harness - v10)

This repository contains a **single-file, reproducible experiment** for validating:

> **Semantic Data Normal Forms (SDNF)**  
> from the paper:  
> *"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces"*

---

## What makes v10 different

Version 10 is a **reviewer-grade audit harness**, designed to:

- Compute all metrics from observed data
- Validate claims with `PASS`, `FAIL`, or `NOT MEASURABLE`
- Support ground-truth evaluation for precision, recall, F1, and leakage
- Provide drift detection validation
- Enable pairwise explainability tracing
- Measure timing and stability/scalability behavior
- Support ablation studies across SDNF components

---

## Core Capabilities

### 1. Evidence-Based Schema Normalization

The experiment evaluates candidate attribute merges using multiple signals:

- Embedding similarity
- Name similarity using token overlap
- Ontology-root alignment
- Regex / value-format matching
- Value Semantic Signature (VSS)
- Shape-based similarity

---

### 2. SDNF Validation Layers

The script reports validation status for the following Semantic Data Normal Forms:

- **EENF** — Entity Embedding Normal Form / embedding stability
- **AANF** — Attribute Alias Normal Form / alias similarity thresholding
- **CMNF** — Context Modulation Normal Form / context and ontology conflict control
- **ECNF** — Evidence Completeness Normal Form / evidence sufficiency
- **DBNF** — Drift-Bounded Normal Form / embedding drift robustness
- **RRNF** — Role-Respecting Normal Form, currently reported as not exercised unless extended
- **PONF** — Partition Orthogonality Normal Form, currently reported as not exercised unless extended

Each normal form is reported with observed values, expected thresholds, and a status.

---

### 3. Claim Verification Engine

The v10 harness prints reviewer-facing tables for:

- Schema reduction percentage
- Precision / Recall / F1
- Cross-context leakage rate
- Drift detection metrics
- EENF variance reduction sweep
- Merge-decision timing metrics
- Claim support summary

Each relevant paper claim is labeled as:

```text
PASS / FAIL / NOT MEASURABLE
```

If required evidence such as ground truth is missing, the script marks the metric as `NOT MEASURABLE` rather than inventing results.

---

## Ground Truth Support

### Alias Ground Truth Format

The script supports alias groups:

```json
{
  "alias_groups": [
    ["acct_num", "PrimaryAccountNumber", "pan"],
    ["txn_amount", "amount"]
  ],
  "negative_pairs": [
    ["card", "playing_card"]
  ]
}
```

It also supports explicit true pairs:

```json
{
  "true_pairs": [
    ["acct_num", "PrimaryAccountNumber"]
  ],
  "negative_pairs": [
    ["account", "merchant_account"]
  ]
}
```

The script converts alias groups into unordered normalized true pairs before computing precision, recall, and F1.

---

### Drift Ground Truth Format

```json
{
  "drift_attributes": [
    "description",
    "iso_currency_code",
    "acct_num"
  ]
}
```

When drift ground truth is provided, the script compares DBNF drift hotspots against known simulated drift attributes and computes drift precision, recall, F1, and accuracy where definable.

---

## Repository Structure

```text
unified_sdnf_experiment_hybrid_v10.py   # Main reviewer-grade experiment harness
readMe.md                              # This file
requirements.txt                       # Python dependencies
data/                                  # Schema JSON files
payloads/                              # Payload JSON files with values
```

---

## Setup

### 1. Create and activate a virtual environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Default hybrid SDNF run

```bash
python unified_sdnf_experiment_hybrid_v10.py
```

---

### V11 Debug SDNF run

```bash
python unified_sdnf_experiment_hybrid_v11.py \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases.json \
  --drift_ground_truth drift_ground_truth.json \
  --trace_pair acct_num PrimaryAccountNumber \
  --trace_pair primary_account_number account_number \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing \
  --export_decisions decisions_v11.csv \
  --export_predicted_pairs predicted_pairs_v11.json \
  --export_false_positives false_positives_v11.csv \
  --export_false_negatives false_negatives_v11.csv \
  --export_ground_truth_template ground_truth_aliases_template_v11.json \
  --export_summary_json summary_v11.json
```

---

### Run all reviewer-grade modes and ablations

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode all
```

---

### Run with alias ground truth validation

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode hybrid \
  --ground_truth_aliases ground_truth_aliases.json
```

---

### Run full audit with alias ground truth, drift ground truth, pair trace, EENF sweep, and timing

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases.json \
  --drift_ground_truth drift_ground_truth.json \
  --trace_pair acct_num PrimaryAccountNumber \
  --eenf_g_sweep 1,10,20 \
  --measure_timing
```

---

### Run DBNF drift simulation using a second embedding model

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode all \
  --drift_model all-mpnet-base-v2 \
  --drift_ground_truth drift_ground_truth.json
```

---

### Run pairwise explainability trace

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --trace_pair acct_num PrimaryAccountNumber
```

Multiple trace pairs can be supplied by repeating `--trace_pair`:

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --trace_pair acct_num PrimaryAccountNumber \
  --trace_pair txn_amount amount
```

---

### Run EENF stability-latency sweep

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --eenf_g_sweep 1,10,20
```

---

### Run timing instrumentation

```bash
python unified_sdnf_experiment_hybrid_v10.py \
  --measure_timing
```

---

## Supported Evidence / Ablation Modes

When `--evidence_mode all` is used, the script runs the following reviewer-grade modes:

| Mode | Description |
|---|---|
| `embed_only_baseline` | Embedding-only baseline. Uses only embedding cosine threshold. |
| `sdnf_hybrid` | Full SDNF-style hybrid evidence mode. |
| `no_ecnf` | Disables evidence completeness gating. |
| `no_cmnf` | Disables context / ontology conflict gating. |
| `no_dbnf` | Disables drift robustness handling while preserving merge rules. |
| `no_value_evidence` | Removes VSS and shape evidence. |
| `vss_only` | Uses value semantic signature evidence only. |
| `shape_only` | Uses value shape evidence only. |
| `name_ontology_only` | Uses name similarity and ontology-root matching. |
| `hybrid` | Backward-compatible hybrid mode. |

---

## Main Output Tables

The script prints the following reviewer-facing tables:

### `RUN CONFIGURATION`

Reports seed, model, evidence mode, thresholds, input directories, and optional ground-truth files.

### `DATASET SUMMARY`

Reports schema files, payload files, raw attribute records, distinct attribute names, value evidence availability, and missing evidence fraction.

### `NORMAL FORM VALIDATION SUMMARY`

Reports EENF, AANF, CMNF, ECNF, DBNF, RRNF, and PONF status.

### `ALIAS MERGE EVALUATION AGAINST GROUND TRUTH`

Reports:

- Predicted pairs
- True pairs
- TP / FP / FN
- Precision
- Recall
- F1

If ground truth is absent, the table marks the relevant values as `NOT MEASURABLE`.

### `CROSS-CONTEXT LEAKAGE EVALUATION`

Reports leakage count, predicted merge count, leakage rate, and representative leakage examples.

### `ABLATION STUDY SUMMARY`

Compares all evidence and ablation modes.

### `PAPER TABLE 2 REPRODUCTION CHECK`

Checks whether the measured values reproduce the paper-aligned claims within configured tolerances.

### `EENF STABILITY-LATENCY SWEEP`

Printed when `--eenf_g_sweep` is supplied.

Reports:

- G
- Mean variance
- q95 variance
- Max variance
- Variance reduction versus G=1
- Encoding time
- Overhead versus G=1

### `MERGE DECISION TIMING SUMMARY`

Reports:

- Candidate-pair count
- Mean latency
- P50 latency
- P95 latency
- P99 latency
- Max latency

It also prints whether average merge decision latency is under 50 ms.

### `PAIRWISE MERGE EVIDENCE TRACE`

Printed when `--trace_pair` is supplied.

Reports pair-level details including:

- Sources
- Contexts
- Cosine similarity
- Name similarity
- Ontology roots
- Ontology match
- Value co-occurrence
- Regex match
- VSS similarity
- Shape similarity
- Aggregate score
- Evidence signal count
- AANF / ECNF / CMNF status
- Final decision
- Lineage ID

### `DBNF DRIFT HOTSPOTS`

Printed when `--drift_model` is supplied.

Reports top drift attributes and drift magnitude.

### `DBNF DRIFT DETECTION EVALUATION`

Printed when drift evaluation is run.

Reports:

- Drift threshold
- Detected count
- True drift count
- TP / FP / FN
- Precision
- Recall
- F1
- Accuracy if definable

### `CLAIM SUPPORT SUMMARY`

Final audit table that maps each paper claim to:

- Measured value
- Expected value
- Status
- Evidence table

---

## Data Expectations

### Schema-style JSON

```json
{
  "schema_id": "INAmex_v1",
  "attributes": [
    {"name": "PrimaryAccountNumber"},
    {"name": "ExpirationDate"}
  ]
}
```

---

### Payload-style JSON

```json
{
  "pan": "4111111111111111",
  "exp": "12/26",
  "cvv": "123"
}
```

Nested JSON is also supported. The script walks dictionaries and lists recursively to collect fields and values.

---

## Important Integrity Rules

- The script does not tune results to force paper claims to pass.
- If ground truth is missing, precision, recall, and drift metrics are marked `NOT MEASURABLE`.
- If measured values differ from paper claims, the script reports the measured values and marks claims as `FAIL`.
- The output tables are intended to help revise the paper based on measurable evidence.

---

## Dependency and Fallback Behavior

- If `sentence-transformers` is installed, the requested embedding model is used.
- If `sentence-transformers` is unavailable, deterministic hashing embeddings are used as a fallback and the run configuration reports `hashing-fallback`.
- If ANN infrastructure is unavailable, exhaustive candidate-pair scoring can still support the reviewer audit path.

---

## Research Integrity Note

This artifact is intended for research, reproducibility, and reviewer validation. Avoid embedding or logging raw sensitive values in production environments. For production use, replace raw-value evidence with privacy-preserving summaries or hashed/aggregated statistics.

---

## Summary

Version 9 transforms the SDNF experiment from a basic consolidation run into a **measurable, auditable, falsifiable reviewer-grade experiment harness**. It is designed so that every major empirical claim in the SDNF paper can be confirmed, rejected, or marked not measurable based on explicit evidence.
