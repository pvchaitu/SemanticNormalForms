# SDNF Unified Experiment — Reviewer-Grade Audit Harness v13

This repository contains a single-file, reproducible experiment for validating the research framework:

**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**

The main experiment file for this version is:

```text
unified_sdnf_experiment_hybrid_v13.py
```

Version 13 evolves the earlier v10/v12 audit harness into a cleaner, more defensible, reviewer-facing experiment artifact. It preserves the successful capabilities from prior versions while correcting the v12 issues around explicit-negative merges, absent ground-truth handling, metadata filtering, DBNF interpretation, console verbosity, and missing SRS output.

---

## 1. What Changed in v13

v13 introduces the following major improvements:

- **Explicit-negative hard veto** before bridge rules and final merge acceptance.
- **Role-sensitive bridge guards** to prevent unsafe account/role merges.
- **Canonical-equivalence AANF pass** for pairs that normalize to the same canonical key.
- **Absent-ground-truth handling** so unavailable attributes do not incorrectly count as false negatives by default.
- **Paper / audit / dev profiles** to control output verbosity.
- **Metadata filtering policy** so paper-mode metrics are not polluted by schema metadata fields.
- **SRS evolved schema export** so the experiment produces an actual Semantic Representation Schema, not only pairwise merge decisions.
- **Corrected DBNF semantics** separating same-family version drift, controlled drift, and cross-backbone migration diagnostics.
- **Cleaner claim-support tables** that distinguish measured results from paper claims.
- **Valid CSV/JSON exports** for reviewer reproducibility and debugging.

---

## 2. SDNF Normal Forms in v13

v13 uses the following normal-form taxonomy.

### EENF — Embedding Existence / Embedding Stability Normal Form

EENF evaluates whether embeddings remain stable under repeated encoding or regeneration.

Typical evidence:

```text
q95 embedding variance
max embedding variance
variance reduction at G=10 or G=20
```

### AANF — Attribute Alias Normal Form

AANF evaluates whether two attributes are semantically admissible as aliases.

Signals include:

```text
embedding cosine similarity
name similarity
canonical synonym equivalence
ontology-root compatibility
```

v13 adds canonical-equivalence handling. For example, if `pan` and `primary_account_number` normalize to the same canonical key, AANF can pass through:

```text
CANONICAL_EQUIVALENCE_PASS
```

unless an explicit negative, role conflict, or context conflict blocks it.

### ECNF — Evidence Completeness Normal Form

ECNF evaluates whether enough independent evidence signals support a merge.

Signals include:

```text
name similarity
ontology match
value co-occurrence
regex compatibility
value semantic signature similarity
shape similarity
aggregate score
evidence signal count
```

### CMNF — Contextual Merge Normal Form

CMNF governs context/domain/business-boundary safety.

If only one context is present, v13 marks cross-context CMNF as:

```text
NA_SINGLE_CONTEXT / NOT_EXERCISED
```

not as a failure.

### DBNF — Drift Boundary Normal Form

DBNF governs model-version or representation drift.

v13 separates DBNF into:

```text
DBNF-V: same-family model-version/checkpoint drift
DBNF-M: cross-backbone migration diagnostic
Controlled DBNF: claim-bearing controlled drift benchmark
```

A cross-backbone comparison such as:

```text
all-MiniLM-L6-v2 -> all-mpnet-base-v2
```

is treated as a migration diagnostic by default, not as primary DBNF claim evidence.

### SRS — Semantic Representation Schema

v13 explicitly exports an evolved SRS. The experiment output no longer stops at predicted alias pairs.

---

## 3. Repository Structure

Recommended structure:

```text
unified_sdnf_experiment_hybrid_v13.py   # Main v13 experiment harness
readMe.md                               # This README
requirements.txt                        # Python dependencies
data/                                   # Schema JSON files
payloads/                               # Payload JSON files with values
ground_truth_aliases_closed_world_v12.json
controlled_drift_cases.json             # Optional controlled DBNF input
```

---

## 4. Setup

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

### Dependency and fallback behavior

If `sentence-transformers` is installed, the requested embedding model is used. If it is unavailable, v13 uses deterministic hashing embeddings so that the harness remains executable offline.

---

## 5. Ground Truth Format

v13 supports object-style alias groups:

```json
{
  "closed_world": true,
  "alias_groups": [
    {
      "canonical": "primary_account_number",
      "aliases": ["account_number", "acct_num", "pan"],
      "basis": "Payment account/card-number identifiers."
    },
    {
      "canonical": "transaction_amount",
      "aliases": ["txn_amount", "amount", "instd_amount"],
      "basis": "Payment amount fields across schemas."
    }
  ],
  "true_pairs": [
    ["transaction_id", "txn_id"]
  ],
  "negative_pairs": [
    ["account_number", "routing_number"],
    ["payer_account", "primary_account_number"],
    ["account_type", "debtor_account"]
  ]
}
```

v13 expands alias groups into unordered normalized true pairs and applies explicit negative pairs as hard vetoes for production SDNF modes.

---

## 6. Important v13 Integrity Rules

v13 follows these rules:

- Metrics are computed from current data.
- Paper claims are not hardcoded as successful outcomes.
- Explicit negative pairs cannot be accepted by production SDNF/hybrid modes.
- Absent ground-truth pairs are excluded from the main evaluation by default.
- Cross-backbone DBNF is diagnostic-only unless explicitly allowed.
- Single-context CMNF is marked as not exercised, not failed.
- SRS schema and mapping exports are first-class outputs.
- Paper profile avoids large verbose console logs.

---

## 7. Run Profiles

v13 supports three run profiles.

### Paper profile

Use this for clean paper-facing runs.

```text
--profile paper
```

Default behavior:

- concise console output
- excludes metadata-like fields by default
- exports core summary and SRS artifacts
- avoids large decision tables unless explicitly requested

### Audit profile

Use this for reviewer/debug runs.

```text
--profile audit
```

Default behavior:

- includes full diagnostics
- can include metadata fields
- produces decision logs, FP/FN files, trace rows, bridge summaries, candidate coverage, and DBNF diagnostics

### Dev profile

Use this for local debugging.

```text
--profile dev
```

Default behavior:

- audit outputs plus extra console self-checks

---

## 8. Recommended Paper Run

This is the cleanest run for paper-facing evidence.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile paper \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy paper \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing \
  --export_summary_json summary_v13.json \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_srs_mapping srs_attribute_mapping_v13.csv \
  --export_claim_support_summary claim_support_summary_v13.csv \
  --export_normal_form_summary normal_form_summary_v13.csv \
  --export_alias_confusion alias_confusion_v13.csv
```

Expected primary outputs:

```text
summary_v13.json
srs_evolved_schema_v13.json
srs_attribute_mapping_v13.csv
claim_support_summary_v13.csv
normal_form_summary_v13.csv
alias_confusion_v13.csv
```

---

## 9. Full Reviewer Audit Run

Use this when you need all diagnostics for reviewer analysis.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy audit \
  --trace_pair acct_num PrimaryAccountNumber \
  --trace_pair primary_account_number account_number \
  --trace_pair currency iso_currency_code \
  --trace_pair instd_amt txn_amount \
  --trace_pair amount txn_amount \
  --trace_pair pan account_number \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing \
  --export_summary_json summary_v13.json \
  --export_decisions decisions_v13.csv \
  --export_predicted_pairs predicted_pairs_v13.json \
  --export_false_positives false_positives_v13.csv \
  --export_false_negatives false_negatives_v13.csv \
  --export_ground_truth_pairs ground_truth_pairs_expanded_v13.csv \
  --export_candidate_coverage candidate_coverage_v13.csv \
  --export_alias_confusion alias_confusion_v13.csv \
  --export_absent_ground_truth_pairs absent_ground_truth_pairs_v13.csv \
  --export_fn_root_causes fn_root_causes_v13.csv \
  --export_fp_clusters fp_clusters_v13.csv \
  --export_bridged_merges bridged_merges_v13.csv \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_srs_mapping srs_attribute_mapping_v13.csv \
  --export_srs_lineage srs_lineage_v13.csv \
  --export_srs_conflicts srs_conflicts_v13.csv \
  --export_trace_pairs trace_pairs_v13.csv
```

Expected diagnostic outputs:

```text
decisions_v13.csv
predicted_pairs_v13.json
false_positives_v13.csv
false_negatives_v13.csv
ground_truth_pairs_expanded_v13.csv
candidate_coverage_v13.csv
alias_confusion_v13.csv
absent_ground_truth_pairs_v13.csv
fn_root_causes_v13.csv
fp_clusters_v13.csv
bridged_merges_v13.csv
srs_evolved_schema_v13.json
srs_attribute_mapping_v13.csv
srs_lineage_v13.csv
srs_conflicts_v13.csv
trace_pairs_v13.csv
summary_v13.json
```

---

## 10. Cross-Backbone Migration Diagnostic Run

Use this when replacing one embedding backbone with another, for example due to security, compliance, or deprecation.

This run is diagnostic by default, not primary DBNF claim evidence.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --model all-MiniLM-L6-v2 \
  --drift_model all-mpnet-base-v2 \
  --dbnf_mode migration \
  --migration_reason "security-driven model replacement" \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_dbnf_lineage dbnf_lineage_v13.csv \
  --export_dbnf_forks dbnf_forks_v13.json \
  --export_cross_model_sensitivity cross_model_sensitivity_v13.csv \
  --export_summary_json summary_v13_migration.json
```

Expected outputs:

```text
summary_v13_migration.json
srs_evolved_schema_v13.json
dbnf_lineage_v13.csv
dbnf_forks_v13.json
cross_model_sensitivity_v13.csv
```

Interpretation:

```text
all-MiniLM-L6-v2 -> all-mpnet-base-v2
```

is treated as **DBNF-M / cross-backbone migration diagnostic** unless `--allow_cross_backbone_dbnf_claim` is explicitly supplied.

---

## 11. Same-Family DBNF Version Drift Run

Use this when comparing versions or checkpoints of the same embedding model family.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --model enterprise-embedder-v1 \
  --drift_model enterprise-embedder-v2 \
  --dbnf_mode version \
  --model_family enterprise-embedder \
  --target_model_family enterprise-embedder \
  --base_model_version v1 \
  --target_model_version v2 \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --export_dbnf_summary dbnf_summary_v13.csv \
  --export_dbnf_lineage dbnf_lineage_v13.csv \
  --export_dbnf_forks dbnf_forks_v13.json \
  --export_summary_json summary_v13_dbnf_version.json
```

Use this run when the research claim is about model-version drift rather than model-backbone replacement.

---

## 12. Controlled DBNF Run

Use this when you have controlled drift cases.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --dbnf_mode controlled \
  --controlled_drift_json controlled_drift_cases.json \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --export_dbnf_summary dbnf_summary_v13.csv \
  --export_dbnf_lineage dbnf_lineage_v13.csv \
  --export_dbnf_forks dbnf_forks_v13.json \
  --export_summary_json summary_v13_controlled_dbnf.json
```

Example controlled drift input:

```json
{
  "controlled_drift_cases": [
    {
      "attribute": "description",
      "drifted_name": "merchant narrative text",
      "basis": "Controlled semantic rename"
    },
    {
      "attribute": "payer_name",
      "drifted_name": "debtor identity label",
      "basis": "Controlled role-sensitive drift"
    }
  ]
}
```

---

## 13. Pairwise Trace Run

Use trace pairs to inspect specific merge decisions.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --trace_pair acct_num PrimaryAccountNumber \
  --trace_pair primary_account_number account_number \
  --trace_pair currency iso_currency_code \
  --trace_pair instd_amt txn_amount \
  --trace_pair amount txn_amount \
  --trace_pair pan account_number \
  --export_trace_pairs trace_pairs_v13.csv \
  --export_summary_json summary_v13_trace.json
```

---

## 14. EENF Stability Sweep Only

Use this to focus on embedding stability.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile paper \
  --evidence_mode hybrid \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --export_eenf_sweep eenf_sweep_v13.csv \
  --export_summary_json summary_v13_eenf.json
```

---

## 15. Timing Instrumentation Run

Use this to measure merge-decision latency.

```bash
python unified_sdnf_experiment_hybrid_v13.py \
  --profile paper \
  --evidence_mode all \
  --measure_timing \
  --export_timing_summary timing_summary_v13.csv \
  --export_summary_json summary_v13_timing.json
```

---

## 16. Supported Evidence / Ablation Modes

When `--evidence_mode all` is used, v13 runs:

```text
embed_only_baseline
sdnf_hybrid
no_ecnf
no_cmnf
no_dbnf
no_value_evidence
vss_only
shape_only
name_ontology_only
hybrid
```

Mode meanings:

```text
embed_only_baseline   Embedding-only baseline.
sdnf_hybrid           Full SDNF hybrid evidence mode.
no_ecnf               ECNF ablated.
no_cmnf               CMNF ablated.
no_dbnf               DBNF handling ablated.
no_value_evidence     Removes VSS/shape/value evidence.
vss_only              Uses value semantic signature only.
shape_only            Uses value shape only.
name_ontology_only    Uses name and ontology signals only.
hybrid                Backward-compatible hybrid mode.
```

---

## 17. Main Outputs

### `summary_v13.json`

Top-level JSON summary containing:

```text
run_configuration
dataset_summary
ground_truth_audit
alias_eval_summary
leakage_eval_summary
normal_form_summary
srs_summary
eenf_sweep
timing_summary
dbnf_summary
self_checks
claim_support_summary
```

### `alias_confusion_v13.csv`

Alias precision/recall/F1 by mode.

Includes:

```text
eval_scope
mode
predicted_pairs_count
true_pairs_count
TP
FP
FN
precision
labeled_precision
recall
F1
closed_world
absent_pairs_excluded_count
explicit_negative_veto_count
canonical_equivalence_pass_count
metadata_excluded_count
```

### `srs_evolved_schema_v13.json`

Evolved SRS schema containing:

```text
srs_version
dataset_summary
model
context
ground_truth_source
canonical_attributes
rejected_merges
conflicts
```

### `srs_attribute_mapping_v13.csv`

Mapping of raw attributes to canonical SRS nodes.

Includes:

```text
raw_attribute
normalized_attribute
canonical_attribute
source_file
context
ontology_root
srs_node_id
lineage_action
merge_decision
reason
```

### `dbnf_lineage_v13.csv`

DBNF lineage actions for model-version or migration runs.

Possible actions:

```text
PRESERVE
REMAP
FORK
DEPRECATE
REVIEW
BLOCKED_BY_NEGATIVE_VETO
```

### `cross_model_sensitivity_v13.csv`

Cross-backbone diagnostic distances when a different target model is supplied.

---

## 18. Self-Checks in v13

v13 reports self-checks in `summary_v13.json` and claim-support output.

Checks include:

```text
No accepted production merge is in negative_pairs
No bridge accepts role-sensitive unsafe merge
Absent ground-truth pairs are separated from main evaluation by default
Single-context CMNF is marked NOT_EXERCISED
Cross-backbone DBNF is DIAGNOSTIC_ONLY unless explicitly allowed
SRS export contains canonical attributes and rejected merges
```

---

## 19. Data Expectations

Schema-style JSON example:

```json
{
  "schema_id": "INAmex_v1",
  "attributes": [
    {"name": "PrimaryAccountNumber"},
    {"name": "ExpirationDate"}
  ]
}
```

Payload-style JSON example:

```json
{
  "pan": "4111111111111111",
  "exp": "12/26",
  "cvv": "123"
}
```

Nested JSON is supported. The script recursively walks dictionaries and lists to collect fields and values.

---

## 20. Research Integrity Note

This artifact is intended for research, reproducibility, and reviewer validation. Avoid logging raw sensitive values in production. For production use, replace raw-value evidence with privacy-preserving summaries, hashed values, or aggregated statistics.

---

## 21. Summary

v13 turns the SDNF experiment into a cleaner and more defensible reviewer-grade artifact. It preserves the auditability of v10/v12 while adding explicit-negative safety, canonical-equivalence handling, absent-ground-truth policy, SRS exports, profile-based verbosity control, and corrected DBNF semantics.

The core v13 experiment is no longer just pairwise alias detection. It is a full SDNF-to-SRS pipeline:

```text
heterogeneous schemas
  -> evidence extraction
  -> threshold-based SDNF validation
  -> safe merge decisions
  -> explicit-negative vetoes
  -> evolved Semantic Representation Schema
  -> paper/audit claim support
```
