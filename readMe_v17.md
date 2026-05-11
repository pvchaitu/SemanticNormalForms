# SDNF Unified Experiment — Schema-First Master Payment SRS Harness v17

## 1. Overview

This repository contains a **single-file, reproducible, reviewer-grade experiment harness** for validating and iteratively improving the research direction described in:

**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**

Main experiment file:

```text
unified_sdnf_experiment_hybrid_v17.py
```

Version **v17.0.0** restores the fuller **v15/v14 implementation backbone** while incorporating the useful **v16 evaluator fixes**. It is designed as an incremental, precision-governed, reviewer-auditable experiment harness rather than a hard-coded reproduction of paper claims.

The central v17 objective is:

- preserve schema-first Payment SRS construction,
- preserve payload-evidenced governance,
- preserve strict output budgeting,
- preserve conservative claim labeling,
- add explicit **HUMAN_REVIEW** governance for ambiguous candidate merges,
- keep roadmap scaffolds ready for future SDNF geometry and HNSW enhancements without treating those scaffolds as evaluated paper claims.

---

## 2. What Changed in v17

### 2.1 v15/v14 Backbone Restored

v17 is intentionally not built from the smaller v16 scaffold alone. It restores the fuller implementation direction from v14/v15, including:

- schema-first Payment SRS construction,
- payment schema descriptor ingestion,
- payload evidence extraction,
- canonical SRS nodes,
- `build_srs`,
- `evaluate_pair`,
- field evidence construction,
- payload compliance validation,
- SRS evolved-schema export,
- output budgeting,
- DBNF/EENF/scale artifact surfaces.

### 2.2 v16 Evaluator Fixes Preserved

v17 preserves the reviewer-safety improvements introduced in v16:

- self-pair exclusion from strict metrics,
- raw predicted pair count vs. unique predicted pair count,
- duplicate predicted pair normalization,
- pair-based alias metrics separated from canonical-cluster membership metrics,
- strict metrics separated from reviewer-diagnosed metrics,
- ground-truth repair modes,
- conservative claim support statuses,
- FP/FN diagnostic surfaces,
- semantic-veto-aware review handling.

### 2.3 New Human Review Governance

v17 introduces a conservative decision type:

```text
HUMAN_REVIEW
```

This is used when a pair has promising evidence but is not safe enough for automatic merge.

Examples of cases routed to human review:

- canonical hints differ,
- semantic subtype is ambiguous,
- rail/context compatibility is uncertain,
- evidence score is above review threshold but below auto-merge threshold,
- ground truth conflicts with semantic veto,
- the pair may otherwise become a likely false positive,
- the pair is plausible but lacks sufficient evidence for automatic merge.

Important metric rule:

```text
HUMAN_REVIEW is not counted as a strict predicted positive merge.
```

Only `ACCEPT_MERGE` contributes to strict predicted alias pairs.

### 2.4 Human Review GT Conflict Handling

v17 also supports:

```text
HUMAN_REVIEW_GT_CONFLICT
```

This is intended for cases where a pair appears in ground truth but semantic vetoes block an automatic merge. Instead of silently treating this as a plain false negative, v17 makes the conflict auditable.

### 2.5 Roadmap Scaffolds Added Without New Claims

v17 includes future-ready scaffolding for planned SDNF enhancements, but these are **not evaluated claims** in v17.

Roadmap scaffolds include:

- `CandidateRetriever`
- `CanonicalEmbeddingBuilder`
- `SemanticGeometryAuditScaffold`
- `SrsEvolutionSnapshotHook`
- configurable evidence scoring registry

These scaffolds support future work such as canonical node embeddings, semantic geometry compactness/separation, partition leakage, HNSW candidate retrieval, and SRS evolution snapshots.

In v17, geometry-related outputs are marked:

```text
SCAFFOLDED_NOT_EVALUATED
```

---

## 3. Core Design Principles

v17 preserves the core SDNF experiment philosophy:

1. **Measured results only**
   - No paper claims are hardcoded as measured outcomes.

2. **Schema-first governance**
   - Schema descriptors define the contract.
   - Payloads provide evidence.
   - Payloads are not silently treated as schema.

3. **Precision-first merge policy**
   - Ambiguous merges go to review.
   - Unsafe merges are rejected.
   - Only clearly supported merges are accepted.

4. **Normal-form safety enforcement**
   - EENF
   - AANF
   - ECNF
   - RRNF
   - CMNF
   - DBNF
   - PONF

5. **Auditability over optimism**
   - Claims can be `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`, `NOT_APPLICABLE`, `NOT_EVALUATED`, or `SCAFFOLDED_NOT_EVALUATED`.

6. **Output discipline**
   - All output writes go through `OutputBudgetWriter`.
   - Paper-mode output remains below the 15-file cap.

7. **Roadmap readiness without overclaiming**
   - HNSW and geometry hooks are scaffolds unless explicitly evaluated in a future version.

---

## 4. Important v17 Concepts

### 4.1 Strict Metrics

Strict metrics count only automatic accepted merges:

```text
ACCEPT_MERGE
```

Strict metrics do not count:

```text
HUMAN_REVIEW
HUMAN_REVIEW_GT_CONFLICT
DEFER
REJECT_UNSAFE
```

### 4.2 Reviewer-Diagnosed Metrics

Reviewer-diagnosed metrics are reported separately from strict metrics. They help identify whether review-queued items could recover likely false negatives or prevent likely false positives.

They do not silently replace strict precision/recall.

### 4.3 Pair-Based Alias Metrics vs Membership Metrics

v17 keeps two separate evaluation views:

- **Pair-based alias metrics**
  - strict TP / FP / FN over predicted alias pairs.

- **Canonical-cluster membership metrics**
  - evaluates whether canonical SRS grouping is directionally aligned with expected membership.

These are intentionally separate to avoid inflated or misleading precision/recall.

### 4.4 Ground-Truth Repair Modes

v17 preserves controlled ground-truth handling:

- `closed_world_only`
  - strict paper-safe mode.
  - no implicit ground-truth expansion.

- `schema_supported_review`
  - likely missing aliases are routed for review.
  - they are not silently added as strict ground truth.

- `schema_supported_include`
  - schema-supported likely missing aliases may be included for expanded evaluation.
  - use carefully and do not mix with strict paper claims.

### 4.5 Candidate Retrieval Backends

v17 supports:

- `pairwise`
- `hnsw`
- `auto`

Important:

```text
HNSW is candidate retrieval only. It never decides merges.
```

If `hnswlib` is unavailable, v17 falls back to pairwise candidate retrieval and records the backend used in the manifest or scale/timing audit surfaces.

---

## 5. Output Profiles

v17 supports the following profiles:

| Profile | File Count Intent | Purpose |
|---|---:|---|
| `minimal` | 3 | Quick validation |
| `paper` | 13 | Paper-ready reviewer outputs |
| `audit` | 14 | Full diagnostics including debug bundle |
| `debug` | 15 | Deep introspection including readme output |

All profiles are governed by:

```text
OutputBudgetWriter
```

No output file should be written outside the output budget writer.

---

## 6. Core Outputs in v17

### 6.1 Paper Profile Outputs

The paper profile is expected to emit these v17-named files:

```text
out_audit_v17.txt
run_manifest_v17.json
summary_audit_v17.json
srs_evolved_schema_v17.compact.json
schema_ingestion_audit_v17.csv
field_evidence_audit_v17.csv
schema_deltas_audit_v17.csv
decisions_audit_v17.csv
alias_evaluation_audit_v17.csv
payload_compliance_audit_v17.csv
normal_forms_and_claims_audit_v17.csv
scale_timing_drift_audit_v17.csv
review_queue_audit_v17.csv
```

### 6.2 Optional Audit / Debug Outputs

Depending on profile and budget:

```text
sdnf_debug_bundle_v17.zip
readme_v17.md
```

---

## 7. Output File Purpose

### `out_audit_v17.txt`

Console-style run summary:

- version,
- profile,
- ground-truth repair mode,
- candidate backend,
- total attributes,
- strict alias metrics if measurable,
- reviewer-diagnosed metrics if measurable,
- review queue count,
- cross-context merge rate,
- duplicate-pair self-check,
- claim support summary.

### `run_manifest_v17.json`

Run configuration and reproducibility metadata:

- version,
- output profile,
- output directory,
- ground-truth repair mode,
- candidate backend requested,
- embedding backend,
- evidence weights,
- ground-truth audit details,
- output budget details.

### `summary_audit_v17.json`

Structured summary of:

- dataset counts,
- strict alias metrics,
- membership metrics,
- cross-context safety,
- review queue statistics,
- self-checks,
- normal-form summaries,
- roadmap scaffold summaries.

### `srs_evolved_schema_v17.compact.json`

Compact canonical SRS schema:

- canonical node id,
- canonical name,
- semantic family,
- role,
- domain,
- rails,
- providers,
- source fields,
- review candidates,
- rejected near misses,
- deferred candidates.

### `schema_ingestion_audit_v17.csv`

Schema and payload ingestion audit rows.

### `field_evidence_audit_v17.csv`

Payload-derived field evidence, where payloads are available:

- observed type,
- regex,
- shape,
- examples,
- presence ratio,
- presence class.

### `schema_deltas_audit_v17.csv`

Unexpected or unmatched payload fields compared with schema descriptors.

### `decisions_audit_v17.csv`

Detailed pairwise decision audit:

- attributes compared,
- decision type,
- normal-form statuses,
- evidence score,
- embedding similarity,
- name similarity,
- support count,
- hard vetoes,
- lineage action.

### `alias_evaluation_audit_v17.csv`

Alias evaluation metrics and self-checks:

- TP,
- FP,
- FN,
- strict precision/recall/F1,
- reviewer-diagnosed precision/recall/F1,
- raw predicted pair count,
- unique predicted pair count,
- duplicate-pair check,
- self-pair check.

### `payload_compliance_audit_v17.csv`

Payload compliance decisions when payload data is available.

### `normal_forms_and_claims_audit_v17.csv`

Claim support and normal-form status rows.

### `scale_timing_drift_audit_v17.csv`

Timing, candidate backend, embedding backend, and DBNF/EENF diagnostic surfaces.

### `review_queue_audit_v17.csv`

Human review queue:

- ambiguous candidates,
- GT/semantic-veto conflicts,
- evidence margin issues,
- canonical hint conflicts,
- subtype ambiguity,
- cross-rail uncertainty.

---

## 8. Setup

### 8.1 Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 8.2 Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 8.3 Optional Dependencies

The experiment is designed to be offline-safe where possible.

Optional packages:

```text
numpy
sentence-transformers
hnswlib
```

Fallback behavior:

- If `sentence-transformers` is unavailable, deterministic hashing embeddings are used.
- If `hnswlib` is unavailable, candidate retrieval falls back to pairwise mode.
- HNSW is never merge authority in v17.

---

## 9. Recommended Runs

### 9.1 Paper Run

Use this for the main reviewer-facing paper profile.

```bash
python unified_sdnf_experiment_hybrid_v17.py ^
  --output_profile paper ^
  --schemas_dir data ^
  --payloads_root payloads/payment ^
  --seed_srs_schema INAmex.schema.json ^
  --ground_truth_aliases ground_truth_aliases_closed_world_v17.json ^
  --ground_truth_closed_world ^
  --evaluation_track both ^
  --dbnf_mode version_drift ^
  --ground_truth_repair_mode closed_world_only ^
  --candidate_backend pairwise ^
  --measure_timing
```

### 9.2 Audit Run

Use this for deeper diagnostics and review queue inspection.

```bash
python unified_sdnf_experiment_hybrid_v17.py ^
  --output_profile audit ^
  --schemas_dir data ^
  --payloads_root payloads/payment ^
  --seed_srs_schema INAmex.schema.json ^
  --ground_truth_aliases ground_truth_aliases_closed_world_v17.json ^
  --ground_truth_closed_world ^
  --evaluation_track both ^
  --dbnf_mode both ^
  --ground_truth_repair_mode schema_supported_review ^
  --candidate_backend auto ^
  --count_semantic_veto_conflicts_as_fn ^
  --measure_timing
```

### 9.3 Minimal Run

Use this for a quick sanity check.

```bash
python unified_sdnf_experiment_hybrid_v17.py ^
  --output_profile minimal ^
  --schemas_dir data ^
  --payloads_root payloads/payment
```

### 9.4 Version-Drift DBNF Diagnostic Run

Use this to exercise the DBNF version-drift surface.

```bash
python unified_sdnf_experiment_hybrid_v17.py ^
  --output_profile paper ^
  --dbnf_mode version_drift ^
  --measure_timing
```

### 9.5 Migration Utility Run

Use this for operational model migration diagnostics. This is a utility mode, not automatically a paper claim.

```bash
python unified_sdnf_experiment_hybrid_v17.py ^
  --output_profile audit ^
  --dbnf_mode migration ^
  --dbnf_migration_model all-mpnet-base-v2
```

---

## 10. Important CLI Arguments

### Output and reproducibility

```text
--output_profile {minimal,paper,audit,debug}
--output_dir output_v17
--max_output_files 15
--seed 42
--measure_timing
```

### Data inputs

```text
--schemas_dir data
--schema_glob *.schema.json
--payloads_root payloads/payment
--seed_srs_schema INAmex.schema.json
```

### Ground truth

```text
--ground_truth_aliases ground_truth_aliases_closed_world_v17.json
--ground_truth_closed_world
--no_ground_truth_closed_world
--ground_truth_repair_mode {closed_world_only,schema_supported_review,schema_supported_include}
--count_semantic_veto_conflicts_as_fn
```

### Evaluation tracks

```text
--evaluation_track {production,discovery,both}
```

### Merge/review thresholds

```text
--review_threshold 0.62
--auto_merge_threshold 0.86
--review_margin 0.10
--tau_aanf 0.72
--name_threshold 0.45
--candidate_name_threshold 0.30
--m_min_schema 2
```

### Candidate retrieval

```text
--candidate_backend {pairwise,hnsw,auto}
--hnsw_top_k 20
```

### DBNF / EENF

```text
--dbnf_mode {none,version_drift,migration,both}
--dbnf_model_version <model-or-version>
--dbnf_migration_model all-mpnet-base-v2
--eenf_mode {not_evaluated,perturbation_stress_test}
--eenf_repeats 10
```

### Semantic safety

```text
--allow_cross_rail_amount_currency
--strict_semantic_vetoes
--precision_first
```

---

## 11. Decision Types

v17 may emit the following decision types.

### `ACCEPT_MERGE`

Used only when the candidate passes normal-form and evidence gates strongly enough for automatic merge.

Strict predicted alias pairs are derived only from this decision type.

### `HUMAN_REVIEW`

Used for plausible but ambiguous candidates.

These are not counted as strict positives.

### `HUMAN_REVIEW_GT_CONFLICT`

Used when ground truth says a pair should merge but semantic vetoes or safety rules block automatic merge.

This makes GT/code/schema conflicts auditable.

### `REJECT_UNSAFE`

Used when a hard veto or unsafe semantic conflict blocks the merge.

### `DEFER`

Used when evidence is insufficient but the pair is not clearly unsafe.

---

## 12. Normal Forms in v17

v17 keeps all seven normal-form concepts visible in audit outputs.

### EENF — Entity Embedding Normal Form

Embedding stability / perturbation diagnostic surface.

### AANF — Attribute Alias Normal Form

Alias admissibility based on semantic, lexical, canonical, and embedding evidence.

### ECNF — Evidence Completeness Normal Form

Requires sufficient evidence count and score before merge or review.

### RRNF — Role-Respecting Normal Form

Prevents role-inconsistent merges such as payer/payee or debtor/creditor conflicts.

### CMNF — Context Modulation Normal Form

Prevents unsafe context or rail mixing unless explicitly allowed for global concepts such as amount/currency.

### DBNF — Drift-Bounded Normal Form

Preserves version-drift and migration diagnostic modes, but does not mark DBNF claims as supported without appropriate drift evidence.

### PONF — Partition Orthogonality Normal Form

Preserves semantic partition safety and blocks unsafe cross-partition merges.

---

## 13. Claim Status Discipline

v17 uses conservative claim labeling.

Allowed statuses:

```text
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED
NOT_APPLICABLE
NOT_EVALUATED
SCAFFOLDED_NOT_EVALUATED
```

Guidance:

- Use `SUPPORTED` only when the current run actually computes supporting evidence.
- Use `PARTIALLY_SUPPORTED` when the run gives directional evidence but not full claim proof.
- Use `NOT_EVALUATED` when the run does not exercise the claim.
- Use `NOT_APPLICABLE` when the claim cannot apply to the current deterministic or missing-evidence setup.
- Use `SCAFFOLDED_NOT_EVALUATED` for roadmap hooks that are present but not evaluated claims.

---

## 14. Roadmap Scaffolds

### 14.1 CandidateRetriever

Provides a reusable candidate-generation seam.

Backends:

```text
pairwise
hnsw
auto
```

In v17, HNSW is constrained to candidate retrieval only.

### 14.2 CanonicalEmbeddingBuilder

Computes compact centroid summaries when embeddings are available.

These centroid summaries are not paper claims in v17.

### 14.3 SemanticGeometryAuditScaffold

Future-ready surface for:

- canonical compactness,
- inter-canonical separation,
- semantic margin,
- partition leakage.

Marked as:

```text
SCAFFOLDED_NOT_EVALUATED
```

### 14.4 SrsEvolutionSnapshotHook

Records minimal SRS snapshot information:

- version label,
- canonical node count,
- member count,
- review candidate count.

Future versions can extend this into full geometry evolution snapshots.

---

## 15. Recommended Review Workflow

After a paper or audit run:

1. Open `summary_audit_v17.json`.
2. Check `self_checks`.
3. Confirm:
   - `no_self_pairs_in_predictions = true`
   - `no_duplicate_pairs_in_predictions = true`
   - `alias_vs_membership_evaluated_separately = true`
   - `human_review_not_counted_as_strict_positive = true`
4. Open `review_queue_audit_v17.csv`.
5. Review ambiguous pairs before tuning thresholds or updating ground truth.
6. Open `decisions_audit_v17.csv`.
7. Inspect all `REJECT_UNSAFE`, `HUMAN_REVIEW`, and `HUMAN_REVIEW_GT_CONFLICT` cases.
8. Only after review, consider updating schema descriptors, alias ground truth, or thresholds.

---

## 16. Interpreting v17 Results

### Good Signs

- No duplicate predicted pairs.
- No self-pairs in strict predictions.
- Human review queue captures ambiguous cases instead of auto-merging them.
- Cross-context merge rate is low or zero.
- Claim statuses are conservative and evidence-backed.

### Warning Signs

- Many `HUMAN_REVIEW_GT_CONFLICT` rows.
- High number of semantic veto conflicts.
- Unexpected payload fields in `schema_deltas_audit_v17.csv`.
- Large gap between strict recall and reviewer-diagnosed recall.
- DBNF marked supported without explicit drift evidence. This should not happen in a paper-safe run.

---

## 17. Version Guidance

### v16 Role

v16 introduced valuable evaluator fixes, especially around alias evaluation correctness and conservative metrics.

### v17 Role

v17 is the recommended next working baseline because it combines:

- v14/v15 implementation backbone,
- v16 evaluator correctness improvements,
- Human Review governance,
- roadmap scaffolds for future SDNF geometry and HNSW enhancements.

### Future Versions

Future versions may implement, evaluate, and export full semantic geometry metrics such as:

- canonical compactness,
- inter-canonical separation,
- semantic margin,
- partition leakage,
- SRS geometry evolution snapshots,
- SDNF-governed HNSW candidate graph diagnostics.

Those should remain future evaluated enhancements and should not be claimed from v17 scaffolding alone.

---

## 18. Summary

v17 is a reviewer-grade, precision-governed SDNF experiment harness that:

- restores the fuller v15/v14 experiment structure,
- preserves v16 evaluator fixes,
- adds explicit Human Review governance,
- keeps strict and reviewer-diagnosed metrics separate,
- preserves output budgeting under the 15-file limit,
- keeps roadmap scaffolds ready without overclaiming,
- supports reproducible paper, audit, and minimal runs.

Recommended baseline file:

```text
unified_sdnf_experiment_hybrid_v17.py
```

Recommended README file:

```text
readMe_v17.md
```
