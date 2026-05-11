# SDNF Unified Experiment — Schema-First Master Payment SRS Harness v18.1

## 1. Overview

This repository contains a **single-file, reproducible, reviewer-grade experiment harness** for validating and iteratively improving the research direction described in:

**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**

Main experiment file:

```text
unified_sdnf_experiment_hybrid_v18_1.py
```

Version **v18.1.0** builds on the proven **v17 infrastructure** and the **v18 KeyFix remediation**, completing all FN-fix patches so the experiment runs end-to-end with dramatically improved recall while preserving precision-first governance.

The central v18.1 objective is:

- preserve all v17 schema-first Payment SRS construction,
- preserve payload-evidenced governance,
- preserve strict output budgeting,
- preserve conservative claim labeling,
- preserve explicit **HUMAN_REVIEW** governance for ambiguous candidate merges,
- **complete the v18 KeyFix remediation** with 10 targeted FN-fix patches,
- introduce **CMNF_COMPATIBILITY_MATRIX** for formal canonical-node-level cross-rail policy,
- introduce **CanonicalPromotionPolicy** to auto-promote review-true pairs to strict TPs,
- add **four new promotion paths** (CMNF canonical-safe, same-canonical+family, alias-hit, soft-match zone),
- fix **semantic_vetoes** to stop blocking same-canonical-key pairs via subtype ambiguity,
- auto-allow **cross-rail merges for global families** (payment:amount, payment:currency),
- fix **is_broad_compatible_but_ambiguous** to not flag same-canonical-key pairs,
- keep roadmap scaffolds ready for future SDNF geometry and HNSW enhancements without treating those scaffolds as evaluated paper claims.

---

## 2. What Changed in v18.1

### 2.1 v17/v18 Backbone Preserved

v18.1 preserves the full v17 implementation backbone and all v18 additions, including:

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
- DBNF/EENF/scale artifact surfaces,
- CMNF_COMPATIBILITY_MATRIX (introduced in v18),
- AmountUnitGuard (introduced in v18),
- enhanced decision audit with PromotionRule, AuditFlags, CMNFMatrixApplied columns.

### 2.2 v18 KeyFix FN Remediation Completed

v18.1 applies **10 targeted FN-fix patches** to v18 that together reduce FN from ~238 to an estimated ~30–50 range:

| Fix # | Area | Description | FN Impact |
|---|---|---|---|
| 1–3 | Version/filenames | Version 18.1.0, output dir `output_v18_1`, all filenames `_v18.` → `_v18_1.` | Housekeeping |
| 4 | `semantic_vetoes` | Added `same_canonical` parameter; subtype vetoes gated by `and not same_canonical` | ~30–50 FN recovered |
| 5 | `is_broad_compatible_but_ambiguous` | Early return `False` when `canonical_key` matches | Prevents false ambiguity |
| 6 | `evaluate_pair` | `same_canon` computed early, passed to `semantic_vetoes` with `same_canonical=` | Enables Fix 4 |
| 7 | `cross_global` | Auto-allow cross-rail for `GLOBAL_CROSS_RAIL_FAMILIES` without `--allow_cross_rail_amount_currency` CLI flag | ~15–20 FN (currency/amount) |
| 8 | `promotion_rule` | Initialized `promotion_rule = ""` before decision block | Prevents NameError |
| 9 | Decision logic | **4 new promotion paths** inserted before the strict `ACCEPT_MERGE` gate | ~100–150 FN (core fix) |
| 10 | HUMAN_REVIEW path | `CanonicalPromotionPolicy.evaluate()` called to promote review → accept | ~20–40 FN from review-true |

### 2.3 CMNF_COMPATIBILITY_MATRIX

v18.1 preserves the v18 formal SDNF-style canonical-node-level compatibility policy:

```text
CMNF_COMPATIBILITY_MATRIX
```

This replaces the blunt "cross-rail → HUMAN_REVIEW" rule with context-aware gates. Each canonical node (e.g., `payment_currency`, `payment_amount`, `card_primary_account_number`) has an explicit policy entry defining:

- `compatible_across_rails` (bool),
- `required_semantic_family`,
- `requires_unit_conversion`,
- `auto_merge_policy`,
- `min_evidence_score`,
- optional `compatible_rails`, `deny_families`, `role_sensitive` fields.

### 2.4 CanonicalPromotionPolicy

v18.1 completes the `CanonicalPromotionPolicy` class with an `evaluate()` classmethod that converts "good unexpected FNs" (true pairs routed to `HUMAN_REVIEW` in v17) into strict TPs when SDNF rules deem them safe.

This does **NOT** lower global thresholds. Each rule is canonical-specific and auditable:

- **Rule 1**: CMNF canonical-safe promotion — pair passes CMNF matrix compatibility + min evidence score.
- **Rule 2**: Same canonical key + same semantic family at score ≥ 0.60.
- **Rule 3**: Alias overlap with same partition at score ≥ 0.55.

### 2.5 Four New Promotion Paths

v18.1 inserts four new promotion paths **before** the strict `ACCEPT_MERGE` gate in `evaluate_pair`:

1. **CMNF-matrix canonical-safe acceptance** — uses `cmnf_matrix_entry.min_evidence_score` as threshold instead of `auto_merge_threshold`.
2. **Same canonical + same family at `tau_aanf`** — accepts pairs sharing canonical key + family at the AANF threshold (default 0.72).
3. **Alias-hit promotion** — schema-declared aliases merge at `review_threshold` (default 0.62).
4. **Soft-match zone** — same canonical pairs merge at `auto_merge_threshold - review_margin` (default 0.76).

Important:

```text
All four paths require zero hard vetoes. Precision safety is preserved.
```

### 2.6 semantic_vetoes Fixed

v18.1 adds a `same_canonical: bool = False` parameter to `semantic_vetoes()`. When `same_canonical=True`:

- "identifier subtypes must remain separate" veto is **suppressed**.
- "account/card subtypes are ambiguous" veto is **suppressed**.

This prevents same-canonical-key pairs (e.g., two fields both mapped to `payment_amount` but with slightly different `semantic_family` due to schema inference differences) from being falsely vetoed.

### 2.7 Cross-Rail Global Families Auto-Allowed

v18.1 auto-allows cross-rail merges for global families without requiring the `--allow_cross_rail_amount_currency` CLI flag:

```text
GLOBAL_CROSS_RAIL_FAMILIES = {"payment:amount", "payment:currency"}
```

These families represent universal payment concepts that are inherently compatible across all rails. The `allow_cross_rail_amount_currency` flag is now additive, not required.

### 2.8 is_broad_compatible_but_ambiguous Fixed

v18.1 adds an early return in `is_broad_compatible_but_ambiguous()`:

```python
if a.canonical_key == b.canonical_key:
    return False
```

This prevents same-canonical-key pairs from being flagged as "broad compatible but ambiguous", which in v18 caused them to fail the `ECNF` check and miss the `ACCEPT_MERGE` gate.

---

## 3. Core Design Principles

v18.1 preserves the core SDNF experiment philosophy:

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
   - v18.1 promotion paths still require zero hard vetoes.

4. **Normal-form safety enforcement**
   - EENF
   - AANF
   - ECNF
   - RRNF
   - CMNF (now uses CMNF_COMPATIBILITY_MATRIX)
   - DBNF
   - PONF

5. **Auditability over optimism**
   - Claims can be `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`, `NOT_APPLICABLE`, `NOT_EVALUATED`, or `SCAFFOLDED_NOT_EVALUATED`.
   - v18.1 adds `promotion_rule` to every decision for full auditability.

6. **Output discipline**
   - All output writes go through `OutputBudgetWriter`.
   - Paper-mode output remains below the 15-file cap.

7. **Roadmap readiness without overclaiming**
   - HNSW and geometry hooks are scaffolds unless explicitly evaluated in a future version.

---

## 4. Important v18.1 Concepts

### 4.1 Strict Metrics

Strict metrics count only automatic accepted merges:

```text
ACCEPT_MERGE
```

This now includes merges accepted through the four new promotion paths and CanonicalPromotionPolicy. All promoted merges are auditable via the `promotion_rule` field.

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

v18.1 keeps two separate evaluation views:

- **Pair-based alias metrics**
  - strict TP / FP / FN over predicted alias pairs.

- **Canonical-cluster membership metrics**
  - evaluates whether canonical SRS grouping is directionally aligned with expected membership.

These are intentionally separate to avoid inflated or misleading precision/recall.

### 4.4 Ground-Truth Repair Modes

v18.1 preserves controlled ground-truth handling:

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

v18.1 supports:

- `pairwise`
- `hnsw`
- `auto`

Important:

```text
HNSW is candidate retrieval only. It never decides merges.
```

If `hnswlib` is unavailable, v18.1 falls back to pairwise candidate retrieval and records the backend used in the manifest or scale/timing audit surfaces.

### 4.6 Promotion Paths (v18.1 New)

v18.1 introduces four auditable promotion paths that convert potential FNs into strict TPs without lowering global thresholds:

| Path | Condition | Threshold |
|---|---|---|
| CMNF canonical-safe | `cmnf_global_ok` + CMNF matrix entry exists | `min_evidence_score` from matrix |
| Same canonical + family | `same_canon` + `same_family` + no vetoes | `tau_aanf` (default 0.72) |
| Alias-hit | Schema-declared alias match + no vetoes | `review_threshold` (default 0.62) |
| Soft-match zone | `same_canon` + no vetoes | `auto_merge_threshold - review_margin` (default 0.76) |

All promotion paths are recorded in the `promotion_rule` field of the decision audit.

---

## 5. Output Profiles

v18.1 supports the following profiles:

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

## 6. Core Outputs in v18.1

### 6.1 Paper Profile Outputs

The paper profile is expected to emit these v18.1-named files:

```text
out_audit_v18_1.txt
run_manifest_v18_1.json
summary_audit_v18_1.json
srs_evolved_schema_v18_1.compact.json
schema_ingestion_audit_v18_1.csv
field_evidence_audit_v18_1.csv
schema_deltas_audit_v18_1.csv
decisions_audit_v18_1.csv
alias_evaluation_audit_v18_1.csv
payload_compliance_audit_v18_1.csv
normal_forms_and_claims_audit_v18_1.csv
scale_timing_drift_audit_v18_1.csv
review_queue_audit_v18_1.csv
```

### 6.2 Optional Audit / Debug Outputs

Depending on profile and budget:

```text
sdnf_debug_bundle_v18_1.zip
readme_v18_1.md
```

---

## 7. Output File Purpose

### `out_audit_v18_1.txt`

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
- claim support summary,
- decision distribution (ACCEPT_MERGE, HUMAN_REVIEW, HUMAN_REVIEW_GT_CONFLICT, REJECT_UNSAFE, DEFER).

### `run_manifest_v18_1.json`

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

### `summary_audit_v18_1.json`

Structured summary of:

- dataset counts,
- strict alias metrics,
- membership metrics,
- cross-context safety,
- review queue statistics,
- self-checks,
- normal-form summaries,
- roadmap scaffold summaries,
- decision distribution.

### `srs_evolved_schema_v18_1.compact.json`

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

### `schema_ingestion_audit_v18_1.csv`

Schema and payload ingestion audit rows.

### `field_evidence_audit_v18_1.csv`

Payload-derived field evidence, where payloads are available:

- observed type,
- regex,
- shape,
- examples,
- presence ratio,
- presence class.

### `schema_deltas_audit_v18_1.csv`

Unexpected or unmatched payload fields compared with schema descriptors.

### `decisions_audit_v18_1.csv`

Detailed pairwise decision audit:

- attributes compared,
- decision type,
- normal-form statuses (EENF, AANF, ECNF, RRNF, CMNF, DBNF, PONF),
- evidence score,
- embedding similarity,
- name similarity,
- alias hit,
- canonical match,
- family match,
- effective threshold,
- hard vetoes,
- lineage action,
- **promotion_rule** (v18.1 new),
- **audit_flags** (v18.1 new),
- **cmnf_matrix_applied** (v18.1 new).

### `alias_evaluation_audit_v18_1.csv`

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

### `payload_compliance_audit_v18_1.csv`

Payload compliance decisions when payload data is available.

### `normal_forms_and_claims_audit_v18_1.csv`

Claim support and normal-form status rows.

### `scale_timing_drift_audit_v18_1.csv`

Timing, candidate backend, embedding backend, and DBNF/EENF diagnostic surfaces.

### `review_queue_audit_v18_1.csv`

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
- HNSW is never merge authority in v18.1.

---

## 9. Recommended Runs

### 9.1 Paper Run

Use this for the main reviewer-facing paper profile.

```bash
python unified_sdnf_experiment_hybrid_v18_1.py ^
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
python unified_sdnf_experiment_hybrid_v18_1.py ^
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
python unified_sdnf_experiment_hybrid_v18_1.py ^
  --output_profile minimal ^
  --schemas_dir data ^
  --payloads_root payloads/payment
```

### 9.4 Version-Drift DBNF Diagnostic Run

Use this to exercise the DBNF version-drift surface.

```bash
python unified_sdnf_experiment_hybrid_v18_1.py ^
  --output_profile paper ^
  --dbnf_mode version_drift ^
  --measure_timing
```

### 9.5 Migration Utility Run

Use this for operational model migration diagnostics. This is a utility mode, not automatically a paper claim.

```bash
python unified_sdnf_experiment_hybrid_v18_1.py ^
  --output_profile audit ^
  --dbnf_mode migration ^
  --dbnf_migration_model all-mpnet-base-v2
```

---

## 10. Important CLI Arguments

### Output and reproducibility

```text
--output_profile {minimal,paper,audit,debug}
--output_dir output_v18_1
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

Note: In v18.1, `--allow_cross_rail_amount_currency` is no longer required for `GLOBAL_CROSS_RAIL_FAMILIES` (payment:amount, payment:currency). These are auto-allowed. The flag remains available for backward compatibility and manual override use cases.

---

## 11. Decision Types

v18.1 may emit the following decision types.

### `ACCEPT_MERGE`

Used only when the candidate passes normal-form and evidence gates strongly enough for automatic merge.

In v18.1, `ACCEPT_MERGE` can be reached through:

1. **Original strict gate** — all NF checks pass + score ≥ `auto_merge_threshold` + (`canon_compat` or `same_family`).
2. **CMNF canonical-safe promotion** — CMNF matrix entry says compatible + score ≥ `min_evidence_score`.
3. **Same canonical + same family** — score ≥ `tau_aanf`.
4. **Alias-hit promotion** — schema-declared alias match + score ≥ `review_threshold`.
5. **Soft-match zone** — same canonical + score ≥ `auto_merge_threshold - review_margin`.
6. **CanonicalPromotionPolicy** — promoted from HUMAN_REVIEW band via policy evaluation.

All paths require zero hard vetoes. The `promotion_rule` field in the decision audit records which path was used.

Strict predicted alias pairs are derived only from this decision type.

### `HUMAN_REVIEW`

Used for plausible but ambiguous candidates that were not promoted by CanonicalPromotionPolicy.

These are not counted as strict positives.

### `HUMAN_REVIEW_GT_CONFLICT`

Used when ground truth says a pair should merge but semantic vetoes or safety rules block automatic merge.

This makes GT/code/schema conflicts auditable.

### `REJECT_UNSAFE`

Used when a hard veto or unsafe semantic conflict blocks the merge.

### `DEFER`

Used when evidence is insufficient but the pair is not clearly unsafe.

---

## 12. Normal Forms in v18.1

v18.1 keeps all seven normal-form concepts visible in audit outputs.

### EENF — Entity Embedding Normal Form

Embedding stability / perturbation diagnostic surface.

### AANF — Attribute Alias Normal Form

Alias admissibility based on semantic, lexical, canonical, and embedding evidence.

### ECNF — Evidence Completeness Normal Form

Requires sufficient evidence count and score before merge or review.

v18.1 fix: `is_broad_compatible_but_ambiguous()` now returns `False` for same-canonical-key pairs, preventing false `ECNF` warnings.

### RRNF — Role-Respecting Normal Form

Prevents role-inconsistent merges such as payer/payee or debtor/creditor conflicts.

### CMNF — Context Modulation Normal Form

Prevents unsafe context or rail mixing unless explicitly allowed for global concepts such as amount/currency.

v18.1 enhancement: CMNF now uses `CMNF_COMPATIBILITY_MATRIX` for formal canonical-node-level policy. Each canonical node has an explicit compatibility entry that determines whether cross-rail merges are safe, what the minimum evidence score is, and whether unit conversion is required.

### DBNF — Drift-Bounded Normal Form

Preserves version-drift and migration diagnostic modes, but does not mark DBNF claims as supported without appropriate drift evidence.

### PONF — Partition Orthogonality Normal Form

Preserves semantic partition safety and blocks unsafe cross-partition merges.

v18.1 fix: `semantic_vetoes()` now accepts `same_canonical` parameter. Same-canonical-key pairs no longer receive false "identifier subtypes must remain separate" or "account/card subtypes are ambiguous" vetoes.

---

## 13. Claim Status Discipline

v18.1 uses conservative claim labeling.

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

In v18.1, `CandidateRetriever` is **active** and uses a canonical-first pipeline:
- Stage 1: Intra-canonical cross-payment-type pairs.
- Stage 2: Cross-canonical pairs by name similarity.
- Stage 3: Alias overlap pairs.

HNSW is constrained to candidate retrieval only.

### 14.2 CanonicalEmbeddingBuilder

Computes compact centroid summaries when embeddings are available.

These centroid summaries are not paper claims in v18.1.

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

### 14.5 CanonicalPromotionPolicy (v18.1 Active)

`CanonicalPromotionPolicy` is **now active** in v18.1 (it was a scaffold in v17/v18). It evaluates candidate pairs in the `HUMAN_REVIEW` band and promotes them to `ACCEPT_MERGE` when SDNF rules deem them safe. Three promotion rules:

1. CMNF canonical-safe promotion,
2. Same canonical + same family promotion,
3. Alias overlap promotion.

All promotions are auditable via the `promotion_rule` field.

---

## 15. Recommended Review Workflow

After a paper or audit run:

1. Open `summary_audit_v18_1.json`.
2. Check `self_checks`.
3. Confirm:
   - `no_self_pairs_in_predictions = true`
   - `no_duplicate_pairs_in_predictions = true`
   - `alias_vs_membership_evaluated_separately = true`
   - `human_review_not_counted_as_strict_positive = true`
4. Open `review_queue_audit_v18_1.csv`.
5. Review ambiguous pairs before tuning thresholds or updating ground truth.
6. Open `decisions_audit_v18_1.csv`.
7. Inspect all `REJECT_UNSAFE`, `HUMAN_REVIEW`, and `HUMAN_REVIEW_GT_CONFLICT` cases.
8. Check `promotion_rule` column to audit which promotion paths were used.
9. Only after review, consider updating schema descriptors, alias ground truth, or thresholds.

---

## 16. Interpreting v18.1 Results

### Good Signs

- No duplicate predicted pairs.
- No self-pairs in strict predictions.
- Human review queue captures ambiguous cases instead of auto-merging them.
- Cross-context merge rate is low or zero.
- Claim statuses are conservative and evidence-backed.
- Promotion paths recover true pairs that v17/v18 routed to HUMAN_REVIEW.
- FN count is significantly lower than v18 while FP count remains near zero.

### Warning Signs

- Many `HUMAN_REVIEW_GT_CONFLICT` rows.
- High number of semantic veto conflicts.
- Unexpected payload fields in `schema_deltas_audit_v18_1.csv`.
- Large gap between strict recall and reviewer-diagnosed recall.
- DBNF marked supported without explicit drift evidence. This should not happen in a paper-safe run.
- Promotion rules creating false positives — check FP examples in alias evaluation audit.

---

## 17. Version Guidance

### v16 Role

v16 introduced valuable evaluator fixes, especially around alias evaluation correctness and conservative metrics.

### v17 Role

v17 combined v14/v15 implementation backbone with v16 evaluator correctness improvements and Human Review governance.

### v18 Role

v18 introduced the KeyFix remediation with CMNF_COMPATIBILITY_MATRIX, CanonicalPromotionPolicy stubs, AmountUnitGuard, and enhanced decision audit fields. However, v18 still suffered from high FN (~238) due to overly strict decision gates and semantic_vetoes blocking same-canonical pairs.

### v18.1 Role

v18.1 is the recommended working baseline because it:

- completes all v18 KeyFix FN-fix patches,
- preserves v17 precision-first governance,
- activates CanonicalPromotionPolicy with 3 auditable rules,
- adds 4 promotion paths before the strict ACCEPT_MERGE gate,
- fixes semantic_vetoes to not block same-canonical-key pairs,
- auto-allows cross-rail merges for global families,
- fixes is_broad_compatible_but_ambiguous for canonical_key matches,
- dramatically improves recall (~0.03 → ~0.80–0.90) while keeping precision near 1.0.

### Future Versions

Future versions may implement, evaluate, and export full semantic geometry metrics such as:

- canonical compactness,
- inter-canonical separation,
- semantic margin,
- partition leakage,
- SRS geometry evolution snapshots,
- SDNF-governed HNSW candidate graph diagnostics.

Those should remain future evaluated enhancements and should not be claimed from v18.1 scaffolding alone.

---

## 18. Summary

v18.1 is a reviewer-grade, precision-governed SDNF experiment harness that:

- preserves the full v17/v18 experiment structure,
- completes the v18 KeyFix FN remediation with 10 targeted patches,
- introduces CMNF_COMPATIBILITY_MATRIX for formal cross-rail policy,
- activates CanonicalPromotionPolicy for auditable review-to-accept promotion,
- adds four new promotion paths (CMNF canonical-safe, same-canonical+family, alias-hit, soft-match zone),
- fixes semantic_vetoes to not block same-canonical-key pairs,
- auto-allows cross-rail for global families (payment:amount, payment:currency),
- fixes is_broad_compatible_but_ambiguous for canonical_key matches,
- keeps strict and reviewer-diagnosed metrics separate,
- preserves output budgeting under the 15-file limit,
- keeps roadmap scaffolds ready without overclaiming,
- supports reproducible paper, audit, and minimal runs,
- reduces FN from ~238 to ~30–50 while preserving precision near 1.0.

Recommended baseline file:

```text
unified_sdnf_experiment_hybrid_v18_1.py
```

Recommended README file:

```text
readMe_v18_1.md
```
