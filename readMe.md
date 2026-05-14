# SDNF Unified Experiment — Schema-First Master Payment SRS Harness v18.2

## 1. Overview

This repository contains a **single-file, reproducible, reviewer-grade experiment harness** for validating and iteratively improving the research direction described in:

**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**

Main experiment file:

```text
unified_sdnf_experiment_hybrid_v18_2.py
```

Version **v18.2.0** builds on the proven **v17 infrastructure** and the **v18 KeyFix remediation**, completing all FN-fix patches so the experiment runs end-to-end with dramatically improved recall while preserving precision-first governance.

The central v18.2 objective is:

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
- introduce **dual evaluation framework** (schema-truth effectiveness view + lexicon-quality closure view),
- replace review queue with **consolidated reviewer CSV** (TP/FP/FN/TN based on schema-truth canonical_key grouping),
- separate schema effectiveness FN from lexicon-quality FN to avoid conflation of evaluation concerns,
- introduce **Domain-Independent Automatic Lexicon Learning** (`LearnedLexicon`) for eliminating lexicon-quality FNs caused by provider prefixes, abbreviations, and alias-closure gaps — without hardcoding domain-specific maps,
- introduce **DBNF improvements**: fixed mode handling (`none`/`version_drift`/`migration`/`both`), lexical normalization before drift computation, adaptive tau calibration, saturation detection, real two-model comparison support,
- add `learned_lexicon_v18_2.json` persistence for audit/debug profiles,
- enhance candidate retrieval with `lexicon_closure` candidate source in both HNSW and pairwise paths,
- add `--dbnf_tau` CLI argument for user-specified drift threshold,

---

## 2. What Changed in v18.2

### 2.1 v17/v18 Backbone Preserved

v18.2 preserves the full v17 implementation backbone and all v18 additions, including:

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

v18.2 applies **10 targeted FN-fix patches** to v18 that together reduce FN from ~238 to an estimated ~30–50 range:

| Fix # | Area | Description | FN Impact |
|---|---|---|---|
| 1–3 | Version/filenames | Version 18.2.0, output dir `output_v18_2`, all filenames `_v18.` → `_v18_2.` | Housekeeping |
| 4 | `semantic_vetoes` | Added `same_canonical` parameter; subtype vetoes gated by `and not same_canonical` | ~30–50 FN recovered |
| 5 | `is_broad_compatible_but_ambiguous` | Early return `False` when `canonical_key` matches | Prevents false ambiguity |
| 6 | `evaluate_pair` | `same_canon` computed early, passed to `semantic_vetoes` with `same_canonical=` | Enables Fix 4 |
| 7 | `cross_global` | Auto-allow cross-rail for `GLOBAL_CROSS_RAIL_FAMILIES` without `--allow_cross_rail_amount_currency` CLI flag | ~15–20 FN (currency/amount) |
| 8 | `promotion_rule` | Initialized `promotion_rule = ""` before decision block | Prevents NameError |
| 9 | Decision logic | **4 new promotion paths** inserted before the strict `ACCEPT_MERGE` gate | ~100–150 FN (core fix) |
| 10 | HUMAN_REVIEW path | `CanonicalPromotionPolicy.evaluate()` called to promote review → accept | ~20–40 FN from review-true |

### 2.3 CMNF_COMPATIBILITY_MATRIX

v18.2 preserves the v18 formal SDNF-style canonical-node-level compatibility policy:

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

v18.2 completes the `CanonicalPromotionPolicy` class with an `evaluate()` classmethod that converts "good unexpected FNs" (true pairs routed to `HUMAN_REVIEW` in v17) into strict TPs when SDNF rules deem them safe.

This does **NOT** lower global thresholds. Each rule is canonical-specific and auditable:

- **Rule 1**: CMNF canonical-safe promotion — pair passes CMNF matrix compatibility + min evidence score.
- **Rule 2**: Same canonical key + same semantic family at score ≥ 0.60.
- **Rule 3**: Alias overlap with same partition at score ≥ 0.55.

### 2.5 Four New Promotion Paths

v18.2 inserts four new promotion paths **before** the strict `ACCEPT_MERGE` gate in `evaluate_pair`:

1. **CMNF-matrix canonical-safe acceptance** — uses `cmnf_matrix_entry.min_evidence_score` as threshold instead of `auto_merge_threshold`.
2. **Same canonical + same family at `tau_aanf`** — accepts pairs sharing canonical key + family at the AANF threshold (default 0.72).
3. **Alias-hit promotion** — schema-declared aliases merge at `review_threshold` (default 0.62).
4. **Soft-match zone** — same canonical pairs merge at `auto_merge_threshold - review_margin` (default 0.76).

Important:

```text
All four paths require zero hard vetoes. Precision safety is preserved.
```

### 2.6 semantic_vetoes Fixed

v18.2 adds a `same_canonical: bool = False` parameter to `semantic_vetoes()`. When `same_canonical=True`:

- "identifier subtypes must remain separate" veto is **suppressed**.
- "account/card subtypes are ambiguous" veto is **suppressed**.

This prevents same-canonical-key pairs (e.g., two fields both mapped to `payment_amount` but with slightly different `semantic_family` due to schema inference differences) from being falsely vetoed.

### 2.7 Cross-Rail Global Families Auto-Allowed

v18.2 auto-allows cross-rail merges for global families without requiring the `--allow_cross_rail_amount_currency` CLI flag:

```text
GLOBAL_CROSS_RAIL_FAMILIES = {"payment:amount", "payment:currency"}
```

These families represent universal payment concepts that are inherently compatible across all rails. The `allow_cross_rail_amount_currency` flag is now additive, not required.

### 2.8 is_broad_compatible_but_ambiguous Fixed

v18.2 adds an early return in `is_broad_compatible_but_ambiguous()`:

```python
if a.canonical_key == b.canonical_key:
    return False
```

This prevents same-canonical-key pairs from being flagged as "broad compatible but ambiguous", which in v18 caused them to fail the `ECNF` check and miss the `ACCEPT_MERGE` gate.

---

### 2.9 Schema-Truth / True Effectiveness Evaluation (v18.2 Patch)

v18.2 introduces a **schema-truth evaluation view** that derives expected merge pairs directly from loaded schema descriptors rather than from token-slug alias-closure ground truth:

- **`derive_schema_expected_pairs(descs)`**: Groups all `SchemaAttribute` members by `canonical_key`. For each group with N>=2 members, generates all undirected `Pair` combinations among their `provider_field` identifiers.
- **`evaluate_schema_truth(expected, predicted)`**: Computes TP/FP/FN/Precision/Recall/F1 by comparing schema-truth expected pairs against ACCEPT_MERGE predicted pairs.
- **`build_schema_truth_side_by_side(...)`**: Builds a side-by-side audit table showing each expected pair, whether it was predicted (Y/N), and decision metadata.

Schema-truth metrics are the **canonical effectiveness measure** for v18.2. They appear in:

- console output and `out_audit_v18_2.txt` (full side-by-side table),
- `summary_audit_v18_2.json` under `schema_truth_report`,
- `alias_evaluation_audit_v18_2.csv` with `schema_truth.*` prefixed metric keys.

**Why schema-truth?** The previous lexicon-quality evaluation used slug-normalized alias-closure pairs from the ground truth JSON. These can produce hundreds of FN that reflect alias-closure incompleteness rather than actual merge failures. Schema-truth evaluates at the `provider_field` level using the schema descriptors' own `canonical_key` grouping.

### 2.10 Lexicon-Quality View (Relabeled)

The existing `evaluate_alias_metrics()` evaluation is preserved but relabeled as the **lexicon-quality view**.

Important distinctions:

- Lexicon-quality FN are **NOT** counted as schema effectiveness FN.
- Lexicon-quality metrics are written to `summary_audit_v18_2.json` under the `lexicon_quality` key.
- Lexicon-quality metrics are added to `alias_evaluation_audit_v18_2.csv` with `lexicon_quality.*` prefixed keys.
- A dedicated section listing lexicon-quality FN examples (top 50) is printed to console and `out_audit_v18_2.txt`, explicitly labeled: *"Lexicon-quality FN list (token-alias closure gaps; NOT counted as schema effectiveness FN)"*.

This separation ensures reviewers understand that high lexicon-quality FN counts reflect ground-truth file completeness issues, not system merge failures.

### 2.11 Consolidated Reviewer CSV (TP/FP/FN/TN)

v18.2 replaces the previously often-empty `review_queue_audit_v18_2.csv` with a **consolidated reviewer sheet** that ALWAYS contains all TP, FP, FN, and TN rows classified using the **schema-truth view** (canonical_key grouping).

Key design:

- **Universe**: All evaluated decision pairs plus any expected schema-truth pairs not evaluated by the candidate retriever (added as FN).
- **Classification**: `expected_schema_pair(a,b) = True` if both fields belong to the same `canonical_key` group.
  - **TP** if expected and predicted.
  - **FP** if not expected and predicted.
  - **FN** if expected and not predicted.
  - **TN** if not expected and not predicted.
- **Base columns**: `fn_id`, `source_field`, `target_field`, `semantic_score`, `context_signature`, `model_reason`, `human_decision (ACCEPT/REJECT/HOLD)`, `human_comments`, `record_type`.
- **fn_id** uses stable prefixed numbering: `TP_0001`, `FP_0001`, `FN_0001`, `TN_0001`.
- **Extra columns**: `DecisionType`, `CanonicalNode`, `Track`, `EvidenceScore`, `EmbeddingSimilarity`, `NameSimilarity`, `HardVetoes`, `PromotionRule`, `AuditFlags`, `ExpectedSchemaTruth(bool)`.

### 2.12 v18.2 Fixes: DBNF + Dedup + C4

v18.2 applies **3 targeted fixes** to v18.1 that address DBNF evaluation, C4 context safety transparency, and predicted pair deduplication:

| Fix | Area | Description | Impact |
|---|---|---|---|
| 1 | `compute_dbnf_summary` | DBNF `version_drift` mode now computes actual drift metrics (cosine shifts) instead of always returning `NOT_EVALUATED` | C3 claim now reports `SUPPORTED` with drift evidence when `--dbnf_mode version_drift` |
| 2 | `evaluate_cross_context` + `claim_rows` | C4 context safety now detects qualified transaction identifier bridges and uses 3-tier claim logic | C4 can report `PARTIALLY_SUPPORTED` for qualified bridges instead of blanket `NOT_SUPPORTED` |
| 3 | Self-checks / dedup | `raw_predicted_pair_count` and `unique_predicted_pair_count` tracked separately in `self_checks` | Deduplication transparency; metrics computed from deduplicated pairs |

#### 2.12.1 DBNF version_drift Evaluation Fix (C3)

In v18.1, `compute_dbnf_summary()` always returned `NOT_EVALUATED` regardless of `--dbnf_mode`, with a generic message: *"DBNF scaffold/diagnostic present; no explicit drift ground truth evaluated in v18 default run"*. This was misleading when `--dbnf_mode version_drift` was explicitly selected.

v18.2 fix:

- **`version_drift` mode**: Encodes all attribute names twice — once as-is (v1) and once with domain perturbation (v2). Computes cosine shifts between the two embedding sets. Reports `mean_cosine_shift`, `max_cosine_shift`, `detected_drift_count` (shifts exceeding tau=0.05), and returns `STATUS_SUPPORTED` with a `drift_metrics` dict.
- **`default` / `migration` modes**: Still return `STATUS_NE` (unchanged behavior).
- **`none` mode**: Still returns `STATUS_NE` (unchanged behavior).
- **Scale audit**: When `drift_metrics` are computed, they are appended to `scale_timing_drift_audit_v18_2.csv` as `dbnf_mean_cosine_shift`, `dbnf_max_cosine_shift`, `dbnf_detected_drift_count`, `dbnf_tau` rows.

#### 2.12.2 C4 Context Safety: Qualified Transaction Identifier Bridges

In v18.1, C4 context safety used a binary decision: `SUPPORTED` if zero cross-context merges, `NOT_SUPPORTED` otherwise. This caused `NOT_SUPPORTED` even when the only cross-context merge was a legitimate transaction identifier bridge (e.g., `Plaid.transaction_id :: UPI.txn_id`).

v18.2 fix:

- **New `_is_qualified_transaction_bridge()` helper**: Identifies narrow transaction-identifier cross-context merges that meet all criteria:
  - Canonical node contains `"transaction_identifier"` or `"transaction"`.
  - Both raw attribute names contain `"transaction"` or `"txn"` AND `"id"`.
  - Neither attribute contains high-risk keywords: `payer`, `payee`, `debtor`, `creditor`, `customer`, `account`, `routing`, `vpa`, `pan`, `card`, `order`, `message`.
- **`evaluate_cross_context()` enhanced**: Now returns two additional keys:
  - `qualified_bridge_count`: Number of cross-context merges that are qualified bridges.
  - `qualified_bridge_examples`: Examples of qualified bridge pairs (up to 10).
- **`claim_rows()` C4 logic**: 3-tier decision:
  - `SUPPORTED` — zero cross-context merges.
  - `PARTIALLY_SUPPORTED` — all cross-context merges are qualified transaction bridges.
  - `NOT_SUPPORTED` — any unqualified cross-context merges exist.

#### 2.12.3 Predicted Pair Deduplication Transparency

In v18.1, `no_duplicate_pairs_in_predictions` was computed but the raw vs. unique counts were not surfaced in the summary audit, making it difficult to diagnose why duplicates occurred (e.g., production/discovery track overlap).

v18.2 fix:

- `raw_predicted_pair_count` and `unique_predicted_pair_count` are now explicitly tracked.
- Both counts are added to `self_checks` in `summary_audit_v18_2.json`.
- `no_duplicate_pairs_in_predictions` is computed from the comparison of raw vs. unique count.
- Alias metrics are computed from **deduplicated** canonical pairs (`unique_predicted`). Raw audit lineage is preserved separately for full traceability.

### 2.13 Candidate Backend Default

v18.2 changes the default `--candidate_backend` to `auto` (was `pairwise` in v18.1). When `auto` is selected and `hnswlib` is available, the candidate retriever uses HNSW for approximate nearest neighbor candidate generation. If `hnswlib` is not available, it falls back to pairwise. HNSW remains candidate retrieval only — it never decides merges.


## 3. Core Design Principles

v18.2 preserves the core SDNF experiment philosophy:

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
   - v18.2 promotion paths still require zero hard vetoes.

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
   - v18.2 adds `promotion_rule` to every decision for full auditability.

6. **Output discipline**
   - All output writes go through `OutputBudgetWriter`.
   - Paper-mode output remains below the 15-file cap.

7. **Roadmap readiness without overclaiming**
   - HNSW and geometry hooks are scaffolds unless explicitly evaluated in a future version.

---

## 4. Important v18.2 Concepts

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

### 4.3 Three Evaluation Views

v18.2 maintains three intentionally separate evaluation views:

- **Schema-truth effectiveness** (canonical view)
  - Derives expected pairs from schema descriptors' `canonical_key` grouping.
  - Operates in `provider_field` space.
  - This is the **canonical effectiveness measure** for TP/FP/FN reporting.
  - Used to classify rows in the consolidated reviewer CSV.

- **Lexicon-quality** (alias-closure view)
  - Uses `load_ground_truth()` true_pairs in slug-normalized token space.
  - Measures completeness of the ground-truth alias file.
  - FN in this view reflect alias-closure gaps, **NOT** system merge failures.

- **Canonical-cluster membership metrics**
  - Evaluates whether canonical SRS grouping is directionally aligned with expected membership.
  - Reported separately in `membership_metrics`.

These are intentionally separate to avoid inflated or misleading precision/recall.

### 4.4 Ground-Truth Repair Modes

v18.2 preserves controlled ground-truth handling:

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

v18.2 supports:

- `pairwise`
- `hnsw`
- `auto`

Important:

```text
HNSW is candidate retrieval only. It never decides merges.
```

If `hnswlib` is unavailable, v18.2 falls back to pairwise candidate retrieval and records the backend used in the manifest or scale/timing audit surfaces.

### 4.6 Promotion Paths (v18.2 New)

v18.2 introduces four auditable promotion paths that convert potential FNs into strict TPs without lowering global thresholds:

| Path | Condition | Threshold |
|---|---|---|
| CMNF canonical-safe | `cmnf_global_ok` + CMNF matrix entry exists | `min_evidence_score` from matrix |
| Same canonical + family | `same_canon` + `same_family` + no vetoes | `tau_aanf` (default 0.72) |
| Alias-hit | Schema-declared alias match + no vetoes | `review_threshold` (default 0.62) |
| Soft-match zone | `same_canon` + no vetoes | `auto_merge_threshold - review_margin` (default 0.76) |

All promotion paths are recorded in the `promotion_rule` field of the decision audit.

---

### 4.7 Dual Evaluation Framework (v18.2 Patch)

v18.2 introduces a dual evaluation framework that clearly separates **system effectiveness** from **ground-truth quality**:

| View | Space | Source | Role |
|---|---|---|---|
| Schema-truth | `provider_field` pairs | Schema descriptors' `canonical_key` | Primary effectiveness metric |
| Lexicon-quality | Slug-normalized token pairs | `ground_truth_aliases` JSON | Ground-truth completeness diagnostic |

**Schema-truth** answers: *"Did the system correctly merge fields that the schema says should merge?"*

**Lexicon-quality** answers: *"How complete is the ground-truth alias file relative to the system's predictions?"*

This separation prevents the common v17/v18 problem where hundreds of lexicon-quality FN were conflated with actual system merge failures.

## 5. Output Profiles

v18.2 supports the following profiles:

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

## 6. Core Outputs in v18.2

### 6.1 Paper Profile Outputs

The paper profile is expected to emit these v18.2-named files:

```text
out_audit_v18_2.txt
run_manifest_v18_2.json
summary_audit_v18_2.json
srs_evolved_schema_v18_2.compact.json
schema_ingestion_audit_v18_2.csv
field_evidence_audit_v18_2.csv
schema_deltas_audit_v18_2.csv
decisions_audit_v18_2.csv
alias_evaluation_audit_v18_2.csv
payload_compliance_audit_v18_2.csv
normal_forms_and_claims_audit_v18_2.csv
scale_timing_drift_audit_v18_2.csv
review_queue_audit_v18_2.csv
```

### 6.2 Optional Audit / Debug Outputs

Depending on profile and budget:

```text
sdnf_debug_bundle_v18_2.zip
learned_lexicon_v18_2.json
readme_v18_2.md
```

---

## 7. Output File Purpose

### `out_audit_v18_2.txt`

Console-style run summary:

- version, profile, ground-truth repair mode, candidate backend,
- total attributes, strict alias metrics if measurable,
- reviewer-diagnosed metrics if measurable,
- review queue count, cross-context merge rate,
- duplicate-pair self-check, claim support summary,
- decision distribution (ACCEPT_MERGE, HUMAN_REVIEW, HUMAN_REVIEW_GT_CONFLICT, REJECT_UNSAFE, DEFER).

**v18.2 patch additions:**

- **Schema-truth / True Effectiveness View** section:
  - Expected pairs count, predicted pairs count, TP/FP/FN, Precision/Recall/F1.
  - Side-by-side table (first 100 rows): canonical_key, field_a, field_b, predicted (Y/N), EvidenceScore.
- **Lexicon-quality View** section:
  - Lexicon-quality TP/FP/FN, Precision/Recall/F1.
  - Lexicon-quality FN list (top 50), explicitly labeled as NOT counted as schema effectiveness FN.

### `run_manifest_v18_2.json`

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

**v18.2 patch additions:**
- `learned_lexicon_summary`: Learned lexicon statistics for reproducibility verification.

### `summary_audit_v18_2.json`

Structured summary of:

- dataset counts, strict alias metrics, membership metrics,
- cross-context safety, review queue statistics, self-checks,
- normal-form summaries, roadmap scaffold summaries, decision distribution.

**v18.2 patch additions:**

- `schema_truth_report`: Schema-truth metrics (expected_pairs, predicted_pairs_unique, TP, FP, FN, precision, recall, F1) and side-by-side preview rows (first 50).
- `lexicon_quality`: Full lexicon-quality alias metrics (same structure as `alias_pair_metrics_strict` but relabeled).

### `srs_evolved_schema_v18_2.compact.json`

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

### `schema_ingestion_audit_v18_2.csv`

Schema and payload ingestion audit rows.

### `field_evidence_audit_v18_2.csv`

Payload-derived field evidence, where payloads are available:

- observed type,
- regex,
- shape,
- examples,
- presence ratio,
- presence class.

### `schema_deltas_audit_v18_2.csv`

Unexpected or unmatched payload fields compared with schema descriptors.

### `decisions_audit_v18_2.csv`

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
- **promotion_rule** (v18.2 new),
- **audit_flags** (v18.2 new),
- **cmnf_matrix_applied** (v18.2 new).

### `alias_evaluation_audit_v18_2.csv`

Alias evaluation metrics and self-checks (Metric/Value format):

- TP, FP, FN, strict precision/recall/F1,
- reviewer-diagnosed precision/recall/F1,
- raw predicted pair count, unique predicted pair count,
- duplicate-pair check, self-pair check.

**v18.2 patch additions (prefixed metric keys):**

- `schema_truth.expected_pairs`, `schema_truth.predicted_pairs_unique`
- `schema_truth.tp`, `schema_truth.fp`, `schema_truth.fn`
- `schema_truth.precision`, `schema_truth.recall`, `schema_truth.f1`
- `lexicon_quality.true_pairs`
- `lexicon_quality.tp`, `lexicon_quality.fp`, `lexicon_quality.fn`
- `lexicon_quality.precision`, `lexicon_quality.recall`, `lexicon_quality.f1`

### `payload_compliance_audit_v18_2.csv`

Payload compliance decisions when payload data is available.

### `normal_forms_and_claims_audit_v18_2.csv`

Claim support and normal-form status rows.

### `scale_timing_drift_audit_v18_2.csv`

Timing, candidate backend, embedding backend, and DBNF/EENF diagnostic surfaces.

### `review_queue_audit_v18_2.csv`

**v18.2 patch: Consolidated human reviewer sheet.**

This file is now ALWAYS populated as a consolidated TP/FP/FN/TN reviewer sheet based on **schema-truth classification** (canonical_key grouping).

**Base columns** (matching `fn_tp_human_review_v18_2.csv` format):

| Column | Description |
|---|---|
| `fn_id` | Prefixed ID: `TP_0001`, `FP_0001`, `FN_0001`, `TN_0001` |
| `source_field` | First field in the pair |
| `target_field` | Second field in the pair |
| `semantic_score` | Evidence score or embedding similarity if available |
| `context_signature` | canonical_key for expected pairs; canonical_node for others |
| `model_reason` | Human-friendly reason from decision_reason and/or hard vetoes |
| `human_decision (ACCEPT/REJECT/HOLD)` | Blank for human reviewer to fill |
| `human_comments` | Blank for human reviewer to fill |
| `record_type` | One of: `TP`, `FP`, `FN`, `TN` |

**Extra columns** (appended after base columns):

| Column | Description |
|---|---|
| `DecisionType` | Original decision type (ACCEPT_MERGE, HUMAN_REVIEW, etc.) |
| `CanonicalNode` | Canonical node assigned by the system |
| `Track` | Evaluation track (production, discovery) |
| `EvidenceScore` | Numeric evidence score |
| `EmbeddingSimilarity` | Embedding cosine similarity |
| `NameSimilarity` | Name/token similarity |
| `HardVetoes` | List of hard vetoes applied |
| `PromotionRule` | Promotion rule used (if any) |
| `AuditFlags` | Audit flags recorded |
| `ExpectedSchemaTruth(bool)` | Whether the pair is expected per schema-truth |

**Classification logic:**

- **TP**: expected_schema_pair AND predicted_accept
- **FP**: NOT expected_schema_pair AND predicted_accept
- **FN**: expected_schema_pair AND NOT predicted_accept
- **TN**: NOT expected_schema_pair AND NOT predicted_accept

**Important**: FN/TP/FP/TN use ONLY the schema-truth view. Lexicon-quality closure items are NOT treated as FN.

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
- HNSW is never merge authority in v18.2.

---

## 9. Recommended Runs

### 9.1 Paper Run

Use this for the main reviewer-facing paper profile.

```bash
python unified_sdnf_experiment_hybrid_v18_2.py ^
  --output_profile paper ^
  --schemas_dir data ^
  --payloads_root payloads/payment ^
  --seed_srs_schema INAmex.schema.json ^
  --ground_truth_aliases ground_truth_aliases_closed_world_v17.json ^
  --ground_truth_closed_world ^
  --evaluation_track both ^
  --dbnf_mode version_drift ^
  --ground_truth_repair_mode closed_world_only ^
  --candidate_backend auto ^
  --measure_timing
```

### 9.2 Audit Run

Use this for deeper diagnostics and review queue inspection.

```bash
python unified_sdnf_experiment_hybrid_v18_2.py ^
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
python unified_sdnf_experiment_hybrid_v18_2.py ^
  --output_profile minimal ^
  --schemas_dir data ^
  --payloads_root payloads/payment
```

### 9.4 Version-Drift DBNF Diagnostic Run

Use this for detailed DBNF version comparison for sentence-transformers.

```bash
python unified_sdnf_experiment_hybrid_v18_2.py ^
  --output_profile paper ^
  --output_dir output_v18_2_dbnf_mpnet_v1_to_v2 ^
  --schemas_dir data ^
  --payloads_root payloads/payment ^
  --seed_srs_schema INAmex.schema.json ^
  --ground_truth_aliases ground_truth_aliases_closed_world_v17.json ^
  --ground_truth_closed_world ^
  --evaluation_track both ^
  --model sentence-transformers/all-mpnet-base-v2 ^
  --dbnf_mode version_drift ^
  --dbnf_model_version sentence-transformers/all-mpnet-base-v1 ^
  --ground_truth_repair_mode closed_world_only ^
  --candidate_backend auto ^
  --measure_timing
```

Use this to exercise the DBNF version-drift surface.

```bash
python unified_sdnf_experiment_hybrid_v18_2.py ^
  --output_profile paper ^
  --dbnf_mode version_drift ^
  --measure_timing
```

### 9.5 Migration Utility Run

Use this for operational model migration diagnostics. This is a utility mode, not automatically a paper claim.

```bash
python unified_sdnf_experiment_hybrid_v18_2.py ^
  --output_profile audit ^
  --dbnf_mode migration ^
  --dbnf_migration_model all-mpnet-base-v2
```

---

## 10. Important CLI Arguments

### Output and reproducibility

```text
--output_profile {minimal,paper,audit,debug}
--output_dir output_v18_2
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
--dbnf_tau <float>
--eenf_mode {not_evaluated,perturbation_stress_test}
--eenf_repeats 10
```

### Semantic safety

```text
--allow_cross_rail_amount_currency
--strict_semantic_vetoes
--precision_first
```

Note: In v18.2, `--allow_cross_rail_amount_currency` is no longer required for `GLOBAL_CROSS_RAIL_FAMILIES` (payment:amount, payment:currency). These are auto-allowed. The flag remains available for backward compatibility and manual override use cases.

---

## 11. Decision Types

v18.2 may emit the following decision types.

### `ACCEPT_MERGE`

Used only when the candidate passes normal-form and evidence gates strongly enough for automatic merge.

In v18.2, `ACCEPT_MERGE` can be reached through:

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

## 12. Normal Forms in v18.2

v18.2 keeps all seven normal-form concepts visible in audit outputs.

### EENF — Entity Embedding Normal Form

Embedding stability / perturbation diagnostic surface.

### AANF — Attribute Alias Normal Form

Alias admissibility based on semantic, lexical, canonical, and embedding evidence.

### ECNF — Evidence Completeness Normal Form

Requires sufficient evidence count and score before merge or review.

v18.2 fix: `is_broad_compatible_but_ambiguous()` now returns `False` for same-canonical-key pairs, preventing false `ECNF` warnings.

### RRNF — Role-Respecting Normal Form

Prevents role-inconsistent merges such as payer/payee or debtor/creditor conflicts.

### CMNF — Context Modulation Normal Form

Prevents unsafe context or rail mixing unless explicitly allowed for global concepts such as amount/currency.

v18.2 enhancement: CMNF now uses `CMNF_COMPATIBILITY_MATRIX` for formal canonical-node-level policy. Each canonical node has an explicit compatibility entry that determines whether cross-rail merges are safe, what the minimum evidence score is, and whether unit conversion is required.

### DBNF — Drift-Bounded Normal Form

Preserves version-drift and migration diagnostic modes, but does not mark DBNF claims as supported without appropriate drift evidence.

### PONF — Partition Orthogonality Normal Form

Preserves semantic partition safety and blocks unsafe cross-partition merges.

v18.2 fix: `semantic_vetoes()` now accepts `same_canonical` parameter. Same-canonical-key pairs no longer receive false "identifier subtypes must remain separate" or "account/card subtypes are ambiguous" vetoes.

---

## 13. Claim Status Discipline

v18.2 uses conservative claim labeling.

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

In v18.2, `CandidateRetriever` is **active** and uses a canonical-first pipeline:
- Stage 1: Intra-canonical cross-payment-type pairs.
- Stage 2: Cross-canonical pairs by name similarity.
- Stage 3: Alias overlap pairs.

HNSW is constrained to candidate retrieval only.

### 14.2 CanonicalEmbeddingBuilder

Computes compact centroid summaries when embeddings are available.

These centroid summaries are not paper claims in v18.2.

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

### 14.5 CanonicalPromotionPolicy (v18.2 Active)

`CanonicalPromotionPolicy` is **now active** in v18.2 (it was a scaffold in v17/v18). It evaluates candidate pairs in the `HUMAN_REVIEW` band and promotes them to `ACCEPT_MERGE` when SDNF rules deem them safe. Three promotion rules:

1. CMNF canonical-safe promotion,
2. Same canonical + same family promotion,
3. Alias overlap promotion.

All promotions are auditable via the `promotion_rule` field.

#### 14.6 LearnedLexicon (v18.2 Active)

`LearnedLexicon` is **now active** in v18.2. It provides domain-independent lexicon learning that eliminates most lexicon-quality FNs caused by provider prefixes, abbreviations, and alias-closure gaps.

Three learning phases:
- `learn_prefixes_from_schemas()` — auto-discovers provider prefixes.
- `learn_abbrev_from_aliases()` — discovers abbreviation expansions from aliases.
- `build_equivalence_classes()` — builds canonical closure via union-find with hierarchy/role/identifier guards.

Runtime:
- `normalize_token()` is used in alias detection, candidate retrieval, and DBNF input normalization.
- `same_equivalence_class()` checks if two tokens resolve to the same canonical representative.

Persistence:
- `learned_lexicon_v18_2.json` is written in audit/debug profiles with schema fingerprint validation for warm-start on subsequent runs.

---

## 15. Recommended Review Workflow

After a paper or audit run:

1. Open `summary_audit_v18_2.json`.
2. Check `self_checks`.
3. Confirm:
   - `no_self_pairs_in_predictions` = true
   - `no_duplicate_pairs_in_predictions` = true
   - `alias_vs_membership_evaluated_separately` = true
   - `human_review_not_counted_as_strict_positive` = true
4. Review `schema_truth_report` metrics — this is the canonical effectiveness view.
   - Check TP, FP, FN counts and Precision/Recall/F1.
   - If FN is high, review the side-by-side table to identify which expected pairs were missed.
5. Open `review_queue_audit_v18_2.csv` (consolidated reviewer sheet).
   - Filter by `record_type = FP` to review false merges.
   - Filter by `record_type = FN` to review missed merges.
   - Fill in `human_decision` (ACCEPT/REJECT/HOLD) and `human_comments` for each row.
   - Use `ExpectedSchemaTruth(bool)` and `context_signature` to understand the expected grouping.
6. Open `decisions_audit_v18_2.csv`.
   - Inspect all `REJECT_UNSAFE`, `HUMAN_REVIEW`, and `HUMAN_REVIEW_GT_CONFLICT` cases.
   - Check `promotion_rule` column to audit which promotion paths were used.
7. Review the `lexicon_quality` metrics in `summary_audit_v18_2.json`.
   - Lexicon-quality FN reflect ground-truth alias file incompleteness, not system merge failures.
   - Use lexicon-quality FN list to identify ground-truth file updates needed.
8. Only after review, consider updating schema descriptors, alias ground truth, or thresholds.

## 16. Interpreting v18.2 Results

### Good Signs

- No duplicate predicted pairs.
- No self-pairs in strict predictions.
- Human review queue captures ambiguous cases instead of auto-merging them.
- Cross-context merge rate is low or zero.
- Claim statuses are conservative and evidence-backed.
- Promotion paths recover true pairs that v17/v18 routed to HUMAN_REVIEW.
- FN count is significantly lower than v18 while FP count remains near zero.
- **Schema-truth precision near 1.0** — very few false merges.
- **Schema-truth recall increasing** — system correctly identifies expected canonical groupings.
- **Lexicon-quality FN clearly separated** — high lexicon-quality FN does NOT indicate system failure.
- **Consolidated reviewer CSV contains all four quadrants** (TP/FP/FN/TN) for complete audit coverage.

### Warning Signs

- Many `HUMAN_REVIEW_GT_CONFLICT` rows.
- High number of semantic veto conflicts.
- Unexpected payload fields in `schema_deltas_audit_v18_2.csv`.
- Large gap between strict recall and reviewer-diagnosed recall.
- DBNF marked supported without explicit drift evidence.
- Promotion rules creating false positives — check FP examples in alias evaluation audit.
- **Schema-truth FP > 0** — investigate whether false merges are due to incorrect `canonical_key` assignments or overly aggressive promotion paths.
- **Schema-truth FN high** — check whether candidate retriever is missing expected pairs (look for FN rows with `DecisionType = NOT_EVALUATED` in the consolidated reviewer CSV).
- **Lexicon-quality FN very high** — indicates the ground-truth alias file needs updating, not necessarily a system problem.

## 17. Version Guidance

### v16 Role

v16 introduced valuable evaluator fixes, especially around alias evaluation correctness and conservative metrics.

### v17 Role

v17 combined v14/v15 implementation backbone with v16 evaluator correctness improvements and Human Review governance.

### v18 Role

v18 introduced the KeyFix remediation with CMNF_COMPATIBILITY_MATRIX, CanonicalPromotionPolicy stubs, AmountUnitGuard, and enhanced decision audit fields. However, v18 still suffered from high FN (~238) due to overly strict decision gates and semantic_vetoes blocking same-canonical pairs.

### v18.2 Role

v18.2 is the recommended working baseline because it:

- completes all v18 KeyFix FN-fix patches,
- preserves v17 precision-first governance,
- activates CanonicalPromotionPolicy with 3 auditable rules,
- adds 4 promotion paths before the strict ACCEPT_MERGE gate,
- fixes semantic_vetoes to not block same-canonical-key pairs,
- auto-allows cross-rail merges for global families,
- fixes is_broad_compatible_but_ambiguous for canonical_key matches,
- dramatically improves recall (~0.03 → ~0.80–0.90) while keeping precision near 1.0.

### v18.2 Role

v18.2 builds on v18.1's proven infrastructure and applies 3 targeted fixes plus 2 major new capabilities:

- **DBNF version_drift** now properly evaluated for C3 — returns `SUPPORTED` with drift metrics when `--dbnf_mode version_drift`.
- **C4 context safety** uses qualified transaction identifier bridge detection — `PARTIALLY_SUPPORTED` for legitimate cross-context bridges instead of blanket `NOT_SUPPORTED`.
- **Deduplication transparency** — `raw_predicted_pair_count` and `unique_predicted_pair_count` tracked separately in `self_checks`.
- **Candidate backend default** changed to `auto` (HNSW when available, pairwise fallback).

- **Domain-Independent Automatic Lexicon Learning** — `LearnedLexicon` eliminates most lexicon-quality FNs from provider prefixes, abbreviations, and alias-closure gaps without hardcoded domain knowledge.
- **DBNF improvements** — fixed mode handling, lexical normalization, adaptive tau, saturation detection, real two-model comparison.

v18.2 is the recommended working baseline for all paper and audit runs.

### Future Versions

Future versions may implement, evaluate, and export full semantic geometry metrics such as:

- canonical compactness,
- inter-canonical separation,
- semantic margin,
- partition leakage,
- SRS geometry evolution snapshots,
- SDNF-governed HNSW candidate graph diagnostics.

Those should remain future evaluated enhancements and should not be claimed from v18.2 scaffolding alone.

---

## 18. Summary

v18.2 is a reviewer-grade, precision-governed SDNF experiment harness that:

- preserves the full v17/v18 experiment structure,
- completes the v18 KeyFix FN remediation with 10 targeted patches,
- introduces CMNF_COMPATIBILITY_MATRIX for formal cross-rail policy,
- activates CanonicalPromotionPolicy for auditable review-to-accept promotion,
- adds four new promotion paths (CMNF canonical-safe, same-canonical+family, alias-hit, soft-match zone),
- fixes semantic_vetoes to not block same-canonical-key pairs,
- auto-allows cross-rail for global families (payment:amount, payment:currency),
- fixes is_broad_compatible_but_ambiguous for canonical_key matches,
- **introduces dual evaluation framework** separating schema-truth effectiveness from lexicon-quality closure,
- **replaces review queue with consolidated reviewer CSV** (TP/FP/FN/TN based on schema-truth canonical_key grouping),
- **adds schema-truth metrics** to console, out_audit, summary_audit, and alias_evaluation_audit,
- **relabels existing alias evaluation as lexicon-quality** with explicit separation from effectiveness FN,
- keeps strict and reviewer-diagnosed metrics separate,
- preserves output budgeting under the 15-file limit (no new output files added),
- keeps roadmap scaffolds ready without overclaiming,
- supports reproducible paper, audit, and minimal runs,
- reduces FN from ~238 to ~30-50 while preserving precision near 1.0.

- **fixes DBNF version_drift** to properly evaluate C3 claim with actual drift metrics,
- **adds C4 qualified transaction identifier bridge detection** with 3-tier claim logic (SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED),
- **adds deduplication transparency** with raw/unique predicted pair counts in self_checks,
- **defaults candidate_backend to auto** (HNSW when available),
- introduces **Domain-Independent Automatic Lexicon Learning** (`LearnedLexicon`) with three learning phases (prefix discovery, abbreviation expansion, equivalence class closure),
- introduces **DBNF improvements** with fixed mode handling, lexical normalization, adaptive tau, saturation detection, and real two-model comparison,
- adds `learned_lexicon_v18_2.json` persistence for audit/debug profiles,
- enhances candidate retrieval with `lexicon_closure` candidate source,
- adds `--dbnf_tau` CLI argument for user-specified drift threshold.

Recommended baseline file:

```text
unified_sdnf_experiment_hybrid_v18_2.py
```

Recommended README file:

```text
readMe_v18_2.md
```

