## SDNF Unified Experiment — Schema-First Master Payment SRS Harness v16

### 1. Overview

This repository contains a **single-file, reproducible experiment** for validating:
**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**.

Main experiment file:
unified_sdnf_experiment_hybrid_v16.py

Version v16 evolves v15 into a **reviewer-safe, precision-governed SDNF evaluation harness** with:
- strict correctness discipline
- explicit claim validation
- conservative evaluation semantics
- improved alias evaluation fidelity

---

### 2. What’s New in v16

#### 2.1 Alias Evaluation Fixes (Critical)
- Self-pairs excluded from metrics (diagnostics-only)
- Duplicate alias pairs normalized and deduplicated
- Separation of:
  - **Pair-based evaluation**
  - **Canonical cluster membership evaluation**
- Prevents inflated precision/recall

#### 2.2 Ground Truth Repair Modes
New controlled evaluation mode:

| Mode | Behavior |
|------|---------|
| closed_world_only | Strict evaluation (default, paper-safe) |
| schema_supported_review | Flags potential GT gaps for reviewer |
| schema_supported_include | Includes schema-supported missing aliases |

✅ No implicit expansion — always controlled

#### 2.3 False Positive / False Negative Diagnostics
- FP classification by root cause:
  - Algorithmic error (counted in strict precision)
  - Likely GT gap (excluded from reviewer precision)
- FN enhancements:
  - Optional semantic-veto counting via:
    --count_semantic_veto_conflicts_as_fn

#### 2.4 Cross-Context Merge Safety
- Explicit detection of unsafe merges across contexts
- New metric:
  - **cross_rail_merge_rate**
- Any unsafe merges → claim compliance failure

#### 2.5 Claim & Normal Form Safeguards
Every paper claim now explicitly labeled:

- SUPPORTED
- PARTIALLY_SUPPORTED
- REVISED
- NOT_SUPPORTED
- NOT_APPLICABLE
- NOT_EVALUATED

Important:
- **EENF** → NOT_APPLICABLE (if deterministic run)
- **DBNF** → NOT_EVALUATED without drift ground truth

✅ Ensures fully reviewer-auditable claims

---

### 3. Output Profiles (Maintained with Discipline)

| Profile | Files | Purpose |
|--------|------|--------|
| minimal | 3 | Quick validation |
| paper | ~12 | Paper-ready outputs |
| audit | ~13 | Full diagnostics |
| debug | ≤15 | Deep introspection |

✅ Output cap preserved (≤15 files)

---

### 4. Core Outputs (v16)

#### Mandatory (paper profile)
- out_audit_v16.txt
- run_manifest_v16.json
- summary_audit_v16.json
- srs_evolved_schema_v16.compact.json
- schema_ingestion_audit_v16.csv
- field_evidence_audit_v16.csv
- decisions_audit_v16.csv
- alias_evaluation_audit_v16.csv
- payload_compliance_audit_v16.csv
- normal_forms_and_claims_audit_v16.csv
- scale_timing_drift_audit_v16.csv

#### Optional
- sdnf_debug_bundle_v16.zip

---

### 5. Core Design Principles (Strictly Preserved)

- Schema-first governance
- Payload = evidence (NOT schema)
- Normal-form safety enforcement:
  - AANF
  - ECNF
  - RRNF
  - CMNF
  - DBNF
  - PONF
- Semantic veto enforcement
- Context-aware merging (no unsafe cross-rail joins)
- Conservative evaluation philosophy

---

### 6. Key Improvements Over v15

| Area | v15 | v16 |
|------|-----|-----|
| Alias evaluation | basic pair metrics | corrected + deduplicated + separated |
| Ground truth | static | repair modes with governance |
| Precision | inflated risk | reviewer-safe precision |
| FP handling | unified | root-cause classified |
| FN handling | basic | semantic veto-aware |
| Cross-context safety | implicit | explicit + enforced |
| Claim validation | implicit | explicit statuses |

---

### 7. Setup

#### Linux / macOS
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional:
- numpy
- sentence-transformers
- hnswlib

Fallback:
- deterministic hashing (offline safe)

---

### 8. Recommended Runs (v16)

#### Paper-Grade Run (Strict Closed World)
```
python unified_sdnf_experiment_hybrid_v16.py ^
  --profile paper ^
  --ground_truth_repair_mode closed_world_only
```

#### Expanded Evaluation Run
```
python unified_sdnf_experiment_hybrid_v16.py ^
  --profile paper ^
  --ground_truth_repair_mode schema_supported_include
```

#### Full Audit Run
```
python unified_sdnf_experiment_hybrid_v16.py ^
  --profile audit ^
  --ground_truth_repair_mode schema_supported_review ^
  --count_semantic_veto_conflicts_as_fn
```

---

### 9. Research Positioning (v16 Clarification)

#### Paper-Valid Claims
- Precision-first semantic merging
- Normal-form-based governance
- Controlled alias evaluation
- Safe schema evolution

#### Conditionally Valid
- DBNF (requires drift ground truth)
- EENF (requires perturbation evidence)

#### Not Paper Claims
- Debug bundles
- Repair modes (evaluation utilities)

---

### 10. Summary

v16 represents a **reviewer-safe evolution** of SDNF experimentation by:
- fixing alias evaluation correctness
- enforcing conservative precision accounting
- introducing ground-truth governance
- explicitly validating all research claims

✅ This version is the **recommended baseline for TMLR submission**.

