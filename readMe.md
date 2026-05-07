# SDNF Unified Experiment — Schema-First Master Payment SRS Harness v15

## 1. Overview
This repository contains a **single-file, reproducible experiment** for validating:
**Semantic Data Normal Forms (SDNF): Extending Normalization Theory to Vector Embedding Spaces**.

The main experiment file:

unified_sdnf_experiment_hybrid_v15.py

v15 evolves v14 into a **production-grade, output-governed SDNF evaluation harness** with strict audit discipline and scalable evaluation design.

---

## 2. What’s New in v15

### 2.1 Output Budget Enforcement
- Hard **file output ceiling (default = 12, max = 15)**
- Centralized writer ensures **no uncontrolled file generation**

### 2.2 Output Profiles
| Profile | Files | Purpose |
|--------|------|--------|
| minimal | 3 | Quick sanity check |
| paper | 12 | Paper-ready reproducible outputs |
| audit | 13 | Reviewer-grade audit (includes debug ZIP) |
| debug | 14 | Full introspection including README export |

### 2.3 Dual Evaluation Tracks
- **Production Track** → precision-first (paper-safe)
- **Discovery Track** → recall-oriented (exploratory)

### 2.4 DBNF Enhancements
Two explicit modes:

1. **version_drift (paper claim)**
   - Same-model version evolution
   - Supports **drift detection + fork governance**

2. **migration (utility only)**
   - Cross-model transitions (e.g., MiniLM → MPNet)
   - Uses **rank-order geometry preservation**, not cosine

### 2.5 EENF Stress Testing
- Deterministic mode
- Perturbation stress-test mode (G sweep)
- Reports variance reduction metrics

### 2.6 Payload Compliance Diagnostics
- Constraint validation with **pattern mismatch reasoning**
- Schema vs payload mismatch detection
- Decision types:
  - ALLOW
  - REJECT
  - DEFER_REVIEW
  - ROUTE_SCHEMA_ONBOARDING

### 2.7 HNSW Scale Audit
- Brute-force vs ANN comparison
- Optional `hnswlib` integration
- Partition-aware estimation

---

## 3. Core v15 Outputs

### Mandatory Outputs (paper profile)
1. out_audit_v15.txt
2. run_manifest_v15.json
3. summary_audit_v15.json
4. srs_evolved_schema_v15.compact.json
5. schema_ingestion_audit_v15.csv
6. field_evidence_audit_v15.csv
7. schema_deltas_audit_v15.csv
8. decisions_audit_v15.csv
9. alias_evaluation_audit_v15.csv
10. payload_compliance_audit_v15.csv
11. normal_forms_and_claims_audit_v15.csv
12. scale_timing_drift_audit_v15.csv

Optional:
- sdnf_debug_bundle_v15.zip
- readme_v15.md

---

## 4. Core Design Principles (Unchanged from v14)
- Schema-first governance
- Payload as evidence (not schema source)
- SDNF normal forms enforce safety:
  - AANF
  - ECNF
  - RRNF
  - CMNF
  - DBNF
  - PONF
- Strict semantic vetoes
- Partition-aware merging

---

## 5. Repository Structure

```
.
├── unified_sdnf_experiment_hybrid_v15.py
├── readme.md
├── requirements.txt
├── data/
├── payloads/payment/
├── ground_truth_aliases_closed_world_v12.json
└── drift_ground_truth.json
```

---

## 6. Setup

### Linux / macOS
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```
python -m venv venv
venv\Scripts\activate
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional dependencies:
- numpy
- sentence-transformers
- hnswlib

Fallback embedding:
- Deterministic hashing (offline-safe)

---

## 7. Recommended Runs

### Paper Run
```
python unified_sdnf_experiment_hybrid_v15.py ^
--output_profile paper ^
--schemas_dir data ^
--payloads_root payloads/payment ^
--seed_srs_schema INAmex.schema.json ^
--ground_truth_aliases ground_truth_aliases_closed_world_v12.json ^
--ground_truth_closed_world ^
--evaluation_track both ^
--dbnf_mode version_drift ^
--eenf_mode perturbation_stress_test
```

### Full Audit Run
```
python unified_sdnf_experiment_hybrid_v15.py ^
--output_profile audit ^
--schemas_dir data ^
--payloads_root payloads/payment ^
--seed_srs_schema INAmex.schema.json ^
--ground_truth_aliases ground_truth_aliases_closed_world_v12.json ^
--ground_truth_closed_world ^
--evaluation_track both ^
--dbnf_mode version_drift ^
--eenf_mode deterministic_report ^
--measure_timing
```

### Minimal Run
```
python unified_sdnf_experiment_hybrid_v15.py --output_profile minimal
```

---

## 8. Key Differences vs v14

| Area | v14 | v15 |
|------|----|----|
| Output files | 30+ | capped ≤15 |
| DBNF | delta detection | version drift + migration |
| Evaluation | single | production + discovery |
| EENF | static | stress testing |
| Scale | conceptual | measurable HNSW audit |
| Output quality | verbose | paper-optimized |

---

## 9. Research Positioning (Important)

### Paper Claims (Supported)
- DBNF version drift governance
- Precision-first merging
- Normal-form-driven safety
- Payload compliance system

### Utilities (NOT paper claims)
- DBNF migration
- Debug bundle
- Deep introspection outputs

---

## 10. Summary

v15 transforms the SDNF experiment into a **paper-ready, audit-safe, and scale-aware system** by:

- Enforcing output discipline
- Separating production vs discovery semantics
- Formalizing DBNF into publishable vs utility modes
- Adding measurable scale characteristics (HNSW)

This version is the **recommended baseline for TMLR/JMLR submission**.
