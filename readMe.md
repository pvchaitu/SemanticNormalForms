# SDNF Unified Experiment (Real Embeddings + HNSW ANN)

This repo contains a **single-file, reproducible experiment** for **Semantic Data Normal Forms (SDNF)** aligned to the paper claims:

- **Real embeddings** via **Sentence-Transformers**
- **Approximate Nearest Neighbor (ANN)** retrieval via **HNSW (hnswlib)**
- **Evidence-gated schema evolution** (ECNF) with configurable evidence modes:
  - `embed_only` (baseline)
  - `vss` (Value Semantic Signature)
  - `shape` (Shape-token embedding)
  - `hybrid` (default: VSS + shape-token)
- **JSON-driven ingestion** from `./data` (supports both schema-style and payload-style JSON)

> **Main script:** `unified_sdnf_experiment_hybrid_v1.py`

---

## What the experiment does

1. **Bootstraps a canonical schema** from `data/INAmex.json` if present (otherwise first JSON file).
2. Builds **multi-level attribute name embeddings** per the SDNF/SRS narrative:
   - *fine:* `encode(name)`
   - *abstract:* `encode(normalize(name))`
   - *contextual:* `encode(f"{name} in {context} context")`
   - concatenate + L2 normalize
3. Builds an **HNSW index** over canonical attribute embeddings.
4. Ingests each JSON file and for each new field:
   - retrieves top ANN candidate(s)
   - computes evidence based on `--evidence_mode`
   - applies **ECNF gating**: `merge_auto` vs `merge_pending` vs `create`
5. Optional **DBNF drift test** (if `--drift_model` is provided).
6. Prints **concise logs**: per-file ingest summary + final metrics summary.

---

## Repository layout

- `unified_sdnf_experiment_hybrid_v1.py` — the unified experiment (single file)
- `data/` — JSON artifacts used by the experiment
- `requirements.txt` — minimal dependencies to run the experiment

---

## Quick start

### 1) Create & activate a virtual environment

**Linux/macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to run

### Run default (recommended): **hybrid evidence**

```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --contexts Payments Risk
```

### Compare modes quickly

Baseline (**embedding-only**):
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --evidence_mode embed_only
```

Value Semantic Signature (VSS):
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --evidence_mode vss
```

Shape-token embedding evidence:
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --evidence_mode shape
```

Hybrid (default):
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --evidence_mode hybrid
```

### Drift test (DBNF simulation)

Use a second embedding model to simulate a model upgrade and trigger drift detection:
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --drift_model all-mpnet-base-v2
```

### More diagnostics

Increase verbosity:
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --log_level DEBUG
```

Limit runtime for quick iteration:
```bash
python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --max_files 3 --max_fields 20
```

---

## Evidence modes (what changes)

All modes use **ANN retrieval** over **real embeddings** to propose merge candidates.

- `embed_only`: uses only embedding similarity (baseline; expect more false merges).
- `vss`: adds **Value Semantic Signature** evidence derived from values (domain-agnostic numeric signature).
- `shape`: adds **shape-token embedding** evidence derived from values (embeds shape tokens, not raw values).
- `hybrid`: combines `vss` + `shape` (recommended; best precision/recall trade-off in most cases).

> **Important:** VSS/shape evidence requires **payload-style JSON** (values present). For schema-only JSON (e.g., files that only define attribute metadata), value evidence may be missing. The script prints guidance if value evidence is missing for a large fraction of fields.

---

## Data folder expectations

The experiment supports two JSON styles:

### 1) Schema-style JSON
Example structure (contains an `attributes` list):
```json
{
  "schema_id": "INAmex_v1",
  "entity": "PaymentCard",
  "attributes": [
    {"name": "PrimaryAccountNumber", "aliases": ["pan", "cardNumber"]},
    {"name": "ExpirationDate", "aliases": ["exp"]}
  ]
}
```

### 2) Payload-style JSON
Example structure (flat dict of fields):
```json
{
  "pan": "4111111111111111",
  "exp": "12/26",
  "cvv": "123"
}
```

---

## Output (what you’ll see)

At `INFO` level, the script prints:

- Calibrated taus (EENF, AANF, DBNF)
- HNSW index build summary
- Per-file ingestion summary (`fields`, `merges`, `pending`, `new`)
- DBNF drift summary (if enabled)
- Evidence-availability guidance (if value evidence is often missing)
- Final summary (`canon`, `aliases_total`, `redundancy_reduction`, `merges`, `pending`, `forks`)
- Up to 5 example merges

---

## Reproducibility tips

- Prefer running with a fixed environment (venv / container)
- Use `--max_files` and `--max_fields` to iterate quickly
- Keep the `data/` set consistent across runs when reporting numbers in the paper

---

## License / Research note

This is a research artifact intended for experimentation and reproducibility. You should avoid embedding or logging raw sensitive values in production environments.
