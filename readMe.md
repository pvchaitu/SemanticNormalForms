# SDNF Unified Experiment (Real Embeddings + HNSW ANN)

This repo contains a **single-file, reproducible experiment** for **Semantic Data Normal Forms (SDNF)** using:
- **Real embeddings** via **Sentence-Transformers**
- **Approximate Nearest Neighbor (ANN)** retrieval via **HNSW (hnswlib)**
- **Evidence-gated schema evolution (ECNF)** with multiple evidence modes
- **Schema + payload ingestion** from local folders (`data/` and `payloads/`)

> **Main script (latest):** `unified_sdnf_experiment_hybrid_v7.py`

---

## What the experiment does

1. **Bootstraps a canonical schema** from `data/INAmex.json` if present (otherwise first JSON file).
2. Builds **multi-level attribute name embeddings**:
   - **fine:** `encode(name)`
   - **abstract:** `encode(normalize(name))`
   - **contextual:** `encode(f"{name} in {context} context")`
   - concatenates and L2-normalizes
3. Builds an **HNSW index** over canonical attribute embeddings.
4. Ingests:
   - **Schema JSONs** from `--data_dir` (default: `data`)
   - **Payload JSONs** from `--payloads_dir` (default: `payloads`)
5. For each new field, proposes a merge via ANN and applies **ECNF gating**:
   - `merge_auto` vs `merge_pending` vs `create`
6. Optional **DBNF drift test** if `--drift_model` is provided.
7. Prints consolidated tables:
   - Run configuration
   - SDNF validation summary (expected vs actual + PASS/FAIL)
   - Per-source ingestion summary
   - Evidence availability
8. If `--evidence_mode all` is chosen, runs **all** modes and prints a **comparison table** including EENF/CMNF/DBNF actual vs tau.

---

## Repository layout

- `unified_sdnf_experiment_hybrid_v7.py` — unified experiment (single file)
- `data/` — schema JSON artifacts
- `payloads/` — payload JSON artifacts (values present; used by VSS/shape)
- `requirements.txt` — dependencies

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

### Default run (recommended)
Runs `hybrid` mode using defaults (`data/`, `payloads/`):
```bash
python unified_sdnf_experiment_hybrid_v7.py
```

### Specify folders explicitly
```bash
python unified_sdnf_experiment_hybrid_v7.py --data_dir ./data --payloads_dir ./payloads
```

### Compare all evidence modes (prints a comparison table)
```bash
python unified_sdnf_experiment_hybrid_v7.py --evidence_mode all
```

### Drift test (DBNF simulation)
Use a second embedding model to simulate a model upgrade:
```bash
python unified_sdnf_experiment_hybrid_v7.py --drift_model all-mpnet-base-v2
```

### Quiet/noisy console logs
By default, external HuggingFace/HTTP INFO noise is suppressed.
To show it:
```bash
python unified_sdnf_experiment_hybrid_v7.py --show_external
```

### Limit runtime for quick iteration
```bash
python unified_sdnf_experiment_hybrid_v7.py --max_schema_files 3 --max_payload_files 10 --max_fields 20
```

---

## CLI options (argparse)

Run this anytime:
```bash
python unified_sdnf_experiment_hybrid_v7.py --help
```

Key options:

- `--data_dir` (default: `data`)
  - Folder containing schema JSONs.

- `--payloads_dir` (default: `payloads`)
  - Folder containing payload JSONs (values present).

- `--model` (default: `all-MiniLM-L6-v2`)
  - Sentence-Transformer model used for embeddings.

- `--drift_model` (default: None)
  - Optional second model; enables DBNF drift comparison.

- `--contexts` (default: `Payments Risk`)
  - Space-separated context labels.

- `--evidence_mode` (default: `hybrid`)
  - Choices: `embed_only | vss | shape | hybrid | all`
  - `all` runs every mode and prints a comparison table.

- `--promote_sources` (default: 2)
  - Auto-promote pending merges after seeing the pair in N distinct sources.

- `--promote_hits` (default: 3)
  - Auto-promote pending merges after N repeated observations.

- `--max_schema_files`, `--max_payload_files`, `--max_fields`
  - Debug/runtime limiters.

- `--log_level` (default: INFO)
  - SDNF log verbosity.

- `--quiet_external` / `--show_external`
  - Control third-party library logging noise.

---

## Evidence modes (when to choose which)

All modes use ANN retrieval over real embeddings to propose merge candidates.

- `embed_only`
  - **Fastest**, name-embedding-only baseline. Good for quick smoke tests.

- `vss`
  - Adds **Value Semantic Signature** evidence derived from **values**.
  - Needs payload JSONs with values.

- `shape`
  - Adds **shape-token embedding** evidence derived from **values**.
  - Needs payload JSONs with values.

- `hybrid` (recommended)
  - Combines name signals + VSS + shape-token (best overall trade-off).

- `all`
  - Runs `embed_only`, `vss`, `shape`, `hybrid` sequentially and prints a comparison table.

> **Important:** VSS/shape evidence requires **payload-style JSON** (values present). If you ingest only schema JSONs, value evidence can be missing; the script prints evidence-availability guidance.

---

## Data expectations

The experiment supports two JSON styles:

### 1) Schema-style JSON
Contains an `attributes` list:
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
Flat dict of fields and values:
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
- Calibrated taus (EENF, AANF, CMNF, DBNF)
- Per-source ingestion summary (merges/pending/promoted/new)
- SDNF validation summary table (expected vs actual + PASS/FAIL)
- Evidence availability (how often value evidence was present)
- If `--evidence_mode all`: a multi-mode comparison table

---

## Research note
This is a research artifact intended for experimentation and reproducibility. Avoid embedding or logging raw sensitive values in production environments.
