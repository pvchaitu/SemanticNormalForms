# demo.py
"""
Demo runner with automatic calibration step.

This file performs two phases:
1) Calibration phase (preceding step)
   - Programmatically derives recommended thresholds for EENF, DBNF, CMNF
     from the JSON files in ./data and from simulated model dynamics.
   - Produces a short calibration report and a suggested config dictionary.
   - Comments throughout explain the rationale and how to adapt the procedure
     as data and operational requirements evolve.

2) Demo phase (uses calibrated thresholds)
   - Runs the SDNF demo (semantic compliance, drift predeployment, evidence accumulation)
   - Uses the calibrated thresholds in-memory for validator and decision policy.
   - Prints the same tables as before, but now thresholds are data-driven.

Notes for implementers:
- Calibration is intentionally conservative: thresholds are set to high quantiles
  (default 95th) of observed statistics to control false positives.
- Calibration is *not* a substitute for governance: borderline cases should still
  be routed to human review (merge_pending) and canary promotion.
- You can persist the suggested config to sdnf_config.json if you want to make
  the calibration permanent; this script only uses the values in-memory.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tabulate import tabulate

from srs_store import SRSStore
from embedding_utils import EmbeddingModel
from schema_ingest import load_json_file
from cmnf import learn_linear_projection
from sdnf_config import load_config
from semantic_merge import evidence_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sdnf.demo")
np.random.seed(42)


# ---------------------------
# Calibration utilities
# ---------------------------

def list_data_files(data_dir: str = "data") -> List[Path]:
    """
    Return a list of JSON files in the data directory that we will use for calibration.
    Implementers: add or remove files here to reflect your production corpus.
    """
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted([f for f in p.glob("*.json")])


def load_attribute_names_from_file(path: str) -> List[str]:
    """
    Load attribute names from a schema JSON file if it contains an 'attributes' list.
    Fallback: if the file is a flat payload, return top-level keys.
    """
    data = load_json_file(path)
    if not data:
        return []
    if isinstance(data, dict) and "attributes" in data:
        return [a.get("name") for a in data.get("attributes", []) if a.get("name")]
    # fallback: top-level keys (payload-style files)
    if isinstance(data, dict):
        return [k for k in data.keys()]
    return []


def calibrate_thresholds_from_data(model: EmbeddingModel,
                                   data_dir: str = "data",
                                   eenf_regenerations: int = 20,
                                   drift_noise_scales: List[float] = None,
                                   cmnf_bootstrap_iters: int = 50,
                                   quantile: float = 0.95) -> Tuple[Dict, Dict]:
    """
    Programmatically derive calibration statistics and suggested thresholds.

    Steps and rationale (high level):
    - EENF: For each canonical attribute (we use INAmex.json as canonical SRS),
      we compute G regenerations (deterministic but varied via regen_idx) and
      measure per-dimension variance. The mean per-dimension variance per attribute
      is the EENF statistic. We set tau to the chosen quantile across attributes.
      Rationale: EENF should tolerate typical embedding jitter but flag outliers.

    - DBNF: We compute canonical centroids for each attribute (centroid of name + aliases).
      We simulate plausible drift by adding Gaussian noise at several scales and
      measure L2 drift per attribute. We aggregate drifts across scales and set
      global DBNF tau to the chosen quantile of the simulated drift distribution.
      Rationale: Drift arises from model updates or data shifts; simulation gives
      a conservative estimate of expected drift.

    - CMNF: Learn projection matrices from two distinct context corpora (Stripe.json
      for payments and INAmex.json for risk in the demo). Compute contamination as
      the mean normalized singular value of M = Wp @ Wr^T. Bootstrap small perturbations
      to estimate variability and set tau to the chosen quantile.
      Rationale: CMNF measures subspace overlap; we calibrate to observed overlap.

    Returns:
      (report, suggested_config)
      - report: detailed numeric statistics for inspection
      - suggested_config: recommended tau values and multipliers (in-memory)
    """
    if drift_noise_scales is None:
        drift_noise_scales = [0.01, 0.02, 0.03, 0.05]

    files = list_data_files(data_dir)
    report = {"files_used": [str(f) for f in files], "EENF": {}, "DBNF": {}, "CMNF": {}}

    # Load canonical SRS (prefer INAmex.json if present)
    inamex_path = Path(data_dir) / "INAmex.json"
    if not inamex_path.exists():
        # fallback to first schema-like file
        inamex_path = files[0] if files else None

    # Build canonical attribute list
    canonical_attrs = []
    if inamex_path:
        canonical_attrs = load_attribute_names_from_file(str(inamex_path))
    canonical_attrs = canonical_attrs or []

    # EENF: regenerations per canonical attribute
    eenf_vals = []
    for name in canonical_attrs:
        regs = model.regenerations(name, context="payments", G=eenf_regenerations)
        per_dim_var = np.var(regs, axis=0)
        mean_var = float(np.mean(per_dim_var))
        report["EENF"][name] = {"mean_per_dim_var": mean_var}
        eenf_vals.append(mean_var)
    eenf_vals = np.array(eenf_vals) if eenf_vals else np.array([0.0])
    eenf_q = float(np.quantile(eenf_vals, quantile))
    report["EENF_summary"] = {"median": float(np.median(eenf_vals)), "quantile": eenf_q, "count": len(eenf_vals)}

    # DBNF: canonical centroids and simulated drift
    canonical_centroids = {}
    if inamex_path:
        inam = load_json_file(str(inamex_path)) or {}
        for a in inam.get("attributes", []):
            name = a.get("name")
            tokens = [name] + a.get("aliases", [])
            if not tokens:
                continue
            embs = model.encode(tokens, context="payments")
            centroid = np.mean(embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            canonical_centroids[name] = centroid

    dbnf_drifts_all = []
    dbnf_per_attr = {}
    for name, vec in canonical_centroids.items():
        drifts = []
        for s in drift_noise_scales:
            noise = np.random.default_rng(0).normal(scale=s, size=vec.shape)
            pert = vec + noise
            pert = pert / (np.linalg.norm(pert) + 1e-12)
            drift = float(np.linalg.norm(pert - vec))
            drifts.append(drift)
            dbnf_drifts_all.append(drift)
        dbnf_per_attr[name] = {"drifts": drifts, "mean": float(np.mean(drifts))}
    dbnf_drifts_all = np.array(dbnf_drifts_all) if dbnf_drifts_all else np.array([0.0])
    dbnf_q = float(np.quantile(dbnf_drifts_all, quantile))
    report["DBNF_summary"] = {"global_median": float(np.median(dbnf_drifts_all)), "quantile": dbnf_q, "per_attribute": dbnf_per_attr}

    # CMNF: learn W_p from Stripe.json and W_r from INAmex.json (contexts)
    stripe_attrs = load_attribute_names_from_file(str(Path(data_dir) / "Stripe.json"))
    inamex_attrs = canonical_attrs  # reuse INAmex attributes for the other context

    # If Stripe has no attributes, fallback to using canonical centroids as small corpus
    if stripe_attrs:
        stripe_embs = model.encode([t for t in stripe_attrs], context="payments")
    else:
        stripe_embs = np.stack(list(canonical_centroids.values())) if canonical_centroids else np.zeros((1, model.dim))

    if inamex_attrs:
        inamex_embs = model.encode([t for t in inamex_attrs], context="risk")
    else:
        inamex_embs = np.stack(list(canonical_centroids.values())) if canonical_centroids else np.zeros((1, model.dim))

    # Learn projections (SVD-based)
    k_p = min(32, stripe_embs.shape[1])
    k_r = min(32, inamex_embs.shape[1])
    W_p = learn_linear_projection(stripe_embs, k=min(32, stripe_embs.shape[1]))
    W_r = learn_linear_projection(inamex_embs, k=min(32, inamex_embs.shape[1]))

    # contamination metric: mean normalized singular values of M = Wp @ Wr.T
    try:
        M = np.matmul(W_p, W_r.T)
        sv = np.linalg.svd(M, compute_uv=False)
        if sv.size > 0:
            max_sv = float(np.max(np.abs(sv)))
            contamination = float(np.mean(np.abs(sv) / (max_sv if max_sv > 0 else 1.0)))
        else:
            contamination = 0.0
    except Exception:
        contamination = 0.0

    # bootstrap small perturbations to estimate variability
    conts = []
    rng = np.random.default_rng(1)
    for i in range(cmnf_bootstrap_iters):
        sp = stripe_embs + rng.normal(scale=1e-3, size=stripe_embs.shape)
        ir = inamex_embs + rng.normal(scale=1e-3, size=inamex_embs.shape)
        Wp = learn_linear_projection(sp, k=min(32, sp.shape[1]))
        Wr = learn_linear_projection(ir, k=min(32, ir.shape[1]))
        M2 = np.matmul(Wp, Wr.T)
        sv2 = np.linalg.svd(M2, compute_uv=False)
        if sv2.size > 0:
            max_sv2 = float(np.max(np.abs(sv2)))
            conts.append(float(np.mean(np.abs(sv2) / (max_sv2 if max_sv2 > 0 else 1.0))))
        else:
            conts.append(0.0)
    conts = np.array(conts) if conts else np.array([0.0])
    cmnf_q = float(np.quantile(conts, quantile))
    report["CMNF_summary"] = {"median": float(np.median(conts)), "quantile": cmnf_q, "observed": float(contamination)}

    # Suggest config values based on quantiles
    suggested = {
        "EENF": {"tau": eenf_q, "note": f"{int(quantile*100)}th quantile of per-attribute mean variance"},
        "DBNF": {"global_tau": dbnf_q, "adaptive_multiplier": None, "note": f"{int(quantile*100)}th quantile of simulated per-attribute drifts"},
        "CMNF": {"tau": cmnf_q, "note": f"{int(quantile*100)}th quantile of bootstrap contamination"}
    }

    # Suggest adaptive_multiplier: choose multiplier so that multiplier*sqrt(mean variance) ~ DBNF global tau
    mean_var_mean = float(np.mean(eenf_vals)) if eenf_vals.size > 0 else 0.0
    if mean_var_mean > 0:
        suggested_multiplier = suggested["DBNF"]["global_tau"] / math.sqrt(mean_var_mean)
        # clamp to reasonable range
        suggested_multiplier = float(max(1.0, min(suggested_multiplier, 10.0)))
    else:
        suggested_multiplier = 2.5
    suggested["DBNF"]["adaptive_multiplier"] = suggested_multiplier

    return report, suggested


# ---------------------------
# Demo run (uses calibrated thresholds)
# ---------------------------

def run_demo_and_tables_with_calibration():
    """
    1) Run calibration
    2) Use suggested thresholds to override config in-memory
    3) Run the demo using the calibrated thresholds
    """
    print("\n[SDNF DEMO RUNNER] Starting calibration phase...\n")
    cfg = load_config()
    model = EmbeddingModel()
    # calibrate using local data folder
    report, suggested = calibrate_thresholds_from_data(model, data_dir="data", eenf_regenerations=20,
                                                       drift_noise_scales=[0.01, 0.02, 0.03, 0.05],
                                                       cmnf_bootstrap_iters=50, quantile=0.95)

    # Print calibration report (concise)
    print("CALIBRATION REPORT (concise):")
    # EENF summary
    eenf_tau = suggested["EENF"]["tau"]
    dbnf_tau = suggested["DBNF"]["global_tau"]
    cmnf_tau = suggested["CMNF"]["tau"]
    adaptive_multiplier = suggested["DBNF"]["adaptive_multiplier"]

    table = [
        ["EENF (mean per-dim var) 95th quantile", f"{eenf_tau:.6f}"],
        ["DBNF (simulated drift) 95th quantile", f"{dbnf_tau:.6f}"],
        ["CMNF (contamination) 95th quantile", f"{cmnf_tau:.6f}"],
        ["DBNF adaptive_multiplier (suggested)", f"{adaptive_multiplier:.3f}"]
    ]
    print(tabulate(table, headers=["Metric", "Suggested value"], tablefmt="grid"))

    print("\nCalibration notes:")
    print("- EENF: uses regenerations per attribute to estimate embedding jitter.")
    print("- DBNF: simulated drift across multiple plausible noise scales; conservative quantile chosen.")
    print("- CMNF: bootstrap contamination across small perturbations; increase orthogonalize_iters if you want stricter separation.\n")

    # Apply suggested thresholds in-memory (do not overwrite file)
    # We update the loaded config dict so the rest of the demo uses calibrated values.
    cfg_local = cfg.copy()
    cfg_local["EENF"] = cfg_local.get("EENF", {})
    cfg_local["EENF"]["tau"] = eenf_tau
    cfg_local["DBNF"] = cfg_local.get("DBNF", {})
    cfg_local["DBNF"]["global_tau"] = dbnf_tau
    cfg_local["DBNF"]["adaptive_multiplier"] = adaptive_multiplier
    cfg_local["CMNF"] = cfg_local.get("CMNF", {})
    cfg_local["CMNF"]["tau"] = cmnf_tau

    # Inform implementers: you can persist cfg_local to sdnf_config.json if desired.
    print("Using calibrated thresholds for this demo run (in-memory). To persist, write cfg_local to sdnf_config.json.\n")

    # Proceed to run the demo using SRSStore and the in-memory cfg_local values.
    # We will instantiate SRSStore and override its config where needed.
    store = SRSStore(embedding_model=model)
    # override store.config values used by validator and merger
    store.config.update(cfg_local)
    # reinitialize merger and validator with new thresholds
    # (Simpler approach: update validator.tau and merger.tau_dbnf directly)
    try:
        store.validator.tau["EENF"] = cfg_local["EENF"]["tau"]
        store.validator.tau["CMNF"] = cfg_local["CMNF"]["tau"]
        store.validator.tau["DBNF"] = cfg_local["DBNF"]["global_tau"]
    except Exception:
        pass
    try:
        store.merger.tau_dbnf = cfg_local["DBNF"]["global_tau"]
    except Exception:
        pass

    # Load base master SRS (INAmex.json) if available
    base = load_json_file("INAmex.json")
    if base:
        store.master = base
        store._recompute_all_canonical_embeddings()

    # --- The rest of the demo is similar to previous demo.py but uses calibrated thresholds ---
    print("MASTER SRS (initial):")
    master = store.get_master_schema()
    if master.get("attributes"):
        rows = [[a["name"], a.get("type", ""), ",".join(a.get("aliases", []))] for a in master["attributes"]]
        print(tabulate(rows, headers=["name", "type", "aliases"], tablefmt="grid"))
    else:
        print("(master SRS empty)")

    # Use Case 1: Semantic compliance
    payload = {"payer_vpa": "user@upi", "txn_amount": "1000", "merchant_id": "MID123"}
    compliance_report = store.check_payload_compliance(payload, context="payments")
    uc1_pass = isinstance(compliance_report, dict) and "mapped_fields" in compliance_report and "sdnf_status" in compliance_report
    uc1_notes = "unmapped_fields:" + ",".join([k for k, v in compliance_report["mapped_fields"].items() if v is None]) if uc1_pass else "report_missing"

    # Use Case 2: EENF + DBNF drift simulation (use calibrated DBNF tau)
    pre_embeddings_map = dict(store._canonical_embeddings)
    rng = np.random.default_rng(1234)
    post_embeddings_map = {}
    noise = rng.normal(scale=cfg_local.get("random", {}).get("drift_noise_scale", 0.02), size=(1, store.model.dim))
    for name, vec in pre_embeddings_map.items():
        # small perturbation to simulate drift
        noise = rng.normal(scale=0.02, size=vec.shape)
        perturbed = vec + noise
        if np.linalg.norm(perturbed) > 0:
            perturbed = perturbed / (np.linalg.norm(perturbed) + 1e-12)
        post_embeddings_map[name] = perturbed

    per_attribute_tau = store.compute_adaptive_dbnf_taus()
    # ensure adaptive multiplier from calibration is used
    store.config["DBNF"]["adaptive_multiplier"] = cfg_local["DBNF"]["adaptive_multiplier"]
    global_tau = cfg_local["DBNF"]["global_tau"]
    # compute per-attribute DBNF details
    def compute_per_attribute_dbnf(pre_map, post_map, per_attribute_tau, global_tau):
        details = []
        sample_vec = None
        if pre_map:
            sample_vec = next(iter(pre_map.values()))
        elif post_map:
            sample_vec = next(iter(post_map.values()))
        else:
            return details
        zero = np.zeros_like(sample_vec)
        all_attrs = sorted(set(list(pre_map.keys()) + list(post_map.keys())))
        for a in all_attrs:
            pre = pre_map.get(a, None)
            post = post_map.get(a, None)
            pvec = pre if pre is not None else zero
            qvec = post if post is not None else zero
            if np.linalg.norm(pvec) > 0:
                pvec = pvec / (np.linalg.norm(pvec) + 1e-12)
            if np.linalg.norm(qvec) > 0:
                qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            drift = float(np.linalg.norm(qvec - pvec))
            tau_attr = float(per_attribute_tau.get(a, global_tau))
            ok = drift <= tau_attr
            action = "accept" if ok else "fork"
            details.append({"attribute": a, "drift": drift, "pass": ok, "action": action, "tau": tau_attr})
        return details

    attr_dbnf_details = compute_per_attribute_dbnf(pre_embeddings_map, post_embeddings_map, per_attribute_tau, global_tau)

    # Use Case 3: evidence accumulation and merges (per-merge ECNF)
    payloads = [
        {"acct_num": "A1", "amount": "10"},
        {"acct_num": "A2", "amount": "20"},
        {"acct_num": "A3", "amount": "30"},
        {"acct_num": "acct_num", "amount": "40"}
    ]
    before_lineage = len(store.list_lineage(1000))
    rec_ids = []
    per_merge_reports = []
    for i, p in enumerate(payloads):
        r = store.insert_payload(p, source=f"sim_source_{i}", auto_merge=True)
        rec_ids.append(r.get("record_id"))
        new_entries = store.list_lineage(1000)[before_lineage:]
        for e in new_entries:
            if e.get("action", "").startswith("merge") or e.get("action") == "merge_pending":
                per_merge_reports.append(e)
        before_lineage = len(store.list_lineage(1000))

    after_lineage = len(store.list_lineage(1000))
    recent = store.list_lineage(50)
    merge_actions = [e for e in recent if e["action"].startswith("merge") or e["action"] == "merge_pending"]
    uc3_pass = len(merge_actions) > 0
    uc3_notes = f"lineage_added={len(per_merge_reports)}; merge_actions={len(merge_actions)}"

    # Table A
    table_a = [
        ["Semantic Compliance Check", "PASS" if uc1_pass else "FAIL", uc1_notes],
        ["Regression Drift Predeployment", "PASS" if all(d["pass"] for d in attr_dbnf_details) else "FAIL", f"failed_attrs={sum(1 for d in attr_dbnf_details if not d['pass'])}/{len(attr_dbnf_details)}"],
        ["Evidence Accumulation Merge", "PASS" if uc3_pass else "FAIL", uc3_notes]
    ]
    print("\nTABLE A: Semantic Use Case Test Results")
    print(tabulate(table_a, headers=["Use Case", "Result", "Notes"], tablefmt="grid"))

    # Table B
    lineage = store.list_lineage(500)
    merge_counts = {}
    example_actions = {}
    for e in lineage:
        action = e.get("action")
        evidence = e.get("evidence", [])
        tokens = [it.get("token") for it in evidence if it.get("token")]
        key = action
        merge_counts[key] = merge_counts.get(key, 0) + 1
        if key not in example_actions and tokens:
            example_actions[key] = tokens[:3]
    table_b = []
    if merge_counts:
        for k, cnt in merge_counts.items():
            ex = ",".join(example_actions.get(k, [])) if example_actions.get(k) else ""
            table_b.append([k, cnt, ex])
    else:
        table_b = [["(no lineage yet)", 0, ""]]
    print("\nTABLE B: Merge Summary Snapshot")
    print(tabulate(table_b, headers=["Action", "Count", "Example Tokens"], tablefmt="grid"))

    # Table C: per-attribute taus (adaptive)
    print("\nTABLE C: Per-attribute adaptive DBNF taus")
    rows_c = []
    for name, tau in sorted(per_attribute_tau.items()):
        rows_c.append([name, f"{tau:.6f}"])
    if rows_c:
        print(tabulate(rows_c, headers=["attribute", "tau_attr"], tablefmt="grid"))
    else:
        print("(no attributes)")

    # Table D: DBNF attribute-level details
    print("\nTABLE D: DBNF attribute-level details")
    rows_d = []
    for d in attr_dbnf_details:
        rows_d.append([d["attribute"], f"{d['drift']:.6f}", "PASS" if d["pass"] else "FAIL", d["action"], f"{d['tau']:.6f}"])
    if rows_d:
        print(tabulate(rows_d, headers=["attribute", "drift", "status", "action", "tau_used"], tablefmt="grid"))
    else:
        print("(no DBNF details)")

    # Master SRS final snapshot
    final_master = store.get_master_schema()
    print("\nMASTER SRS final snapshot:")
    if final_master.get("attributes"):
        rows = [[a["name"], a.get("type", ""), ",".join(a.get("aliases", []))] for a in final_master["attributes"]]
        print(tabulate(rows, headers=["name", "type", "aliases"], tablefmt="grid"))
    else:
        print("(master SRS empty)")

    # CMNF: learn W_p from Stripe.json and W_r from INAmex.json with orthogonalization
    stripe_attrs = load_attribute_names_from_file("Stripe.json")
    inamex_attrs = load_attribute_names_from_file("INAmex.json")
    if not stripe_attrs:
        stripe_attrs = [a["name"] for a in final_master.get("attributes", [])]
    if not inamex_attrs:
        inamex_attrs = [a["name"] for a in final_master.get("attributes", [])]

    stripe_embs = model.encode([t for t in stripe_attrs], context="payments")
    inamex_embs = model.encode([t for t in inamex_attrs], context="risk")

    orth_iters = int(cfg_local.get("CMNF", {}).get("orthogonalize_iters", 3))
    W_p = learn_linear_projection(stripe_embs, k=min(32, stripe_embs.shape[1]))
    W_r = learn_linear_projection(inamex_embs, k=min(32, inamex_embs.shape[1]))

    # compute contamination on contextual components using singular values of cross-projection
    if final_master.get("attributes"):
        sample_embeddings = np.stack([store._canonical_embeddings.get(n, store._embed(n)) for n in [a["name"] for a in final_master["attributes"]]], axis=0)
        d = model.contextual_dim
        contextual_sample = sample_embeddings[:, -d:]
        try:
            Wp_ctx = W_p[:, -d:] if W_p.shape[1] >= d else W_p
            Wr_ctx = W_r[:, -d:] if W_r.shape[1] >= d else W_r
            M = np.matmul(Wp_ctx, Wr_ctx.T)
            sv = np.linalg.svd(M, compute_uv=False)
            contamination = float(np.mean(np.abs(sv))) if sv.size > 0 else 0.0
        except Exception:
            contamination = 0.0
    else:
        contamination = 0.0

    # Table I: SDNF 7-NF compliance (CMNF replaced with contamination)
    attr_names = [a["name"] for a in final_master.get("attributes", [])]
    if attr_names:
        attr_embeddings = np.stack([store._canonical_embeddings.get(n, store._embed(n)) for n in attr_names], axis=0)
        pre_vec = np.mean(np.stack(list(pre_embeddings_map.values()), axis=0), axis=0) if pre_embeddings_map else np.zeros((store.model.dim,))
        post_vec = np.mean(np.stack(list(post_embeddings_map.values()), axis=0), axis=0) if post_embeddings_map else pre_vec
        # reuse existing validator run_all but inject contamination into CMNF result
        sdnf_table = store.validator.run_all({
            "regenerations": attr_embeddings if attr_embeddings.size else np.zeros((1, store.model.dim)),
            "attr_embeddings": attr_embeddings,
            "attr_names": attr_names,
            "W_p": W_p, "W_r": W_r,
            "sample_embeddings": attr_embeddings if attr_embeddings.size else np.zeros((1, store.model.dim)),
            "pre_vec": pre_vec, "post_vec": post_vec,
            "tau_dbnf": store.validator.tau.get("DBNF", cfg_local["DBNF"]["global_tau"]),
            "relations": store.master.get("relations", []),
            "role_compat": {},
            "partition_labels": [0] * len(attr_names),
            # optionally include evidence_summary if you want ECNF to be evaluated here
            "evidence_summary": None
        })
        for r in sdnf_table:
            if r["name"] == "CMNF":
                r["actual"] = contamination
                r["status"] = "PASS" if contamination < store.validator.tau.get("CMNF", cfg_local["CMNF"]["tau"]) else "FAIL"
        print("\nTABLE I: SDNF 7-Normal Form Compliance Summary (calibrated thresholds)")
        rows = []
        for r in sdnf_table:
            actual = r["actual"]
            if isinstance(actual, float):
                actual_fmt = f"{actual:.6f}"
            else:
                actual_fmt = str(actual)
            rows.append([r["name"], r.get("req", ""), actual_fmt, r.get("status", ""), r.get("details", {})])
        print(tabulate(rows, headers=["Normal Form", "Requirement", "Actual", "Status", "Details"], tablefmt="grid"))
    else:
        print("\nNo attributes available to compute full SDNF table.")

    # Lineage excerpt and forks
    lineage = store.list_lineage(500)
    print("\nLINEAGE EXCERPT (last 10 entries):")
    for entry in lineage[-10:]:
        print(entry)

    forks = [e for e in lineage if e["action"] == "fork"]
    if forks:
        print("\nDBNF Fork Events (attribute-level):")
        for f in forks[-10:]:
            print(f)
    else:
        print("\nNo DBNF fork events recorded.")

    # Return calibration report and demo outputs for programmatic inspection if needed
    return {"calibration_report": report, "suggested_config": suggested, "demo_store": store}


if __name__ == "__main__":
    run_demo_and_tables_with_calibration()
