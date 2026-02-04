# calibrate_thresholds.py
"""
Calibration job to compute adaptive thresholds from historical data.

Usage:
    python calibrate_thresholds.py

This script is intentionally simple: it loads sample metrics from a
small local store (or synthetic generator) and updates sdnf_config.json.
In production, replace `collect_samples()` with real telemetry queries.
"""

import numpy as np
from sdnf_config import load_config, save_config
import json
import os

# Replace this with real telemetry collection in production
def collect_samples():
    # Synthetic example: distributions for each metric
    rng = np.random.default_rng(123)
    samples = {
        "EENF": np.abs(rng.normal(0.001, 0.0005, size=1000)).tolist(),
        "AANF": np.clip(rng.normal(0.5, 0.1, size=1000), 0, 1).tolist(),
        "CMNF": np.abs(rng.normal(0.03, 0.01, size=1000)).tolist(),
        "DBNF": np.abs(rng.normal(0.12, 0.08, size=1000)).tolist(),
        "PONF": np.abs(rng.normal(0.02, 0.01, size=1000)).tolist()
    }
    return samples

def calibrate():
    cfg = load_config()
    samples = collect_samples()
    for key in ["EENF", "AANF", "CMNF", "DBNF", "PONF"]:
        if key in samples:
            q = cfg.get(key, {}).get("calibration_quantile", 0.95)
            val = float(np.quantile(samples[key], q))
            # smoothing: blend with existing
            prev = cfg.get(key, {}).get("tau", None)
            if prev is None:
                new_tau = val
            else:
                alpha = 0.3
                new_tau = alpha * val + (1 - alpha) * prev
            cfg.setdefault(key, {})["tau"] = float(new_tau)
            print(f"Calibrated {key}: quantile({q})={val:.6f} -> tau={new_tau:.6f}")
    save_config(cfg)
    print("Saved updated config to sdnf_config.json")

if __name__ == "__main__":
    calibrate()
