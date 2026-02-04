# sdnf_config.py
import json
import os
from typing import Dict, Any

CONFIG_PATH = "sdnf_config.json"

DEFAULT_CONFIG = {
    "EENF": {"tau": 0.01, "calibration_quantile": 0.95},
    "AANF": {"tau": 0.90, "calibration_quantile": 0.99},
    "CMNF": {"tau": 0.05, "calibration_quantile": 0.95},
    "DBNF": {"global_tau": 0.25, "calibration_quantile": 0.90, "adaptive_multiplier": 1.5},
    "ECNF": {"m_min": 3, "score_threshold": 0.55, "strong_score_threshold": 0.72},
    "RRNF": {"tau": 0.70},
    "PONF": {"tau": 0.10},
    "decision_policy": {
        "dbnf_fail_fraction": 0.5,
        "dbnf_median_factor": 2.0,
        "auto_merge_min_score": 0.80,
        "auto_merge_min_count": 4
    }
}


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # merge defaults for missing keys
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            return merged
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
