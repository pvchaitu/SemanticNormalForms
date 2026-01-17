# semantic_merge.py
import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger("sdnf.merge")


def evidence_score(items: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Weighted aggregation of evidence items.
    Default weights reflect relative trust: nn > ontology > cooccurrence > heuristics.
    """
    if weights is None:
        weights = {"nn": 0.5, "ontology": 0.3, "cooccurrence": 0.15, "type_match": 0.05}
    total = 0.0
    for it in items:
        t = it.get("type", "nn")
        s = float(it.get("score", 0.0))
        total += weights.get(t, 0.0) * s
    # Clip to [0,1]
    return max(0.0, min(1.0, total))


class SemanticMerger:
    """
    Responsible for AANF (alias detection/merge), ECNF checks, and DBNF drift testing.

    - evidence_map: expected to be {derived_attr_name: [evidence_items...]}
      where each evidence_item has keys: type, token, score, source
    - Maintains an in-memory lineage log; in production this should be persisted.
    """

    def __init__(self, m_min: int = 3, tau_dbnf: float = 0.15):
        self.m_min = int(m_min)
        self.tau_dbnf = float(tau_dbnf)
        self.lineage: List[Dict[str, Any]] = []

    def check_ecnf(self, evidence_items: List[Dict[str, Any]]) -> Tuple[bool, int, float]:
        """Return (pass, count, aggregate_score)."""
        count = len(evidence_items or [])
        score = evidence_score(evidence_items or [])
        return (count >= self.m_min and score > 0.0), count, score

    def check_dbnf(self, pre_vec: Optional[np.ndarray], post_vec: Optional[np.ndarray]) -> Tuple[bool, float]:
        """Compute L2 drift and compare with tau_dbnf. If embeddings are normalized,
        drift is in [0, 2]."""
        if pre_vec is None or post_vec is None:
            return True, 0.0
        drift = float(np.linalg.norm(pre_vec - post_vec))
        return (drift <= self.tau_dbnf), drift

    def apply_merge(self, base_srs: Dict[str, Any], derived_schema: Dict[str, Any], evidence_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Apply alias merges from derived_schema into base_srs when ECNF satisfied.
        Returns the updated SRS (shallow copy) and appends lineage entries.
        """
        # Ensure base has 'attributes' and alias lists
        post_attrs = [dict(a) for a in base_srs.get("attributes", [])]
        name_to_idx = {a["name"]: i for i, a in enumerate(post_attrs)}
        for da in derived_schema.get("attributes", []):
            dname = da["name"]
            evidence_items = evidence_map.get(dname, [])
            ecnf_pass, count, agg_score = self.check_ecnf(evidence_items)
            if not ecnf_pass:
                logger.debug(f"ECNF not satisfied for {dname} (count={count}, score={agg_score:.3f}) -> skip auto-merge")
                # record pending evidence for human review
                self.record_lineage(action="merge_pending", from_attr=dname, to=None, evidence=evidence_items, auto=False)
                continue
            # Choose canonical target token from evidence (prefer ontology or high-score nn)
            # Fallback: use first evidence token
            candidate_tokens = [it.get("token") or it.get("target") for it in evidence_items if it.get("token") or it.get("target")]
            if not candidate_tokens:
                logger.debug(f"No canonical target found in evidence for {dname}.")
                self.record_lineage(action="merge_pending", from_attr=dname, to=None, evidence=evidence_items, auto=False)
                continue
            canonical = candidate_tokens[0]
            # If canonical exists in base SRS attributes, append alias
            if canonical in name_to_idx:
                idx = name_to_idx[canonical]
                aliases = post_attrs[idx].setdefault("aliases", [])
                if dname not in aliases and dname != canonical:
                    aliases.append(dname)
                    post_attrs[idx]["aliases"] = aliases
                    # record auto-merge lineage
                    self.record_lineage(action="merge", from_attr=dname, to=canonical, evidence=evidence_items, auto=True)
            else:
                # If canonical not present, create a new attribute entry
                new_attr = {
                    "name": canonical,
                    "type": da.get("type", "string"),
                    "aliases": [dname],
                    "provenance": da.get("provenance", {"source": "derived"})
                }
                post_attrs.append(new_attr)
                name_to_idx[canonical] = len(post_attrs) - 1
                self.record_lineage(action="merge_create", from_attr=dname, to=canonical, evidence=evidence_items, auto=True)
        return {"attributes": post_attrs}

    def execute_merge(self, srs_base: Dict[str, Any], derived_schema: Dict[str, Any], evidence_map: Dict[str, List[Dict[str, Any]]], pre_vec: Optional[np.ndarray] = None, post_vec: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        High-level merge: apply AANF/ECNF (apply_merge), then DBNF check (pre/post drift).
        Returns structure: {'after': {...}, 'dbnf': {...}, 'lineage': [...]}
        """
        after = self.apply_merge(srs_base, derived_schema, evidence_map)
        dbnf_ok, drift_val = self.check_dbnf(pre_vec, post_vec)
        dbnf_res = {"pass": bool(dbnf_ok), "drift": float(drift_val)}
        result = {"after": after, "dbnf": dbnf_res, "lineage": list(self.lineage)}
        return result

    def record_lineage(self, action: str, from_attr: Optional[str], to: Optional[str], evidence: Optional[List[Dict[str, Any]]], auto: bool = False):
        entry = {
            "action": action,
            "from": from_attr if isinstance(from_attr, list) else [from_attr] if from_attr else [],
            "to": to,
            "evidence": evidence or [],
            "actor": "auto" if auto else "pending_human",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        self.lineage.append(entry)
        logger.debug(f"Lineage recorded: {entry}")
