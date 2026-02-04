# semantic_merge.py
"""
SemanticMerger with:
- per-merge ECNF evaluation (config-driven)
- canonical selection by highest aggregate evidence score
- DBNF per-attribute tau usage and lineage recording (tau included)
- auto-merge policy parameterized (canary fraction support placeholder)
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sdnf_config import load_config

logger = logging.getLogger("sdnf.merge")
cfg = load_config()

def evidence_score(items: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {"nn": 0.35, "abstract_nn": 0.20, "ontology": 0.20, "cooccurrence": 0.15, "type_match": 0.06, "heuristic": 0.04}
    total_w = 0.0
    weighted = 0.0
    for it in items or []:
        t = it.get("type", "nn")
        s = float(it.get("score", 0.5))
        w = weights.get(t, 0.0)
        weighted += w * s
        total_w += w
    if total_w <= 0:
        return 0.0
    return max(0.0, min(1.0, weighted / total_w))

def distinct_signal_types(evidence_items: List[Dict[str, Any]]) -> int:
    return len(set(it.get("type") for it in (evidence_items or [])))

class SemanticMerger:
    def __init__(self, m_min: int = None, tau_dbnf: float = None):
        cfg_local = load_config()
        self.m_min = int(m_min or cfg_local["ECNF"]["m_min"])
        self.tau_dbnf = float(tau_dbnf or cfg_local["DBNF"]["global_tau"])
        self.lineage: List[Dict[str, Any]] = []
        self.config = cfg_local

    def check_ecnf_per_merge(self, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(evidence_items or [])
        score = evidence_score(evidence_items or [])
        req_count = int(self.config["ECNF"]["m_min"])
        score_threshold = float(self.config["ECNF"]["score_threshold"])
        strong_score_threshold = float(self.config["ECNF"]["strong_score_threshold"])
        required_signals = int(self.config["ECNF"].get("require_distinct_signals", 2))
        distinct = distinct_signal_types(evidence_items)
        pass_flag = False
        reason = ""
        if (count >= req_count and score >= score_threshold and distinct >= required_signals):
            pass_flag = True
            reason = "count_score_and_diversity"
        elif (score >= strong_score_threshold and distinct >= 1):
            pass_flag = True
            reason = "strong_score"
        else:
            pass_flag = False
            reason = "insufficient_evidence"
        return {"pass": pass_flag, "count": count, "score": score, "distinct_signals": distinct, "reason": reason}

    def recompute_canonical(self, alias_embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        if not alias_embeddings:
            raise ValueError("No alias embeddings provided")
        X = np.stack(alias_embeddings, axis=0)
        if weights is None:
            return np.mean(X, axis=0)
        w = np.asarray(weights).reshape(-1, 1)
        return (w * X).sum(axis=0) / w.sum()

    def check_dbnf_attr(self, pre_vec: Optional[np.ndarray], post_vec: Optional[np.ndarray], tau_attr: float) -> Tuple[bool, float]:
        if pre_vec is None or post_vec is None:
            return True, 0.0
        if np.linalg.norm(pre_vec) > 0:
            pre_vec = pre_vec / (np.linalg.norm(pre_vec) + 1e-12)
        if np.linalg.norm(post_vec) > 0:
            post_vec = post_vec / (np.linalg.norm(post_vec) + 1e-12)
        d = float(np.linalg.norm(pre_vec - post_vec))
        return (d <= tau_attr), d

    def apply_merge(self,
                    base_srs: Dict[str, Any],
                    derived_schema: Dict[str, Any],
                    evidence_map: Dict[str, List[Dict[str, Any]]],
                    embed_fn) -> Dict[str, Any]:
        post_attrs = [dict(a) for a in base_srs.get("attributes", [])]
        name_to_idx = {a["name"]: i for i, a in enumerate(post_attrs)}
        diff = {"added": [], "removed": [], "aliases_added": []}
        to_remove = set()

        for da in derived_schema.get("attributes", []):
            dname = da["name"]
            evidence_items = evidence_map.get(dname, [])
            ecnf_report = self.check_ecnf_per_merge(evidence_items)
            ecnf_pass = ecnf_report["pass"]
            if not ecnf_pass:
                self.record_lineage("merge_pending", dname, None, evidence_items, auto=False, extra={"ecnf": ecnf_report})
                continue

            # choose canonical by highest evidence score among candidate tokens
            candidate_scores = {}
            for ev in evidence_items:
                token = ev.get("token")
                if not token:
                    continue
                candidate_scores.setdefault(token, []).append(ev)
            # compute aggregate score per candidate
            best_candidate = None
            best_score = -1.0
            for cand, evs in candidate_scores.items():
                s = evidence_score(evs)
                if s > best_score:
                    best_score = s
                    best_candidate = cand

            if not best_candidate:
                # fallback: use first evidence token
                best_candidate = evidence_items[0].get("token") if evidence_items else dname

            canonical = best_candidate
            if canonical in name_to_idx:
                idx = name_to_idx[canonical]
                aliases = post_attrs[idx].setdefault("aliases", [])
                if dname not in aliases and dname != canonical:
                    aliases.append(dname)
                    diff["aliases_added"].append((canonical, dname))
                    self.record_lineage("merge", dname, canonical, evidence_items, auto=True, extra={"ecnf": ecnf_report, "agg_score": best_score})
                if dname in name_to_idx and dname != canonical:
                    to_remove.add(dname)
            else:
                new_attr = {
                    "name": canonical,
                    "type": da.get("type", "string"),
                    "aliases": [dname],
                    "provenance": da.get("provenance", {"source": "derived"})
                }
                post_attrs.append(new_attr)
                name_to_idx[canonical] = len(post_attrs) - 1
                diff["added"].append(canonical)
                self.record_lineage("merge_create", dname, canonical, evidence_items, auto=True, extra={"ecnf": ecnf_report, "agg_score": best_score})

        compact = []
        for a in post_attrs:
            if a["name"] in to_remove:
                diff["removed"].append(a["name"])
                continue
            compact.append(a)

        return {"attributes": compact, "diff": diff}

    def execute_merge(self,
                      srs_base: Dict[str, Any],
                      derived_schema: Dict[str, Any],
                      evidence_map: Dict[str, List[Dict[str, Any]]],
                      embed_fn,
                      pre_canonical_embeddings: Optional[Dict[str, np.ndarray]] = None,
                      per_attribute_tau: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        before = {"attributes": [dict(a) for a in srs_base.get("attributes", [])]}
        after_struct = self.apply_merge(srs_base, derived_schema, evidence_map, embed_fn)

        post_embeddings = {}
        dbnf_details = []
        for a in after_struct["attributes"]:
            name = a["name"]
            tokens = [name] + a.get("aliases", [])
            alias_embs = [embed_fn(t) for t in tokens]
            new_emb = self.recompute_canonical(alias_embs)
            post_embeddings[name] = new_emb
            pre_emb = None
            if pre_canonical_embeddings and name in pre_canonical_embeddings:
                pre_emb = pre_canonical_embeddings[name]
            tau_attr = None
            if per_attribute_tau and name in per_attribute_tau:
                tau_attr = per_attribute_tau[name]
            else:
                tau_attr = self.tau_dbnf
            ok, drift = self.check_dbnf_attr(pre_emb, new_emb, tau_attr)
            action = "accept" if ok else "fork"
            dbnf_details.append({"attribute": name, "drift": float(drift), "pass": bool(ok), "action": action, "tau": float(tau_attr)})
            if not ok:
                # create a new version entry in lineage (simple versioning)
                self.record_lineage("fork", name, None, [], auto=True, extra={"tau": float(tau_attr), "action": "fork"})
        result = {"before": before, "after": after_struct, "dbnf_details": dbnf_details, "lineage": list(self.lineage)}
        return result

    def record_lineage(self, action: str, from_attr: Optional[str], to: Optional[str], evidence: Optional[List[Dict[str, Any]]], auto: bool = True, extra: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "action": action,
            "from": [from_attr] if from_attr else [],
            "to": to,
            "evidence": evidence or [],
            "actor": "auto" if auto else "pending_human"
        }
        if extra:
            entry.update(extra)
        self.lineage.append(entry)
        logger.debug("Lineage recorded: %s", entry)
