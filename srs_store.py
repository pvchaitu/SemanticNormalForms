# srs_store.py
"""
SRSStore updated to:
- use multi-level embeddings with regenerations for EENF
- build richer per-merge evidence (component-wise, ontology, cooccurrence, source diversity)
- compute per-attribute adaptive DBNF taus and use them in merges
- create simple schema versioning on forks (append new version entries)
- parameterize domain heuristics and thresholds via sdnf_config.json
"""

import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import re
from collections import defaultdict

from embedding_utils import EmbeddingModel
from semantic_merge import SemanticMerger, evidence_score
from validators import SDNFValidator
from sdnf_config import load_config

logger = logging.getLogger("sdnf.srs_store")
logging.basicConfig(level=logging.INFO)

class SRSStore:
    def __init__(self,
                 initial_master: Optional[Dict[str, Any]] = None,
                 embedding_model: Optional[EmbeddingModel] = None,
                 merger: Optional[SemanticMerger] = None,
                 validator: Optional[SDNFValidator] = None):
        self.config = load_config()
        self.model = embedding_model or EmbeddingModel()
        self.merger = merger or SemanticMerger(m_min=self.config["ECNF"]["m_min"], tau_dbnf=self.config["DBNF"]["global_tau"])
        self.validator = validator or SDNFValidator({
            "EENF": self.config["EENF"]["tau"],
            "AANF": self.config["AANF"]["tau"],
            "CMNF": self.config["CMNF"]["tau"],
            "DBNF": self.config["DBNF"]["global_tau"],
            "ECNF": self.config["ECNF"]["m_min"],
            "RRNF": self.config["RRNF"]["tau"],
            "PONF": self.config["PONF"]["tau"]
        })

        if initial_master is None:
            self.master = {"attributes": [], "relations": [], "versions": [{"version": 1, "attributes": []}]}
        else:
            self.master = initial_master
            if "versions" not in self.master:
                self.master["versions"] = [{"version": 1, "attributes": list(self.master.get("attributes", []))}]

        self.records: List[Dict[str, Any]] = []
        self.lineage: List[Dict[str, Any]] = list(self.merger.lineage)
        self._canonical_embeddings: Dict[str, np.ndarray] = {}
        self.attribute_stats: Dict[str, Dict[str, Any]] = {}
        self._recompute_all_canonical_embeddings()

    def _now_iso(self) -> str:
        return datetime.datetime.utcnow().isoformat() + "Z"

    def _embed(self, token: str, context: Optional[str] = None, regen_idx: Optional[int] = None) -> np.ndarray:
        emb = self.model.encode([token], context=context, regen_idx=regen_idx)[0]
        n = np.linalg.norm(emb) + 1e-12
        return (emb / n).astype(np.float32)

    def _recompute_all_canonical_embeddings(self):
        self._canonical_embeddings = {}
        for a in self.master.get("attributes", []):
            name = a["name"]
            tokens = [name] + a.get("aliases", [])
            recent_aliases = self._collect_recent_aliases_for(name, window=1000)
            tokens.extend(recent_aliases)
            emb_list = [self._embed(t) for t in tokens]
            if emb_list:
                centroid = np.mean(np.stack(emb_list, axis=0), axis=0)
                if np.linalg.norm(centroid) > 0:
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                self._canonical_embeddings[name] = centroid
                stats = self.attribute_stats.setdefault(name, {})
                stats.setdefault("alias_embs", [])
                stats["alias_embs"].extend(emb_list)
                stats["alias_embs"] = stats["alias_embs"][-500:]
                arr = np.stack(stats["alias_embs"], axis=0)
                var = float(np.mean(np.var(arr, axis=0)))
                stats["variance"] = var
                stats["count"] = arr.shape[0]
            else:
                self._canonical_embeddings[name] = self._embed(name)
                self.attribute_stats.setdefault(name, {})["variance"] = 0.0
                self.attribute_stats.setdefault(name, {})["count"] = 0

    def _collect_recent_aliases_for(self, canonical_name: str, window: int = 500) -> List[str]:
        aliases = []
        for rec in reversed(self.records[-window:]):
            mapped = rec.get("mapped_to", {})
            for k, v in mapped.items():
                if v == canonical_name:
                    aliases.append(k)
        seen = set()
        out = []
        for t in aliases:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _find_best_match(self, token: str, threshold: float = 0.82) -> Optional[Tuple[str, float]]:
        if not self._canonical_embeddings:
            return None
        q = self._embed(token)
        names = list(self._canonical_embeddings.keys())
        mats = np.stack([self._canonical_embeddings[n] for n in names], axis=0)
        sims = mats @ q
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            return names[best_idx], best_score
        return None

    def _type_of(self, attr_name: str) -> str:
        if not attr_name:
            return "string"
        for a in self.master.get("attributes", []):
            if a.get("name") == attr_name:
                return a.get("type", "string")
        return "string"

    def _ontology_lookup(self, token: str) -> Optional[str]:
        token_l = token.lower()
        mappings = self.config.get("ontology", {}).get("mappings", {})
        for k, v in mappings.items():
            if k in token_l:
                return v
        return None

    def compute_adaptive_dbnf_taus(self) -> Dict[str, float]:
        taus: Dict[str, float] = {}
        cfg = self.config["DBNF"]
        global_tau = float(cfg["global_tau"])
        multiplier = float(cfg.get("adaptive_multiplier", 2.5))
        min_alias_count = int(cfg.get("min_alias_count", 3))
        for name, stats in self.attribute_stats.items():
            var = float(stats.get("variance", 0.0))
            count = int(stats.get("count", 0))
            variability = float(np.sqrt(var)) if var >= 0 else 0.0
            tau_candidate = multiplier * variability
            if count < min_alias_count:
                tau_attr = max(global_tau, 0.9 * tau_candidate)
            else:
                tau_attr = max(global_tau, tau_candidate)
            taus[name] = float(tau_attr)
        for a in self.master.get("attributes", []):
            if a["name"] not in taus:
                taus[a["name"]] = float(global_tau)
        return taus

    def get_master_schema(self) -> Dict[str, Any]:
        return {"attributes": [dict(a) for a in self.master.get("attributes", [])],
                "relations": [dict(r) for r in self.master.get("relations", [])],
                "versions": list(self.master.get("versions", []))}

    def get_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return list(self.records[:limit]) if limit else list(self.records)

    def list_lineage(self, last_n: Optional[int] = 50) -> List[Dict[str, Any]]:
        entries = list(self.merger.lineage)
        return entries[-last_n:] if last_n else entries

    def check_payload_compliance(self, payload: Dict[str, Any], context: Optional[str] = None) -> Dict[str, Any]:
        mapped = {}
        violations = []
        for k, v in payload.items():
            match = self._find_best_match(k, threshold=self.validator.tau.get("AANF", 0.88))
            mapped[k] = match[0] if match else None

        if context:
            for k, attr in mapped.items():
                if attr:
                    attr_meta = next((a for a in self.master.get("attributes", []) if a["name"] == attr), None)
                    if attr_meta:
                        allowed = attr_meta.get("contexts")
                        if allowed and context not in allowed:
                            violations.append({"field": k, "reason": "ContextMismatch", "details": {"expected": allowed, "found": context}})

        attr_names = [a["name"] for a in self.master.get("attributes", [])]
        attr_embeddings = np.stack([self._canonical_embeddings.get(n, self._embed(n)) for n in attr_names], axis=0) if attr_names else np.zeros((0, self.model.dim))
        pre_vec = np.mean(attr_embeddings, axis=0) if attr_embeddings.size else np.zeros((self.model.dim,))
        payload_embs = np.stack([self._embed(k, context=context) for k in payload.keys()], axis=0) if payload else np.zeros((0, self.model.dim))
        post_vec = np.mean(np.vstack([pre_vec, np.mean(payload_embs, axis=0)]), axis=0) if payload_embs.size else pre_vec

        data_context = {
            "regenerations": np.vstack([attr_embeddings, attr_embeddings + 1e-6]) if attr_embeddings.size else np.zeros((1, self.model.dim)),
            "attr_embeddings": attr_embeddings,
            "attr_names": attr_names,
            "W_p": np.eye(self.model.dim),
            "W_r": np.eye(self.model.dim),
            "sample_embeddings": attr_embeddings if attr_embeddings.size else np.zeros((1, self.model.dim)),
            "pre_vec": pre_vec,
            "post_vec": post_vec,
            "tau_dbnf": self.validator.tau.get("DBNF", self.config["DBNF"]["global_tau"]),
            "relations": self.master.get("relations", []),
            "role_compat": {},
            "partition_labels": [0] * len(attr_names)
        }

        sdnf_results = self.validator.run_all(data_context)
        compliant = (len(violations) == 0) and all(r["status"] == "PASS" for r in sdnf_results)
        return {"compliant": compliant, "mapped_fields": mapped, "violations": violations, "sdnf_status": sdnf_results}

    def _build_evidence_for_attr(self, derived_name: str, derived_type: str, derived_vec: np.ndarray, context: Optional[str] = None):
        evidence = []
        best_token, best_score = None, 0.0
        if self._canonical_embeddings:
            keys = list(self._canonical_embeddings.keys())
            mats = np.stack([self._canonical_embeddings[k] for k in keys], axis=0)
            sims = np.dot(mats, derived_vec)
            idx = int(np.argmax(sims))
            best_score = float(sims[idx])
            best_token = keys[idx]
            comp_sim = self.model.component_similarity(self._canonical_embeddings[best_token], derived_vec)
            evidence.append({"type": "nn", "score": float(max(0.0, min(1.0, (best_score + 1) / 2))), "token": best_token})
            evidence.append({"type": "abstract_nn", "score": float(max(0.0, min(1.0, (comp_sim['abstract'] + 1) / 2))), "token": best_token})
            evidence.append({"type": "contextual_nn", "score": float(max(0.0, min(1.0, (comp_sim['contextual'] + 1) / 2))), "token": best_token})

        try:
            ont_root = self._ontology_lookup(derived_name)
            if ont_root:
                evidence.append({"type": "ontology", "score": 0.85, "token": ont_root})
        except Exception:
            pass

        type_score = 1.0 if derived_type and best_token and self._type_of(best_token) == derived_type else 0.0
        evidence.append({"type": "type_match", "score": float(type_score), "token": best_token or derived_name})

        coocc_score = 0.0
        source_set = set()
        if self.records:
            coocc_count = 0
            total_count = 0
            for rec in self.records[-2000:]:
                total_count += 1
                if derived_name in rec.get("payload", {}):
                    mapped = rec.get("mapped_to", {})
                    if best_token and best_token in mapped.values():
                        coocc_count += 1
                        source_set.add(rec.get("source", "unknown"))
            if total_count > 0:
                coocc_score = float(coocc_count) / float(total_count)
                if coocc_score > 0:
                    evidence.append({"type": "cooccurrence", "score": min(1.0, coocc_score), "token": best_token or derived_name, "sources": list(source_set)})

        heur_patterns = self.config.get("heuristics", {}).get("token_patterns", [])
        if any(p in derived_name.lower() for p in heur_patterns):
            evidence.append({"type": "heuristic", "score": 0.6, "token": derived_name})

        pan_regex = self.config.get("heuristics", {}).get("pan_regex")
        if pan_regex and re.match(pan_regex, derived_name):
            evidence.append({"type": "heuristic", "score": 0.85, "token": derived_name})

        if source_set and len(source_set) >= 2:
            evidence.append({"type": "source_diversity", "score": 1.0, "token": derived_name, "sources": list(source_set)})

        return evidence

    def insert_payload(self, payload: Dict[str, Any], source: str = "unknown", context: Optional[str] = None, auto_merge: bool = True) -> Dict[str, Any]:
        compliance = self.check_payload_compliance(payload, context=context)
        mapped = compliance["mapped_fields"]
        derived_schema = {"entity": f"DerivedFrom_{source}", "attributes": [], "derived_at": self._now_iso()}
        for k, v in payload.items():
            derived_schema["attributes"].append({
                "name": k,
                "type": "string",
                "aliases": [],
                "provenance": {"source": source, "first_seen": self._now_iso()}
            })

        evidence_map = {}
        for da in derived_schema.get("attributes", []):
            name = da["name"]
            dtype = da.get("type", "string")
            vec = self._embed(name.lower(), context=context)
            evidence_map[name] = self._build_evidence_for_attr(name, dtype, vec, context=context)

        pre_canonical_embeddings = dict(self._canonical_embeddings)
        per_attribute_tau = self.compute_adaptive_dbnf_taus()

        merge_result = None
        if auto_merge:
            merge_result = self.merger.execute_merge(self.master, derived_schema, evidence_map, embed_fn=lambda t: self._embed(t, context=context), pre_canonical_embeddings=pre_canonical_embeddings, per_attribute_tau=per_attribute_tau)
            if merge_result and merge_result.get("after"):
                # apply after-structure and create new version if forks occurred
                self.master = merge_result["after"]
                # if any fork actions, create a new version snapshot
                forks = [d for d in merge_result.get("dbnf_details", []) if not d.get("pass", True)]
                if forks:
                    new_version_id = max(v.get("version", 1) for v in self.master.get("versions", [{"version": 1}])) + 1
                    self.master.setdefault("versions", []).append({"version": new_version_id, "attributes": [dict(a) for a in self.master.get("attributes", [])], "created_at": self._now_iso()})
                self._recompute_all_canonical_embeddings()
                self.lineage.extend(self.merger.lineage)
        else:
            merge_result = {"after": self.master, "dbnf_details": [], "lineage": []}

        record = {
            "id": len(self.records) + 1,
            "payload": payload,
            "source": source,
            "inserted_at": self._now_iso(),
            "mapped_to": mapped,
            "compliance": compliance,
            "merge_result": merge_result
        }
        self.records.append(record)
        return {"record_id": record["id"], "compliance": compliance, "merge_result": merge_result}

    def export_snapshot(self) -> Dict[str, Any]:
        return {
            "master": self.get_master_schema(),
            "records_count": len(self.records),
            "recent_lineage": self.list_lineage(100),
            "timestamp": self._now_iso()
        }
