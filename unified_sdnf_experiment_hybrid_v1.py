#!/usr/bin/env python3
"""unified_sdnf_experiment_hybrid_v1.py"""

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- FIX: Move PATS and VSS_DIM to Global level ---
PATS = ["DDDD", "DDDDDD", "AAAA", "SS", "DS", "SD", "DA", "AD"]
VSS_DIM = 14 + len(PATS) 

logger = logging.getLogger("sdnf.unified")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_json_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.json") if p.is_file()])

def camel_to_tokens(s: str) -> str:
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s2 = s2.replace("_", " ")
    return " ".join(s2.split())

def normalize_name(name: str) -> str:
    return camel_to_tokens(name).strip()

def safe_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return (x / n).astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    ent = 0.0
    for k, v in c.items():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return float(ent)

def vss_from_value(v: Any) -> Optional[np.ndarray]:
    if v is None: return None
    s = str(v)
    if s == "": return None
    raw = s
    s = s.strip()
    L = len(s)
    digits = sum(ch.isdigit() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    alnum = sum(ch.isalnum() for ch in s)
    spaces = sum(ch.isspace() for ch in raw)
    specials = L - alnum
    has_slash = 1.0 if "/" in s else 0.0
    has_dash = 1.0 if "-" in s else 0.0
    has_at = 1.0 if "@" in s else 0.0
    has_dot = 1.0 if "." in s else 0.0
    is_float = 0.0
    is_int = 0.0
    try:
        float(s)
        is_float = 1.0
        if re.fullmatch(r"[-+]?\d+", s): is_int = 1.0
    except Exception: pass
    is_date_like = 1.0 if re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2}|\d{4})", s) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else 0.0
    uniq_ratio = len(set(s)) / max(1, L)
    ent = _shannon_entropy(s)

    vec = np.zeros((VSS_DIM,), dtype=np.float32)
    vec[0] = min(1.0, L / 32.0)
    vec[1] = digits / max(1, L)
    vec[2] = alpha / max(1, L)
    vec[3] = specials / max(1, L)
    vec[4] = spaces / max(1, max(1, len(raw)))
    vec[5] = has_slash
    vec[6] = has_dash
    vec[7] = has_at
    vec[8] = has_dot
    vec[9] = is_float
    vec[10] = is_int
    vec[11] = is_date_like
    vec[12] = float(min(1.0, uniq_ratio))
    vec[13] = float(min(1.0, ent / 4.0))

    classes = []
    for ch in s[:64]:
        if ch.isdigit(): classes.append('D')
        elif ch.isalpha(): classes.append('A')
        else: classes.append('S')
    sketch = "".join(classes)    
    for i, p in enumerate(PATS):
        vec[14 + i] = 1.0 if p in sketch else 0.0
    return safe_norm(vec)

def shape_tokens(v: Any) -> Optional[str]:
    if v is None: return None
    s = str(v).strip()
    if s == "": return None
    L = len(s)
    toks = [f"LEN_{min(L,64)}"]
    if s.isdigit(): toks.append("ALL_DIGITS")
    if s.isalpha(): toks.append("ALL_ALPHA")
    if s.isalnum() and not s.isdigit() and not s.isalpha(): toks.append("ALNUM")
    if "/" in s: toks.append("HAS_SLASH")
    if "-" in s: toks.append("HAS_DASH")
    if "." in s: toks.append("HAS_DOT")
    if "@" in s: toks.append("HAS_AT")
    if re.fullmatch(r"\d{15,16}", s): toks.append("LEN_15_16_DIGITS")
    if re.fullmatch(r"\d{3,4}", s): toks.append("LEN_3_4_DIGITS")
    if re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2}|\d{4})", s): toks.append("DATE_MM_YY")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): toks.append("DATE_ISO")
    digits = sum(ch.isdigit() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    toks.append(f"DIGIT_FRAC_{int(10*digits/max(1,L))}")
    toks.append(f"ALPHA_FRAC_{int(10*alpha/max(1,L))}")
    return " ".join(toks)

class MultiLevelEmbedder:
    def __init__(self, st_model):
        self.model = st_model
        # --- FIX: Silencing Batch output ---
        v = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        self.base_dim = int(v.shape[1])
        self.dim = self.base_dim * 3
    def embed_many(self, tokens: List[str], context: str) -> np.ndarray:
        # --- FIX: Silencing Batch output ---
        fine = self.model.encode(tokens, normalize_embeddings=True, show_progress_bar=False)
        abstract = self.model.encode([normalize_name(t) for t in tokens], normalize_embeddings=True, show_progress_bar=False)
        contextual = self.model.encode([f"{t} in {context} context" for t in tokens], normalize_embeddings=True, show_progress_bar=False)
        X = np.concatenate([fine, abstract, contextual], axis=1).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X
    def embed(self, token: str, context: str) -> np.ndarray:
        return self.embed_many([token], context=context)[0]
    def regenerations(self, token: str, context: str, G: int = 10) -> np.ndarray:
        variants = [token, token + " ", token + ".", token + "!", f"{token} field"]
        variants = (variants * ((G + len(variants) - 1) // len(variants)))[:G]
        return self.embed_many(variants, context=context)

@dataclass
class CanonicalAttribute:
    name: str
    context: str
    embedding: np.ndarray
    aliases: List[str] = field(default_factory=list)
    vss_centroid: Optional[np.ndarray] = None
    vss_count: int = 0
    shape_centroid: Optional[np.ndarray] = None
    shape_count: int = 0

@dataclass
class EvidenceItem:
    type: str
    score: float

class SDNFExperiment:
    def __init__(self, data_dir: Path, model_name: str, drift_model_name: Optional[str], contexts: List[str], evidence_mode: str = "hybrid", seed: int = 42, log_level: str = "INFO", hnsw_m: int = 32, ef_construction: int = 200, ef_search: int = 50):
        setup_logging(log_level)
        self.data_dir = data_dir
        self.model_name = model_name
        self.drift_model_name = drift_model_name
        self.contexts = contexts
        self.evidence_mode = evidence_mode
        self.seed = seed
        np.random.seed(seed)
        from sentence_transformers import SentenceTransformer
        import hnswlib
        self.hnswlib = hnswlib
        self.st = SentenceTransformer(model_name)
        self.embedder = MultiLevelEmbedder(self.st)
        self.shape_st = self.st
        # --- FIX: Silencing Batch output ---
        self.shape_dim = int(self.shape_st.encode(["test"], normalize_embeddings=True, show_progress_bar=False).shape[1])
        self.drift_st = SentenceTransformer(drift_model_name) if drift_model_name else None
        self.drift_embedder = MultiLevelEmbedder(self.drift_st) if self.drift_st else None
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.tau_aanf = 0.88 
        self.m_min_default = 3
        self.score_threshold = 0.60
        self.gamma = 0.72
        self.canon: Dict[str, CanonicalAttribute] = {}
        self.lineage: List[Dict[str, Any]] = []
        self.index = None
        self.id_to_name: Dict[int, str] = {}
        self.value_evidence_available = 0
        self.value_evidence_missing = 0

    def parse_file(self, path: Path) -> Tuple[List[str], Dict[str, Any]]:
        obj = load_json(path)
        if isinstance(obj, dict) and "attributes" in obj and isinstance(obj["attributes"], list):
            names = [str(a["name"]) for a in obj["attributes"] if isinstance(a, dict) and a.get("name")]
            return sorted(set(names)), obj
        if isinstance(obj, dict):
            return [k for k in obj.keys() if not str(k).startswith("_")], obj
        return [], obj

    def infer_context_for_file(self, filename: str) -> str:
        f = filename.lower()
        if any(x in f for x in ["risk", "fraud", "score"]): return self.contexts[1] if len(self.contexts) > 1 else self.contexts[0]
        return self.contexts[0]

    def rebuild_index(self):
        if not self.canon: return
        dim = self.embedder.dim
        names = list(self.canon.keys())
        X = np.stack([self.canon[n].embedding for n in names], axis=0).astype(np.float32)
        index = self.hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=max(1000, len(names) * 5), ef_construction=self.ef_construction, M=self.hnsw_m)
        ids = np.arange(len(names))
        index.add_items(X, ids)
        index.set_ef(self.ef_search)
        self.index = index
        self.id_to_name = {int(i): names[int(i)] for i in ids}

    def bootstrap_master(self):
        inamex = self.data_dir / "INAmex.json"
        if not inamex.exists():
            raise FileNotFoundError(f"CRITICAL: Required bootstrap file 'INAmex.json' not found in {self.data_dir}.")
        names, _ = self.parse_file(inamex)
        if not names:
            raise ValueError(f"CRITICAL: 'INAmex.json' found but no attributes could be parsed from it.")
        ctx = self.infer_context_for_file(inamex.name)
        canonical_names = sorted(set(names))
        logger.info("Bootstrapping from %s | canon=%d", inamex.name, len(canonical_names))
        embs = self.embedder.embed_many(canonical_names, context=ctx)
        for n, e in zip(canonical_names, embs):
            self.canon[n] = CanonicalAttribute(name=n, context=ctx, embedding=e, aliases=[n])
            self.lineage.append({"action": "create", "to": n, "source": inamex.name})

    def _shape_embed(self, shape_str: str) -> np.ndarray:
        # --- FIX: Silencing Batch output ---
        v = self.shape_st.encode([shape_str], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        return safe_norm(v)

    def build_evidence(self, derived_name: str, derived_value: Any, candidate: CanonicalAttribute, nn_sim: float) -> List[EvidenceItem]:
        ev = [EvidenceItem(type="nn", score=float(nn_sim))]
        if self.evidence_mode == "embed_only" or derived_value is None: 
            if derived_value is None: self.value_evidence_missing += 1
            return ev
        self.value_evidence_available += 1
        if self.evidence_mode in ("vss", "hybrid"):
            vss = vss_from_value(derived_value)
            if vss is not None and candidate.vss_centroid is not None:
                ev.append(EvidenceItem(type="vss", score=float(cosine_sim(vss, candidate.vss_centroid))))
        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                shv = self._shape_embed(sh)
                if candidate.shape_centroid is not None:
                    ev.append(EvidenceItem(type="shape", score=float(cosine_sim(shv, candidate.shape_centroid))))
        return ev

    def aggregate_score(self, evidence: List[EvidenceItem]) -> float:
        weights = {"nn": 0.60, "vss": 0.20, "shape": 0.20} if self.evidence_mode == "hybrid" else {"nn": 1.0}
        total_w = acc = 0.0
        for it in evidence:
            w = weights.get(it.type, 0.0)
            total_w += w
            acc += w * max(0.0, min(1.0, (float(it.score) + 1.0) / 2.0))
        return float(acc / total_w) if total_w > 0 else 0.0

    def ecnf_pass(self, evidence: List[EvidenceItem]) -> Tuple[bool, str, float, int]:
        score = self.aggregate_score(evidence)
        distinct = len(set(it.type for it in evidence))
        m_min = 1 if self.evidence_mode == "embed_only" else self.m_min_default
        if distinct >= m_min and score >= self.score_threshold: return True, "count_score", score, distinct
        if score >= self.gamma: return True, "strong_score", score, distinct
        return False, "insufficient", score, distinct

    def update_validator_state(self, canonical: CanonicalAttribute, derived_value: Any):
        if derived_value is None: return
        if self.evidence_mode in ("vss", "hybrid"):
            v = vss_from_value(derived_value)
            if v is not None:
                alpha = 1.0 / float(min(50, canonical.vss_count + 1))
                canonical.vss_centroid = v if canonical.vss_centroid is None else safe_norm((1 - alpha) * canonical.vss_centroid + alpha * v)
                canonical.vss_count += 1
        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                sv = self._shape_embed(sh)
                alpha = 1.0 / float(min(50, canonical.shape_count + 1))
                canonical.shape_centroid = sv if canonical.shape_centroid is None else safe_norm((1 - alpha) * canonical.shape_centroid + alpha * sv)
                canonical.shape_count += 1

    def ingest_file(self, path: Path, max_fields: Optional[int] = None):
        names, raw = self.parse_file(path)
        ctx = self.infer_context_for_file(path.name)
        if not names: return
        if max_fields: names = names[:max_fields]
        payload = raw if isinstance(raw, dict) else {}
        X = self.embedder.embed_many(names, context=ctx)
        merges = creates = 0
        
        for derived_name, emb in zip(names, X):
            if derived_name in self.canon: continue

            # --- FIX: More robust HNSW check to prevent RuntimeError ---
            if self.index is None or self.index.element_count == 0:
                self.canon[derived_name] = CanonicalAttribute(name=derived_name, context=ctx, embedding=emb, aliases=[derived_name])
                self.lineage.append({"action": "create", "to": derived_name, "source": path.name})
                creates += 1
                self.rebuild_index() # Rebuild immediately if empty so next field can query
                continue

            labels, dists = self.index.knn_query(emb.reshape(1, -1), k=min(5, self.index.element_count))
            best_id, best_sim = int(labels[0][0]), float(1.0 - float(dists[0][0]))
            best_name = self.id_to_name[best_id]
            candidate = self.canon[best_name]
            derived_value = payload.get(derived_name)
            evidence = self.build_evidence(derived_name, derived_value, candidate, best_sim)
            ok, reason, score, signals = self.ecnf_pass(evidence)

            if best_sim >= self.tau_aanf and ok:
                if derived_name not in candidate.aliases: candidate.aliases.append(derived_name)
                candidate.embedding = safe_norm(candidate.embedding + emb)
                self.update_validator_state(candidate, derived_value)
                self.lineage.append({"action": "merge_auto", "from": derived_name, "to": best_name, "sim": round(best_sim, 3)})
                merges += 1
            else:
                self.canon[derived_name] = CanonicalAttribute(name=derived_name, context=ctx, embedding=emb, aliases=[derived_name])
                self.lineage.append({"action": "create", "to": derived_name, "source": path.name})
                creates += 1

        if creates > 0: self.rebuild_index()
        logger.info("Ingested %-18s | merges=%2d | new=%2d", path.name, merges, creates)

    def run(self, max_files: Optional[int] = None, max_fields: Optional[int] = None):
        self.bootstrap_master()
        self.rebuild_index()
        files = iter_json_files(self.data_dir)
        if max_files: files = files[:max_files]
        for p in files:
            if p.name == "INAmex.json": continue
            self.ingest_file(p, max_fields=max_fields)
        logger.info("Summary | mode=%s | canon=%d", self.evidence_mode, len(self.canon))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--drift_model", type=str, default=None)
    ap.add_argument("--contexts", nargs="+", default=["Payments", "Risk"])
    ap.add_argument("--evidence_mode", type=str, default="hybrid", choices=["embed_only", "vss", "shape", "hybrid"])
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_fields", type=int, default=None)
    ap.add_argument("--log_level", type=str, default="INFO")
    args = ap.parse_args()
    exp = SDNFExperiment(Path(args.data_dir), args.model, args.drift_model, args.contexts, args.evidence_mode, log_level=args.log_level)
    exp.run(max_files=args.max_files, max_fields=args.max_fields)

if __name__ == "__main__":
    main()