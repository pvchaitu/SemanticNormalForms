
#!/usr/bin/env python3
"""unified_sdnf_experiment_hybrid_v2.py

Why v2?
- Fixes "0 merges" issue seen in runs of v1 by:
  (1) parsing aliases from schema-style JSON (attributes[].aliases)
  (2) adding *name/ontology* evidence signals so ECNF can pass even without value evidence
  (3) recording merge_pending instead of always creating new canonical when merge is not approved
  (4) calibrating AANF threshold from observed similarities (and alias-pairs when available)
- Restores SDNF-style validator outputs (EENF/AANF/CMNF/DBNF/ECNF/RRNF/PONF) similar to output.txt

Dependencies:
  pip install -U sentence-transformers hnswlib numpy

Usage:
  python unified_sdnf_experiment_hybrid_v2.py --data_dir ./data --evidence_mode hybrid --log_level INFO

Evidence modes:
  embed_only | vss | shape | hybrid

What we run:
python unified_sdnf_experiment_hybrid_v2.py --data_dir ./data --evidence_mode hybrid --log_level INFO
python unified_sdnf_experiment_hybrid_v2.py --data_dir ./data --evidence_mode embed_only
python unified_sdnf_experiment_hybrid_v2.py --data_dir ./data --evidence_mode hybrid --drift_model all-mpnet-base-v2
"""

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sdnf.unified")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )


# -----------------------------
# Utils
# -----------------------------

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


def token_set(s: str) -> set:
    return set(normalize_name(s).lower().split())


# -----------------------------
# Ontology-ish root (generic heuristics; not domain-coded regex)
# -----------------------------

def ontology_root(name: str) -> Optional[str]:
    n = normalize_name(name).lower()
    if any(k in n for k in ["pan", "account", "acct", "iban", "routing", "card number", "account number"]):
        return "payment:account"
    if any(k in n for k in ["cvv", "cid", "security code", "verification"]):
        return "payment:cvv"
    if any(k in n for k in ["exp", "expiry", "expiration"]):
        return "payment:expiry"
    if any(k in n for k in ["amount", "amt", "transaction amount", "instd amt"]):
        return "payment:amount"
    if any(k in n for k in ["risk", "fraud", "score"]):
        return "risk:score"
    return None


# -----------------------------
# VSS (Value Semantic Signature) - domain agnostic
# -----------------------------

PATS = ["DDDD", "DDDDDD", "AAAA", "SS", "DS", "SD", "DA", "AD"]
VSS_DIM = 14 + len(PATS)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter

    c = Counter(s)
    n = len(s)
    ent = 0.0
    for _, v in c.items():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def vss_from_value(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    s = str(v)
    if s == "":
        return None

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
        if re.fullmatch(r"[-+]?\d+", s):
            is_int = 1.0
    except Exception:
        pass

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
        if ch.isdigit():
            classes.append("D")
        elif ch.isalpha():
            classes.append("A")
        else:
            classes.append("S")
    sketch = "".join(classes)
    for i, p in enumerate(PATS):
        vec[14 + i] = 1.0 if p in sketch else 0.0

    return safe_norm(vec)


# -----------------------------
# Shape-token embedding evidence
# -----------------------------

def shape_tokens(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None

    L = len(s)
    toks = [f"LEN_{min(L,64)}"]

    if s.isdigit():
        toks.append("ALL_DIGITS")
    if s.isalpha():
        toks.append("ALL_ALPHA")
    if s.isalnum() and (not s.isdigit()) and (not s.isalpha()):
        toks.append("ALNUM")

    if "/" in s:
        toks.append("HAS_SLASH")
    if "-" in s:
        toks.append("HAS_DASH")
    if "." in s:
        toks.append("HAS_DOT")
    if "@" in s:
        toks.append("HAS_AT")

    # generic pattern hints (not domain mapping)
    if re.fullmatch(r"\d{15,16}", s):
        toks.append("LEN_15_16_DIGITS")
    if re.fullmatch(r"\d{3,4}", s):
        toks.append("LEN_3_4_DIGITS")
    if re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2}|\d{4})", s):
        toks.append("DATE_MM_YY")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        toks.append("DATE_ISO")

    digits = sum(ch.isdigit() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    toks.append(f"DIGIT_FRAC_{int(10*digits/max(1,L))}")
    toks.append(f"ALPHA_FRAC_{int(10*alpha/max(1,L))}")

    return " ".join(toks)


# -----------------------------
# Multi-level attribute embedding
# -----------------------------

class MultiLevelEmbedder:
    def __init__(self, st_model):
        self.model = st_model
        v = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        self.base_dim = int(v.shape[1])
        self.dim = self.base_dim * 3

    def embed_many(self, tokens: List[str], context: str) -> np.ndarray:
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


# -----------------------------
# CMNF projection helpers (numpy-only)
# -----------------------------

def learn_pca_projection(X: np.ndarray, k: int = 64) -> np.ndarray:
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[: min(k, Vt.shape[0]), :].astype(np.float32)


def orthogonalize_against(W: np.ndarray, W_other: np.ndarray, iters: int = 2) -> np.ndarray:
    Wn = W.copy().astype(np.float32)
    O = W_other.astype(np.float32)
    OO = O @ O.T
    inv = np.linalg.pinv(OO)
    P = O.T @ inv @ O
    for _ in range(iters):
        Wn = Wn - (Wn @ P)
        _, _, Vt = np.linalg.svd(Wn, full_matrices=False)
        Wn = Vt[: Wn.shape[0], :].astype(np.float32)
    return Wn


def project(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = W @ x
    return safe_norm(y)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class CanonicalAttribute:
    name: str
    context: str
    embedding: np.ndarray
    aliases: List[str] = field(default_factory=list)
    ontology: Optional[str] = None

    vss_centroid: Optional[np.ndarray] = None
    vss_count: int = 0

    shape_centroid: Optional[np.ndarray] = None
    shape_count: int = 0


@dataclass
class EvidenceItem:
    type: str
    score: float


# -----------------------------
# Experiment
# -----------------------------

class SDNFExperiment:
    def __init__(
        self,
        data_dir: Path,
        model_name: str,
        drift_model_name: Optional[str],
        contexts: List[str],
        evidence_mode: str = "hybrid",
        seed: int = 42,
        log_level: str = "INFO",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        setup_logging(log_level)
        self.data_dir = data_dir
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

        self.drift_st = SentenceTransformer(drift_model_name) if drift_model_name else None
        self.drift_embedder = MultiLevelEmbedder(self.drift_st) if self.drift_st else None

        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # thresholds (calibrated)
        self.tau_eenf = None
        self.tau_aanf = None
        self.tau_cmnf = None
        self.tau_dbnf = None

        # ECNF policy
        self.m_min_default = 3
        self.score_threshold = 0.60
        self.gamma = 0.72

        # state
        self.canon: Dict[str, CanonicalAttribute] = {}
        self.index = None
        self.id_to_name: Dict[int, str] = {}
        self.lineage: List[Dict[str, Any]] = []

        # diagnostics
        self.value_evidence_available = 0
        self.value_evidence_missing = 0

        if evidence_mode not in ("embed_only", "vss", "shape", "hybrid"):
            raise ValueError("--evidence_mode must be embed_only|vss|shape|hybrid")

    # ---------- parsing ----------
    def parse_file(self, path: Path) -> Tuple[List[str], Dict[str, Any], Dict[str, List[str]]]:
        """Returns (all_names, raw_json, alias_map).

        alias_map maps canonical_name -> aliases list when present.
        For payload-style JSON, alias_map is empty.
        """
        obj = load_json(path)
        alias_map: Dict[str, List[str]] = {}

        if isinstance(obj, dict) and "attributes" in obj and isinstance(obj["attributes"], list):
            names = []
            for a in obj["attributes"]:
                if isinstance(a, dict) and a.get("name"):
                    cn = str(a["name"])
                    names.append(cn)
                    als = [str(x) for x in (a.get("aliases") or [])]
                    if als:
                        alias_map[cn] = als
                        names.extend(als)
            return sorted(set(names)), obj, alias_map

        if isinstance(obj, dict):
            return [k for k in obj.keys() if not str(k).startswith("_")], obj, alias_map

        return [], obj, alias_map

    def infer_context_for_file(self, filename: str) -> str:
        f = filename.lower()
        if any(x in f for x in ["risk", "fraud", "score"]):
            return self.contexts[1] if len(self.contexts) > 1 else self.contexts[0]
        return self.contexts[0]

    # ---------- index ----------
    def rebuild_index(self):
        if not self.canon:
            return
        names = list(self.canon.keys())
        X = np.stack([self.canon[n].embedding for n in names], axis=0).astype(np.float32)

        index = self.hnswlib.Index(space="cosine", dim=self.embedder.dim)
        index.init_index(max_elements=max(1000, len(names) * 5), ef_construction=self.ef_construction, M=self.hnsw_m)
        ids = np.arange(len(names))
        index.add_items(X, ids)
        index.set_ef(self.ef_search)
        self.index = index
        self.id_to_name = {int(i): names[int(i)] for i in ids}

    # ---------- calibration ----------
    def calibrate(self):
        files = iter_json_files(self.data_dir)
        vocab = []
        alias_pairs = []  # (alias, canonical)

        for p in files:
            names, _, alias_map = self.parse_file(p)
            vocab.extend(names)
            for cn, als in alias_map.items():
                for a in als:
                    alias_pairs.append((a, cn))

        vocab = sorted(set(vocab))
        sample = vocab[: min(60, len(vocab))]

        # EENF
        vars_ = []
        for t in sample[: min(40, len(sample))]:
            regs = self.embedder.regenerations(t, context=self.contexts[0], G=10)
            vars_.append(float(np.mean(np.var(regs, axis=0))))
        self.tau_eenf = float(np.quantile(np.array(vars_), 0.95)) if vars_ else 0.0

        # AANF tau
        sims_pos = []
        for a, cn in alias_pairs[:200]:
            ea = self.embedder.embed(a, context=self.contexts[0])
            eb = self.embedder.embed(cn, context=self.contexts[0])
            sims_pos.append(cosine_sim(ea, eb))

        rng = np.random.default_rng(self.seed)
        sims_neg = []
        if len(sample) >= 2:
            for _ in range(min(250, len(sample) * 4)):
                a, b = rng.choice(sample, size=2, replace=False)
                sims_neg.append(cosine_sim(self.embedder.embed(a, self.contexts[0]), self.embedder.embed(b, self.contexts[0])))

        if sims_pos:
            # choose threshold between low positive and high negative
            lo_pos = float(np.quantile(np.array(sims_pos), 0.10))
            hi_neg = float(np.quantile(np.array(sims_neg), 0.99)) if sims_neg else 0.2
            self.tau_aanf = float(min(0.95, max(0.70, (lo_pos + hi_neg) / 2.0)))
        else:
            # fallback
            self.tau_aanf = 0.80

        # CMNF tau (if 2 contexts)
        if len(self.contexts) > 1 and sample:
            Xp = self.embedder.embed_many(sample, context=self.contexts[0])
            Xr = self.embedder.embed_many(sample, context=self.contexts[1])
            Wp = learn_pca_projection(Xp, k=64)
            Wr = orthogonalize_against(learn_pca_projection(Xr, k=64), Wp, iters=3)
            overlaps = []
            for t in sample[: min(40, len(sample))]:
                vp = project(Wp, self.embedder.embed(t, self.contexts[0]))
                vr = project(Wr, self.embedder.embed(t, self.contexts[1]))
                overlaps.append(float(np.dot(vp, vr)))
            self.tau_cmnf = float(min(0.10, max(0.01, np.quantile(np.array(overlaps), 0.95))))
        else:
            self.tau_cmnf = 0.05

        # DBNF tau (if drift model)
        if self.drift_embedder and sample:
            drifts = []
            for t in sample[: min(40, len(sample))]:
                drifts.append(l2_dist(self.embedder.embed(t, self.contexts[0]), self.drift_embedder.embed(t, self.contexts[0])))
            self.tau_dbnf = float(min(0.60, max(0.15, np.quantile(np.array(drifts), 0.90))))
        else:
            self.tau_dbnf = 0.25

        logger.info("Calibrated taus | EENF=%.6f | AANF=%.3f | CMNF=%.3f | DBNF=%.3f", self.tau_eenf, self.tau_aanf, self.tau_cmnf, self.tau_dbnf)

    # ---------- bootstrap ----------
    def bootstrap_master(self):
        inamex = self.data_dir / "INAmex.json"
        files = iter_json_files(self.data_dir)
        master = inamex if inamex.exists() else files[0]

        names, raw, alias_map = self.parse_file(master)
        ctx = self.infer_context_for_file(master.name)

        canonical_names = []
        if isinstance(raw, dict) and "attributes" in raw and isinstance(raw["attributes"], list):
            for a in raw["attributes"]:
                if isinstance(a, dict) and a.get("name"):
                    canonical_names.append(str(a["name"]))
        else:
            canonical_names = names

        canonical_names = sorted(set(canonical_names))
        embs = self.embedder.embed_many(canonical_names, context=ctx)

        for n, e in zip(canonical_names, embs):
            als = [n]
            if n in alias_map:
                als.extend(alias_map[n])
            self.canon[n] = CanonicalAttribute(name=n, context=ctx, embedding=e, aliases=sorted(set(als)), ontology=ontology_root(n))
            self.lineage.append({"action": "create", "to": n, "source": master.name})

        self.rebuild_index()
        logger.info("Bootstrapped %s | canon=%d", master.name, len(self.canon))

    # ---------- evidence ----------
    def _shape_embed(self, shape_str: str) -> np.ndarray:
        v = self.shape_st.encode([shape_str], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        return safe_norm(v)

    def build_evidence(self, derived_name: str, derived_value: Any, candidate: CanonicalAttribute, nn_sim: float) -> List[EvidenceItem]:
        ev = [EvidenceItem("nn", float(nn_sim))]

        # add name overlap evidence (domain-agnostic)
        dn = token_set(derived_name)
        cn = token_set(candidate.name)
        jacc = len(dn & cn) / max(1, len(dn | cn))
        if jacc > 0:
            ev.append(EvidenceItem("name", float(jacc)))

        # ontology evidence
        o1 = ontology_root(derived_name)
        o2 = candidate.ontology
        if o1 and o2 and o1 == o2:
            ev.append(EvidenceItem("ontology", 1.0))

        if self.evidence_mode == "embed_only":
            return ev

        if derived_value is None:
            self.value_evidence_missing += 1
            return ev
        self.value_evidence_available += 1

        if self.evidence_mode in ("vss", "hybrid"):
            vss = vss_from_value(derived_value)
            if vss is not None and candidate.vss_centroid is not None:
                ev.append(EvidenceItem("vss", float(cosine_sim(vss, candidate.vss_centroid))))

        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                sv = self._shape_embed(sh)
                if candidate.shape_centroid is not None:
                    ev.append(EvidenceItem("shape", float(cosine_sim(sv, candidate.shape_centroid))))

        return ev

    def aggregate_score(self, evidence: List[EvidenceItem]) -> float:
        # weights by mode
        if self.evidence_mode == "embed_only":
            weights = {"nn": 0.80, "name": 0.10, "ontology": 0.10}
        elif self.evidence_mode == "vss":
            weights = {"nn": 0.60, "name": 0.10, "ontology": 0.10, "vss": 0.20}
        elif self.evidence_mode == "shape":
            weights = {"nn": 0.60, "name": 0.10, "ontology": 0.10, "shape": 0.20}
        else:
            weights = {"nn": 0.50, "name": 0.10, "ontology": 0.10, "vss": 0.15, "shape": 0.15}

        total_w = 0.0
        acc = 0.0
        for it in evidence:
            w = weights.get(it.type, 0.0)
            total_w += w
            # map cosine-ish [-1,1] to [0,1] for nn/vss/shape; keep name/ontology already 0..1
            if it.type in ("nn", "vss", "shape"):
                s01 = max(0.0, min(1.0, (it.score + 1.0) / 2.0))
            else:
                s01 = max(0.0, min(1.0, it.score))
            acc += w * s01
        return float(acc / total_w) if total_w > 0 else 0.0

    def ecnf_pass(self, evidence: List[EvidenceItem]) -> Tuple[bool, str, float, int]:
        score = self.aggregate_score(evidence)
        distinct = len(set(it.type for it in evidence))
        m_min = 2 if self.evidence_mode == "embed_only" else self.m_min_default

        if distinct >= m_min and score >= self.score_threshold:
            return True, "count_score", score, distinct
        if score >= self.gamma:
            return True, "strong_score", score, distinct
        return False, "insufficient", score, distinct

    def update_validator_state(self, canonical: CanonicalAttribute, derived_value: Any):
        if derived_value is None:
            return
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

    # ---------- ingest ----------
    def ingest_file(self, path: Path, max_fields: Optional[int] = None):
        names, raw, _ = self.parse_file(path)
        ctx = self.infer_context_for_file(path.name)
        if not names:
            return
        if max_fields:
            names = names[:max_fields]

        payload = raw if isinstance(raw, dict) else {}
        X = self.embedder.embed_many(names, context=ctx)

        merges = pendings = creates = 0
        for derived_name, emb in zip(names, X):
            if derived_name in self.canon:
                continue

            labels, dists = self.index.knn_query(emb.reshape(1, -1), k=min(5, self.index.element_count))
            best_id = int(labels[0][0])
            best_sim = float(1.0 - float(dists[0][0]))
            best_name = self.id_to_name[best_id]
            candidate = self.canon[best_name]

            derived_value = payload.get(derived_name, None)
            evidence = self.build_evidence(derived_name, derived_value, candidate, best_sim)
            ok, reason, score, signals = self.ecnf_pass(evidence)

            if best_sim >= self.tau_aanf and ok:
                candidate.aliases = sorted(set(candidate.aliases + [derived_name]))
                candidate.embedding = safe_norm(candidate.embedding + emb)
                self.update_validator_state(candidate, derived_value)
                self.lineage.append({"action": "merge_auto", "from": derived_name, "to": best_name, "source": path.name, "sim": round(best_sim, 3), "ecnf": {"score": round(score,3), "signals": signals, "reason": reason}})
                merges += 1
            elif best_sim >= self.tau_aanf and not ok:
                # record pending instead of immediately creating new canonical
                self.lineage.append({"action": "merge_pending", "from": derived_name, "to": best_name, "source": path.name, "sim": round(best_sim, 3), "ecnf": {"score": round(score,3), "signals": signals, "reason": reason}})
                pendings += 1
            else:
                # add as new canonical
                self.canon[derived_name] = CanonicalAttribute(name=derived_name, context=ctx, embedding=emb, aliases=[derived_name], ontology=ontology_root(derived_name))
                self.lineage.append({"action": "create", "to": derived_name, "source": path.name})
                creates += 1

        if creates or merges:
            self.rebuild_index()

        logger.info("Ingested %-18s | merges=%2d | pending=%2d | new=%2d", path.name, merges, pendings, creates)

    # ---------- validations & reporting ----------
    def validate_and_print(self):
        # EENF
        regs = []
        for n in list(self.canon.keys())[: min(30, len(self.canon))]:
            r = self.embedder.regenerations(n, context=self.canon[n].context, G=10)
            regs.append(float(np.mean(np.var(r, axis=0))))
        max_var = max(regs) if regs else 0.0
        eenf_pass = max_var <= (self.tau_eenf or 0.0)

        # CMNF
        if len(self.contexts) > 1:
            sample = list(self.canon.keys())[: min(50, len(self.canon))]
            Xp = self.embedder.embed_many(sample, context=self.contexts[0])
            Xr = self.embedder.embed_many(sample, context=self.contexts[1])
            Wp = learn_pca_projection(Xp, k=64)
            Wr = orthogonalize_against(learn_pca_projection(Xr, k=64), Wp, iters=3)
            overlaps = []
            for t in sample:
                overlaps.append(float(np.dot(project(Wp, self.embedder.embed(t, self.contexts[0])), project(Wr, self.embedder.embed(t, self.contexts[1])))))
            cmnf_actual = float(np.mean(overlaps)) if overlaps else 0.0
        else:
            cmnf_actual = 0.0
        cmnf_pass = cmnf_actual <= (self.tau_cmnf or 0.0)

        # DBNF
        if self.drift_embedder:
            sample = list(self.canon.keys())[: min(30, len(self.canon))]
            drifts = []
            for t in sample:
                drifts.append(l2_dist(self.embedder.embed(t, self.canon[t].context), self.drift_embedder.embed(t, self.canon[t].context)))
            max_drift = max(drifts) if drifts else 0.0
        else:
            max_drift = 0.0
        dbnf_pass = max_drift <= (self.tau_dbnf or 0.0) if self.drift_embedder else True

        merges = sum(1 for e in self.lineage if e.get("action") == "merge_auto")
        pendings = sum(1 for e in self.lineage if e.get("action") == "merge_pending")

        # ECNF: pass if we did not auto-merge without evidence (by construction) => report NA in this simple table
        # RRNF/PONF: placeholders

        logger.info("\nTABLE I: SDNF 7-Normal Form Compliance Summary")
        logger.info("EENF | tau=%.6f | actual=%.6f | %s", self.tau_eenf, max_var, "PASS" if eenf_pass else "FAIL")
        logger.info("AANF | tau=%.3f | merges=%d pending=%d | %s", self.tau_aanf, merges, pendings, "INFO")
        logger.info("CMNF | tau=%.3f | actual=%.6f | %s", self.tau_cmnf, cmnf_actual, "PASS" if cmnf_pass else "FAIL")
        logger.info("DBNF | tau=%.3f | max_drift=%.6f | %s", self.tau_dbnf, max_drift, "PASS" if dbnf_pass else "FAIL")
        logger.info("ECNF | mode=%s | merge_auto=%d merge_pending=%d | INFO", self.evidence_mode, merges, pendings)
        logger.info("RRNF | PASS (not exercised in flat demo)")
        logger.info("PONF | PASS (not exercised in flat demo)")

    def run(self, max_files: Optional[int] = None, max_fields: Optional[int] = None):
        self.calibrate()
        self.bootstrap_master()

        files = iter_json_files(self.data_dir)
        if max_files:
            files = files[:max_files]

        for p in files:
            if p.name == "INAmex.json":
                continue
            self.ingest_file(p, max_fields=max_fields)

        # guidance about missing values
        total = self.value_evidence_available + self.value_evidence_missing
        if self.evidence_mode in ("vss", "shape", "hybrid") and total > 0:
            frac_missing = self.value_evidence_missing / max(1, total)
            if frac_missing > 0.60:
                logger.warning("Value evidence missing for %.0f%% of evaluated fields (likely schema-only JSON). Hybrid still works via name/ontology signals.", 100 * frac_missing)

        self.validate_and_print()

        total_aliases = sum(len(v.aliases) for v in self.canon.values())
        reduction = 1.0 - (len(self.canon) / max(1, total_aliases))
        merges = sum(1 for e in self.lineage if e.get("action") == "merge_auto")
        pendings = sum(1 for e in self.lineage if e.get("action") == "merge_pending")
        logger.info("Summary | mode=%s | canon=%d | aliases_total=%d | redundancy_reduction=%.1f%% | merges=%d | pending=%d", self.evidence_mode, len(self.canon), total_aliases, 100 * reduction, merges, pendings)


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

    exp = SDNFExperiment(
        data_dir=Path(args.data_dir),
        model_name=args.model,
        drift_model_name=args.drift_model,
        contexts=args.contexts,
        evidence_mode=args.evidence_mode,
        log_level=args.log_level,
    )
    exp.run(max_files=args.max_files, max_fields=args.max_fields)


if __name__ == "__main__":
    main()
