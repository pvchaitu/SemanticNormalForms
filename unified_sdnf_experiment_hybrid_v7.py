#!/usr/bin/env python3
"""unified_sdnf_experiment_hybrid_v7.py

SDNF Unified Experiment (v7)

Polish added (requested):
- In --evidence_mode all comparison table, include EENF/CMNF/DBNF actual vs tau (a/t) columns.

v7 retains v6 features:
- Defaults: --data_dir=data, --payloads_dir=payloads, --evidence_mode=hybrid, --drift_model optional
- evidence_mode=all runs embed_only/vss/shape/hybrid and prints comparison table
- Quiet external logs (HuggingFace/HTTP) ON by default; override with --show_external
- Consolidated reporting tables for single-mode runs

Dependencies:
  pip install -U sentence-transformers hnswlib numpy

"""

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sdnf.unified")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )


def suppress_external_logs(enable: bool = True):
    """Reduce noisy console output from HuggingFace / Transformers / HTTP stack."""
    if not enable:
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    for name in [
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as tr_logging
        tr_logging.set_verbosity_error()
    except Exception:
        pass


# -----------------------------
# Formatting helpers
# -----------------------------

def render_table(headers: List[str], rows: List[List[Any]], title: Optional[str] = None) -> str:
    srows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(ch: str = '-'):
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def fmt_row(r):
        return "|" + "|".join(" " + r[i].ljust(widths[i]) + " " for i in range(len(headers))) + "|"

    out = []
    if title:
        out.append(title)
    out.append(line('-'))
    out.append(fmt_row(headers))
    out.append(line('='))
    for r in srows:
        out.append(fmt_row(r))
    out.append(line('-'))
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{100.0*x:.1f}%"


# -----------------------------
# IO + basic math
# -----------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.json") if p.is_file()])


def camel_to_tokens(s: str) -> str:
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s2 = s2.replace("_", " ")
    return " ".join(s2.split())


def normalize_name(name: str) -> str:
    return camel_to_tokens(name).strip()


def token_set(s: str) -> set:
    return set(normalize_name(s).lower().split())


def safe_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return (x / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Drift alignment helper

def fit_linear_map(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (np.linalg.pinv(B) @ A).astype(np.float32)


def apply_linear_map(b: np.ndarray, M: np.ndarray) -> np.ndarray:
    return safe_norm(b @ M)


# -----------------------------
# Ontology-ish root (lightweight heuristic) — evidence, not hard mapping
# -----------------------------

def ontology_root(name: str) -> Optional[str]:
    n = normalize_name(name).lower()
    if any(k in n for k in ["pan", "account", "acct", "iban", "routing", "card number", "account number", "primary account number", "acct_num"]):
        return "payment:account"
    if any(k in n for k in ["cvv", "cid", "security code", "verification", "cardverificationvalue"]):
        return "payment:cvv"
    if any(k in n for k in ["exp", "expiry", "expiration", "expirationdate"]):
        return "payment:expiry"
    if any(k in n for k in ["amount", "amt", "transaction amount", "instd amt", "txn amount", "txn_amount", "instd_amt"]):
        return "payment:amount"
    if any(k in n for k in ["risk", "fraud", "score"]):
        return "risk:score"
    return None


# -----------------------------
# VSS (Value Semantic Signature)
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
# Shape-token evidence
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
# Multi-level embeddings
# -----------------------------

class MultiLevelEmbedder:
    def __init__(self, st_model):
        self.model = st_model
        v = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        self.base_dim = int(v.shape[1])
        self.dim = self.base_dim * 3

    def embed_components_many(self, tokens: List[str], context: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fine = self.model.encode(tokens, normalize_embeddings=True, show_progress_bar=False)
        abstract = self.model.encode([normalize_name(t) for t in tokens], normalize_embeddings=True, show_progress_bar=False)
        contextual = self.model.encode([f"{t} in {context} context" for t in tokens], normalize_embeddings=True, show_progress_bar=False)
        return fine.astype(np.float32), abstract.astype(np.float32), contextual.astype(np.float32)

    def embed_many(self, tokens: List[str], context: str) -> np.ndarray:
        fine, abstract, contextual = self.embed_components_many(tokens, context)
        X = np.concatenate([fine, abstract, contextual], axis=1).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def embed_components(self, token: str, context: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, a, c = self.embed_components_many([token], context)
        return f[0], a[0], c[0]

    def embed(self, token: str, context: str) -> np.ndarray:
        return self.embed_many([token], context)[0]

    def regenerations(self, token: str, context: str, G: int = 10) -> np.ndarray:
        variants = [token, token + " ", token + ".", token + "!", f"{token} field"]
        variants = (variants * ((G + len(variants) - 1) // len(variants)))[:G]
        return self.embed_many(variants, context=context)


# -----------------------------
# CMNF projection (numpy-only)
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
    return safe_norm(W @ x)


@dataclass
class EvidenceItem:
    type: str
    score: float


@dataclass
class CanonicalAttribute:
    name: str
    context: str
    emb: np.ndarray
    fine: np.ndarray
    abstract: np.ndarray
    contextual: np.ndarray
    aliases: List[str] = field(default_factory=list)
    ontology: Optional[str] = None
    alias_emb: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    vss_centroid: Optional[np.ndarray] = None
    vss_count: int = 0
    shape_centroid: Optional[np.ndarray] = None
    shape_count: int = 0


class SDNFExperiment:
    def __init__(
        self,
        data_dir: Path,
        payloads_dir: Path,
        model_name: str,
        drift_model_name: Optional[str],
        contexts: List[str],
        evidence_mode: str,
        seed: int,
        log_level: str,
        promote_sources: int,
        promote_hits: int,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        setup_logging(log_level)
        self.data_dir = data_dir
        self.payloads_dir = payloads_dir
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
        self.drift_map = None

        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.tau_eenf = None
        self.tau_aanf = None
        self.tau_cmnf = None
        self.tau_dbnf = None

        self.m_min_default = 4
        self.score_threshold = 0.62
        self.gamma = 0.75

        self.promote_sources = promote_sources
        self.promote_hits = promote_hits
        self.pending_support: Dict[Tuple[str, str], set] = {}
        self.pending_hits: Dict[Tuple[str, str], int] = {}

        self.canon: Dict[str, CanonicalAttribute] = {}
        self.index = None
        self.id_to_name: Dict[int, str] = {}
        self.lineage: List[Dict[str, Any]] = []
        self.source_stats: Dict[str, Dict[str, int]] = {}

        self.value_evidence_available = 0
        self.value_evidence_missing = 0

    def parse_schema_file(self, path: Path) -> Tuple[List[str], Dict[str, Any], Dict[str, List[str]]]:
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

    def parse_payload_file(self, path: Path) -> Tuple[List[str], Dict[str, Any]]:
        obj = load_json(path)
        if isinstance(obj, dict):
            keys = [k for k in obj.keys() if not str(k).startswith("_")]
            return keys, obj
        return [], {}

    def infer_context_for_source(self, source: str) -> str:
        f = source.lower()
        if any(x in f for x in ["risk", "fraud", "score"]):
            return self.contexts[1] if len(self.contexts) > 1 else self.contexts[0]
        return self.contexts[0]

    def rebuild_index(self):
        if not self.canon:
            return
        names = list(self.canon.keys())
        X = np.stack([self.canon[n].emb for n in names], axis=0).astype(np.float32)
        index = self.hnswlib.Index(space="cosine", dim=self.embedder.dim)
        index.init_index(max_elements=max(2000, len(names) * 10), ef_construction=self.ef_construction, M=self.hnsw_m)
        ids = np.arange(len(names))
        index.add_items(X, ids)
        index.set_ef(self.ef_search)
        self.index = index
        self.id_to_name = {int(i): names[int(i)] for i in ids}

    def _ensure_alias_embeddings(self, ca: CanonicalAttribute):
        for al in ca.aliases:
            if al in ca.alias_emb:
                continue
            f, a, c = self.embedder.embed_components(al, ca.context)
            e = safe_norm(np.concatenate([f, a, c], axis=0))
            ca.alias_emb[al] = (e, f, a, c)

    def calibrate(self):
        schema_files = iter_json_files(self.data_dir)
        vocab = []
        alias_pairs = []

        for p in schema_files:
            names, _, alias_map = self.parse_schema_file(p)
            vocab.extend(names)
            for cn, als in alias_map.items():
                for a in als:
                    alias_pairs.append((a, cn))

        vocab = sorted(set(vocab))
        sample = vocab[: min(80, len(vocab))]

        # EENF tau
        vars_ = []
        for t in sample[: min(40, len(sample))]:
            regs = self.embedder.regenerations(t, context=self.contexts[0], G=10)
            vars_.append(float(np.mean(np.var(regs, axis=0))))
        self.tau_eenf = float(np.quantile(np.array(vars_), 0.95)) if vars_ else 0.0

        # AANF tau
        sims_pos = []
        for a, cn in alias_pairs[:300]:
            fa, aa, ca = self.embedder.embed_components(a, self.contexts[0])
            fb, ab, cb = self.embedder.embed_components(cn, self.contexts[0])
            sims_pos.append(max(cosine_sim(fa, fb), cosine_sim(aa, ab), cosine_sim(ca, cb)))

        rng = np.random.default_rng(self.seed)
        sims_neg = []
        if len(sample) >= 2:
            for _ in range(min(400, len(sample) * 4)):
                x, y = rng.choice(sample, size=2, replace=False)
                fx, ax, cx = self.embedder.embed_components(x, self.contexts[0])
                fy, ay, cy = self.embedder.embed_components(y, self.contexts[0])
                sims_neg.append(max(cosine_sim(fx, fy), cosine_sim(ax, ay), cosine_sim(cx, cy)))

        if sims_pos:
            lo_pos = float(np.quantile(np.array(sims_pos), 0.10))
            hi_neg = float(np.quantile(np.array(sims_neg), 0.99)) if sims_neg else 0.2
            self.tau_aanf = float(min(0.95, max(0.65, (lo_pos + hi_neg) / 2.0)))
        else:
            self.tau_aanf = 0.75

        # CMNF tau
        if len(self.contexts) > 1 and sample:
            Xp = self.embedder.embed_many(sample, context=self.contexts[0])
            Xr = self.embedder.embed_many(sample, context=self.contexts[1])
            Wp = learn_pca_projection(Xp, k=64)
            Wr = orthogonalize_against(learn_pca_projection(Xr, k=64), Wp, iters=3)
            overlaps = []
            for t in sample[: min(40, len(sample))]:
                overlaps.append(float(np.dot(project(Wp, self.embedder.embed(t, self.contexts[0])), project(Wr, self.embedder.embed(t, self.contexts[1])))))
            self.tau_cmnf = float(min(0.10, max(0.01, np.quantile(np.array(overlaps), 0.95))))
        else:
            self.tau_cmnf = 0.05

        # DBNF tau
        if self.drift_embedder and sample:
            A = self.embedder.embed_many(sample[: min(60, len(sample))], context=self.contexts[0])
            B = self.drift_embedder.embed_many(sample[: min(60, len(sample))], context=self.contexts[0])
            if A.shape[1] != B.shape[1]:
                self.drift_map = fit_linear_map(B, A)
                logger.warning("DBNF: drift_model dim (%d) != base dim (%d). Learned alignment map.", B.shape[1], A.shape[1])
                Bm = B @ self.drift_map
                Bm = Bm / (np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-12)
                drifts = [l2_dist(A[i], Bm[i]) for i in range(A.shape[0])]
            else:
                drifts = [l2_dist(A[i], B[i]) for i in range(A.shape[0])]
            self.tau_dbnf = float(min(0.60, max(0.15, np.quantile(np.array(drifts), 0.90))))
        else:
            self.tau_dbnf = 0.25

        logger.info("Calibrated taus | EENF=%.6f | AANF=%.3f | CMNF=%.3f | DBNF=%.3f", self.tau_eenf, self.tau_aanf, self.tau_cmnf, self.tau_dbnf)

    def bootstrap_master(self):
        inamex = self.data_dir / "INAmex.json"
        schema_files = iter_json_files(self.data_dir)
        master = inamex if inamex.exists() else (schema_files[0] if schema_files else None)
        if master is None:
            raise RuntimeError(f"No schema JSON files found in {self.data_dir}")

        names, raw, alias_map = self.parse_schema_file(master)
        ctx = self.infer_context_for_source(master.name)

        canonical_names = []
        if isinstance(raw, dict) and "attributes" in raw and isinstance(raw["attributes"], list):
            for a in raw["attributes"]:
                if isinstance(a, dict) and a.get("name"):
                    canonical_names.append(str(a["name"]))
        else:
            canonical_names = names

        canonical_names = sorted(set(canonical_names))
        fine, abstract, contextual = self.embedder.embed_components_many(canonical_names, context=ctx)
        concat = np.concatenate([fine, abstract, contextual], axis=1).astype(np.float32)
        concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-12)

        for n, e, f, a, c in zip(canonical_names, concat, fine, abstract, contextual):
            als = [n]
            if n in alias_map:
                als.extend(alias_map[n])
            ca = CanonicalAttribute(
                name=n,
                context=ctx,
                emb=e,
                fine=f,
                abstract=a,
                contextual=c,
                aliases=sorted(set(als)),
                ontology=ontology_root(n),
            )
            self._ensure_alias_embeddings(ca)
            self.canon[n] = ca
            self.lineage.append({"action": "create", "to": n, "source": master.name})

        self.rebuild_index()
        logger.info("Bootstrapped %s | canon=%d", master.name, len(self.canon))

    def _shape_embed(self, shape_str: str) -> np.ndarray:
        v = self.shape_st.encode([shape_str], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        return safe_norm(v)

    def alias_aware_sims(self, derived: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ca: CanonicalAttribute) -> Dict[str, float]:
        d_emb, d_f, d_a, d_c = derived
        self._ensure_alias_embeddings(ca)
        best = {
            "nn": cosine_sim(d_emb, ca.emb),
            "fine_nn": cosine_sim(d_f, ca.fine),
            "abstract_nn": cosine_sim(d_a, ca.abstract),
            "contextual_nn": cosine_sim(d_c, ca.contextual),
        }
        for _, (e, f, a, c) in ca.alias_emb.items():
            best["nn"] = max(best["nn"], cosine_sim(d_emb, e))
            best["fine_nn"] = max(best["fine_nn"], cosine_sim(d_f, f))
            best["abstract_nn"] = max(best["abstract_nn"], cosine_sim(d_a, a))
            best["contextual_nn"] = max(best["contextual_nn"], cosine_sim(d_c, c))
        return best

    def build_evidence(self, derived_name: str, derived_value: Any, ca: CanonicalAttribute, sims: Dict[str, float]) -> List[EvidenceItem]:
        ev = [
            EvidenceItem("nn", float(sims["nn"])),
            EvidenceItem("fine_nn", float(sims["fine_nn"])),
            EvidenceItem("abstract_nn", float(sims["abstract_nn"])),
            EvidenceItem("contextual_nn", float(sims["contextual_nn"])),
        ]

        dn = token_set(derived_name)
        cn = token_set(ca.name)
        jacc = len(dn & cn) / max(1, len(dn | cn))
        if jacc > 0:
            ev.append(EvidenceItem("name", float(jacc)))

        o1 = ontology_root(derived_name)
        if o1 and ca.ontology and o1 == ca.ontology:
            ev.append(EvidenceItem("ontology", 1.0))

        if self.evidence_mode == "embed_only":
            return ev

        if derived_value is None:
            self.value_evidence_missing += 1
            return ev
        self.value_evidence_available += 1

        if self.evidence_mode in ("vss", "hybrid"):
            vss = vss_from_value(derived_value)
            if vss is not None and ca.vss_centroid is not None:
                ev.append(EvidenceItem("vss", float(cosine_sim(vss, ca.vss_centroid))))

        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                sv = self._shape_embed(sh)
                if ca.shape_centroid is not None:
                    ev.append(EvidenceItem("shape", float(cosine_sim(sv, ca.shape_centroid))))

        return ev

    def aggregate_score(self, evidence: List[EvidenceItem]) -> float:
        if self.evidence_mode == "embed_only":
            weights = {"nn": 0.30, "fine_nn": 0.15, "abstract_nn": 0.25, "contextual_nn": 0.15, "name": 0.10, "ontology": 0.05}
        elif self.evidence_mode == "vss":
            weights = {"nn": 0.25, "fine_nn": 0.10, "abstract_nn": 0.25, "contextual_nn": 0.10, "name": 0.10, "ontology": 0.05, "vss": 0.15}
        elif self.evidence_mode == "shape":
            weights = {"nn": 0.25, "fine_nn": 0.10, "abstract_nn": 0.25, "contextual_nn": 0.10, "name": 0.10, "ontology": 0.05, "shape": 0.15}
        else:
            weights = {"nn": 0.22, "fine_nn": 0.10, "abstract_nn": 0.22, "contextual_nn": 0.10, "name": 0.10, "ontology": 0.06, "vss": 0.10, "shape": 0.10}

        total_w = 0.0
        acc = 0.0
        for it in evidence:
            w = weights.get(it.type, 0.0)
            if w <= 0:
                continue
            total_w += w
            if it.type in ("nn", "fine_nn", "abstract_nn", "contextual_nn", "vss", "shape"):
                s01 = max(0.0, min(1.0, (it.score + 1.0) / 2.0))
            else:
                s01 = max(0.0, min(1.0, it.score))
            acc += w * s01
        return float(acc / total_w) if total_w > 0 else 0.0

    def ecnf_pass(self, evidence: List[EvidenceItem]) -> Tuple[bool, str, float, int]:
        score = self.aggregate_score(evidence)
        distinct = len(set(it.type for it in evidence))
        m_min = 3 if self.evidence_mode == "embed_only" else self.m_min_default
        if distinct >= m_min and score >= self.score_threshold:
            return True, "count_score", score, distinct
        if score >= self.gamma:
            return True, "strong_score", score, distinct
        return False, "insufficient", score, distinct

    def update_value_state(self, ca: CanonicalAttribute, derived_value: Any):
        if derived_value is None:
            return
        if self.evidence_mode in ("vss", "hybrid"):
            v = vss_from_value(derived_value)
            if v is not None:
                alpha = 1.0 / float(min(50, ca.vss_count + 1))
                ca.vss_centroid = v if ca.vss_centroid is None else safe_norm((1 - alpha) * ca.vss_centroid + alpha * v)
                ca.vss_count += 1
        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                sv = self._shape_embed(sh)
                alpha = 1.0 / float(min(50, ca.shape_count + 1))
                ca.shape_centroid = sv if ca.shape_centroid is None else safe_norm((1 - alpha) * ca.shape_centroid + alpha * sv)
                ca.shape_count += 1

    def ingest_names(self, names: List[str], payload: Dict[str, Any], source: str):
        ctx = self.infer_context_for_source(source)
        if not names:
            return

        fine, abstract, contextual = self.embedder.embed_components_many(names, context=ctx)
        concat = np.concatenate([fine, abstract, contextual], axis=1).astype(np.float32)
        concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-12)

        merges = pendings = creates = promotions = 0

        for derived_name, d_emb, d_f, d_a, d_c in zip(names, concat, fine, abstract, contextual):
            if derived_name in self.canon:
                continue
            labels, _ = self.index.knn_query(d_emb.reshape(1, -1), k=min(8, self.index.element_count))
            best_id = int(labels[0][0])
            best_name = self.id_to_name[best_id]
            ca = self.canon[best_name]

            sims = self.alias_aware_sims((d_emb, d_f, d_a, d_c), ca)
            best_sim = sims["nn"]

            derived_value = payload.get(derived_name, None)
            evidence = self.build_evidence(derived_name, derived_value, ca, sims)
            ok, reason, score, signals = self.ecnf_pass(evidence)

            key = (derived_name, best_name)
            if best_sim >= self.tau_aanf and not ok:
                self.pending_hits[key] = self.pending_hits.get(key, 0) + 1
                self.pending_support.setdefault(key, set()).add(source)
                if len(self.pending_support[key]) >= self.promote_sources or self.pending_hits[key] >= self.promote_hits:
                    ok = True
                    reason = "promoted"
                    promotions += 1

            if best_sim >= self.tau_aanf and ok:
                if derived_name not in ca.aliases:
                    ca.aliases.append(derived_name)

                ca.emb = safe_norm(ca.emb + d_emb)
                ca.fine = safe_norm(ca.fine + d_f)
                ca.abstract = safe_norm(ca.abstract + d_a)
                ca.contextual = safe_norm(ca.contextual + d_c)

                ca.alias_emb.pop(derived_name, None)
                self._ensure_alias_embeddings(ca)
                self.update_value_state(ca, derived_value)

                self.lineage.append({
                    "action": "merge_auto",
                    "from": derived_name,
                    "to": best_name,
                    "source": source,
                    "sim": round(best_sim, 3),
                    "ecnf": {"pass": True, "reason": reason, "score": round(score, 3), "signals": signals},
                    "sims": {k: round(float(v), 3) for k, v in sims.items()},
                })
                merges += 1
            elif best_sim >= self.tau_aanf and not ok:
                self.lineage.append({
                    "action": "merge_pending",
                    "from": derived_name,
                    "to": best_name,
                    "source": source,
                    "sim": round(best_sim, 3),
                    "ecnf": {"pass": False, "reason": reason, "score": round(score, 3), "signals": signals},
                    "sims": {k: round(float(v), 3) for k, v in sims.items()},
                })
                pendings += 1
            else:
                ca2 = CanonicalAttribute(
                    name=derived_name,
                    context=ctx,
                    emb=d_emb,
                    fine=d_f,
                    abstract=d_a,
                    contextual=d_c,
                    aliases=[derived_name],
                    ontology=ontology_root(derived_name),
                )
                self._ensure_alias_embeddings(ca2)
                self.canon[derived_name] = ca2
                self.lineage.append({"action": "create", "to": derived_name, "source": source})
                creates += 1

        if creates or merges:
            self.rebuild_index()

        self.source_stats[source] = {"merges": merges, "pending": pendings, "promoted": promotions, "new": creates}
        logger.info("Ingested %-22s | merges=%2d | pending=%2d | promoted=%2d | new=%2d", source, merges, pendings, promotions, creates)

    def ingest_schema_file(self, path: Path):
        names, _, _ = self.parse_schema_file(path)
        self.ingest_names(names, payload={}, source=path.name)

    def ingest_payload_file(self, path: Path):
        names, payload = self.parse_payload_file(path)
        self.ingest_names(names, payload=payload, source=path.name)

    def report(self, run_cfg: Dict[str, Any]):
        cfg_rows = []
        for k in [
            "timestamp", "evidence_mode", "model", "drift_model", "contexts", "promote_sources", "promote_hits",
            "schema_files_ingested", "payload_files_ingested", "payloads_dir", "data_dir",
        ]:
            cfg_rows.append([k, run_cfg.get(k)])

        total_aliases = sum(len(v.aliases) for v in self.canon.values())
        reduction = 1.0 - (len(self.canon) / max(1, total_aliases))
        cfg_rows += [["canon_final", len(self.canon)], ["aliases_total", total_aliases], ["redundancy_reduction", pct(reduction)]]
        logger.info("\n" + render_table(["Option", "Value"], cfg_rows, title="RUN CONFIGURATION"))

        # compute actuals
        regs = []
        for n in list(self.canon.keys())[: min(40, len(self.canon))]:
            r = self.embedder.regenerations(n, context=self.canon[n].context, G=10)
            regs.append(float(np.mean(np.var(r, axis=0))))
        eenf_actual = max(regs) if regs else 0.0
        eenf_tau = float(self.tau_eenf or 0.0)
        eenf_pass = eenf_actual <= eenf_tau

        if len(self.contexts) > 1 and len(self.canon) > 0:
            sample = list(self.canon.keys())[: min(80, len(self.canon))]
            Xp = self.embedder.embed_many(sample, context=self.contexts[0])
            Xr = self.embedder.embed_many(sample, context=self.contexts[1])
            Wp = learn_pca_projection(Xp, k=64)
            Wr = orthogonalize_against(learn_pca_projection(Xr, k=64), Wp, iters=3)
            overlaps = [float(np.dot(project(Wp, self.embedder.embed(t, self.contexts[0])), project(Wr, self.embedder.embed(t, self.contexts[1])))) for t in sample]
            cmnf_actual = float(np.mean(overlaps)) if overlaps else 0.0
        else:
            cmnf_actual = 0.0
        cmnf_tau = float(self.tau_cmnf or 0.0)
        cmnf_pass = cmnf_actual <= cmnf_tau

        if self.drift_embedder and len(self.canon) > 0:
            sample = list(self.canon.keys())[: min(60, len(self.canon))]
            drifts = []
            for t in sample:
                a = self.embedder.embed(t, self.canon[t].context)
                b = self.drift_embedder.embed(t, self.canon[t].context)
                if self.drift_map is not None:
                    b = apply_linear_map(b, self.drift_map)
                drifts.append(l2_dist(a, b))
            dbnf_actual = max(drifts) if drifts else 0.0
        else:
            dbnf_actual = 0.0
        dbnf_tau = float(self.tau_dbnf or 0.0)
        dbnf_pass = True if not self.drift_embedder else (dbnf_actual <= dbnf_tau)

        merges = [e for e in self.lineage if e.get("action") == "merge_auto"]
        aanf_tau = float(self.tau_aanf or 0.0)
        min_merge_sim = min((float(e.get("sim", 0.0)) for e in merges), default=None)
        aanf_pass = (min_merge_sim is None) or (min_merge_sim >= aanf_tau)

        min_sig = min((int(e.get("ecnf", {}).get("signals", 0)) for e in merges), default=None)
        ecnf_expected = 3 if self.evidence_mode == "embed_only" else self.m_min_default
        ecnf_pass = (min_sig is None) or (min_sig >= ecnf_expected)

        nf_rows = [
            ["EENF", "max_var <= tau", f"{eenf_actual:.6f}", f"{eenf_tau:.6f}", "PASS" if eenf_pass else "FAIL"],
            ["AANF", "min_merge_sim >= tau", "NA" if min_merge_sim is None else f"{min_merge_sim:.3f}", f"{aanf_tau:.3f}", "PASS" if aanf_pass else "FAIL"],
            ["CMNF", "mean_overlap <= tau", f"{cmnf_actual:.6f}", f"{cmnf_tau:.3f}", "PASS" if cmnf_pass else "FAIL"],
            ["DBNF", "max_drift <= tau", f"{dbnf_actual:.6f}", f"{dbnf_tau:.3f}", "PASS" if dbnf_pass else "FAIL"],
            ["ECNF", "min_signals >= m_min", "NA" if min_sig is None else str(min_sig), str(ecnf_expected), "PASS" if ecnf_pass else "FAIL"],
            ["RRNF", "(not exercised)", "NA", "NA", "INFO"],
            ["PONF", "(not exercised)", "NA", "NA", "INFO"],
        ]
        logger.info("\n" + render_table(["NormalForm", "Rule", "Actual", "Expected", "Status"], nf_rows, title="SDNF VALIDATION SUMMARY"))

        merge_rows = []
        for src, st in sorted(self.source_stats.items()):
            merge_rows.append([src, st.get("merges", 0), st.get("pending", 0), st.get("promoted", 0), st.get("new", 0)])
        if merge_rows:
            logger.info("\n" + render_table(["Source", "Merges", "Pending", "Promoted", "New"], merge_rows, title="SRS INGEST SUMMARY (PER SOURCE)"))

        total_ev = self.value_evidence_available + self.value_evidence_missing
        frac_missing = (self.value_evidence_missing / total_ev) if total_ev else 0.0
        ev_rows = [["value_evidence_available", self.value_evidence_available], ["value_evidence_missing", self.value_evidence_missing], ["missing_fraction", pct(frac_missing)]]
        logger.info("\n" + render_table(["Metric", "Value"], ev_rows, title="EVIDENCE AVAILABILITY"))

        if not eenf_pass:
            logger.warning("EENF FAIL: instability > tau. Consider increasing regenerations G or stabilizing prompts/model.")
        if not cmnf_pass:
            logger.warning("CMNF FAIL: context overlap > tau. Consider increasing orthogonalization iters or refining context prompts.")
        if self.drift_embedder and (not dbnf_pass):
            logger.warning("DBNF FAIL: drift > tau. Consider forking schema versions for high-drift attributes.")
        if (self.evidence_mode in ("vss", "shape", "hybrid")) and frac_missing > 0.5:
            logger.warning("Value evidence missing for many fields. Ensure payload JSONs exist in payloads_dir.")


# -----------------------------
# Multi-run helper (evidence_mode=all)
# -----------------------------

def run_single(args, evidence_mode: str) -> Dict[str, Any]:
    exp = SDNFExperiment(
        data_dir=Path(args.data_dir),
        payloads_dir=Path(args.payloads_dir),
        model_name=args.model,
        drift_model_name=args.drift_model,
        contexts=args.contexts,
        evidence_mode=evidence_mode,
        seed=42,
        log_level=args.log_level,
        promote_sources=args.promote_sources,
        promote_hits=args.promote_hits,
    )

    exp.calibrate()
    exp.bootstrap_master()

    schema_files = [p for p in iter_json_files(Path(args.data_dir)) if p.name != "INAmex.json"]
    if args.max_schema_files:
        schema_files = schema_files[: args.max_schema_files]
    for p in schema_files:
        exp.ingest_schema_file(p)

    payload_files = iter_json_files(Path(args.payloads_dir))
    if args.max_payload_files:
        payload_files = payload_files[: args.max_payload_files]
    for p in payload_files:
        if args.max_fields is None:
            exp.ingest_payload_file(p)
        else:
            names, payload = exp.parse_payload_file(p)
            names = names[: args.max_fields]
            payload = {k: payload.get(k) for k in names}
            exp.ingest_names(names, payload=payload, source=p.name)

    # drift alignment if needed
    if exp.drift_embedder and exp.drift_map is None:
        anchor = list(exp.canon.keys())[: min(60, len(exp.canon))]
        if anchor:
            A = exp.embedder.embed_many(anchor, context=exp.contexts[0])
            B = exp.drift_embedder.embed_many(anchor, context=exp.contexts[0])
            if A.shape[1] != B.shape[1]:
                exp.drift_map = fit_linear_map(B, A)

    total_aliases = sum(len(v.aliases) for v in exp.canon.values())
    reduction = 1.0 - (len(exp.canon) / max(1, total_aliases))

    merges = len([e for e in exp.lineage if e.get("action") == "merge_auto"])
    pending = len([e for e in exp.lineage if e.get("action") == "merge_pending"])
    promoted = sum(st.get("promoted", 0) for st in exp.source_stats.values())

    total_ev = exp.value_evidence_available + exp.value_evidence_missing
    missing_frac = (exp.value_evidence_missing / total_ev) if total_ev else 0.0

    # EENF actual/tau
    regs = []
    for n in list(exp.canon.keys())[: min(40, len(exp.canon))]:
        r = exp.embedder.regenerations(n, context=exp.canon[n].context, G=10)
        regs.append(float(np.mean(np.var(r, axis=0))))
    eenf_actual = max(regs) if regs else 0.0

    # CMNF actual/tau
    if len(exp.contexts) > 1 and len(exp.canon) > 0:
        sample = list(exp.canon.keys())[: min(80, len(exp.canon))]
        Xp = exp.embedder.embed_many(sample, context=exp.contexts[0])
        Xr = exp.embedder.embed_many(sample, context=exp.contexts[1])
        Wp = learn_pca_projection(Xp, k=64)
        Wr = orthogonalize_against(learn_pca_projection(Xr, k=64), Wp, iters=3)
        overlaps = [float(np.dot(project(Wp, exp.embedder.embed(t, exp.contexts[0])), project(Wr, exp.embedder.embed(t, exp.contexts[1])))) for t in sample]
        cmnf_actual = float(np.mean(overlaps)) if overlaps else 0.0
    else:
        cmnf_actual = 0.0

    # DBNF actual/tau
    if exp.drift_embedder and len(exp.canon) > 0:
        sample = list(exp.canon.keys())[: min(60, len(exp.canon))]
        drifts = []
        for t in sample:
            a = exp.embedder.embed(t, exp.canon[t].context)
            b = exp.drift_embedder.embed(t, exp.canon[t].context)
            if exp.drift_map is not None:
                b = apply_linear_map(b, exp.drift_map)
            drifts.append(l2_dist(a, b))
        dbnf_actual = max(drifts) if drifts else 0.0
    else:
        dbnf_actual = None

    return {
        "mode": evidence_mode,
        "merges": merges,
        "pending": pending,
        "promoted": promoted,
        "canon": len(exp.canon),
        "aliases": total_aliases,
        "reduction": reduction,
        "aanf_tau": float(exp.tau_aanf or 0.0),
        "missingV": missing_frac,
        "EENF_actual": float(eenf_actual),
        "EENF_tau": float(exp.tau_eenf or 0.0),
        "CMNF_actual": float(cmnf_actual),
        "CMNF_tau": float(exp.tau_cmnf or 0.0),
        "DBNF_actual": (None if dbnf_actual is None else float(dbnf_actual)),
        "DBNF_tau": (None if not exp.drift_embedder else float(exp.tau_dbnf or 0.0)),
    }


def run_all_modes(args):
    modes = ["embed_only", "vss", "shape", "hybrid"]
    results = [run_single(args, m) for m in modes]

    rows = []
    for r in results:
        rows.append([
            r["mode"],
            r["merges"],
            r["pending"],
            r["promoted"],
            r["canon"],
            r["aliases"],
            pct(r["reduction"]),
            f"{r['aanf_tau']:.3f}",
            f"{r['EENF_actual']:.6f}/{r['EENF_tau']:.6f}",
            f"{r['CMNF_actual']:.6f}/{r['CMNF_tau']:.3f}",
            ("NA" if args.drift_model is None else f"{(r['DBNF_actual'] or 0.0):.6f}/{(r['DBNF_tau'] or 0.0):.3f}"),
            pct(r["missingV"]),
        ])

    logger.info("\n" + render_table(
        ["mode", "merges", "pending", "promoted", "canon", "aliases", "redundancy", "AANF_tau", "EENF a/t", "CMNF a/t", "DBNF a/t", "missingV"],
        rows,
        title="EVIDENCE MODE COMPARISON (all)"
    ))


def main():
    epilog = """When to use which option:
  --evidence_mode hybrid    : best default (name + value evidence; safest merges)
  --evidence_mode embed_only: baseline (fast; relies on name similarity)
  --evidence_mode vss       : value signature evidence (needs payload values)
  --evidence_mode shape     : shape-token evidence (needs payload values)
  --evidence_mode all       : runs all modes and prints comparison

Promotion knobs:
  --promote_sources N : auto-promote pending merges seen in N distinct sources
  --promote_hits N    : auto-promote after N repeated observations
"""

    ap = argparse.ArgumentParser(
        description="SDNF unified experiment with configurable evidence modes and consolidated reporting.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    ap.add_argument("--data_dir", type=str, default="data", help="Schema JSON directory (default: data)")
    ap.add_argument("--payloads_dir", type=str, default="payloads", help="Payload JSON directory (default: payloads)")

    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence-Transformer model for embeddings")
    ap.add_argument("--drift_model", type=str, default=None, help="Optional second model to simulate drift (DBNF)")

    ap.add_argument("--contexts", nargs="+", default=["Payments", "Risk"], help="Context labels (default: Payments Risk)")

    ap.add_argument(
        "--evidence_mode",
        type=str,
        default="hybrid",
        choices=["embed_only", "vss", "shape", "hybrid", "all"],
        help="Evidence mode (default: hybrid). Use 'all' to run all modes and compare."
    )

    ap.add_argument("--promote_sources", type=int, default=2, help="Promote pending merges after N distinct sources (default: 2)")
    ap.add_argument("--promote_hits", type=int, default=3, help="Promote pending merges after N total hits (default: 3)")

    ap.add_argument("--max_schema_files", type=int, default=None, help="Limit number of schema files ingested")
    ap.add_argument("--max_payload_files", type=int, default=None, help="Limit number of payload files ingested")
    ap.add_argument("--max_fields", type=int, default=None, help="Limit number of fields per payload ingested")

    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level for SDNF logs (default: INFO)")
    ap.add_argument("--quiet_external", action="store_true", default=True, help="Suppress noisy HuggingFace/HTTP INFO logs (default: on)")
    ap.add_argument("--show_external", action="store_true", help="Show external library INFO logs")

    args = ap.parse_args()

    setup_logging(args.log_level)
    suppress_external_logs(enable=(args.quiet_external and not args.show_external))

    if args.evidence_mode == "all":
        run_all_modes(args)
        return

    exp = SDNFExperiment(
        data_dir=Path(args.data_dir),
        payloads_dir=Path(args.payloads_dir),
        model_name=args.model,
        drift_model_name=args.drift_model,
        contexts=args.contexts,
        evidence_mode=args.evidence_mode,
        seed=42,
        log_level=args.log_level,
        promote_sources=args.promote_sources,
        promote_hits=args.promote_hits,
    )

    exp.calibrate()
    exp.bootstrap_master()

    schema_files = [p for p in iter_json_files(Path(args.data_dir)) if p.name != "INAmex.json"]
    if args.max_schema_files:
        schema_files = schema_files[: args.max_schema_files]
    for p in schema_files:
        exp.ingest_schema_file(p)

    payload_files = iter_json_files(Path(args.payloads_dir))
    if args.max_payload_files:
        payload_files = payload_files[: args.max_payload_files]
    for p in payload_files:
        if args.max_fields is None:
            exp.ingest_payload_file(p)
        else:
            names, payload = exp.parse_payload_file(p)
            names = names[: args.max_fields]
            payload = {k: payload.get(k) for k in names}
            exp.ingest_names(names, payload=payload, source=p.name)

    # DBNF alignment map if needed
    if exp.drift_embedder and exp.drift_map is None:
        anchor = list(exp.canon.keys())[: min(60, len(exp.canon))]
        if anchor:
            A = exp.embedder.embed_many(anchor, context=exp.contexts[0])
            B = exp.drift_embedder.embed_many(anchor, context=exp.contexts[0])
            if A.shape[1] != B.shape[1]:
                exp.drift_map = fit_linear_map(B, A)
                logger.warning("DBNF: drift_model dim (%d) != base dim (%d). Learned alignment map (post-ingest).", B.shape[1], A.shape[1])

    run_cfg = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "evidence_mode": args.evidence_mode,
        "model": args.model,
        "drift_model": args.drift_model,
        "contexts": " ".join(args.contexts),
        "promote_sources": args.promote_sources,
        "promote_hits": args.promote_hits,
        "schema_files_ingested": len(schema_files),
        "payload_files_ingested": len(payload_files),
        "payloads_dir": str(Path(args.payloads_dir)),
        "data_dir": str(Path(args.data_dir)),
    }

    exp.report(run_cfg)


if __name__ == "__main__":
    main()
