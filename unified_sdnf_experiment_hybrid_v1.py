
#!/usr/bin/env python3
"""unified_sdnf_experiment_hybrid_v1.py

Unified SDNF experiment (single file) with real embeddings + HNSW ANN and configurable evidence mode.

Key features:
- Attribute-name embeddings: Sentence-Transformers + Multi-level embedding (fine + abstract + contextual) then concat+normalize.
- ANN: HNSW (hnswlib) over canonical attribute embeddings.
- Evidence modes for ECNF gating:
    --evidence_mode embed_only   : embedding similarity only (baseline)
    --evidence_mode vss          : embedding similarity + Value Semantic Signature evidence
    --evidence_mode shape        : embedding similarity + shape-token embedding evidence
    --evidence_mode hybrid       : embedding similarity + VSS + shape-token embedding (default)
- Uses JSON files from ./data (schema-style with 'attributes' list, and payload-style flat dict).

Why VSS/shape?
- They provide domain-agnostic, learned validation signals to reduce false merges without hard-coded domain regex.
- They are treated as *evidence* for ECNF, not as primary matching logic.

Dependencies:
  pip install -U sentence-transformers hnswlib numpy

Examples:
  python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --contexts Payments Risk
  python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --evidence_mode embed_only
  python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --evidence_mode vss --log_level DEBUG

Optional drift test (DBNF) by comparing two models:
  python unified_sdnf_experiment_hybrid_v1.py --data_dir ./data --model all-MiniLM-L6-v2 --drift_model all-mpnet-base-v2

"""

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

logger = logging.getLogger("sdnf.unified")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )


# -----------------------------
# Basic utilities
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


# -----------------------------
# Domain-agnostic VSS (Value Semantic Signature)
# -----------------------------

VSS_DIM = 20


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    ent = 0.0
    for k, v in c.items():
        p = v / n
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def vss_from_value(v: Any) -> Optional[np.ndarray]:
    """Compute a compact, domain-agnostic signature vector from a single value.

    Returns None if value is missing.
    """
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

    # parse attempts (generic)
    is_float = 0.0
    is_int = 0.0
    try:
        float(s)
        is_float = 1.0
        if re.fullmatch(r"[-+]?\d+", s):
            is_int = 1.0
    except Exception:
        pass

    # loose date-like indicator (generic, not domain-specific)
    is_date_like = 1.0 if re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2}|\d{4})", s) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else 0.0

    uniq_ratio = len(set(s)) / max(1, L)
    ent = _shannon_entropy(s)

    vec = np.zeros((VSS_DIM,), dtype=np.float32)
    # Normalize scalar features into 0..1-ish ranges
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

    # A tiny hashed character-class sketch for extra discrimination (still generic)
    # classes: D digit, A alpha, S special
    classes = []
    for ch in s[:64]:
        if ch.isdigit():
            classes.append('D')
        elif ch.isalpha():
            classes.append('A')
        else:
            classes.append('S')
    sketch = "".join(classes)
    # count short patterns
    pats = ["DDDD", "DDDDDD", "AAAA", "SS", "DS", "SD", "DA", "AD"]
    for i, p in enumerate(pats):
        vec[14 + i] = 1.0 if p in sketch else 0.0

    return safe_norm(vec)


# -----------------------------
# Shape-token embedding (embedding-native evidence)
# -----------------------------

def shape_tokens(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None

    L = len(s)
    toks = [f"LEN_{min(L,64)}"]

    only_digits = s.isdigit()
    only_alpha = s.isalpha()
    only_alnum = s.isalnum()

    if only_digits:
        toks.append("ALL_DIGITS")
    if only_alpha:
        toks.append("ALL_ALPHA")
    if only_alnum and not only_digits and not only_alpha:
        toks.append("ALNUM")

    if "/" in s:
        toks.append("HAS_SLASH")
    if "-" in s:
        toks.append("HAS_DASH")
    if "." in s:
        toks.append("HAS_DOT")
    if "@" in s:
        toks.append("HAS_AT")

    # generic numeric hints
    if re.fullmatch(r"\d{15,16}", s):
        toks.append("LEN_15_16_DIGITS")
    if re.fullmatch(r"\d{3,4}", s):
        toks.append("LEN_3_4_DIGITS")
    if re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2}|\d{4})", s):
        toks.append("DATE_MM_YY")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        toks.append("DATE_ISO")

    # fraction of digit/alpha
    digits = sum(ch.isdigit() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    toks.append(f"DIGIT_FRAC_{int(10*digits/max(1,L))}")
    toks.append(f"ALPHA_FRAC_{int(10*alpha/max(1,L))}")

    return " ".join(toks)


# -----------------------------
# Multi-level embedding for attribute names (real embeddings)
# -----------------------------

class MultiLevelEmbedder:
    """Multi-level representation per paper using Sentence-Transformer backbone.

    fine: encode(name)
    abstract: encode(normalize_name(name))
    contextual: encode(f"{name} in {context} context")

    concat + normalize.
    """

    def __init__(self, st_model):
        self.model = st_model
        v = self.model.encode(["test"], normalize_embeddings=True)
        self.base_dim = int(v.shape[1])
        self.dim = self.base_dim * 3

    def embed_many(self, tokens: List[str], context: str) -> np.ndarray:
        fine = self.model.encode(tokens, normalize_embeddings=True)
        abstract = self.model.encode([normalize_name(t) for t in tokens], normalize_embeddings=True)
        contextual = self.model.encode([f"{t} in {context} context" for t in tokens], normalize_embeddings=True)
        X = np.concatenate([fine, abstract, contextual], axis=1).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def embed(self, token: str, context: str) -> np.ndarray:
        return self.embed_many([token], context=context)[0]

    def regenerations(self, token: str, context: str, G: int = 10) -> np.ndarray:
        # prompt variants to emulate regeneration jitter
        variants = [token, token + " ", token + ".", token + "!", f"{token} field"]
        variants = (variants * ((G + len(variants) - 1) // len(variants)))[:G]
        return self.embed_many(variants, context=context)


# -----------------------------
# Canonical attribute state
# -----------------------------

@dataclass
class CanonicalAttribute:
    name: str
    context: str
    embedding: np.ndarray
    aliases: List[str] = field(default_factory=list)

    # learned validator state (optional)
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
        self.model_name = model_name
        self.drift_model_name = drift_model_name
        self.contexts = contexts
        self.evidence_mode = evidence_mode
        self.seed = seed
        np.random.seed(seed)

        # dependencies
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("Missing dependency: sentence-transformers. Install: pip install sentence-transformers") from e
        try:
            import hnswlib
        except Exception as e:
            raise RuntimeError("Missing dependency: hnswlib. Install: pip install hnswlib") from e

        self.hnswlib = hnswlib
        self.st = SentenceTransformer(model_name)
        self.embedder = MultiLevelEmbedder(self.st)

        # For shape-token embeddings we reuse the same ST model; these are not raw values.
        self.shape_st = self.st
        self.shape_dim = int(self.shape_st.encode(["test"], normalize_embeddings=True).shape[1])

        # Optional drift model (DBNF)
        self.drift_st = SentenceTransformer(drift_model_name) if drift_model_name else None
        self.drift_embedder = MultiLevelEmbedder(self.drift_st) if self.drift_st else None

        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # thresholds (calibrated)
        self.tau_eenf = None
        self.tau_aanf = None
        self.tau_dbnf = None

        # ECNF decision policy (can be tuned)
        # distinct signal minimum (for embed_only, we relax)
        self.m_min_default = 3
        self.score_threshold = 0.60
        self.gamma = 0.72

        # state
        self.canon: Dict[str, CanonicalAttribute] = {}
        self.lineage: List[Dict[str, Any]] = []
        self.index = None
        self.id_to_name: Dict[int, str] = {}

        # diagnostics
        self.value_evidence_available = 0
        self.value_evidence_missing = 0

        self._mode_sanity_banner()

    def _mode_sanity_banner(self):
        if self.evidence_mode == "embed_only":
            logger.warning("evidence_mode=embed_only: this is a baseline; expect more false merges vs hybrid ECNF.")
        elif self.evidence_mode in ("vss", "shape", "hybrid"):
            logger.info("evidence_mode=%s: value-derived evidence will be used when values exist in payload JSON.", self.evidence_mode)
        else:
            raise ValueError("Invalid --evidence_mode. Use embed_only|vss|shape|hybrid")

    # ---------- parsing ----------
    def parse_file(self, path: Path) -> Tuple[List[str], Dict[str, Any]]:
        obj = load_json(path)
        if isinstance(obj, dict) and "attributes" in obj and isinstance(obj["attributes"], list):
            names = []
            for a in obj["attributes"]:
                if isinstance(a, dict) and a.get("name"):
                    names.append(str(a["name"]))
                    for al in a.get("aliases", []) or []:
                        names.append(str(al))
            return sorted(set(names)), obj
        if isinstance(obj, dict):
            return [k for k in obj.keys() if not str(k).startswith("_")], obj
        return [], obj

    def infer_context_for_file(self, filename: str) -> str:
        f = filename.lower()
        if any(x in f for x in ["risk", "fraud", "score"]):
            return self.contexts[1] if len(self.contexts) > 1 else self.contexts[0]
        return self.contexts[0]

    # ---------- calibration ----------
    def calibrate(self, eenf_G: int = 10):
        files = iter_json_files(self.data_dir)
        if not files:
            raise RuntimeError(f"No .json files found in {self.data_dir}")

        # build vocab
        vocab = []
        for p in files:
            names, _ = self.parse_file(p)
            vocab.extend(names)
        vocab = sorted(set(vocab))
        sample = vocab[: min(60, len(vocab))]

        # EENF
        vars_ = []
        for t in sample[: min(40, len(sample))]:
            regs = self.embedder.regenerations(t, context=self.contexts[0], G=eenf_G)
            vars_.append(float(np.mean(np.var(regs, axis=0))))
        self.tau_eenf = float(np.quantile(np.array(vars_), 0.95)) if vars_ else 0.0

        # AANF: similarity quantile among random pairs (conservative)
        rng = np.random.default_rng(self.seed)
        sims = []
        if len(sample) >= 2:
            for _ in range(min(250, len(sample) * 4)):
                a, b = rng.choice(sample, size=2, replace=False)
                ea = self.embedder.embed(a, context=self.contexts[0])
                eb = self.embedder.embed(b, context=self.contexts[0])
                sims.append(cosine_sim(ea, eb))
        self.tau_aanf = float(np.quantile(np.array(sims), 0.995)) if sims else 0.88
        self.tau_aanf = float(min(0.95, max(0.75, self.tau_aanf)))

        # DBNF: if drift model provided, calibrate distances
        if self.drift_embedder:
            drifts = []
            for t in sample[: min(40, len(sample))]:
                v1 = self.embedder.embed(t, context=self.contexts[0])
                v2 = self.drift_embedder.embed(t, context=self.contexts[0])
                drifts.append(l2_dist(v1, v2))
            self.tau_dbnf = float(np.quantile(np.array(drifts), 0.90)) if drifts else 0.25
            self.tau_dbnf = float(min(0.60, max(0.15, self.tau_dbnf)))
        else:
            self.tau_dbnf = 0.25

        logger.info(
            "Calibrated taus | EENF=%.6f | AANF=%.3f | DBNF=%.3f",
            self.tau_eenf,
            self.tau_aanf,
            self.tau_dbnf,
        )

    # ---------- HNSW ----------
    def rebuild_index(self):
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

        logger.info("Built HNSW | items=%d | dim=%d | M=%d | ef_search=%d", len(names), dim, self.hnsw_m, self.ef_search)

    # ---------- bootstrap ----------
    def bootstrap_master(self):
        files = iter_json_files(self.data_dir)
        inamex = self.data_dir / "INAmex.json"
        master_path = inamex if inamex.exists() else files[0]

        names, raw = self.parse_file(master_path)
        ctx = self.infer_context_for_file(master_path.name)

        canonical_names = []
        if isinstance(raw, dict) and "attributes" in raw and isinstance(raw["attributes"], list):
            for a in raw["attributes"]:
                if isinstance(a, dict) and a.get("name"):
                    canonical_names.append(str(a["name"]))
        else:
            canonical_names = names

        canonical_names = sorted(set(canonical_names))
        logger.info("Bootstrapping master from %s | canon=%d", master_path.name, len(canonical_names))

        embs = self.embedder.embed_many(canonical_names, context=ctx)
        for n, e in zip(canonical_names, embs):
            self.canon[n] = CanonicalAttribute(name=n, context=ctx, embedding=e, aliases=[n])
            self.lineage.append({"action": "create", "to": n, "source": master_path.name})

    # ---------- evidence computation ----------
    def _shape_embed(self, shape_str: str) -> np.ndarray:
        v = self.shape_st.encode([shape_str], normalize_embeddings=True)[0].astype(np.float32)
        return safe_norm(v)

    def build_evidence(self, derived_name: str, derived_value: Any, candidate: CanonicalAttribute, nn_sim: float) -> List[EvidenceItem]:
        ev: List[EvidenceItem] = []
        ev.append(EvidenceItem(type="nn", score=float(nn_sim)))

        if self.evidence_mode == "embed_only":
            return ev

        # Value-derived evidence availability
        if derived_value is None:
            self.value_evidence_missing += 1
            return ev
        self.value_evidence_available += 1

        if self.evidence_mode in ("vss", "hybrid"):
            vss = vss_from_value(derived_value)
            if vss is not None and candidate.vss_centroid is not None:
                sim = cosine_sim(vss, candidate.vss_centroid)
                ev.append(EvidenceItem(type="vss", score=float(sim)))

        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                shv = self._shape_embed(sh)
                if candidate.shape_centroid is not None:
                    sim = cosine_sim(shv, candidate.shape_centroid)
                    ev.append(EvidenceItem(type="shape", score=float(sim)))

        return ev

    def aggregate_score(self, evidence: List[EvidenceItem]) -> float:
        # weight nn strongest; vss/shape as confirmers
        if self.evidence_mode == "embed_only":
            weights = {"nn": 1.0}
        elif self.evidence_mode == "vss":
            weights = {"nn": 0.75, "vss": 0.25}
        elif self.evidence_mode == "shape":
            weights = {"nn": 0.75, "shape": 0.25}
        else:  # hybrid
            weights = {"nn": 0.60, "vss": 0.20, "shape": 0.20}

        total_w = 0.0
        acc = 0.0
        for it in evidence:
            w = weights.get(it.type, 0.0)
            total_w += w
            # cosine similarity in [-1,1], map to [0,1]
            s01 = max(0.0, min(1.0, (float(it.score) + 1.0) / 2.0))
            acc += w * s01
        return float(acc / total_w) if total_w > 0 else 0.0

    def ecnf_pass(self, evidence: List[EvidenceItem]) -> Tuple[bool, str, float, int]:
        score = self.aggregate_score(evidence)
        distinct = len(set(it.type for it in evidence))

        # adapt signal requirement by evidence mode
        m_min = 1 if self.evidence_mode == "embed_only" else self.m_min_default

        if distinct >= m_min and score >= self.score_threshold:
            return True, "count_score", score, distinct
        if score >= self.gamma:
            return True, "strong_score", score, distinct
        return False, "insufficient", score, distinct

    # ---------- update learned validator state ----------
    def update_validator_state(self, canonical: CanonicalAttribute, derived_value: Any):
        if derived_value is None:
            return

        # VSS state
        if self.evidence_mode in ("vss", "hybrid"):
            v = vss_from_value(derived_value)
            if v is not None:
                if canonical.vss_centroid is None:
                    canonical.vss_centroid = v
                    canonical.vss_count = 1
                else:
                    # EMA update
                    alpha = 1.0 / float(min(50, canonical.vss_count + 1))
                    canonical.vss_centroid = safe_norm((1 - alpha) * canonical.vss_centroid + alpha * v)
                    canonical.vss_count += 1

        # shape state
        if self.evidence_mode in ("shape", "hybrid"):
            sh = shape_tokens(derived_value)
            if sh is not None:
                sv = self._shape_embed(sh)
                if canonical.shape_centroid is None:
                    canonical.shape_centroid = sv
                    canonical.shape_count = 1
                else:
                    alpha = 1.0 / float(min(50, canonical.shape_count + 1))
                    canonical.shape_centroid = safe_norm((1 - alpha) * canonical.shape_centroid + alpha * sv)
                    canonical.shape_count += 1

    # ---------- ingestion ----------
    def ingest_file(self, path: Path, max_fields: Optional[int] = None):
        names, raw = self.parse_file(path)
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

            labels, dists = self.index.knn_query(emb.reshape(1, -1), k=min(5, len(self.canon)))
            best_id = int(labels[0][0])
            best_sim = float(1.0 - float(dists[0][0]))
            best_name = self.id_to_name[best_id]
            candidate = self.canon[best_name]

            derived_value = payload.get(derived_name, None)
            evidence = self.build_evidence(derived_name, derived_value, candidate, best_sim)
            ok, reason, score, signals = self.ecnf_pass(evidence)

            if best_sim >= self.tau_aanf and ok:
                # merge
                if derived_name not in candidate.aliases:
                    candidate.aliases.append(derived_name)
                candidate.embedding = safe_norm(candidate.embedding + emb)
                self.update_validator_state(candidate, derived_value)

                self.lineage.append({
                    "action": "merge_auto",
                    "from": derived_name,
                    "to": best_name,
                    "source": path.name,
                    "sim": round(best_sim, 3),
                    "ecnf": {"pass": True, "reason": reason, "score": round(score, 3), "signals": signals, "mode": self.evidence_mode},
                })
                merges += 1

            elif best_sim >= self.tau_aanf and not ok:
                # pending
                self.lineage.append({
                    "action": "merge_pending",
                    "from": derived_name,
                    "to": best_name,
                    "source": path.name,
                    "sim": round(best_sim, 3),
                    "ecnf": {"pass": False, "reason": reason, "score": round(score, 3), "signals": signals, "mode": self.evidence_mode},
                })
                pendings += 1

                # Keep as canonical (so the system can proceed)
                self.canon[derived_name] = CanonicalAttribute(name=derived_name, context=ctx, embedding=emb, aliases=[derived_name])
                creates += 1

            else:
                # new canonical
                self.canon[derived_name] = CanonicalAttribute(name=derived_name, context=ctx, embedding=emb, aliases=[derived_name])
                self.lineage.append({"action": "create", "to": derived_name, "source": path.name})
                creates += 1

        if creates > 0:
            self.rebuild_index()

        logger.info(
            "Ingested %-18s | ctx=%-8s | fields=%3d | merges=%2d | pending=%2d | new=%2d",
            path.name,
            ctx,
            len(names),
            merges,
            pendings,
            creates,
        )

    # ---------- drift test ----------
    def run_drift_test(self, sample_k: int = 25):
        if not self.drift_embedder:
            logger.info("DBNF drift test skipped (no --drift_model provided)")
            return 0

        names = list(self.canon.keys())[: min(sample_k, len(self.canon))]
        forks = 0
        for n in names:
            v1 = self.embedder.embed(n, context=self.canon[n].context)
            v2 = self.drift_embedder.embed(n, context=self.canon[n].context)
            d = l2_dist(v1, v2)
            if d > self.tau_dbnf:
                vname = f"{n}_v2"
                if vname not in self.canon:
                    self.canon[vname] = CanonicalAttribute(name=vname, context=self.canon[n].context, embedding=v2, aliases=[n])
                    self.lineage.append({"action": "fork_auto", "from": n, "to": vname, "drift": round(float(d), 3)})
                    forks += 1

        if forks:
            self.rebuild_index()
        logger.info("DBNF drift test | checked=%d | forks=%d | tau=%.3f", len(names), forks, self.tau_dbnf)
        return forks

    # ---------- debug guidance ----------
    def debug_guidance(self):
        total = self.value_evidence_available + self.value_evidence_missing
        if self.evidence_mode in ("vss", "shape", "hybrid") and total > 0:
            frac_missing = self.value_evidence_missing / max(1, total)
            if frac_missing > 0.60:
                logger.warning(
                    "Value-evidence missing for %.0f%% of evaluated fields (many schema-only files). "
                    "If results look unstable, consider --evidence_mode embed_only for a baseline or ingest more payload-style JSONs.",
                    100.0 * frac_missing,
                )
            elif frac_missing > 0.30:
                logger.info(
                    "Value-evidence missing for %.0f%% of evaluated fields. Hybrid/VSS/Shape will still work, but evidence strength varies by file.",
                    100.0 * frac_missing,
                )

    # ---------- summary ----------
    def summary(self):
        merges = sum(1 for e in self.lineage if e.get("action") == "merge_auto")
        pendings = sum(1 for e in self.lineage if e.get("action") == "merge_pending")
        forks = sum(1 for e in self.lineage if e.get("action") == "fork_auto")

        total_aliases = sum(len(v.aliases) for v in self.canon.values())
        total_canon = len(self.canon)
        reduction = 1.0 - (total_canon / max(1, total_aliases))

        logger.info(
            "Summary | mode=%s | canon=%d | aliases_total=%d | redundancy_reduction=%.1f%% | merges=%d | pending=%d | forks=%d",
            self.evidence_mode,
            total_canon,
            total_aliases,
            100.0 * reduction,
            merges,
            pendings,
            forks,
        )

        # show up to 5 merge examples
        examples = [e for e in self.lineage if e.get("action") == "merge_auto"][:5]
        if examples:
            logger.info("Example merges (up to 5):")
            for e in examples:
                logger.info("  %s -> %s (sim=%s, score=%s, signals=%s)", e.get("from"), e.get("to"), e.get("sim"), e.get("ecnf", {}).get("score"), e.get("ecnf", {}).get("signals"))

    def run(self, max_files: Optional[int] = None, max_fields: Optional[int] = None):
        self.calibrate()
        self.bootstrap_master()
        self.rebuild_index()

        files = iter_json_files(self.data_dir)
        if max_files:
            files = files[:max_files]

        for p in files:
            if p.name == "INAmex.json":
                continue
            self.ingest_file(p, max_fields=max_fields)

        self.run_drift_test()
        self.debug_guidance()
        self.summary()


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
