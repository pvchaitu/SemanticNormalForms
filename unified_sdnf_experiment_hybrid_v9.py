#!/usr/bin/env python3
"""
unified_sdnf_experiment_hybrid_v9.py

Reviewer-grade SDNF experiment/audit harness for the paper:
"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces".

Design goals
------------
1. Never hardcode claimed paper metrics.
2. Compute claim-support metrics from explicit data, predicted lineage, optional ground truth,
   pairwise evidence, and timing instrumentation.
3. Print reviewer-facing tables that either confirm, falsify, or mark claims as NOT MEASURABLE.

Supported examples
------------------
python unified_sdnf_experiment_hybrid_v9.py \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases.json \
  --drift_ground_truth drift_ground_truth.json \
  --trace_pair acct_num PrimaryAccountNumber \
  --eenf_g_sweep 1,10,20 \
  --measure_timing

python unified_sdnf_experiment_hybrid_v9.py \
  --evidence_mode hybrid \
  --ground_truth_aliases ground_truth_aliases.json \
  --trace_pair acct_num PrimaryAccountNumber

python unified_sdnf_experiment_hybrid_v9.py \
  --evidence_mode all \
  --drift_model all-mpnet-base-v2 \
  --drift_ground_truth drift_ground_truth.json

Ground-truth alias formats supported
------------------------------------
A. Alias groups:
{
  "alias_groups": [
    ["acct_num", "PrimaryAccountNumber", "pan"],
    ["txn_amount", "amount"]
  ],
  "negative_pairs": [["card", "playing_card"]]
}

B. True pairs:
{
  "true_pairs": [["acct_num", "PrimaryAccountNumber"]],
  "negative_pairs": [["account", "merchant_account"]]
}

Drift ground-truth formats supported
------------------------------------
{
  "drift_attributes": ["description", "iso_currency_code", "acct_num", "payer_name"]
}

Notes
-----
- If sentence-transformers is installed, the script uses the requested model. Otherwise it falls back
  to deterministic hashing embeddings and clearly prints that fallback in RUN CONFIGURATION.
- If hnswlib is unavailable, the script uses exhaustive candidate-pair scoring.
- This is intentionally explicit and audit-oriented rather than optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# -----------------------------
# Constants / paper claims
# -----------------------------

DEFAULT_SEED = 42
PAPER_CLAIMS = {
    "schema_files_ingested": 7,
    "payload_files_ingested_current_output": 40,
    "input_attributes": 80,
    "canon_final": 49,
    "schema_reduction_pct": 38.7,
    "sdnf_precision_pct": 95.0,
    "sdnf_recall_pct": 90.0,
    "sdnf_leakage_pct": 2.0,
    "baseline_precision_pct": 86.0,
    "baseline_recall_pct": 95.0,
    "baseline_leakage_pct": 9.0,
    "g10_variance_reduction_pct": 40.0,
    "g20_variance_reduction_pct": 70.0,
    "avg_merge_decision_ms": 50.0,
}

# Paper tolerance rules.
TOLERANCES = {
    "schema_reduction_pct": 0.2,
    "precision_pct": 1.0,
    "recall_pct": 1.0,
    "leakage_pct": 0.5,
}

# Evidence weights from the paper appendix. Missing evidence is re-normalized.
DEFAULT_WEIGHTS = {
    "embedding": 0.4,
    "name": 0.2,
    "ontology": 0.1,
    "shape": 0.1,
    "vss": 0.2,
}

# -----------------------------
# Dataclasses
# -----------------------------

@dataclass(frozen=True, order=True)
class Pair:
    """Normalized unordered pair used for ground-truth and predictions."""
    a: str
    b: str

    @staticmethod
    def make(x: str, y: str) -> "Pair":
        nx, ny = normalize_key(x), normalize_key(y)
        if nx <= ny:
            return Pair(nx, ny)
        return Pair(ny, nx)

    def display(self) -> str:
        return f"{self.a} <-> {self.b}"


@dataclass
class AttributeRecord:
    name: str
    source: str
    context: str
    path: str
    values: List[Any] = field(default_factory=list)
    ontology_root: Optional[str] = None
    regex: str = ""
    canonical: str = ""
    embedding: Optional[np.ndarray] = None
    vss: Optional[np.ndarray] = None
    shape: str = ""

    @property
    def key(self) -> str:
        # Names are normalized for evaluation. Path/source are retained for audit traces.
        return normalize_key(self.name)


@dataclass
class PairEvidence:
    attr_a: str
    attr_b: str
    source_a: str = ""
    source_b: str = ""
    context_a: str = ""
    context_b: str = ""
    cosine_similarity: Optional[float] = None
    name_similarity: Optional[float] = None
    ontology_root_a: Optional[str] = None
    ontology_root_b: Optional[str] = None
    ontology_match: Optional[bool] = None
    value_cooccurrence: Optional[float] = None
    regex_a: str = ""
    regex_b: str = ""
    regex_match: Optional[bool] = None
    vss_similarity: Optional[float] = None
    shape_similarity: Optional[float] = None
    aggregate_score: Optional[float] = None
    evidence_signal_count: int = 0
    aanf_status: str = "NA"
    ecnf_status: str = "NA"
    cmnf_status: str = "NA"
    final_decision: str = "DEFER"
    lineage_id: str = ""
    reason: str = ""


@dataclass
class TimingRecord:
    candidate_generation_ms: float = 0.0
    evidence_scoring_ms: float = 0.0
    sdnf_validation_ms: float = 0.0
    total_decision_ms: float = 0.0


@dataclass
class MergeDecision:
    pair: Pair
    evidence: PairEvidence
    timing: TimingRecord
    accepted: bool
    mode: str


@dataclass
class ModeResult:
    mode: str
    attributes: List[AttributeRecord]
    decisions: List[MergeDecision]
    predicted_pairs: Set[Pair]
    canon_final: int
    input_attributes: int
    schema_reduction_pct: float
    nf_metrics: Dict[str, Any]
    drift_hotspots: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class AliasEvalResult:
    mode: str
    predicted_pairs: int
    true_pairs: int
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    false_positive_examples: List[str] = field(default_factory=list)
    false_negative_examples: List[str] = field(default_factory=list)
    measurable: bool = True


@dataclass
class LeakageEvalResult:
    mode: str
    leakage_count: Optional[int]
    predicted_merge_count: int
    leakage_rate: Optional[float]
    examples: List[str] = field(default_factory=list)
    measurable: bool = True


@dataclass
class DriftEvalResult:
    mode: str
    drift_tau: float
    detected_count: int
    true_drift_count: int
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    accuracy_if_defined: Optional[float]
    measurable: bool = True


# -----------------------------
# Formatting helpers
# -----------------------------

def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.{nd}f}"
    return str(x)


def pct(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.{nd}f}%"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> str:
    str_rows = [[fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    head = "|" + "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))) + "|"
    out = []
    if title:
        out.append("\n" + title)
    out.append(sep)
    out.append(head)
    out.append(sep.replace("-", "="))
    for row in str_rows:
        out.append("|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(sep)
    return "\n".join(out)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> None:
    print(render_table(headers, rows, title))


# -----------------------------
# Normalization / similarity
# -----------------------------

def camel_to_tokens(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split())


def normalize_key(s: str) -> str:
    return camel_to_tokens(str(s)).lower().strip()


def token_set(s: str) -> Set[str]:
    return set(normalize_key(s).split())


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def safe_norm(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)


def ontology_root(name: str) -> Optional[str]:
    n = normalize_key(name)
    if any(k in n for k in ["pan", "account", "acct", "iban", "routing", "card number", "account number", "primary account number"]):
        return "payment:account"
    if any(k in n for k in ["cvv", "cid", "security code", "verification", "card verification value"]):
        return "payment:cvv"
    if any(k in n for k in ["exp", "expiry", "expiration", "expiration date"]):
        return "payment:expiry"
    if any(k in n for k in ["amount", "amt", "transaction amount", "instd amt", "txn amount"]):
        return "payment:amount"
    if any(k in n for k in ["risk", "fraud", "score"]):
        return "risk:score"
    if any(k in n for k in ["currency", "iso currency", "ccy"]):
        return "payment:currency"
    if any(k in n for k in ["description", "comment", "memo", "note"]):
        return "text:description"
    if any(k in n for k in ["name", "payer", "payee", "merchant", "holder"]):
        return "party:name"
    return None


def infer_regex(values: Sequence[Any]) -> str:
    samples = [str(v) for v in values if v is not None and str(v) != ""]
    if not samples:
        return ""
    if all(re.fullmatch(r"\d{13,19}", s) for s in samples):
        return r"^[0-9]{13,19}$"
    if all(re.fullmatch(r"\d+", s) for s in samples):
        lengths = sorted({len(s) for s in samples})
        if len(lengths) == 1:
            return rf"^[0-9]{{{lengths[0]}}}$"
        return r"^[0-9]+$"
    if all(re.fullmatch(r"[A-Z]{3}", s) for s in samples):
        return r"^[A-Z]{3}$"
    if all(re.fullmatch(r"\d+(\.\d+)?", s) for s in samples):
        return r"^[0-9]+(\.[0-9]+)?$"
    return "mixed"


def shape_token(v: Any) -> str:
    s = str(v)
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("D")
        elif ch.isalpha():
            out.append("A")
        elif ch.isspace():
            out.append("S")
        else:
            out.append("P")
    # compact runs to avoid extremely long shapes
    compact = []
    for k, group in itertools.groupby(out):
        compact.append(k + str(len(list(group))))
    return "".join(compact)


def value_shape_signature(values: Sequence[Any]) -> str:
    if not values:
        return ""
    counts = defaultdict(int)
    for v in values[:100]:
        counts[shape_token(v)] += 1
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))


def vss_from_values(values: Sequence[Any]) -> Optional[np.ndarray]:
    samples = [str(v) for v in values if v is not None and str(v) != ""][:200]
    if not samples:
        return None
    lengths = np.array([len(s) for s in samples], dtype=np.float32)
    digit_frac = np.array([sum(ch.isdigit() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    alpha_frac = np.array([sum(ch.isalpha() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    punct_frac = np.array([sum(not ch.isalnum() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    numeric = np.array([1.0 if re.fullmatch(r"\d+(\.\d+)?", s) else 0.0 for s in samples], dtype=np.float32)
    vec = np.array([
        float(np.mean(lengths)), float(np.std(lengths)), float(np.min(lengths)), float(np.max(lengths)),
        float(np.mean(digit_frac)), float(np.mean(alpha_frac)), float(np.mean(punct_frac)), float(np.mean(numeric)),
        float(len(set(samples)) / max(1, len(samples)))
    ], dtype=np.float32)
    return safe_norm(vec)


# -----------------------------
# Embedding provider
# -----------------------------

class EmbeddingProvider:
    def __init__(self, model_name: str, seed: int, dim: int = 384):
        self.model_name = model_name
        self.seed = seed
        self.dim = dim
        self.backend = "hashing-fallback"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name)
            probe = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
            self.dim = int(probe.shape[1])
            self.backend = "sentence-transformers"
        except Exception:
            self.model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            arr = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)
        return np.vstack([self._hash_embed(t) for t in texts]).astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        # Deterministic char/token hashing fallback for reproducible offline audits.
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = normalize_key(text).split()
        if not toks:
            toks = [normalize_key(text)]
        for tok in toks:
            digest = hashlib.sha256((tok + str(self.seed)).encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i:i+2], "little") % self.dim
                sign = 1.0 if digest[i] % 2 == 0 else -1.0
                vec[idx] += sign
        return safe_norm(vec)

    def regenerations(self, text: str, context: str, G: int) -> np.ndarray:
        base = self.encode([f"{text} | context={context}"])[0]
        regs = []
        for g in range(G):
            rng = np.random.default_rng(stable_int(f"{text}|{context}|{self.seed}|{g}"))
            # Small deterministic perturbation approximates repeated encoder nondeterminism.
            noise = rng.normal(0.0, 0.003, size=base.shape).astype(np.float32)
            regs.append(safe_norm(base + noise))
        return np.vstack(regs)


def stable_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little", signed=False)


# -----------------------------
# JSON data extraction
# -----------------------------

def iter_json_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.json") if p.is_file()], key=lambda p: p.name.lower())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def walk_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from walk_json(v, path)
            else:
                yield path, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                yield from walk_json(v, path)
            else:
                yield path, v


def context_from_source(path: Path, default_context: str) -> str:
    # Keep conservative: current paper experiment is payments; source filename retained separately.
    return default_context


def collect_attributes(data_dir: Path, payloads_dir: Path, context: str, embedder: EmbeddingProvider) -> Tuple[List[AttributeRecord], Dict[str, Any]]:
    schema_files = iter_json_files(data_dir)
    payload_files = iter_json_files(payloads_dir)
    by_key: Dict[Tuple[str, str], AttributeRecord] = {}

    # Schema files contribute field names.
    for fp in schema_files:
        try:
            obj = load_json(fp)
        except Exception as e:
            print(f"WARNING: failed to read schema file {fp}: {e}")
            continue
        fields = list(walk_json(obj))
        # If a schema is a flat field-list, fallback to top-level keys/strings.
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    fields.append((item, None))
                elif isinstance(item, dict):
                    for candidate in ["name", "field", "attribute", "key"]:
                        if candidate in item:
                            fields.append((str(item[candidate]), None))
        for path, value in fields:
            field_name = path.split(".")[-1]
            key = (normalize_key(field_name), fp.name)
            if key not in by_key:
                by_key[key] = AttributeRecord(
                    name=field_name,
                    source=fp.name,
                    context=context_from_source(fp, context),
                    path=path,
                )
            if value not in (None, ""):
                by_key[key].values.append(value)

    # Payloads contribute field names and value evidence.
    for fp in payload_files:
        try:
            obj = load_json(fp)
        except Exception as e:
            print(f"WARNING: failed to read payload file {fp}: {e}")
            continue
        for path, value in walk_json(obj):
            field_name = path.split(".")[-1]
            key = (normalize_key(field_name), fp.name)
            if key not in by_key:
                by_key[key] = AttributeRecord(
                    name=field_name,
                    source=fp.name,
                    context=context_from_source(fp, context),
                    path=path,
                )
            by_key[key].values.append(value)

    attrs = list(by_key.values())
    attrs.sort(key=lambda a: (normalize_key(a.name), a.source.lower(), a.path.lower()))

    # Fill metadata.
    texts = [f"{a.name} | context={a.context}" for a in attrs]
    embs = embedder.encode(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
    for a, e in zip(attrs, embs):
        a.ontology_root = ontology_root(a.name)
        a.regex = infer_regex(a.values)
        a.shape = value_shape_signature(a.values)
        a.vss = vss_from_values(a.values)
        a.embedding = e
        a.canonical = a.key

    summary = {
        "schema_files_ingested": len(schema_files),
        "payload_files_ingested": len(payload_files),
        "raw_attribute_records": len(attrs),
        "distinct_attribute_names": len({a.key for a in attrs}),
        "value_evidence_available": sum(len(a.values) for a in attrs),
        "value_evidence_missing": sum(1 for a in attrs if not a.values),
    }
    total_value_slots = summary["value_evidence_available"] + summary["value_evidence_missing"]
    summary["missing_fraction"] = (summary["value_evidence_missing"] / total_value_slots) if total_value_slots else None
    return attrs, summary


# -----------------------------
# Ground truth loading / evaluation
# -----------------------------

def pairs_from_alias_groups(groups: Sequence[Sequence[str]]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for group in groups:
        normed = sorted({normalize_key(x) for x in group if str(x).strip()})
        for a, b in itertools.combinations(normed, 2):
            pairs.add(Pair.make(a, b))
    return pairs


def load_ground_truth_aliases(path: Optional[str]) -> Tuple[Optional[Set[Pair]], Set[Pair]]:
    if not path:
        return None, set()
    p = Path(path)
    if not p.exists():
        print(f"WARNING: ground truth alias file not found: {p}. Alias precision/recall marked NOT MEASURABLE.")
        return None, set()
    data = load_json(p)
    true_pairs: Set[Pair] = set()
    negative_pairs: Set[Pair] = set()
    if "alias_groups" in data:
        true_pairs |= pairs_from_alias_groups(data.get("alias_groups", []))
    if "true_pairs" in data:
        for a, b in data.get("true_pairs", []):
            true_pairs.add(Pair.make(a, b))
    if "negative_pairs" in data:
        for a, b in data.get("negative_pairs", []):
            negative_pairs.add(Pair.make(a, b))
    if not true_pairs:
        print(f"WARNING: no true alias pairs found in {p}. Alias precision/recall marked NOT MEASURABLE.")
        return None, negative_pairs
    return true_pairs, negative_pairs


def evaluate_alias_merges(mode: str, predicted: Set[Pair], true_pairs: Optional[Set[Pair]]) -> AliasEvalResult:
    if true_pairs is None:
        return AliasEvalResult(mode, len(predicted), 0, 0, 0, 0, None, None, None, measurable=False)
    tp_set = predicted & true_pairs
    fp_set = predicted - true_pairs
    fn_set = true_pairs - predicted
    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return AliasEvalResult(
        mode=mode,
        predicted_pairs=len(predicted),
        true_pairs=len(true_pairs),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_examples=[p.display() for p in sorted(fp_set)[:10]],
        false_negative_examples=[p.display() for p in sorted(fn_set)[:10]],
        measurable=True,
    )


# -----------------------------
# Evidence scoring and mode rules
# -----------------------------

def aggregate_score(scores: Dict[str, Optional[float]], weights: Dict[str, float] = DEFAULT_WEIGHTS) -> Optional[float]:
    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return None
    total_w = sum(weights.get(k, 0.0) for k in available)
    if total_w <= 0:
        return None
    return sum((weights.get(k, 0.0) / total_w) * float(v) for k, v in available.items())


def value_cooccurrence(a: AttributeRecord, b: AttributeRecord) -> Optional[float]:
    # Conservative file/source-aware approximation: if values exist for both, compare non-empty counts.
    if not a.values or not b.values:
        return None
    n = max(len(a.values), len(b.values))
    if n == 0:
        return None
    return min(len(a.values), len(b.values)) / n


def compute_pair_evidence(a: AttributeRecord, b: AttributeRecord, mode: str, tau_aanf: float, gamma: float, m_min: int) -> PairEvidence:
    cos_sim = cosine(a.embedding, b.embedding)
    name_sim = jaccard(token_set(a.name), token_set(b.name))
    ont_match = bool(a.ontology_root and b.ontology_root and a.ontology_root == b.ontology_root)
    cooc = value_cooccurrence(a, b)
    regex_match = bool(a.regex and b.regex and a.regex == b.regex)
    vss_sim = cosine(a.vss, b.vss)
    shape_sim = 1.0 if (a.shape and b.shape and a.shape == b.shape) else (jaccard(set(a.shape.split(";")), set(b.shape.split(";"))) if a.shape and b.shape else None)

    # Evidence availability count. This maps directly to ECNF EvidenceSet >= m_min.
    signal_scores = {
        "embedding": cos_sim,
        "name": name_sim,
        "ontology": 1.0 if ont_match else (0.0 if a.ontology_root or b.ontology_root else None),
        "shape": shape_sim,
        "vss": vss_sim,
    }

    # Mode-specific evidence disabling.
    if mode in {"embed_only", "embed_only_baseline"}:
        signal_scores = {"embedding": cos_sim}
    elif mode == "vss_only":
        signal_scores = {"vss": vss_sim}
    elif mode == "shape_only":
        signal_scores = {"shape": shape_sim}
    elif mode == "name_ontology_only":
        signal_scores = {"name": name_sim, "ontology": 1.0 if ont_match else 0.0}
    elif mode == "no_value_evidence":
        signal_scores = {"embedding": cos_sim, "name": name_sim, "ontology": 1.0 if ont_match else 0.0}

    agg = aggregate_score(signal_scores)
    signal_count = sum(1 for v in signal_scores.values() if v is not None and v > 0.0)

    aanf = "PASS" if cos_sim is not None and cos_sim >= tau_aanf else "FAIL"
    ecnf = "PASS" if (signal_count >= m_min and agg is not None and agg >= gamma) else "FAIL"
    cmnf = "PASS"
    # In this payments-only setup, incompatible known ontology roots are treated as a CMNF/context contamination risk.
    if a.context != b.context:
        cmnf = "FAIL"
    if a.ontology_root and b.ontology_root and a.ontology_root != b.ontology_root:
        cmnf = "FAIL"

    return PairEvidence(
        attr_a=a.name,
        attr_b=b.name,
        source_a=a.source,
        source_b=b.source,
        context_a=a.context,
        context_b=b.context,
        cosine_similarity=cos_sim,
        name_similarity=name_sim,
        ontology_root_a=a.ontology_root,
        ontology_root_b=b.ontology_root,
        ontology_match=ont_match,
        value_cooccurrence=cooc,
        regex_a=a.regex,
        regex_b=b.regex,
        regex_match=regex_match,
        vss_similarity=vss_sim,
        shape_similarity=shape_sim,
        aggregate_score=agg,
        evidence_signal_count=signal_count,
        aanf_status=aanf,
        ecnf_status=ecnf,
        cmnf_status=cmnf,
    )


def decision_for_mode(e: PairEvidence, mode: str, gamma: float, m_min: int) -> Tuple[bool, str]:
    # Baseline: only embedding threshold; no ECNF/CMNF/ontology/value gates.
    if mode in {"embed_only", "embed_only_baseline"}:
        return e.aanf_status == "PASS", "embedding cosine threshold only"

    # Ablations.
    if mode == "no_ecnf":
        return e.aanf_status == "PASS" and e.cmnf_status == "PASS", "AANF+CMNF; ECNF disabled"
    if mode == "no_cmnf":
        return e.aanf_status == "PASS" and e.ecnf_status == "PASS", "AANF+ECNF; CMNF disabled"
    if mode == "no_dbnf":
        return e.aanf_status == "PASS" and e.ecnf_status == "PASS" and e.cmnf_status == "PASS", "DBNF disabled; merge rules unchanged"
    if mode in {"vss_only", "shape_only", "name_ontology_only", "no_value_evidence"}:
        return e.ecnf_status == "PASS" and e.cmnf_status == "PASS", f"{mode} evidence rule"

    # SDNF hybrid / hybrid: AANF + ECNF + CMNF gates.
    return e.aanf_status == "PASS" and e.ecnf_status == "PASS" and e.cmnf_status == "PASS", "AANF+ECNF+CMNF hybrid rule"


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra <= rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


# -----------------------------
# Mode run / metrics
# -----------------------------

def mode_list(requested: str) -> List[str]:
    if requested == "all":
        return [
            "embed_only_baseline", "sdnf_hybrid", "no_ecnf", "no_cmnf", "no_dbnf",
            "no_value_evidence", "vss_only", "shape_only", "name_ontology_only", "hybrid"
        ]
    # Backward compatibility aliases.
    if requested == "embed_only":
        return ["embed_only_baseline"]
    return [requested]


def run_mode(attrs: List[AttributeRecord], mode: str, args: argparse.Namespace, embedder: EmbeddingProvider) -> ModeResult:
    tau_aanf = args.tau_aanf
    gamma = args.gamma
    m_min = args.m_min
    t0 = time.perf_counter()
    pairs = list(itertools.combinations(attrs, 2))
    t_candidate_ms = (time.perf_counter() - t0) * 1000.0

    decisions: List[MergeDecision] = []
    predicted: Set[Pair] = set()
    uf = UnionFind([a.key for a in attrs])

    for idx, (a, b) in enumerate(pairs):
        start = time.perf_counter()
        # Candidate generation time is amortized over all candidate pairs.
        cand_ms = t_candidate_ms / max(1, len(pairs))

        ev_start = time.perf_counter()
        evidence = compute_pair_evidence(a, b, mode, tau_aanf, gamma, m_min)
        evidence_ms = (time.perf_counter() - ev_start) * 1000.0

        val_start = time.perf_counter()
        accepted, reason = decision_for_mode(evidence, mode, gamma, m_min)
        validation_ms = (time.perf_counter() - val_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0 + cand_ms

        pair = Pair.make(a.name, b.name)
        evidence.final_decision = "MERGE" if accepted else "DEFER"
        evidence.reason = reason
        evidence.lineage_id = f"{mode}-{idx:06d}"
        timing = TimingRecord(cand_ms, evidence_ms, validation_ms, total_ms)
        decisions.append(MergeDecision(pair, evidence, timing, accepted, mode))
        if accepted:
            predicted.add(pair)
            uf.union(a.key, b.key)

    # Assign canonical groups.
    for a in attrs:
        a.canonical = uf.find(a.key)
    canon_final = len({uf.find(a.key) for a in attrs})
    input_attributes = len({a.key for a in attrs})
    schema_reduction = ((input_attributes - canon_final) / input_attributes * 100.0) if input_attributes else 0.0

    nf_metrics = compute_nf_metrics(attrs, decisions, args, embedder)
    drift_hotspots = []
    if args.drift_model:
        drift_hotspots = compute_drift_hotspots(attrs, args, embedder)
        nf_metrics["DBNF_max"] = max([d for _, d in drift_hotspots], default=0.0)
        nf_metrics["DBNF_tau"] = args.tau_dbnf_drift if args.drift_model else args.tau_dbnf
        nf_metrics["DBNF_status"] = "PASS" if nf_metrics["DBNF_max"] <= nf_metrics["DBNF_tau"] else "FAIL"

    return ModeResult(mode, attrs, decisions, predicted, canon_final, input_attributes, schema_reduction, nf_metrics, drift_hotspots)


def compute_nf_metrics(attrs: List[AttributeRecord], decisions: List[MergeDecision], args: argparse.Namespace, embedder: EmbeddingProvider) -> Dict[str, Any]:
    regs = []
    for a in attrs[: min(60, len(attrs))]:
        r = embedder.regenerations(a.name, a.context, G=10)
        regs.append(float(np.mean(np.var(r, axis=0))))
    eenf_q95 = float(np.quantile(np.array(regs), 0.95)) if regs else 0.0
    eenf_max = max(regs) if regs else 0.0
    accepted = [d for d in decisions if d.accepted]
    min_merge_sim = min([d.evidence.cosine_similarity for d in accepted if d.evidence.cosine_similarity is not None], default=None)
    # Mean overlap proxies current paper output's CMNF mean-overlap style.
    overlaps = []
    for d in decisions:
        if d.evidence.cosine_similarity is not None and d.evidence.cmnf_status == "FAIL":
            overlaps.append(d.evidence.cosine_similarity)
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    min_signals = min([d.evidence.evidence_signal_count for d in accepted], default=0)
    return {
        "EENF_q95": eenf_q95,
        "EENF_max": eenf_max,
        "EENF_tau": args.tau_eenf,
        "EENF_status": "PASS" if eenf_q95 <= args.tau_eenf else "FAIL",
        "AANF_min_merge_sim": min_merge_sim,
        "AANF_tau": args.tau_aanf,
        "AANF_status": "PASS" if (min_merge_sim is not None and min_merge_sim >= args.tau_aanf) else ("NA" if not accepted else "FAIL"),
        "CMNF_mean_overlap": mean_overlap,
        "CMNF_tau": args.tau_cmnf,
        "CMNF_status": "PASS" if mean_overlap <= args.tau_cmnf else "FAIL",
        "ECNF_min_signals": min_signals,
        "ECNF_m_min": args.m_min,
        "ECNF_status": "PASS" if min_signals >= args.m_min and accepted else ("NA" if not accepted else "FAIL"),
        "DBNF_status": "NA" if not args.drift_model else "PENDING",
    }


# -----------------------------
# Leakage evaluation
# -----------------------------

def evaluate_cross_context_leakage(mode: str, decisions: List[MergeDecision], negative_pairs: Set[Pair]) -> LeakageEvalResult:
    accepted = [d for d in decisions if d.accepted]
    if not accepted:
        return LeakageEvalResult(mode, 0, 0, 0.0, [], measurable=True)
    leaks = []
    for d in accepted:
        ev = d.evidence
        incompatible_context = ev.context_a != ev.context_b
        incompatible_ontology = bool(ev.ontology_root_a and ev.ontology_root_b and ev.ontology_root_a != ev.ontology_root_b)
        explicit_negative = d.pair in negative_pairs
        if incompatible_context or incompatible_ontology or explicit_negative:
            why = []
            if incompatible_context:
                why.append("context")
            if incompatible_ontology:
                why.append("ontology")
            if explicit_negative:
                why.append("explicit_negative")
            leaks.append(f"{d.pair.display()} ({'/'.join(why)})")
    rate = len(leaks) / len(accepted) if accepted else 0.0
    return LeakageEvalResult(mode, len(leaks), len(accepted), rate, leaks[:5], measurable=True)


# -----------------------------
# EENF sweep / timing
# -----------------------------

def run_eenf_g_sweep(attrs: List[AttributeRecord], embedder: EmbeddingProvider, g_values: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    baseline_mean = None
    baseline_time = None
    for G in g_values:
        start = time.perf_counter()
        vars_ = []
        for a in attrs[: min(60, len(attrs))]:
            regs = embedder.regenerations(a.name, a.context, G=G)
            vars_.append(float(np.mean(np.var(regs, axis=0))))
        elapsed = time.perf_counter() - start
        mean_v = float(np.mean(vars_)) if vars_ else 0.0
        q95_v = float(np.quantile(np.array(vars_), 0.95)) if vars_ else 0.0
        max_v = max(vars_) if vars_ else 0.0
        if baseline_mean is None:
            baseline_mean = mean_v
            baseline_time = elapsed
        # If G=1 variance is zero due a single sample, use q95/max interpretably and avoid division by zero.
        reduction = None if not baseline_mean else max(0.0, (baseline_mean - mean_v) / baseline_mean)
        overhead = None if not baseline_time else elapsed / baseline_time
        rows.append({
            "G": G,
            "mean_variance": mean_v,
            "q95_variance": q95_v,
            "max_variance": max_v,
            "variance_reduction_vs_G1": reduction,
            "encoding_time_sec": elapsed,
            "overhead_vs_G1": overhead,
        })
    return rows


def timing_summary(mode: str, decisions: List[MergeDecision]) -> List[Any]:
    vals = [d.timing.total_decision_ms for d in decisions]
    if not vals:
        return [mode, 0, "NA", "NA", "NA", "NA", "NA"]
    arr = np.array(vals, dtype=np.float64)
    return [mode, len(vals), float(np.mean(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)), float(np.max(arr))]


# -----------------------------
# Pair trace
# -----------------------------

def trace_pairwise_evidence(results: List[ModeResult], trace_pairs: Sequence[Sequence[str]]) -> None:
    if not trace_pairs:
        return
    rows = []
    for tp in trace_pairs:
        if len(tp) != 2:
            continue
        target = Pair.make(tp[0], tp[1])
        for result in results:
            matches = [d for d in result.decisions if d.pair == target]
            if not matches:
                rows.append([result.mode, tp[0], tp[1], "NOT FOUND", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue
            d = matches[0]
            e = d.evidence
            rows.append([
                result.mode, e.attr_a, e.attr_b, e.source_a, e.source_b, e.context_a, e.context_b,
                fmt(e.cosine_similarity), fmt(e.name_similarity), e.ontology_root_a, e.ontology_root_b,
                e.ontology_match, fmt(e.value_cooccurrence), e.regex_a, e.regex_b, e.regex_match,
                fmt(e.vss_similarity), fmt(e.shape_similarity), fmt(e.aggregate_score),
                e.evidence_signal_count, e.aanf_status, e.ecnf_status, e.cmnf_status,
                e.final_decision, e.lineage_id,
            ])
    print_table([
        "mode", "attr_a", "attr_b", "source_a", "source_b", "context_a", "context_b",
        "cosine_similarity", "name_similarity", "ontology_root_a", "ontology_root_b",
        "ontology_match", "value_cooccurrence", "regex_a", "regex_b", "regex_match",
        "vss_similarity", "shape_similarity", "aggregate_score", "evidence_signal_count",
        "AANF_status", "ECNF_status", "CMNF_status", "final_decision", "lineage_id"
    ], rows, "PAIRWISE MERGE EVIDENCE TRACE")


# -----------------------------
# Drift evaluation
# -----------------------------

def compute_drift_hotspots(attrs: List[AttributeRecord], args: argparse.Namespace, base_embedder: EmbeddingProvider) -> List[Tuple[str, float]]:
    drift_embedder = EmbeddingProvider(args.drift_model, args.seed)
    names = [f"{a.name} | context={a.context}" for a in attrs]
    if not names:
        return []
    base = base_embedder.encode(names)
    drift = drift_embedder.encode(names)
    # Align dimensionality conservatively by truncating to common dimension.
    k = min(base.shape[1], drift.shape[1])
    distances = np.linalg.norm(base[:, :k] - drift[:, :k], axis=1)
    by_name: Dict[str, float] = {}
    for a, d in zip(attrs, distances):
        by_name[a.key] = max(by_name.get(a.key, 0.0), float(d))
    return sorted(by_name.items(), key=lambda x: (-x[1], x[0]))[: args.drift_top_k]


def load_drift_ground_truth(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"WARNING: drift ground truth file not found: {p}. Drift detection marked NOT MEASURABLE.")
        return None
    data = load_json(p)
    attrs = data.get("drift_attributes", data.get("true_drift_attributes", []))
    s = {normalize_key(x) for x in attrs}
    return s if s else None


def evaluate_drift_detection(result: ModeResult, drift_truth: Optional[Set[str]], tau: float) -> DriftEvalResult:
    detected = {name for name, d in result.drift_hotspots if d > tau}
    if drift_truth is None:
        return DriftEvalResult(result.mode, tau, len(detected), 0, 0, 0, 0, None, None, None, None, measurable=False)
    tp = len(detected & drift_truth)
    fp = len(detected - drift_truth)
    fn = len(drift_truth - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # True negatives require a closed universe. Here the universe is attributes in result.
    universe = {a.key for a in result.attributes}
    tn = len((universe - drift_truth) - detected)
    accuracy = (tp + tn) / max(1, len(universe))
    return DriftEvalResult(result.mode, tau, len(detected), len(drift_truth), tp, fp, fn, precision, recall, f1, accuracy, measurable=True)


# -----------------------------
# Paper-aligned / claim support reports
# -----------------------------

def status_with_tolerance(measured: Optional[float], expected: float, tol: float, as_fraction: bool = False) -> str:
    if measured is None:
        return "NOT MEASURABLE"
    m = measured * 100.0 if as_fraction else measured
    return "PASS" if abs(m - expected) <= tol else "FAIL"


def paper_table_2_rows(results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult]) -> List[List[Any]]:
    rows = []
    for r in results:
        # Map paper approaches to modes.
        if r.mode in {"sdnf_hybrid", "hybrid"}:
            exp_prec, exp_rec, exp_leak = PAPER_CLAIMS["sdnf_precision_pct"], PAPER_CLAIMS["sdnf_recall_pct"], PAPER_CLAIMS["sdnf_leakage_pct"]
        elif r.mode == "embed_only_baseline":
            exp_prec, exp_rec, exp_leak = PAPER_CLAIMS["baseline_precision_pct"], PAPER_CLAIMS["baseline_recall_pct"], PAPER_CLAIMS["baseline_leakage_pct"]
        else:
            exp_prec = exp_rec = exp_leak = None
        ae = alias_evals.get(r.mode)
        le = leakage_evals.get(r.mode)
        statuses = [status_with_tolerance(r.schema_reduction_pct, PAPER_CLAIMS["schema_reduction_pct"], TOLERANCES["schema_reduction_pct"])]
        if exp_prec is not None:
            statuses.append(status_with_tolerance(ae.precision if ae else None, exp_prec, TOLERANCES["precision_pct"], as_fraction=True))
            statuses.append(status_with_tolerance(ae.recall if ae else None, exp_rec, TOLERANCES["recall_pct"], as_fraction=True))
            statuses.append(status_with_tolerance(le.leakage_rate if le else None, exp_leak, TOLERANCES["leakage_pct"], as_fraction=True))
        else:
            statuses.append("NA")
        rows.append([
            r.mode,
            f"{r.schema_reduction_pct:.1f}%",
            pct(ae.precision if ae else None),
            pct(ae.recall if ae else None),
            pct(ae.f1 if ae else None),
            pct(le.leakage_rate if le else None),
            r.canon_final,
            r.input_attributes,
            ";".join(statuses),
        ])
    return rows


def print_claim_support_summary(
    dataset_summary: Dict[str, Any],
    results: List[ModeResult],
    alias_evals: Dict[str, AliasEvalResult],
    leakage_evals: Dict[str, LeakageEvalResult],
    eenf_rows: Optional[List[Dict[str, Any]]],
    drift_evals: Optional[Dict[str, DriftEvalResult]],
    trace_requested: bool,
) -> None:
    # Prefer sdnf_hybrid for SDNF claims, fallback to hybrid.
    by_mode = {r.mode: r for r in results}
    sdnf = by_mode.get("sdnf_hybrid") or by_mode.get("hybrid")
    base = by_mode.get("embed_only_baseline")
    rows = []

    def add(claim: str, measured: Any, expected: Any, status: str, evidence_table: str):
        rows.append([claim, measured, expected, status, evidence_table])

    add("7 schema files ingested", dataset_summary.get("schema_files_ingested"), 7,
        "PASS" if dataset_summary.get("schema_files_ingested") == 7 else "FAIL", "DATASET SUMMARY")
    add("40 payload files ingested from current output basis", dataset_summary.get("payload_files_ingested"), 40,
        "PASS" if dataset_summary.get("payload_files_ingested") == 40 else "FAIL", "DATASET SUMMARY")
    if sdnf:
        add("80 input attributes", sdnf.input_attributes, 80, "PASS" if sdnf.input_attributes == 80 else "FAIL", "DATASET SUMMARY")
        add("49 final canonical attributes", sdnf.canon_final, 49, "PASS" if sdnf.canon_final == 49 else "FAIL", "PAPER TABLE 2 REPRODUCTION CHECK")
        add("38.7% schema reduction", f"{sdnf.schema_reduction_pct:.1f}%", "38.7%",
            status_with_tolerance(sdnf.schema_reduction_pct, 38.7, 0.2), "PAPER TABLE 2 REPRODUCTION CHECK")
        ae = alias_evals.get(sdnf.mode)
        le = leakage_evals.get(sdnf.mode)
        add("SDNF precision 95%", pct(ae.precision if ae else None), "95.0%",
            status_with_tolerance(ae.precision if ae else None, 95.0, 1.0, True), "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
        add("SDNF recall 90%", pct(ae.recall if ae else None), "90.0%",
            status_with_tolerance(ae.recall if ae else None, 90.0, 1.0, True), "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
        add("SDNF cross-context leakage approximately 2%", pct(le.leakage_rate if le else None), "2.0%",
            status_with_tolerance(le.leakage_rate if le else None, 2.0, 0.5, True), "CROSS-CONTEXT LEAKAGE EVALUATION")
    else:
        add("SDNF metrics", "NA", "paper Table 2", "NOT MEASURABLE", "PAPER TABLE 2 REPRODUCTION CHECK")

    if base:
        ae = alias_evals.get(base.mode)
        le = leakage_evals.get(base.mode)
        add("baseline precision 86%", pct(ae.precision if ae else None), "86.0%",
            status_with_tolerance(ae.precision if ae else None, 86.0, 1.0, True), "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
        add("baseline recall 95%", pct(ae.recall if ae else None), "95.0%",
            status_with_tolerance(ae.recall if ae else None, 95.0, 1.0, True), "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
        add("baseline cross-context leakage approximately 9%", pct(le.leakage_rate if le else None), "9.0%",
            status_with_tolerance(le.leakage_rate if le else None, 9.0, 0.5, True), "CROSS-CONTEXT LEAKAGE EVALUATION")

    if eenf_rows:
        by_g = {r["G"]: r for r in eenf_rows}
        for G, expected in [(10, 40.0), (20, 70.0)]:
            val = by_g.get(G, {}).get("variance_reduction_vs_G1")
            add(f"G={G} variance reduction approximately {expected:.0f}%", pct(val), f"{expected:.1f}%",
                status_with_tolerance(val, expected, 5.0, True), "EENF STABILITY-LATENCY SWEEP")
    else:
        add("G=10/G=20 variance reduction", "NA", "40% / 70%", "NOT MEASURABLE", "EENF STABILITY-LATENCY SWEEP")

    if sdnf:
        vals = [d.timing.total_decision_ms for d in sdnf.decisions]
        mean_ms = float(np.mean(vals)) if vals else None
        add("average merge decision under 50ms", fmt(mean_ms), "<50ms",
            "PASS" if mean_ms is not None and mean_ms < 50.0 else ("FAIL" if mean_ms is not None else "NOT MEASURABLE"),
            "MERGE DECISION TIMING SUMMARY")

    if drift_evals:
        de = drift_evals.get(sdnf.mode if sdnf else "hybrid") or next(iter(drift_evals.values()))
        add("DBNF drift detection precision/recall", f"P={pct(de.precision)} R={pct(de.recall)}", "ground-truth evaluated",
            "PASS" if de.measurable else "NOT MEASURABLE", "DBNF DRIFT DETECTION EVALUATION")
    else:
        add("DBNF drift detection accuracy / precision / recall", "NA", "ground-truth evaluated", "NOT MEASURABLE", "DBNF DRIFT DETECTION EVALUATION")

    add("acct_num / PrimaryAccountNumber evidence trace values", "printed" if trace_requested else "NA", "computed trace", "PASS" if trace_requested else "NOT MEASURABLE", "PAIRWISE MERGE EVIDENCE TRACE")
    print_table(["paper_claim", "measured_value", "expected_value", "status", "evidence_table"], rows, "CLAIM SUPPORT SUMMARY")


# -----------------------------
# Main printing
# -----------------------------

def parse_trace_pairs(flat: Sequence[str]) -> List[List[str]]:
    # argparse with nargs=2 append gives list[list[str]] if configured; this fallback supports flat too.
    if not flat:
        return []
    if all(isinstance(x, list) for x in flat):  # type: ignore
        return list(flat)  # type: ignore
    vals = list(flat)
    return [vals[i:i+2] for i in range(0, len(vals), 2) if len(vals[i:i+2]) == 2]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reviewer-grade SDNF experiment/audit harness")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--payloads_dir", default="payloads")
    p.add_argument("--evidence_mode", default="hybrid")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--drift_model", default=None)
    p.add_argument("--ground_truth_aliases", default=None)
    p.add_argument("--drift_ground_truth", default=None)
    p.add_argument("--trace_pair", nargs=2, action="append", default=[])
    p.add_argument("--eenf_g_sweep", default=None, help="comma-separated G values, e.g. 1,10,20")
    p.add_argument("--measure_timing", action="store_true")
    p.add_argument("--context", default="Payments Risk")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tau_eenf", type=float, default=0.000129)
    p.add_argument("--tau_aanf", type=float, default=0.650)
    p.add_argument("--tau_cmnf", type=float, default=0.100)
    p.add_argument("--tau_dbnf", type=float, default=0.250)
    p.add_argument("--tau_dbnf_drift", type=float, default=0.150)
    p.add_argument("--gamma", type=float, default=0.70)
    p.add_argument("--m_min", type=int, default=4)
    p.add_argument("--drift_top_k", type=int, default=10)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    embedder = EmbeddingProvider(args.model, args.seed)
    attrs, dataset_summary = collect_attributes(Path(args.data_dir), Path(args.payloads_dir), args.context, embedder)

    print_table(["Option", "Value"], [
        ["timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())],
        ["seed", args.seed],
        ["evidence_mode", args.evidence_mode],
        ["model", args.model],
        ["embedding_backend", embedder.backend],
        ["drift_model", args.drift_model or ""],
        ["data_dir", args.data_dir],
        ["payloads_dir", args.payloads_dir],
        ["context", args.context],
        ["tau_EENF", args.tau_eenf],
        ["tau_AANF", args.tau_aanf],
        ["tau_CMNF", args.tau_cmnf],
        ["tau_DBNF", args.tau_dbnf_drift if args.drift_model else args.tau_dbnf],
        ["gamma", args.gamma],
        ["m_min", args.m_min],
        ["ground_truth_aliases", args.ground_truth_aliases or ""],
        ["drift_ground_truth", args.drift_ground_truth or ""],
    ], "RUN CONFIGURATION")

    print_table(["Metric", "Value"], [
        ["schema_files_ingested", dataset_summary.get("schema_files_ingested")],
        ["payload_files_ingested", dataset_summary.get("payload_files_ingested")],
        ["raw_attribute_records", dataset_summary.get("raw_attribute_records")],
        ["distinct_attribute_names", dataset_summary.get("distinct_attribute_names")],
        ["value_evidence_available", dataset_summary.get("value_evidence_available")],
        ["value_evidence_missing", dataset_summary.get("value_evidence_missing")],
        ["missing_fraction", pct(dataset_summary.get("missing_fraction"))],
    ], "DATASET SUMMARY")

    if not attrs:
        print("WARNING: no attributes found. Provide JSON files in --data_dir and/or --payloads_dir.")

    true_pairs, negative_pairs = load_ground_truth_aliases(args.ground_truth_aliases)

    results = [run_mode(attrs, m, args, embedder) for m in mode_list(args.evidence_mode)]

    # Normal-form validation summary.
    nf_rows = []
    for r in results:
        n = r.nf_metrics
        nf_rows.extend([
            [r.mode, "EENF", "q95(var) <= tau", f"q95={n['EENF_q95']:.6f}; max={n['EENF_max']:.6f}", f"tau={n['EENF_tau']:.6f}", n["EENF_status"]],
            [r.mode, "AANF", "min_merge_sim >= tau", fmt(n.get("AANF_min_merge_sim")), f"tau={n['AANF_tau']:.3f}", n["AANF_status"]],
            [r.mode, "CMNF", "mean_overlap <= tau", fmt(n.get("CMNF_mean_overlap")), f"tau={n['CMNF_tau']:.3f}", n["CMNF_status"]],
            [r.mode, "ECNF", "min_signals >= m_min", n.get("ECNF_min_signals"), f"m_min={n['ECNF_m_min']}", n["ECNF_status"]],
            [r.mode, "DBNF", "max_drift <= tau" if args.drift_model else "drift model not enabled", fmt(n.get("DBNF_max")), f"tau={n.get('DBNF_tau', args.tau_dbnf)}", n["DBNF_status"]],
            [r.mode, "RRNF", "not exercised", "NA", "NA", "INFO"],
            [r.mode, "PONF", "not exercised", "NA", "NA", "INFO"],
        ])
    print_table(["mode", "NormalForm", "Rule", "Actual", "Expected", "Status"], nf_rows, "NORMAL FORM VALIDATION SUMMARY")

    # Alias eval.
    alias_evals: Dict[str, AliasEvalResult] = {r.mode: evaluate_alias_merges(r.mode, r.predicted_pairs, true_pairs) for r in results}
    print_table(["mode", "predicted_pairs", "true_pairs", "TP", "FP", "FN", "precision", "recall", "F1"], [
        [ae.mode, ae.predicted_pairs, ae.true_pairs if ae.measurable else "NOT MEASURABLE", ae.tp, ae.fp, ae.fn, pct(ae.precision), pct(ae.recall), pct(ae.f1)]
        for ae in alias_evals.values()
    ], "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
    for ae in alias_evals.values():
        if ae.false_positive_examples or ae.false_negative_examples:
            print_table(["mode", "type", "examples_top10"], [
                [ae.mode, "false_positive", "; ".join(ae.false_positive_examples) or ""],
                [ae.mode, "false_negative", "; ".join(ae.false_negative_examples) or ""],
            ], f"ALIAS ERROR EXAMPLES ({ae.mode})")

    # Leakage eval.
    leakage_evals: Dict[str, LeakageEvalResult] = {r.mode: evaluate_cross_context_leakage(r.mode, r.decisions, negative_pairs) for r in results}
    print_table(["mode", "leakage_count", "predicted_merge_count", "leakage_rate", "top_leakage_examples"], [
        [le.mode, le.leakage_count, le.predicted_merge_count, pct(le.leakage_rate), "; ".join(le.examples)]
        for le in leakage_evals.values()
    ], "CROSS-CONTEXT LEAKAGE EVALUATION")

    # Ablation summary.
    print_table(["mode", "predicted_pairs", "canon_final", "input_attributes", "schema_reduction", "precision", "recall", "leakage"], [
        [r.mode, len(r.predicted_pairs), r.canon_final, r.input_attributes, f"{r.schema_reduction_pct:.1f}%", pct(alias_evals[r.mode].precision), pct(alias_evals[r.mode].recall), pct(leakage_evals[r.mode].leakage_rate)]
        for r in results
    ], "ABLATION STUDY SUMMARY")

    # Paper Table 2 check.
    print_table(["approach", "schema_reduction", "merge_precision", "merge_recall", "F1", "cross_context_leakage", "canon_final", "input_attributes", "supported_paper_claim_status"],
                paper_table_2_rows(results, alias_evals, leakage_evals), "PAPER TABLE 2 REPRODUCTION CHECK")

    # EENF sweep.
    eenf_rows = None
    if args.eenf_g_sweep:
        g_values = [int(x.strip()) for x in args.eenf_g_sweep.split(",") if x.strip()]
        eenf_rows = run_eenf_g_sweep(attrs, embedder, g_values)
        print_table(["G", "mean_variance", "q95_variance", "max_variance", "variance_reduction_vs_G1", "encoding_time_sec", "overhead_vs_G1"], [
            [r["G"], fmt(r["mean_variance"], 8), fmt(r["q95_variance"], 8), fmt(r["max_variance"], 8), pct(r["variance_reduction_vs_G1"]), fmt(r["encoding_time_sec"]), fmt(r["overhead_vs_G1"])]
            for r in eenf_rows
        ], "EENF STABILITY-LATENCY SWEEP")

    # Timing summary always printed; --measure_timing kept for CLI compatibility.
    timing_rows = [timing_summary(r.mode, r.decisions) for r in results]
    print_table(["mode", "candidate_pairs", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"], timing_rows, "MERGE DECISION TIMING SUMMARY")
    for row in timing_rows:
        mean_ms = row[2]
        status = "PASS" if isinstance(mean_ms, float) and mean_ms < 50.0 else "FAIL"
        print(f"Average merge decision under 50ms ({row[0]}): {status}")

    # Pair traces.
    trace_pairs = parse_trace_pairs(args.trace_pair)
    trace_pairwise_evidence(results, trace_pairs)

    # Drift hotspots / evaluation.
    drift_evals = None
    if args.drift_model:
        for r in results:
            print_table(["TopDriftAttribute", "drift"], [[name, fmt(d)] for name, d in r.drift_hotspots], f"DBNF DRIFT HOTSPOTS ({r.mode})")
        drift_truth = load_drift_ground_truth(args.drift_ground_truth)
        drift_evals = {r.mode: evaluate_drift_detection(r, drift_truth, args.tau_dbnf_drift) for r in results}
        print_table(["mode", "drift_tau", "detected_count", "true_drift_count", "TP", "FP", "FN", "precision", "recall", "F1", "accuracy_if_defined"], [
            [de.mode, de.drift_tau, de.detected_count, de.true_drift_count if de.measurable else "NOT MEASURABLE", de.tp, de.fp, de.fn, pct(de.precision), pct(de.recall), pct(de.f1), pct(de.accuracy_if_defined)]
            for de in drift_evals.values()
        ], "DBNF DRIFT DETECTION EVALUATION")

    print_claim_support_summary(dataset_summary, results, alias_evals, leakage_evals, eenf_rows, drift_evals, bool(trace_pairs))


if __name__ == "__main__":
    main()
