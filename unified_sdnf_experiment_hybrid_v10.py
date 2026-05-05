#!/usr/bin/env python3
"""
unified_sdnf_experiment_hybrid_v10.py
Reviewer-grade SDNF experiment/audit harness for:
"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces".

Design goals
------------
- Never hardcode claimed paper metrics as measured outcomes.
- Compute claim-support metrics from explicit data, predicted lineage, optional ground truth,
  pairwise evidence, and timing instrumentation.
- Print reviewer-facing tables that PASS, FAIL, or mark claims as NOT MEASURABLE.
- Improve v9 auditability around: ground-truth loading, alias normalization, value/regex evidence,
  CMNF single-context handling, EENF G-sweep measurement, DBNF drift truth, and claim actions.

Execution examples
------------------
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases.json \
  --drift_model all-mpnet-base-v2 \
  --drift_ground_truth drift_ground_truth.json \
  --trace_pair acct_num PrimaryAccountNumber \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing

Strict paper reproduction mode:
python unified_sdnf_experiment_hybrid_v10.py \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases.json \
  --drift_model all-mpnet-base-v2 \
  --drift_ground_truth drift_ground_truth.json \
  --strict_paper_reproduction \
  --trace_pair acct_num PrimaryAccountNumber \
  --eenf_g_sweep 1,10,20 \
  --measure_timing

Supported ground_truth_aliases.json formats
-------------------------------------------
1) List-of-lists alias groups:
{
  "alias_groups": [
    ["acct_num", "PrimaryAccountNumber", "pan"],
    ["txn_amount", "amount", "transaction_amount"]
  ],
  "negative_pairs": [["account_id", "amount"]]
}

2) Explicit true pairs:
{
  "true_pairs": [["acct_num", "PrimaryAccountNumber"]],
  "negative_pairs": [["account_id", "amount"]]
}

3) Object-style alias groups for reviewer metadata:
{
  "alias_groups": [
    {
      "canonical": "primary_account_number",
      "aliases": ["acct_num", "PrimaryAccountNumber", "pan"],
      "basis": "Payment account/card number identifiers."
    }
  ],
  "negative_pairs": [["payer_name", "payee_name"]]
}

Supported drift_ground_truth.json formats
-----------------------------------------
1) Simple list:
{
  "drift_attributes": ["description", "iso_currency_code", "acct_num", "payer_name"]
}

2) Alternative key:
{
  "true_drift_attributes": ["description", "iso_currency_code"]
}

3) Object-style drift cases:
{
  "drift_cases": [
    {"attribute": "description", "basis": "Known simulated semantic drift case."}
  ]
}

Notes
-----
- sentence_transformers is optional. If unavailable, a deterministic hashing backend is used.
- This harness is explicit and audit-oriented rather than optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

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

TOLERANCES = {
    "schema_count": 0.0,
    "payload_count": 0.0,
    "input_attributes": 0.0,
    "canon_final": 0.0,
    "schema_reduction_pct": 0.2,
    "precision_pct": 1.0,
    "recall_pct": 1.0,
    "leakage_pct": 0.5,
    "variance_reduction_pct": 5.0,
}

DEFAULT_WEIGHTS = {
    "embedding": 0.4,
    "name": 0.2,
    "ontology": 0.1,
    "shape": 0.1,
    "vss": 0.2,
}

SYNONYM_CANON = {
    "acct": "account",
    "acc": "account",
    "acctnum": "account number",
    "accountnum": "account number",
    "num": "number",
    "nbr": "number",
    "pan": "primary account number",
    "ccy": "currency",
    "txn": "transaction",
    "tx": "transaction",
    "amt": "amount",
    "desc": "description",
    "memo": "description",
    "note": "description",
    "payer": "payer",
    "payee": "payee",
}

ROLE_TOKENS = {"payer", "payee", "merchant", "cardholder", "holder"}

# -----------------------------
# Dataclasses
# -----------------------------

@dataclass(frozen=True, order=True)
class Pair:
    """Normalized unordered pair used for ground truth and predictions."""
    a: str
    b: str

    @staticmethod
    def make(a: str, b: str) -> "Pair":
        x, y = sorted([canonical_pair_key(a), canonical_pair_key(b)])
        return Pair(x, y)

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
        return canonical_pair_key(self.name)

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
        return "NOT MEASURABLE"
    return f"{100.0 * x:.{nd}f}%"

def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> str:
    str_rows = [[fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out: List[str] = []
    if title:
        out.append("\n" + title)
    out.append(sep)
    out.append("|" + "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
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

def normalize_key_raw(s: str) -> str:
    return camel_to_tokens(str(s)).lower().strip()

def expand_synonym_token(t: str) -> List[str]:
    repl = SYNONYM_CANON.get(t, t)
    return repl.split()

def normalize_key(s: str) -> str:
    toks = normalize_key_raw(s).split()
    expanded: List[str] = []
    for t in toks:
        expanded.extend(expand_synonym_token(t))
    return " ".join(expanded).strip()

def canonical_pair_key(s: str) -> str:
    return normalize_key(s)

def token_set(s: str) -> Set[str]:
    return set(normalize_key(s).split())

def role_conflict(a: str, b: str) -> bool:
    ta, tb = token_set(a), token_set(b)
    if "payer" in ta and "payee" in tb:
        return True
    if "payee" in ta and "payer" in tb:
        return True
    return False

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
    if any(k in n for k in ["primary account number", "account number", "account", "iban", "routing", "card number"]):
        return "payment:account"
    if any(k in n for k in ["cvv", "cid", "security code", "verification", "card verification value"]):
        return "payment:cvv"
    if any(k in n for k in ["exp", "expiry", "expiration", "expiration date"]):
        return "payment:expiry"
    if any(k in n for k in ["amount", "transaction amount", "instd amount"]):
        return "payment:amount"
    if any(k in n for k in ["risk", "fraud", "score"]):
        return "risk:score"
    if any(k in n for k in ["currency", "iso currency"]):
        return "payment:currency"
    if any(k in n for k in ["description", "comment"]):
        return "text:description"
    if any(k in n for k in ["payer", "payee", "merchant", "holder", "name"]):
        return "party:name"
    return None

def is_account_like(name: str) -> bool:
    toks = token_set(name)
    n = " ".join(toks)
    return bool({"account", "card", "number", "primary"} & toks) or "primary account number" in n

def normalize_value_for_shape(name: str, value: Any) -> str:
    s = str(value).strip()
    if is_account_like(name):
        s = re.sub(r"[\s\-]", "", s)
        s = re.sub(r"[xX*]", "0", s)
    return s

def infer_regex(values: Sequence[Any], attr_name: str = "") -> str:
    samples = [normalize_value_for_shape(attr_name, v) for v in values if v is not None and str(v).strip() != ""]
    if not samples:
        return ""
    if all(re.fullmatch(r"\d{13,19}", s) for s in samples):
        return r"^[0-9]{13,19}$"
    if is_account_like(attr_name) and all(re.fullmatch(r"\d{6,19}", s) for s in samples):
        return r"^[0-9]{6,19}$"
    if all(re.fullmatch(r"[A-Z]{3}", s) for s in samples):
        return r"^[A-Z]{3}$"
    if all(re.fullmatch(r"\d+(\.\d+)?", s) for s in samples):
        return r"^[0-9]+(\.[0-9]+)?$"
    if all(re.fullmatch(r"\d+", s) for s in samples):
        lengths = sorted({len(s) for s in samples})
        if len(lengths) == 1:
            return rf"^[0-9]{{{lengths[0]}}}$"
        return r"^[0-9]+$"
    return "mixed"

def shape_token(v: Any) -> str:
    s = str(v)
    out: List[str] = []
    for ch in s:
        if ch.isdigit():
            out.append("D")
        elif ch.isalpha():
            out.append("A")
        elif ch.isspace():
            out.append("S")
        else:
            out.append("P")
    compact: List[str] = []
    for k, group in itertools.groupby(out):
        compact.append(k + str(len(list(group))))
    return "".join(compact)

def value_shape_signature(values: Sequence[Any], attr_name: str = "") -> str:
    if not values:
        return ""
    counts: Dict[str, int] = defaultdict(int)
    for v in values[:100]:
        counts[shape_token(normalize_value_for_shape(attr_name, v))] += 1
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))

def vss_from_values(values: Sequence[Any], attr_name: str = "") -> Optional[np.ndarray]:
    samples = [normalize_value_for_shape(attr_name, v) for v in values if v is not None and str(v).strip() != ""][:200]
    if not samples:
        return None
    lengths = np.array([len(s) for s in samples], dtype=np.float32)
    digit_frac = np.array([sum(ch.isdigit() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    alpha_frac = np.array([sum(ch.isalpha() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    punct_frac = np.array([sum(not ch.isalnum() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    numeric = np.array([1.0 if re.fullmatch(r"\d+(\.\d+)?", s) else 0.0 for s in samples], dtype=np.float32)
    unique_ratio = float(len(set(samples)) / max(1, len(samples)))
    vec = np.array([
        float(np.mean(lengths)), float(np.std(lengths)), float(np.min(lengths)), float(np.max(lengths)),
        float(np.mean(digit_frac)), float(np.mean(alpha_frac)), float(np.mean(punct_frac)), float(np.mean(numeric)),
        unique_ratio,
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

    @staticmethod
    def stable_int(s: str) -> int:
        return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little", signed=False)

    def _hash_vec(self, text: str, nonce: int = 0) -> np.ndarray:
        seed = self.stable_int(f"{self.model_name}|{self.seed}|{nonce}|{text}")
        rng = np.random.default_rng(seed)
        return safe_norm(rng.normal(size=self.dim).astype(np.float32))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            arr = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)
        return np.vstack([self._hash_vec(t, 0) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def regenerations(self, name: str, context: str, G: int, nonce: int = 0) -> np.ndarray:
        text = f"{name} context={context}"
        if self.model is not None:
            # Most sentence-transformers inference is deterministic. Add nonce only to measurement text
            # so repeated-batch EENF can expose backend sensitivity if any. This is disclosed by output.
            texts = [f"{text} regen={nonce}:{g}" for g in range(G)]
            return self.encode(texts)
        return np.vstack([self._hash_vec(text, nonce * 100000 + g) for g in range(G)]).astype(np.float32)

# -----------------------------
# File / JSON extraction
# -----------------------------

def iter_json_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.json") if p.is_file()], key=lambda p: p.name.lower())

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def resolve_optional_input_file(path: Optional[str], label: str, strict: bool = False, search_dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    if not path:
        if strict:
            raise FileNotFoundError(f"{label} is required in strict paper reproduction mode but was not provided.")
        return None
    script_dir = Path(__file__).resolve().parent
    raw = Path(path)
    candidates = [raw, Path.cwd() / path, script_dir / path]
    for d in (search_dirs or []):
        candidates.append(Path(d) / path)
    candidates.extend([script_dir / "data" / path, script_dir / "payloads" / path])
    seen: Set[str] = set()
    uniq: List[Path] = []
    for c in candidates:
        s = str(c.resolve() if c.exists() else c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)
    for c in uniq:
        if c.exists():
            return c
    searched = "\n".join(f"  - {c}" for c in uniq)
    msg = f"WARNING: {label} file not found: {path}. Searched:\n{searched}"
    if strict:
        raise FileNotFoundError(msg.replace("WARNING: ", ""))
    print(msg + "\nCorresponding metrics will be marked NOT MEASURABLE.")
    return None

def walk_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from walk_json(v, path)
            else:
                yield path, v
    elif isinstance(obj, list):
        for v in obj:
            path = f"{prefix}[]" if prefix else "[]"
            if isinstance(v, (dict, list)):
                yield from walk_json(v, path)
            else:
                yield path, v

def context_from_source(path: Path, default_context: str) -> str:
    # Conservative: unless data contains explicit context, use configured paper context.
    return default_context

def collect_attributes(data_dir: Path, payloads_dir: Path, context: str, embedder: EmbeddingProvider) -> Tuple[List[AttributeRecord], Dict[str, Any]]:
    schema_files = iter_json_files(data_dir)
    payload_files = iter_json_files(payloads_dir)
    by_key: Dict[Tuple[str, str], AttributeRecord] = {}

    def add_value(field_path: str, value: Any, source: Path, ctx: str) -> None:
        name = field_path.split(".")[-1].replace("[]", "") or field_path
        key = (canonical_pair_key(name), ctx)
        if key not in by_key:
            by_key[key] = AttributeRecord(name=name, source=source.name, context=ctx, path=field_path)
        rec = by_key[key]
        if len(rec.values) < 500:
            rec.values.append(value)

    for f in schema_files + payload_files:
        try:
            data = load_json(f)
        except Exception as ex:
            print(f"WARNING: could not load {f}: {ex}")
            continue
        ctx = context_from_source(f, context)
        for path, value in walk_json(data):
            add_value(path, value, f, ctx)

    attrs = list(by_key.values())
    texts = [f"{a.name} context={a.context}" for a in attrs]
    embeddings = embedder.encode(texts) if attrs else np.zeros((0, embedder.dim), dtype=np.float32)
    for a, e in zip(attrs, embeddings):
        a.embedding = e
        a.ontology_root = ontology_root(a.name)
        a.regex = infer_regex(a.values, a.name)
        a.shape = value_shape_signature(a.values, a.name)
        a.vss = vss_from_values(a.values, a.name)
        a.canonical = a.key

    summary = {
        "schema_files_ingested": len(schema_files),
        "payload_files_ingested": len(payload_files),
        "raw_attribute_records": sum(len(a.values) for a in attrs),
        "distinct_attribute_names": len({a.key for a in attrs}),
        "value_evidence_available": sum(1 for a in attrs if a.values),
        "value_evidence_missing": sum(1 for a in attrs if not a.values),
    }
    total = max(1, summary["value_evidence_available"] + summary["value_evidence_missing"])
    summary["missing_fraction"] = summary["value_evidence_missing"] / total
    return attrs, summary

# -----------------------------
# Ground truth loading / evaluation
# -----------------------------

def pairs_from_alias_groups(groups: Sequence[Any]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for group in groups:
        if isinstance(group, dict):
            aliases = list(group.get("aliases", []))
            canonical = group.get("canonical")
            if canonical:
                aliases = [canonical] + aliases
        else:
            aliases = list(group)
        normed = sorted({canonical_pair_key(x) for x in aliases if str(x).strip()})
        for a, b in itertools.combinations(normed, 2):
            pairs.add(Pair(a, b))
    return pairs

def load_ground_truth_aliases(path: Optional[str], args: argparse.Namespace) -> Tuple[Optional[Set[Pair]], Set[Pair]]:
    p = resolve_optional_input_file(path, "ground truth aliases", args.strict_paper_reproduction, [Path(args.data_dir), Path(args.payloads_dir)])
    if p is None:
        return None, set()
    data = load_json(p)
    true_pairs: Set[Pair] = set()
    negative_pairs: Set[Pair] = set()
    if "alias_groups" in data:
        true_pairs |= pairs_from_alias_groups(data.get("alias_groups", []))
    if "true_pairs" in data:
        for pair in data.get("true_pairs", []):
            if len(pair) == 2:
                true_pairs.add(Pair.make(pair[0], pair[1]))
    if "negative_pairs" in data:
        for pair in data.get("negative_pairs", []):
            if len(pair) == 2:
                negative_pairs.add(Pair.make(pair[0], pair[1]))
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AliasEvalResult(
        mode, len(predicted), len(true_pairs), tp, fp, fn, precision, recall, f1,
        [p.display() for p in sorted(fp_set)[:10]],
        [p.display() for p in sorted(fn_set)[:10]],
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
    if not a.values or not b.values:
        return None
    n = max(len(a.values), len(b.values))
    return min(len(a.values), len(b.values)) / n if n else None

def compute_pair_evidence(a: AttributeRecord, b: AttributeRecord, mode: str, tau_aanf: float, gamma: float, m_min: int) -> PairEvidence:
    cos_sim = cosine(a.embedding, b.embedding)
    name_sim = jaccard(token_set(a.name), token_set(b.name))
    ont_match = bool(a.ontology_root and b.ontology_root and a.ontology_root == b.ontology_root)
    cooc = value_cooccurrence(a, b)
    regex_match = bool(a.regex and b.regex and a.regex == b.regex)
    vss_sim = cosine(a.vss, b.vss)
    shape_sim = 1.0 if (a.shape and b.shape and a.shape == b.shape) else (jaccard(set(a.shape.split(";")), set(b.shape.split(";"))) if a.shape and b.shape else None)

    if role_conflict(a.name, b.name):
        ont_match = False

    # Mode-specific available evidence. This preserves ablation semantics.
    signal_scores: Dict[str, Optional[float]] = {}
    if mode not in {"shape_only", "name_ontology_only"}:
        signal_scores["embedding"] = cos_sim
    if mode not in {"vss_only", "shape_only"}:
        signal_scores["name"] = name_sim
        signal_scores["ontology"] = 1.0 if ont_match else 0.0
    if mode not in {"no_value_evidence", "name_ontology_only"}:
        if mode != "vss_only":
            signal_scores["shape"] = max(float(shape_sim or 0.0), 1.0 if regex_match else 0.0) if (shape_sim is not None or regex_match) else None
        if mode != "shape_only":
            signal_scores["vss"] = vss_sim
    if mode == "shape_only":
        signal_scores = {"shape": max(float(shape_sim or 0.0), 1.0 if regex_match else 0.0)}
    if mode == "vss_only":
        signal_scores = {"vss": vss_sim}
    if mode == "name_ontology_only":
        signal_scores = {"name": name_sim, "ontology": 1.0 if ont_match else 0.0}

    agg = aggregate_score(signal_scores)

    supportive = 0
    if cos_sim is not None and cos_sim >= tau_aanf:
        supportive += 1
    if name_sim is not None and name_sim >= 0.50:
        supportive += 1
    if ont_match:
        supportive += 1
    if cooc is not None and cooc >= 0.50:
        supportive += 1
    if regex_match:
        supportive += 1
    if vss_sim is not None and vss_sim >= 0.90:
        supportive += 1
    if shape_sim is not None and shape_sim >= 0.80:
        supportive += 1

    aanf = "PASS" if cos_sim is not None and cos_sim >= tau_aanf else "FAIL"
    ecnf = "PASS" if supportive >= m_min and agg is not None and agg >= gamma else "FAIL"
    cmnf = "PASS" if a.context == b.context and not role_conflict(a.name, b.name) else "FAIL"

    ev = PairEvidence(
        attr_a=a.name, attr_b=b.name, source_a=a.source, source_b=b.source,
        context_a=a.context, context_b=b.context,
        cosine_similarity=cos_sim, name_similarity=name_sim,
        ontology_root_a=a.ontology_root, ontology_root_b=b.ontology_root, ontology_match=ont_match,
        value_cooccurrence=cooc, regex_a=a.regex, regex_b=b.regex, regex_match=regex_match,
        vss_similarity=vss_sim, shape_similarity=shape_sim, aggregate_score=agg,
        evidence_signal_count=supportive, aanf_status=aanf, ecnf_status=ecnf, cmnf_status=cmnf,
    )
    return ev

def decision_for_mode(e: PairEvidence, mode: str, gamma: float, m_min: int) -> Tuple[bool, str]:
    if mode in {"embed_only", "embed_only_baseline"}:
        ok = e.aanf_status == "PASS"
        return ok, "embedding cosine threshold only" if ok else "embedding below AANF threshold"
    if mode == "no_ecnf":
        ok = e.aanf_status == "PASS" and e.cmnf_status == "PASS"
        return ok, "AANF+CMNF only; ECNF ablated" if ok else "AANF or CMNF failed"
    if mode == "no_cmnf":
        ok = e.aanf_status == "PASS" and e.ecnf_status == "PASS"
        return ok, "AANF+ECNF only; CMNF ablated" if ok else "AANF or ECNF failed"
    if mode in {"vss_only", "shape_only", "name_ontology_only"}:
        ok = e.aggregate_score is not None and e.aggregate_score >= gamma
        return ok, "single/limited evidence mode threshold" if ok else "limited evidence below gamma"
    ok = e.aanf_status == "PASS" and e.ecnf_status == "PASS" and e.cmnf_status == "PASS"
    if ok:
        return True, "AANF, ECNF, and CMNF passed"
    reasons = []
    if e.aanf_status != "PASS":
        reasons.append("AANF failed")
    if e.ecnf_status != "PASS":
        reasons.append("ECNF failed")
    if e.cmnf_status != "PASS":
        reasons.append("CMNF failed")
    return False, "; ".join(reasons)

class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

# -----------------------------
# Mode run / metrics
# -----------------------------

def mode_list(requested: str) -> List[str]:
    if requested == "all":
        return [
            "embed_only_baseline", "sdnf_hybrid", "no_ecnf", "no_cmnf", "no_dbnf",
            "no_value_evidence", "vss_only", "shape_only", "name_ontology_only", "hybrid",
        ]
    if requested == "embed_only":
        return ["embed_only_baseline"]
    return [requested]

def compute_cmnf_status(attrs: List[AttributeRecord], tau_cmnf: float) -> Dict[str, Any]:
    contexts = sorted({a.context for a in attrs})
    if len(contexts) < 2:
        return {"CMNF_mean_overlap": None, "CMNF_tau": tau_cmnf, "CMNF_status": "NA", "CMNF_reason": "single context; cross-context CMNF not exercised"}
    overlaps: List[float] = []
    by_primitive: Dict[str, List[AttributeRecord]] = defaultdict(list)
    for a in attrs:
        by_primitive[a.key].append(a)
    for group in by_primitive.values():
        for a, b in itertools.combinations(group, 2):
            if a.context != b.context:
                sim = cosine(a.embedding, b.embedding)
                if sim is not None:
                    overlaps.append(sim)
    if not overlaps:
        return {"CMNF_mean_overlap": None, "CMNF_tau": tau_cmnf, "CMNF_status": "NA", "CMNF_reason": "no repeated primitive across distinct contexts"}
    mean_overlap = float(np.mean(overlaps))
    return {"CMNF_mean_overlap": mean_overlap, "CMNF_tau": tau_cmnf, "CMNF_status": "PASS" if mean_overlap <= tau_cmnf else "FAIL", "CMNF_reason": "same normalized primitive across distinct contexts"}

def compute_nf_metrics(attrs: List[AttributeRecord], decisions: List[MergeDecision], args: argparse.Namespace, embedder: EmbeddingProvider) -> Dict[str, Any]:
    regs = []
    for a in attrs[: min(60, len(attrs))]:
        r = embedder.regenerations(a.name, a.context, G=10)
        regs.append(float(np.mean(np.var(r, axis=0))))
    accepted = [d for d in decisions if d.accepted]
    min_merge_sim = min([d.evidence.cosine_similarity for d in accepted if d.evidence.cosine_similarity is not None], default=None)
    min_signals = min([d.evidence.evidence_signal_count for d in accepted], default=0)
    cmnf = compute_cmnf_status(attrs, args.tau_cmnf)
    out = {
        "EENF_q95": float(np.quantile(np.array(regs), 0.95)) if regs else 0.0,
        "EENF_max": max(regs) if regs else 0.0,
        "EENF_tau": args.tau_eenf,
        "EENF_status": "PASS" if (float(np.quantile(np.array(regs), 0.95)) if regs else 0.0) <= args.tau_eenf else "FAIL",
        "AANF_min_merge_sim": min_merge_sim,
        "AANF_tau": args.tau_aanf,
        "AANF_status": "PASS" if (min_merge_sim is not None and min_merge_sim >= args.tau_aanf) else ("NA" if not accepted else "FAIL"),
        "ECNF_min_signals": min_signals,
        "ECNF_m_min": args.m_min,
        "ECNF_status": "PASS" if min_signals >= args.m_min and accepted else ("NA" if not accepted else "FAIL"),
        "DBNF_status": "NA" if not args.drift_model else "PENDING",
    }
    out.update(cmnf)
    return out

def run_mode(attrs: List[AttributeRecord], mode: str, args: argparse.Namespace, embedder: EmbeddingProvider) -> ModeResult:
    t0 = time.perf_counter()
    pairs = list(itertools.combinations(attrs, 2))
    t_candidate_ms = (time.perf_counter() - t0) * 1000.0
    decisions: List[MergeDecision] = []
    for idx, (a, b) in enumerate(pairs):
        start = time.perf_counter()
        evidence = compute_pair_evidence(a, b, mode, args.tau_aanf, args.gamma, args.m_min)
        scoring_ms = (time.perf_counter() - start) * 1000.0
        vstart = time.perf_counter()
        accepted, reason = decision_for_mode(evidence, mode, args.gamma, args.m_min)
        evidence.final_decision = "MERGE" if accepted else "DEFER"
        evidence.reason = reason
        evidence.lineage_id = f"{mode}-{idx:06d}"
        validation_ms = (time.perf_counter() - vstart) * 1000.0
        total_ms = t_candidate_ms / max(1, len(pairs)) + scoring_ms + validation_ms
        decisions.append(MergeDecision(Pair.make(a.name, b.name), evidence, TimingRecord(t_candidate_ms, scoring_ms, validation_ms, total_ms), accepted, mode))
    predicted = {d.pair for d in decisions if d.accepted}
    uf = UnionFind([a.key for a in attrs])
    for d in decisions:
        if d.accepted:
            uf.union(d.pair.a, d.pair.b)
    roots = {uf.find(a.key) for a in attrs}
    input_n = len({a.key for a in attrs})
    canon_final = len(roots)
    reduction = 100.0 * (input_n - canon_final) / input_n if input_n else 0.0
    nf = compute_nf_metrics(attrs, decisions, args, embedder)
    return ModeResult(mode, attrs, decisions, predicted, canon_final, input_n, reduction, nf)

# -----------------------------
# Evaluation helpers
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

def run_eenf_g_sweep(attrs: List[AttributeRecord], embedder: EmbeddingProvider, g_values: Sequence[int], repeats: int) -> List[Dict[str, Any]]:
    rows = []
    baseline_mean = None
    baseline_time = None
    for G in g_values:
        start = time.perf_counter()
        attr_vars = []
        for a in attrs[: min(60, len(attrs))]:
            batch_means = []
            for r in range(repeats):
                regs = embedder.regenerations(a.name, a.context, G=G, nonce=r)
                batch_means.append(np.mean(regs, axis=0))
            arr = np.stack(batch_means, axis=0)
            attr_vars.append(float(np.mean(np.var(arr, axis=0))))
        elapsed = time.perf_counter() - start
        mean_v = float(np.mean(attr_vars)) if attr_vars else 0.0
        q95_v = float(np.quantile(np.array(attr_vars), 0.95)) if attr_vars else 0.0
        max_v = max(attr_vars) if attr_vars else 0.0
        if baseline_mean is None:
            baseline_mean = mean_v
            baseline_time = elapsed
        reduction = None if baseline_mean is None or baseline_mean <= 1e-15 else max(0.0, (baseline_mean - mean_v) / baseline_mean)
        note = "" if reduction is not None else "deterministic or zero-variance backend; reduction not measurable"
        overhead = None if baseline_time is None or baseline_time <= 0 else elapsed / baseline_time
        rows.append({"G": G, "mean_variance": mean_v, "q95_variance": q95_v, "max_variance": max_v, "variance_reduction_vs_G1": reduction, "encoding_time_sec": elapsed, "overhead_vs_G1": overhead, "note": note})
    return rows

def timing_summary(mode: str, decisions: List[MergeDecision]) -> List[Any]:
    vals = [d.timing.total_decision_ms for d in decisions]
    if not vals:
        return [mode, 0, "NA", "NA", "NA", "NA", "NA"]
    arr = np.array(vals, dtype=np.float64)
    return [mode, len(vals), float(np.mean(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)), float(np.max(arr))]

def compute_drift_hotspots(attrs: List[AttributeRecord], args: argparse.Namespace, base_embedder: EmbeddingProvider) -> List[Tuple[str, float]]:
    if not args.drift_model:
        return []
    drift_embedder = EmbeddingProvider(args.drift_model, args.seed)
    names = [f"{a.name} context={a.context}" for a in attrs]
    if not names:
        return []
    base = base_embedder.encode(names)
    drift = drift_embedder.encode(names)
    k = min(base.shape[1], drift.shape[1])
    distances = np.linalg.norm(base[:, :k] - drift[:, :k], axis=1)
    by_name: Dict[str, float] = {}
    for a, d in zip(attrs, distances):
        by_name[a.key] = max(by_name.get(a.key, 0.0), float(d))
    return sorted(by_name.items(), key=lambda x: (-x[1], x[0]))[: args.drift_top_k]

def load_drift_ground_truth(path: Optional[str], args: argparse.Namespace) -> Optional[Set[str]]:
    p = resolve_optional_input_file(path, "drift ground truth", args.strict_paper_reproduction, [Path(args.data_dir), Path(args.payloads_dir)])
    if p is None:
        return None
    data = load_json(p)
    attrs: List[str] = []
    attrs.extend(data.get("drift_attributes", []))
    attrs.extend(data.get("true_drift_attributes", []))
    for c in data.get("drift_cases", []):
        if isinstance(c, dict) and c.get("attribute"):
            attrs.append(c["attribute"])
    s = {canonical_pair_key(x) for x in attrs if str(x).strip()}
    if not s:
        print(f"WARNING: no drift attributes found in {p}. Drift metrics marked NOT MEASURABLE.")
        return None
    return s

def evaluate_drift_detection(result: ModeResult, drift_truth: Optional[Set[str]], tau: float) -> DriftEvalResult:
    detected = {name for name, d in result.drift_hotspots if d > tau}
    if drift_truth is None:
        return DriftEvalResult(result.mode, tau, len(detected), 0, 0, 0, 0, None, None, None, None, measurable=False)
    tp = len(detected & drift_truth)
    fp = len(detected - drift_truth)
    fn = len(drift_truth - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    universe = {a.key for a in result.attributes}
    tn = len((universe - drift_truth) - detected)
    accuracy = (tp + tn) / max(1, len(universe))
    return DriftEvalResult(result.mode, tau, len(detected), len(drift_truth), tp, fp, fn, precision, recall, f1, accuracy, measurable=True)

# -----------------------------
# Reports
# -----------------------------

def status_with_tolerance(measured: Optional[float], expected: float, tol: float, as_fraction: bool = False, less_than: bool = False) -> str:
    if measured is None:
        return "NOT MEASURABLE"
    m = measured * 100.0 if as_fraction else measured
    if less_than:
        return "PASS" if m < expected else "FAIL"
    return "PASS" if abs(m - expected) <= tol else "FAIL"

def action_for_status(status: str, measurable_needed: bool = False, mismatch_kind: str = "") -> str:
    if status == "PASS":
        return "keep claim"
    if status == "NOT MEASURABLE" or measurable_needed:
        return "provide ground truth and rerun" if not mismatch_kind else mismatch_kind
    return "revise paper claim to measured value" if not mismatch_kind else mismatch_kind

def paper_table_2_rows(results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult]) -> List[List[Any]]:
    rows = []
    for r in results:
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
        rows.append([r.mode, f"{r.schema_reduction_pct:.1f}%", pct(ae.precision if ae else None), pct(ae.recall if ae else None), pct(ae.f1 if ae else None), pct(le.leakage_rate if le else None), r.canon_final, r.input_attributes, ";".join(statuses)])
    return rows

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
                rows.append([result.mode, tp[0], tp[1], "NOT FOUND", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue
            d = matches[0]
            e = d.evidence
            rows.append([result.mode, e.attr_a, e.attr_b, e.source_a, e.source_b, e.context_a, e.context_b, fmt(e.cosine_similarity), fmt(e.name_similarity), e.ontology_root_a, e.ontology_root_b, e.ontology_match, fmt(e.value_cooccurrence), e.regex_a, e.regex_b, e.regex_match, fmt(e.vss_similarity), fmt(e.shape_similarity), fmt(e.aggregate_score), e.evidence_signal_count, e.aanf_status, e.ecnf_status, e.cmnf_status, e.final_decision, e.reason, e.lineage_id])
    print_table(["mode", "attr_a", "attr_b", "source_a", "source_b", "context_a", "context_b", "cosine_similarity", "name_similarity", "ontology_root_a", "ontology_root_b", "ontology_match", "value_cooccurrence", "regex_a", "regex_b", "regex_match", "vss_similarity", "shape_similarity", "aggregate_score", "evidence_signal_count", "AANF_status", "ECNF_status", "CMNF_status", "final_decision", "reason", "lineage_id"], rows, "PAIRWISE MERGE EVIDENCE TRACE")

def print_claim_support_summary(dataset_summary: Dict[str, Any], results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult], eenf_rows: Optional[List[Dict[str, Any]]], drift_evals: Optional[Dict[str, DriftEvalResult]], trace_requested: bool) -> None:
    by_mode = {r.mode: r for r in results}
    sdnf = by_mode.get("sdnf_hybrid") or by_mode.get("hybrid")
    base = by_mode.get("embed_only_baseline")
    rows: List[List[Any]] = []

    def add(claim: str, measured: Any, expected: Any, status: str, table: str, action: str) -> None:
        rows.append([claim, measured, expected, status, table, action])

    sc = dataset_summary.get("schema_files_ingested")
    st = "PASS" if sc == PAPER_CLAIMS["schema_files_ingested"] else "FAIL"
    add("7 schema files ingested", sc, PAPER_CLAIMS["schema_files_ingested"], st, "DATASET SUMMARY", "dataset mismatch; align data or revise paper" if st == "FAIL" else "keep claim")

    pc = dataset_summary.get("payload_files_ingested")
    st = "PASS" if pc == PAPER_CLAIMS["payload_files_ingested_current_output"] else "FAIL"
    add("40 payload files ingested from current output basis", pc, PAPER_CLAIMS["payload_files_ingested_current_output"], st, "DATASET SUMMARY", "keep claim" if st == "PASS" else "dataset mismatch; align data or revise paper")

    inp = sdnf.input_attributes if sdnf else None
    st = "PASS" if inp == PAPER_CLAIMS["input_attributes"] else "FAIL"
    add("80 input attributes", inp, PAPER_CLAIMS["input_attributes"], st, "PAPER TABLE 2 REPRODUCTION CHECK", "revise paper claim to measured value" if st == "FAIL" else "keep claim")

    cf = sdnf.canon_final if sdnf else None
    st = "PASS" if cf == PAPER_CLAIMS["canon_final"] else "FAIL"
    add("49 final canonical attributes", cf, PAPER_CLAIMS["canon_final"], st, "PAPER TABLE 2 REPRODUCTION CHECK", "revise paper claim to measured value" if st == "FAIL" else "keep claim")

    red = sdnf.schema_reduction_pct if sdnf else None
    st = status_with_tolerance(red, PAPER_CLAIMS["schema_reduction_pct"], TOLERANCES["schema_reduction_pct"])
    add("38.7% schema reduction", f"{red:.1f}%" if red is not None else "NA", f"{PAPER_CLAIMS['schema_reduction_pct']}%", st, "PAPER TABLE 2 REPRODUCTION CHECK", action_for_status(st))

    if sdnf:
        ae = alias_evals.get(sdnf.mode)
        le = leakage_evals.get(sdnf.mode)
        stp = status_with_tolerance(ae.precision if ae else None, PAPER_CLAIMS["sdnf_precision_pct"], TOLERANCES["precision_pct"], as_fraction=True)
        add("SDNF precision 95%", pct(ae.precision if ae else None), "95.0%", stp, "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH", action_for_status(stp))
        strc = status_with_tolerance(ae.recall if ae else None, PAPER_CLAIMS["sdnf_recall_pct"], TOLERANCES["recall_pct"], as_fraction=True)
        add("SDNF recall 90%", pct(ae.recall if ae else None), "90.0%", strc, "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH", action_for_status(strc))
        stl = status_with_tolerance(le.leakage_rate if le else None, PAPER_CLAIMS["sdnf_leakage_pct"], TOLERANCES["leakage_pct"], as_fraction=True)
        add("SDNF cross-context leakage approximately 2%", pct(le.leakage_rate if le else None), "2.0%", stl, "CROSS-CONTEXT LEAKAGE EVALUATION", action_for_status(stl))

    if base:
        ae = alias_evals.get(base.mode)
        le = leakage_evals.get(base.mode)
        stp = status_with_tolerance(ae.precision if ae else None, PAPER_CLAIMS["baseline_precision_pct"], TOLERANCES["precision_pct"], as_fraction=True)
        add("baseline precision 86%", pct(ae.precision if ae else None), "86.0%", stp, "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH", action_for_status(stp))
        strc = status_with_tolerance(ae.recall if ae else None, PAPER_CLAIMS["baseline_recall_pct"], TOLERANCES["recall_pct"], as_fraction=True)
        add("baseline recall 95%", pct(ae.recall if ae else None), "95.0%", strc, "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH", action_for_status(strc))
        stl = status_with_tolerance(le.leakage_rate if le else None, PAPER_CLAIMS["baseline_leakage_pct"], TOLERANCES["leakage_pct"], as_fraction=True)
        add("baseline cross-context leakage approximately 9%", pct(le.leakage_rate if le else None), "9.0%", stl, "CROSS-CONTEXT LEAKAGE EVALUATION", action_for_status(stl))

    if eenf_rows:
        for target_g, exp_key in [(10, "g10_variance_reduction_pct"), (20, "g20_variance_reduction_pct")]:
            row = next((r for r in eenf_rows if int(r["G"]) == target_g), None)
            val = row.get("variance_reduction_vs_G1") if row else None
            st = status_with_tolerance(val, PAPER_CLAIMS[exp_key], TOLERANCES["variance_reduction_pct"], as_fraction=True)
            add(f"G={target_g} variance reduction approximately {PAPER_CLAIMS[exp_key]:.0f}%", pct(val), f"{PAPER_CLAIMS[exp_key]:.1f}%", st, "EENF STABILITY-LATENCY SWEEP", "not exercised in current run" if row is None else action_for_status(st))

    if sdnf:
        vals = [d.timing.total_decision_ms for d in sdnf.decisions]
        mean_ms = float(np.mean(vals)) if vals else None
        st = "PASS" if mean_ms is not None and mean_ms < PAPER_CLAIMS["avg_merge_decision_ms"] else "FAIL"
        add("average merge decision under 50ms", fmt(mean_ms), "<50ms", st, "MERGE DECISION TIMING SUMMARY", "keep claim" if st == "PASS" else "revise paper claim to measured value")

    if drift_evals and sdnf:
        de = drift_evals.get(sdnf.mode)
        st = "PASS" if de and de.measurable else "NOT MEASURABLE"
        measured = f"P={pct(de.precision)}/R={pct(de.recall)}/F1={pct(de.f1)}" if de and de.measurable else "NOT MEASURABLE"
        add("DBNF drift detection accuracy / precision / recall", measured, "ground-truth evaluated", st, "DBNF DRIFT DETECTION EVALUATION", action_for_status(st))
    else:
        add("DBNF drift detection accuracy / precision / recall", "NOT EXERCISED", "ground-truth evaluated", "NOT MEASURABLE", "DBNF DRIFT DETECTION EVALUATION", "not exercised in current run")

    add("trace-pair evidence printed", "printed" if trace_requested else "not requested", "computed trace", "PASS" if trace_requested else "NOT MEASURABLE", "PAIRWISE MERGE EVIDENCE TRACE", "keep claim" if trace_requested else "request --trace_pair and rerun")

    print_table(["paper_claim", "measured_value", "expected_value", "status", "evidence_table", "recommended_paper_action"], rows, "CLAIM SUPPORT SUMMARY")

# -----------------------------
# CLI / Main
# -----------------------------

def parse_trace_pairs(flat: Sequence[Any]) -> List[List[str]]:
    if not flat:
        return []
    if all(isinstance(x, list) for x in flat):
        return list(flat)  # type: ignore
    vals = list(flat)
    return [vals[i:i + 2] for i in range(0, len(vals), 2) if len(vals[i:i + 2]) == 2]

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reviewer-grade SDNF experiment/audit harness v10")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--payloads_dir", default="payloads")
    p.add_argument("--evidence_mode", default="hybrid")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--drift_model", default=None)
    p.add_argument("--ground_truth_aliases", default=None)
    p.add_argument("--drift_ground_truth", default=None)
    p.add_argument("--strict_paper_reproduction", action="store_true")
    p.add_argument("--trace_pair", nargs=2, action="append", default=[])
    p.add_argument("--eenf_g_sweep", default=None, help="comma-separated G values, e.g. 1,10,20")
    p.add_argument("--eenf_repeats", type=int, default=20)
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
        ["drift_model", args.drift_model or "NA"],
        ["data_dir", args.data_dir],
        ["payloads_dir", args.payloads_dir],
        ["context", args.context],
        ["tau_EENF", args.tau_eenf],
        ["tau_AANF", args.tau_aanf],
        ["tau_C MNF", args.tau_cmnf],
        ["tau_DBNF", args.tau_dbnf],
        ["gamma", args.gamma],
        ["m_min", args.m_min],
        ["ground_truth_aliases", args.ground_truth_aliases or "NA"],
        ["drift_ground_truth", args.drift_ground_truth or "NA"],
        ["strict_paper_reproduction", args.strict_paper_reproduction],
    ], "RUN CONFIGURATION")

    print_table(["Metric", "Value"], [
        ["schema_files_ingested", dataset_summary.get("schema_files_ingested")],
        ["payload_files_ingested", dataset_summary.get("payload_files_ingested")],
        ["raw_attribute_records", dataset_summary.get("raw_attribute_records")],
        ["distinct_attribute_names", dataset_summary.get("distinct_attribute_names")],
        ["value_evidence_available", dataset_summary.get("value_evidence_available")],
        ["value_evidence_missing", dataset_summary.get("value_evidence_missing")],
        ["missing_fraction", f"{100.0 * dataset_summary.get('missing_fraction', 0.0):.1f}%"],
    ], "DATASET SUMMARY")

    true_pairs, negative_pairs = load_ground_truth_aliases(args.ground_truth_aliases, args)

    results: List[ModeResult] = []
    for mode in mode_list(args.evidence_mode):
        r = run_mode(attrs, mode, args, embedder)
        if args.drift_model:
            r.drift_hotspots = compute_drift_hotspots(attrs, args, embedder)
            r.nf_metrics["DBNF_status"] = "EVALUATED"
        results.append(r)

    nf_rows = []
    for r in results:
        nf = r.nf_metrics
        nf_rows.extend([
            [r.mode, "EENF", "q95(var) <= tau", f"q95={fmt(nf.get('EENF_q95'))}; max={fmt(nf.get('EENF_max'))}", f"tau={fmt(nf.get('EENF_tau'))}", nf.get("EENF_status")],
            [r.mode, "AANF", "min_merge_sim >= tau", fmt(nf.get("AANF_min_merge_sim")), f"tau={fmt(nf.get('AANF_tau'))}", nf.get("AANF_status")],
            [r.mode, "CMNF", nf.get("CMNF_reason"), fmt(nf.get("CMNF_mean_overlap")), f"tau={fmt(nf.get('CMNF_tau'))}", nf.get("CMNF_status")],
            [r.mode, "ECNF", "min_signals >= m_min", nf.get("ECNF_min_signals"), f"m_min={nf.get('ECNF_m_min')}", nf.get("ECNF_status")],
            [r.mode, "DBNF", "drift model enabled" if args.drift_model else "drift model not enabled", "NA", f"tau={fmt(args.tau_dbnf)}", nf.get("DBNF_status")],
            [r.mode, "RRNF", "not exercised", "NA", "NA", "INFO"],
            [r.mode, "PONF", "not exercised", "NA", "NA", "INFO"],
        ])
    print_table(["mode", "NormalForm", "Rule", "Actual", "Expected", "Status"], nf_rows, "NORMAL FORM VALIDATION SUMMARY")

    alias_evals = {r.mode: evaluate_alias_merges(r.mode, r.predicted_pairs, true_pairs) for r in results}
    print_table(["mode", "predicted_pairs", "true_pairs", "TP", "FP", "FN", "precision", "recall", "F1"], [
        [ae.mode, ae.predicted_pairs, ae.true_pairs if ae.measurable else "NOT MEASURABLE", ae.tp, ae.fp, ae.fn, pct(ae.precision), pct(ae.recall), pct(ae.f1)] for ae in alias_evals.values()
    ], "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")

    leakage_evals = {r.mode: evaluate_cross_context_leakage(r.mode, r.decisions, negative_pairs) for r in results}
    print_table(["mode", "leakage_count", "predicted_merge_count", "leakage_rate", "top_leakage_examples"], [
        [le.mode, le.leakage_count, le.predicted_merge_count, pct(le.leakage_rate), "; ".join(le.examples)] for le in leakage_evals.values()
    ], "CROSS-CONTEXT LEAKAGE EVALUATION")

    print_table(["mode", "predicted_pairs", "canon_final", "input_attributes", "schema_reduction", "precision", "recall", "leakage"], [
        [r.mode, len(r.predicted_pairs), r.canon_final, r.input_attributes, f"{r.schema_reduction_pct:.1f}%", pct(alias_evals[r.mode].precision), pct(alias_evals[r.mode].recall), pct(leakage_evals[r.mode].leakage_rate)] for r in results
    ], "ABLATION STUDY SUMMARY")

    print_table(["approach", "schema_reduction", "merge_precision", "merge_recall", "F1", "cross_context_leakage", "canon_final", "input_attributes", "supported_paper_claim_status"], paper_table_2_rows(results, alias_evals, leakage_evals), "PAPER TABLE 2 REPRODUCTION CHECK")

    eenf_rows = None
    if args.eenf_g_sweep:
        g_values = [int(x.strip()) for x in args.eenf_g_sweep.split(",") if x.strip()]
        eenf_rows = run_eenf_g_sweep(attrs, embedder, g_values, args.eenf_repeats)
        print_table(["G", "mean_variance", "q95_variance", "max_variance", "variance_reduction_vs_G1", "encoding_time_sec", "overhead_vs_G1", "note"], [
            [r["G"], r["mean_variance"], r["q95_variance"], r["max_variance"], pct(r["variance_reduction_vs_G1"]), r["encoding_time_sec"], r["overhead_vs_G1"], r["note"]] for r in eenf_rows
        ], "EENF STABILITY-LATENCY SWEEP")

    if args.measure_timing:
        print_table(["mode", "candidate_pairs", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"], [timing_summary(r.mode, r.decisions) for r in results], "MERGE DECISION TIMING SUMMARY")
        for r in results:
            mean_ms = timing_summary(r.mode, r.decisions)[2]
            status = "PASS" if isinstance(mean_ms, float) and mean_ms < PAPER_CLAIMS["avg_merge_decision_ms"] else "FAIL"
            print(f"Average merge decision under 50ms ({r.mode}): {status}")

    drift_evals = None
    if args.drift_model:
        drift_truth = load_drift_ground_truth(args.drift_ground_truth, args)
        drift_evals = {r.mode: evaluate_drift_detection(r, drift_truth, args.tau_dbnf_drift) for r in results}
        print_table(["mode", "drift_tau", "detected_count", "true_drift_count", "TP", "FP", "FN", "precision", "recall", "F1", "accuracy"], [
            [de.mode, de.drift_tau, de.detected_count, de.true_drift_count if de.measurable else "NOT MEASURABLE", de.tp, de.fp, de.fn, pct(de.precision), pct(de.recall), pct(de.f1), pct(de.accuracy_if_defined)] for de in drift_evals.values()
        ], "DBNF DRIFT DETECTION EVALUATION")
        first = results[0] if results else None
        if first and first.drift_hotspots:
            print_table(["attribute", "drift_distance"], first.drift_hotspots, "DBNF DRIFT HOTSPOTS")

    trace_pairs = parse_trace_pairs(args.trace_pair)
    trace_pairwise_evidence(results, trace_pairs)

    print_claim_support_summary(dataset_summary, results, alias_evals, leakage_evals, eenf_rows, drift_evals, bool(trace_pairs))

if __name__ == "__main__":
    main()
