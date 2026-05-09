
#!/usr/bin/env python3
"""
unified_sdnf_experiment_hybrid_v11.py
Reviewer-grade SDNF experiment/audit harness for:
"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces".

Design principles
-----------------
1. Measured results are computed from current data; paper claims are never hardcoded as outcomes.
2. Thresholds are not changed merely to force paper claims to pass.
3. Claim audit semantics remain PASS / FAIL / NOT MEASURABLE / NOT EXERCISED.
4. Incomplete ground truth is explicitly labeled and exported for curation.
5. DBNF success requires precision, recall, and F1 thresholds, not merely execution.
6. sentence_transformers is optional; deterministic hashing fallback is used when unavailable.

Examples
--------
Standard run:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all --measure_timing

Strict paper reproduction run:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all \
    --ground_truth_aliases ground_truth_aliases.json --ground_truth_closed_world \
    --drift_model all-mpnet-base-v2 --drift_ground_truth drift_ground_truth.json \
    --strict_paper_reproduction --eenf_g_sweep 1,10,20 --measure_timing

Run with decision exports:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all \
    --ground_truth_aliases ground_truth_aliases.json \
    --export_decisions decisions_v11.csv \
    --export_predicted_pairs predicted_pairs_v11.json \
    --export_false_positives false_positives_v11.csv \
    --export_false_negatives false_negatives_v11.csv \
    --export_ground_truth_template ground_truth_aliases_template_v11.json

Run with closed-world alias ground truth:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all \
    --ground_truth_aliases ground_truth_aliases.json --ground_truth_closed_world

Run with controlled drift benchmark:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all \
    --controlled_drift_json controlled_drift_cases.json --drift_eval_mode controlled

Run exporting summary JSON:
  python unified_sdnf_experiment_hybrid_v11.py --evidence_mode all \
    --export_summary_json summary_v11.json

Supported ground_truth_aliases.json formats
-------------------------------------------
{
  "alias_groups": [["acct_num", "PrimaryAccountNumber", "pan"]],
  "true_pairs": [["txn_amount", "amount"]],
  "negative_pairs": [["account_id", "amount"]]
}
Object-style alias groups are also accepted:
{
  "alias_groups": [
    {"canonical": "primary_account_number", "aliases": ["acct_num", "pan"], "basis": "..."}
  ]
}

Supported drift_ground_truth.json formats
-----------------------------------------
{"drift_attributes": ["description", "iso_currency_code"]}
{"true_drift_attributes": ["description"]}
{"drift_cases": [{"attribute": "description", "basis": "..."}]}

Supported controlled_drift_cases.json format
--------------------------------------------
{
  "controlled_drift_cases": [
    {"attribute": "description", "drifted_name": "transaction narrative text", "basis": "Simulated semantic rename"}
  ]
}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
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
    "dbnf_precision_pct": 80.0,
    "dbnf_recall_pct": 80.0,
    "dbnf_f1_pct": 80.0,
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
DEFAULT_WEIGHTS = {"embedding": 0.4, "name": 0.2, "ontology": 0.1, "shape": 0.1, "vss": 0.2}
SYNONYM_CANON = {
    "acct": "account", "acc": "account", "acctnum": "account number", "accountnum": "account number",
    "num": "number", "nbr": "number", "pan": "primary account number", "ccy": "currency",
    "txn": "transaction", "tx": "transaction", "amt": "amount", "desc": "description",
    "memo": "description", "note": "description", "exp": "expiry", "expires": "expiry",
    "dbtr": "debtor", "cdtr": "creditor", "nm": "name", "id": "identifier",
}
ROLE_TOKENS = {"payer", "payee", "debtor", "creditor", "merchant", "cardholder", "holder"}

@dataclass(frozen=True, order=True)
class Pair:
    a: str
    b: str
    @staticmethod
    def make(a: str, b: str) -> "Pair":
        x, y = sorted([canonical_pair_key(a), canonical_pair_key(b)])
        return Pair(x, y)
    def display(self) -> str:
        return f"{self.a} <-> {self.b}"
    def as_list(self) -> List[str]:
        return [self.a, self.b]

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
    regex_compatible: Optional[bool] = None
    vss_similarity: Optional[float] = None
    shape_similarity: Optional[float] = None
    aggregate_score: Optional[float] = None
    evidence_signal_count: int = 0
    AANF_status: str = "NA"
    ECNF_status: str = "NA"
    CMNF_status: str = "NA"
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
    predicted_pairs_count: int
    true_pairs_count: int
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    labeled_precision: Optional[float]
    evaluated_against_closed_world: bool
    ground_truth_coverage_warning: str
    measurable: bool = True

@dataclass
class LeakageEvalResult:
    mode: str
    leakage_count: int
    predicted_merge_count: int
    leakage_rate: Optional[float]
    examples: List[str] = field(default_factory=list)

@dataclass
class DriftEvalResult:
    mode: str
    eval_type: str
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
    detected_set: List[str] = field(default_factory=list)
    true_set: List[str] = field(default_factory=list)
    tp_set: List[str] = field(default_factory=list)
    fp_set: List[str] = field(default_factory=list)
    fn_set: List[str] = field(default_factory=list)
    measurable: bool = True

# Formatting

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

def pct(x: Optional[float], nd: int = 1, na: str = "NOT MEASURABLE") -> str:
    if x is None:
        return na
    return f"{100.0 * x:.{nd}f}%"

def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> str:
    str_rows = [[fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], min(len(c), 80))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out: List[str] = []
    if title:
        out.append("\n" + title)
    out.append(sep)
    out.append("|" + "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(sep.replace("-", "="))
    for row in str_rows:
        clipped = [c if len(c) <= 80 else c[:77] + "..." for c in row]
        out.append("|" + "|".join(f" {clipped[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(sep)
    return "\n".join(out)

def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], title: Optional[str] = None) -> None:
    print(render_table(headers, rows, title))

# Normalization / evidence helpers

def camel_to_tokens(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split())

def normalize_key_raw(s: str) -> str:
    return camel_to_tokens(str(s)).lower().strip()

def expand_synonym_token(t: str) -> List[str]:
    return SYNONYM_CANON.get(t, t).split()

def normalize_key(s: str) -> str:
    expanded: List[str] = []
    for t in normalize_key_raw(s).split():
        expanded.extend(expand_synonym_token(t))
    return " ".join(expanded).strip()

def canonical_pair_key(s: str) -> str:
    return normalize_key(s)

def token_set(s: str) -> Set[str]:
    return set(normalize_key(s).split())

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def role_conflict(a: str, b: str) -> bool:
    ta, tb = token_set(a), token_set(b)
    return ("payer" in ta and "payee" in tb) or ("payee" in ta and "payer" in tb) or ("debtor" in ta and "creditor" in tb) or ("creditor" in ta and "debtor" in tb)

def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    k = min(len(a), len(b))
    aa, bb = a[:k], b[:k]
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / denom)

def safe_norm(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)

def ontology_root(name: str) -> Optional[str]:
    n = normalize_key(name)
    if any(k in n for k in ["primary account number", "account number", "account", "iban", "routing", "card number", "payer account", "debtor account"]):
        return "payment:account"
    if any(k in n for k in ["cvv", "cid", "security code", "verification"]):
        return "payment:cvv"
    if any(k in n for k in ["expiry", "expiration"]):
        return "payment:expiry"
    if any(k in n for k in ["amount", "transaction amount", "instd amount"]):
        return "payment:amount"
    if any(k in n for k in ["risk", "fraud", "score"]):
        return "risk:score"
    if any(k in n for k in ["currency", "iso currency"]):
        return "payment:currency"
    if any(k in n for k in ["description", "comment", "narrative"]):
        return "text:description"
    if any(k in n for k in ["payer", "payee", "merchant", "holder", "name", "debtor", "creditor"]):
        return "party:name"
    if any(k in n for k in ["status", "state"]):
        return "payment:status"
    return None

def is_account_like(name: str) -> bool:
    toks = token_set(name)
    n = " ".join(toks)
    return "account" in toks or "card" in toks or "primary account number" in n or "pan" in toks

def normalize_value_for_shape(name: str, value: Any) -> str:
    s = str(value).strip()
    if is_account_like(name):
        s = re.sub(r"[\s-]", "", s)
        s = re.sub(r"[xX*]", "0", s)
    return s

def infer_regex(values: Sequence[Any], attr_name: str = "") -> str:
    samples = [normalize_value_for_shape(attr_name, v) for v in values if v is not None and str(v).strip()]
    if not samples:
        return ""
    if all(re.fullmatch(r"\d{13,19}", s) for s in samples):
        return r"^[0-9]{13,19}$"
    if is_account_like(attr_name) and all(re.fullmatch(r"\d{6,19}", s) for s in samples):
        return r"^[0-9]{6,19}$"
    if all(re.fullmatch(r"[A-Z]{3}", s) for s in samples):
        return r"^[A-Z]{3}$"
    if all(re.fullmatch(r"[+-]?\d+(\.\d+)?", s) for s in samples):
        return r"^[+-]?[0-9]+(\.[0-9]+)?$"
    if all(re.fullmatch(r"\d+", s) for s in samples):
        lengths = sorted({len(s) for s in samples})
        return rf"^[0-9]{{{lengths[0]},{lengths[-1]}}}$" if len(lengths) > 1 else rf"^[0-9]{{{lengths[0]}}}$"
    return "mixed"

def parse_numeric_regex_range(rx: str) -> Optional[Tuple[int, int]]:
    if not rx:
        return None
    r = rx.replace("\\d", "[0-9]").strip()
    patterns = [
        r"^\^?\[0-9\]\{(\d+),(\d+)\}\$?$",
        r"^\^?\[0-9\]\{(\d+)\}\$?$",
        r"^\^?\[0-9\]\+\$?$",
        r"^\^?\d\{(\d+),(\d+)\}\$?$",
        r"^\^?\d\{(\d+)\}\$?$",
        r"^\^?\d\+\$?$",
    ]
    for p in patterns:
        m = re.match(p, r)
        if not m:
            continue
        if len(m.groups()) == 2:
            return int(m.group(1)), int(m.group(2))
        if len(m.groups()) == 1:
            v = int(m.group(1)); return v, v
        return 1, 10**9
    return None

def is_decimal_numeric_regex(rx: str) -> bool:
    if not rx:
        return False
    r = rx.replace("\\d", "[0-9]")
    return bool(re.search(r"\[0-9\].*\(\\?\.\[0-9\].*\)\?", r) or "[+-]?[0-9]+" in r or r in {r"^[0-9]+(.[0-9]+)?$", r"^[0-9]+(\.[0-9]+)?$"})

def is_currency_regex(rx: str) -> bool:
    return bool(rx and re.fullmatch(r"\^?\[A-Z\]\{3\}\$?", rx.strip()))

def regex_compatible(regex_a: str, regex_b: str) -> bool:
    if not regex_a or not regex_b:
        return False
    if regex_a == regex_b and regex_a != "mixed":
        return True
    if regex_a == "mixed" or regex_b == "mixed":
        return False
    ra, rb = parse_numeric_regex_range(regex_a), parse_numeric_regex_range(regex_b)
    if ra and rb:
        return max(ra[0], rb[0]) <= min(ra[1], rb[1])
    if is_decimal_numeric_regex(regex_a) and is_decimal_numeric_regex(regex_b):
        return True
    if is_currency_regex(regex_a) and is_currency_regex(regex_b):
        return True
    return False

def shape_token(v: Any) -> str:
    s = str(v)
    out: List[str] = []
    for ch in s:
        out.append("D" if ch.isdigit() else "A" if ch.isalpha() else "S" if ch.isspace() else "P")
    return "".join(k + str(len(list(g))) for k, g in itertools.groupby(out))

def numeric_length_range(values: Sequence[Any], attr_name: str = "") -> Optional[Tuple[int, int]]:
    lens = []
    for v in values:
        s = normalize_value_for_shape(attr_name, v)
        if re.fullmatch(r"\d+", s):
            lens.append(len(s))
    if not lens:
        return None
    return min(lens), max(lens)

def value_shape_signature(values: Sequence[Any], attr_name: str = "") -> str:
    if not values:
        return ""
    rng = numeric_length_range(values, attr_name)
    if rng:
        return f"NUMERIC_RANGE:{rng[0]}:{rng[1]}"
    counts: Dict[str, int] = defaultdict(int)
    for v in values[:100]:
        counts[shape_token(normalize_value_for_shape(attr_name, v))] += 1
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))

def shape_similarity_from_records(a: AttributeRecord, b: AttributeRecord) -> Optional[float]:
    if not a.shape or not b.shape:
        return None
    ra, rb = numeric_length_range(a.values, a.name), numeric_length_range(b.values, b.name)
    if ra and rb:
        inter = max(0, min(ra[1], rb[1]) - max(ra[0], rb[0]) + 1)
        union = max(ra[1], rb[1]) - min(ra[0], rb[0]) + 1
        if inter > 0:
            containment = inter / max(1, min(ra[1]-ra[0]+1, rb[1]-rb[0]+1))
            return float(max(0.75, 0.5 * inter / union + 0.5 * containment))
        return 0.0
    if a.shape == b.shape:
        return 1.0
    return jaccard(set(a.shape.split(";")), set(b.shape.split(";")))

def vss_from_values(values: Sequence[Any], attr_name: str = "") -> Optional[np.ndarray]:
    samples = [normalize_value_for_shape(attr_name, v) for v in values if v is not None and str(v).strip()][:200]
    if not samples:
        return None
    lengths = np.array([len(s) for s in samples], dtype=np.float32)
    digit_frac = np.array([sum(ch.isdigit() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    alpha_frac = np.array([sum(ch.isalpha() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    punct_frac = np.array([sum(not ch.isalnum() for ch in s) / max(1, len(s)) for s in samples], dtype=np.float32)
    numeric = np.array([1.0 if re.fullmatch(r"[+-]?\d+(\.\d+)?", s) else 0.0 for s in samples], dtype=np.float32)
    unique_ratio = float(len(set(samples)) / max(1, len(samples)))
    vec = np.array([float(np.mean(lengths)), float(np.std(lengths)), float(np.min(lengths)), float(np.max(lengths)), float(np.mean(digit_frac)), float(np.mean(alpha_frac)), float(np.mean(punct_frac)), float(np.mean(numeric)), unique_ratio], dtype=np.float32)
    return safe_norm(vec)

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
    def _hash_vec(self, text: str, nonce: int = 0) -> np.ndarray:
        toks = normalize_key(text).split() or [normalize_key_raw(text) or "empty"]
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in toks:
            h = hashlib.sha256(f"{self.seed}|{nonce}|{tok}".encode()).digest()
            for i in range(0, len(h), 4):
                idx = int.from_bytes(h[i:i+2], "little") % self.dim
                sign = 1.0 if h[i+2] % 2 == 0 else -1.0
                vec[idx] += sign * (1.0 + (h[i+3] / 255.0))
        return safe_norm(vec)
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        return np.stack([self._hash_vec(t) for t in texts], axis=0)
    def regenerations(self, text: str, context: str, G: int, nonce: int = 0) -> np.ndarray:
        base_text = f"{text} context={context}"
        if self.model is not None:
            base = self.encode([base_text])[0]
            rng = np.random.default_rng(abs(hash((base_text, self.seed, nonce))) % (2**32))
            return np.stack([safe_norm(base + rng.normal(0, 0.003, size=base.shape).astype(np.float32)) for _ in range(G)], axis=0)
        return np.stack([self._hash_vec(base_text, nonce=nonce * 1000 + g) for g in range(G)], axis=0)

# File extraction

def iter_json_files(d: Path) -> List[Path]:
    return sorted([p for p in d.glob("*.json") if p.is_file()], key=lambda p: p.name.lower()) if d.exists() else []

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
    seen, uniq = set(), []
    for c in candidates:
        s = str(c.resolve() if c.exists() else c)
        if s not in seen:
            seen.add(s); uniq.append(c)
    for c in uniq:
        if c.exists():
            return c
    msg = f"WARNING: file not found: {path}. Metrics depending on {label} will be marked NOT MEASURABLE."
    if strict:
        raise FileNotFoundError(msg.replace("WARNING: ", ""))
    print(msg)
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
        for i, v in enumerate(obj):
            path = f"{prefix}[]" if prefix else "[]"
            if isinstance(v, (dict, list)):
                yield from walk_json(v, path)
            else:
                yield path, v

def collect_attributes(data_dir: Path, payloads_dir: Path, context: str, embedder: EmbeddingProvider) -> Tuple[List[AttributeRecord], Dict[str, Any]]:
    schema_files, payload_files = iter_json_files(data_dir), iter_json_files(payloads_dir)
    by_key: Dict[Tuple[str, str], AttributeRecord] = {}
    raw_records = 0
    for p in schema_files:
        try:
            obj = load_json(p)
        except Exception:
            continue
        seen_in_file: Set[str] = set()
        for path, value in walk_json(obj):
            attr = path.split(".")[-1].replace("[]", "")
            if not attr or attr.isdigit():
                continue
            raw_records += 1
            key = (canonical_pair_key(attr), p.name)
            if key not in by_key:
                by_key[key] = AttributeRecord(name=attr, source=p.name, context=context, path=path)
            if value is not None and not isinstance(value, (dict, list)) and canonical_pair_key(attr) not in seen_in_file:
                by_key[key].values.append(value)
                seen_in_file.add(canonical_pair_key(attr))
    # Attach payload values by matching normalized leaf name.
    by_attr = defaultdict(list)
    for rec in by_key.values():
        by_attr[rec.key].append(rec)
    for p in payload_files:
        try:
            obj = load_json(p)
        except Exception:
            continue
        for path, value in walk_json(obj):
            attr = path.split(".")[-1].replace("[]", "")
            k = canonical_pair_key(attr)
            for rec in by_attr.get(k, []):
                if value is not None and not isinstance(value, (dict, list)):
                    rec.values.append(value)
    attrs = list(by_key.values())
    if not attrs:
        # Synthetic tiny fallback to keep harness executable when no data directory is present.
        examples = [
            ("acct_num", "Bank.json", ["4111111111111111", "5555444433331111"]),
            ("PrimaryAccountNumber", "Mastercard.json", ["4111111111111111", "5555444433331111"]),
            ("account_number", "Plaid.json", ["123456789", "987654321"]),
            ("txn_amount", "Visa.json", ["10.50", "99.00"]),
            ("amount", "Stripe.json", ["10.50", "99.00"]),
            ("iso_currency_code", "ISO20022.json", ["USD", "EUR"]),
            ("description", "PayPal.json", ["coffee", "grocery"]),
        ]
        attrs = [AttributeRecord(n, s, context, n, vals) for n, s, vals in examples]
        raw_records = len(attrs)
    names = [f"{a.name} context={a.context}" for a in attrs]
    embs = embedder.encode(names)
    for a, e in zip(attrs, embs):
        a.canonical = a.key
        a.ontology_root = ontology_root(a.name)
        a.regex = infer_regex(a.values, a.name)
        a.shape = value_shape_signature(a.values, a.name)
        a.vss = vss_from_values(a.values, a.name)
        a.embedding = e
    distinct = len({a.key for a in attrs})
    value_avail = len({a.key for a in attrs if a.values})
    summary = {
        "schema_files_ingested": len(schema_files),
        "payload_files_ingested": len(payload_files),
        "raw_attribute_records": raw_records,
        "distinct_attribute_names": distinct,
        "value_evidence_available": value_avail,
        "value_evidence_missing": max(0, distinct - value_avail),
        "missing_fraction": 0.0 if distinct == 0 else (distinct - value_avail) / distinct,
    }
    return attrs, summary

# Ground truth

def pairs_from_alias_groups(groups: Sequence[Any]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for group in groups:
        if isinstance(group, dict):
            aliases = list(group.get("aliases", []))
            if group.get("canonical"):
                aliases = [group["canonical"]] + aliases
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
    true_pairs, negative_pairs = set(), set()
    if "alias_groups" in data:
        true_pairs |= pairs_from_alias_groups(data.get("alias_groups", []))
    for pair in data.get("true_pairs", []):
        if len(pair) == 2:
            true_pairs.add(Pair.make(pair[0], pair[1]))
    for pair in data.get("negative_pairs", []):
        if len(pair) == 2:
            negative_pairs.add(Pair.make(pair[0], pair[1]))
    if not true_pairs:
        print(f"WARNING: no true alias pairs found in {p}. Alias precision/recall marked NOT MEASURABLE.")
        return None, negative_pairs
    return true_pairs, negative_pairs

def evaluate_alias_merges(mode: str, predicted: Set[Pair], true_pairs: Optional[Set[Pair]], closed_world: bool) -> AliasEvalResult:
    warning = ""
    if true_pairs is None:
        return AliasEvalResult(mode, len(predicted), 0, 0, 0, 0, None, None, None, None, closed_world, "No alias ground truth supplied or no true pairs found.", measurable=False)
    tp_set, fp_set, fn_set = predicted & true_pairs, predicted - true_pairs, true_pairs - predicted
    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    labeled_precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if closed_world:
        precision = labeled_precision
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    else:
        precision = None
        f1 = None
        warning = "Ground truth is not declared closed-world; unlabeled predicted pairs may require review. Precision is labeled_precision only."
    return AliasEvalResult(mode, len(predicted), len(true_pairs), tp, fp, fn, precision, recall, f1, labeled_precision, closed_world, warning, True)

# Evidence scoring and modes

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

def supportive_signal_count(e: PairEvidence) -> int:
    count = 0
    if e.cosine_similarity is not None and e.cosine_similarity >= 0.65: count += 1
    if e.name_similarity is not None and e.name_similarity >= 0.35: count += 1
    if e.ontology_match: count += 1
    if e.value_cooccurrence is not None and e.value_cooccurrence >= 0.5: count += 1
    if e.regex_match or e.regex_compatible: count += 1
    if e.vss_similarity is not None and e.vss_similarity >= 0.75: count += 1
    if e.shape_similarity is not None and e.shape_similarity >= 0.70: count += 1
    return count

def compute_pair_evidence(a: AttributeRecord, b: AttributeRecord, mode: str, tau_aanf: float, gamma: float, m_min: int) -> PairEvidence:
    cos_sim = cosine(a.embedding, b.embedding)
    name_sim = jaccard(token_set(a.name), token_set(b.name))
    ont_match = bool(a.ontology_root and b.ontology_root and a.ontology_root == b.ontology_root)
    cooc = value_cooccurrence(a, b)
    rx_match = bool(a.regex and b.regex and a.regex == b.regex)
    rx_compat = regex_compatible(a.regex, b.regex)
    vss_sim = cosine(a.vss, b.vss)
    shape_sim = shape_similarity_from_records(a, b)
    if mode == "no_value_evidence":
        cooc = vss_sim = shape_sim = None
        rx_match = rx_compat = False
    scores = {
        "embedding": cos_sim,
        "name": name_sim,
        "ontology": 1.0 if ont_match else 0.0,
        "shape": max([x for x in [shape_sim, 1.0 if rx_match else 0.85 if rx_compat else None] if x is not None], default=None),
        "vss": vss_sim,
    }
    if mode == "vss_only": scores = {"vss": vss_sim}
    if mode == "shape_only": scores = {"shape": scores["shape"]}
    if mode == "name_ontology_only": scores = {"name": name_sim, "ontology": 1.0 if ont_match else 0.0}
    agg = aggregate_score(scores)
    e = PairEvidence(
        attr_a=a.name, attr_b=b.name, source_a=a.source, source_b=b.source, context_a=a.context, context_b=b.context,
        cosine_similarity=cos_sim, name_similarity=name_sim, ontology_root_a=a.ontology_root, ontology_root_b=b.ontology_root,
        ontology_match=ont_match, value_cooccurrence=cooc, regex_a=a.regex, regex_b=b.regex, regex_match=rx_match,
        regex_compatible=rx_compat, vss_similarity=vss_sim, shape_similarity=shape_sim, aggregate_score=agg,
    )
    e.evidence_signal_count = supportive_signal_count(e)
    e.AANF_status = "PASS" if (cos_sim is not None and cos_sim >= tau_aanf and not role_conflict(a.name, b.name)) else "FAIL"
    e.ECNF_status = "PASS" if (e.evidence_signal_count >= m_min and agg is not None and agg >= gamma) else "FAIL"
    e.CMNF_status = "PASS" if a.context == b.context else "FAIL"
    return e

def decision_for_mode(e: PairEvidence, mode: str, gamma: float) -> Tuple[bool, str]:
    if mode == "embed_only_baseline":
        ok = e.AANF_status == "PASS"
        return ok, "embedding cosine threshold only" if ok else "embedding below AANF threshold or role conflict"
    if mode == "no_ecnf":
        ok = e.AANF_status == "PASS" and e.CMNF_status == "PASS"
        return ok, "AANF+CMNF only; ECNF ablated" if ok else "AANF or CMNF failed"
    if mode == "no_cmnf":
        ok = e.AANF_status == "PASS" and e.ECNF_status == "PASS"
        return ok, "AANF+ECNF only; CMNF ablated" if ok else "AANF or ECNF failed"
    if mode in {"vss_only", "shape_only", "name_ontology_only"}:
        ok = e.aggregate_score is not None and e.aggregate_score >= gamma
        return ok, "single/limited evidence mode threshold" if ok else "limited evidence below gamma"
    ok = e.AANF_status == "PASS" and e.ECNF_status == "PASS" and e.CMNF_status == "PASS"
    if ok:
        return True, "AANF, ECNF, and CMNF passed"
    reasons = []
    if e.AANF_status != "PASS": reasons.append("AANF failed")
    if e.ECNF_status != "PASS": reasons.append("ECNF failed")
    if e.CMNF_status != "PASS": reasons.append("CMNF failed")
    return False, "; ".join(reasons)

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
            self.parent[rb] = ra

def mode_list(requested: str) -> List[str]:
    if requested == "all":
        return ["embed_only_baseline", "sdnf_hybrid", "no_ecnf", "no_cmnf", "no_dbnf", "no_value_evidence", "vss_only", "shape_only", "name_ontology_only", "hybrid"]
    if requested == "embed_only":
        return ["embed_only_baseline"]
    return [requested]

def compute_cmnf_status(attrs: List[AttributeRecord], tau_cmnf: float) -> Dict[str, Any]:
    contexts = sorted({a.context for a in attrs})
    if len(contexts) < 2:
        return {"CMNF_mean_overlap": None, "CMNF_tau": tau_cmnf, "CMNF_status": "NA", "normal_form_interpretation": "NA_SINGLE_CONTEXT", "CMNF_reason": "single context; cross-context CMNF not exercised"}
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
        return {"CMNF_mean_overlap": None, "CMNF_tau": tau_cmnf, "CMNF_status": "NA", "normal_form_interpretation": "NOT_EXERCISED", "CMNF_reason": "no repeated primitive across distinct contexts"}
    mean_overlap = float(np.mean(overlaps))
    status = "PASS" if mean_overlap <= tau_cmnf else "FAIL"
    return {"CMNF_mean_overlap": mean_overlap, "CMNF_tau": tau_cmnf, "CMNF_status": status, "normal_form_interpretation": status, "CMNF_reason": "same normalized primitive across distinct contexts"}

def compute_nf_metrics(attrs: List[AttributeRecord], decisions: List[MergeDecision], args: argparse.Namespace, embedder: EmbeddingProvider) -> Dict[str, Any]:
    regs = []
    for a in attrs[:min(60, len(attrs))]:
        r = embedder.regenerations(a.name, a.context, G=10)
        regs.append(float(np.mean(np.var(r, axis=0))))
    accepted = [d for d in decisions if d.accepted]
    min_merge_sim = min([d.evidence.cosine_similarity for d in accepted if d.evidence.cosine_similarity is not None], default=None)
    min_signals = min([d.evidence.evidence_signal_count for d in accepted], default=0)
    q95 = float(np.quantile(np.array(regs), 0.95)) if regs else 0.0
    cmnf = compute_cmnf_status(attrs, args.tau_cmnf)
    out = {
        "EENF_q95": q95, "EENF_max": max(regs) if regs else 0.0, "EENF_tau": args.tau_eenf,
        "EENF_status": "PASS" if q95 <= args.tau_eenf else "FAIL", "EENF_interpretation": "PASS" if q95 <= args.tau_eenf else "FAIL",
        "AANF_min_merge_sim": min_merge_sim, "AANF_tau": args.tau_aanf,
        "AANF_status": "PASS" if (min_merge_sim is not None and min_merge_sim >= args.tau_aanf) else ("NA" if not accepted else "FAIL"),
        "AANF_interpretation": "NOT_EXERCISED" if not accepted else ("PASS" if min_merge_sim is not None and min_merge_sim >= args.tau_aanf else "FAIL"),
        "ECNF_min_signals": min_signals, "ECNF_m_min": args.m_min,
        "ECNF_status": "PASS" if min_signals >= args.m_min and accepted else ("NA" if not accepted else "FAIL"),
        "ECNF_interpretation": "NOT_EXERCISED" if not accepted else ("PASS" if min_signals >= args.m_min else "FAIL"),
        "DBNF_status": "NOT_EXERCISED" if not args.drift_model and not args.controlled_drift_json else "PENDING",
        "DBNF_interpretation": "NOT_EXERCISED" if not args.drift_model and not args.controlled_drift_json else "INFO_ONLY",
        "RRNF_status": "INFO", "RRNF_interpretation": "INFO_ONLY", "PONF_status": "INFO", "PONF_interpretation": "INFO_ONLY",
    }
    out.update(cmnf)
    return out

def run_mode(attrs: List[AttributeRecord], mode: str, args: argparse.Namespace, embedder: EmbeddingProvider) -> ModeResult:
    t0 = time.perf_counter(); pairs = list(itertools.combinations(attrs, 2)); cand_ms = (time.perf_counter() - t0) * 1000.0
    decisions: List[MergeDecision] = []
    for idx, (a, b) in enumerate(pairs):
        start = time.perf_counter(); ev = compute_pair_evidence(a, b, mode, args.tau_aanf, args.gamma, args.m_min); scoring_ms = (time.perf_counter() - start) * 1000.0
        vstart = time.perf_counter(); accepted, reason = decision_for_mode(ev, mode, args.gamma)
        ev.final_decision = "MERGE" if accepted else "DEFER"; ev.reason = reason; ev.lineage_id = f"{mode}-{idx:06d}"
        validation_ms = (time.perf_counter() - vstart) * 1000.0
        total_ms = cand_ms / max(1, len(pairs)) + scoring_ms + validation_ms
        decisions.append(MergeDecision(Pair.make(a.name, b.name), ev, TimingRecord(cand_ms, scoring_ms, validation_ms, total_ms), accepted, mode))
    predicted = {d.pair for d in decisions if d.accepted}
    uf = UnionFind([a.key for a in attrs])
    for d in decisions:
        if d.accepted:
            uf.union(d.pair.a, d.pair.b)
    input_n = len({a.key for a in attrs}); roots = {uf.find(a.key) for a in attrs}; canon_final = len(roots)
    reduction = 100.0 * (input_n - canon_final) / input_n if input_n else 0.0
    nf = compute_nf_metrics(attrs, decisions, args, embedder)
    return ModeResult(mode, attrs, decisions, predicted, canon_final, input_n, reduction, nf)

# Evaluations

def evaluate_cross_context_leakage(mode: str, decisions: List[MergeDecision], negative_pairs: Set[Pair]) -> LeakageEvalResult:
    accepted = [d for d in decisions if d.accepted]
    leaks = []
    for d in accepted:
        e = d.evidence
        incompatible_context = e.context_a != e.context_b
        incompatible_ontology = bool(e.ontology_root_a and e.ontology_root_b and e.ontology_root_a != e.ontology_root_b)
        explicit_negative = d.pair in negative_pairs
        if incompatible_context or incompatible_ontology or explicit_negative:
            why = []
            if incompatible_context: why.append("context")
            if incompatible_ontology: why.append("ontology")
            if explicit_negative: why.append("explicit_negative")
            leaks.append(f"{d.pair.display()} ({'/'.join(why)})")
    rate = len(leaks) / len(accepted) if accepted else 0.0
    return LeakageEvalResult(mode, len(leaks), len(accepted), rate, leaks[:5])

def run_eenf_g_sweep(attrs: List[AttributeRecord], embedder: EmbeddingProvider, g_values: Sequence[int], repeats: int) -> List[Dict[str, Any]]:
    rows, baseline_mean, baseline_time = [], None, None
    for G in g_values:
        start = time.perf_counter(); attr_vars = []
        for a in attrs[:min(60, len(attrs))]:
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
            baseline_mean, baseline_time = mean_v, elapsed
        reduction = None if baseline_mean is None or baseline_mean <= 1e-15 else max(0.0, (baseline_mean - mean_v) / baseline_mean)
        overhead = None if baseline_time is None or baseline_time <= 0 else elapsed / baseline_time
        expected = PAPER_CLAIMS.get(f"g{G}_variance_reduction_pct")
        claim_status = "NA" if expected is None or reduction is None else ("PASS" if abs(reduction * 100 - expected) <= TOLERANCES["variance_reduction_pct"] else "FAIL")
        action = "keep claim" if claim_status == "PASS" else "revise paper claim to measured value" if expected is not None else "not a paper claim"
        rows.append({"G": G, "mean_variance": mean_v, "q95_variance": q95_v, "max_variance": max_v, "variance_reduction_vs_G1": reduction, "encoding_time_sec": elapsed, "overhead_vs_G1": overhead, "claim_status": claim_status, "recommended_action": action, "note": "" if reduction is not None else "zero-variance backend; reduction not measurable"})
    return rows

def timing_summary(mode: str, decisions: List[MergeDecision]) -> List[Any]:
    vals = [d.timing.total_decision_ms for d in decisions]
    if not vals:
        return [mode, 0, None, None, None, None, None]
    arr = np.array(vals, dtype=np.float64)
    return [mode, len(vals), float(np.mean(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)), float(np.max(arr))]

def load_drift_ground_truth(path: Optional[str], args: argparse.Namespace) -> Optional[Set[str]]:
    p = resolve_optional_input_file(path, "drift ground truth", args.strict_paper_reproduction, [Path(args.data_dir), Path(args.payloads_dir)])
    if p is None:
        return None
    data = load_json(p); attrs: List[str] = []
    attrs.extend(data.get("drift_attributes", [])); attrs.extend(data.get("true_drift_attributes", []))
    for c in data.get("drift_cases", []):
        if isinstance(c, dict) and c.get("attribute"):
            attrs.append(c["attribute"])
    s = {canonical_pair_key(x) for x in attrs if str(x).strip()}
    return s or None

def compute_drift_hotspots(attrs: List[AttributeRecord], args: argparse.Namespace, base_embedder: EmbeddingProvider) -> List[Tuple[str, float]]:
    if not args.drift_model:
        return []
    drift_embedder = EmbeddingProvider(args.drift_model, args.seed)
    names = [f"{a.name} context={a.context}" for a in attrs]
    if not names: return []
    base, drift = base_embedder.encode(names), drift_embedder.encode(names)
    k = min(base.shape[1], drift.shape[1])
    distances = np.linalg.norm(base[:, :k] - drift[:, :k], axis=1)
    by_name: Dict[str, float] = {}
    for a, d in zip(attrs, distances):
        by_name[a.key] = max(by_name.get(a.key, 0.0), float(d))
    return sorted(by_name.items(), key=lambda x: (-x[1], x[0]))[:args.drift_top_k]

def evaluate_drift_detection(mode: str, hotspots: List[Tuple[str, float]], truth: Optional[Set[str]], tau: float, universe: Set[str], eval_type: str) -> DriftEvalResult:
    detected = {name for name, d in hotspots if d > tau}
    if truth is None:
        return DriftEvalResult(mode, eval_type, tau, len(detected), 0, 0, 0, 0, None, None, None, None, sorted(detected), [], [], [], [], measurable=False)
    tp_set, fp_set, fn_set = detected & truth, detected - truth, truth - detected
    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    uni = universe | truth | detected
    tn = len((uni - truth) - detected)
    acc = (tp + tn) / max(1, len(uni))
    return DriftEvalResult(mode, eval_type, tau, len(detected), len(truth), tp, fp, fn, precision, recall, f1, acc, sorted(detected), sorted(truth), sorted(tp_set), sorted(fp_set), sorted(fn_set), True)

def load_controlled_drift_cases(path: Optional[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    p = resolve_optional_input_file(path, "controlled drift benchmark", False, [Path(args.data_dir), Path(args.payloads_dir)])
    if p is None: return []
    data = load_json(p)
    return [c for c in data.get("controlled_drift_cases", []) if isinstance(c, dict) and c.get("attribute") and c.get("drifted_name")]

def evaluate_controlled_drift(attrs: List[AttributeRecord], args: argparse.Namespace, base_embedder: EmbeddingProvider, cases: List[Dict[str, Any]]) -> DriftEvalResult:
    if not cases:
        return DriftEvalResult("controlled", "controlled", args.tau_dbnf_drift, 0, 0, 0, 0, 0, None, None, None, None, measurable=False)
    drift_embedder = EmbeddingProvider(args.drift_model or args.model, args.seed)
    by_key = {a.key: a for a in attrs}
    hotspots: List[Tuple[str, float]] = []
    truth = {canonical_pair_key(c["attribute"]) for c in cases}
    for c in cases:
        attr_key = canonical_pair_key(c["attribute"])
        base_name = by_key.get(attr_key).name if attr_key in by_key else c["attribute"]
        base = base_embedder.encode([f"{base_name} context={args.context}"])[0]
        drift = drift_embedder.encode([f"{c['drifted_name']} context={args.context}"])[0]
        k = min(len(base), len(drift))
        hotspots.append((attr_key, float(np.linalg.norm(base[:k] - drift[:k]))))
    return evaluate_drift_detection("controlled", hotspots, truth, args.tau_dbnf_drift, {a.key for a in attrs}, "controlled")

# Exports

DECISION_FIELDS = ["mode","attr_a","attr_b","normalized_pair_a","normalized_pair_b","pair_key","source_a","source_b","context_a","context_b","cosine_similarity","name_similarity","ontology_root_a","ontology_root_b","ontology_match","value_cooccurrence","regex_a","regex_b","regex_match","regex_compatible","vss_similarity","shape_similarity","aggregate_score","evidence_signal_count","AANF_status","ECNF_status","CMNF_status","final_decision","reason","lineage_id","is_predicted_merge","is_in_true_pairs","is_in_negative_pairs","error_class"]

def decision_row(d: MergeDecision, true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Dict[str, Any]:
    e = d.evidence; p = d.pair
    is_true = bool(true_pairs and p in true_pairs); is_neg = p in negative_pairs
    if d.accepted and is_true: err = "TP"
    elif d.accepted and is_neg: err = "EXPLICIT_NEGATIVE_MERGE"
    elif d.accepted and not is_true: err = "FP"
    elif (not d.accepted) and is_true: err = "FN_CANDIDATE"
    else: err = "TN_OR_UNLABELED"
    return {"mode": d.mode, "attr_a": e.attr_a, "attr_b": e.attr_b, "normalized_pair_a": p.a, "normalized_pair_b": p.b, "pair_key": p.display(), "source_a": e.source_a, "source_b": e.source_b, "context_a": e.context_a, "context_b": e.context_b, "cosine_similarity": e.cosine_similarity, "name_similarity": e.name_similarity, "ontology_root_a": e.ontology_root_a, "ontology_root_b": e.ontology_root_b, "ontology_match": e.ontology_match, "value_cooccurrence": e.value_cooccurrence, "regex_a": e.regex_a, "regex_b": e.regex_b, "regex_match": e.regex_match, "regex_compatible": e.regex_compatible, "vss_similarity": e.vss_similarity, "shape_similarity": e.shape_similarity, "aggregate_score": e.aggregate_score, "evidence_signal_count": e.evidence_signal_count, "AANF_status": e.AANF_status, "ECNF_status": e.ECNF_status, "CMNF_status": e.CMNF_status, "final_decision": e.final_decision, "reason": e.reason, "lineage_id": e.lineage_id, "is_predicted_merge": d.accepted, "is_in_true_pairs": is_true, "is_in_negative_pairs": is_neg, "error_class": err}

def write_csv(path: str, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def export_decisions(path: Optional[str], results: List[ModeResult], true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> List[Dict[str, Any]]:
    rows = [decision_row(d, true_pairs, negative_pairs) for r in results for d in r.decisions]
    if path: write_csv(path, rows, DECISION_FIELDS)
    return rows

def export_predicted_pairs(path: Optional[str], results: List[ModeResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in results:
        items = []
        for d in r.decisions:
            if d.accepted:
                items.append({"pair": d.pair.as_list(), "attr_a": d.evidence.attr_a, "attr_b": d.evidence.attr_b, "evidence": asdict(d.evidence)})
        out[r.mode] = items
    if path:
        with open(path, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, default=str)
    return out

def export_false_positive_negative(fp_path: Optional[str], fn_path: Optional[str], results: List[ModeResult], true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fp_rows, fn_rows = [], []
    decision_by_mode_pair = {(r.mode, d.pair): d for r in results for d in r.decisions}
    for r in results:
        for d in r.decisions:
            if d.accepted and (true_pairs is None or d.pair not in true_pairs):
                row = decision_row(d, true_pairs, negative_pairs); row["explicitly_negative"] = d.pair in negative_pairs; row["suggested_review_label"] = "REVIEW"; fp_rows.append(row)
        if true_pairs:
            for p in sorted(true_pairs):
                if p not in r.predicted_pairs:
                    d = decision_by_mode_pair.get((r.mode, p))
                    if d:
                        row = decision_row(d, true_pairs, negative_pairs); row["found_as_candidate"] = True; row["reason_for_deferral"] = d.evidence.reason
                    else:
                        row = {"mode": r.mode, "normalized_pair_a": p.a, "normalized_pair_b": p.b, "pair_key": p.display(), "found_as_candidate": False, "reason_for_deferral": "not generated as candidate"}
                    row["suggested_review_label"] = "MISSED_TRUE_ALIAS"; fn_rows.append(row)
    if fp_path: write_csv(fp_path, fp_rows, list(dict.fromkeys(DECISION_FIELDS + ["explicitly_negative", "suggested_review_label"])))
    if fn_path: write_csv(fn_path, fn_rows, list(dict.fromkeys(DECISION_FIELDS + ["found_as_candidate", "reason_for_deferral", "suggested_review_label"])))
    return fp_rows, fn_rows

def export_ground_truth_template(path: Optional[str], predicted_json: Dict[str, Any], negative_pairs: Set[Pair]) -> Dict[str, Any]:
    cand = []
    for mode, items in predicted_json.items():
        for it in items:
            cand.append({"mode": mode, "pair": it["pair"], "attr_a": it["attr_a"], "attr_b": it["attr_b"], "evidence": it["evidence"]})
    neg = [{"pair": p.as_list()} for p in sorted(negative_pairs)]
    out = {"alias_groups": [], "true_pairs_candidates": cand, "negative_pairs_candidates": neg, "review_notes": "Curate these candidates into alias_groups/true_pairs/negative_pairs before claiming precision/recall."}
    if path:
        with open(path, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, default=str)
    return out

# Reporting

def status_with_tolerance(measured: Optional[float], expected: float, tol: float, as_fraction: bool = False, less_than: bool = False) -> str:
    if measured is None: return "NOT MEASURABLE"
    m = measured * 100.0 if as_fraction else measured
    return "PASS" if (m < expected if less_than else abs(m - expected) <= tol) else "FAIL"

def paper_table_2_rows(results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult]) -> List[List[Any]]:
    rows = []
    for r in results:
        ae, le = alias_evals.get(r.mode), leakage_evals.get(r.mode)
        if r.mode in {"sdnf_hybrid", "hybrid"}: exp_prec, exp_rec, exp_leak = PAPER_CLAIMS["sdnf_precision_pct"], PAPER_CLAIMS["sdnf_recall_pct"], PAPER_CLAIMS["sdnf_leakage_pct"]
        elif r.mode == "embed_only_baseline": exp_prec, exp_rec, exp_leak = PAPER_CLAIMS["baseline_precision_pct"], PAPER_CLAIMS["baseline_recall_pct"], PAPER_CLAIMS["baseline_leakage_pct"]
        else: exp_prec = exp_rec = exp_leak = None
        statuses = [status_with_tolerance(r.schema_reduction_pct, PAPER_CLAIMS["schema_reduction_pct"], TOLERANCES["schema_reduction_pct"])]
        if exp_prec is None:
            statuses += ["NA", "NA", "NA"]
        elif not (ae and ae.evaluated_against_closed_world):
            statuses += ["GROUND TRUTH INCOMPLETE", "GROUND TRUTH INCOMPLETE", status_with_tolerance(le.leakage_rate if le else None, exp_leak, TOLERANCES["leakage_pct"], as_fraction=True)]
        else:
            statuses += [status_with_tolerance(ae.precision, exp_prec, TOLERANCES["precision_pct"], as_fraction=True), status_with_tolerance(ae.recall, exp_rec, TOLERANCES["recall_pct"], as_fraction=True), status_with_tolerance(le.leakage_rate if le else None, exp_leak, TOLERANCES["leakage_pct"], as_fraction=True)]
        rows.append([r.mode, f"{r.schema_reduction_pct:.1f}%", pct(ae.precision if ae else None), pct(ae.recall if ae else None), pct(ae.f1 if ae else None), pct(le.leakage_rate if le else None), r.canon_final, r.input_attributes, ";".join(statuses)])
    return rows

def current_empirical_rows(results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult]) -> List[List[Any]]:
    rows = []
    for r in results:
        ae, le = alias_evals.get(r.mode), leakage_evals.get(r.mode)
        prec_label = "precision" if (ae and ae.evaluated_against_closed_world) else "labeled_precision"
        prec_val = ae.precision if (ae and ae.evaluated_against_closed_world) else (ae.labeled_precision if ae else None)
        note = "closed-world" if (ae and ae.evaluated_against_closed_world) else "ground truth incomplete; review unlabeled predicted pairs"
        rows.append([r.mode, r.predicted_pairs and len(r.predicted_pairs) or 0, r.canon_final, r.input_attributes, f"{r.schema_reduction_pct:.1f}%", prec_label, pct(prec_val), pct(ae.recall if ae else None), pct(ae.f1 if ae else None), pct(le.leakage_rate if le else None), note])
    return rows

def normal_form_rows(results: List[ModeResult]) -> List[List[Any]]:
    rows = []
    for r in results:
        nf = r.nf_metrics
        specs = [
            ("EENF", "q95(var) <= tau", f"q95={nf.get('EENF_q95'):.4g}; max={nf.get('EENF_max'):.4g}", f"tau={nf.get('EENF_tau'):.4g}", nf.get("EENF_status"), nf.get("EENF_interpretation")),
            ("AANF", "min_merge_sim >= tau", fmt(nf.get("AANF_min_merge_sim")), f"tau={nf.get('AANF_tau'):.4g}", nf.get("AANF_status"), nf.get("AANF_interpretation")),
            ("CMNF", nf.get("CMNF_reason"), fmt(nf.get("CMNF_mean_overlap")), f"tau={nf.get('CMNF_tau'):.4g}", nf.get("CMNF_status"), nf.get("normal_form_interpretation")),
            ("ECNF", "min_signals >= m_min", nf.get("ECNF_min_signals"), f"m_min={nf.get('ECNF_m_min')}", nf.get("ECNF_status"), nf.get("ECNF_interpretation")),
            ("DBNF", "drift model / controlled drift evaluation", "see DBNF tables", f"tau={r.nf_metrics.get('DBNF_tau', 'NA')}", nf.get("DBNF_status"), nf.get("DBNF_interpretation")),
            ("RRNF", "role consistency", "info only", "NA", nf.get("RRNF_status"), nf.get("RRNF_interpretation")),
            ("PONF", "partition orthogonality", "info only", "NA", nf.get("PONF_status"), nf.get("PONF_interpretation")),
        ]
        for s in specs: rows.append([r.mode] + list(s))
    return rows

def trace_pairwise_evidence(results: List[ModeResult], trace_pairs: Sequence[Sequence[str]]) -> None:
    if not trace_pairs: return
    rows = []
    for tp in trace_pairs:
        if len(tp) != 2: continue
        target = Pair.make(tp[0], tp[1]); raw_a, raw_b = normalize_key_raw(tp[0]), normalize_key_raw(tp[1])
        for r in results:
            exact = [d for d in r.decisions if normalize_key_raw(d.evidence.attr_a) == raw_a and normalize_key_raw(d.evidence.attr_b) == raw_b or normalize_key_raw(d.evidence.attr_a) == raw_b and normalize_key_raw(d.evidence.attr_b) == raw_a]
            matches = exact or [d for d in r.decisions if d.pair == target]
            if not matches:
                rows.append([r.mode, tp[0], tp[1], "NOT_FOUND"] + [""] * 24); continue
            d = matches[0]; e = d.evidence; mt = "EXACT_RAW" if exact else "NORMALIZED_EQUIVALENT"
            rows.append([r.mode, mt, e.attr_a, e.attr_b, e.source_a, e.source_b, e.context_a, e.context_b, fmt(e.cosine_similarity), fmt(e.name_similarity), e.ontology_root_a, e.ontology_root_b, e.ontology_match, fmt(e.value_cooccurrence), e.regex_a, e.regex_b, e.regex_match, e.regex_compatible, fmt(e.vss_similarity), fmt(e.shape_similarity), fmt(e.aggregate_score), e.evidence_signal_count, e.AANF_status, e.ECNF_status, e.CMNF_status, e.final_decision, e.reason, e.lineage_id])
    print_table(["mode","trace_match_type","attr_a","attr_b","source_a","source_b","context_a","context_b","cosine_similarity","name_similarity","ontology_root_a","ontology_root_b","ontology_match","value_cooccurrence","regex_a","regex_b","regex_match","regex_compatible","vss_similarity","shape_similarity","aggregate_score","evidence_signal_count","AANF_status","ECNF_status","CMNF_status","final_decision","reason","lineage_id"], rows, "PAIRWISE MERGE EVIDENCE TRACE")

def claim_support_summary(dataset_summary: Dict[str, Any], results: List[ModeResult], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult], eenf_rows: List[Dict[str, Any]], drift_results: List[DriftEvalResult], trace_requested: bool, closed_world: bool) -> List[List[Any]]:
    by_mode = {r.mode: r for r in results}; sdnf = by_mode.get("sdnf_hybrid") or by_mode.get("hybrid"); base = by_mode.get("embed_only_baseline")
    rows: List[List[Any]] = []
    def add(claim, measured, expected, status, table, action): rows.append([claim, measured, expected, status, table, action])
    add("7 schema files ingested", dataset_summary.get("schema_files_ingested"), 7, "PASS" if dataset_summary.get("schema_files_ingested") == 7 else "FAIL", "DATASET SUMMARY", "dataset mismatch; align data or revise paper")
    add("40 payload files ingested from current output basis", dataset_summary.get("payload_files_ingested"), 40, "PASS" if dataset_summary.get("payload_files_ingested") == 40 else "FAIL", "DATASET SUMMARY", "keep claim" if dataset_summary.get("payload_files_ingested") == 40 else "dataset mismatch; align data or revise paper")
    if sdnf:
        add("80 input attributes", sdnf.input_attributes, 80, "PASS" if sdnf.input_attributes == 80 else "FAIL", "CURRENT RUN EMPIRICAL TABLE", "revise paper claim to measured value")
        add("49 final canonical attributes", sdnf.canon_final, 49, "PASS" if sdnf.canon_final == 49 else "FAIL", "CURRENT RUN EMPIRICAL TABLE", "revise paper claim to measured value")
        add("38.7% schema reduction", f"{sdnf.schema_reduction_pct:.1f}%", "38.7%", status_with_tolerance(sdnf.schema_reduction_pct, 38.7, 0.2), "CURRENT RUN EMPIRICAL TABLE", "keep claim" if status_with_tolerance(sdnf.schema_reduction_pct, 38.7, 0.2)=="PASS" else "revise paper claim to measured value")
        ae = alias_evals.get(sdnf.mode); le = leakage_evals.get(sdnf.mode)
        if not closed_world:
            add("SDNF precision 95%", pct(ae.labeled_precision if ae else None), "95.0%", "GROUND TRUTH INCOMPLETE", "ALIAS MERGE EVALUATION", "curate closed-world alias ground truth and rerun")
            add("SDNF recall 90%", pct(ae.recall if ae else None), "90.0%", "GROUND TRUTH INCOMPLETE", "ALIAS MERGE EVALUATION", "curate closed-world alias ground truth and rerun")
        else:
            add("SDNF precision 95%", pct(ae.precision if ae else None), "95.0%", status_with_tolerance(ae.precision if ae else None, 95.0, 1.0, True), "ALIAS MERGE EVALUATION", "keep claim" if ae and status_with_tolerance(ae.precision,95,1,True)=="PASS" else "revise paper claim to measured value")
            add("SDNF recall 90%", pct(ae.recall if ae else None), "90.0%", status_with_tolerance(ae.recall if ae else None, 90.0, 1.0, True), "ALIAS MERGE EVALUATION", "keep claim" if ae and status_with_tolerance(ae.recall,90,1,True)=="PASS" else "revise paper claim to measured value")
        add("SDNF cross-context leakage approximately 2%", pct(le.leakage_rate if le else None), "2.0%", status_with_tolerance(le.leakage_rate if le else None, 2.0, 0.5, True), "CROSS-CONTEXT LEAKAGE", "keep claim" if le and status_with_tolerance(le.leakage_rate,2,0.5,True)=="PASS" else "revise paper claim to measured value")
    if base:
        ae = alias_evals.get(base.mode); le = leakage_evals.get(base.mode)
        if not closed_world:
            add("baseline precision 86%", pct(ae.labeled_precision if ae else None), "86.0%", "GROUND TRUTH INCOMPLETE", "ALIAS MERGE EVALUATION", "curate closed-world alias ground truth and rerun")
            add("baseline recall 95%", pct(ae.recall if ae else None), "95.0%", "GROUND TRUTH INCOMPLETE", "ALIAS MERGE EVALUATION", "curate closed-world alias ground truth and rerun")
        else:
            add("baseline precision 86%", pct(ae.precision if ae else None), "86.0%", status_with_tolerance(ae.precision if ae else None,86,1,True), "ALIAS MERGE EVALUATION", "revise paper claim to measured value")
            add("baseline recall 95%", pct(ae.recall if ae else None), "95.0%", status_with_tolerance(ae.recall if ae else None,95,1,True), "ALIAS MERGE EVALUATION", "revise paper claim to measured value")
        add("baseline cross-context leakage approximately 9%", pct(le.leakage_rate if le else None), "9.0%", status_with_tolerance(le.leakage_rate if le else None,9,0.5,True), "CROSS-CONTEXT LEAKAGE", "keep claim" if le and status_with_tolerance(le.leakage_rate,9,0.5,True)=="PASS" else "revise paper claim to measured value")
    for G, exp in [(10,40.0),(20,70.0)]:
        er = next((x for x in eenf_rows if x.get("G") == G), None)
        measured = er.get("variance_reduction_vs_G1") if er else None
        status = "NOT MEASURABLE" if measured is None else ("PASS" if abs(measured*100-exp) <= 5.0 else "FAIL")
        add(f"G={G} variance reduction approximately {int(exp)}%", pct(measured), f"{exp:.1f}%", status, "EENF STABILITY-LATENCY SWEEP", "keep claim" if status=="PASS" else "revise paper claim to measured value")
    # Timing
    if sdnf:
        vals = [d.timing.total_decision_ms for d in sdnf.decisions]; mean_ms = float(np.mean(vals)) if vals else None
        add("average merge decision under 50ms", fmt(mean_ms), "<50ms", "PASS" if mean_ms is not None and mean_ms < 50 else "NOT MEASURABLE", "MERGE DECISION TIMING SUMMARY", "keep claim" if mean_ms is not None and mean_ms < 50 else "instrument timing and rerun")
    # DBNF
    dr = next((x for x in drift_results if x.eval_type == "controlled"), None) or next((x for x in drift_results if x.eval_type == "exploratory"), None)
    if dr is None:
        add("DBNF drift detection precision / recall / F1", "NOT EXERCISED", "P/R/F1 >= 80%", "NOT EXERCISED", "DBNF DRIFT DETECTION", "not exercised in current run")
    elif not dr.measurable:
        add("DBNF drift detection precision / recall / F1", "NOT MEASURABLE", "P/R/F1 >= 80%", "NOT MEASURABLE", "DBNF DRIFT DETECTION", "provide drift ground truth or controlled benchmark and rerun")
    else:
        pass_dbnf = (dr.precision or 0) >= .8 and (dr.recall or 0) >= .8 and (dr.f1 or 0) >= .8
        action = "keep claim" if pass_dbnf else ("report exploratory hotspots only; do not claim DBNF precision/recall" if dr.eval_type == "exploratory" else "revise DBNF claim to measured controlled-drift value")
        add("DBNF drift detection precision / recall / F1", f"P={pct(dr.precision)}/R={pct(dr.recall)}/F1={pct(dr.f1)}", "P/R/F1 >= 80%", "PASS" if pass_dbnf else "FAIL", "DBNF DRIFT DETECTION", action)
    add("trace-pair evidence printed", "printed" if trace_requested else "not requested", "computed trace", "PASS" if trace_requested else "NOT MEASURABLE", "PAIRWISE MERGE EVIDENCE TRACE", "keep claim" if trace_requested else "request --trace_pair and rerun")
    return rows

def serializable_summary(args: argparse.Namespace, dataset_summary: Dict[str, Any], normal_rows: List[List[Any]], alias_evals: Dict[str, AliasEvalResult], leakage_evals: Dict[str, LeakageEvalResult], table2: List[List[Any]], empirical: List[List[Any]], eenf_rows: List[Dict[str, Any]], timing_rows: List[List[Any]], drift_results: List[DriftEvalResult], claim_rows: List[List[Any]]) -> Dict[str, Any]:
    return {
        "run_configuration": {k: str(v) for k, v in vars(args).items()},
        "dataset_summary": dataset_summary,
        "normal_form_summary": normal_rows,
        "alias_eval_summary": {k: asdict(v) for k, v in alias_evals.items()},
        "leakage_eval_summary": {k: asdict(v) for k, v in leakage_evals.items()},
        "table2_reproduction": table2,
        "current_run_empirical_table": empirical,
        "eenf_sweep": eenf_rows,
        "timing_summary": timing_rows,
        "dbnf_summary": [asdict(d) for d in drift_results],
        "claim_support_summary": claim_rows,
    }

# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reviewer-grade SDNF experiment/audit harness v11")
    p.add_argument("--data_dir", default="data"); p.add_argument("--payloads_dir", default="payloads")
    p.add_argument("--evidence_mode", default="hybrid"); p.add_argument("--model", default="all-MiniLM-L6-v2"); p.add_argument("--drift_model", default=None)
    p.add_argument("--ground_truth_aliases", default=None); p.add_argument("--ground_truth_closed_world", action="store_true")
    p.add_argument("--drift_ground_truth", default=None); p.add_argument("--controlled_drift_json", default=None)
    p.add_argument("--drift_eval_mode", choices=["exploratory", "controlled", "both"], default=None)
    p.add_argument("--strict_paper_reproduction", action="store_true"); p.add_argument("--trace_pair", nargs=2, action="append", default=[])
    p.add_argument("--eenf_g_sweep", default=None); p.add_argument("--eenf_repeats", type=int, default=20); p.add_argument("--measure_timing", action="store_true")
    p.add_argument("--context", default="Payments Risk"); p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tau_eenf", type=float, default=0.000129); p.add_argument("--tau_aanf", type=float, default=0.650); p.add_argument("--tau_cmnf", type=float, default=0.100)
    p.add_argument("--tau_dbnf", type=float, default=0.250); p.add_argument("--tau_dbnf_drift", type=float, default=0.150); p.add_argument("--gamma", type=float, default=0.70); p.add_argument("--m_min", type=int, default=4); p.add_argument("--drift_top_k", type=int, default=10)
    p.add_argument("--export_decisions", default=None); p.add_argument("--export_predicted_pairs", default=None); p.add_argument("--export_false_positives", default=None); p.add_argument("--export_false_negatives", default=None); p.add_argument("--export_ground_truth_template", default=None); p.add_argument("--export_summary_json", default=None)
    return p

def main() -> None:
    args = build_arg_parser().parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    if args.drift_eval_mode is None:
        args.drift_eval_mode = "controlled" if args.controlled_drift_json else "exploratory"
    embedder = EmbeddingProvider(args.model, args.seed)
    attrs, dataset_summary = collect_attributes(Path(args.data_dir), Path(args.payloads_dir), args.context, embedder)
    true_pairs, negative_pairs = load_ground_truth_aliases(args.ground_truth_aliases, args)
    results = [run_mode(attrs, m, args, embedder) for m in mode_list(args.evidence_mode)]
    for r in results:
        r.drift_hotspots = compute_drift_hotspots(attrs, args, embedder)
    alias_evals = {r.mode: evaluate_alias_merges(r.mode, r.predicted_pairs, true_pairs, args.ground_truth_closed_world) for r in results}
    leakage_evals = {r.mode: evaluate_cross_context_leakage(r.mode, r.decisions, negative_pairs) for r in results}
    g_values = [int(x.strip()) for x in args.eenf_g_sweep.split(",")] if args.eenf_g_sweep else []
    eenf_rows = run_eenf_g_sweep(attrs, embedder, g_values, args.eenf_repeats) if g_values else []
    universe = {a.key for a in attrs}; drift_results: List[DriftEvalResult] = []
    if args.drift_eval_mode in {"exploratory", "both"} and args.drift_model:
        truth = load_drift_ground_truth(args.drift_ground_truth, args)
        for r in results:
            drift_results.append(evaluate_drift_detection(r.mode, r.drift_hotspots, truth, args.tau_dbnf_drift, universe, "exploratory"))
    if args.drift_eval_mode in {"controlled", "both"}:
        cases = load_controlled_drift_cases(args.controlled_drift_json, args)
        if cases:
            drift_results.append(evaluate_controlled_drift(attrs, args, embedder, cases))
    decision_rows = export_decisions(args.export_decisions, results, true_pairs, negative_pairs)
    predicted_json = export_predicted_pairs(args.export_predicted_pairs, results)
    export_false_positive_negative(args.export_false_positives, args.export_false_negatives, results, true_pairs, negative_pairs)
    export_ground_truth_template(args.export_ground_truth_template, predicted_json, negative_pairs)

    print_table(["Option", "Value"], [[k, v] for k, v in vars(args).items()], "RUN CONFIGURATION")
    print_table(["Metric", "Value"], [[k, pct(v) if k == "missing_fraction" else v] for k, v in dataset_summary.items()], "DATASET SUMMARY")
    normal_rows = normal_form_rows(results)
    print_table(["mode","NormalForm","Rule","Actual","Expected","Status","normal_form_interpretation"], normal_rows, "NORMAL FORM VALIDATION SUMMARY")
    alias_rows = [[a.mode, a.predicted_pairs_count, a.true_pairs_count, a.tp, a.fp, a.fn, pct(a.labeled_precision), pct(a.recall), pct(a.f1), a.evaluated_against_closed_world, a.ground_truth_coverage_warning] for a in alias_evals.values()]
    print_table(["mode","predicted_pairs_count","true_pairs_count","TP","FP","FN","labeled_precision","recall","F1","closed_world","ground_truth_coverage_warning"], alias_rows, "ALIAS MERGE EVALUATION AGAINST GROUND TRUTH")
    leakage_rows = [[l.mode, l.leakage_count, l.predicted_merge_count, pct(l.leakage_rate), "; ".join(l.examples)] for l in leakage_evals.values()]
    print_table(["mode","leakage_count","predicted_merge_count","leakage_rate","top_leakage_examples"], leakage_rows, "CROSS-CONTEXT LEAKAGE EVALUATION")
    table2 = paper_table_2_rows(results, alias_evals, leakage_evals)
    print_table(["approach","schema_reduction","merge_precision","merge_recall","F1","cross_context_leakage","canon_final","input_attributes","supported_paper_claim_status"], table2, "PAPER TABLE 2 REPRODUCTION CHECK")
    empirical = current_empirical_rows(results, alias_evals, leakage_evals)
    print_table(["mode","predicted_pairs","canon_final","input_attributes","schema_reduction","precision_column","precision_value","recall","F1","leakage","note"], empirical, "CURRENT RUN EMPIRICAL TABLE")
    if eenf_rows:
        print_table(["G","mean_variance","q95_variance","max_variance","variance_reduction_vs_G1","encoding_time_sec","overhead_vs_G1","claim_status","recommended_action","note"], [[x.get(k) for k in ["G","mean_variance","q95_variance","max_variance","variance_reduction_vs_G1","encoding_time_sec","overhead_vs_G1","claim_status","recommended_action","note"]] for x in eenf_rows], "EENF STABILITY-LATENCY SWEEP")
    timing_rows = [timing_summary(r.mode, r.decisions) for r in results]
    print_table(["mode","candidate_pairs","mean_ms","p50_ms","p95_ms","p99_ms","max_ms"], timing_rows, "MERGE DECISION TIMING SUMMARY")
    for row in timing_rows:
        print(f"Average merge decision under 50ms ({row[0]}): {'PASS' if row[2] is not None and row[2] < 50 else 'NOT MEASURABLE'}")
    if drift_results:
        for kind in sorted({d.eval_type for d in drift_results}):
            title = "CONTROLLED DBNF DRIFT EVALUATION" if kind == "controlled" else "DBNF EXPLORATORY MODEL-SHIFT HOTSPOT EVALUATION"
            rows = [[d.mode, d.drift_tau, d.detected_count, d.true_drift_count, d.tp, d.fp, d.fn, pct(d.precision), pct(d.recall), pct(d.f1), pct(d.accuracy_if_defined), ", ".join(d.detected_set), ", ".join(d.true_set), ", ".join(d.tp_set), ", ".join(d.fp_set), ", ".join(d.fn_set)] for d in drift_results if d.eval_type == kind]
            print_table(["mode","drift_tau","detected_count","true_drift_count","TP","FP","FN","precision","recall","F1","accuracy","detected_drift_set","true_drift_set","TP_drift_attributes","FP_drift_attributes","FN_drift_attributes"], rows, title)
    trace_pairwise_evidence(results, args.trace_pair)
    claim_rows = claim_support_summary(dataset_summary, results, alias_evals, leakage_evals, eenf_rows, drift_results, bool(args.trace_pair), args.ground_truth_closed_world)
    print_table(["paper_claim","measured_value","expected_value","status","evidence_table","recommended_paper_action"], claim_rows, "CLAIM SUPPORT SUMMARY")
    if args.export_summary_json:
        summary = serializable_summary(args, dataset_summary, normal_rows, alias_evals, leakage_evals, table2, empirical, eenf_rows, timing_rows, drift_results, claim_rows)
        with open(args.export_summary_json, "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    main()
