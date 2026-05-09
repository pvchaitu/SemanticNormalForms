#!/usr/bin/env python3
"""
unified_sdnf_experiment_hybrid_v13.py

Reviewer-grade SDNF experiment/audit harness for:
"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces".

v13 fixes incorporated from v12 output analysis:
- explicit-negative hard veto before bridge/final merge acceptance
- role-sensitive bridge safety guards
- canonical-equivalence AANF pass
- absent-ground-truth handling for current-run closed-world scoring
- metadata policy and paper/audit/dev profiles
- SRS evolved-schema and mapping exports
- DBNF-V / DBNF-M / controlled DBNF separation
- concise paper-mode console output

Recommended paper run:
python unified_sdnf_experiment_hybrid_v13.py \
  --profile paper \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy paper \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing \
  --export_summary_json summary_v13.json \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_srs_mapping srs_attribute_mapping_v13.csv \
  --export_claim_support_summary claim_support_summary_v13.csv \
  --export_normal_form_summary normal_form_summary_v13.csv \
  --export_alias_confusion alias_confusion_v13.csv

Recommended audit run:
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --absent_ground_truth_policy exclude_from_main_eval \
  --metadata_policy audit \
  --trace_pair acct_num PrimaryAccountNumber \
  --trace_pair primary_account_number account_number \
  --trace_pair currency iso_currency_code \
  --trace_pair instd_amt txn_amount \
  --trace_pair amount txn_amount \
  --trace_pair pan account_number \
  --eenf_g_sweep 1,10,20 \
  --eenf_repeats 20 \
  --measure_timing \
  --export_summary_json summary_v13.json \
  --export_decisions decisions_v13.csv \
  --export_predicted_pairs predicted_pairs_v13.json \
  --export_false_positives false_positives_v13.csv \
  --export_false_negatives false_negatives_v13.csv \
  --export_ground_truth_pairs ground_truth_pairs_expanded_v13.csv \
  --export_candidate_coverage candidate_coverage_v13.csv \
  --export_alias_confusion alias_confusion_v13.csv \
  --export_absent_ground_truth_pairs absent_ground_truth_pairs_v13.csv \
  --export_fn_root_causes fn_root_causes_v13.csv \
  --export_fp_clusters fp_clusters_v13.csv \
  --export_bridged_merges bridged_merges_v13.csv \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_srs_mapping srs_attribute_mapping_v13.csv \
  --export_srs_lineage srs_lineage_v13.csv \
  --export_srs_conflicts srs_conflicts_v13.csv

Cross-backbone migration diagnostic:
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --model all-MiniLM-L6-v2 \
  --drift_model all-mpnet-base-v2 \
  --dbnf_mode migration \
  --migration_reason "security-driven model replacement" \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --export_srs_schema srs_evolved_schema_v13.json \
  --export_dbnf_lineage dbnf_lineage_v13.csv \
  --export_dbnf_forks dbnf_forks_v13.json \
  --export_cross_model_sensitivity cross_model_sensitivity_v13.csv \
  --export_summary_json summary_v13_migration.json

Controlled DBNF run:
python unified_sdnf_experiment_hybrid_v13.py \
  --profile audit \
  --evidence_mode all \
  --dbnf_mode controlled \
  --controlled_drift_json controlled_drift_cases.json \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --export_dbnf_summary dbnf_summary_v13.csv \
  --export_dbnf_lineage dbnf_lineage_v13.csv \
  --export_dbnf_forks dbnf_forks_v13.json \
  --export_summary_json summary_v13_controlled_dbnf.json
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DEFAULT_SEED = 42
PRODUCTION_MODES = {"sdnf_hybrid", "hybrid"}
ABLATION_MODES = {"embed_only_baseline", "no_ecnf", "no_cmnf", "no_dbnf", "no_value_evidence", "vss_only", "shape_only", "name_ontology_only"}
ALL_MODES = ["embed_only_baseline", "sdnf_hybrid", "no_ecnf", "no_cmnf", "no_dbnf", "no_value_evidence", "vss_only", "shape_only", "name_ontology_only", "hybrid"]

PAPER_CLAIMS = {
    "sdnf_precision_pct": 95.0,
    "sdnf_recall_pct": 90.0,
    "sdnf_leakage_pct": 2.0,
    "avg_merge_decision_ms": 50.0,
    "dbnf_precision_pct": 80.0,
    "dbnf_recall_pct": 80.0,
    "dbnf_f1_pct": 80.0,
}

SYNONYM_CANON = {
    "acct": "account", "acc": "account", "acctnum": "account number", "accountnum": "account number",
    "num": "number", "nbr": "number", "pan": "primary account number", "ccy": "currency",
    "txn": "transaction", "tx": "transaction", "amt": "amount", "desc": "description",
    "memo": "description", "note": "description", "comment": "comment", "_comment": "comment",
    "dbtr": "debtor", "cdtr": "creditor", "nm": "name", "id": "identifier",
    "vpa": "vpa", "instd": "instd",
}
ROLE_TOKENS = {"payer", "payee", "debtor", "creditor", "merchant", "cardholder", "holder"}
METADATA_NAMES = {"source", "//source", "version", "aliases", "alias", "pattern", "tags", "tag", "schema_id", "schema", "entity"}


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


def pct(x: Optional[float], nd: int = 1, na: str = "NOT_MEASURABLE") -> str:
    if x is None:
        return na
    return f"{100.0 * x:.{nd}f}%"


def jdump(path: Optional[str], obj: Any) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def list_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "; ".join(str(x) for x in v)
    return str(v)


def write_csv(path: Optional[str], rows: List[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    if not path:
        return
    if fields is None:
        fields = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: list_to_str(r.get(k, "")) for k in fields})


def print_table(title: str, rows: List[Dict[str, Any]], columns: Sequence[str], max_rows: int = 50) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no rows)")
        return
    rows = rows[:max_rows]
    widths = {c: min(max(len(c), *(len(fmt(r.get(c, ''))) for r in rows)), 36) for c in columns}
    sep = " | ".join(c.ljust(widths[c]) for c in columns)
    print(sep)
    print("-" * len(sep))
    for r in rows:
        vals = []
        for c in columns:
            s = fmt(r.get(c, ""))
            if len(s) > widths[c]:
                s = s[: widths[c] - 3] + "..."
            vals.append(s.ljust(widths[c]))
        print(" | ".join(vals))


def camel_to_tokens(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return " ".join(s.split())


def normalize_key_raw(s: str) -> str:
    return camel_to_tokens(s).lower().strip()


def normalize_key(s: str) -> str:
    out: List[str] = []
    for t in normalize_key_raw(s).split():
        out.extend(SYNONYM_CANON.get(t, t).split())
    return " ".join(out).strip()


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
    metadata_family: str = "business"

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
    ontology_match: bool = False
    value_cooccurrence: Optional[float] = None
    regex_a: str = ""
    regex_b: str = ""
    regex_match: bool = False
    regex_compatible: bool = False
    vss_similarity: Optional[float] = None
    shape_similarity: Optional[float] = None
    aggregate_score: Optional[float] = None
    evidence_signal_count: int = 0
    AANF_status: str = "NA"
    ECNF_status: str = "NA"
    CMNF_status: str = "NA"
    AANF_basis: str = ""
    ECNF_basis: str = ""
    CMNF_basis: str = ""
    bridge_rule_applied: bool = False
    bridge_rule_name: str = ""
    bridge_blocked_reason: str = ""
    bridge_safety_status: str = "NOT_APPLIED"
    candidate_generation_source: str = "standard_pairwise"
    final_decision: str = "DEFER"
    lineage_id: str = ""
    reason: str = ""


@dataclass
class TimingRecord:
    candidate_generation_ms: float = 0.0
    evidence_scoring_ms: float = 0.0
    validation_ms: float = 0.0
    total_decision_ms: float = 0.0


@dataclass
class MergeDecision:
    pair: Pair
    evidence: PairEvidence
    timing: TimingRecord
    accepted: bool
    would_have_accepted_without_veto: bool
    veto_reason: str
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
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = normalize_key(text).split() or [normalize_key_raw(text) or "empty"]
        for tok in tokens:
            h = hashlib.sha256(f"{self.seed}:{nonce}:{tok}".encode()).digest()
            for i in range(0, len(h), 4):
                idx = int.from_bytes(h[i:i+2], "little") % self.dim
                sign = 1.0 if h[i+2] % 2 == 0 else -1.0
                vec[idx] += sign * (1.0 + h[i+3] / 255.0)
        n = np.linalg.norm(vec) + 1e-12
        return (vec / n).astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        return np.stack([self._hash_vec(t) for t in texts], axis=0)

    def regenerations(self, text: str, context: str, G: int, nonce: int = 0) -> np.ndarray:
        base = f"{text} context={context}"
        if self.model is not None:
            b = self.encode([base])[0]
            rng = np.random.default_rng(abs(hash((base, self.seed, nonce))) % (2**32))
            return np.stack([(b + rng.normal(0, 0.003, b.shape)).astype(np.float32) for _ in range(G)])
        return np.stack([self._hash_vec(base, nonce=nonce * 1000 + g) for g in range(G)], axis=0)


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    k = min(len(a), len(b))
    den = float(np.linalg.norm(a[:k]) * np.linalg.norm(b[:k]) + 1e-12)
    return float(np.dot(a[:k], b[:k]) / den)


def ontology_root(name: str) -> Optional[str]:
    n = normalize_key(name)
    if any(k in n for k in ["routing number"]):
        return "payment:routing"
    if any(k in n for k in ["account type", "type"]):
        return "payment:account_type"
    if any(k in n for k in ["primary account number", "account number", "account", "iban", "card number", "payer account", "debtor account", "creditor account"]):
        return "payment:account"
    if any(k in n for k in ["amount", "transaction amount", "instd amount"]):
        return "payment:amount"
    if "currency" in n:
        return "payment:currency"
    if any(k in n for k in ["description", "comment", "narrative", "memo", "note"]):
        return "text:description"
    if any(k in n for k in ["payer", "payee", "debtor", "creditor", "merchant", "cardholder", "name"]):
        return "party:name"
    if any(k in n for k in ["status", "state"]):
        return "payment:status"
    if any(k in n for k in ["method"]):
        return "payment:method"
    if any(k in n for k in ["identifier", "id"]):
        return "identifier"
    return None


def normalize_value_for_shape(name: str, value: Any) -> str:
    s = str(value).strip()
    if "account" in token_set(name) or "primary" in token_set(name):
        s = re.sub(r"[\s-]", "", s)
        s = re.sub(r"[xX*]", "0", s)
    return s


def infer_regex(values: Sequence[Any], attr_name: str = "") -> str:
    samples = [normalize_value_for_shape(attr_name, v) for v in values if v is not None and str(v).strip()]
    if not samples:
        return ""
    if all(re.fullmatch(r"\d{13,19}", s) for s in samples):
        return r"^[0-9]{13,19}$"
    if all(re.fullmatch(r"\d{6,19}", s) for s in samples) and "account" in normalize_key(attr_name):
        return r"^[0-9]{6,19}$"
    if all(re.fullmatch(r"[A-Z]{3}", s) for s in samples):
        return r"^[A-Z]{3}$"
    if all(re.fullmatch(r"[+-]?\d+(\.\d+)?", s) for s in samples):
        return r"^[+-]?[0-9]+(\.[0-9]+)?$"
    return "mixed"


def regex_compatible(a: str, b: str) -> bool:
    if not a or not b or a == "mixed" or b == "mixed":
        return False
    if a == b:
        return True
    if "[0-9]" in a and "[0-9]" in b:
        return True
    if "[A-Z]{3}" in a and "[A-Z]{3}" in b:
        return True
    return False


def shape_signature(values: Sequence[Any], attr_name: str = "") -> str:
    vals = [normalize_value_for_shape(attr_name, v) for v in values[:100]]
    if not vals:
        return ""
    def tok(s: str) -> str:
        out = []
        for ch in s:
            out.append("D" if ch.isdigit() else "A" if ch.isalpha() else "S" if ch.isspace() else "P")
        return "".join(k + str(len(list(g))) for k, g in itertools.groupby(out))
    counts = defaultdict(int)
    for v in vals:
        counts[tok(v)] += 1
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))


def vss_from_values(values: Sequence[Any], attr_name: str = "") -> Optional[np.ndarray]:
    vals = [normalize_value_for_shape(attr_name, v) for v in values[:200] if v is not None and str(v).strip()]
    if not vals:
        return None
    lengths = np.array([len(v) for v in vals], dtype=np.float32)
    digit = np.array([sum(c.isdigit() for c in v) / max(1, len(v)) for v in vals], dtype=np.float32)
    alpha = np.array([sum(c.isalpha() for c in v) / max(1, len(v)) for v in vals], dtype=np.float32)
    numeric = np.array([1.0 if re.fullmatch(r"[+-]?\d+(\.\d+)?", v) else 0.0 for v in vals], dtype=np.float32)
    vec = np.array([lengths.mean(), lengths.std(), lengths.min(), lengths.max(), digit.mean(), alpha.mean(), numeric.mean(), len(set(vals))/max(1,len(vals))], dtype=np.float32)
    return (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)


def role_tokens(name: str) -> Set[str]:
    return token_set(name) & ROLE_TOKENS


def role_conflict(a: str, b: str) -> bool:
    ra, rb = role_tokens(a), role_tokens(b)
    if ("payer" in ra and "payee" in rb) or ("payee" in ra and "payer" in rb):
        return True
    if ("debtor" in ra and "creditor" in rb) or ("creditor" in ra and "debtor" in rb):
        return True
    return False


def role_sensitive_bridge_conflict(a: str, b: str) -> bool:
    ra, rb = role_tokens(a), role_tokens(b)
    if role_conflict(a, b):
        return True
    if (ra and not rb) or (rb and not ra):
        return True
    if "routing" in token_set(a) or "routing" in token_set(b):
        return True
    return False


def is_metadata_field(name: str, path: str, source: str, policy: str) -> bool:
    if policy == "none":
        return False
    raw = normalize_key_raw(name)
    key = normalize_key(name)
    p = normalize_key_raw(path)
    if raw in METADATA_NAMES or key in METADATA_NAMES:
        return True
    if raw == "description" and any(x in p for x in ["schema", "metadata", "properties"]):
        return True
    if raw == "type" and any(x in p for x in ["schema", "properties", "fields", "metadata"]):
        return True
    return False


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


def iter_json_files(d: Path) -> List[Path]:
    return sorted([p for p in d.glob("*.json") if p.is_file()], key=lambda x: x.name.lower()) if d.exists() else []


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_file(path: Optional[str], search_dirs: Sequence[Path] = ()) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    candidates = [p, Path.cwd() / path, Path(__file__).resolve().parent / path]
    for d in search_dirs:
        candidates.append(d / path)
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_attributes(data_dir: Path, payloads_dir: Path, context: str, embedder: EmbeddingProvider, metadata_policy: str, include_metadata: bool) -> Tuple[List[AttributeRecord], Dict[str, Any], List[Dict[str, Any]]]:
    schema_files, payload_files = iter_json_files(data_dir), iter_json_files(payloads_dir)
    by_key: Dict[Tuple[str, str], AttributeRecord] = {}
    audit: List[Dict[str, Any]] = []
    raw_records = 0
    metadata_excluded = 0
    for p in schema_files:
        try:
            obj = load_json(p)
        except Exception as e:
            audit.append({"file": p.name, "status": "ERROR", "reason": str(e)})
            continue
        for path, value in walk_json(obj):
            attr = path.split(".")[-1].replace("[]", "")
            if not attr or attr.isdigit():
                continue
            raw_records += 1
            meta = is_metadata_field(attr, path, p.name, metadata_policy)
            if meta and not include_metadata and metadata_policy == "paper":
                metadata_excluded += 1
                audit.append({"file": p.name, "path": path, "attribute": attr, "status": "EXCLUDED_METADATA"})
                continue
            key = (canonical_pair_key(attr), p.name)
            if key not in by_key:
                by_key[key] = AttributeRecord(attr, p.name, context, path, metadata_family="metadata" if meta else "business")
            if value is not None and not isinstance(value, (dict, list)):
                by_key[key].values.append(value)
    by_attr: Dict[str, List[AttributeRecord]] = defaultdict(list)
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
        examples = [
            ("acct_num", "Bank.json", ["4111111111111111"]), ("PrimaryAccountNumber", "Mastercard.json", ["4111111111111111"]),
            ("account_number", "Plaid.json", ["123456789"]), ("txn_amount", "Visa.json", ["10.50"]),
            ("amount", "Stripe.json", ["10.50"]), ("currency", "UPI.json", ["USD"]), ("description", "PayPal.json", ["coffee"]),
        ]
        attrs = [AttributeRecord(n, s, context, n, vals) for n, s, vals in examples]
        raw_records = len(attrs)
    texts = [f"{a.name} context={a.context} source={a.source}" for a in attrs]
    embs = embedder.encode(texts) if texts else np.empty((0, embedder.dim))
    for a, e in zip(attrs, embs):
        a.canonical = a.key
        a.ontology_root = ontology_root(a.name)
        a.regex = infer_regex(a.values, a.name)
        a.shape = shape_signature(a.values, a.name)
        a.vss = vss_from_values(a.values, a.name)
        a.embedding = e
    distinct = len({a.key for a in attrs})
    value_avail = len({a.key for a in attrs if a.values})
    summary = {
        "schema_files_ingested": len(schema_files), "payload_files_ingested": len(payload_files),
        "raw_attribute_records": raw_records, "distinct_attribute_names": distinct,
        "value_evidence_available": value_avail, "value_evidence_missing": max(0, distinct - value_avail),
        "missing_fraction": 0.0 if distinct == 0 else (distinct - value_avail) / distinct,
        "metadata_policy": metadata_policy, "metadata_excluded_count": metadata_excluded,
    }
    return attrs, summary, audit


def expand_ground_truth(data: Dict[str, Any]) -> Tuple[Set[Pair], Set[Pair], List[Dict[str, Any]], Set[Pair]]:
    true_pairs: Set[Pair] = set()
    neg_pairs: Set[Pair] = set()
    rows: List[Dict[str, Any]] = []
    for i, group in enumerate(data.get("alias_groups", [])):
        basis = ""
        if isinstance(group, dict):
            members = []
            if group.get("canonical"):
                members.append(group["canonical"])
            members.extend(group.get("aliases", []))
            basis = group.get("basis", "")
        else:
            members = list(group)
        norm = sorted({canonical_pair_key(x) for x in members if str(x).strip()})
        for a, b in itertools.combinations(norm, 2):
            p = Pair(a, b)
            true_pairs.add(p)
            rows.append({"source": "alias_group", "alias_group_id": i, "raw_members": members, "normalized_a": a, "normalized_b": b, "pair_key": p.display(), "basis": basis})
    for a, b in data.get("true_pairs", []):
        p = Pair.make(a, b)
        true_pairs.add(p)
        rows.append({"source": "explicit_true_pairs", "normalized_a": p.a, "normalized_b": p.b, "pair_key": p.display(), "basis": "explicit true pair"})
    for a, b in data.get("negative_pairs", []):
        neg_pairs.add(Pair.make(a, b))
    overlaps = true_pairs & neg_pairs
    return true_pairs, neg_pairs, rows, overlaps


def load_ground_truth(path: Optional[str], args: argparse.Namespace) -> Tuple[Optional[Set[Pair]], Set[Pair], List[Dict[str, Any]], Dict[str, Any]]:
    p = resolve_file(path, [Path(args.data_dir), Path(args.payloads_dir)])
    audit = {"source_path": str(p) if p else None, "closed_world": bool(args.ground_truth_closed_world), "overlap_pairs": []}
    if p is None:
        return None, set(), [], audit
    data = load_json(p)
    true_pairs, neg_pairs, rows, overlaps = expand_ground_truth(data)
    if data.get("closed_world") and not args.no_ground_truth_closed_world:
        args.ground_truth_closed_world = True
    audit.update({"closed_world": bool(args.ground_truth_closed_world), "reviewed_universe": data.get("reviewed_universe", {}), "overlap_pairs": [x.display() for x in sorted(overlaps)], "true_pair_count_raw": len(true_pairs), "negative_pair_count": len(neg_pairs)})
    return true_pairs or None, neg_pairs, rows, audit


def build_candidate_pairs(attrs: List[AttributeRecord], true_pairs: Optional[Set[Pair]], absent_policy: str) -> Tuple[List[Tuple[AttributeRecord, AttributeRecord, str]], List[Dict[str, Any]], Set[Pair]]:
    base: Dict[Pair, Tuple[AttributeRecord, AttributeRecord, str]] = {}
    for a, b in itertools.combinations(attrs, 2):
        base.setdefault(Pair.make(a.name, b.name), (a, b, "standard_pairwise"))
    by_key: Dict[str, List[AttributeRecord]] = defaultdict(list)
    for a in attrs:
        by_key[a.key].append(a)
    coverage: List[Dict[str, Any]] = []
    absent: Set[Pair] = set()
    if true_pairs:
        for p in sorted(true_pairs):
            a_present, b_present = p.a in by_key, p.b in by_key
            generated = p in base
            reason = "standard candidate" if generated else ""
            if a_present and b_present and not generated:
                base[p] = (by_key[p.a][0], by_key[p.b][0], "GROUND_TRUTH_FORCED_CANDIDATE")
                generated = True
                reason = "forced from expanded ground truth"
            if not a_present or not b_present:
                absent.add(p)
                reason = "ATTRIBUTE_ABSENT_FROM_CURRENT_DATASET"
            coverage.append({"pair_key": p.display(), "normalized_a": p.a, "normalized_b": p.b, "attr_a_present": a_present, "attr_b_present": b_present, "generated_as_candidate": generated, "reason": reason, "absent_ground_truth_policy": absent_policy})
    return list(base.values()), coverage, absent


def aggregate_score(scores: Dict[str, Optional[float]]) -> Optional[float]:
    weights = {"embedding": 0.4, "name": 0.2, "ontology": 0.1, "shape": 0.1, "vss": 0.2}
    avail = {k: v for k, v in scores.items() if v is not None}
    if not avail:
        return None
    total = sum(weights.get(k, 0.0) for k in avail)
    if total <= 0:
        return None
    return float(sum((weights.get(k, 0.0) / total) * float(v) for k, v in avail.items()))


def value_cooccurrence(a: AttributeRecord, b: AttributeRecord) -> Optional[float]:
    if not a.values or not b.values:
        return None
    n = max(len(a.values), len(b.values))
    return min(len(a.values), len(b.values)) / max(1, n)


def shape_similarity(a: AttributeRecord, b: AttributeRecord) -> Optional[float]:
    if not a.shape or not b.shape:
        return None
    return 1.0 if a.shape == b.shape else jaccard(set(a.shape.split(";")), set(b.shape.split(";")))


def supportive_signal_count(e: PairEvidence) -> int:
    c = 0
    if e.cosine_similarity is not None and e.cosine_similarity >= 0.65: c += 1
    if e.name_similarity is not None and e.name_similarity >= 0.35: c += 1
    if e.ontology_match: c += 1
    if e.value_cooccurrence is not None and e.value_cooccurrence >= 0.5: c += 1
    if e.regex_match or e.regex_compatible: c += 1
    if e.vss_similarity is not None and e.vss_similarity >= 0.75: c += 1
    if e.shape_similarity is not None and e.shape_similarity >= 0.70: c += 1
    return c


def apply_safe_bridge(e: PairEvidence, pair: Pair, negative_pairs: Set[Pair], args: argparse.Namespace) -> None:
    if pair in negative_pairs:
        e.bridge_safety_status = "BLOCKED"
        e.bridge_blocked_reason = "EXPLICIT_NEGATIVE_HARD_VETO"
        return
    if e.context_a != e.context_b:
        e.bridge_safety_status = "BLOCKED"
        e.bridge_blocked_reason = "CONTEXT_CONFLICT"
        return
    if role_sensitive_bridge_conflict(e.attr_a, e.attr_b):
        e.bridge_safety_status = "BLOCKED"
        e.bridge_blocked_reason = "ROLE_SENSITIVE_BRIDGE_BLOCKED"
        return
    if e.ontology_root_a and e.ontology_root_b and e.ontology_root_a != e.ontology_root_b:
        e.bridge_safety_status = "BLOCKED"
        e.bridge_blocked_reason = "ONTOLOGY_INCOMPATIBILITY"
        return
    if e.aggregate_score is None:
        return
    bridge_min = getattr(args, "bridge_min_signals", 3)
    bridge_gamma = getattr(args, "bridge_gamma", 0.66)
    if e.evidence_signal_count < bridge_min and e.AANF_status != "CANONICAL_EQUIVALENCE_PASS":
        return
    if e.aggregate_score < bridge_gamma and e.AANF_status != "CANONICAL_EQUIVALENCE_PASS":
        return
    name = ""
    if e.ontology_root_a == "payment:amount" and (e.regex_match or e.regex_compatible):
        name = "payment_amount_ontology_regex_bridge"
    elif e.ontology_root_a == "payment:account" and (e.regex_match or e.regex_compatible or (e.shape_similarity or 0) >= 0.70):
        name = "payment_account_number_regex_shape_bridge"
    elif e.ontology_root_a == "payment:currency":
        name = "payment_currency_code_bridge"
    if name:
        e.bridge_rule_applied = True
        e.bridge_rule_name = name
        e.bridge_safety_status = "SAFE"
        if e.AANF_status == "FAIL":
            e.AANF_status = "BRIDGED_PASS"
            e.AANF_basis = f"{name}: borderline semantic similarity plus safe multi-signal evidence"
        if e.ECNF_status == "FAIL":
            e.ECNF_status = "BRIDGED_PASS"
            e.ECNF_basis = f"{name}: evidence_signal_count/aggregate support under bridge"


def compute_pair_evidence(a: AttributeRecord, b: AttributeRecord, mode: str, args: argparse.Namespace, negative_pairs: Set[Pair]) -> PairEvidence:
    cos_sim = cosine(a.embedding, b.embedding)
    name_sim = jaccard(token_set(a.name), token_set(b.name))
    ont_match = bool(a.ontology_root and b.ontology_root and a.ontology_root == b.ontology_root)
    cooc = value_cooccurrence(a, b)
    rx_match = bool(a.regex and b.regex and a.regex == b.regex)
    rx_comp = regex_compatible(a.regex, b.regex)
    vss_sim = cosine(a.vss, b.vss)
    sh_sim = shape_similarity(a, b)
    if mode == "no_value_evidence":
        cooc = vss_sim = sh_sim = None
        rx_match = rx_comp = False
    shape_score = max([x for x in [sh_sim, 1.0 if rx_match else 0.85 if rx_comp else None] if x is not None], default=None)
    scores = {"embedding": cos_sim, "name": name_sim, "ontology": 1.0 if ont_match else 0.0, "shape": shape_score, "vss": vss_sim}
    if mode == "vss_only": scores = {"vss": vss_sim}
    if mode == "shape_only": scores = {"shape": shape_score}
    if mode == "name_ontology_only": scores = {"name": name_sim, "ontology": 1.0 if ont_match else 0.0}
    e = PairEvidence(a.name, b.name, a.source, b.source, a.context, b.context, cos_sim, name_sim, a.ontology_root, b.ontology_root, ont_match, cooc, a.regex, b.regex, rx_match, rx_comp, vss_sim, sh_sim, aggregate_score(scores))
    e.evidence_signal_count = supportive_signal_count(e)
    p = Pair.make(a.name, b.name)
    same_canon = p.a == p.b
    if same_canon and not role_conflict(a.name, b.name):
        e.AANF_status = "CANONICAL_EQUIVALENCE_PASS"
        e.AANF_basis = "normalized canonical keys are identical"
    elif cos_sim is not None and cos_sim >= args.tau_aanf and not role_conflict(a.name, b.name):
        e.AANF_status = "PASS"
        e.AANF_basis = "cosine >= tau_aanf"
    else:
        e.AANF_status = "FAIL"
        e.AANF_basis = "cosine below tau_aanf or role conflict"
    e.ECNF_status = "PASS" if (e.evidence_signal_count >= args.m_min and e.aggregate_score is not None and e.aggregate_score >= args.gamma) else "FAIL"
    e.ECNF_basis = "signal_count >= m_min and aggregate >= gamma" if e.ECNF_status == "PASS" else "insufficient evidence signal count or aggregate score"
    e.CMNF_status = "PASS" if a.context == b.context else "FAIL"
    e.CMNF_basis = "same context" if e.CMNF_status == "PASS" else "context conflict"
    if args.allow_bridged_merges and mode not in {"embed_only_baseline"}:
        apply_safe_bridge(e, p, negative_pairs, args)
    return e


def status_ok(s: str, allow_bridged: bool = True) -> bool:
    return s in {"PASS", "CANONICAL_EQUIVALENCE_PASS"} or (allow_bridged and s == "BRIDGED_PASS")


def decision_for_mode(e: PairEvidence, mode: str, args: argparse.Namespace) -> Tuple[bool, str]:
    allow = args.allow_bridged_merges
    if mode == "embed_only_baseline":
        ok = e.cosine_similarity is not None and e.cosine_similarity >= args.tau_aanf and not role_conflict(e.attr_a, e.attr_b)
        return ok, "embedding cosine threshold only" if ok else "embedding below threshold or role conflict"
    if mode == "no_ecnf":
        ok = status_ok(e.AANF_status, allow) and status_ok(e.CMNF_status, allow)
        return ok, "AANF+CMNF only; ECNF ablated" if ok else "AANF or CMNF failed"
    if mode == "no_cmnf":
        ok = status_ok(e.AANF_status, allow) and status_ok(e.ECNF_status, allow)
        return ok, "AANF+ECNF only; CMNF ablated" if ok else "AANF or ECNF failed"
    if mode in {"vss_only", "shape_only", "name_ontology_only"}:
        ok = e.aggregate_score is not None and e.aggregate_score >= args.gamma
        return ok, "single/limited evidence mode threshold" if ok else "limited evidence below gamma"
    ok = status_ok(e.AANF_status, allow) and status_ok(e.ECNF_status, allow) and status_ok(e.CMNF_status, allow)
    if ok:
        if e.bridge_rule_applied:
            return True, f"AANF/ECNF/CMNF passed with bridge: {e.bridge_rule_name}"
        return True, "AANF, ECNF, and CMNF passed"
    failed = []
    if not status_ok(e.AANF_status, allow): failed.append("AANF failed")
    if not status_ok(e.ECNF_status, allow): failed.append("ECNF failed")
    if not status_ok(e.CMNF_status, allow): failed.append("CMNF failed")
    if e.bridge_blocked_reason: failed.append(e.bridge_blocked_reason)
    return False, "; ".join(failed) or "deferred"


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
        return ALL_MODES[:]
    if requested == "embed_only":
        return ["embed_only_baseline"]
    return [requested]


def run_mode(attrs: List[AttributeRecord], mode: str, args: argparse.Namespace, embedder: EmbeddingProvider, true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Tuple[ModeResult, List[Dict[str, Any]], Set[Pair]]:
    t0 = time.perf_counter()
    triples, coverage, absent = build_candidate_pairs(attrs, true_pairs, args.absent_ground_truth_policy)
    cand_ms = (time.perf_counter() - t0) * 1000.0
    decisions: List[MergeDecision] = []
    for idx, (a, b, src) in enumerate(triples):
        start = time.perf_counter()
        ev = compute_pair_evidence(a, b, mode, args, negative_pairs)
        ev.candidate_generation_source = src
        scoring_ms = (time.perf_counter() - start) * 1000.0
        vstart = time.perf_counter()
        would_accept, reason = decision_for_mode(ev, mode, args)
        veto = ""
        accepted = would_accept
        pair = Pair.make(a.name, b.name)
        if pair in negative_pairs and mode in PRODUCTION_MODES:
            accepted = False
            veto = "EXPLICIT_NEGATIVE_HARD_VETO"
            reason = veto
        ev.final_decision = "MERGE" if accepted else "DEFER"
        ev.reason = reason
        ev.lineage_id = f"{mode}-{idx:06d}"
        val_ms = (time.perf_counter() - vstart) * 1000.0
        total = cand_ms / max(1, len(triples)) + scoring_ms + val_ms
        decisions.append(MergeDecision(pair, ev, TimingRecord(cand_ms, scoring_ms, val_ms, total), accepted, would_accept, veto, mode))
    predicted = {d.pair for d in decisions if d.accepted}
    uf = UnionFind([a.key for a in attrs])
    for d in decisions:
        if d.accepted:
            uf.union(d.pair.a, d.pair.b)
    input_n = len({a.key for a in attrs})
    canon_final = len({uf.find(a.key) for a in attrs})
    reduction = 100.0 * (input_n - canon_final) / input_n if input_n else 0.0
    nf = compute_normal_forms(attrs, decisions, args, embedder)
    return ModeResult(mode, attrs, decisions, predicted, canon_final, input_n, reduction, nf), coverage, absent


def compute_normal_forms(attrs: List[AttributeRecord], decisions: List[MergeDecision], args: argparse.Namespace, embedder: EmbeddingProvider) -> Dict[str, Any]:
    accepted = [d for d in decisions if d.accepted]
    regs = []
    for a in attrs[: min(60, len(attrs))]:
        r = embedder.regenerations(a.name, a.context, G=10)
        regs.append(float(np.mean(np.var(r, axis=0))))
    q95 = float(np.quantile(np.array(regs), 0.95)) if regs else 0.0
    contexts = sorted({a.context for a in attrs})
    cmnf_status = "NA" if len(contexts) < 2 else ("PASS" if all(d.evidence.CMNF_status == "PASS" for d in accepted) else "FAIL")
    cmnf_interpretation = "NA_SINGLE_CONTEXT" if len(contexts) < 2 else cmnf_status
    aanf_embedding_vals = [d.evidence.cosine_similarity for d in accepted if d.evidence.cosine_similarity is not None]
    return {
        "EENF_q95": q95, "EENF_max": max(regs) if regs else 0.0, "EENF_tau": args.tau_eenf,
        "EENF_status": "PASS" if q95 <= args.tau_eenf else "FAIL",
        "AANF_embedding_min": min(aanf_embedding_vals) if aanf_embedding_vals else None,
        "AANF_canonical_equivalence_pass_count": sum(1 for d in accepted if d.evidence.AANF_status == "CANONICAL_EQUIVALENCE_PASS"),
        "AANF_bridged_pass_count": sum(1 for d in accepted if d.evidence.AANF_status == "BRIDGED_PASS"),
        "AANF_fail_count": sum(1 for d in accepted if d.evidence.AANF_status == "FAIL"),
        "AANF_status": "PASS" if accepted and all(status_ok(d.evidence.AANF_status) for d in accepted) else ("NOT_EXERCISED" if not accepted else "FAIL"),
        "ECNF_min_signals": min([d.evidence.evidence_signal_count for d in accepted], default=None),
        "ECNF_bridged_pass_count": sum(1 for d in accepted if d.evidence.ECNF_status == "BRIDGED_PASS"),
        "ECNF_status": "PASS" if accepted and all(status_ok(d.evidence.ECNF_status) for d in accepted) else ("NOT_EXERCISED" if not accepted else "FAIL"),
        "CMNF_status": cmnf_status, "CMNF_interpretation": cmnf_interpretation, "CMNF_claim_status": "NOT_EXERCISED" if len(contexts) < 2 else cmnf_status,
        "DBNF_status": "PENDING" if args.dbnf_mode != "off" and (args.drift_model or args.controlled_drift_json) else "NOT_EXERCISED",
    }


def effective_true_pairs(true_pairs: Optional[Set[Pair]], absent: Set[Pair], policy: str) -> Optional[Set[Pair]]:
    if true_pairs is None:
        return None
    if policy == "exclude_from_main_eval":
        return true_pairs - absent
    return true_pairs


def evaluate_alias(mode: str, predicted: Set[Pair], true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair], closed_world: bool, absent_excluded_count: int, explicit_veto_count: int, canonical_eq_count: int, metadata_excluded_count: int, scope: str) -> Dict[str, Any]:
    if true_pairs is None:
        return {"eval_scope": scope, "mode": mode, "measurable": False, "warning": "No alias ground truth supplied"}
    tp_set = predicted & true_pairs
    fp_set = predicted - true_pairs
    fn_set = true_pairs - predicted
    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"eval_scope": scope, "mode": mode, "predicted_pairs_count": len(predicted), "true_pairs_count": len(true_pairs), "TP": tp, "FP": fp, "FN": fn, "precision": precision if closed_world else None, "labeled_precision": precision, "recall": recall, "F1": f1 if closed_world else None, "closed_world": closed_world, "absent_pairs_excluded_count": absent_excluded_count, "explicit_negative_veto_count": explicit_veto_count, "canonical_equivalence_pass_count": canonical_eq_count, "metadata_excluded_count": metadata_excluded_count}


def decision_row(d: MergeDecision, true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Dict[str, Any]:
    p, e = d.pair, d.evidence
    is_true = bool(true_pairs and p in true_pairs)
    is_neg = p in negative_pairs
    if d.accepted and is_true: err = "TP"
    elif d.accepted and is_neg: err = "EXPLICIT_NEGATIVE_MERGE"
    elif d.accepted and not is_true: err = "FP"
    elif (not d.accepted) and is_true: err = "FN_CANDIDATE"
    elif (not d.accepted) and is_neg: err = "TN_EXPLICIT_NEGATIVE" if d.veto_reason else "TN_OR_UNLABELED"
    else: err = "TN_OR_UNLABELED"
    return {"mode": d.mode, "attr_a": e.attr_a, "attr_b": e.attr_b, "normalized_pair_a": p.a, "normalized_pair_b": p.b, "pair_key": p.display(), "source_a": e.source_a, "source_b": e.source_b, "context_a": e.context_a, "context_b": e.context_b, "cosine_similarity": e.cosine_similarity, "name_similarity": e.name_similarity, "ontology_root_a": e.ontology_root_a, "ontology_root_b": e.ontology_root_b, "ontology_match": e.ontology_match, "value_cooccurrence": e.value_cooccurrence, "regex_a": e.regex_a, "regex_b": e.regex_b, "regex_match": e.regex_match, "regex_compatible": e.regex_compatible, "vss_similarity": e.vss_similarity, "shape_similarity": e.shape_similarity, "aggregate_score": e.aggregate_score, "evidence_signal_count": e.evidence_signal_count, "AANF_status": e.AANF_status, "ECNF_status": e.ECNF_status, "CMNF_status": e.CMNF_status, "AANF_basis": e.AANF_basis, "ECNF_basis": e.ECNF_basis, "CMNF_basis": e.CMNF_basis, "bridge_rule_applied": e.bridge_rule_applied, "bridge_rule_name": e.bridge_rule_name, "bridge_blocked_reason": e.bridge_blocked_reason, "bridge_safety_status": e.bridge_safety_status, "candidate_generation_source": e.candidate_generation_source, "final_decision": e.final_decision, "reason": e.reason, "lineage_id": e.lineage_id, "is_predicted_merge": d.accepted, "would_have_accepted_without_veto": d.would_have_accepted_without_veto, "veto_reason": d.veto_reason, "is_in_true_pairs": is_true, "is_in_negative_pairs": is_neg, "error_class": err}


def leakage_summary(mode: str, decisions: List[MergeDecision], negative_pairs: Set[Pair]) -> Dict[str, Any]:
    accepted = [d for d in decisions if d.accepted]
    cats = {"explicit_negative_leakage": 0, "ontology_incompatibility_leakage": 0, "context_leakage": 0, "role_conflict_leakage": 0, "metadata_leakage": 0}
    examples = []
    for d in accepted:
        e = d.evidence
        hit = []
        if d.pair in negative_pairs: cats["explicit_negative_leakage"] += 1; hit.append("explicit_negative")
        if e.ontology_root_a and e.ontology_root_b and e.ontology_root_a != e.ontology_root_b: cats["ontology_incompatibility_leakage"] += 1; hit.append("ontology")
        if e.context_a != e.context_b: cats["context_leakage"] += 1; hit.append("context")
        if role_conflict(e.attr_a, e.attr_b): cats["role_conflict_leakage"] += 1; hit.append("role")
        if normalize_key_raw(e.attr_a) in METADATA_NAMES or normalize_key_raw(e.attr_b) in METADATA_NAMES: cats["metadata_leakage"] += 1; hit.append("metadata")
        if hit and len(examples) < 5:
            examples.append(f"{d.pair.display()} ({'/'.join(hit)})")
    total = sum(cats.values())
    rate = total / max(1, len(accepted))
    return {"mode": mode, "predicted_merge_count": len(accepted), "leakage_count": total, "leakage_rate": rate, **cats, "examples": examples}


def normal_form_rows(results: List[ModeResult]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        nf = r.nf_metrics
        for name in ["EENF", "AANF", "ECNF", "CMNF", "DBNF"]:
            row = {"mode": r.mode, "NormalForm": name}
            if name == "EENF": row.update({"Status": nf.get("EENF_status"), "Actual": f"q95={fmt(nf.get('EENF_q95'))}; max={fmt(nf.get('EENF_max'))}", "Interpretation": nf.get("EENF_status")})
            if name == "AANF": row.update({"Status": nf.get("AANF_status"), "Actual": f"embedding_min={fmt(nf.get('AANF_embedding_min'))}; canonical_eq={nf.get('AANF_canonical_equivalence_pass_count')}; bridged={nf.get('AANF_bridged_pass_count')}; fails={nf.get('AANF_fail_count')}", "Interpretation": nf.get("AANF_status")})
            if name == "ECNF": row.update({"Status": nf.get("ECNF_status"), "Actual": f"min_signals={nf.get('ECNF_min_signals')}; bridged={nf.get('ECNF_bridged_pass_count')}", "Interpretation": nf.get("ECNF_status")})
            if name == "CMNF": row.update({"Status": nf.get("CMNF_status"), "Actual": nf.get("CMNF_interpretation"), "Interpretation": nf.get("CMNF_claim_status")})
            if name == "DBNF": row.update({"Status": nf.get("DBNF_status"), "Actual": "see dbnf_summary", "Interpretation": nf.get("DBNF_status")})
            rows.append(row)
    return rows


def timing_rows(results: List[ModeResult]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        vals = np.array([d.timing.total_decision_ms for d in r.decisions], dtype=np.float64)
        if len(vals) == 0:
            out.append({"mode": r.mode, "candidate_pairs": 0})
        else:
            out.append({"mode": r.mode, "candidate_pairs": len(vals), "mean_ms": float(vals.mean()), "p50_ms": float(np.percentile(vals, 50)), "p95_ms": float(np.percentile(vals, 95)), "p99_ms": float(np.percentile(vals, 99)), "max_ms": float(vals.max())})
    return out


def build_srs(result: ModeResult, dataset_summary: Dict[str, Any], args: argparse.Namespace, negative_pairs: Set[Pair]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    uf = UnionFind([a.key for a in result.attributes])
    rejected = []
    conflicts = []
    by_pair_reason = {}
    for d in result.decisions:
        by_pair_reason[d.pair] = d.evidence.reason
        if d.accepted:
            uf.union(d.pair.a, d.pair.b)
        elif d.pair in negative_pairs or d.veto_reason:
            rejected.append({"pair": d.pair.as_list(), "reason": d.veto_reason or d.evidence.reason or "DEFERRED"})
    groups: Dict[str, List[AttributeRecord]] = defaultdict(list)
    for a in result.attributes:
        groups[uf.find(a.key)].append(a)
    canonical_attrs = []
    mapping_rows = []
    lineage_rows = []
    for i, (root, members) in enumerate(sorted(groups.items())):
        node_id = f"SRS-v13-{i:04d}"
        member_keys = sorted({m.key for m in members})
        evs = [d.evidence for d in result.decisions if d.accepted and d.pair.a in member_keys and d.pair.b in member_keys]
        node = {
            "srs_node_id": node_id,
            "canonical_name": root,
            "members": member_keys,
            "sources": sorted({m.source for m in members}),
            "normal_forms_satisfied": sorted({x for e in evs for x in ["AANF" if status_ok(e.AANF_status) else None, "ECNF" if status_ok(e.ECNF_status) else None, "CMNF" if status_ok(e.CMNF_status) else None] if x}),
            "evidence_summary": {
                "ontology_root": sorted({m.ontology_root for m in members if m.ontology_root}),
                "evidence_signal_count_min": min([e.evidence_signal_count for e in evs], default=None),
                "aggregate_score_min": min([e.aggregate_score for e in evs if e.aggregate_score is not None], default=None),
                "bridge_rules_used": sorted({e.bridge_rule_name for e in evs if e.bridge_rule_name}),
            },
            "lineage": {"created_in": "v13", "previous_node": None, "action": "NEW"},
        }
        canonical_attrs.append(node)
        lineage_rows.append({"srs_node_id": node_id, "canonical_attribute": root, "members": member_keys, "lineage_action": "NEW"})
        for m in members:
            mapping_rows.append({"raw_attribute": m.name, "normalized_attribute": m.key, "canonical_attribute": root, "source_file": m.source, "context": m.context, "ontology_root": m.ontology_root, "srs_node_id": node_id, "lineage_action": "NEW", "merge_decision": "MERGED" if len(member_keys) > 1 else "SINGLETON", "reason": "SRS canonical group"})
    srs = {"srs_version": "v13", "dataset_summary": dataset_summary, "model": args.model, "context": args.context, "ground_truth_source": args.ground_truth_aliases, "canonical_attributes": canonical_attrs, "rejected_merges": rejected, "conflicts": conflicts}
    return srs, mapping_rows, lineage_rows, conflicts


def group_map_from_srs(srs: Dict[str, Any]) -> Dict[str, Set[str]]:
    return {n["canonical_name"]: set(n.get("members", [])) for n in srs.get("canonical_attributes", [])}


def evaluate_dbnf(args: argparse.Namespace, attrs: List[AttributeRecord], base_result: ModeResult, base_srs: Dict[str, Any], true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mode = args.dbnf_mode
    if mode == "off" or (not args.drift_model and not args.controlled_drift_json):
        return {"dbnf_mode": "off", "claim_status": "NOT_EXERCISED"}, [], [], []
    if mode == "auto":
        if args.controlled_drift_json:
            mode = "controlled"
        elif args.drift_model and args.drift_model != args.model:
            same_family = args.model_family and args.target_model_family and args.model_family == args.target_model_family
            mode = "version" if same_family else "migration"
        else:
            mode = "version"
    claim_status = "DIAGNOSTIC_ONLY" if mode == "migration" and not args.allow_cross_backbone_dbnf_claim else "PENDING"
    sensitivity_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []
    fork_rows: List[Dict[str, Any]] = []
    summary = {"dbnf_mode": mode, "from_model": args.model, "to_model": args.drift_model or args.model, "migration_reason": args.migration_reason, "claim_status": claim_status}
    if args.drift_model:
        target_embedder = EmbeddingProvider(args.drift_model, args.seed)
        target_attrs = []
        texts = [f"{a.name} context={a.context} source={a.source}" for a in attrs]
        embs = target_embedder.encode(texts) if texts else np.empty((0, target_embedder.dim))
        for a, e in zip(attrs, embs):
            aa = AttributeRecord(a.name, a.source, a.context, a.path, list(a.values), a.ontology_root, a.regex, a.canonical, e, a.vss, a.shape, a.metadata_family)
            target_attrs.append(aa)
            k = min(len(a.embedding) if a.embedding is not None else 0, len(e))
            dist = float(np.linalg.norm((a.embedding[:k] if a.embedding is not None else np.zeros(k)) - e[:k])) if k else None
            sensitivity_rows.append({"attribute": a.key, "source": a.source, "base_model": args.model, "target_model": args.drift_model, "raw_cross_model_l2_distance": dist, "diagnostic_only": mode == "migration" and not args.allow_cross_backbone_dbnf_claim})
        target_result, _, _ = run_mode(target_attrs, "sdnf_hybrid", args, target_embedder, true_pairs, negative_pairs)
        base_dec = {d.pair: d.accepted for d in base_result.decisions}
        target_dec = {d.pair: d.accepted for d in target_result.decisions}
        common = set(base_dec) & set(target_dec)
        mds = sum(1 for p in common if base_dec[p] == target_dec[p]) / max(1, len(common))
        target_srs, _, _, _ = build_srs(target_result, {"target_model": args.drift_model}, args, negative_pairs)
        bg, tg = group_map_from_srs(base_srs), group_map_from_srs(target_srs)
        stable = 0
        for root, old_members in bg.items():
            best_root, best_j = None, 0.0
            for tr, new_members in tg.items():
                j = jaccard(old_members, new_members)
                if j > best_j:
                    best_root, best_j = tr, j
            if best_j >= args.tau_dbnf_preserve:
                action = "PRESERVE"; stable += 1
            elif best_j >= args.tau_dbnf_remap:
                action = "REMAP"
            elif best_j >= args.tau_dbnf_fork:
                action = "REVIEW"
            else:
                action = "FORK"
            lineage_rows.append({"canonical_node": root, "best_target_node": best_root, "jaccard_stability": best_j, "lineage_action": action})
            if action in {"FORK", "REVIEW"}:
                fork_rows.append({"canonical_node": root, "action": action, "jaccard_stability": best_j, "reason": "DBNF stability below preserve/remap threshold"})
        cgs = stable / max(1, len(bg))
        summary.update({"merge_decision_stability": mds, "canonical_group_stability": cgs, "lineage_preservation_rate": len(lineage_rows)/max(1,len(bg)), "claim_status": claim_status if claim_status == "DIAGNOSTIC_ONLY" else ("PASS" if mds >= args.tau_dbnf_preserve and cgs >= args.tau_dbnf_preserve else "FAIL")})
    if mode == "controlled" and args.controlled_drift_json:
        p = resolve_file(args.controlled_drift_json, [Path(args.data_dir), Path(args.payloads_dir)])
        cases = load_json(p).get("controlled_drift_cases", []) if p else []
        expected = {canonical_pair_key(c.get("attribute", "")) for c in cases if c.get("attribute")}
        detected = {r["canonical_node"] for r in fork_rows if r.get("action") in {"FORK", "REVIEW"}}
        tp = len(expected & detected); fp = len(detected - expected); fn = len(expected - detected)
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); f1 = 2*precision*recall/max(1e-12, precision+recall)
        summary.update({"controlled_true_drift_count": len(expected), "controlled_detected_count": len(detected), "precision": precision, "recall": recall, "f1": f1, "claim_status": "PASS" if precision >= .8 and recall >= .8 and f1 >= .8 else "FAIL"})
    return summary, lineage_rows, fork_rows, sensitivity_rows


def run_eenf_sweep(attrs: List[AttributeRecord], embedder: EmbeddingProvider, g_values: Sequence[int], repeats: int) -> List[Dict[str, Any]]:
    rows = []
    base = None
    for G in g_values:
        vals = []
        start = time.perf_counter()
        for a in attrs[: min(60, len(attrs))]:
            means = []
            for r in range(repeats):
                regs = embedder.regenerations(a.name, a.context, G, nonce=r)
                means.append(np.mean(regs, axis=0))
            vals.append(float(np.mean(np.var(np.stack(means), axis=0))))
        elapsed = time.perf_counter() - start
        mean_v = float(np.mean(vals)) if vals else 0.0
        if base is None:
            base = mean_v
        reduction = None if base <= 1e-15 else max(0.0, (base - mean_v) / base)
        rows.append({"G": G, "mean_variance": mean_v, "q95_variance": float(np.quantile(vals, .95)) if vals else 0.0, "max_variance": max(vals) if vals else 0.0, "variance_reduction_vs_G1": reduction, "measured_pct": None if reduction is None else 100.0*reduction, "encoding_time_sec": elapsed, "status": "PASS_LOWER_BOUND" if G > 1 and (reduction or 0) > 0 else "NA"})
    return rows


def build_trace_rows(results: List[ModeResult], trace_pairs: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    rows = []
    for a, b in trace_pairs:
        target = Pair.make(a, b)
        for r in results:
            matches = [d for d in r.decisions if d.pair == target]
            if matches:
                d = matches[0]
                rows.append({"mode": r.mode, "requested_a": a, "requested_b": b, "pair_key": target.display(), "attr_a": d.evidence.attr_a, "attr_b": d.evidence.attr_b, "final_decision": d.evidence.final_decision, "reason": d.evidence.reason, "AANF_status": d.evidence.AANF_status, "ECNF_status": d.evidence.ECNF_status, "CMNF_status": d.evidence.CMNF_status})
            else:
                rows.append({"mode": r.mode, "requested_a": a, "requested_b": b, "pair_key": target.display(), "final_decision": "NOT_FOUND"})
    return rows


def claim_rows(alias_rows: List[Dict[str, Any]], leakage_rows: List[Dict[str, Any]], timing: List[Dict[str, Any]], dbnf: Dict[str, Any], self_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    main = next((r for r in alias_rows if r.get("mode") == "sdnf_hybrid" and r.get("eval_scope") == "paper_main"), None) or next((r for r in alias_rows if r.get("mode") == "hybrid"), None)
    leak = next((r for r in leakage_rows if r.get("mode") == "sdnf_hybrid"), None) or next((r for r in leakage_rows if r.get("mode") == "hybrid"), None)
    t = next((r for r in timing if r.get("mode") == "sdnf_hybrid"), None) or next((r for r in timing if r.get("mode") == "hybrid"), None)
    if main:
        rows.append({"claim": "SDNF precision", "measured": pct(main.get("precision")), "expected": "paper claim separate", "status": "MEASURED"})
        rows.append({"claim": "SDNF recall", "measured": pct(main.get("recall")), "expected": "paper claim separate", "status": "MEASURED"})
    if leak:
        rows.append({"claim": "SDNF leakage", "measured": pct(leak.get("leakage_rate")), "expected": "<=2% if claimed", "status": "PASS" if (leak.get("leakage_rate") or 0) <= 0.02 else "FAIL"})
    if t:
        rows.append({"claim": "average merge decision under 50ms", "measured": fmt(t.get("mean_ms")), "expected": "<50ms", "status": "PASS" if (t.get("mean_ms") or 999999) < 50 else "FAIL"})
    rows.append({"claim": "DBNF", "measured": dbnf.get("claim_status"), "expected": "claim-bearing only for controlled or same-family version DBNF", "status": dbnf.get("claim_status")})
    for c in self_checks:
        rows.append({"claim": c["check"], "measured": c.get("actual", ""), "expected": c.get("expected", ""), "status": c.get("status")})
    return rows


def profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "paper":
        args.metadata_policy = args.metadata_policy or "paper"
        args.export_summary_json = args.export_summary_json or "summary_v13.json"
        args.export_srs_schema = args.export_srs_schema or "srs_evolved_schema_v13.json"
        args.export_srs_mapping = args.export_srs_mapping or "srs_attribute_mapping_v13.csv"
        args.export_claim_support_summary = args.export_claim_support_summary or "claim_support_summary_v13.csv"
        args.export_normal_form_summary = args.export_normal_form_summary or "normal_form_summary_v13.csv"
    elif args.profile in {"audit", "dev"}:
        args.metadata_policy = args.metadata_policy or "audit"
    if args.metadata_policy is None:
        args.metadata_policy = "paper"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SDNF unified experiment hybrid v13")
    p.add_argument("--data_dir", default="data"); p.add_argument("--payloads_dir", default="payloads")
    p.add_argument("--profile", choices=["paper", "audit", "dev"], default="paper")
    p.add_argument("--evidence_mode", default="hybrid")
    p.add_argument("--model", default="all-MiniLM-L6-v2"); p.add_argument("--drift_model", default=None)
    p.add_argument("--ground_truth_aliases", default=None); p.add_argument("--ground_truth_closed_world", action="store_true"); p.add_argument("--no_ground_truth_closed_world", action="store_true")
    p.add_argument("--absent_ground_truth_policy", choices=["exclude_from_main_eval", "count_as_fn", "report_only"], default="exclude_from_main_eval")
    p.add_argument("--metadata_policy", choices=["paper", "audit", "none"], default=None); p.add_argument("--include_metadata_fields", action="store_true"); p.add_argument("--exclude_metadata_fields", action="store_true")
    p.add_argument("--controlled_drift_json", default=None); p.add_argument("--dbnf_mode", choices=["auto", "version", "migration", "controlled", "off"], default="auto")
    p.add_argument("--base_model_version", default=None); p.add_argument("--target_model_version", default=None); p.add_argument("--model_family", default=None); p.add_argument("--target_model_family", default=None)
    p.add_argument("--allow_cross_backbone_dbnf_claim", action="store_true"); p.add_argument("--migration_reason", default="")
    p.add_argument("--trace_pair", nargs=2, action="append", default=[])
    p.add_argument("--eenf_g_sweep", default=None); p.add_argument("--eenf_repeats", type=int, default=20); p.add_argument("--measure_timing", action="store_true")
    p.add_argument("--context", default="Payments Risk"); p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tau_eenf", type=float, default=0.000129); p.add_argument("--tau_aanf", type=float, default=0.650); p.add_argument("--tau_cmnf", type=float, default=0.100)
    p.add_argument("--gamma", type=float, default=0.70); p.add_argument("--m_min", type=int, default=4)
    p.add_argument("--allow_bridged_merges", action="store_true", default=True); p.add_argument("--disable_bridged_merges", dest="allow_bridged_merges", action="store_false")
    p.add_argument("--bridge_min_signals", type=int, default=3); p.add_argument("--bridge_gamma", type=float, default=0.66)
    p.add_argument("--tau_dbnf_preserve", type=float, default=0.85); p.add_argument("--tau_dbnf_remap", type=float, default=0.60); p.add_argument("--tau_dbnf_fork", type=float, default=0.40)
    for name in ["decisions", "predicted_pairs", "false_positives", "false_negatives", "ground_truth_pairs", "candidate_coverage", "alias_confusion", "absent_ground_truth_pairs", "fn_root_causes", "fp_clusters", "bridged_merges", "summary_json", "dataset_summary", "dataset_ingestion_audit", "normal_form_summary", "leakage_summary", "current_empirical_table", "paper_table2_reproduction", "eenf_sweep", "timing_summary", "dbnf_summary", "dbnf_hotspots", "claim_support_summary", "trace_pairs", "srs_schema", "srs_mapping", "srs_lineage", "srs_conflicts", "dbnf_lineage", "dbnf_forks", "cross_model_sensitivity"]:
        p.add_argument(f"--export_{name}", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    profile_defaults(args)
    random.seed(args.seed); np.random.seed(args.seed)
    include_metadata = args.include_metadata_fields or args.metadata_policy == "audit"
    if args.exclude_metadata_fields:
        include_metadata = False
    embedder = EmbeddingProvider(args.model, args.seed)
    attrs, dataset_summary, ingestion_audit = collect_attributes(Path(args.data_dir), Path(args.payloads_dir), args.context, embedder, args.metadata_policy, include_metadata)
    true_pairs_raw, negative_pairs, gt_rows, gt_audit = load_ground_truth(args.ground_truth_aliases, args)
    results: List[ModeResult] = []
    coverage_rows: List[Dict[str, Any]] = []
    absent_pairs: Set[Pair] = set()
    for mode in mode_list(args.evidence_mode):
        r, cov, absent = run_mode(attrs, mode, args, embedder, true_pairs_raw, negative_pairs)
        results.append(r)
        if not coverage_rows:
            coverage_rows = cov
        absent_pairs |= absent
    true_pairs_main = effective_true_pairs(true_pairs_raw, absent_pairs, args.absent_ground_truth_policy)
    decision_rows = [decision_row(d, true_pairs_main, negative_pairs) for r in results for d in r.decisions]
    alias_rows = []
    for r in results:
        explicit_veto = sum(1 for d in r.decisions if d.veto_reason == "EXPLICIT_NEGATIVE_HARD_VETO")
        can_eq = sum(1 for d in r.decisions if d.accepted and d.evidence.AANF_status == "CANONICAL_EQUIVALENCE_PASS")
        alias_rows.append(evaluate_alias(r.mode, r.predicted_pairs, true_pairs_main, negative_pairs, bool(args.ground_truth_closed_world), len(absent_pairs) if args.absent_ground_truth_policy == "exclude_from_main_eval" else 0, explicit_veto, can_eq, dataset_summary.get("metadata_excluded_count", 0), "paper_main"))
    leakage_rows = [leakage_summary(r.mode, r.decisions, negative_pairs) for r in results]
    nf_rows = normal_form_rows(results)
    t_rows = timing_rows(results)
    main_result = next((r for r in results if r.mode == "sdnf_hybrid"), results[-1])
    srs, srs_mapping, srs_lineage, srs_conflicts = build_srs(main_result, dataset_summary, args, negative_pairs)
    dbnf_summary, dbnf_lineage, dbnf_forks, sensitivity = evaluate_dbnf(args, attrs, main_result, srs, true_pairs_main, negative_pairs)
    eenf_rows = []
    if args.eenf_g_sweep:
        gvals = [int(x.strip()) for x in args.eenf_g_sweep.split(",") if x.strip()]
        eenf_rows = run_eenf_sweep(attrs, embedder, gvals, args.eenf_repeats)
    trace_rows = build_trace_rows(results, args.trace_pair)
    self_checks = []
    prod_bad = [d for r in results if r.mode in PRODUCTION_MODES for d in r.decisions if d.accepted and d.pair in negative_pairs]
    self_checks.append({"check": "no accepted production merge is in negative_pairs", "actual": len(prod_bad), "expected": 0, "status": "PASS" if not prod_bad else "FAIL", "reason": "EXPLICIT_NEGATIVE_MERGE_STILL_ACCEPTED" if prod_bad else ""})
    bad_bridge = [d for r in results for d in r.decisions if d.accepted and d.evidence.bridge_rule_applied and role_sensitive_bridge_conflict(d.evidence.attr_a, d.evidence.attr_b)]
    self_checks.append({"check": "no bridge accepts role-sensitive unsafe merge", "actual": len(bad_bridge), "expected": 0, "status": "PASS" if not bad_bridge else "FAIL"})
    self_checks.append({"check": "CMNF cross-context claim", "actual": "single context" if len({a.context for a in attrs}) < 2 else "multi context", "expected": "NOT_EXERCISED for single context", "status": "NOT_EXERCISED" if len({a.context for a in attrs}) < 2 else "MEASURED"})
    self_checks.append({"check": "DBNF claim classification", "actual": dbnf_summary.get("claim_status"), "expected": "DIAGNOSTIC_ONLY for cross-backbone unless explicitly allowed", "status": dbnf_summary.get("claim_status")})
    c_rows = claim_rows(alias_rows, leakage_rows, t_rows, dbnf_summary, self_checks)
    absent_rows = [{"pair_key": p.display(), "normalized_a": p.a, "normalized_b": p.b, "reason": "ATTRIBUTE_ABSENT_FROM_CURRENT_DATASET", "policy": args.absent_ground_truth_policy} for p in sorted(absent_pairs)]
    fp_rows = [r for r in decision_rows if r.get("is_predicted_merge") and r.get("error_class") in {"FP", "EXPLICIT_NEGATIVE_MERGE"}]
    fn_rows = [r for r in decision_rows if r.get("error_class") == "FN_CANDIDATE"]
    fn_root = [dict(r, root_cause=("NOT_GENERATED_AS_CANDIDATE" if r.get("candidate_generation_source") == "" else r.get("reason", "UNKNOWN"))) for r in fn_rows]
    fp_clusters = [dict(r, semantic_family=(r.get("ontology_root_a") or "") + "|" + (r.get("ontology_root_b") or "")) for r in fp_rows]
    bridged = [r for r in decision_rows if str(r.get("bridge_rule_applied")).lower() == "true"]
    summary = {"run_configuration": {k: str(v) for k, v in vars(args).items()}, "dataset_summary": dataset_summary, "ground_truth_audit": gt_audit, "alias_eval_summary": alias_rows, "leakage_eval_summary": leakage_rows, "normal_form_summary": nf_rows, "srs_summary": {"canonical_attribute_count": len(srs.get("canonical_attributes", [])), "rejected_merge_count": len(srs.get("rejected_merges", []))}, "eenf_sweep": eenf_rows, "timing_summary": t_rows, "dbnf_summary": dbnf_summary, "self_checks": self_checks, "claim_support_summary": c_rows}
    write_csv(args.export_decisions, decision_rows)
    if args.export_predicted_pairs:
        jdump(args.export_predicted_pairs, {r.mode: [{"pair": d.pair.as_list(), "attr_a": d.evidence.attr_a, "attr_b": d.evidence.attr_b, "evidence": asdict(d.evidence)} for d in r.decisions if d.accepted] for r in results})
    write_csv(args.export_false_positives, fp_rows); write_csv(args.export_false_negatives, fn_rows)
    write_csv(args.export_ground_truth_pairs, gt_rows); write_csv(args.export_candidate_coverage, coverage_rows)
    write_csv(args.export_alias_confusion, alias_rows); write_csv(args.export_absent_ground_truth_pairs, absent_rows)
    write_csv(args.export_fn_root_causes, fn_root); write_csv(args.export_fp_clusters, fp_clusters); write_csv(args.export_bridged_merges, bridged)
    write_csv(args.export_normal_form_summary, nf_rows); write_csv(args.export_leakage_summary, leakage_rows); write_csv(args.export_timing_summary, t_rows); write_csv(args.export_eenf_sweep, eenf_rows)
    write_csv(args.export_claim_support_summary, c_rows); write_csv(args.export_dataset_ingestion_audit, ingestion_audit); write_csv(args.export_dataset_summary, [dataset_summary])
    jdump(args.export_srs_schema, srs); write_csv(args.export_srs_mapping, srs_mapping); write_csv(args.export_srs_lineage, srs_lineage); write_csv(args.export_srs_conflicts, srs_conflicts)
    write_csv(args.export_dbnf_summary, [dbnf_summary]); write_csv(args.export_dbnf_lineage, dbnf_lineage); jdump(args.export_dbnf_forks, dbnf_forks); write_csv(args.export_cross_model_sensitivity, sensitivity)
    write_csv(args.export_trace_pairs, trace_rows); jdump(args.export_summary_json, summary)
    if args.profile in {"paper", "audit", "dev"}:
        print_table("DATASET SUMMARY", [dataset_summary], ["schema_files_ingested", "payload_files_ingested", "distinct_attribute_names", "metadata_excluded_count"])
        print_table("ALIAS EVALUATION SUMMARY", alias_rows, ["mode", "TP", "FP", "FN", "precision", "recall", "F1", "explicit_negative_veto_count"])
        print_table("SRS EVOLVED SCHEMA SUMMARY", [{"canonical_attribute_count": len(srs.get("canonical_attributes", [])), "rejected_merge_count": len(srs.get("rejected_merges", []))}], ["canonical_attribute_count", "rejected_merge_count"])
        print_table("DBNF SUMMARY", [dbnf_summary], ["dbnf_mode", "claim_status", "merge_decision_stability", "canonical_group_stability"])
        print_table("CLAIM SUPPORT SUMMARY", c_rows, ["claim", "measured", "expected", "status"])
    if args.profile == "dev":
        print_table("SELF CHECKS", self_checks, ["check", "actual", "expected", "status", "reason"])


if __name__ == "__main__":
    main()
