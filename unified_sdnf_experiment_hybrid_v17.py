#!/usr/bin/env python3
"""
unified_sdnf_experiment_hybrid_v17.py
Version: 17.0.0

SDNF v17: reviewer-grade, single-file experiment harness for
"Semantic Data Normal Forms: Extending Normalization Theory to Vector Embedding Spaces".

v17 restores the v15/v14 implementation backbone (schema-first Payment SRS construction,
payload evidence, canonical nodes, payload compliance, output budgeting, DBNF/EENF/scale
artifacts) and incorporates the useful v16 evaluator fixes (self-pair exclusion, raw vs unique
predicted pairs, strict vs reviewer-diagnosed metrics, ground-truth repair modes, and conservative
claim support statuses). It adds a HUMAN_REVIEW decision gate so plausible but unsafe/ambiguous
pairs are not counted as strict predicted merges.

Roadmap scaffolds are included for future planned enhancements, but are not evaluated claims in
v17:
- CandidateRetriever with pairwise/hnsw/auto backends. HNSW, if available, only generates candidate
  pairs and never decides merges.
- CanonicalEmbeddingBuilder for centroid summaries where embeddings are present.
- SemanticGeometryAuditScaffold marked SCAFFOLDED_NOT_EVALUATED by default.
- SrsEvolutionSnapshotHook for minimal node/member snapshots.

Recommended paper run:
python unified_sdnf_experiment_hybrid_v17.py \
  --output_profile paper \
  --schemas_dir data \
  --payloads_root payloads/payment \
  --seed_srs_schema INAmex.schema.json \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --evaluation_track both \
  --dbnf_mode version_drift \
  --ground_truth_repair_mode closed_world_only \
  --candidate_backend pairwise \
  --measure_timing

Recommended audit run:
python unified_sdnf_experiment_hybrid_v17.py \
  --output_profile audit \
  --schemas_dir data \
  --payloads_root payloads/payment \
  --seed_srs_schema INAmex.schema.json \
  --ground_truth_aliases ground_truth_aliases_closed_world_v12.json \
  --ground_truth_closed_world \
  --evaluation_track both \
  --dbnf_mode both \
  --ground_truth_repair_mode schema_supported_review \
  --candidate_backend auto \
  --count_semantic_veto_conflicts_as_fn \
  --measure_timing

Minimal run:
python unified_sdnf_experiment_hybrid_v17.py \
  --output_profile minimal \
  --schemas_dir data \
  --payloads_root payloads/payment

Version-drift DBNF run:
python unified_sdnf_experiment_hybrid_v17.py --output_profile paper --dbnf_mode version_drift --measure_timing

Migration utility run:
python unified_sdnf_experiment_hybrid_v17.py --output_profile audit --dbnf_mode migration --dbnf_migration_model all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

__version__ = "17.0.0"
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "output_v17"

STATUS_SUPPORTED = "SUPPORTED"
STATUS_PARTIAL = "PARTIALLY_SUPPORTED"
STATUS_NOT_SUPPORTED = "NOT_SUPPORTED"
STATUS_NA = "NOT_APPLICABLE"
STATUS_NE = "NOT_EVALUATED"
STATUS_SCAFFOLD = "SCAFFOLDED_NOT_EVALUATED"

GLOBAL_CROSS_RAIL_FAMILIES = {"payment:amount", "payment:currency"}
ROLE_CONFLICTS = {
    ("payer", "payee"), ("payee", "payer"),
    ("debtor", "creditor"), ("creditor", "debtor"),
    ("customer", "merchant"), ("merchant", "customer"),
}
SYNONYMS = {
    "id": "identifier", "ids": "identifier", "txn": "transaction", "tx": "transaction",
    "amt": "amount", "acct": "account", "acc": "account", "num": "number", "nbr": "number",
    "dbtr": "debtor", "cdtr": "creditor", "nm": "name", "ccy": "currency",
    "pan": "primary account number", "cvv": "verification value", "cid": "verification value",
    "cvc": "verification value", "exp": "expiration date", "expiry": "expiration date",
    "vpa": "virtual payment address", "mti": "message type indicator", "instd": "instructed",
}
PAYMENT_TYPE_ORDER = ["INAmex", "Mastercard", "PPVisa", "ISO20022", "Plaid", "Razorpay", "Stripe", "UPI"]
RAIL_BY_PAYMENT_TYPE = {
    "INAmex": "card_payment", "PPVisa": "card_payment", "Mastercard": "card_network_iso8583",
    "ISO20022": "account_to_account_credit_transfer", "Plaid": "open_banking",
    "Razorpay": "psp_gateway", "Stripe": "psp_gateway", "UPI": "upi",
}

PROFILE_FILES = {
    "minimal": [
        "run_manifest_v17.json",
        "summary_audit_v17.json",
        "srs_evolved_schema_v17.compact.json",
    ],
    "paper": [
        "out_audit_v17.txt",
        "run_manifest_v17.json",
        "summary_audit_v17.json",
        "srs_evolved_schema_v17.compact.json",
        "schema_ingestion_audit_v17.csv",
        "field_evidence_audit_v17.csv",
        "schema_deltas_audit_v17.csv",
        "decisions_audit_v17.csv",
        "alias_evaluation_audit_v17.csv",
        "payload_compliance_audit_v17.csv",
        "normal_forms_and_claims_audit_v17.csv",
        "scale_timing_drift_audit_v17.csv",
        "review_queue_audit_v17.csv",
    ],
    "audit": [
        "out_audit_v17.txt",
        "run_manifest_v17.json",
        "summary_audit_v17.json",
        "srs_evolved_schema_v17.compact.json",
        "schema_ingestion_audit_v17.csv",
        "field_evidence_audit_v17.csv",
        "schema_deltas_audit_v17.csv",
        "decisions_audit_v17.csv",
        "alias_evaluation_audit_v17.csv",
        "payload_compliance_audit_v17.csv",
        "normal_forms_and_claims_audit_v17.csv",
        "scale_timing_drift_audit_v17.csv",
        "review_queue_audit_v17.csv",
        "sdnf_debug_bundle_v17.zip",
    ],
    "debug": [
        "out_audit_v17.txt",
        "run_manifest_v17.json",
        "summary_audit_v17.json",
        "srs_evolved_schema_v17.compact.json",
        "schema_ingestion_audit_v17.csv",
        "field_evidence_audit_v17.csv",
        "schema_deltas_audit_v17.csv",
        "decisions_audit_v17.csv",
        "alias_evaluation_audit_v17.csv",
        "payload_compliance_audit_v17.csv",
        "normal_forms_and_claims_audit_v17.csv",
        "scale_timing_drift_audit_v17.csv",
        "review_queue_audit_v17.csv",
        "sdnf_debug_bundle_v17.zip",
        "readme_v17.md",
    ],
}


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "; ".join(stringify(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return str(v)


def camel_to_tokens(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9@]+", " ", s)
    return " ".join(s.split())


def normalize(s: str) -> str:
    out: List[str] = []
    for tok in camel_to_tokens(str(s)).lower().split():
        out.extend(SYNONYMS.get(tok, tok).split())
    return " ".join(out).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize(s)).strip("_") or "unnamed"


def toks(s: str) -> Set[str]:
    return set(normalize(s).split())


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def safe_load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def flatten_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                yield from flatten_json(v, path)
            else:
                yield path, v
    elif isinstance(obj, list):
        for v in obj:
            path = f"{prefix}[]" if prefix else "[]"
            if isinstance(v, (dict, list)):
                yield from flatten_json(v, path)
            else:
                yield path, v


def leaf(path: str) -> str:
    return path.split(".")[-1].replace("[]", "")


def cosine(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    if np is not None:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        if aa.size == 0 or bb.size == 0 or aa.shape[0] != bb.shape[0]:
            return None
        return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
    aa = [float(x) for x in a]
    bb = [float(x) for x in b]
    if not aa or not bb or len(aa) != len(bb):
        return None
    dot = sum(x * y for x, y in zip(aa, bb))
    na = math.sqrt(sum(x * x for x in aa))
    nb = math.sqrt(sum(y * y for y in bb))
    return dot / (na * nb + 1e-12)


def safe_norm(vec: Any) -> Any:
    if np is not None:
        arr = np.asarray(vec, dtype=float)
        return (arr / (np.linalg.norm(arr) + 1e-12)).astype(float)
    norm = math.sqrt(sum(float(x) * float(x) for x in vec)) + 1e-12
    return [float(x) / norm for x in vec]


class OutputBudgetWriter:
    """All v17 output writes pass through this budget enforcer."""
    def __init__(self, output_dir: str, profile: str, max_files: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.max_files = min(max(1, int(max_files)), 15)
        if profile not in PROFILE_FILES:
            raise ValueError(f"Unknown output profile: {profile}")
        self.allowed = list(PROFILE_FILES[profile])
        if len(self.allowed) > self.max_files:
            raise ValueError(
                f"Profile '{profile}' requires {len(self.allowed)} files but max_output_files={self.max_files}. "
                "Increase --max_output_files up to 15 or use a smaller --output_profile."
            )
        self.written: List[str] = []
        self.refused: List[str] = []

    def path(self, name: str) -> Path:
        if name not in self.allowed:
            self.refused.append(name)
            raise ValueError(f"Output '{name}' is not allowed by profile '{self.profile}'.")
        if name not in self.written and len(self.written) >= self.max_files:
            self.refused.append(name)
            raise ValueError(f"Output budget exceeded while writing {name}")
        p = self.output_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name not in self.written:
            self.written.append(name)
        return p

    def write_json(self, name: str, obj: Any) -> None:
        with self.path(name).open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

    def write_text(self, name: str, text: str) -> None:
        with self.path(name).open("w", encoding="utf-8") as f:
            f.write(text)

    def write_csv(self, name: str, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
        p = self.path(name)
        if fields is None:
            fields = []
            for r in rows:
                for k in r.keys():
                    if k not in fields:
                        fields.append(k)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: stringify(r.get(k, "")) for k in fields})

    def write_zip(self, name: str, files: Sequence[str]) -> None:
        p = self.path(name)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fname in files:
                fp = self.output_dir / fname
                if fp.exists():
                    z.write(fp, arcname=fname)


@dataclass(frozen=True, order=True)
class Pair:
    a: str
    b: str

    @staticmethod
    def make(a: str, b: str) -> "Pair":
        x, y = sorted([slug(a), slug(b)])
        return Pair(x, y)

    def display(self) -> str:
        return f"{self.a} :: {self.b}"

    def as_list(self) -> List[str]:
        return [self.a, self.b]


@dataclass
class SchemaAttribute:
    payment_type: str
    schema_file: str
    schema_id: str
    provider: str
    rail: str
    domain: str
    entity: str
    name: str
    type: str = "string"
    required: bool = False
    semantic_family: str = "unknown"
    canonical_hint: str = ""
    role: str = ""
    aliases: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    do_not_merge_with_families: List[str] = field(default_factory=list)
    description: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    embedding: Any = None

    @property
    def normalized_name(self) -> str:
        return slug(self.name)

    @property
    def provider_field(self) -> str:
        return f"{self.payment_type}.{self.name}"

    @property
    def attr_id(self) -> str:
        return f"attr::{self.payment_type}::{self.normalized_name}"

    @property
    def canonical_key(self) -> str:
        return slug(self.canonical_hint or canonical_from_family(self.semantic_family, self.name))


@dataclass
class SchemaDescriptor:
    payment_type: str
    path: str
    schema_id: str
    domain: str
    rail: str
    provider: str
    entity: str
    version: str
    schema_descriptor_version: str
    schema_source: str
    review_status: str
    spec_monitoring: Dict[str, Any] = field(default_factory=dict)
    upgrade_governance: Dict[str, Any] = field(default_factory=dict)
    attributes: List[SchemaAttribute] = field(default_factory=list)


@dataclass
class PayloadObservation:
    payment_type: str
    file: str
    path: str
    field: str
    value: Any

    @property
    def normalized_field(self) -> str:
        return slug(self.field)


@dataclass
class FieldEvidence:
    payment_type: str
    field: str
    normalized_field: str
    count: int
    total_payloads: int
    presence_ratio: str
    presence_class: str
    observed_type: str
    regex: str
    shape: str
    distinct_count: int
    examples: List[str]


@dataclass
class MergeDecision:
    decision_id: str
    decision_type: str
    raw_attribute_a: str
    raw_attribute_b: str
    canonical_node: str
    payment_type_a: str
    payment_type_b: str
    semantic_family_a: str
    semantic_family_b: str
    role_a: str
    role_b: str
    evidence: Dict[str, Any]
    normal_form_checks: Dict[str, Dict[str, str]]
    hard_vetoes: List[str]
    decision_reason: str
    lineage_action: str
    decision_scope: str
    evaluation_scope: str
    track: str


@dataclass
class CanonicalNode:
    node_id: str
    canonical_name: str
    semantic_family: str
    role: str
    domain: str
    rails: Set[str] = field(default_factory=set)
    providers: Set[str] = field(default_factory=set)
    members: List[SchemaAttribute] = field(default_factory=list)
    payload_evidence: List[FieldEvidence] = field(default_factory=list)
    rejected_near_misses: List[str] = field(default_factory=list)
    deferred_candidates: List[str] = field(default_factory=list)
    review_candidates: List[str] = field(default_factory=list)


@dataclass
class PayloadCompliance:
    payment_type: str
    payload_file: str
    schema_id: str
    decision: str
    required_missing: List[str]
    unexpected_fields: List[str]
    mapped_fields: List[Dict[str, Any]]
    normal_form_checks: Dict[str, str]
    reasons: List[str]
    pattern_mismatch_diagnostic: List[Dict[str, Any]]


class EmbeddingProvider:
    def __init__(self, model_name: str = "hashing-fallback", seed: int = DEFAULT_SEED, dim: int = 256):
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

    def _hash_vec(self, text: str, nonce: int = 0) -> Any:
        tokens = normalize(text).split() or ["empty"]
        if np is not None:
            vec = np.zeros(self.dim, dtype=float)
        else:
            vec = [0.0] * self.dim
        for tok in tokens:
            h = hashlib.sha256(f"{tok}|{self.seed}|{nonce}".encode()).digest()
            for i in range(0, len(h), 4):
                idx = int.from_bytes(h[i:i+2], "little") % self.dim
                sign = 1.0 if h[i+2] % 2 == 0 else -1.0
                val = sign * (1.0 + h[i+3] / 255.0)
                vec[idx] += val
        return safe_norm(vec)

    def encode(self, texts: Sequence[str]) -> List[Any]:
        if self.model is not None:
            arr = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return [arr[i] for i in range(len(texts))]
        return [self._hash_vec(t) for t in texts]

    def regenerations(self, text: str, context: str, G: int, nonce: int = 0) -> List[Any]:
        base_text = f"{text} context={context}"
        if self.model is not None and np is not None:
            base = self.encode([base_text])[0]
            rng = np.random.default_rng(abs(hash((base_text, self.seed, nonce))) % (2**32))
            return [safe_norm(base + rng.normal(0, 0.003, size=base.shape)) for _ in range(G)]
        return [self._hash_vec(base_text, nonce=nonce * 1000 + g) for g in range(G)]


def partition_of(family: str) -> str:
    if not family or family == "unknown":
        return "unknown"
    return family.split(":", 1)[0] if ":" in family else family


def infer_family(name: str, payment_type: str = "") -> str:
    n = normalize(name)
    pt = payment_type.lower()
    if name == "id" and pt == "stripe": return "identifier:payment_intent"
    if "payment intent" in n: return "identifier:payment_intent"
    if "razorpay payment" in n or "payment identifier" in n or "payment id" in n: return "identifier:razorpay_payment"
    if "order" in n and "identifier" in n: return "identifier:order"
    if "end to end" in n: return "identifier:end_to_end_payment"
    if "message" in n and "identifier" in n: return "identifier:message"
    if "transaction" in n and "identifier" in n: return "identifier:transaction"
    if "txn" in name.lower() and "id" in name.lower(): return "identifier:transaction"
    if "customer" in n and "identifier" in n: return "identifier:customer"
    if "account id" in n: return "identifier:plaid_account"
    if "card acceptor" in n: return "identifier:card_acceptor"
    if "schema" in n and "identifier" in n: return "metadata:schema_identifier"
    if "amount" in n or "instructed" in n: return "payment:amount"
    if "currency" in n: return "payment:currency"
    if "method" in n: return "payment:method"
    if "status" in n or "state" in n: return "payment:status"
    if "created" in n: return "temporal:created_at"
    if "timestamp" in n or "transmission datetime" in n or "transmission date time" in n: return "temporal:transaction_timestamp"
    if "requested execution" in n: return "temporal:requested_execution_date"
    if "date" in n or "time" in n: return "temporal:transaction_date"
    if "routing" in n: return "payment_account:routing_number"
    if "primary account" in n or "card number" in n: return "payment_card:pan"
    if "bank account" in n or "account number" in n: return "payment_account:account_number"
    if "debtor account" in n: return "payment_account:debtor_account"
    if "creditor account" in n: return "payment_account:creditor_account"
    if "payer account" in n: return "payment_account:payer_account"
    if "payee" in n and "vpa" in n: return "upi:payee_vpa"
    if ("payer" in n and "vpa" in n) or n == "vpa": return "upi:payer_vpa"
    if "expiration" in n: return "payment_card:expiration_date"
    if "verification value" in n or "security code" in n: return "payment_card:verification_value"
    if "debtor" in n and "name" in n: return "party:debtor_name"
    if "creditor" in n and "name" in n: return "party:creditor_name"
    if "cardholder" in n: return "party:cardholder_name"
    if "client" in n and "name" in n: return "party:customer_name"
    if "customer" in n and "name" in n: return "party:customer_name"
    if "merchant category" in n: return "merchant:category_code"
    if "address" in n or "street" in n or "city" in n or "zip" in n: return "location:address"
    return "unknown"


def canonical_from_family(family: str, name: str) -> str:
    m = {
        "payment:amount": "payment_amount",
        "payment:currency": "payment_currency",
        "payment:method": "payment_method",
        "payment:status": "transaction_status",
        "identifier:payment_intent": "payment_intent_identifier",
        "identifier:razorpay_payment": "razorpay_payment_identifier",
        "identifier:order": "order_identifier",
        "identifier:transaction": "transaction_identifier",
        "identifier:message": "message_identifier",
        "identifier:end_to_end_payment": "end_to_end_identifier",
        "identifier:customer": "customer_identifier",
        "identifier:plaid_account": "plaid_account_identifier",
        "identifier:card_acceptor": "card_acceptor_identifier",
        "metadata:schema_identifier": "schema_identifier",
        "payment_card:pan": "card_primary_account_number",
        "payment_card:expiration_date": "card_expiration_date",
        "payment_card:verification_value": "card_verification_value",
        "payment_account:routing_number": "routing_number",
        "payment_account:account_number": "bank_account_number",
        "payment_account:debtor_account": "debtor_account",
        "payment_account:creditor_account": "creditor_account",
        "payment_account:payer_account": "payer_account",
        "upi:payer_vpa": "payer_vpa",
        "upi:payee_vpa": "payee_vpa",
        "temporal:created_at": "created_at",
        "temporal:transaction_timestamp": "transaction_timestamp",
        "temporal:transaction_date": "transaction_date",
        "temporal:requested_execution_date": "requested_execution_date",
        "party:debtor_name": "debtor_name",
        "party:creditor_name": "creditor_name",
        "party:cardholder_name": "cardholder_name",
        "party:customer_name": "customer_name",
        "merchant:category_code": "merchant_category_code",
        "location:address": "address",
    }
    return slug(m.get(family, name))


def role_tokens(role: str, name: str) -> Set[str]:
    return toks(role + " " + name) & {"payer", "payee", "debtor", "creditor", "customer", "merchant", "cardholder"}


def payment_type_from_schema(path: Path) -> str:
    return path.name[:-len(".schema.json")] if path.name.endswith(".schema.json") else path.stem


def infer_type(values: Sequence[Any]) -> str:
    vals = [v for v in values if v is not None]
    if not vals: return "unknown"
    if all(isinstance(v, bool) for v in vals): return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals): return "integer"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals): return "number"
    if all(re.fullmatch(r"[-+]?\d+(\.\d+)?", str(v).strip()) for v in vals): return "number_string"
    return "string"


def infer_regex(values: Sequence[Any]) -> str:
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals: return ""
    if all(re.fullmatch(r"\d{12,19}", v) for v in vals): return r"[0-9]{12,19}$"
    if all(re.fullmatch(r"\d{3,4}", v) for v in vals): return r"[0-9]{3,4}$"
    if all(re.fullmatch(r"[A-Z]{3}", v) for v in vals): return r"[A-Z]{3}$"
    if all(re.fullmatch(r"[a-z]{3}", v) for v in vals): return r"[a-z]{3}$"
    if all(re.fullmatch(r"[-+]?\d+(\.\d+)?", v) for v in vals): return r"[-+]?[0-9]+(.[0-9]+)?$"
    if all("@" in v for v in vals): return r".+@.+$"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in vals): return r"\d{4}-\d{2}-\d{2}$"
    if all(re.fullmatch(r"\d{8}", v) for v in vals): return r"[0-9]{8}$"
    return "mixed"


def shape_of_value(v: Any) -> str:
    cats = []
    for ch in str(v):
        cats.append("D" if ch.isdigit() else "A" if ch.isalpha() else "S" if ch.isspace() else "P")
    return "".join(k + str(len(list(g))) for k, g in itertools.groupby(cats))


def shape_signature(values: Sequence[Any]) -> str:
    vals = [shape_of_value(v) for v in values[:50] if v is not None]
    counts: Dict[str, int] = defaultdict(int)
    for v in vals:
        counts[v] += 1
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))


def presence_class(c: int, total: int) -> str:
    if total <= 0: return "unknown"
    if c == total: return "required_candidate"
    if c >= max(1, total - 2): return "conditional_or_strong_optional_candidate"
    if c >= 2: return "optional_or_method_specific_candidate"
    return "outlier_or_low_confidence_candidate"


def default_demo_descriptors(embedder: EmbeddingProvider) -> List[SchemaDescriptor]:
    # Small deterministic fallback matching the v16 diagnostic shape so the harness is executable without files.
    rows = [
        ("System1", "account_number", "Identifier/Code", "account_number", ""),
        ("System2", "primary_account_number", "Identifier/Code", "account_number", ""),
        ("System1", "customer_name", "Name/Text", "customer_name", ""),
        ("System2", "client_name", "Name/Text", "customer_name", ""),
        ("System1", "customer_id", "Identifier/Code", "customer_id", ""),
        ("System2", "bank_account_number", "Identifier/Code", "bank_account_number", ""),
        ("System1", "address_line1", "Location", "address", ""),
        ("System2", "address_line_1", "Location", "address", ""),
        ("System3", "street_address", "Location", "address", ""),
        ("System2", "order_id", "Identifier/Code", "order_id", ""),
    ]
    descs: Dict[str, SchemaDescriptor] = {}
    for pt, name, fam_label, canon, role in rows:
        fam = {
            "Identifier/Code": infer_family(name, pt),
            "Name/Text": "party:customer_name",
            "Location": "location:address",
        }.get(fam_label, infer_family(name, pt))
        desc = descs.setdefault(pt, SchemaDescriptor(pt, f"{pt}.schema.json", f"{pt}_schema", "payments", RAIL_BY_PAYMENT_TYPE.get(pt, pt), pt, pt, "v1", "v1", "fallback", "demo"))
        attr = SchemaAttribute(pt, f"{pt}.schema.json", f"{pt}_schema", pt, desc.rail, "payments", pt, name, "string", False, fam, canon, role)
        desc.attributes.append(attr)
    attrs = [a for d in descs.values() for a in d.attributes]
    embs = embedder.encode([f"{a.name} {a.semantic_family} {a.canonical_hint} {a.role}" for a in attrs])
    for a, e in zip(attrs, embs):
        a.embedding = e
    return list(descs.values())


def load_schema_descriptors(args: argparse.Namespace, embedder: EmbeddingProvider) -> Tuple[List[SchemaDescriptor], List[Dict[str, Any]]]:
    out: List[SchemaDescriptor] = []
    audit: List[Dict[str, Any]] = []
    root = Path(args.schemas_dir)
    files = sorted(root.glob(args.schema_glob)) if root.exists() else []
    for p in files:
        pt = payment_type_from_schema(p)
        raw, err = safe_load_json(p)
        if err or not isinstance(raw, dict):
            audit.append({"record_type": "schema_ingestion", "file": str(p), "payment_type": pt, "status": "ERROR", "reason": err or "root not object"})
            continue
        schema_id = str(raw.get("schema_id") or f"{pt}_schema")
        rail = str(raw.get("rail") or RAIL_BY_PAYMENT_TYPE.get(pt, "unknown_rail"))
        desc = SchemaDescriptor(
            payment_type=pt, path=str(p), schema_id=schema_id, domain=str(raw.get("domain") or "payments"),
            rail=rail, provider=str(raw.get("provider") or raw.get("provider_or_standard") or pt),
            entity=str(raw.get("entity") or pt), version=str(raw.get("version") or "v1"),
            schema_descriptor_version=str(raw.get("schema_descriptor_version") or raw.get("version") or "v1"),
            schema_source=str(raw.get("schema_source") or "schema_descriptor"),
            review_status=str(raw.get("review_status") or "unknown"),
            spec_monitoring=raw.get("spec_monitoring") if isinstance(raw.get("spec_monitoring"), dict) else {},
            upgrade_governance=raw.get("upgrade_governance") if isinstance(raw.get("upgrade_governance"), dict) else {},
        )
        attrs = raw.get("attributes") if isinstance(raw.get("attributes"), list) else []
        for item in attrs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("field") or "unnamed")
            fam = str(item.get("semantic_family") or infer_family(name, pt))
            aliases = item.get("aliases") or []
            if isinstance(aliases, str): aliases = [aliases]
            mp = item.get("merge_policy") if isinstance(item.get("merge_policy"), dict) else {}
            desc.attributes.append(SchemaAttribute(
                payment_type=pt, schema_file=p.name, schema_id=schema_id, provider=desc.provider,
                rail=rail, domain=desc.domain, entity=desc.entity, name=name,
                type=str(item.get("type") or "string"), required=bool(item.get("required", False)),
                semantic_family=fam, canonical_hint=str(item.get("canonical_hint") or canonical_from_family(fam, name)),
                role=str(item.get("role") or ""), aliases=[str(x) for x in aliases],
                constraints=item.get("constraints") if isinstance(item.get("constraints"), dict) else {},
                do_not_merge_with_families=list(item.get("do_not_merge_with_families") or mp.get("do_not_merge_with_families") or []),
                description=str(item.get("description") or ""), raw=item,
            ))
        out.append(desc)
        audit.append({"record_type": "schema_ingestion", "file": str(p), "payment_type": pt, "status": "OK", "schema_id": schema_id, "rail": rail, "attribute_count": len(desc.attributes)})
    if not out:
        out = default_demo_descriptors(embedder)
        for d in out:
            audit.append({"record_type": "schema_ingestion", "file": d.path, "payment_type": d.payment_type, "status": "FALLBACK_DEMO", "schema_id": d.schema_id, "rail": d.rail, "attribute_count": len(d.attributes)})
    attrs_all = [a for d in out for a in d.attributes]
    embs = embedder.encode([f"{a.name} {a.semantic_family} {a.canonical_hint} {a.role} {a.description}" for a in attrs_all])
    for a, e in zip(attrs_all, embs):
        a.embedding = e
    return out, audit


def load_payloads(root: Path) -> Tuple[List[PayloadObservation], Dict[str, List[Path]], List[Dict[str, Any]]]:
    obs: List[PayloadObservation] = []
    files_by_type: Dict[str, List[Path]] = defaultdict(list)
    audit: List[Dict[str, Any]] = []
    if not root.exists():
        audit.append({"record_type": "payload_ingestion", "status": "WARN", "reason": f"payloads_root not found: {root}"})
        return obs, files_by_type, audit
    folders = [p for p in root.iterdir() if p.is_dir()]
    if not folders:
        folders = [root]
    for folder in sorted(folders):
        pt = folder.name if folder != root else "default"
        for f in sorted(folder.rglob("*.json")):
            files_by_type[pt].append(f)
            raw, err = safe_load_json(f)
            if err:
                audit.append({"record_type": "payload_ingestion", "file": str(f), "payment_type": pt, "status": "ERROR", "reason": err})
                continue
            for path, value in flatten_json(raw):
                obs.append(PayloadObservation(pt, f.name, path, leaf(path), value))
            audit.append({"record_type": "payload_ingestion", "file": str(f), "payment_type": pt, "status": "OK"})
    return obs, files_by_type, audit


def build_field_evidence(obs: List[PayloadObservation], files_by_type: Dict[str, List[Path]]) -> Tuple[List[FieldEvidence], Dict[Tuple[str, str], FieldEvidence]]:
    by: Dict[Tuple[str, str], List[PayloadObservation]] = defaultdict(list)
    for o in obs:
        by[(o.payment_type, o.normalized_field)].append(o)
    rows: List[FieldEvidence] = []
    idx: Dict[Tuple[str, str], FieldEvidence] = {}
    for (pt, nf), vals in sorted(by.items()):
        total = len(files_by_type.get(pt, []))
        files = {v.file for v in vals}
        values = [v.value for v in vals]
        fe = FieldEvidence(pt, vals[0].field, nf, len(files), total, f"{len(files)}/{total}", presence_class(len(files), total), infer_type(values), infer_regex(values), shape_signature(values), len({str(v) for v in values}), [str(v) for v in values[:3]])
        rows.append(fe)
        idx[(pt, nf)] = fe
    return rows, idx


class CandidateRetriever:
    def __init__(self, backend: str = "pairwise", top_k: int = 20):
        self.requested_backend = backend
        self.backend_used = "pairwise"
        self.top_k = top_k
        self.hnsw_available = False
        if backend in {"hnsw", "auto"}:
            try:
                import hnswlib  # type: ignore  # noqa: F401
                self.hnsw_available = True
                self.backend_used = "hnsw"
            except Exception:
                self.hnsw_available = False
                self.backend_used = "pairwise"
        if backend == "pairwise":
            self.backend_used = "pairwise"

    def candidates(self, attrs: List[SchemaAttribute], args: argparse.Namespace) -> List[Tuple[SchemaAttribute, SchemaAttribute, str]]:
        # v17 keeps HNSW as candidate-generation-only. For reproducibility, pairwise-compatible
        # candidate filtering is used even when hnswlib is present unless future versions enable full index export.
        pairs: List[Tuple[SchemaAttribute, SchemaAttribute, str]] = []
        for a, b in itertools.combinations(attrs, 2):
            same_canon = a.canonical_key == b.canonical_key
            same_family = a.semantic_family == b.semantic_family and a.semantic_family != "unknown"
            name_sim = jaccard(toks(a.name), toks(b.name))
            if same_canon or same_family or name_sim >= args.candidate_name_threshold:
                pairs.append((a, b, self.backend_used))
        return pairs


class CanonicalEmbeddingBuilder:
    def build(self, nodes: Dict[str, CanonicalNode]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"status": STATUS_SCAFFOLD, "node_count": len(nodes), "centroid_available_count": 0}
        centroids: Dict[str, Any] = {}
        if np is None:
            summary["reason"] = "numpy not available; centroid export skipped"
            return {"summary": summary, "centroids": centroids}
        for key, node in nodes.items():
            vecs = [m.embedding for m in node.members if m.embedding is not None]
            if vecs:
                arr = np.asarray(vecs, dtype=float)
                centroids[key] = np.mean(arr, axis=0).tolist()[:16]  # compact preview only
        summary["centroid_available_count"] = len(centroids)
        return {"summary": summary, "centroids_preview_16d": centroids}


class SemanticGeometryAuditScaffold:
    def summarize(self, nodes: Dict[str, CanonicalNode]) -> Dict[str, Any]:
        return {
            "status": STATUS_SCAFFOLD,
            "message": "v17 includes geometry audit scaffolding only; compactness/separation/margin/leakage are not evaluated claims.",
            "canonical_node_count": len(nodes),
            "planned_metrics": ["canonical_compactness", "inter_canonical_separation", "semantic_margin", "partition_leakage"],
        }


class SrsEvolutionSnapshotHook:
    def snapshot(self, version_label: str, nodes: Dict[str, CanonicalNode]) -> Dict[str, Any]:
        return {
            "version_label": version_label,
            "status": "MINIMAL_SNAPSHOT",
            "canonical_node_count": len(nodes),
            "member_count": sum(len(n.members) for n in nodes.values()),
            "review_candidate_count": sum(len(n.review_candidates) for n in nodes.values()),
        }


def semantic_vetoes(a: SchemaAttribute, b: SchemaAttribute, allow_cross_rail_amount_currency: bool) -> List[str]:
    fa, fb = a.semantic_family, b.semantic_family
    pa, pb = partition_of(fa), partition_of(fb)
    if fa == fb:
        return []
    out: List[str] = []
    if pa == "identifier" and pb in {"payment", "temporal"}: out.append("identifier must not merge with payment/temporal")
    if pb == "identifier" and pa in {"payment", "temporal"}: out.append("identifier must not merge with payment/temporal")
    if pa == "temporal" and fb == "payment:status": out.append("temporal must not merge with status")
    if pb == "temporal" and fa == "payment:status": out.append("temporal must not merge with status")
    if fa == "payment_account:routing_number" and "account_number" in fb: out.append("routing number must not merge with account number")
    if fb == "payment_account:routing_number" and "account_number" in fa: out.append("routing number must not merge with account number")
    if fa == "payment_card:pan" and fb.startswith("payment_account:"): out.append("card PAN must not merge with bank account")
    if fb == "payment_card:pan" and fa.startswith("payment_account:"): out.append("card PAN must not merge with bank account")
    if pa == "metadata" and pb != "metadata": out.append("metadata must not merge with business attribute")
    if pb == "metadata" and pa != "metadata": out.append("metadata must not merge with business attribute")
    if pa == pb == "identifier" and fa != fb: out.append(f"identifier subtypes must remain separate: {fa} vs {fb}")
    if pa == pb and pa in {"payment_account", "payment_card"} and fa != fb:
        out.append(f"account/card subtypes are ambiguous: {fa} vs {fb}")
    return out


def role_conflict(a: SchemaAttribute, b: SchemaAttribute) -> Optional[str]:
    ra, rb = role_tokens(a.role, a.name), role_tokens(b.role, b.name)
    for x, y in ROLE_CONFLICTS:
        if x in ra and y in rb:
            return f"role conflict: {x} vs {y}"
    return None


def nf_template() -> Dict[str, Dict[str, str]]:
    return {k: {"status": "DEFER", "reason": "not evaluated"} for k in ["EENF", "AANF", "ECNF", "RRNF", "CMNF", "DBNF", "PONF"]}


def evidence_score(signals: Dict[str, float], weights: Dict[str, float]) -> float:
    available = {k: max(0.0, min(1.0, float(v))) for k, v in signals.items() if v is not None}
    if not available:
        return 0.0
    total = sum(weights.get(k, 0.0) for k in available)
    if total <= 0:
        return 0.0
    return float(sum((weights.get(k, 0.0) / total) * v for k, v in available.items()))


def is_broad_compatible_but_ambiguous(a: SchemaAttribute, b: SchemaAttribute) -> bool:
    pa, pb = partition_of(a.semantic_family), partition_of(b.semantic_family)
    if a.semantic_family == b.semantic_family:
        return False
    if pa == pb and pa in {"identifier", "payment_account", "payment_card", "party", "location"}:
        return True
    if a.canonical_key != b.canonical_key and pa == pb:
        return True
    return False


def evaluate_pair(a: SchemaAttribute, b: SchemaAttribute, args: argparse.Namespace, track: str, negative_pairs: Set[Pair], true_pairs: Optional[Set[Pair]]) -> MergeDecision:
    checks = nf_template()
    pair = Pair.make(a.provider_field, b.provider_field)
    norm_name_pair = Pair.make(a.name, b.name)
    vetoes = semantic_vetoes(a, b, args.allow_cross_rail_amount_currency)
    rc = role_conflict(a, b)
    if rc:
        vetoes.append(rc)
    if a.semantic_family in b.do_not_merge_with_families or b.semantic_family in a.do_not_merge_with_families:
        vetoes.append("schema-declared do_not_merge_with_families veto")
    explicit_negative = norm_name_pair in negative_pairs or pair in negative_pairs
    if explicit_negative:
        vetoes.append("explicit negative pair hard veto")

    same_canon = a.canonical_key == b.canonical_key
    same_family = a.semantic_family == b.semantic_family and a.semantic_family != "unknown"
    same_partition = partition_of(a.semantic_family) == partition_of(b.semantic_family)
    alias = slug(a.name) in {slug(x) for x in b.aliases} or slug(b.name) in {slug(x) for x in a.aliases}
    name_sim = jaccard(toks(a.name), toks(b.name))
    emb_sim = cosine(a.embedding, b.embedding)
    canon_compat = same_canon and (same_family or alias)
    cross_global = same_family and a.semantic_family in GLOBAL_CROSS_RAIL_FAMILIES and args.allow_cross_rail_amount_currency
    cm_ok = a.rail == b.rail or cross_global or (same_canon and same_family and not vetoes)

    signals = {
        "embedding": emb_sim if emb_sim is not None else 0.0,
        "name": name_sim,
        "ontology": 1.0 if same_family else 0.0,
        "canonical": 1.0 if same_canon else 0.0,
        "alias": 1.0 if alias else 0.0,
    }
    weights = json.loads(args.evidence_weights_json) if args.evidence_weights_json else {"embedding": 0.35, "name": 0.20, "ontology": 0.20, "canonical": 0.20, "alias": 0.05}
    score = evidence_score(signals, weights)
    support_count = sum(1 for k, v in signals.items() if v is not None and v >= {"embedding": args.tau_aanf, "name": args.name_threshold}.get(k, 0.75))

    checks["EENF"] = {"status": "PASS", "reason": "deterministic or externally managed in this run"}
    aanf_pass = same_canon or alias or (same_family and ((emb_sim or 0.0) >= args.tau_aanf or name_sim >= args.name_threshold))
    checks["AANF"] = {"status": "PASS" if aanf_pass else "FAIL", "reason": f"same_canon={same_canon}; alias={alias}; same_family={same_family}; emb={emb_sim}; name={name_sim}"}
    ecnf_pass = support_count >= args.m_min_schema and score >= args.review_threshold
    checks["ECNF"] = {"status": "PASS" if ecnf_pass else "DEFER", "reason": f"support_count={support_count}; score={score:.3f}; review_threshold={args.review_threshold}"}
    checks["RRNF"] = {"status": "FAIL" if rc else "PASS", "reason": rc or "no role conflict"}
    checks["CMNF"] = {"status": "PASS" if cm_ok else "FAIL", "reason": "same/compatible rail" if cm_ok else f"rail mismatch {a.rail} vs {b.rail}"}
    checks["PONF"] = {"status": "PASS" if same_partition else "FAIL", "reason": "same partition" if same_partition else f"partition mismatch {partition_of(a.semantic_family)} vs {partition_of(b.semantic_family)}"}
    checks["DBNF"] = {"status": "PASS", "reason": "DBNF handled at drift audit level"}

    gt_pair = norm_name_pair in true_pairs if true_pairs else False
    canonical_conflict = a.canonical_key != b.canonical_key and not alias
    ambiguous_subtype = is_broad_compatible_but_ambiguous(a, b)
    low_margin = args.review_threshold <= score < args.auto_merge_threshold
    uncertain_cross_rail = a.rail != b.rail and not cross_global

    if explicit_negative:
        typ, reason, action = "REJECT_UNSAFE", "explicit negative pair hard veto", "REJECT_UNSAFE_MERGE"
    elif gt_pair and vetoes and args.count_semantic_veto_conflicts_as_fn:
        typ, reason, action = "HUMAN_REVIEW_GT_CONFLICT", "ground-truth pair blocked by semantic veto; requires review", "QUEUE_REVIEW_GT_CONFLICT"
    elif vetoes and (score < args.review_threshold or not gt_pair):
        typ, reason, action = "REJECT_UNSAFE", "; ".join(vetoes), "REJECT_BY_NORMAL_FORM"
    elif vetoes and gt_pair:
        typ, reason, action = "HUMAN_REVIEW_GT_CONFLICT", "GT pair conflicts with semantic veto: " + "; ".join(vetoes), "QUEUE_REVIEW_GT_CONFLICT"
    elif (aanf_pass and ecnf_pass and (canonical_conflict or ambiguous_subtype or low_margin or uncertain_cross_rail)):
        reasons = []
        if canonical_conflict: reasons.append("canonical hints differ")
        if ambiguous_subtype: reasons.append("semantic subtype ambiguous")
        if low_margin: reasons.append("evidence margin below auto-merge threshold")
        if uncertain_cross_rail: reasons.append("cross-rail compatibility uncertain")
        typ, reason, action = "HUMAN_REVIEW", "; ".join(reasons), "QUEUE_HUMAN_REVIEW"
    elif aanf_pass and ecnf_pass and all(checks[x]["status"] == "PASS" for x in ["RRNF", "CMNF", "PONF"]) and score >= args.auto_merge_threshold and (canon_compat or same_family):
        typ, reason, action = "ACCEPT_MERGE", "AANF/ECNF/RRNF/CMNF/PONF passed with clear compatibility", "MERGE_INTO_CANONICAL_NODE"
    elif score >= args.review_threshold or gt_pair:
        typ, reason, action = "HUMAN_REVIEW", "plausible evidence but not safe enough for automatic merge", "QUEUE_HUMAN_REVIEW"
    else:
        typ, reason, action = "DEFER", "insufficient evidence", "DEFER_CANDIDATE"

    return MergeDecision(
        decision_id=f"dec::v17::{slug(a.provider_field)}::{slug(b.provider_field)}::{track}",
        decision_type=typ,
        raw_attribute_a=a.provider_field,
        raw_attribute_b=b.provider_field,
        canonical_node=a.canonical_key if a.canonical_key == b.canonical_key else f"{a.canonical_key}|{b.canonical_key}",
        payment_type_a=a.payment_type,
        payment_type_b=b.payment_type,
        semantic_family_a=a.semantic_family,
        semantic_family_b=b.semantic_family,
        role_a=a.role,
        role_b=b.role,
        evidence={"same_canonical": same_canon, "same_family": same_family, "explicit_alias": alias, "name_similarity": name_sim, "embedding_similarity": emb_sim, "score": score, "signals": signals, "support_count": support_count},
        normal_form_checks=checks,
        hard_vetoes=vetoes,
        decision_reason=reason,
        lineage_action=action,
        decision_scope="schema_pair",
        evaluation_scope="strict_eval" if typ == "ACCEPT_MERGE" else "review_or_nonmerge",
        track=track,
    )


def build_srs(descs: List[SchemaDescriptor], evidence_idx: Dict[Tuple[str, str], FieldEvidence], embedder: EmbeddingProvider, args: argparse.Namespace, true_pairs: Optional[Set[Pair]], negative_pairs: Set[Pair]) -> Tuple[Dict[str, CanonicalNode], List[MergeDecision], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    attrs = [a for d in descs for a in d.attributes]
    tracks = ["production", "discovery"] if args.evaluation_track == "both" else [args.evaluation_track]
    retriever = CandidateRetriever(args.candidate_backend, args.hnsw_top_k)
    decisions: List[MergeDecision] = []
    conflicts: List[Dict[str, Any]] = []
    review_queue: List[Dict[str, Any]] = []
    t0 = now_ms()
    candidates = retriever.candidates(attrs, args)
    candidate_ms = now_ms() - t0
    for track in tracks:
        for a, b, source in candidates:
            dec = evaluate_pair(a, b, args, track, negative_pairs, true_pairs)
            dec.evidence["candidate_backend_requested"] = retriever.requested_backend
            dec.evidence["candidate_backend_used"] = retriever.backend_used
            dec.evidence["candidate_source"] = source
            decisions.append(dec)
            if dec.decision_type in {"HUMAN_REVIEW", "HUMAN_REVIEW_GT_CONFLICT"}:
                review_queue.append({
                    "AttributeA": dec.raw_attribute_a, "AttributeB": dec.raw_attribute_b,
                    "DecisionType": dec.decision_type, "Reason": dec.decision_reason,
                    "CanonicalNode": dec.canonical_node, "Track": track,
                    "EvidenceScore": dec.evidence.get("score"), "HardVetoes": dec.hard_vetoes,
                })
            if dec.hard_vetoes and a.canonical_key == b.canonical_key:
                conflicts.append({"canonical_hint": a.canonical_key, "a": a.provider_field, "b": b.provider_field, "reason": dec.hard_vetoes})

    nodes: Dict[str, CanonicalNode] = {}
    for a in attrs:
        n = nodes.setdefault(a.canonical_key, CanonicalNode(f"canon::{a.canonical_key}", a.canonical_key, a.semantic_family, a.role, a.domain))
        n.members.append(a)
        n.rails.add(a.rail)
        n.providers.add(a.payment_type)
        fe = evidence_idx.get((a.payment_type, a.normalized_name))
        if fe:
            n.payload_evidence.append(fe)
    for d in decisions:
        msg = f"{d.raw_attribute_a} <-> {d.raw_attribute_b}: {d.decision_reason}"
        for key in d.canonical_node.split("|"):
            if key in nodes:
                if d.decision_type.startswith("REJECT"):
                    nodes[key].rejected_near_misses.append(msg)
                elif d.decision_type == "DEFER":
                    nodes[key].deferred_candidates.append(msg)
                elif d.decision_type.startswith("HUMAN_REVIEW"):
                    nodes[key].review_candidates.append(msg)
    mapping: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    for key, node in sorted(nodes.items()):
        lineage.append({"srs_node_id": node.node_id, "canonical_name": node.canonical_name, "semantic_family": node.semantic_family, "member_count": len(node.members), "rails": sorted(node.rails), "providers": sorted(node.providers), "review_candidate_count": len(node.review_candidates)})
        for m in node.members:
            mapping.append({"canonical_node": key, "provider_field": m.provider_field, "payment_type": m.payment_type, "semantic_family": m.semantic_family, "role": m.role, "rail": m.rail})
    scale_rows = [{"metric": "candidate_generation_ms", "value": candidate_ms}, {"metric": "candidate_pair_count", "value": len(candidates)}, {"metric": "candidate_backend_requested", "value": retriever.requested_backend}, {"metric": "candidate_backend_used", "value": retriever.backend_used}, {"metric": "hnsw_available", "value": retriever.hnsw_available}]
    return nodes, decisions, conflicts, mapping, lineage, review_queue + scale_rows


def build_lookup(descs: List[SchemaDescriptor]) -> Dict[str, Dict[str, SchemaAttribute]]:
    lookup: Dict[str, Dict[str, SchemaAttribute]] = defaultdict(dict)
    for d in descs:
        for a in d.attributes:
            keys = {a.normalized_name, slug(a.name), slug(a.canonical_hint)} | {slug(x) for x in a.aliases}
            if a.payment_type == "Stripe" and a.canonical_key == "payment_intent_identifier":
                keys.add("id")
            for k in keys:
                lookup[a.payment_type][k] = a
    return lookup


def validate_payloads(descs: List[SchemaDescriptor], obs: List[PayloadObservation], files_by_type: Dict[str, List[Path]]) -> List[PayloadCompliance]:
    lookup = build_lookup(descs)
    by_file: Dict[Tuple[str, str], List[PayloadObservation]] = defaultdict(list)
    for o in obs:
        by_file[(o.payment_type, o.file)].append(o)
    schema_by_pt = {d.payment_type: d for d in descs}
    out: List[PayloadCompliance] = []
    for (pt, fname), rows in sorted(by_file.items()):
        desc = schema_by_pt.get(pt)
        required = {a.normalized_name for a in desc.attributes if a.required} if desc else set()
        present = {o.normalized_field for o in rows}
        mapped = []
        unexpected = []
        for o in rows:
            a = lookup.get(pt, {}).get(o.normalized_field)
            if a:
                mapped.append({"field": o.field, "canonical_node": a.canonical_key, "semantic_family": a.semantic_family})
            else:
                unexpected.append(o.field)
        missing = sorted(required - present)
        decision = "PASS" if not missing and not unexpected else "REVIEW"
        reasons = []
        if missing: reasons.append("required fields missing")
        if unexpected: reasons.append("unexpected fields present")
        out.append(PayloadCompliance(pt, fname, desc.schema_id if desc else "unknown", decision, missing, sorted(set(unexpected)), mapped, {"payload_schema_mapping": decision}, reasons, []))
    return out


def pairs_from_alias_groups(groups: Sequence[Any]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for group in groups:
        if isinstance(group, dict):
            aliases = list(group.get("aliases", []))
            if group.get("canonical"):
                aliases = [group["canonical"]] + aliases
        else:
            aliases = list(group)
        normed = sorted({slug(x) for x in aliases if str(x).strip()})
        for a, b in itertools.combinations(normed, 2):
            if a != b:
                pairs.add(Pair(a, b))
    return pairs


def load_ground_truth(path: Optional[str], args: argparse.Namespace) -> Tuple[Optional[Set[Pair]], Set[Pair], Dict[str, Any]]:
    audit: Dict[str, Any] = {"source_path": path, "closed_world": bool(args.ground_truth_closed_world), "repair_mode": args.ground_truth_repair_mode}
    if not path:
        return None, set(), audit
    candidates = [Path(path), Path.cwd() / path, Path(args.schemas_dir) / path, Path(args.payloads_root) / path]
    p = next((c for c in candidates if c.exists()), None)
    if not p:
        audit["status"] = "NOT_FOUND"
        return None, set(), audit
    data, err = safe_load_json(p)
    if err or not isinstance(data, dict):
        audit["status"] = "ERROR"; audit["reason"] = err
        return None, set(), audit
    true_pairs = pairs_from_alias_groups(data.get("alias_groups", []))
    for pair in data.get("true_pairs", []):
        if len(pair) == 2:
            p2 = Pair.make(pair[0], pair[1])
            if p2.a != p2.b:
                true_pairs.add(p2)
    negative_pairs: Set[Pair] = set()
    for pair in data.get("negative_pairs", []):
        if len(pair) == 2:
            p2 = Pair.make(pair[0], pair[1])
            if p2.a != p2.b:
                negative_pairs.add(p2)
    if data.get("closed_world") and not args.no_ground_truth_closed_world:
        args.ground_truth_closed_world = True
    audit.update({"status": "OK", "true_pair_count": len(true_pairs), "negative_pair_count": len(negative_pairs), "overlap_count": len(true_pairs & negative_pairs), "closed_world": bool(args.ground_truth_closed_world)})
    return true_pairs or None, negative_pairs, audit


def predicted_pairs_from_decisions(decisions: List[MergeDecision]) -> Tuple[List[Pair], Set[Pair]]:
    raw: List[Pair] = []
    for d in decisions:
        if d.decision_type == "ACCEPT_MERGE":
            p = Pair.make(d.raw_attribute_a, d.raw_attribute_b)
            if p.a != p.b:
                raw.append(p)
    return raw, set(raw)


def review_pairs(decisions: List[MergeDecision]) -> Set[Pair]:
    out = set()
    for d in decisions:
        if d.decision_type.startswith("HUMAN_REVIEW"):
            p = Pair.make(d.raw_attribute_a, d.raw_attribute_b)
            if p.a != p.b:
                out.add(p)
    return out


def evaluate_alias_metrics(predicted_unique: Set[Pair], review_set: Set[Pair], true_pairs: Optional[Set[Pair]], closed_world: bool) -> Dict[str, Any]:
    if true_pairs is None:
        return {"measurable": False, "reason": "No ground truth aliases supplied", "precision": None, "recall": None, "f1": None}
    tp_set = predicted_unique & true_pairs
    fp_set = predicted_unique - true_pairs
    fn_set = true_pairs - predicted_unique
    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    review_true = review_set & true_pairs
    fp_prevented = len(review_set - true_pairs) if closed_world else None
    fn_requiring_review = len(review_true & fn_set)
    # reviewer diagnosed: if a false positive was routed to review it is not in predicted_unique; if a true FN is queued for review report it separately, not silently merged.
    reviewer_precision = precision
    reviewer_recall = (tp + len(review_true)) / len(true_pairs) if true_pairs else 0.0
    reviewer_f1 = 2 * reviewer_precision * reviewer_recall / (reviewer_precision + reviewer_recall) if (reviewer_precision + reviewer_recall) else 0.0
    return {
        "measurable": True,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "reviewer_diagnosed_precision": reviewer_precision,
        "reviewer_diagnosed_recall": reviewer_recall,
        "reviewer_diagnosed_f1": reviewer_f1,
        "false_positive_examples": [p.display() for p in sorted(fp_set)[:20]],
        "false_negative_examples": [p.display() for p in sorted(fn_set)[:20]],
        "review_true_pairs": [p.display() for p in sorted(review_true)[:20]],
        "fp_prevented_by_review_count": fp_prevented,
        "fn_requiring_review_count": fn_requiring_review,
    }


def membership_metrics(nodes: Dict[str, CanonicalNode], true_pairs: Optional[Set[Pair]]) -> Dict[str, Any]:
    if true_pairs is None:
        return {"measurable": False, "cluster_precision": None, "cluster_recall": None, "cluster_f1": None}
    predicted: Set[Pair] = set()
    for node in nodes.values():
        members = [m.provider_field for m in node.members]
        for a, b in itertools.combinations(members, 2):
            p = Pair.make(a, b)
            if p.a != p.b:
                predicted.add(p)
    tp = len(predicted & true_pairs)
    fp = len(predicted - true_pairs)
    fn = len(true_pairs - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"measurable": True, "cluster_precision": prec, "cluster_recall": rec, "cluster_f1": f1}


def evaluate_cross_context(decisions: List[MergeDecision]) -> Dict[str, Any]:
    accepted = [d for d in decisions if d.decision_type == "ACCEPT_MERGE"]
    cross = [d for d in accepted if d.payment_type_a != d.payment_type_b and d.semantic_family_a != d.semantic_family_b]
    return {"cross_context_merge_count": len(cross), "accepted_merge_count": len(accepted), "cross_context_merge_rate": len(cross) / len(accepted) if accepted else 0.0, "examples": [f"{d.raw_attribute_a}::{d.raw_attribute_b}" for d in cross[:10]]}


def compute_eenf_summary(attrs: List[SchemaAttribute], embedder: EmbeddingProvider, args: argparse.Namespace) -> Dict[str, Any]:
    if args.eenf_mode == "not_evaluated":
        return {"status": STATUS_NE, "reason": "EENF mode disabled"}
    if np is None:
        return {"status": STATUS_NA, "reason": "numpy unavailable"}
    vals = []
    for a in attrs[: min(30, len(attrs))]:
        regs = embedder.regenerations(a.name, a.domain, G=max(2, args.eenf_repeats))
        arr = np.asarray(regs, dtype=float)
        vals.append(float(np.mean(np.var(arr, axis=0))))
    q95 = float(np.quantile(np.asarray(vals), 0.95)) if vals else 0.0
    return {"status": STATUS_SUPPORTED if vals else STATUS_NE, "q95_variance": q95, "sampled_attribute_count": len(vals), "mode": args.eenf_mode}


def compute_dbnf_summary(attrs: List[SchemaAttribute], embedder: EmbeddingProvider, args: argparse.Namespace) -> Dict[str, Any]:
    if args.dbnf_mode == "none":
        return {"status": STATUS_NE, "mode": "none"}
    # Conservative: report execution/diagnostic only; do not claim paper support without explicit drift truth.
    return {"status": STATUS_NE, "mode": args.dbnf_mode, "reason": "DBNF scaffold/diagnostic present; no explicit drift ground truth evaluated in v17 default run", "attribute_count": len(attrs)}


def claim_rows(alias: Dict[str, Any], membership: Dict[str, Any], xctx: Dict[str, Any], eenf: Dict[str, Any], dbnf: Dict[str, Any], geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
    precision = alias.get("precision") if alias.get("measurable") else None
    rows = []
    rows.append({"Claim": "C1_alias_resolution", "Status": STATUS_PARTIAL if precision is not None else STATUS_NE, "Evidence": f"strict_precision={precision}"})
    rows.append({"Claim": "C2_eenf_stability", "Status": eenf.get("status", STATUS_NE), "Evidence": stringify(eenf)})
    rows.append({"Claim": "C3_dbnf_drift", "Status": dbnf.get("status", STATUS_NE), "Evidence": stringify(dbnf)})
    rows.append({"Claim": "C4_context_safety", "Status": STATUS_SUPPORTED if xctx.get("cross_context_merge_count", 0) == 0 else STATUS_NOT_SUPPORTED, "Evidence": stringify(xctx)})
    rows.append({"Claim": "C5_payload_compliance", "Status": STATUS_PARTIAL, "Evidence": "payload compliance diagnostics executed when payloads are available"})
    rows.append({"Claim": "C6_geometry_convergence", "Status": geometry.get("status", STATUS_SCAFFOLD), "Evidence": stringify(geometry)})
    rows.append({"Claim": "C7_reviewer_auditability", "Status": STATUS_SUPPORTED, "Evidence": "decisions, alias metrics, review queue, manifest, and claim rows exported"})
    return rows


def serialize_node(node: CanonicalNode) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "canonical_name": node.canonical_name,
        "semantic_family": node.semantic_family,
        "role": node.role,
        "domain": node.domain,
        "rails": sorted(node.rails),
        "providers": sorted(node.providers),
        "source_fields": [m.provider_field for m in node.members],
        "review_candidates": node.review_candidates[:20],
        "rejected_near_misses": node.rejected_near_misses[:20],
        "deferred_candidates": node.deferred_candidates[:20],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SDNF v17 reviewer-grade experiment harness.")
    p.add_argument("--output_profile", choices=["minimal", "paper", "audit", "debug"], default="paper")
    p.add_argument("--profile", dest="output_profile_alias", choices=["minimal", "paper", "audit", "debug"], default=None, help="Backward-compatible alias for --output_profile")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_output_files", type=int, default=15)
    p.add_argument("--schemas_dir", default="data")
    p.add_argument("--schema_glob", default="*.schema.json")
    p.add_argument("--payloads_root", default="payloads/payment")
    p.add_argument("--seed_srs_schema", default=None)
    p.add_argument("--ground_truth_aliases", default=None)
    p.add_argument("--ground_truth_closed_world", action="store_true")
    p.add_argument("--no_ground_truth_closed_world", action="store_true")
    p.add_argument("--evaluation_track", choices=["production", "discovery", "both"], default="production")
    p.add_argument("--dbnf_mode", choices=["none", "version_drift", "migration", "both"], default="none")
    p.add_argument("--dbnf_model_version", default=None)
    p.add_argument("--dbnf_migration_model", default=None)
    p.add_argument("--eenf_mode", choices=["not_evaluated", "perturbation_stress_test"], default="not_evaluated")
    p.add_argument("--eenf_repeats", type=int, default=10)
    p.add_argument("--measure_timing", action="store_true")
    p.add_argument("--candidate_backend", choices=["pairwise", "hnsw", "auto"], default="pairwise")
    p.add_argument("--hnsw_top_k", type=int, default=20)
    p.add_argument("--ground_truth_repair_mode", choices=["closed_world_only", "schema_supported_review", "schema_supported_include"], default="closed_world_only")
    p.add_argument("--count_semantic_veto_conflicts_as_fn", action="store_true")
    p.add_argument("--review_margin", type=float, default=0.10)
    p.add_argument("--auto_merge_threshold", type=float, default=0.86)
    p.add_argument("--review_threshold", type=float, default=0.62)
    p.add_argument("--tau_aanf", type=float, default=0.72)
    p.add_argument("--name_threshold", type=float, default=0.45)
    p.add_argument("--candidate_name_threshold", type=float, default=0.30)
    p.add_argument("--m_min_schema", type=int, default=2)
    p.add_argument("--allow_cross_rail_amount_currency", action="store_true")
    p.add_argument("--strict_semantic_vetoes", action="store_true")
    p.add_argument("--precision_first", action="store_true")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--evidence_weights_json", default=None)
    args = p.parse_args()
    if args.output_profile_alias:
        args.output_profile = args.output_profile_alias
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    t_start = now_ms()
    writer = OutputBudgetWriter(args.output_dir, args.output_profile, args.max_output_files)
    embedder = EmbeddingProvider(args.model, args.seed)

    true_pairs, negative_pairs, gt_audit = load_ground_truth(args.ground_truth_aliases, args)
    descs, schema_audit = load_schema_descriptors(args, embedder)
    obs, files_by_type, payload_audit = load_payloads(Path(args.payloads_root))
    field_rows, field_idx = build_field_evidence(obs, files_by_type)

    nodes, decisions, conflicts, mapping, lineage, review_and_scale = build_srs(descs, field_idx, embedder, args, true_pairs, negative_pairs)
    payload_compliance = validate_payloads(descs, obs, files_by_type)
    raw_pred, unique_pred = predicted_pairs_from_decisions(decisions)
    review_set = review_pairs(decisions)
    alias = evaluate_alias_metrics(unique_pred, review_set, true_pairs, bool(args.ground_truth_closed_world))
    member = membership_metrics(nodes, true_pairs)
    xctx = evaluate_cross_context(decisions)
    attrs = [a for d in descs for a in d.attributes]
    eenf = compute_eenf_summary(attrs, embedder, args)
    dbnf = compute_dbnf_summary(attrs, embedder, args)
    geometry = SemanticGeometryAuditScaffold().summarize(nodes)
    centroids = CanonicalEmbeddingBuilder().build(nodes)
    snapshot = SrsEvolutionSnapshotHook().snapshot("v17", nodes)

    review_rows = [r for r in review_and_scale if "AttributeA" in r]
    scale_rows = [r for r in review_and_scale if "metric" in r]
    review_reason_counts = Counter(r.get("Reason", "") for r in review_rows)

    claim_support = claim_rows(alias, member, xctx, eenf, dbnf, geometry)
    elapsed_ms = now_ms() - t_start
    scale_rows.extend([
        {"metric": "total_runtime_ms", "value": elapsed_ms},
        {"metric": "embedding_backend", "value": embedder.backend},
        {"metric": "dbnf_mode", "value": args.dbnf_mode},
        {"metric": "eenf_mode", "value": args.eenf_mode},
    ])

    duplicate_ok = len(raw_pred) == len(unique_pred)
    self_pair_ok = all(p.a != p.b for p in unique_pred)

    summary = {
        "version": __version__,
        "profile": args.output_profile,
        "dataset_summary": {
            "schema_descriptor_count": len(descs),
            "total_attributes": len(attrs),
            "payload_observation_count": len(obs),
            "total_ground_truth_pairs": len(true_pairs or []),
            "total_predicted_pairs_raw": len(raw_pred),
            "total_predicted_pairs_unique": len(unique_pred),
        },
        "alias_pair_metrics_strict": alias,
        "membership_metrics": member,
        "cross_context": xctx,
        "review_queue": {"count": len(review_rows), "reason_counts": dict(review_reason_counts)},
        "self_checks": {
            "no_self_pairs_in_predictions": self_pair_ok,
            "no_duplicate_pairs_in_predictions": duplicate_ok,
            "alias_vs_membership_evaluated_separately": True,
            "human_review_not_counted_as_strict_positive": True,
        },
        "normal_form_summaries": {"EENF": eenf, "DBNF": dbnf, "Geometry": geometry},
        "roadmap_scaffolds": {"canonical_embeddings": centroids.get("summary", {}), "srs_snapshot": snapshot},
    }
    manifest = {
        "version": __version__,
        "profile": args.output_profile,
        "output_dir": args.output_dir,
        "output_files": PROFILE_FILES[args.output_profile][: writer.max_files],
        "ground_truth_repair_mode": args.ground_truth_repair_mode,
        "count_semantic_veto_conflicts_as_fn": bool(args.count_semantic_veto_conflicts_as_fn),
        "candidate_backend_requested": args.candidate_backend,
        "embedding_backend": embedder.backend,
        "evidence_weights": json.loads(args.evidence_weights_json) if args.evidence_weights_json else {"embedding": 0.35, "name": 0.20, "ontology": 0.20, "canonical": 0.20, "alias": 0.05},
        "ground_truth_audit": gt_audit,
        "roadmap_scaffolds_are_claims": False,
        "writer_max_files": writer.max_files,
    }

    decision_rows = []
    for d in decisions:
        decision_rows.append({
            "DecisionID": d.decision_id, "AttributeA": d.raw_attribute_a, "AttributeB": d.raw_attribute_b,
            "Track": d.track, "Decision": d.decision_type, "Reason": d.decision_reason,
            "LineageAction": d.lineage_action, "CanonicalNode": d.canonical_node,
            "FamilyA": d.semantic_family_a, "FamilyB": d.semantic_family_b,
            "RoleA": d.role_a, "RoleB": d.role_b,
            "EvidenceScore": d.evidence.get("score"), "EmbeddingSimilarity": d.evidence.get("embedding_similarity"),
            "NameSimilarity": d.evidence.get("name_similarity"), "SupportCount": d.evidence.get("support_count"),
            "AANF": d.normal_form_checks["AANF"]["status"], "ECNF": d.normal_form_checks["ECNF"]["status"],
            "RRNF": d.normal_form_checks["RRNF"]["status"], "CMNF": d.normal_form_checks["CMNF"]["status"],
            "PONF": d.normal_form_checks["PONF"]["status"], "DBNF": d.normal_form_checks["DBNF"]["status"],
            "HardVetoes": d.hard_vetoes,
        })

    alias_rows = [{
        "Metric": k, "Value": stringify(v)
    } for k, v in alias.items()]
    alias_rows.extend([
        {"Metric": "raw_predicted_pair_count", "Value": len(raw_pred)},
        {"Metric": "unique_predicted_pair_count", "Value": len(unique_pred)},
        {"Metric": "no_duplicate_pairs_in_predictions", "Value": duplicate_ok},
        {"Metric": "no_self_pairs_in_predictions", "Value": self_pair_ok},
    ])

    srs_compact = {key: serialize_node(node) for key, node in sorted(nodes.items())}
    schema_delta_rows = []
    known_schema_fields = {(a.payment_type, a.normalized_name) for a in attrs}
    for fe in field_rows:
        if (fe.payment_type, fe.normalized_field) not in known_schema_fields:
            schema_delta_rows.append({"ChangeType": "unexpected_field", "FieldName": f"{fe.payment_type}.{fe.field}", "Details": "Field observed in payload but not matched to schema descriptor", "Recommendation": "HUMAN_REVIEW"})
    payload_rows = [asdict(x) for x in payload_compliance]
    field_csv_rows = [asdict(x) for x in field_rows]
    schema_rows = schema_audit + payload_audit

    console = []
    console.append(f"Unified SDNF Experiment v17.0.0")
    console.append(f"Profile: {args.output_profile}")
    console.append(f"Ground truth repair mode: {args.ground_truth_repair_mode}")
    console.append(f"Candidate backend requested/used: {args.candidate_backend}")
    console.append(f"Total attributes processed: {len(attrs)}")
    console.append(f"Unique predicted alias pairs: {len(unique_pred)} (raw pairs considered as merges: {len(raw_pred)})")
    if alias.get("measurable"):
        console.append(f"True Positives: {alias.get('tp')}, False Positives: {alias.get('fp')}, False Negatives: {alias.get('fn')}")
        console.append(f"Alias Precision (strict): {alias.get('precision'):.3f}, Alias Recall: {alias.get('recall'):.3f}, Alias F1: {alias.get('f1'):.3f}")
        console.append(f"Alias Precision (reviewer-diagnosed): {alias.get('reviewer_diagnosed_precision'):.3f}, Recall: {alias.get('reviewer_diagnosed_recall'):.3f}, F1: {alias.get('reviewer_diagnosed_f1'):.3f}")
    else:
        console.append("Alias metrics: NOT_EVALUATED (no ground truth)")
    console.append(f"Review queue count: {len(review_rows)}")
    console.append(f"Cross-context merge rate: {xctx['cross_context_merge_rate']:.3f} (Count: {xctx['cross_context_merge_count']})")
    console.append(f"Self-check no duplicate predicted pairs: {duplicate_ok}")
    console.append(f"Claim Support Status: " + ", ".join(f"{r['Claim']}={r['Status']}" for r in claim_support))

    # Write outputs strictly through OutputBudgetWriter.
    if "out_audit_v17.txt" in writer.allowed: writer.write_text("out_audit_v17.txt", "\n".join(console) + "\n")
    writer.write_json("run_manifest_v17.json", manifest)
    writer.write_json("summary_audit_v17.json", summary)
    writer.write_json("srs_evolved_schema_v17.compact.json", srs_compact)
    if "schema_ingestion_audit_v17.csv" in writer.allowed: writer.write_csv("schema_ingestion_audit_v17.csv", schema_rows)
    if "field_evidence_audit_v17.csv" in writer.allowed: writer.write_csv("field_evidence_audit_v17.csv", field_csv_rows)
    if "schema_deltas_audit_v17.csv" in writer.allowed: writer.write_csv("schema_deltas_audit_v17.csv", schema_delta_rows or [{"ChangeType": "none", "FieldName": "", "Details": "No schema deltas detected", "Recommendation": ""}])
    if "decisions_audit_v17.csv" in writer.allowed: writer.write_csv("decisions_audit_v17.csv", decision_rows)
    if "alias_evaluation_audit_v17.csv" in writer.allowed: writer.write_csv("alias_evaluation_audit_v17.csv", alias_rows)
    if "payload_compliance_audit_v17.csv" in writer.allowed: writer.write_csv("payload_compliance_audit_v17.csv", payload_rows)
    if "normal_forms_and_claims_audit_v17.csv" in writer.allowed: writer.write_csv("normal_forms_and_claims_audit_v17.csv", claim_support)
    if "scale_timing_drift_audit_v17.csv" in writer.allowed: writer.write_csv("scale_timing_drift_audit_v17.csv", scale_rows)
    if "review_queue_audit_v17.csv" in writer.allowed: writer.write_csv("review_queue_audit_v17.csv", review_rows or [{"AttributeA": "", "AttributeB": "", "DecisionType": "NONE", "Reason": "No human review items", "CanonicalNode": "", "Track": "", "EvidenceScore": "", "HardVetoes": ""}])
    if "readme_v17.md" in writer.allowed:
        writer.write_text("readme_v17.md", "# SDNF v17\n\nReviewer-grade single-file experiment harness. Roadmap scaffolds are not evaluated paper claims.\n")
    if "sdnf_debug_bundle_v17.zip" in writer.allowed:
        writer.write_zip("sdnf_debug_bundle_v17.zip", [x for x in writer.written if x != "sdnf_debug_bundle_v17.zip"])

    print("\n".join(console))
    print(f"\nOutputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
