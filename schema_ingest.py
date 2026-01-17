# schema_ingest.py
import re
import json
import logging
import datetime
import os
from typing import Dict, Any, List

logger = logging.getLogger("sdnf.ingest")


def load_json_file(filename: str):
    """Look for file in given path or in ./data/; return parsed JSON or None."""
    paths = [filename, os.path.join("data", filename)]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    logger.error(f"File not found: {filename}")
    return None


def validate_payload(payload: Dict[str, Any], required_fields: List[str]) -> bool:
    """Simple required-field validator; logs missing fields."""
    missing = [f for f in required_fields if f not in payload]
    if missing:
        logger.error(f"Payload validation failed. Missing fields: {missing}")
        return False
    return True


def infer_type(value):
    """
    Infer a coarse type for a payload value.
    Kept intentionally simple — extendable for domain-specific heuristics.
    """
    if isinstance(value, str) and re.match(r"^[0-9\s\-]{13,19}$", value):
        return "pan_type"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "string"


def derive_schema_from_payload(payload: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """
    Derive a minimal semantic schema object from a single JSON payload.

    Output structure:
    {
      "entity": "DerivedFrom_<source>",
      "attributes": [
         {"name": "<field>", "type": "<inferred>", "aliases": [], "provenance": {"source": source, "first_seen": timestamp}}
      ],
      "derived_at": "<iso timestamp>"
    }
    """
    derived = {
        "entity": f"DerivedFrom_{source}",
        "attributes": [],
        "derived_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    for k, v in payload.items():
        attribute = {
            "name": str(k),
            "type": infer_type(v),
            "aliases": [],
            "constraints": None,
            "provenance": {"source": source, "first_seen": datetime.datetime.utcnow().isoformat() + "Z"},
        }
        # Small heuristics to populate constraints
        if attribute["type"] == "pan_type":
            attribute["constraints"] = {"pattern": r"^[0-9\s\-]{13,19}$"}
        derived["attributes"].append(attribute)
    return derived
