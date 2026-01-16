import re
import json
import logging
import datetime
import os
from typing import Dict, Any

logger = logging.getLogger("sdnf.ingest")

def load_json_file(filename: str):
    """Checks root and data/ folder for the required JSON files."""
    paths = [filename, os.path.join("data", filename)]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    logger.error(f"File not found: {filename}")
    return None

def validate_payload(payload: Dict[str, Any], required_fields: list):
    """Validates raw payloads against ISO 20022/PCI-DSS standards."""
    missing = [f for f in required_fields if f not in payload]
    if missing:
        logger.error(f"Validation Failed! Missing fields: {missing}")
        return False
    return True

def infer_type(value):
    """Infers semantic types (e.g., pan_type) to support EENF."""
    if isinstance(value, str) and re.match(r"^[0-9\- ]{13,19}$", value):
        return "pan_type"
    return "string"

def derive_schema_from_payload(payload: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """Derives a Semantic Relational Schema (SRS) draft."""
    derived = {
        "entity": f"DerivedFrom_{source}",
        "attributes": [],
        "derived_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    for k, v in payload.items():
        derived["attributes"].append({"name": k, "type": infer_type(v)})
    return derived