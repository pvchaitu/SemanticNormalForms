# preprocessing.py
import re

def preprocess(names, source):
    """
    Standardizes attribute names for the SDNF embedding pipeline.
    """
    results = []
    for name in names:
        norm = name.lower().strip()
        norm = re.sub(r'[^a-z0-9]', '', norm)
        results.append({
            "original": name,
            "normalized": norm,
            "source": source
        })
    return results
