# preprocessing.py
import re

def preprocess(names, source):
    """
    Standardizes attribute names for the SDNF embedding pipeline.
    
    Args:
        names (list): List of raw attribute names.
        source (str): The source file name (e.g., 'INAmex.json').
        
    Returns:
        list: A list of dictionaries containing the 'normalized' name.
    """
    results = []
    for name in names:
        # Step 1: Basic normalization (lowercase, strip whitespace)
        norm = name.lower().strip()
        
        # Step 2: Basic regex to handle common camelCase/snake_case issues
        # (Example: 'cardHolder' -> 'cardholder')
        norm = re.sub(r'[^a-z0-9]', '', norm)
        
        results.append({
            "original": name,
            "normalized": norm,
            "source": source
        })
    return results