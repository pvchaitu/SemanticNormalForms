import numpy as np
import json
from tabulate import tabulate
from schema_ingest import load_json_file, validate_payload, derive_schema_from_payload
from embedding_utils import EmbeddingModel, add_gaussian_dp_noise
from semantic_merge import SemanticMerger
from cmnf import learn_linear_projection
from validators import SDNFValidator

def run_sdnf_research_demo():
    # 1. LOAD & INITIAL STATE (Pre-Evolution)
    # ---------------------------------------------------------
    INAmex = load_json_file("INAmex.json")
    PPVisa_data = load_json_file("PPVisa.json")
    
    if not INAmex or not PPVisa_data:
        print("[ERROR] JSON data files missing.")
        return

    if not validate_payload(PPVisa_data, ["pan", "cvv"]): 
        return

    print("\n--- [STAGE 1] SRS STATE BEFORE SEMANTIC MERGE ---")
    # Showing the base schema before any aliases are learned
    print(json.dumps(INAmex, indent=2))

    # 2. EMBEDDINGS (EENF/PONF)
    model = EmbeddingModel()
    all_names = [a['name'] for a in INAmex['attributes']] + list(PPVisa_data.keys())
    vecs = model.encode([n.lower() for n in all_names])
    
    # Apply Differential Privacy noise to the first embedding (PONF compliance)
    vecs[0] = add_gaussian_dp_noise(vecs[0]) 

    # 3. EVOLUTION (AANF/DBNF/ECNF)
    # ---------------------------------------------------------
    merger = SemanticMerger(tau_dbnf=0.15, m_min=3)
    
    # Mocking evidence: we found 3 instances where 'pan' maps to 'PrimaryAccountNumber'
    # This satisfies ECNF (m_min=3)
    evidence = {"pan": [{"target": "PrimaryAccountNumber"}] * 3}
    
    # Execute the merge to evolve the schema
    derived_schema = derive_schema_from_payload(PPVisa_data, source="PPVisa")
    evolution = merger.execute_merge(
        INAmex, 
        derived_schema, 
        evidence, 
        pre_vec=vecs[0], 
        post_vec=vecs[0] * 1.05  # Simulated post-merge embedding vector
    )

    print("\n--- [STAGE 2] SRS STATE AFTER SEMANTIC MERGE (AANF EVOLVED) ---")
    # The 'after' schema contains the updated attribute list with new aliases
    print(json.dumps(evolution['after'], indent=2))

    # 4. NORMAL FORM VALIDATION (CMNF/EENF/DBNF)
    # ---------------------------------------------------------
    # Learn projections for two contexts (e.g., Payment vs Risk)
    W_p = learn_linear_projection(vecs, vecs)
    W_r = learn_linear_projection(vecs, vecs*1.1, orth_penalty=0.01, other_W=W_p)
    
    validator = SDNFValidator(thresholds={'EENF': 0.1, 'AANF': 0.85, 'CMNF': 0.05})
    results = validator.run_all({
        'samples': np.random.normal(0, 0.04, (10, vecs.shape[1])), 
        'embeddings': vecs, 
        'W_p': W_p, 
        'W_r': W_r
    })
    
    # Append DBNF (Drift) result to the validation table
    results.append({
        "name": "DBNF", 
        "req": f"Drift < {merger.tau_dbnf}", 
        "actual": f"{evolution['dbnf']['drift']:.4f}", 
        "status": "PASS" if evolution['dbnf']['pass'] else "FAIL"
    })

    print("\nTABLE I: SDNF COMPLIANCE VERDICTS (IEEE MANUSCRIPT)")
    print(tabulate(results, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    run_sdnf_research_demo()