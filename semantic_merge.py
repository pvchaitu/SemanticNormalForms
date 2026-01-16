import logging
import numpy as np

logger = logging.getLogger("sdnf.merge")

class SemanticMerger:
    def __init__(self, m_min=3, tau_dbnf=0.15):
        self.m_min = m_min  # ECNF threshold
        self.tau_dbnf = tau_dbnf  # DBNF drift threshold

    def check_ecnf(self, evidence_set):
        """ECNF: Verify merge has sufficient evidence."""
        count = len(evidence_set)
        return count >= self.m_min, count

    def check_dbnf(self, pre_vec, post_vec):
        """DBNF: Verify semantic drift is within bounds."""
        drift = np.linalg.norm(pre_vec - post_vec)
        is_compliant = drift <= self.tau_dbnf
        return is_compliant, drift

    def execute_merge(self, srs_base, derived_schema, evidence_map, pre_vec=None, post_vec=None):
        """Executes AANF/ECNF evolution and validates DBNF stability."""
        post_attrs = [dict(a) for a in srs_base['attributes']]
        dbnf_res = {"pass": True, "drift": 0.0}
        
        for attr in derived_schema['attributes']:
            evidence = evidence_map.get(attr['name'], [])
            ecnf_pass, _ = self.check_ecnf(evidence)
            if ecnf_pass:
                target = evidence[0]['target']
                for base in post_attrs:
                    if base['name'] == target and attr['name'] not in base['aliases']:
                        base['aliases'].append(attr['name'])
        
        if pre_vec is not None and post_vec is not None:
            is_valid, val = self.check_dbnf(pre_vec, post_vec)
            dbnf_res = {"pass": is_valid, "drift": val}
            
        return {"after": {"attributes": post_attrs}, "dbnf": dbnf_res}