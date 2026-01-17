# validators.py
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List

logger = logging.getLogger("sdnf.validators")


class SDNFValidator:
    """
    Consolidated SDNF validator suite. Each test returns a dict with fields:
    {name, req, actual, status, details}

    Thresholds: dictionary containing keys for EENF, AANF, CMNF, DBNF defaults.
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.tau = thresholds.copy()

    def test_eenf(self, embedding_regenerations: np.ndarray) -> Dict[str, Any]:
        """
        embedding_regenerations: shape (G, d) for a single entity or (n, d) sample.
        We compute average per-dimension variance and use its mean as the scalar metric.
        """
        arr = np.asarray(embedding_regenerations)
        # compute per-dimension variance
        var = np.var(arr, axis=0)
        metric = float(np.mean(var))
        status = "PASS" if metric < self.tau.get("EENF", 1e-3) else "FAIL"
        return {"name": "EENF", "req": f"MeanVar < {self.tau.get('EENF')}", "actual": f"{metric:.6f}", "status": status, "details": {"per_dim_var_mean": metric}}

    def test_aanf(self, attribute_embeddings: np.ndarray, attribute_names: List[str]) -> Dict[str, Any]:
        """
        Compute pairwise cosine similarities and find top cross-attribute similarity.
        We return both the maximum similarity for non-identical names and
        candidate merge pairs above the threshold for human review.
        """
        X = np.asarray(attribute_embeddings)
        if X.shape[0] < 2:
            return {"name": "AANF", "req": f"Sim < {self.tau.get('AANF')}", "actual": "NA (single attribute)", "status": "PASS", "details": {}}

        sims = cosine_similarity(X)
        # zero out diagonal
        np.fill_diagonal(sims, -999.0)
        max_sim = float(np.max(sims))
        # candidate pairs
        pairs = []
        th = self.tau.get("AANF", 0.88)
        idxs = np.argwhere(sims >= th)
        for i, j in idxs:
            pairs.append({"a": attribute_names[i], "b": attribute_names[j], "sim": float(sims[i, j])})
        status = "PASS" if max_sim < th else "FAIL"
        return {"name": "AANF", "req": f"Sim < {th}", "actual": f"{max_sim:.4f}", "status": status, "details": {"candidates": pairs}}

    def test_cmnf(self, W_a: np.ndarray, W_b: np.ndarray, sample_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Test CMNF contamination via approximate sample-based contamination metric.
        """
        # compute contamination using cmnf.approximate_context_contamination equivalent logic
        # avoid circular import: do inline
        Xs = sample_embeddings
        Xa = Xs @ W_a.T
        Xb = Xs @ W_b.T
        na = np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12
        nb = np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12
        Xa_n = Xa / na
        Xb_n = Xb / nb
        ips = np.abs(np.sum(Xa_n * Xb_n, axis=1))
        contamination = float(np.mean(ips))
        status = "PASS" if contamination < self.tau.get("CMNF", 0.02) else "FAIL"
        return {"name": "CMNF", "req": f"AvgInnerProd < {self.tau.get('CMNF')}", "actual": f"{contamination:.6f}", "status": status, "details": {}}

    def test_dbnf(self, pre_vec: np.ndarray, post_vec: np.ndarray, tau_dbnf: float) -> Dict[str, Any]:
        """
        DBNF: test that L2 drift between pre & post embeddings is <= tau_dbnf.
        """
        d = float(np.linalg.norm(pre_vec - post_vec))
        status = "PASS" if d <= tau_dbnf else "FAIL"
        return {"name": "DBNF", "req": f"Drift <= {tau_dbnf}", "actual": f"{d:.6f}", "status": status, "details": {}}

    def run_all(self, data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        data_context keys:
          - 'regenerations': ndarray (G, d)
          - 'attr_embeddings': ndarray (n, d)
          - 'attr_names': list[str]
          - 'W_p', 'W_r': projection matrices
          - 'sample_embeddings': ndarray (m, d)
          - 'pre_vec', 'post_vec': optional for DBNF
        """
        results = []
        if "regenerations" in data_context:
            results.append(self.test_eenf(data_context["regenerations"]))
        if "attr_embeddings" in data_context and "attr_names" in data_context:
            results.append(self.test_aanf(data_context["attr_embeddings"], data_context["attr_names"]))
        if "W_p" in data_context and "W_r" in data_context and "sample_embeddings" in data_context:
            results.append(self.test_cmnf(data_context["W_p"], data_context["W_r"], data_context["sample_embeddings"]))
        if "pre_vec" in data_context and "post_vec" in data_context and "tau_dbnf" in data_context:
            results.append(self.test_dbnf(data_context["pre_vec"], data_context["post_vec"], data_context["tau_dbnf"]))
        return results
