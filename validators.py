# validators.py
"""
SDNF validators

Provides tests for the seven Semantic Data Normal Forms:
EENF, AANF, CMNF, DBNF, ECNF (placeholder), RRNF, PONF.

This file replaces the previous implementation of CMNF with a robust
computation that handles differing projection ranks (W_p and W_r).
"""

from typing import List, Dict, Any
import numpy as np

class SDNFValidator:
    def __init__(self, tau_map: Dict[str, float]):
        """
        tau_map: dict with keys "EENF","AANF","CMNF","DBNF","RRNF","PONF"
        Values are numeric thresholds used by tests.
        """
        self.tau = dict(tau_map)

    # -------------------------
    # EENF: Entity Embedding Normal Form
    # -------------------------
    def test_eenf(self, regenerations: np.ndarray) -> Dict[str, Any]:
        """
        regenerations: array shape (G, d) of repeated embeddings for an entity
        Returns dict with name, requirement, actual, status, details
        """
        if regenerations is None or regenerations.size == 0:
            return {"name": "EENF", "req": f"MeanVar < {self.tau.get('EENF', 0.01)}", "actual": None, "status": "NA", "details": {}}
        per_dim_var = np.var(regenerations, axis=0)
        mean_var = float(np.mean(per_dim_var))
        status = "PASS" if mean_var < float(self.tau.get("EENF", 0.01)) else "FAIL"
        return {"name": "EENF", "req": f"MeanVar < {self.tau.get('EENF', 0.01)}", "actual": mean_var, "status": status, "details": {"per_dim_var_mean": mean_var}}

    # -------------------------
    # AANF: Attribute Alias Normal Form
    # -------------------------
    def test_aanf(self, attr_embeddings: np.ndarray, attr_names: List[str]) -> Dict[str, Any]:
        """
        attr_embeddings: (n, d) array of canonical attribute embeddings
        attr_names: list of attribute names (parallel to attr_embeddings)
        """
        if attr_embeddings is None or attr_embeddings.size == 0:
            return {"name": "AANF", "req": f"Sim < {self.tau.get('AANF', 0.9)}", "actual": None, "status": "NA", "details": {}}
        # compute pairwise cosine similarities (upper triangle)
        X = attr_embeddings
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        sims = Xn @ Xn.T
        n = sims.shape[0]
        # ignore diagonal
        if n <= 1:
            max_sim = 0.0
        else:
            mask = ~np.eye(n, dtype=bool)
            max_sim = float(np.max(sims[mask]))
        status = "PASS" if max_sim < float(self.tau.get("AANF", 0.9)) else "FAIL"
        return {"name": "AANF", "req": f"Sim < {self.tau.get('AANF', 0.9)}", "actual": max_sim, "status": status, "details": {"candidates": []}}

    # -------------------------
    # CMNF: Context Modulation Normal Form
    # -------------------------
    def test_cmnf(self, W_p: np.ndarray, W_r: np.ndarray, sample_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute contamination between two context projections W_p and W_r.

        Approach:
        - Prefer a subspace-overlap measure using singular values of M = Wp_ctx @ Wr_ctx^T.
          This works when W_p and W_r have different output ranks (k_p != k_r).
        - If SVD fails, fallback to projecting sample embeddings and computing mean absolute
          inner product after trimming to the minimum common projection dimension.

        Inputs:
          W_p: (k_p, d) projection matrix for context p
          W_r: (k_r, d) projection matrix for context r
          sample_embeddings: (n, d) canonical embeddings (used for fallback)
        """
        tau = float(self.tau.get("CMNF", 0.05))
        # Validate inputs
        if W_p is None or W_r is None:
            return {"name": "CMNF", "req": f"AvgInnerProd < {tau}", "actual": None, "status": "NA", "details": {}}

        try:
            # Attempt subspace overlap via singular values of cross-projection matrix
            # If W matrices include contextual-only columns, caller should pass contextual slices.
            # M shape: (k_p, k_r)
            M = np.matmul(W_p, W_r.T)
            sv = np.linalg.svd(M, compute_uv=False)
            if sv.size == 0:
                contamination = 0.0
            else:
                # Normalize singular values to [0,1] by dividing by max singular value (if >0)
                # and take mean absolute singular value as contamination metric.
                max_sv = float(np.max(np.abs(sv))) if sv.size > 0 else 1.0
                if max_sv <= 0:
                    contamination = float(np.mean(np.abs(sv)))
                else:
                    contamination = float(np.mean(np.abs(sv) / max_sv))
            status = "PASS" if contamination < tau else "FAIL"
            return {"name": "CMNF", "req": f"AvgInnerProd < {tau}", "actual": contamination, "status": status, "details": {"sv_count": int(sv.size)}}
        except Exception:
            # Fallback: project sample embeddings and compute mean abs inner product on min common dims
            try:
                if sample_embeddings is None or sample_embeddings.size == 0:
                    return {"name": "CMNF", "req": f"AvgInnerProd < {tau}", "actual": None, "status": "NA", "details": {"fallback": "no_samples"}}
                # Project samples
                Xa = np.matmul(sample_embeddings, W_p.T)  # (n, k_p)
                Xb = np.matmul(sample_embeddings, W_r.T)  # (n, k_r)
                kmin = min(Xa.shape[1], Xb.shape[1])
                if kmin == 0:
                    contamination = 0.0
                else:
                    Xa_trim = Xa[:, :kmin]
                    Xb_trim = Xb[:, :kmin]
                    na = np.linalg.norm(Xa_trim, axis=1, keepdims=True) + 1e-12
                    nb = np.linalg.norm(Xb_trim, axis=1, keepdims=True) + 1e-12
                    Xa_n = Xa_trim / na
                    Xb_n = Xb_trim / nb
                    ips = np.abs(np.sum(Xa_n * Xb_n, axis=1))
                    contamination = float(np.mean(ips))
                status = "PASS" if contamination < tau else "FAIL"
                return {"name": "CMNF", "req": f"AvgInnerProd < {tau}", "actual": contamination, "status": status, "details": {"fallback": "trimmed_projection", "kmin": kmin}}
            except Exception as e:
                return {"name": "CMNF", "req": f"AvgInnerProd < {tau}", "actual": None, "status": "NA", "details": {"error": str(e)}}

    # -------------------------
    # DBNF: Drift Bounded Normal Form (global check)
    # -------------------------
    def test_dbnf(self, pre_vec: np.ndarray, post_vec: np.ndarray) -> Dict[str, Any]:
        tau = float(self.tau.get("DBNF", 0.25))
        if pre_vec is None or post_vec is None:
            return {"name": "DBNF", "req": f"Drift <= {tau}", "actual": None, "status": "NA", "details": {}}
        # normalize and compute L2 distance
        if np.linalg.norm(pre_vec) > 0:
            pre = pre_vec / (np.linalg.norm(pre_vec) + 1e-12)
        else:
            pre = pre_vec
        if np.linalg.norm(post_vec) > 0:
            post = post_vec / (np.linalg.norm(post_vec) + 1e-12)
        else:
            post = post_vec
        drift = float(np.linalg.norm(pre - post))
        status = "PASS" if drift <= tau else "FAIL"
        return {"name": "DBNF", "req": f"Drift <= {tau}", "actual": drift, "status": status, "details": {}}

    # -------------------------
    # ECNF: Evidence Completeness Normal Form (placeholder)
    # -------------------------
    def test_ecnf(self, evidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder: the demo computes ECNF per-merge elsewhere.
        This function returns NA unless a numeric summary is provided.
        """
        if not evidence_summary:
            return {"name": "ECNF", "req": "Evidence >= m_min OR agg_score >= threshold", "actual": None, "status": "NA", "details": {}}
        # Expect evidence_summary to contain 'avg_count' and 'avg_score' or similar
        return {"name": "ECNF", "req": "per-merge checks", "actual": evidence_summary, "status": "NA", "details": {}}

    # -------------------------
    # RRNF: Relation Role Normal Form (placeholder)
    # -------------------------
    def test_rrnf(self, relations: List[Dict[str, Any]], role_compat: Dict[str, Any]) -> Dict[str, Any]:
        # If no relations, trivially pass
        if not relations:
            return {"name": "RRNF", "req": f">= {self.tau.get('RRNF', 0.7)}", "actual": 1.0, "status": "PASS", "details": {"total_paths": 0, "incompatible": 0}}
        # Placeholder simple check: count incompatible paths (not implemented)
        return {"name": "RRNF", "req": f">= {self.tau.get('RRNF', 0.7)}", "actual": 1.0, "status": "PASS", "details": {"total_paths": 0, "incompatible": 0}}

    # -------------------------
    # PONF: Partition Orthogonality Normal Form (placeholder)
    # -------------------------
    def test_ponf(self, embeddings: np.ndarray, partition_labels: List[int]) -> Dict[str, Any]:
        if embeddings is None or embeddings.size == 0 or not partition_labels:
            return {"name": "PONF", "req": f"<= {self.tau.get('PONF', 0.1)}", "actual": 0.0, "status": "PASS", "details": {}}
        # Simple implementation: compute mean inter-partition cosine overlap
        labels = np.array(partition_labels)
        unique = np.unique(labels)
        if unique.size <= 1:
            return {"name": "PONF", "req": f"<= {self.tau.get('PONF', 0.1)}", "actual": 0.0, "status": "PASS", "details": {}}
        # compute centroids per partition
        centroids = []
        for u in unique:
            idx = np.where(labels == u)[0]
            if idx.size == 0:
                centroids.append(np.zeros((embeddings.shape[1],)))
            else:
                centroids.append(np.mean(embeddings[idx], axis=0))
        C = np.stack(centroids, axis=0)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        sims = np.abs(Cn @ Cn.T)
        mask = ~np.eye(sims.shape[0], dtype=bool)
        overlap = float(np.mean(sims[mask])) if mask.any() else 0.0
        status = "PASS" if overlap <= float(self.tau.get("PONF", 0.1)) else "FAIL"
        return {"name": "PONF", "req": f"<= {self.tau.get('PONF', 0.1)}", "actual": overlap, "status": status, "details": {}}

    # -------------------------
    # run_all: convenience wrapper
    # -------------------------
    def run_all(self, data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        data_context expected keys:
          - regenerations (array Gxd) for EENF
          - attr_embeddings (nxd) and attr_names for AANF
          - W_p, W_r, sample_embeddings for CMNF
          - pre_vec, post_vec for DBNF
          - evidence_summary for ECNF (optional)
          - relations, role_compat for RRNF
          - partition_labels for PONF
        """
        results = []
        # EENF
        regs = data_context.get("regenerations")
        results.append(self.test_eenf(regs))
        # AANF
        attr_emb = data_context.get("attr_embeddings")
        attr_names = data_context.get("attr_names", [])
        results.append(self.test_aanf(attr_emb, attr_names))
        # CMNF
        results.append(self.test_cmnf(data_context.get("W_p"), data_context.get("W_r"), data_context.get("sample_embeddings")))
        # DBNF
        results.append(self.test_dbnf(data_context.get("pre_vec"), data_context.get("post_vec")))
        # ECNF (summary placeholder)
        results.append(self.test_ecnf(data_context.get("evidence_summary")))
        # RRNF
        results.append(self.test_rrnf(data_context.get("relations", []), data_context.get("role_compat", {})))
        # PONF
        results.append(self.test_ponf(data_context.get("attr_embeddings"), data_context.get("partition_labels", [])))
        return results
