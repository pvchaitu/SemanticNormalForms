# cmnf.py
"""
Context Modulation helpers.

Provides learn_linear_projection(X, k, orthogonalize_iters, other_W) which returns a projection matrix W (k x d).
Iterative orthogonalization reduces contamination between contexts.
"""

import numpy as np

def learn_linear_projection(X: np.ndarray, k: int = None, orthogonalize_iters: int = 0, other_W: np.ndarray = None):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n x d)")
    n, d = X.shape
    if k is None:
        k = min(64, d)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:k, :].astype(np.float32)
    except Exception:
        rng = np.random.default_rng(0)
        A = rng.normal(size=(k, d))
        Q, R = np.linalg.qr(A.T)
        W = Q.T[:k, :].astype(np.float32)

    # iterative orthogonalization relative to other_W
    if other_W is not None and orthogonalize_iters > 0:
        OW = np.asarray(other_W, dtype=np.float32)
        for _ in range(orthogonalize_iters):
            try:
                P = OW.T @ np.linalg.pinv(OW @ OW.T)
                for i in range(W.shape[0]):
                    w = W[i:i+1, :].T
                    proj = P @ (OW @ w)
                    w_new = w - proj
                    W[i:i+1, :] = (w_new.T)
                # re-orthonormalize rows
                U2, S2, Vt2 = np.linalg.svd(W, full_matrices=False)
                W = Vt2[:W.shape[0], :].astype(np.float32)
            except Exception:
                break
    return W.astype(np.float32)
