# cmnf.py
import numpy as np
import logging
from sklearn.decomposition import PCA
from typing import Optional

logger = logging.getLogger("sdnf.cmnf")


def learn_linear_projection(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3, orth_penalty: Optional[float] = None, other_W: Optional[np.ndarray] = None, max_iters: int = 100) -> np.ndarray:
    """
    Learn a linear projection W that maps X -> Y (least squares / ridge).
    Optionally apply an orthogonality penalty to keep W approximately orthogonal
    to other_W (useful for CMNF).
    Returns W (d x d).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    d = X.shape[1]
    if X.shape[0] < d:
        # regularize by identity to avoid singular matrices
        reg_adj = reg * np.eye(d)
    else:
        reg_adj = reg * np.eye(d)

    # Solve W^T = (X^T X + reg I)^{-1} X^T Y -> W = Y^T X (X^T X + reg I)^{-1}
    A = X.T @ X + reg_adj
    B = X.T @ Y
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    W = (B.T @ A_inv)  # shape (d, d) if Y has dim d

    if orth_penalty and other_W is not None:
        # simple iterative orthogonality refinement (gradient step)
        for i in range(max_iters):
            # penalty gradient: grad = 2 * orth_penalty * (W @ other_W.T) @ other_W
            grad = 2.0 * orth_penalty * (W @ other_W.T) @ other_W
            # step-size heuristics (small)
            step = 1e-3
            W = W - step * grad
            # small convergence check
            if np.linalg.norm(step * grad) < 1e-6:
                break

    return W


def batch_project(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project rows X (n x d) using W (d x d) -> returns (n x d)"""
    return (X @ W.T)


def approximate_context_contamination(W_a: np.ndarray, W_b: np.ndarray, X_sample: np.ndarray, sample_size: int = 512) -> float:
    """
    Compute approximate contamination metric between two projections by sampling
    a subset of primitives and measuring average inner product between their
    projected vectors.

    This avoids O(N^2) pairwise checks and provides a practical scalar contamination.
    """
    n = X_sample.shape[0]
    idx = np.random.default_rng(42).choice(np.arange(n), size=min(sample_size, n), replace=False)
    Xs = X_sample[idx]
    Xa = batch_project(W_a, Xs)
    Xb = batch_project(W_b, Xs)
    # normalize each row
    na = np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12
    nb = np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12
    Xa_n = Xa / na
    Xb_n = Xb / nb
    # average absolute inner product
    ips = np.abs(np.sum(Xa_n * Xb_n, axis=1))
    contamination = float(np.mean(ips))
    return contamination
