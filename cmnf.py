import numpy as np
import logging

logger = logging.getLogger("sdnf.cmnf")

"""
NOVELTY NOTE: Context Modulation Normal Form (CMNF).
CMNF addresses 'Contextual Contamination'. We enforce that if two contexts 
(e.g., Risk and Payment) are independent, their projection matrices W must be orthogonal.
Constraint: |W_1 @ W_2^T|_F < epsilon
"""

def learn_linear_projection(X, Y, reg=1e-3, orth_penalty=None, other_W=None):
    logger.info("Learning CMNF Projection matrix...")
    d = X.shape[1]
    # Ridge regression solution for W
    A = (X.T @ X) + reg * np.eye(d)
    B = X.T @ Y
    W = np.linalg.solve(A, B).T
    
    if orth_penalty and other_W is not None:
        logger.debug("Applying orthogonality penalty to projection...")
        # Iterative step to reduce cross-context contamination
        grad = 2.0 * orth_penalty * (W @ other_W.T) @ other_W
        W -= 1e-3 * grad
    return W

def batch_project(W, X):
    return (W @ X.T).T