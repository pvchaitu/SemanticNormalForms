# embedding_utils.py
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("sdnf.embeddings")

# Try to import sentence-transformers but allow graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

# Deterministic RNG for reproducibility if no transformer is available
GLOBAL_RNG = np.random.default_rng(42)


class EmbeddingModel:
    """
    Encapsulates embedding generation used in SDNF experiments.

    - If sentence-transformers is available, uses the specified model (recommended).
    - Otherwise falls back to a deterministic, reproducible random projection
      embedding (useful for unit tests and CI) that is stable across runs
      given the same seed.

    NOVELTY NOTE: We preserve the multi-level embedding idea by allowing callers
    to later compose fine/abstract/contextual subvectors. Here we provide the
    base textual encoder; multi-level assembly is performed by the pipeline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        if ST_AVAILABLE:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            try:
                self.model = SentenceTransformer(model_name)
                # If model returns different dim than requested, adapt
                emb_dim = self.model.get_sentence_embedding_dimension()
                if emb_dim != dim:
                    logger.warning(f"Model embedding dim {emb_dim} != requested {dim}. Using {emb_dim}.")
                    self.dim = emb_dim
            except Exception as e:
                logger.exception("Failed to load sentence-transformers model. Falling back to RNG embeddings.")
                self.model = None
                self._seed = 42
        else:
            logger.warning("sentence-transformers not available; using deterministic random embeddings.")
            self.model = None
            self._seed = 42

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Return embeddings shaped (len(texts), dim).

        - Normalizes (L2) the output so cosine similarity is valid.
        - Deterministic fallback if real model unavailable.
        """
        if self.model is not None:
            emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
        else:
            # deterministic hash-based pseudo-embedding fallback
            emb = np.stack([self._pseudo_embed(t) for t in texts], axis=0)

        # L2 normalize per-row (important for cosine sim)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        return emb

    def _pseudo_embed(self, text: str) -> np.ndarray:
        """
        Deterministic pseudo-embedding based on hashed seed.
        Useful for unit tests / CI when transformer is not available.
        """
        # Create repeatable seed per text
        h = abs(hash(text)) % (2 ** 31 - 1)
        rng = np.random.default_rng(h)
        vec = rng.normal(0, 1.0, size=(self.dim,))
        return vec.astype(np.float32)


def add_gaussian_dp_noise(vec: np.ndarray, epsilon: Optional[float] = 1.0, delta: Optional[float] = 1e-5) -> np.ndarray:
    """
    Add Gaussian DP noise to an embedding vector.

    We use the standard approximate Gaussian mechanism calibration:
        sigma = c * sensitivity / epsilon,
    where c depends on delta. For simplicity and reproducibility we use:
        sigma = sqrt(2 * log(1.25/delta)) / epsilon

    Notes:
    - Embeddings should be pre-normalized (L2) before adding noise if desired.
    - Caller should re-normalize after adding noise if cosine similarity is used.
    - This is a pragmatic calibration; for production, an audited DP library is recommended.

    Returns the noisy vector (same shape).
    """
    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be positive for DP noise")
    c = np.sqrt(2 * np.log(1.25 / (delta + 1e-30)))
    sigma = (c / epsilon)  # sensitivity assumed 1 for normalized embeddings
    logger.debug(f"Applying Gaussian DP noise: epsilon={epsilon}, delta={delta}, sigma={sigma:.6f}")
    noise = np.random.default_rng(42).normal(0, sigma, size=vec.shape)
    noisy = vec + noise
    # Re-normalize so downstream cosine sims behave sensibly
    noisy = noisy / (np.linalg.norm(noisy) + 1e-12)
    return noisy.astype(np.float32)
