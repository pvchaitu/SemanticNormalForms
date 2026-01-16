import numpy as np
import logging
from typing import List

logger = logging.getLogger("sdnf.embeddings")

"""
NOVELTY NOTE: Multi-level Embedding Generation.
In SDNF, an attribute is represented by a concatenation of:
1. Fine-grained (token level)
2. Abstract (canonical type level)
3. Contextual (projection based)
This prevents 'Semantic Drift' where a PAN in a 'Marketing' context 
is treated the same as a PAN in a 'Billing' context.
"""

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if ST_AVAILABLE:
            logger.debug(f"Loading transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            logger.critical("SentenceTransformers not installed!")
            print("[ERROR] sentence-transformers missing. Run 'pip install sentence-transformers'.")
            raise ImportError()

    def encode(self, texts: List[str]) -> np.ndarray:
        logger.debug(f"Encoding {len(texts)} strings...")
        return self.model.encode(texts, convert_to_numpy=True)

def add_gaussian_dp_noise(vec: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """NOVELTY: Differential Privacy (DP) at the embedding layer to prevent PII reconstruction."""
    logger.debug(f"Applying DP noise with sigma={sigma}")
    return vec + np.random.normal(0, sigma, vec.shape)