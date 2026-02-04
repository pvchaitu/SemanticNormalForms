# embedding_utils.py
"""
Multi-level deterministic embedding model implementing Eq 3.2 with regeneration support.

Emb(p, c) = concat(Emb_fine(p, c), Emb_abstract(p), Emb_contextual(p, c))

Supports deterministic regenerations (simulate repeated embedding calls) for EENF.
"""

from typing import List, Optional, Tuple
import numpy as np
import hashlib

class EmbeddingModel:
    def __init__(self,
                 fine_dim: int = 128,
                 abstract_dim: int = 64,
                 contextual_dim: int = 64,
                 regen_jitter: float = 1e-3):
        self.fine_dim = fine_dim
        self.abstract_dim = abstract_dim
        self.contextual_dim = contextual_dim
        self.dim = fine_dim + abstract_dim + contextual_dim
        self.regen_jitter = regen_jitter

    def _seed_from_text(self, text: str, salt: str = "") -> int:
        h = hashlib.sha256((text + salt).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") & 0xFFFFFFFF

    def _pseudo_vector(self, text: str, dim: int, salt: str = "", regen_idx: Optional[int] = None) -> np.ndarray:
        seed = self._seed_from_text(text, salt)
        if regen_idx is not None:
            seed = (seed ^ (regen_idx * 0x9e3779b1)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(dim,))
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def encode(self, tokens: List[str], context: Optional[str] = None, regen_idx: Optional[int] = None) -> np.ndarray:
        """
        Encode tokens into concatenated embeddings. If regen_idx is provided, deterministic
        regeneration variation is applied to simulate repeated embedding calls.
        """
        out = []
        ctx = context or "global"
        for t in tokens:
            fine = self._pseudo_vector(f"LEX:{t}:{ctx}", self.fine_dim, salt="fine", regen_idx=regen_idx)
            abstract = self._pseudo_vector(f"ABS:{t}", self.abstract_dim, salt="abstract", regen_idx=regen_idx)
            contextual = self._pseudo_vector(f"CTX:{ctx}:{t}", self.contextual_dim, salt="contextual", regen_idx=regen_idx)
            emb = np.concatenate([fine, abstract, contextual], axis=0)
            # small deterministic jitter for regeneration realism
            if regen_idx is not None:
                jitter = (regen_idx % 7) * self.regen_jitter
                emb = emb + jitter * np.sign(emb)
            out.append(emb)
        return np.stack(out, axis=0)

    def regenerations(self, token: str, context: Optional[str], G: int) -> np.ndarray:
        """
        Return G regenerations for token as an array (G x dim).
        """
        reps = [self.encode([token], context=context, regen_idx=i)[0] for i in range(G)]
        return np.stack(reps, axis=0)

    def split_components(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fine = emb[:self.fine_dim]
        abstract = emb[self.fine_dim:self.fine_dim + self.abstract_dim]
        contextual = emb[self.fine_dim + self.abstract_dim:]
        return fine, abstract, contextual

    def component_similarity(self, a: np.ndarray, b: np.ndarray) -> dict:
        def cos(u, v):
            nu = np.linalg.norm(u) + 1e-12
            nv = np.linalg.norm(v) + 1e-12
            return float(np.dot(u, v) / (nu * nv))
        fa, aa, ca = self.split_components(a)
        fb, ab, cb = self.split_components(b)
        return {
            "fine": cos(fa, fb),
            "abstract": cos(aa, ab),
            "contextual": cos(ca, cb),
            "global": cos(a, b)
        }
