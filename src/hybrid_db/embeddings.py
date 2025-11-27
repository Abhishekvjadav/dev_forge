"""
Lightweight embedding pipeline that runs fully offline.
Uses hashing trick to create deterministic dense vectors.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

import numpy as np

from .config import settings


class HashEmbeddingModel:
    """
    Deterministic pseudo-embedding generator suitable for offline demos.
    We hash n-grams into a fixed-size vector and normalize it.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def _tokenize(self, text: str) -> Iterable[str]:
        lowered = "".join(ch if ch.isalnum() else " " for ch in text.lower())
        tokens = [tok for tok in lowered.split() if tok]
        for token in tokens:
            yield token
            if len(token) > 3:
                for i in range(len(token) - 2):
                    yield token[i : i + 3]

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "little") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


embedding_model = HashEmbeddingModel()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


