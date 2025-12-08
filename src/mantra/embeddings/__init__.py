# src/mantra/embeddings/__init__.py
from __future__ import annotations

from .config import EmbeddingConfig
from .hvg_embed import compute_embeddings

__all__ = [
    "EmbeddingConfig",
    "compute_embeddings",
]
