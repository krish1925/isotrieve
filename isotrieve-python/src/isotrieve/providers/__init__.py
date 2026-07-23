"""Embedding providers."""

from isotrieve.providers.base import Embedder
from isotrieve.providers.cached import (
    CachedEmbedder,
    default_cache_dir,
    with_disk_cache,
)
from isotrieve.providers.factory import create_embedder

__all__ = [
    "Embedder",
    "CachedEmbedder",
    "create_embedder",
    "with_disk_cache",
    "default_cache_dir",
]
