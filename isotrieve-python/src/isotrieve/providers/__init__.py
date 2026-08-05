"""Embedding providers."""

from isotrieve.providers.base import Embedder
from isotrieve.providers.cached import (
    CachedEmbedder,
    default_cache_dir,
    with_disk_cache,
)
from isotrieve.providers.factory import create_embedder
from isotrieve.providers.image_providers import CLIPEmbedder, SigLIPEmbedder

__all__ = [
    "Embedder",
    "CachedEmbedder",
    "CLIPEmbedder",
    "SigLIPEmbedder",
    "create_embedder",
    "with_disk_cache",
    "default_cache_dir",
]
