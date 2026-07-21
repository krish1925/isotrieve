"""Provider factory — always returns a disk-cached embedder."""

from __future__ import annotations

from pathlib import Path

from aecp.providers.base import Embedder
from aecp.providers.cached import CachedEmbedder, with_disk_cache


def create_embedder(
    model_id: str,
    *,
    cache_dir: str | Path | None = None,
    cached: bool = True,
) -> Embedder:
    """Create an embedder for ``model_id``, wrapped in disk cache by default.

    Resolution order:
      - ``openai:…`` / known OpenAI model names → OpenAIEmbedder
      - ``voyage:…`` / names starting with ``voyage`` → VoyageEmbedder
      - ``cohere:…`` → CohereEmbedder
      - ``gemini:…`` / ``models/text-embedding`` → GeminiEmbedder
      - otherwise → SentenceTransformerEmbedder (local)

    Parameters
    ----------
    cached:
        If True (default), wrap with :class:`CachedEmbedder` so the same text
        is never embedded twice. Set False only for tests that assert raw calls.
    """
    raw = _create_raw(model_id)
    if cached:
        return with_disk_cache(raw, cache_dir)
    return raw


def _create_raw(model_id: str) -> Embedder:
    mid = model_id
    lower = model_id.lower()

    if mid.startswith("openai:"):
        from aecp.providers.openai import OpenAIEmbedder

        return OpenAIEmbedder(mid.split(":", 1)[1])
    if mid.startswith("voyage:"):
        from aecp.providers.voyage import VoyageEmbedder

        return VoyageEmbedder(mid.split(":", 1)[1])
    if mid.startswith("cohere:"):
        from aecp.providers.cohere import CohereEmbedder

        return CohereEmbedder(mid.split(":", 1)[1])
    if mid.startswith("gemini:"):
        from aecp.providers.gemini import GeminiEmbedder

        return GeminiEmbedder(mid.split(":", 1)[1])

    if mid in {
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    } or lower.startswith("text-embedding-"):
        from aecp.providers.openai import OpenAIEmbedder

        return OpenAIEmbedder(mid)

    if lower.startswith("voyage"):
        from aecp.providers.voyage import VoyageEmbedder

        return VoyageEmbedder(mid)

    if "cohere" in lower or lower.startswith("embed-"):
        from aecp.providers.cohere import CohereEmbedder

        return CohereEmbedder(mid)

    if (
        lower.startswith("models/")
        or "gemini" in lower
        or "text-embedding-004" in lower
    ):
        from aecp.providers.gemini import GeminiEmbedder

        return GeminiEmbedder(mid)

    from aecp.providers.sentence_transformers import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(mid)


__all__ = ["create_embedder", "CachedEmbedder", "with_disk_cache"]
