"""Disk-cached embedder wrapper so calibration never double-pays."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from isotrieve.providers.base import Embedder

_DEFAULT_CACHE_ENV = "ISOTRIEVE_EMBED_CACHE"


def default_cache_dir() -> Path:
    """Resolve disk cache directory (env ``ISOTRIEVE_EMBED_CACHE`` or ``~/.cache/isotrieve/embeds``)."""
    override = os.environ.get(_DEFAULT_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "isotrieve" / "embeds"


def with_disk_cache(
    embedder: Embedder, cache_dir: str | Path | None = None
) -> CachedEmbedder:
    """Wrap any embedder in :class:`CachedEmbedder` (idempotent if already wrapped)."""
    if isinstance(embedder, CachedEmbedder):
        return embedder
    return CachedEmbedder(embedder, cache_dir or default_cache_dir())


class CachedEmbedder(Embedder):
    """Wrap any embedder with a content-addressed disk cache.

    Cache key = sha256(model_id + '\\0' + text). Does not hit the network
    itself; the underlying embedder might.

    Stats ``hits`` / ``misses`` / ``api_calls`` track budget; ``api_calls``
    equals the number of texts forwarded to the inner embedder.
    """

    def __init__(self, inner: Embedder, cache_dir: str | Path) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir) / _safe_model_dir(inner.model_id)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "index.json"
        self._index: dict[str, str] = {}
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
        self.hits = 0
        self.misses = 0

    @property
    def api_calls(self) -> int:
        """Texts that required an inner embed (cache misses)."""
        return self.misses

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dims(self) -> int:
        return self._inner.dims

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed with cache; only uncached texts call the inner embedder."""
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.dims), dtype=np.float64)

        out: list[np.ndarray | None] = [None] * len(texts_list)
        missing_idx: list[int] = []
        missing_texts: list[str] = []

        for i, t in enumerate(texts_list):
            key = self._key(t)
            fname = self._index.get(key)
            if fname is not None:
                path = self._cache_dir / fname
                if path.exists():
                    out[i] = np.load(path)
                    self.hits += 1
                    continue
            missing_idx.append(i)
            missing_texts.append(t)
            self.misses += 1

        if missing_texts:
            fresh = self._inner.embed(missing_texts)
            for j, i in enumerate(missing_idx):
                key = self._key(missing_texts[j])
                fname = f"{key}.npy"
                np.save(self._cache_dir / fname, fresh[j])
                self._index[key] = fname
                out[i] = fresh[j]
            self._index_path.write_text(json.dumps(self._index), encoding="utf-8")

        return np.stack([np.asarray(v, dtype=np.float64) for v in out], axis=0)

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "api_calls": self.api_calls,
            "index_size": len(self._index),
        }


def _safe_model_dir(model_id: str) -> str:
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
