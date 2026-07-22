"""Shared test fakes — deterministic, no network, no store SDKs.

These are *not* mocks.  They implement enough of the real interfaces
(``Embedder``, ``VectorStore``, ``BaseEmbedding``) that wrapper/adapter
tests can exercise real code paths without hitting APIs.

Usage::

    from tests.fakes import FakeEmbedder, FakeStore, make_mapping
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from aecp.mapping.base import Mapping
from aecp.mapping.linear import RidgeMapping
from aecp.stores.base import VectorRecord, VectorStore

# ---------------------------------------------------------------------------
# Mapping helper
# ---------------------------------------------------------------------------


def make_mapping(
    d_src: int = 8,
    d_tgt: int = 12,
    k: int = 120,
    seed: int = 42,
) -> RidgeMapping:
    """Create a fitted RidgeMapping with deterministic data.

    Returns a ready-to-use mapping; no network, no I/O.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, d_src))
    W = rng.normal(size=(d_src, d_tgt))
    Y = X @ W
    m = RidgeMapping(alpha="auto", seed=0)
    m.fit(X, Y)
    return m


def save_mapping(m: Mapping, tmp_path: Any, name: str = "map.aecp") -> str:
    """Save a mapping to a temp directory and return the path string."""
    from pathlib import Path

    p = Path(tmp_path) / name
    m.save(p)
    return str(p)


# ---------------------------------------------------------------------------
# Fake embedder — implements the Embedder ABC surface
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder.  Returns seeded random vectors.

    Parameters
    ----------
    dim:
        Output dimensionality.
    model_id:
        Identifier string (for ``Embedder.model_id``).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self, dim: int = 12, model_id: str = "fake-model", seed: int = 0
    ) -> None:
        self._dim = dim
        self._model_id = model_id
        self._seed = seed

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts → (len(texts), dim) seeded random vectors."""
        rng = np.random.default_rng(self._seed + hash(tuple(texts)) % (2**31))
        return rng.normal(size=(len(texts), self._dim))


# ---------------------------------------------------------------------------
# Fake LangChain Embeddings — duck-types langchain_core Embeddings
# ---------------------------------------------------------------------------


class FakeLangChainEmbeddings:
    """Fake LangChain ``Embeddings`` base class.

    Implements ``embed_documents`` and ``embed_query`` with deterministic
    seeded random output.  Does NOT subclass ``langchain_core.embeddings.Embeddings``
    so the import is never required.
    """

    def __init__(self, dim: int = 12, seed: int = 0) -> None:
        self._dim = dim
        self._seed = seed

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(self._seed)
        return rng.normal(size=(len(texts), self._dim)).tolist()

    def embed_query(self, text: str) -> list[float]:
        rng = np.random.default_rng(self._seed + 1)
        return rng.normal(size=(self._dim,)).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Fake LlamaIndex BaseEmbedding — duck-types the interface
# ---------------------------------------------------------------------------


class FakeLlamaIndexEmbedding:
    """Fake LlamaIndex ``BaseEmbedding``.

    Implements ``_get_text_embedding`` and ``_get_query_embedding`` with
    deterministic seeded random output.  Does NOT subclass
    ``llama_index.core.base.embeddings.BaseEmbedding`` so the import
    is never required.
    """

    def __init__(self, dim: int = 12, seed: int = 0) -> None:
        self._dim = dim
        self._seed = seed

    def _get_text_embedding(self, text: str) -> list[float]:
        rng = np.random.default_rng(self._seed)
        return rng.normal(size=(self._dim,)).tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(self._seed)
        return rng.normal(size=(len(texts), self._dim)).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        rng = np.random.default_rng(self._seed + 1)
        return rng.normal(size=(self._dim,)).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)


# ---------------------------------------------------------------------------
# Fake OpenAI embeddings client — duck-types client.embeddings.create
# ---------------------------------------------------------------------------


class _FakeOpenAIEmbeddingData:
    """Mimics openai.types.CreateEmbeddingResponse.data[0]."""

    def __init__(self, embedding: list[float], index: int = 0) -> None:
        self.embedding = embedding
        self.index = index
        self.object = "embedding"


class _FakeOpenAIUsage:
    """Mimics openai.types.CreateEmbeddingResponse.usage."""

    def __init__(self, prompt_tokens: int = 0, total_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class _FakeOpenAIEmbeddingResponse:
    """Mimics openai.types.CreateEmbeddingResponse."""

    def __init__(self, embeddings: list[list[float]], model: str = "fake") -> None:
        self.data = [_FakeOpenAIEmbeddingData(e, i) for i, e in enumerate(embeddings)]
        self.model = model
        self.usage = _FakeOpenAIUsage()
        self.object = "list"


class FakeOpenAIClient:
    """Fake OpenAI client with ``embeddings.create`` method.

    Matches the call signature of ``openai.OpenAI().embeddings.create``:
    ``create(input=..., model=...)`` → response with ``.data[i].embedding``.
    """

    def __init__(self, dim: int = 12, model: str = "fake-embed", seed: int = 0) -> None:
        self._dim = dim
        self._model = model
        self._seed = seed
        self._call_count = 0

    @property
    def embeddings(self) -> _FakeOpenAIEmbeddingsNamespace:
        return _FakeOpenAIEmbeddingsNamespace(self)


class _FakeOpenAIEmbeddingsNamespace:
    """Namespace for ``client.embeddings.create(...)``."""

    def __init__(self, client: FakeOpenAIClient) -> None:
        self._client = client

    def create(
        self,
        *,
        input: list[str] | str,
        model: str | None = None,
    ) -> _FakeOpenAIEmbeddingResponse:
        if isinstance(input, str):
            input = [input]
        self._client._call_count += 1
        rng = np.random.default_rng(self._client._seed + self._client._call_count)
        embeddings = [rng.normal(size=(self._client._dim,)).tolist() for _ in input]
        return _FakeOpenAIEmbeddingResponse(
            embeddings, model=model or self._client._model
        )


# ---------------------------------------------------------------------------
# Fake vector store — in-memory, implements VectorStore ABC
# ---------------------------------------------------------------------------


class FakeStore(VectorStore):
    """In-memory vector store for tests.

    Implements ``count``, ``iter_vectors``, ``write_vectors`` with
    no I/O.  Vectors live in a plain dict.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def count(self) -> int:
        return len(self._records)

    def iter_vectors(self, batch_size: int = 1024) -> Iterator[list[VectorRecord]]:
        ids = list(self._records.keys())
        for i in range(0, len(ids), batch_size):
            yield [self._records[vid] for vid in ids[i : i + batch_size]]

    def write_vectors(
        self,
        records: Iterator[list[VectorRecord]] | list[VectorRecord],
        *,
        batch_size: int = 1024,
    ) -> int:
        count = 0
        for batch in records:
            for rec in batch:
                self._records[rec.id] = rec
                count += 1
        return count

    def seed(self, n: int, dim: int, seed: int = 0) -> None:
        """Populate with ``n`` random vectors of dimension ``dim``."""
        rng = np.random.default_rng(seed)
        for i in range(n):
            vid = f"vec_{i}"
            self._records[vid] = VectorRecord(
                id=vid,
                vector=rng.normal(size=(dim,)),
                text=f"doc {i}",
                payload={"index": i},
            )

    def get_vectors(self) -> dict[str, VectorRecord]:
        """Return the internal dict (for assertions)."""
        return dict(self._records)


# ---------------------------------------------------------------------------
# Fake ChromaDB — minimal client + collection
# ---------------------------------------------------------------------------


class FakeChromaCollection:
    """Minimal fake for ``chromadb.Collection``."""

    def __init__(
        self,
        embeddings: list[list[float]] | None = None,
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str | None] | None = None,
    ) -> None:
        self._embeddings = embeddings or []
        self._ids = ids or []
        self._metadatas = metadatas or []
        self._documents = documents or []

    def get(
        self,
        offset: int = 0,
        limit: int = 100,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        end = min(offset + limit, len(self._ids))
        ids = self._ids[offset:end]
        out: dict[str, Any] = {"ids": ids}
        if include:
            if "embeddings" in include:
                out["embeddings"] = self._embeddings[offset:end]
            if "metadatas" in include:
                out["metadatas"] = self._metadatas[offset:end]
            if "documents" in include:
                out["documents"] = self._documents[offset:end]
        return out

    def count(self) -> int:
        return len(self._ids)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str | None] | None = None,
    ) -> None:
        self._ids.extend(ids)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas or [{}] * len(ids))
        self._documents.extend(documents or [None] * len(ids))


class FakeChromaClient:
    """Minimal fake for ``chromadb.Client``."""

    def __init__(
        self, collections: dict[str, FakeChromaCollection] | None = None
    ) -> None:
        self._collections = collections or {}

    def get_collection(
        self, name: str, embedding_function: Any = None
    ) -> FakeChromaCollection:
        if name not in self._collections:
            self._collections[name] = FakeChromaCollection()
        return self._collections[name]

    def create_collection(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> FakeChromaCollection:
        col = FakeChromaCollection()
        self._collections[name] = col
        return col

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)
