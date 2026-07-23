"""OpenAI client shim.

Wraps ``openai.OpenAI().embeddings.create`` so query vectors are
mapped through the fitted Isotrieve transform.  Batching is preserved;
usage/token fields are passed through untouched.

Usage::

    import openai
    from isotrieve.wrappers.openai_shim import IsotrieveOpenAI

    client = openai.OpenAI()
    shim = IsotrieveOpenAI(client, "mapping.isotrieve")
    response = shim.embeddings.create(
        input=["query text"], model="text-embedding-3-small"
    )
    # response.data[0].embedding is now in legacy-model space
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.registry import load_mapping


class IsotrieveOpenAI:
    """OpenAI embeddings client wrapper.

    Wraps ``client.embeddings.create`` with the same call signature.
    Output vectors are mapped through the Isotrieve transform before return.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` (or ``openai.AsyncOpenAI``) instance.
    transform_artifact_path:
        Path to a fitted ``.isotrieve`` mapping file.
    direction:
        ``"new_to_old"`` (default) maps new-model vectors to legacy space.
    """

    def __init__(
        self,
        client: Any,
        transform_artifact_path: str,
        *,
        direction: str = "new_to_old",
    ) -> None:
        self._client = client
        self._direction = direction
        self._mapping = load_mapping(transform_artifact_path)
        self._d_src = self._mapping.d_src
        self._d_tgt = self._mapping.d_tgt
        self._embeddings = _IsotrieveEmbeddingsNamespace(self)

    @property
    def embeddings(self) -> _IsotrieveEmbeddingsNamespace:
        return self._embeddings

    def _map_vectors(self, vecs: np.ndarray) -> np.ndarray:
        """Map vectors through the Isotrieve transform."""
        if self._direction == "new_to_old":
            return l2_normalize(self._mapping.inverse_transform(vecs))
        return l2_normalize(self._mapping.transform(vecs))

    @property
    def has_recalibrator(self) -> bool:
        return self._mapping.has_recalibrator

    @property
    def d_src(self) -> int:
        return self._d_src

    @property
    def d_tgt(self) -> int:
        return self._d_tgt

    def __repr__(self) -> str:
        return (
            f"IsotrieveOpenAI(mapping_type={self._mapping.mapping_type!r}, "
            f"d_src={self._d_src}, d_tgt={self._d_tgt})"
        )


class _IsotrieveEmbeddingsNamespace:
    """Namespace matching ``client.embeddings.create(...)``."""

    def __init__(self, shim: IsotrieveOpenAI) -> None:
        self._shim = shim

    def create(
        self,
        *,
        input: Sequence[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create embeddings and map through Isotrieve transform.

        Delegates to the underlying client, then maps each embedding
        vector.  Usage/token fields are passed through untouched.
        """
        response = self._shim._client.embeddings.create(
            input=list(input), model=model, **kwargs
        )

        # Map each embedding
        for item in response.data:
            vec = np.array(item.embedding, dtype=np.float64).reshape(1, -1)
            mapped = self._shim._map_vectors(vec)
            item.embedding = mapped.ravel().tolist()

        return response
