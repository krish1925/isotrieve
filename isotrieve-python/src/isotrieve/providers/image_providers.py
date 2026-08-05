"""Image embedding providers using CLIP/SigLIP via sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from isotrieve.providers.base import Embedder

# Type for anything PIL-image-like: file path, PIL Image, or numpy array
ImageInput = str | Path | Any


def _to_pil(image: ImageInput):
    """Coerce file paths to PIL Images; pass everything else through."""
    if isinstance(image, (str, Path)):
        from PIL import Image

        return Image.open(str(image))
    return image


class CLIPEmbedder(Embedder):
    """Embed images and text using CLIP models via sentence-transformers.

    Supports both image and text embedding in a shared space. Images can be
    file paths, PIL Image objects, or numpy arrays.

    Install with ``pip install isotrieve[sentence-transformers]``.

    Parameters
    ----------
    model_id:
        HuggingFace model ID. Defaults to ``sentence-transformers/clip-ViT-B-32``.
        Other options: ``clip-ViT-B-16``, ``clip-ViT-L-14``, ``clip-ViT-L-14-v1.5``.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/clip-ViT-B-32",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for image embedding. "
                "Install with: pip install isotrieve[sentence-transformers]"
            ) from e
        self._model_id = model_id
        self._model = SentenceTransformer(model_id)
        probe = self._model.encode(["dim probe"], convert_to_numpy=True)
        self._dims = int(np.asarray(probe).shape[-1])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts into the CLIP shared space."""
        if not texts:
            return np.zeros((0, self._dims), dtype=np.float64)
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float64)

    def embed_images(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Embed images into the CLIP shared space.

        Parameters
        ----------
        images:
            Sequence of image inputs: file paths (str/Path), PIL Images,
            or numpy arrays.

        Returns
        -------
        np.ndarray of shape ``(len(images), dims)`` float64.
        """
        if not images:
            return np.zeros((0, self._dims), dtype=np.float64)
        # sentence-transformers CLIP handles PIL Images and numpy arrays natively
        arr = self._model.encode(
            [_to_pil(img) for img in images],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float64)


class SigLIPEmbedder(CLIPEmbedder):
    """Embed images and text using SigLIP models.

    SigLIP uses a sigmoid-based contrastive loss instead of softmax,
    which can produce better retrieval quality for some tasks.

    Default model: ``sentence-transformers/SigLIP-SO400M-patch14-384``.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/SigLIP-SO400M-patch14-384",
    ) -> None:
        super().__init__(model_id=model_id)
