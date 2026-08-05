"""Image calibration pipeline for cross-modal embedding migration."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from isotrieve.providers.base import Embedder


def load_images_from_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"),
) -> list[Path]:
    """Load image file paths from a directory (sorted, deterministic).

    Parameters
    ----------
    directory:
        Path to directory containing images.
    extensions:
        File extensions to include (case-insensitive).

    Returns
    -------
    Sorted list of image file paths.
    """
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"Not a directory: {d}")

    ext_lower = {e.lower() for e in extensions}
    images = [
        p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in ext_lower
    ]
    if not images:
        raise ValueError(f"No images found in {d} with extensions {extensions}")
    return images


def load_paired_images(
    source_dir: str | Path,
    target_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"),
) -> tuple[list[Path], list[Path]]:
    """Load paired images from two directories (same filenames = matched pairs).

    Both directories must contain images with matching filenames. Images are
    paired by name: ``source_dir/cat.jpg`` pairs with ``target_dir/cat.jpg``.

    Returns
    -------
    (source_paths, target_paths) — matched lists of equal length.

    Raises
    ------
    ValueError if no matching filenames are found.
    """
    src_images = load_images_from_directory(source_dir, extensions)
    tgt_images = load_images_from_directory(target_dir, extensions)

    src_names = {p.name: p for p in src_images}
    tgt_names = {p.name: p for p in tgt_images}

    common = sorted(set(src_names.keys()) & set(tgt_names.keys()))
    if not common:
        raise ValueError(
            f"No matching filenames between {source_dir} and {target_dir}. "
            f"Source has {len(src_images)} images, target has {len(tgt_images)}. "
            f"Paired calibration requires matching filenames."
        )

    paired_src = [src_names[n] for n in common]
    paired_tgt = [tgt_names[n] for n in common]
    return paired_src, paired_tgt


def calibration_checksum(images: Sequence[Path]) -> str:
    """SHA256 checksum of image filenames for provenance logging."""
    h = hashlib.sha256()
    for p in images:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def embed_image_pairs(
    source_embedder: Embedder,
    target_embedder: Embedder,
    source_images: Sequence[Path],
    target_images: Sequence[Path],
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed paired images from two different models.

    Parameters
    ----------
    source_embedder:
        Provider with ``embed_images()`` method (e.g., CLIPEmbedder).
    target_embedder:
        Provider with ``embed_images()`` method.
    source_images:
        Paths to source-space images.
    target_images:
        Paths to target-space images (same order = same content).
    batch_size:
        Embedding batch size.

    Returns
    -------
    (X_src, X_tgt) — arrays of shape ``(K, d_src)`` and ``(K, d_tgt)``.
    """
    if len(source_images) != len(target_images):
        raise ValueError(
            f"Image count mismatch: {len(source_images)} source vs "
            f"{len(target_images)} target"
        )

    src_embed_fn = getattr(source_embedder, "embed_images", None)
    tgt_embed_fn = getattr(target_embedder, "embed_images", None)

    if src_embed_fn is None:
        raise TypeError(
            f"{type(source_embedder).__name__} does not support embed_images(). "
            f"Use a CLIP-based provider for image embedding."
        )
    if tgt_embed_fn is None:
        raise TypeError(
            f"{type(target_embedder).__name__} does not support embed_images(). "
            f"Use a CLIP-based provider for image embedding."
        )

    X_src = src_embed_fn(list(source_images))
    X_tgt = tgt_embed_fn(list(target_images))

    return np.asarray(X_src, dtype=np.float64), np.asarray(X_tgt, dtype=np.float64)


def embed_corpus(
    embedder: Embedder,
    images: Sequence[Path],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed a corpus of images for migration.

    Parameters
    ----------
    embedder:
        Provider with ``embed_images()`` method.
    images:
        Paths to images.
    batch_size:
        Embedding batch size.

    Returns
    -------
    Array of shape ``(N, d)`` with embeddings.
    """
    embed_fn = getattr(embedder, "embed_images", None)
    if embed_fn is None:
        raise TypeError(
            f"{type(embedder).__name__} does not support embed_images(). "
            f"Use a CLIP-based provider for image embedding."
        )
    return np.asarray(embed_fn(list(images)), dtype=np.float64)
