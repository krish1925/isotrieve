"""End-to-end consumer test for image embedding migration.

Consumer scenario:
  1. You have a database of images
  2. You encoded them with Model A (e.g., CLIP-ViT-B-32)
  3. You want to migrate to Model B (e.g., SigLIP) without re-encoding
  4. Use isotrieve to fit a mapping, transform your embeddings
  5. Verify retrieval quality is maintained
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isotrieve.mapping.base import l2_normalize
from isotrieve.mapping.contrastive import ContrastiveMapping

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SKIP_REASON = "sentence-transformers not installed or model download fails"


def _make_test_images(n: int = 20, size: int = 32) -> list[Image.Image]:
    """Create n synthetic test images with distinct visual content."""
    rng = np.random.default_rng(42)
    images = []
    for i in range(n):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        gradient = np.linspace(0, 255, size * size, dtype=np.uint8).reshape(size, size)
        arr[:, :, 0] = (arr[:, :, 0].astype(np.int16) + gradient.astype(np.int16)) % 256
        arr[:, :, 1] = (255 - gradient).astype(np.uint8)
        arr[:, :, 2] = (i * 12) % 256
        images.append(Image.fromarray(arr, "RGB"))
    return images


def _save_test_images(images: list[Image.Image], directory: Path) -> list[Path]:
    """Save PIL images to a directory (creating it if needed)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = directory / f"img_{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Unit tests for ContrastiveMapping
# ---------------------------------------------------------------------------


class TestContrastiveMapping:
    """Test ContrastiveMapping with synthetic embeddings (no model download)."""

    def test_basic_fit_transform(self):
        rng = np.random.default_rng(0)
        d = 64
        k = 100

        X = rng.normal(size=(k, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q + 0.01 * rng.normal(size=(k, d))

        m = ContrastiveMapping(temperature=0.1, seed=0)
        m.fit(X, Y)

        assert m.is_fitted
        assert m.d_src == d
        assert m.d_tgt == d

        Z = m.transform(X[:5])
        assert Z.shape == (5, d)
        assert np.all(np.isfinite(Z))

        Z_all = m.transform(X)
        cos_sim = np.sum(l2_normalize(Z_all) * l2_normalize(Y), axis=1)
        assert cos_sim.mean() > 0.5, f"Expected mean cosine > 0.5, got {cos_sim.mean()}"

    def test_inverse_transform(self):
        rng = np.random.default_rng(1)
        d = 32
        k = 80

        X = rng.normal(size=(k, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q

        m = ContrastiveMapping(temperature=0.1, seed=0)
        m.fit(X, Y)

        Z = m.transform(X)
        X_back = m.inverse_transform(Z)
        assert X_back.shape == X.shape

    def test_save_load_roundtrip(self, tmp_path: Path):
        rng = np.random.default_rng(2)
        d = 16
        k = 50

        X = rng.normal(size=(k, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q + 0.02 * rng.normal(size=(k, d))

        m = ContrastiveMapping(temperature=0.1, seed=0)
        m.fit(X, Y)

        path = tmp_path / "contrastive.isotrieve"
        m.save(path)
        assert path.exists()

        loaded = ContrastiveMapping.load(path)
        assert loaded.is_fitted
        assert loaded.d_src == d
        assert loaded.d_tgt == d

        Z_orig = m.transform(X[:5])
        Z_loaded = loaded.transform(X[:5])
        np.testing.assert_allclose(Z_orig, Z_loaded, atol=1e-10)

    def test_different_dims(self):
        rng = np.random.default_rng(3)
        d_src, d_tgt = 512, 768
        k = 200

        X = rng.normal(size=(k, d_src))
        Y = rng.normal(size=(k, d_tgt))

        m = ContrastiveMapping(temperature=0.07, seed=0)
        m.fit(X, Y)

        assert m.d_src == d_src
        assert m.d_tgt == d_tgt
        Z = m.transform(X[:3])
        assert Z.shape == (3, d_tgt)

    def test_validation_report(self):
        rng = np.random.default_rng(4)
        d = 32
        k = 100

        # Use a linear relationship so holdout cosine is a meaningful, positive metric
        X = rng.normal(size=(k, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q + 0.05 * rng.normal(size=(k, d))

        m = ContrastiveMapping(temperature=0.1, seed=0)
        m.fit(X, Y)

        report = m.validation_report()
        assert report.n_train > 0
        assert report.n_holdout > 0
        assert report.n_train + report.n_holdout == k
        assert 0 <= report.holdout_cosine_mean <= 1.0

    def test_rank_compression(self):
        rng = np.random.default_rng(5)
        d = 64
        k = 100

        X = rng.normal(size=(k, d))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y = X @ Q

        m_full = ContrastiveMapping(temperature=0.1, seed=0)
        m_full.fit(X, Y)

        m_compressed = ContrastiveMapping(temperature=0.1, seed=0, rank=16)
        m_compressed.fit(X, Y)

        assert m_full.transform(X[:3]).shape == (3, d)
        assert m_compressed.transform(X[:3]).shape == (3, d)


# ---------------------------------------------------------------------------
# Integration test with real CLIP models
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestImageMigrationEndToEnd:
    """Full consumer test with real CLIP models."""

    @pytest.fixture
    def clip_embedder_b32(self):
        try:
            from isotrieve.providers.image_providers import CLIPEmbedder

            return CLIPEmbedder("sentence-transformers/clip-ViT-B-32")
        except Exception:
            pytest.skip(SKIP_REASON)

    @pytest.fixture
    def clip_embedder_b16(self):
        try:
            from isotrieve.providers.image_providers import CLIPEmbedder

            return CLIPEmbedder("sentence-transformers/clip-ViT-B-16")
        except Exception:
            pytest.skip(SKIP_REASON)

    def test_cross_model_image_migration(
        self,
        clip_embedder_b32,
        clip_embedder_b16,
        tmp_path: Path,
    ):
        images = _make_test_images(n=40, size=64)
        image_paths = _save_test_images(images, tmp_path / "images")

        X_src = clip_embedder_b32.embed_images([str(p) for p in image_paths])
        X_tgt = clip_embedder_b16.embed_images([str(p) for p in image_paths])

        m = ContrastiveMapping(temperature=0.07, seed=0)
        m.fit(X_src, X_tgt)

        Z = m.transform(X_src)
        assert Z.shape == X_tgt.shape

        Z_norm = l2_normalize(Z)
        Y_norm = l2_normalize(X_tgt)
        sims = Z_norm @ Y_norm.T
        top1_correct = sum(sims[i, i] == sims[i].max() for i in range(len(sims)))
        retention = top1_correct / len(sims)

        assert retention > 0.5, (
            f"Migration retention too low: {retention:.1%}. "
            f"Expected >50% for cross-model image migration."
        )

    def test_text_to_image_cross_modal(
        self,
        clip_embedder_b32,
        tmp_path: Path,
    ):
        images = _make_test_images(n=30, size=64)
        image_paths = _save_test_images(images, tmp_path / "images")

        X_img = clip_embedder_b32.embed_images([str(p) for p in image_paths])
        captions = [f"synthetic image {i}" for i in range(len(images))]
        X_txt = clip_embedder_b32.embed(captions)

        m = ContrastiveMapping(temperature=0.07, seed=0)
        m.fit(X_txt, X_img)

        Z = m.transform(X_txt)
        assert Z.shape == X_img.shape

        Z_norm = l2_normalize(Z)
        I_norm = l2_normalize(X_img)
        sims = Z_norm @ I_norm.T
        top1_correct = sum(sims[i, i] == sims[i].max() for i in range(len(sims)))
        retention = top1_correct / len(sims)

        assert retention > 0.3, (
            f"Cross-modal retention too low: {retention:.1%}. "
            f"Expected >30% for text-to-image mapping."
        )


# ---------------------------------------------------------------------------
# Test image calibration pipeline
# ---------------------------------------------------------------------------


class TestImageCalibration:
    """Test image loading and calibration utilities."""

    def test_load_images_from_directory(self, tmp_path: Path):
        from isotrieve.calibration.image_calibration import load_images_from_directory

        images = _make_test_images(5)
        _save_test_images(images, tmp_path)

        loaded = load_images_from_directory(tmp_path)
        assert len(loaded) == 5
        assert all(p.suffix == ".png" for p in loaded)

    def test_paired_images(self, tmp_path: Path):
        from isotrieve.calibration.image_calibration import load_paired_images

        images = _make_test_images(5)
        _save_test_images(images, tmp_path / "src")
        _save_test_images(images, tmp_path / "tgt")

        src, tgt = load_paired_images(tmp_path / "src", tmp_path / "tgt")
        assert len(src) == 5
        assert len(tgt) == 5
        assert [p.name for p in src] == [p.name for p in tgt]

    def test_paired_images_no_match(self, tmp_path: Path):
        from isotrieve.calibration.image_calibration import load_paired_images

        images_b = _make_test_images(3)
        _save_test_images(images_b, tmp_path / "src")
        for i, img in enumerate(images_b):
            (tmp_path / "tgt" / f"photo_{i}.png").parent.mkdir(exist_ok=True)
            img.save(tmp_path / "tgt" / f"photo_{i}.png")

        with pytest.raises(ValueError, match="No matching filenames"):
            load_paired_images(tmp_path / "src", tmp_path / "tgt")

    @pytest.mark.slow
    def test_embed_image_pairs(self, tmp_path: Path):
        from isotrieve.calibration.image_calibration import embed_image_pairs

        images = _make_test_images(5, size=32)
        src_paths = _save_test_images(images, tmp_path / "src")
        tgt_paths = _save_test_images(images, tmp_path / "tgt")

        try:
            from isotrieve.providers.image_providers import CLIPEmbedder

            embedder = CLIPEmbedder()
        except Exception:
            pytest.skip(SKIP_REASON)

        X_src, X_tgt = embed_image_pairs(embedder, embedder, src_paths, tgt_paths)
        assert X_src.shape == (5, embedder.dims)
        assert X_tgt.shape == (5, embedder.dims)
        np.testing.assert_allclose(X_src, X_tgt, atol=1e-6)
