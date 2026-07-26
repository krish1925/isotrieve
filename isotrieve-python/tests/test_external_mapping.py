"""Tests for ExternalMapping (BYOT gate)."""

import numpy as np
import pytest

from isotrieve.mapping.external import ExternalMapping, load_external_callable


class TestExternalMapping:
    def test_transform_basic(self):
        fn = lambda v: v @ np.eye(128, 128)
        m = ExternalMapping(fn, d_src=128, d_tgt=128)
        X = np.random.randn(10, 128)
        result = m.transform(X)
        assert result.shape == (10, 128)
        assert np.isfinite(result).all()

    def test_transform_infer_dims(self):
        fn = lambda v: v @ np.random.randn(64, 32)
        m = ExternalMapping(fn)
        X = np.random.randn(5, 64)
        result = m.transform(X)
        assert result.shape == (5, 32)
        assert m.d_src == 64
        assert m.d_tgt == 32

    def test_transform_l2_normalized(self):
        fn = lambda v: v * 100.0  # huge magnitudes
        m = ExternalMapping(fn, d_src=16, d_tgt=16)
        X = np.random.randn(5, 16)
        result = m.transform(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_transform_1d_input(self):
        fn = lambda v: v
        m = ExternalMapping(fn, d_src=8, d_tgt=8)
        x = np.random.randn(8)
        result = m.transform(x)
        assert result.shape == (1, 8)

    def test_transform_dim_mismatch_raises(self):
        fn = lambda v: v[:, :5]
        m = ExternalMapping(fn, d_src=10, d_tgt=5)
        X = np.random.randn(3, 8)  # wrong dim
        with pytest.raises(ValueError, match="Expected 10-D"):
            m.transform(X)

    def test_transform_output_dim_mismatch_raises(self):
        fn = lambda v: np.random.randn(v.shape[0], 99)
        m = ExternalMapping(fn, d_src=10, d_tgt=10)
        X = np.random.randn(3, 10)
        with pytest.raises(ValueError, match="returned 99-D"):
            m.transform(X)

    def test_is_fitted_from_construction(self):
        fn = lambda v: v
        m = ExternalMapping(fn)
        assert m.is_fitted

    def test_fit_sets_dims(self):
        fn = lambda v: v
        m = ExternalMapping(fn)
        X = np.random.randn(10, 128)
        Y = np.random.randn(10, 256)
        m.fit(X, Y)
        assert m.d_src == 128
        assert m.d_tgt == 256

    def test_inverse_transform_with_fn(self):
        forward = lambda v: v[:, :5]
        inverse = lambda v: np.pad(v, ((0, 0), (0, 5)))
        m = ExternalMapping(fn=forward, d_src=10, d_tgt=5, inverse_fn=inverse)
        X = np.random.randn(3, 10)
        fwd = m.transform(X)
        inv = m.inverse_transform(fwd)
        assert inv.shape == (3, 10)

    def test_inverse_transform_without_fn_raises(self):
        fn = lambda v: v[:, :5]
        m = ExternalMapping(fn, d_src=10, d_tgt=5)
        with pytest.raises(NotImplementedError, match="No inverse_fn"):
            m.inverse_transform(np.random.randn(3, 5))

    def test_save_raises(self):
        fn = lambda v: v
        m = ExternalMapping(fn, d_src=8, d_tgt=8)
        with pytest.raises(NotImplementedError, match="cannot be saved"):
            m.save("/tmp/test.isotrieve")

    def test_mapping_type(self):
        fn = lambda v: v
        m = ExternalMapping(fn)
        assert m.mapping_type == "external"

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 16))
        fn = lambda v: v[:, :8]
        m = ExternalMapping(fn, d_src=16, d_tgt=8)
        r1 = m.transform(X)
        r2 = m.transform(X)
        np.testing.assert_array_equal(r1, r2)

    def test_batch_transform(self):
        fn = lambda v: v[:, :8]
        m = ExternalMapping(fn, d_src=16, d_tgt=8)
        batches = [np.random.randn(3, 16), np.random.randn(2, 16)]
        results = list(m.transform_batches(batches))
        assert len(results) == 2
        assert results[0].shape == (3, 8)
        assert results[1].shape == (2, 8)


class TestLoadExternalCallable:
    def test_load_builtin(self):
        fn, name = load_external_callable("numpy:array")
        assert callable(fn)
        assert name == "numpy:array"

    def test_load_nonexistent_module(self):
        with pytest.raises(ImportError):
            load_external_callable("nonexistent_module_xyz:func")

    def test_load_non_callable(self):
        # numpy.pi is a float, not callable
        with pytest.raises(TypeError, match="not callable"):
            load_external_callable("numpy:pi")

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="expected 'module:callable'"):
            load_external_callable("no_colon_here")
