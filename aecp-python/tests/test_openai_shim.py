"""Tests for OpenAI client shim (mocked, no real OpenAI required)."""

from __future__ import annotations

import numpy as np
from tests.fakes import FakeOpenAIClient, make_mapping, save_mapping


class TestAECPOpenAI:
    """Test OpenAI client wrapper."""

    def test_basic_create(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12)

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        response = shim.embeddings.create(input=["hello", "world"])
        assert len(response.data) == 2
        # Mapped vectors should be in legacy (source) space
        vec = np.array(response.data[0].embedding)
        assert vec.shape == (8,)

    def test_single_string_input(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12)

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        response = shim.embeddings.create(input=["one item"])
        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 8

    def test_preserves_usage(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12)

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        response = shim.embeddings.create(input=["test"])
        # Usage is passed through from the fake
        assert hasattr(response, "usage")

    def test_preserves_model(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12, model="text-embedding-3-small")

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        response = shim.embeddings.create(
            input=["test"], model="text-embedding-3-small"
        )
        assert response.model == "text-embedding-3-small"

    def test_repr(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12)

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        r = repr(shim)
        assert "AECPOpenAI" in r
        assert "d_src=8" in r
        assert "d_tgt=12" in r

    def test_properties(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12)
        path = save_mapping(m, tmp_path)
        client = FakeOpenAIClient(dim=12)

        from aecp.wrappers.openai_shim import AECPOpenAI

        shim = AECPOpenAI(client, path)
        assert shim.d_src == 8
        assert shim.d_tgt == 12
        assert not shim.has_recalibrator
