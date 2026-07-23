"""Tests for wrapper telemetry (local-only JSONL)."""

from __future__ import annotations

import pytest

from isotrieve.wrappers.telemetry import log_query, read_telemetry, summary


@pytest.fixture(autouse=True)
def _clean_telemetry(tmp_path, monkeypatch):
    """Ensure telemetry is off and path points to tmp."""
    monkeypatch.setenv("ISOTRIEVE_TELEMETRY", "off")
    monkeypatch.setattr(
        "isotrieve.wrappers.telemetry._telemetry_path",
        lambda: tmp_path / "telemetry.jsonl",
    )
    yield


class TestTelemetry:
    def test_off_by_default(self, tmp_path):
        log_query("test_wrapper")
        assert not (tmp_path / "telemetry.jsonl").exists()

    def test_local_writes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ISOTRIEVE_TELEMETRY", "local")
        log_query("llamaindex", mapping_type="ridge")
        assert (tmp_path / "telemetry.jsonl").exists()
        events = read_telemetry()
        assert len(events) == 1
        assert events[0]["event"] == "query"
        assert events[0]["wrapper"] == "llamaindex"
        assert events[0]["mapping_type"] == "ridge"

    def test_multiple_queries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ISOTRIEVE_TELEMETRY", "local")
        log_query("openai_shim")
        log_query("langchain")
        log_query("llamaindex")
        events = read_telemetry()
        assert len(events) == 3

    def test_summary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ISOTRIEVE_TELEMETRY", "local")
        log_query("llamaindex")
        log_query("llamaindex")
        log_query("openai_shim")
        s = summary()
        assert s["total_queries"] == 3
        assert s["per_wrapper"]["llamaindex"] == 2
        assert s["per_wrapper"]["openai_shim"] == 1

    def test_empty_summary(self):
        s = summary()
        assert s["total_queries"] == 0
        assert s["per_wrapper"] == {}
