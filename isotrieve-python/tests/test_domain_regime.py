"""Tests for domain-regime inference: gate report + doctor CLI (issue #37)."""

from __future__ import annotations

import json

import numpy as np
from tests.fakes import make_mapping, save_mapping
from typer.testing import CliRunner

from isotrieve.cli import app
from isotrieve.quality.domain import DEFAULT_DOMAIN, DOMAIN_KEYWORDS, infer_domain
from isotrieve.quality.gate import QualityGate

runner = CliRunner()

MEDICAL_TEXTS = [
    "The patient presented with symptoms of diabetes and the diagnosis was documented.",
    "ICD-10 code E11.9 was assigned; a physician ordered screening for cardiac risk.",
    "Post-operative infection was ruled out and the patient was discharged home.",
]
LEGAL_TEXTS = [
    "The court heard the plaintiff's appeal; counsel filed a motion for summary judgment.",
    "The judge entered an opinion and awarded damages to the defendant.",
    "The statute of limitations had not run when the complaint was filed.",
]
CODE_TEXTS = [
    "The function raised a RuntimeError and the exception propagated to the caller.",
    "Import the config module and call the API endpoint; handle HTTP errors from the client.",
    "The retry policy re-raises after the request times out and the server responds 503.",
]
GENERAL_TEXTS = [
    "The invoice was due on 2026-03-14 and the package shipped yesterday.",
    "Your booking confirmation arrived by email this morning.",
]


class TestInferDomain:
    def test_medical(self):
        assert infer_domain(MEDICAL_TEXTS) == "medical"

    def test_legal(self):
        assert infer_domain(LEGAL_TEXTS) == "legal"

    def test_code(self):
        assert infer_domain(CODE_TEXTS) == "code"

    def test_general_fallback(self):
        assert infer_domain(GENERAL_TEXTS) == DEFAULT_DOMAIN == "general"

    def test_empty_defaults_general(self):
        assert infer_domain([]) == "general"

    def test_deterministic(self):
        assert infer_domain(LEGAL_TEXTS) == infer_domain(LEGAL_TEXTS)

    def test_keywords_defined_for_all_domains(self):
        assert set(DOMAIN_KEYWORDS) == {"general", "legal", "medical", "code"}


class TestGateDomainRegime:
    def test_gate_report_defaults_to_general(self):
        m, X, Y = _good_mapping()
        gate = QualityGate()
        report = gate.evaluate(m, X[:50], Y[:50])
        assert report.domain_regime == "general"
        d = report.to_dict()
        assert d["domain_regime"] == "general"

    def test_gate_report_infers_medical_from_corpus_texts(self):
        m, X, Y = _good_mapping()
        gate = QualityGate()
        report = gate.evaluate(m, X[:50], Y[:50], corpus_texts=MEDICAL_TEXTS)
        assert report.domain_regime == "medical"

    def test_gate_to_dict_includes_domain_regime(self):
        m, X, Y = _good_mapping()
        gate = QualityGate()
        report = gate.evaluate(m, X[:50], Y[:50], corpus_texts=CODE_TEXTS)
        assert report.to_dict()["domain_regime"] == "code"


class TestGateCliDomainRegime:
    def test_gate_cli_emits_domain_regime_json(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)
        rng = np.random.default_rng(99)
        np.save(tmp_path / "X.npy", rng.normal(size=(50, 8)))
        np.save(tmp_path / "Y.npy", rng.normal(size=(50, 12)))
        texts = tmp_path / "corpus.txt"
        texts.write_text("\n".join(LEGAL_TEXTS), encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--source-vectors",
                str(tmp_path / "X.npy"),
                "--target-vectors",
                str(tmp_path / "Y.npy"),
                "--corpus-texts",
                str(texts),
                "--format",
                "json",
            ],
        )
        assert result.exit_code in (0, 1)
        data = json.loads(result.output)
        assert data["domain_regime"] == "legal"

    def test_gate_cli_missing_corpus_texts_fails(self, tmp_path):
        m = make_mapping(d_src=8, d_tgt=12, k=200)
        save_mapping(m, tmp_path)
        rng = np.random.default_rng(99)
        np.save(tmp_path / "X.npy", rng.normal(size=(50, 8)))
        np.save(tmp_path / "Y.npy", rng.normal(size=(50, 12)))

        result = runner.invoke(
            app,
            [
                "gate",
                "--mapping",
                str(tmp_path / "map.isotrieve"),
                "--source-vectors",
                str(tmp_path / "X.npy"),
                "--target-vectors",
                str(tmp_path / "Y.npy"),
                "--corpus-texts",
                str(tmp_path / "missing.txt"),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 1


class TestDoctorDomainRegime:
    def _make_medical_store(self, tmp_path):
        from isotrieve.stores.base import VectorRecord
        from isotrieve.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(tmp_path / "store", create=True)
        rng = np.random.default_rng(42)
        records = [
            VectorRecord(
                id=str(i),
                vector=rng.normal(size=(8,)),
                text=MEDICAL_TEXTS[i % len(MEDICAL_TEXTS)],
            )
            for i in range(10)
        ]
        store.write_vectors([records])
        return store

    def test_doctor_numpy_reports_domain_regime(self, tmp_path):
        self._make_medical_store(tmp_path)
        result = runner.invoke(
            app,
            ["doctor", "--store", "numpy", "--url", str(tmp_path / "store"), "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["domain_regime"] == "medical"
        assert data["n_sampled_texts"] == 10

    def test_doctor_numpy_no_text_defaults_general(self, tmp_path):
        from isotrieve.stores.base import VectorRecord
        from isotrieve.stores.numpy_files import NumpyFileStore

        store = NumpyFileStore(tmp_path / "store", create=True)
        rng = np.random.default_rng(0)
        store.write_vectors(
            [
                [
                    VectorRecord(id=str(i), vector=rng.normal(size=(8,)), text=None)
                    for i in range(5)
                ]
            ]
        )
        result = runner.invoke(
            app,
            ["doctor", "--store", "numpy", "--url", str(tmp_path / "store"), "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["domain_regime"] == "general"
        assert data["n_sampled_texts"] == 0

    def test_doctor_text_output_mentions_benchmark(self, tmp_path):
        self._make_medical_store(tmp_path)
        result = runner.invoke(
            app,
            ["doctor", "--store", "numpy", "--url", str(tmp_path / "store")],
        )
        assert result.exit_code == 0
        assert "Domain regime: medical" in result.output
        assert "--dataset scifact" in result.output


def _good_mapping():
    """Well-fitted mapping for gate tests."""
    m = make_mapping(d_src=8, d_tgt=12, k=200)
    rng = np.random.default_rng(99)
    X = rng.normal(size=(50, 8))
    W = rng.normal(size=(8, 12))
    Y = X @ W
    return m, X, Y
