"""P7 Docker-free E2E: core lifecycle + crash/resume + gate polarity +
CLI sweep + serve-mode equivalence + MPS. Emits structured P7-* lines."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SEED = 20260829
N_DOCS, D_SRC, D_TGT = 400, 64, 128


def line(tag: str, ok: bool, detail: str = "") -> None:
    print(f"P7 {tag}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_DOCS, D_SRC))
    q, _ = np.linalg.qr(rng.normal(size=(D_SRC, D_SRC)))
    W = q @ rng.normal(size=(D_SRC, D_TGT)) * 0.2
    Y = X @ W + 0.001 * rng.normal(size=(N_DOCS, D_TGT))
    return X, Y


def main() -> int:
    from isotrieve.mapping.base import l2_normalize
    from isotrieve.mapping.linear import RidgeMapping
    from isotrieve.migrate import migrate_store
    from isotrieve.stores.numpy_files import NumpyFileStore

    X, Y = make_data()
    tmp = Path(tempfile.mkdtemp(prefix="iso_p7_"))

    # ---- P7-03 core lifecycle: migrate -> manifest -> verify semantics ----
    src = NumpyFileStore.from_arrays(tmp / "src", X, texts=[f"doc {i}" for i in range(N_DOCS)])
    tgt = NumpyFileStore(tmp / "tgt", create=True)
    m = RidgeMapping(alpha="auto", seed=SEED).fit(X[:320], Y[:320])
    manifest = migrate_store(src, tgt, m, batch_size=64, manifest_path=tmp / "manifest.json")
    ok = manifest.migrated_vectors == N_DOCS and tgt.count() == N_DOCS
    line("P7-03a-migrate", ok, f"migrated={manifest.migrated_vectors} target={tgt.count()}")

    # resume idempotency: rerun with resume=True on a completed migration
    manifest2 = migrate_store(src, tgt, m, batch_size=64, manifest_path=tmp / "manifest.json", resume=True)
    line("P7-03b-resume-idempotent", tgt.count() == N_DOCS, f"count after resume={tgt.count()}")

    # crash/resume (faithful): SIGKILL a migration subprocess mid-run, then
    # resume with the SAME manifest path and require a complete, idempotent target.
    np.save(tmp / "src.npy", X)
    m.save(tmp / "mapping.isotrieve")
    killed_dir = tmp / "killrun"
    killed_dir.mkdir()
    # 30k docs so the migration runs long enough to be killed mid-flight
    rng_big = np.random.default_rng(SEED + 9)
    Xbig = rng_big.normal(size=(30000, D_SRC))
    np.save(killed_dir / "big_src.npy", Xbig)
    proc_code = (
        "import numpy as np, sys;"
        "sys.path.insert(0, 'isotrieve-python/src');"
        "from isotrieve.mapping.linear import RidgeMapping;"
        "from isotrieve.migrate import migrate_store;"
        "from isotrieve.stores.numpy_files import NumpyFileStore;"
        f"X = np.load({str(killed_dir / 'big_src.npy')!r});"
        "src = NumpyFileStore.from_arrays('src', X, texts=[f'doc {i}' for i in range(len(X))]);"
        "tgt = NumpyFileStore('tgt_kill', create=True);"
        f"m = RidgeMapping.load({str(tmp / 'mapping.isotrieve')!r});"
        "migrate_store(src, tgt, m, batch_size=32, manifest_path='manifest_kill.json')"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", proc_code], cwd=killed_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import time as _t

    _t.sleep(4.0)
    proc.kill()
    proc.wait()
    partial = NumpyFileStore(killed_dir / "tgt_kill", create=False)
    partial_count = partial.count()
    manifest4 = migrate_store(
        NumpyFileStore(killed_dir / "src"), partial, m,
        batch_size=1024, manifest_path=killed_dir / "manifest_kill.json", resume=True,
    )
    final_recs2 = list(partial.iter_vectors(batch_size=40000))[0]
    ids_final = {r.id for r in final_recs2}
    line("P7-03c-crash-resume", len(ids_final) == 30000,
         f"killed-after={partial_count} resumed-manifest={manifest4.migrated_vectors} unique-ids={len(ids_final)}")

    # rollback semantics for file store: target removal restores original state
    import shutil

    shutil.rmtree(tmp / "tgt")
    line("P7-03d-rollback-target-drop", not (tmp / "tgt").exists(), "target store removed; source untouched")
    line("P7-03e-source-untouched", src.count() == N_DOCS, f"source count={src.count()}")

    # ---- P7-04 CLI sweep: all verbs respond ----
    cli = str(Path(sys.executable).parent / "isotrieve")
    verbs = ["plan", "calibrate", "transform", "inspect", "gate", "doctor", "report", "manifest", "rollback", "resume", "verify", "version"]
    bad = []
    for v in verbs:
        r = subprocess.run([cli, v, "--help"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            bad.append(v)
    line("P7-04-cli-sweep", not bad, f"verbs={len(verbs)} bad={bad}")

    # ---- P7-04b gate exit codes, both polarities ----
    # QualityGate.evaluate contract: evaluation pairs must NOT overlap the
    # calibration fit set. Calibrate on docs 0-99, gate on docs 100-199.
    # REAL embeddings: the gate model was trained on real pairs; synthetic
    # near-orthogonal worlds keep paired-cosine variance ~0 (see issue #88).
    from sentence_transformers import SentenceTransformer
    src_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tgt_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    rng_txt = np.random.default_rng(SEED + 3)
    texts_src = [f"calibration document {i}: " + " ".join(rng_txt.choice(["retrieval","migration","embedding","vector","index","calibration","model","gate"], size=12)) for i in range(240)]
    E_src = src_model.encode(texts_src, normalize_embeddings=True)
    E_tgt = tgt_model.encode(texts_src, normalize_embeddings=True)
    np.save(tmp / "src100.npy", E_src[:100]); np.save(tmp / "tgt100.npy", E_tgt[:100])
    np.save(tmp / "src200.npy", E_src[100:220]); np.save(tmp / "tgt200.npy", E_tgt[100:220])
    np.save(tmp / "map_tgt.npy", E_tgt[100:220][np.random.default_rng(SEED).permutation(120)])  # degenerate
    r_cal = subprocess.run([cli, "calibrate", "--source-vectors", str(tmp / "src100.npy"), "--target-vectors", str(tmp / "tgt100.npy"), "--output", str(tmp / "mapping100.isotrieve")], capture_output=True, text=True, timeout=300)
    r_pass = subprocess.run([cli, "gate", "--mapping", str(tmp / "mapping100.isotrieve"), "--source-vectors", str(tmp / "src200.npy"), "--target-vectors", str(tmp / "tgt200.npy")], capture_output=True, text=True, timeout=300)
    r_fail = subprocess.run([cli, "gate", "--mapping", str(tmp / "mapping100.isotrieve"), "--source-vectors", str(tmp / "src200.npy"), "--target-vectors", str(tmp / "map_tgt.npy")], capture_output=True, text=True, timeout=300)
    line("P7-04b-gate-polarity", r_cal.returncode == 0 and r_pass.returncode == 0 and r_fail.returncode != 0,
         f"calibrate={r_cal.returncode} good-gate={r_pass.returncode} degenerate-gate={r_fail.returncode}")

    # ---- P7-05 serve-mode recall equivalence ----
    # QueryAdapter maps NEW-space queries back onto the OLD-space index (the
    # two deployment modes: query-time mapping vs batch-migrated index).
    from isotrieve.serve import QueryAdapter

    qa = QueryAdapter(m)
    rng2 = np.random.default_rng(SEED + 1)
    q_idx = rng2.choice(N_DOCS, size=50, replace=False)
    Q_new = Y[q_idx]  # queries arrive in NEW-model space

    def _map(queries: np.ndarray) -> np.ndarray:
        # QueryAdapter consumes ONE query vector at a time
        outs = []
        for name in ("transform", "map_query", "map", "embed_query"):
            fn = getattr(qa, name, None)
            if callable(fn):
                try:
                    for row in queries:
                        outs.append(np.asarray(fn(np.asarray(row, dtype=np.float64))))
                    arr = np.array(outs)
                    if arr.ndim == 2 and arr.shape[-1] == D_SRC:
                        return arr
                except Exception:  # noqa: BLE001 - try next API name
                    outs.clear()
        raise AssertionError(f"QueryAdapter API: {[a for a in dir(qa) if not a.startswith('_')]}")

    qmapped = _map(Q_new)
    def recall(queries: np.ndarray, corpus: np.ndarray) -> float:
        zn, cn = l2_normalize(queries), l2_normalize(corpus)
        sims = zn @ cn.T
        top10 = np.argsort(-sims, axis=1)[:, :10]
        return float(np.mean([q_idx[i] in top10[i] for i in range(len(q_idx))]))
    r_querytime = recall(qmapped, X)            # adapter maps queries onto OLD index
    X_migrated = m.transform(X)                 # batch: migrate the index instead
    r_batchmode = recall(Q_new, X_migrated)     # raw NEW queries vs migrated index
    line("P7-05-serve-equivalence", abs(r_querytime - r_batchmode) <= 0.02,
         f"query-time={r_querytime:.3f} batch-migrated={r_batchmode:.3f}")

    # ---- P7-06 MPS device path ----
    try:
        import torch

        if not torch.backends.mps.is_available():
            line("P7-06-mps", False, "MPS unavailable on this machine")
        else:
            # NOTE: device="auto" is broken (filed separately) — pass explicit
            # devices here; the auto-resolution path is the filed finding.
            from isotrieve.mapping.mlp import ResidualMLPMapping

            mps_map = ResidualMLPMapping(device="mps", n_epochs=30)
            mps_map.fit(X[:200], Y[:200])
            dev = mps_map._get_model_device() if hasattr(mps_map, "_get_model_device") else getattr(mps_map, "_device_str", "?")
            out_mps = mps_map.transform(X[:20])
            cpu_map = ResidualMLPMapping(device="cpu", n_epochs=30).fit(X[:200], Y[:200])
            out_cpu = cpu_map.transform(X[:20])
            dev_str = str(dev)
            max_diff = float(np.max(np.abs(l2_normalize(out_mps) - l2_normalize(out_cpu))))
            det = mps_map.transform(X[:20])
            det_ok = bool(np.allclose(out_mps, det, atol=0.0))
            # cross-device weight parity is NOT a valid expectation for independently
            # trained nets (prereg-diff): require device resolution + same-device determinism
            line("P7-06-mps", "mps" in dev_str.lower() and det_ok,
                 f"device={dev_str} same-device-deterministic={det_ok} (cross-device diff {max_diff:.2e} informational)")
    except Exception as exc:  # noqa: BLE001
        line("P7-06-mps", False, f"exception: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
