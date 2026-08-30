"""P8 (SOLO): streaming-memory proof for batched migration.
Measures peak RSS two ways for NumpyFileStore -> NumpyFileStore migrate_store."""
from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SNIPPET = """
import resource, sys
import numpy as np
from pathlib import Path
from isotrieve.mapping.linear import RidgeMapping
from isotrieve.migrate import migrate_store
from isotrieve.stores.numpy_files import NumpyFileStore
n, d = {n}, {d}
rng = np.random.default_rng(7)
X = rng.normal(size=(n, d)).astype(np.float32)
import tempfile
tmp = Path(tempfile.mkdtemp())
src = NumpyFileStore.from_arrays(tmp / "src", X, texts=[f"doc {{i}}" for i in range(n)])
tgt = NumpyFileStore(tmp / "tgt", create=True)
m = RidgeMapping(alpha="auto", seed=0).fit(X[:2000].astype(np.float64), (X[:2000] @ rng.normal(size=(d, d))).astype(np.float64))
corpus_bytes = X.nbytes
manifest = migrate_store(src, tgt, m, batch_size=5000)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024  # -> GB
print(f"CORPUS_BYTES={{corpus_bytes}} PEAK_GB={{peak:.3f}} MIGRATED={{manifest.migrated_vectors}}")
"""


def run(n: int, d: int) -> dict:
    # Method 1: in-process rusage reported by the child
    r1 = subprocess.run([sys.executable, "-c", SNIPPET.format(n=n, d=d)],
                        capture_output=True, text=True, timeout=1200, cwd="isotrieve-python")
    # Method 2: /usr/bin/time -l wrapper (peak RSS of the whole process)
    r2 = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-c", SNIPPET.format(n=n, d=d)],
                        capture_output=True, text=True, timeout=1200, cwd="isotrieve-python")
    out1 = [l for l in r1.stdout.splitlines() if l.startswith("CORPUS_BYTES")]
    time_line = [l for l in r2.stderr.splitlines() if "maximum resident set size" in l]
    return {
        "n": n, "d": d,
        "method1": out1[0] if out1 else r1.stderr[-200:],
        "method2_max_rss_bytes": (int(time_line[0].split()[0]) if time_line else None),
        "corpus_gb": n * d * 4 / 1e9,
        "budget_gb": 0.4 * n * d * 4 / 1e9,
    }


if __name__ == "__main__":
    for n, d in [(200_000, 768), (1_000_000, 384)]:  # 1M may ENOSPC -> recorded honestly
        res = run(n, d)
        print("P8-RESULT", res, flush=True)
