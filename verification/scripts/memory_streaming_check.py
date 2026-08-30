"""P8-delta: attribute memory to the LIBRARY (migrate_store) not the fixture.
Child builds a 200kx768 store via batched writes (no big arrays), then a second
process opens it, records RSS before/after migrate_store."""
import subprocess, sys
SETUP = """
import numpy as np
from pathlib import Path
from isotrieve.stores.numpy_files import NumpyFileStore
from isotrieve.stores.base import VectorRecord
rng = np.random.default_rng(7)
tmp = Path("p8store"); tmp.mkdir(exist_ok=True)
store = NumpyFileStore(tmp, create=True)
B, d = 10000, 768
for off in range(0, 200000, B):
    Xb = rng.normal(size=(B, d)).astype(np.float32)
    store.write_vectors([VectorRecord(id=f"doc{off+i}", vector=Xb[i], text=f"doc {off+i}") for i in range(B)])
print("SETUP_DONE count=", store.count())
"""
MIGRATE = """
import resource, numpy as np
from isotrieve.mapping.linear import RidgeMapping
from isotrieve.migrate import migrate_store
from isotrieve.stores.numpy_files import NumpyFileStore
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes
src = NumpyFileStore("p8store")
tgt = NumpyFileStore("p8tgt", create=True)
rng = np.random.default_rng(0)
W = rng.normal(size=(768, 768)) * 0.02
m = RidgeMapping(alpha="auto", seed=0).fit(rng.normal(size=(2000, 768)), rng.normal(size=(2000, 768)) @ W)
before = rss_gb()
manifest = migrate_store(src, tgt, m, batch_size=5000)
after = rss_gb()
print(f"BEFORE_GB={before:.3f} PEAK_GB={after:.3f} DELTA_GB={after-before:.3f} MIGRATED={manifest.migrated_vectors} CORPUS_GB=0.614 BUDGET_GB=0.246")
"""
r1 = subprocess.run([sys.executable, "-c", SETUP], capture_output=True, text=True, timeout=1800, cwd="isotrieve-python")
print(r1.stdout.strip()[-60:] or r1.stderr[-300:])
r2 = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-c", MIGRATE], capture_output=True, text=True, timeout=3600, cwd="isotrieve-python")
print([l for l in r2.stdout.splitlines() if "BEFORE_GB" in l])
print([l for l in r2.stderr.splitlines() if "maximum resident" in l])
