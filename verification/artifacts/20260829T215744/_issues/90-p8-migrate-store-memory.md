## Check
check_id: P8-01-streaming-memory — pre-registered: batched migration peak RSS < 40% of corpus bytes.

## Observed
NumpyFileStore→NumpyFileStore `migrate_store` (batch_size=5000) on a **200k×768 float32 corpus (614 MB)**:

- RSS before migration: 0.416 GB
- Peak during migration: **4.572 GB** (delta **4.156 GB = 6.8× corpus**, 17× the streaming budget)
- Both measurement methods agree (in-process ru_maxrss 4.57 GB; `/usr/bin/time -l` 4,571,807,744 bytes)
- Migrated=200000, results correct — this is a memory-scaling defect, not a correctness one

## Root cause (source-verified)
`NumpyFileStore.write_vectors` (src/isotrieve/stores/numpy_files.py:130-179) is called **once per batch** by `migrate_store`, and each call:

1. `existing = np.load(self.vectors_path)` — loads the ENTIRE target array into RAM (grows every call)
2. upcasts to **float64** (`np.asarray(rec.vector, dtype=np.float64)`), doubling memory vs the float32 store
3. `np.concatenate([existing, arr])` — second full copy
4. `np.save(self.vectors_path, arr)` — full rewrite of the file

→ O(target-size) RAM and O(N²/batch) rewrite I/O. At batch i of N: ~2 × (i/N) × 2 × corpus_bytes float64 live.

## Impact
A 10M×1024 float32 corpus (~40 GB on disk) would need ~160 GB+ RAM and quadratic rewrite time — OOM/abort on any realistic production store. This is the same defect class as fixed bug #9 (Qdrant materializing iterators), now in the reference file store.

## Reproduce

    python verification/artifacts/20260829T215744/P8/memory_delta.py
    # → BEFORE_GB=0.416 PEAK_GB=4.572 DELTA_GB=4.156 (corpus 0.614 GB)

Base SHA: 51c3538 · deterministic: yes

## Evidence
- verification/artifacts/20260829T215744/P8/p8_final.log
- verification/artifacts/20260829T215744/P8/memory_delta.py (output in run log)

## Triage
bucket: product-bug (confidence: high)

## Suggested fix
Append without full reload: pre-allocate a capacity-padded target via `np.lib.format.open_memmap(mode="r+")` and grow geometrically, or write batch shards and concatenate exactly once when the manifest closes (the batch tmp-file scaffolding already exists). Keep dtype float32 end-to-end (the float64 upcast in `iter_vectors`/`write_vectors` doubles resident memory for no numerical benefit at store boundaries). Add a streaming regression test asserting peak RSS < corpus × k on a 200k-row store.

## Status
[ ] Fixed in this run   [x] Deferred to owner (product code)
