# Deprecation Playbook: ada-002 → text-embedding-3-large

> **Source:** OpenAI is deprecating `text-embedding-ada-002` (1536d).
> **Target:** `text-embedding-3-large` (3072d).
> **Note:** Dimension change (1536 → 3072). Non-square transform path required.

## Why not re-embed?

Re-embedding a 1M-vector corpus costs ~$0.26 in API calls. AECP handles the dimension
change via ridge regression (non-square) without re-embedding. The 3072d target gives
better retrieval quality than 1536d — you get an upgrade, not just a migration.

## Cost worksheet

| Item | Re-embed | AECP |
|------|----------|------|
| Embedding calls | 1,000,000 | 2,000 (calibration) |
| Estimated cost | ~$0.26 | ~$0.0005 |
| Downtime | Required | None (serve mode) |
| Rollback | Re-embed again | Instant (drop mapping) |

## Dimension handling

AECP's `RidgeMapping` handles non-square transforms (1536 → 3072) natively.
`OrthogonalProcrustesMapping` requires equal dimensions and will refuse with a
clear error — use the default RidgeMapping.

## Calibration size guidance

| Corpus size | Recommended K | Gate verdict |
|-------------|---------------|--------------|
| < 10K | 2,000 | PASS expected |
| 10K–100K | 2,000–4,000 | PASS expected |
| 100K–1M | 4,000 | PASS expected |
| > 1M | 4,000–8,000 | Run `aecp gate` |

## Step-by-step

```bash
# 1. Estimate cost
aecp plan --source-model text-embedding-ada-002 \
          --target-model text-embedding-3-large \
          --corpus-size 1000000

# 2. Calibrate (use in-domain texts from your corpus)
aecp calibrate --source-model text-embedding-ada-002 \
               --target-model text-embedding-3-large \
               --texts your_corpus_sample.txt \
               --k 2000 \
               -o ada002_to_te3l.aecp

# 3. Gate — verify retention
aecp gate --mapping ada002_to_te3l.aecp \
          --source-vectors X_sample.npy \
          --target-vectors Y_sample.npy

# 4. If PASS, migrate
aecp transform --mapping ada002_to_te3l.aecp \
               --source-dir ./old_store \
               --target-dir ./new_store

# 5. Update your index schema for 3072d vectors
# 6. Switch your application to query against new_store
# 7. Keep old_store for rollback
```

## Rollback

If post-migration results are unsatisfactory:

1. Point your application back to the old store.
2. The old vectors are untouched — zero data loss.
3. If you used serve mode (no migration), simply stop using the mapping.

## What can go wrong

- **Out-of-distribution queries**: retention drops hardest on queries
  very different from calibration texts. Use in-domain calibration.
- **Index schema change**: 3072d vectors need more storage. Budget ~2x disk.
- **Quality gate FAIL**: do not migrate. Re-embed instead.
- **K too small**: below 2000, mappings are noisy. Increase calibration size.
