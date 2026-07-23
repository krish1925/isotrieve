# Deprecation Playbook: ada-002 → text-embedding-3-small

> **Source:** OpenAI is deprecating `text-embedding-ada-002`.
> **Target:** `text-embedding-3-small` (1536d → 1536d).
> **Note:** Same dimension, but different geometric space. Raw cross-space retrieval = 0.

## Why not re-embed?

Re-embedding a 1M-vector corpus costs ~$0.13 in API calls (1M embedding calls × $0.00000013).
Isotrieve fits a mapping from 2K calibration texts for a fraction of that — without touching stored vectors.

## Cost worksheet

| Item | Re-embed | Isotrieve |
|------|----------|------|
| Embedding calls | 1,000,000 | 2,000 (calibration) |
| Estimated cost | ~$0.13 | ~$0.0003 |
| Downtime | Required | None (serve mode) |
| Rollback | Re-embed again | Instant (drop mapping) |

## Calibration size guidance

| Corpus size | Recommended K | Gate verdict |
|-------------|---------------|--------------|
| < 10K | 2,000 | PASS expected |
| 10K–100K | 2,000–4,000 | PASS expected |
| 100K–1M | 4,000 | PASS expected |
| > 1M | 4,000–8,000 | Run `isotrieve gate` |

## Step-by-step

```bash
# 1. Estimate cost
isotrieve plan --source-model text-embedding-ada-002 \
          --target-model text-embedding-3-small \
          --corpus-size 1000000

# 2. Calibrate (use in-domain texts from your corpus)
isotrieve calibrate --source-model text-embedding-ada-002 \
               --target-model text-embedding-3-small \
               --texts your_corpus_sample.txt \
               --k 2000 \
               -o ada002_to_te3s.isotrieve

# 3. Gate — verify retention
isotrieve gate --mapping ada002_to_te3s.isotrieve \
          --source-vectors X_sample.npy \
          --target-vectors Y_sample.npy

# 4. If PASS, migrate
isotrieve transform --mapping ada002_to_te3s.isotrieve \
               --source-dir ./old_store \
               --target-dir ./new_store

# 5. Switch your application to query against new_store
# 6. Keep old_store for rollback
```

## Rollback

If post-migration results are unsatisfactory:

1. Point your application back to the old store.
2. The old vectors are untouched — zero data loss.
3. If you used serve mode (no migration), simply stop using the mapping.

## What can go wrong

- **Out-of-distribution queries**: retention drops hardest on queries
  very different from calibration texts. Use in-domain calibration.
- **Quality gate FAIL**: do not migrate. Re-embed instead.
- **K too small**: below 2000, mappings are noisy. Increase calibration size.
