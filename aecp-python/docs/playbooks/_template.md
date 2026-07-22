# Deprecation Playbook: {{ SOURCE_MODEL }} → {{ TARGET_MODEL }}

> **Source:** {{ source_provider }} is deprecating `{{ SOURCE_MODEL }}`.
> **Target:** `{{ TARGET_MODEL }}` ({{ target_dim }}d).
> **Timeline:** {{ deprecation_date }}.

## Why not re-embed?

Re-embedding a {{ corpus_size }}-vector corpus costs ~${{ reembed_cost_usd }} in API calls
({{ reembed_calls }} embedding calls × ${{ cost_per_call }}). AECP fits a mapping from
{{ cal_k }} calibration texts for ~${{ cal_cost_usd }} and achieves {{ expected_retention }}%
retention — without touching stored vectors.

## Cost worksheet

| Item | Re-embed | AECP |
|------|----------|------|
| Embedding calls | {{ reembed_calls:, }} | {{ cal_k }} (calibration) |
| Estimated cost | ${{ reembed_cost_usd }} | ${{ cal_cost_usd }} |
| Downtime | Required | None (serve mode) |
| Rollback | Re-embed again | Instant (drop mapping) |

## Calibration size guidance

| Corpus size | Recommended K | Gate verdict |
|-------------|---------------|--------------|
| < 10K | 2,000 | PASS expected |
| 10K–100K | 2,000–4,000 | PASS expected |
| 100K–1M | 4,000 | PASS expected |
| > 1M | 4,000–8,000 | Run `aecp gate` |

> K ≥ 2000 is required for a reliable mapping. If no CLAIMS.md row exists for your
> model pair, run `aecp gate` on your corpus — never guess.

## Step-by-step

```bash
# 1. Estimate cost
aecp plan --source-model {{ SOURCE_MODEL }} \
          --target-model {{ TARGET_MODEL }} \
          --corpus-size {{ corpus_size }}

# 2. Calibrate (use in-domain texts from your corpus)
aecp calibrate --source-model {{ SOURCE_MODEL }} \
               --target-model {{ TARGET_MODEL }} \
               --texts your_corpus_sample.txt \
               --k {{ cal_k }} \
               -o mapping.aecp

# 3. Gate — verify retention
aecp gate --mapping mapping.aecp \
          --source-vectors X_sample.npy \
          --target-vectors Y_sample.npy

# 4. If PASS, migrate
aecp transform --mapping mapping.aecp \
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
- **Dimension change** ({{ source_dim }} → {{ target_dim }}): AECP handles
  non-square transforms. Procrustes will refuse with a clear error — use
  RidgeMapping (default).
- **Quality gate FAIL**: do not migrate. Re-embed instead.
- **K too small**: below 2000, mappings are noisy. Increase calibration size.
