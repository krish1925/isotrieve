# When re-embedding is impossible

There are cases where switching embedding models is necessary but re-embedding the
corpus is not possible. AECP is the only option in these scenarios.

## The four scenarios

### 1. Chunking drift

Your chunking pipeline changed — different chunk size, different overlap, different
splitter. The original text is still in your database, but the chunks that produced
the vectors no longer match. Re-embedding would require re-chunking + re-embedding
the entire corpus, which may be prohibitively expensive or slow.

**AECP's approach:** The gate measures retention using *queries*, which usually still
exist even when the original chunks don't. Calibrate on the overlap between old and
new chunking (whatever texts are available in both formats), fit a mapping, and
transform stored vectors.

### 2. Compliance deletion

Regulations (GDPR, CCPA) required you to delete source documents. The vectors
remain in your index — they're derived data, not the source text itself. But you
can no longer re-embed because the source text is gone.

**AECP's approach:** Use `aecp calibrate --queries-only` with a query log. Queries
usually still exist (they're what your users searched for). Embed the queries under
both models, fit a mapping against the stored vectors, and transform.

### 3. Dead sources

Scraped web pages are gone. APIs have been sunset. OCR/transcription pipelines
have been decommissioned. The vectors exist but the text that produced them does not.

**AECP's approach:** Same as compliance deletion — query-only calibration. If you
have any subset of the original texts (backups, samples, logs), use those for
standard calibration instead.

### 4. Expensive preprocessing

Your pipeline includes steps that are expensive to reproduce: custom tokenization,
domain-specific preprocessing, embedding-specific prompt templates. Re-embedding
would require re-running the entire preprocessing pipeline, not just the embedding
step.

**AECP's approach:** Standard calibration with paired texts. The preprocessing
only needs to run on the calibration sample (2K texts), not the full corpus.

## Key insight

The gate measures retrieval retention using **queries** — and queries usually still
exist even when documents don't. This is the fundamental reason AECP works in these
scenarios. You don't need the source text to measure whether the migration preserved
retrieval quality; you just need queries and their expected results.

## When NOT to use AECP

Even in these scenarios, AECP may not be the right choice:

- If you have backups of the source text, re-embedding is simpler and guarantees
  perfect quality.
- If the calibration sample is too small (< 2000 texts), the mapping quality
  degrades. Use `aecp gate` to verify.
- If the query distribution has shifted dramatically since the vectors were created,
  retention may be poor regardless of the migration method.

## What can go wrong

- **Out-of-distribution queries**: retention drops hardest on queries very different
  from calibration texts. Use in-domain calibration whenever possible.
- **Domain mismatch**: calibration texts from one domain don't transfer well to
  another. A code search engine calibrated on prose will have poor retention.
- **Quality gate FAIL**: do not migrate. The gate is telling you the mapping is
  unreliable. Consider re-embedding with whatever text is available, or accept the
  quality loss.
