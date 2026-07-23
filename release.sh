#!/bin/bash
# Release script for isotrieve v0.2.0
# Run after: gh auth login

set -e

VERSION="0.2.0"

echo "=== Step 0: Build dist ==="
cd isotrieve-python
pip install build 2>/dev/null
python -m build
cd ..

echo "=== Step 1: Make repo public ==="
gh repo edit krish1925/Isotrieve --visibility public --accept-visibility-change-consequences

echo "=== Step 2: Edit repo description and topics ==="
gh repo edit krish1925/Isotrieve \
  --description "Migrate vector DBs to a new embedding model without re-embedding. Linear maps from ~2K calibration texts, ~87-91% retrieval retention (BEIR-benchmarked)." \
  --add-topic embeddings \
  --add-topic vector-database \
  --add-topic rag \
  --add-topic qdrant \
  --add-topic pgvector \
  --add-topic machine-learning \
  --add-topic embedding-migration \
  --add-topic python

echo "=== Step 3: Tag v${VERSION} ==="
cd isotrieve-python
git tag -a "v${VERSION}" -m "isotrieve ${VERSION} — adapters, serve mode, quality gate v2, recalibration"
git push origin "v${VERSION}"
cd ..

echo "=== Step 4: Watch the workflow ==="
gh run watch

echo "=== Step 5: Create GitHub release ==="
gh release create "v${VERSION}" isotrieve-python/dist/* \
  --title "isotrieve ${VERSION}" \
  --notes-file isotrieve-python/RELEASE_NOTES.md

echo "=== Step 6: Verify PyPI ==="
python -m venv /tmp/isotrieve_verify && /tmp/isotrieve_verify/bin/pip install isotrieve && /tmp/isotrieve_verify/bin/isotrieve version

echo "=== Done! ==="
echo "Next steps:"
echo "1. Submit to Context7: https://context7.com/add-project"
echo "2. Zenodo: link repo at zenodo.org → GitHub integration"
echo "3. Show HN post (Tuesday-Thursday morning PT)"
