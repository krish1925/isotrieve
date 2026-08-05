#!/bin/bash
# Release script for isotrieve — tags a version and lets release.yml do the rest.
# Usage: ./release.sh <version>   (e.g. ./release.sh 0.4.0)
# Run from a clean checkout of main after merging the Release PR.

set -e

VERSION="${1:?Usage: ./release.sh <version> (e.g. 0.4.0)}"
REPO="krish1925/isotrieve"

echo "=== Step 1: Tag v${VERSION} on main ==="
git tag -a "v${VERSION}" -m "isotrieve ${VERSION}"
git push origin "v${VERSION}"

echo "=== Step 2: release.yml runs tests, publishes to PyPI, creates the GitHub Release ==="
gh run watch

echo "=== Step 3: Verify PyPI install ==="
python -m venv /tmp/isotrieve_verify && /tmp/isotrieve_verify/bin/pip install isotrieve && /tmp/isotrieve_verify/bin/isotrieve version

echo "=== Done! ==="
echo "Next steps:"
echo "1. Close the milestone in GitHub: https://github.com/${REPO}/milestones"
echo "2. Verify the live site: https://krish1925.github.io/isotrieve/"
echo "3. Submit to Context7: https://context7.com/add-project"
echo "4. Zenodo: link repo at zenodo.org → GitHub integration"
echo "5. Show HN post (Tuesday-Thursday morning PT)"
