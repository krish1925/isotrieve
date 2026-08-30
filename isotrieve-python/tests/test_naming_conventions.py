"""Cross-package naming-convention regression tests.

The repo was renamed AECP -> Isotrieve in v0.2.1. These tests pin the
rename so remnants cannot silently reappear in either package (Python
or npm), in live docs, or in packaging metadata.

Historical records (CHANGELOG history, verification/ reports, git
history) intentionally keep the old name and are never scanned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NPM_PACKAGES = REPO_ROOT / "isotrieve-npm" / "packages"

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "isotrieve-npm").exists(),
    reason="monorepo layout not available (running outside the repo checkout)",
)

# Live source surfaces where the pre-rename name must never appear.
NO_AECP_GLOBS = [
    "isotrieve-python/src/**/*.py",
    "isotrieve-python/notebooks/*.ipynb",
    "isotrieve-npm/packages/*/src/**/*.ts",
    "isotrieve-npm/packages/*/README.md",
    "isotrieve-npm/packages/*/package.json",
    "isotrieve-website/*.html",
]


def _iter_files(pattern: str):
    # Only tracked-ish surfaces; skip caches and build output.
    for path in REPO_ROOT.glob(pattern):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and "dist" not in path.parts
        ):
            yield path


class TestLicenseAttribution:
    """Both packages attribute copyright to Isotrieve, not the pre-rename name."""

    def test_python_license(self):
        content = (REPO_ROOT / "isotrieve-python" / "LICENSE").read_text()
        assert "Isotrieve Contributors" in content
        assert "AECP Contributors" not in content

    def test_npm_license(self):
        content = (REPO_ROOT / "isotrieve-npm" / "LICENSE").read_text()
        assert "Isotrieve Contributors" in content
        assert "AECP Contributors" not in content


class TestNoPreRenameNameInLiveSurfaces:
    """No 'aecp' (any case) in live source, docs, or packaging metadata."""

    @pytest.mark.parametrize("pattern", NO_AECP_GLOBS)
    def test_no_aecp(self, pattern: str):
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in _iter_files(pattern)
            if "aecp" in path.read_text(errors="ignore").lower()
        ]
        assert not offenders, f"pre-rename 'aecp' found in: {offenders}"


class TestNotebookColabLinks:
    """Notebook Colab badges must point at the current repo, not the old one."""

    def test_colab_urls(self):
        notebooks = sorted(
            (REPO_ROOT / "isotrieve-python" / "notebooks").glob("*.ipynb")
        )
        assert notebooks, "expected demo notebooks to exist"
        offenders = [
            nb.name
            for nb in notebooks
            if "krish1925/AECP" in nb.read_text(errors="ignore")
        ]
        assert not offenders, f"notebooks link to the pre-rename repo: {offenders}"


class TestNpmPackageLayout:
    """npm workspace package names/paths match the rename."""

    def test_demo_cli_directory(self):
        assert (NPM_PACKAGES / "isotrieve-demo-cli").is_dir()
        assert not (NPM_PACKAGES / "aecp-demo-cli").exists()

    def test_lockfile_matches_directory(self):
        lock = REPO_ROOT / "isotrieve-npm" / "package-lock.json"
        content = lock.read_text()
        assert "packages/isotrieve-demo-cli" in content
        assert "packages/aecp-demo-cli" not in content

    def test_workspace_core_ranges_resolvable(self):
        """@isotrieve/core is an unpublished workspace package.

        Version ranges like ^1.0.0 cannot resolve against the local
        workspace (core is 0.1.0) or the registry, which breaks
        `npm install` for the whole monorepo. Workspace deps must use '*'.
        """
        core = json.loads((NPM_PACKAGES / "core" / "package.json").read_text())
        core_version = core["version"]
        offenders = []
        for pkg_json in NPM_PACKAGES.glob("*/package.json"):
            if pkg_json.parent.name == "core":
                continue
            data = json.loads(pkg_json.read_text())
            for section in ("dependencies", "devDependencies"):
                dep = data.get(section, {}).get("@isotrieve/core")
                if dep is not None and dep != "*":
                    offenders.append(f"{pkg_json.parent.name} [{section}]: {dep}")
        assert not offenders, (
            f"unresolvable @isotrieve/core ranges (core@{core_version} "
            f"is an unpublished workspace dep): {offenders}"
        )


class TestRootReadme:
    """The root README documents the actual repo layout."""

    def test_references_skills_not_agents(self):
        content = (REPO_ROOT / "README.md").read_text()
        assert "SKILLS.md" in content
        assert "AGENTS.md" not in content
