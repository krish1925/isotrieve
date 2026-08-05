#!/usr/bin/env python3
"""Lint CLAIMS.md: verify artifact paths resolve to committed files.

Run from repo root:
    python isotrieve-python/scripts/lint_claims.py

Exit 0 if all claims are backed by artifacts, exit 1 otherwise.
"""

from __future__ import annotations

import glob as globmod
import re
import sys
from pathlib import Path

CLAIMS_PATH = Path("isotrieve-python/CLAIMS.md")
# Benchmark results live at the monorepo root
RESULTS_BASES = [
    Path("benchmarks/results"),
    Path("isotrieve-python/benchmarks/results"),
]
# Source files live under isotrieve-python/
SRC_BASES = [
    Path("isotrieve-python"),
    Path("."),
]


def parse_claims_table(text: str) -> list[dict[str, str]]:
    """Parse the Active claims markdown table into rows."""
    rows: list[dict[str, str]] = []
    in_active = False
    header_seen = False

    for line in text.splitlines():
        if line.strip().startswith("## Active claims"):
            in_active = True
            continue
        if line.strip().startswith("## "):
            in_active = False
            continue
        if not in_active:
            continue

        if re.match(r"^\|[-|]+\|$", line.strip()):
            header_seen = True
            continue
        if not header_seen:
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue

        rows.append(
            {
                "claim": cells[0],
                "where_used": cells[1] if len(cells) > 1 else "",
                "artifact_path": cells[2] if len(cells) > 2 else "",
                "verified": cells[3] if len(cells) > 3 else "",
                "notes": cells[4] if len(cells) > 4 else "",
            }
        )

    return rows


def extract_patterns(raw: str) -> list[str]:
    """Extract glob patterns from a CLAIMS.md artifact path cell.

    Handles backtick wrapping, brace expansion like {0,1,2},
    and comma-separated patterns.
    """
    # Strip backticks
    cleaned = raw.strip().strip("`")

    # If it's a non-glob marker, return as-is
    if cleaned in ("same", "—", "N/A", ""):
        return [cleaned]

    # Expand brace patterns: {a,b,c} -> [a, b, c]
    # Iteratively expand all brace groups
    expanded_parts: list[str] = [cleaned]
    brace_re = re.compile(r"\{([^}]+)\}")

    changed = True
    while changed:
        changed = False
        new_parts: list[str] = []
        for part in expanded_parts:
            match = brace_re.search(part)
            if not match:
                new_parts.append(part)
                continue
            options = match.group(1).split(",")
            before = part[: match.start()]
            after = part[match.end() :]
            for opt in options:
                new_parts.append(before + opt.strip() + after)
            changed = True
        expanded_parts = new_parts

    # Now split on commas for any remaining comma-separated patterns
    # But only top-level commas (not inside remaining braces)
    result: list[str] = []
    for part in expanded_parts:
        if "{" in part:
            # Has remaining braces — treat as single pattern
            result.append(part.strip())
        else:
            result.extend(p.strip() for p in part.split(",") if p.strip())

    return result


def expand_pattern(pattern: str) -> list[Path]:
    """Expand a single glob pattern against known base directories."""
    if pattern in ("same", "—", "N/A", ""):
        return [Path(".")]  # sentinel: "verified by reference"

    results: list[Path] = []

    # Determine which bases to search
    if pattern.startswith("src/") or pattern.startswith("src\\"):
        bases = SRC_BASES
    elif pattern.startswith("benchmarks/"):
        bases = [Path(".")]  # relative to repo root
    else:
        bases = RESULTS_BASES + [Path(".")]

    for base in bases:
        full = str(base / pattern)
        matches = globmod.glob(full, recursive=True)
        results.extend(Path(m) for m in matches)

    return results


def check_retired_claims(readme_text: str) -> list[str]:
    """Check that retired claim strings don't appear in README."""
    errors: list[str] = []

    retired = [
        "97% semantic fidelity",
        "97.2%",
        "97.35%",
        "<10ms transfer",
        "<1ms transfer",
        "85% Top-1",
        "86% corpus fidelity",
        "43% text baseline",
        "150x faster",
        "200x faster",
        "300k vocab",
        "zero overfitting",
    ]

    for pattern in retired:
        if pattern.lower() in readme_text.lower():
            errors.append(f"RETIRED CLAIM in README.md: '{pattern}'")

    return errors


def main() -> int:
    if not CLAIMS_PATH.exists():
        print(f"ERROR: {CLAIMS_PATH} not found")
        return 1

    text = CLAIMS_PATH.read_text()
    rows = parse_claims_table(text)
    errors: list[str] = []
    warnings: list[str] = []

    if not rows:
        errors.append("No active claims found in CLAIMS.md table")
        return 1

    print(f"Found {len(rows)} active claims in CLAIMS.md")

    for i, row in enumerate(rows, 1):
        claim_short = row["claim"][:60] + ("..." if len(row["claim"]) > 60 else "")
        raw_path = row["artifact_path"].strip()

        if not raw_path:
            warnings.append(f"Claim {i} ({claim_short}): no artifact path")
            continue

        patterns = extract_patterns(raw_path)
        any_found = False

        for pattern in patterns:
            if pattern in ("same", "—", "N/A", ""):
                any_found = True
                continue

            matches = expand_pattern(pattern)
            if matches:
                any_found = True
            else:
                warnings.append(f"Claim {i}: artifact '{pattern}' -> 0 files")

        if not any_found and patterns and patterns[0] not in ("same", "—", "N/A", ""):
            errors.append(f"Claim {i} ({claim_short}): NO artifacts found")

    # Check retired claims
    readme_path = Path("isotrieve-python/README.md")
    if readme_path.exists():
        retired = check_retired_claims(readme_path.read_text())
        errors.extend(retired)

    # Report
    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print("\nFAIL")
        return 1

    print("OK: All claims verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
