#!/usr/bin/env python3
"""Audit two benchmark result files field-by-field.

Usage:
    python audit_configs.py result1.json result2.json

Prints every differing field. Keeps the comparison honest.
"""
import json
import sys
from pathlib import Path


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, key))
        else:
            items[key] = v
    return items


def main():
    if len(sys.argv) != 3:
        print("Usage: audit_configs.py result1.json result2.json")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]
    d1 = json.loads(Path(path1).read_text())
    d2 = json.loads(Path(path2).read_text())

    f1 = flatten(d1)
    f2 = flatten(d2)

    all_keys = sorted(set(f1.keys()) | set(f2.keys()))

    diffs = []
    for key in all_keys:
        v1 = f1.get(key)
        v2 = f2.get(key)
        if v1 != v2:
            diffs.append((key, v1, v2))

    if not diffs:
        print("Files are identical.")
    else:
        print(f"Found {len(diffs)} differing fields:\n")
        for key, v1, v2 in diffs:
            print(f"  {key}:")
            print(f"    file1: {v1}")
            print(f"    file2: {v2}")
            print()


if __name__ == "__main__":
    main()
