#!/usr/bin/env python3
"""Regenerate README results table from benchmarks/results/*.json.

Never hand-edit the table — run this script instead.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "aecp-python" / "README.md"
MARKER_START = "<!-- BEGIN AUTO-RESULTS -->"
MARKER_END = "<!-- END AUTO-RESULTS -->"


def load_results() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def render_table(rows: list[dict]) -> str:
    if not rows:
        return (
            "_No committed results yet. Run "
            "`python benchmarks/run_benchmark.py` and re-run this script._\n"
        )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["dataset"],
            r["source_model"],
            r["target_model"],
            r["calibration_mode"],
            r["k_cal"],
        )
        groups[key].append(r)

    lines = [
        "| Dataset | Pair | K | nDCG@10 floor | nDCG@10 AECP | nDCG@10 ceiling | Retention (AECP÷ceiling) | Artifacts |",
        "|---------|------|---|---------------|--------------|-----------------|--------------------------|-----------|",
    ]
    for key, items in sorted(groups.items()):
        dataset, src, tgt, mode, k = key
        pair = f"{src.split('/')[-1]} → {tgt.split('/')[-1]}"
        floors = [i["floor"]["nDCG@10"] for i in items]
        aecps = [i["aecp"]["nDCG@10"] for i in items]
        ceils = [i["ceiling"]["nDCG@10"] for i in items]
        rets = [
            i["retention"]["nDCG@10"]
            for i in items
            if i["retention"]["nDCG@10"] is not None
        ]
        ret_s = (
            f"{statistics.mean(rets):.3f} ± {statistics.stdev(rets):.3f}"
            if len(rets) > 1
            else f"{rets[0]:.3f}"
            if rets
            else "n/a"
        )
        art = ", ".join(f"`{Path(i.get('_path', '')).name}`" for i in items)
        # Prefer listing config hashes
        art = ", ".join(f"`…{i['config_hash']}.json`" for i in items)
        lines.append(
            f"| {dataset} | {pair} ({mode}) | {k} | "
            f"{statistics.mean(floors):.3f} | "
            f"{statistics.mean(aecps):.3f} ± "
            f"{(statistics.stdev(aecps) if len(aecps)>1 else 0):.3f} | "
            f"{statistics.mean(ceils):.3f} | "
            f"**{ret_s}** | {art} |"
        )
    lines.append("")
    lines.append(
        f"_Generated from {len(rows)} file(s) in `benchmarks/results/`. "
        "Retention is the headline metric; cosine alone is misleading._"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_results()
    for r, path in zip(
        rows, sorted(RESULTS.glob("*.json")), strict=False
    ):
        r["_path"] = str(path)
    table = render_table(rows)
    block = f"{MARKER_START}\n{table}{MARKER_END}"
    text = README.read_text(encoding="utf-8")
    if MARKER_START in text and MARKER_END in text:
        pre = text.split(MARKER_START)[0]
        post = text.split(MARKER_END)[1]
        README.write_text(pre + block + post, encoding="utf-8")
    else:
        raise SystemExit("README missing AUTO-RESULTS markers")
    print("Updated", README)


if __name__ == "__main__":
    main()
