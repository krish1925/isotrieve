"""HTML gate report generator."""

from __future__ import annotations

from typing import Any


def generate_gate_html(report: Any, ci: dict[str, Any]) -> str:
    """Generate a self-contained HTML gate report."""
    verdict = report.verdict.value
    colors = {"PASS": "#16a34a", "WARN": "#ca8a04", "FAIL": "#dc2626"}
    color = colors.get(verdict, "#6b7280")

    ci_rows = ""
    for metric, bounds in ci.items():
        if isinstance(bounds, dict):
            lo, hi = bounds["lower"], bounds["upper"]
        else:
            lo, hi = bounds
        ci_rows += f"<tr><td>{metric}</td><td>{lo:.4f}</td><td>{hi:.4f}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AEC Gate Report — {verdict}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ color: {color}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f3f4f6; }}
  .muted {{ color: #6b7280; font-size: 0.875rem; }}
</style>
</head>
<body>
<h1>Quality Gate: {verdict}</h1>
<p class="muted">Predicted retention: {report.predicted_retention:.4f}
(80% CI: [{report.prediction_interval[0]:.4f}, {report.prediction_interval[1]:.4f}])</p>
<p>{report.rationale}</p>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Cosine mean</td><td>{report.cosine_mean:.4f}</td></tr>
<tr><td>Cosine median</td><td>{report.cosine_median:.4f}</td></tr>
<tr><td>Cosine p5</td><td>{report.cosine_p5:.4f}</td></tr>
<tr><td>Top-1 retention</td><td>{report.top1_retention:.4f}</td></tr>
<tr><td>Top-10 retention</td><td>{report.top10_retention:.4f}</td></tr>
<tr><td>Holdout rank corr</td><td>{report.holdout_rank_corr:.4f}</td></tr>
<tr><td>Sample size</td><td>{report.n_sample}</td></tr>
</table>

<h2>Confidence Intervals (Bootstrap)</h2>
<table>
<tr><th>Metric</th><th>Lower</th><th>Upper</th></tr>
{ci_rows}
</table>

<p class="muted">Gate model: {report.gate_model_used} | Scope: {report.gate_model_scope or "N/A"} | LOPO MAE: {report.lopo_error or "N/A"}</p>
</body>
</html>"""
