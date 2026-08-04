"""HTML gate report generator."""

from __future__ import annotations

from typing import Any


def generate_gate_html(
    report: Any, ci: dict[str, Any], ss_report: Any | None = None
) -> str:
    """Generate a self-contained HTML gate report.

    ``report`` may be None when only seed-sensitivity was requested; the
    seed-sensitivity section is rendered when ``ss_report`` is provided.
    """
    ci_rows = ""
    for metric, bounds in ci.items():
        if isinstance(bounds, dict):
            lo, hi = bounds["lower"], bounds["upper"]
        else:
            lo, hi = bounds
        ci_rows += f"<tr><td>{metric}</td><td>{lo:.4f}</td><td>{hi:.4f}</td></tr>\n"

    if report is not None:
        verdict = report.verdict.value
        colors = {"PASS": "#16a34a", "WARN": "#ca8a04", "FAIL": "#dc2626"}
        color = colors.get(verdict, "#6b7280")
        metrics_rows = (
            f"<tr><td>Cosine mean</td><td>{report.cosine_mean:.4f}</td></tr>\n"
            f"<tr><td>Cosine median</td><td>{report.cosine_median:.4f}</td></tr>\n"
            f"<tr><td>Cosine p5</td><td>{report.cosine_p5:.4f}</td></tr>\n"
            f"<tr><td>Top-1 retention</td><td>{report.top1_retention:.4f}</td></tr>\n"
            f"<tr><td>Top-10 retention</td><td>{report.top10_retention:.4f}</td></tr>\n"
            f"<tr><td>Holdout rank corr</td><td>{report.holdout_rank_corr:.4f}</td></tr>\n"
            f"<tr><td>Sample size</td><td>{report.n_sample}</td></tr>\n"
        )
        header = f"<h1>Quality Gate: {verdict}</h1>"
        predicted = (
            f"<p class=\"muted\">Predicted retention: {report.predicted_retention:.4f} "
            f"(80% CI: [{report.prediction_interval[0]:.4f}, "
            f"{report.prediction_interval[1]:.4f}])</p>\n"
            f"<p>{report.rationale}</p>\n"
        )
        metrics_section = f"<h2>Metrics</h2>\n<table>\n<tr><th>Metric</th><th>Value</th></tr>\n{metrics_rows}</table>\n"
        footer = (
            f"<p class=\"muted\">Gate model: {report.gate_model_used} | "
            f"Scope: {report.gate_model_scope or 'N/A'} | "
            f"LOPO MAE: {report.lopo_error or 'N/A'}</p>\n"
        )
    else:
        color = "#6b7280"
        verdict = "SEED-SENSITIVITY"
        header = "<h1>Quality Gate: Seed Sensitivity</h1>"
        predicted = "<p class=\"muted\">Point-estimate gate skipped (no mapping).</p>\n"
        metrics_section = ""
        footer = ""

    ci_section = (
        "<h2>Confidence Intervals (Bootstrap)</h2>\n<table>\n"
        "<tr><th>Metric</th><th>Lower</th><th>Upper</th></tr>\n"
        f"{ci_rows}</table>\n"
        if ci_rows
        else ""
    )

    ss_section = ""
    if ss_report is not None:
        stability_color = "#dc2626" if ss_report.unstable else "#16a34a"
        per_seed = ", ".join(f"{v:.3f}" for v in ss_report.per_seed_retention)
        ss_section = (
            "<h2>Seed Sensitivity</h2>\n"
            "<table>\n"
            "<tr><th>Metric</th><th>Value</th></tr>\n"
            f"<tr><td>Refit runs</td><td>{ss_report.runs}</td></tr>\n"
            f"<tr><td>Retention mean ± std</td><td>{ss_report.mean_retention:.4f} ± "
            f"{ss_report.std_retention:.4f}</td></tr>\n"
            f"<tr><td>Range</td><td>[{ss_report.min_retention:.4f}, "
            f"{ss_report.max_retention:.4f}]</td></tr>\n"
            f"<tr><td>Per-seed retention</td><td>{per_seed}</td></tr>\n"
            f"<tr><td>Stability</td><td style=\"color:{stability_color};font-weight:bold\">"
            f"{'UNSTABLE' if ss_report.unstable else 'STABLE'}</td></tr>\n"
            "</table>\n"
        )

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
{header}
{predicted}
{metrics_section}
{ci_section}
{ss_section}
{footer}
</body>
</html>"""
