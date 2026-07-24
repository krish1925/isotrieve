"""Typer CLI — calibrate / transform / inspect / plan / gate / doctor / report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from isotrieve import __version__
from isotrieve.calibration.corpus import (
    builtin_calibration_texts,
    corpus_checksum,
    sample_from_texts,
)
from isotrieve.calibration.planner import plan_calibration
from isotrieve.mapping.base import read_isotrieve_header
from isotrieve.mapping.linear import RidgeMapping
from isotrieve.stores.base import VectorRecord
from isotrieve.stores.numpy_files import NumpyFileStore

app = typer.Typer(
    name="isotrieve",
    help="Embedding migration without re-embedding.",
    no_args_is_help=True,
)
console = Console()

# Register additional commands
from isotrieve.cli_doctor import register_doctor_command
from isotrieve.cli_gate import register_gate_command
from isotrieve.cli_report import register_report_command

register_gate_command(app)
register_doctor_command(app)
register_report_command(app)


def _print_json(data: object) -> None:
    console.print_json(json.dumps(data, default=str))


@app.callback()
def main_callback() -> None:
    """Isotrieve — Embedding migration without re-embedding."""


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    console.print(__version__)


@app.command("plan")
def plan_cmd(
    source_model: str = typer.Option(..., "--source-model"),
    target_model: str = typer.Option(..., "--target-model"),
    corpus_size: int = typer.Option(..., "--corpus-size"),
    d_src: int = typer.Option(384, "--d-src"),
    d_tgt: int = typer.Option(1024, "--d-tgt"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Estimate calibration vs full re-embed cost."""
    plan = plan_calibration(
        corpus_size=corpus_size,
        source_model=source_model,
        target_model=target_model,
        d_src=d_src,
        d_tgt=d_tgt,
    )
    if as_json:
        _print_json(plan.__dict__)
        return
    table = Table(title="Calibration plan")
    table.add_column("Item")
    table.add_column("Value", justify="right")
    table.add_row("Corpus size", f"{plan.corpus_size:,}")
    table.add_row("Recommended K", f"{plan.recommended_k:,}")
    table.add_row("Calibration embed calls", f"{plan.est_calibration_calls:,}")
    table.add_row("Full re-embed calls", f"{plan.est_reembed_calls:,}")
    table.add_row("Est. calibration cost", f"${plan.est_calibration_usd:.2f}")
    table.add_row("Est. full re-embed cost", f"${plan.est_reembed_usd:.2f}")
    console.print(table)
    for note in plan.notes:
        console.print(f"  • {note}")
    console.print(
        "\nNext: [bold]isotrieve calibrate[/bold] with --sample-from-texts "
        "or local models, then run a quality gate before migrating."
    )


def _load_npy(path: Path, label: str) -> np.ndarray:
    """Load a .npy file with a clear error message on failure."""
    if not path.exists():
        console.print(f"[red]{label}: file not found: {path}[/red]")
        console.print(
            f"  Hint: provide a valid .npy file with shape (K, d). "
            f"Generate one with: np.save('{path}', embeddings)"
        )
        raise typer.Exit(1)
    try:
        return np.load(path)
    except Exception as exc:
        console.print(f"[red]{label}: failed to load {path}: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("calibrate")
def calibrate_cmd(
    source_vectors: Path | None = typer.Option(
        None, "--source-vectors", help="NPY of source embeddings (K, d_src)"
    ),
    target_vectors: Path | None = typer.Option(
        None, "--target-vectors", help="NPY of target embeddings (K, d_tgt)"
    ),
    source_model: str | None = typer.Option(
        None, "--source-model", help="sentence-transformers model id"
    ),
    target_model: str | None = typer.Option(
        None, "--target-model", help="sentence-transformers model id"
    ),
    texts_file: Path | None = typer.Option(
        None, "--texts", help="Text file (one text per line) for in-domain calib"
    ),
    queries_only: bool = typer.Option(
        False,
        "--queries-only",
        help="Calibrate from query log + stored vectors (no source docs)",
    ),
    queries_file: Path | None = typer.Option(
        None,
        "--queries",
        help="NPY of query embeddings in source space (for --queries-only)",
    ),
    k: int = typer.Option(2000, "--k"),
    seed: int = typer.Option(0, "--seed"),
    output: Path = typer.Option(..., "-o", "--output"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Fit a RidgeMapping and write a ``.isotrieve`` file.

    Provide either precomputed ``--source-vectors/--target-vectors`` NPYs,
    or ``--source-model/--target-model`` plus optional ``--texts``.
    Use ``--queries-only`` when source docs are unavailable.
    """
    # Queries-only mode: fit mapping from query pairs + stored vectors
    if queries_only:
        if queries_file is None or target_vectors is None:
            console.print(
                "[red]--queries-only requires --queries (NPY) and --target-vectors (NPY)[/red]"
            )
            raise typer.Exit(2)
        X = _load_npy(queries_file, "queries")
        Y = _load_npy(target_vectors, "target-vectors")
        src_id = source_model or "query-source"
        tgt_id = target_model or "query-target"
        texts = None  # Skip text-based calibration
    elif source_vectors is not None and target_vectors is not None:
        X = _load_npy(source_vectors, "source-vectors")
        Y = _load_npy(target_vectors, "target-vectors")
        src_id = source_model or "source"
        tgt_id = target_model or "target"
        texts = None  # Not needed for NPY-based calibration
    else:
        texts = None

    # Text-based calibration (only if not queries-only and not NPY-based)
    if texts is None and not queries_only and source_vectors is None:
        if texts_file is not None:
            raw = [
                ln.strip()
                for ln in texts_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            texts = sample_from_texts(raw, min(k, len(raw)), seed=seed)
        else:
            texts = builtin_calibration_texts(k, seed=seed)
            console.print(
                "[yellow]Using built-in demo texts (not isotrieve-calib-v1). "
                "Prefer --texts from your corpus.[/yellow]"
            )

    if source_model and target_model and texts is not None:
        from isotrieve.providers.factory import create_embedder

        src = create_embedder(source_model)
        tgt = create_embedder(target_model)
        console.print(f"Embedding {len(texts)} texts with {source_model} …")
        X = src.embed(texts)
        console.print(f"Embedding {len(texts)} texts with {target_model} …")
        Y = tgt.embed(texts)
        src_id, tgt_id = source_model, target_model

    if "X" not in dir() or "Y" not in dir():
        console.print(
            "[red]Provide --source-vectors/--target-vectors, "
            "--source-model/--target-model, or --queries-only[/red]"
        )
        raise typer.Exit(2)

    mapping = RidgeMapping(alpha="auto", seed=seed)
    try:
        mapping.fit(X, Y)
    except ValueError as exc:
        msg = str(exc)
        if "NaN" in msg or "Inf" in msg:
            console.print(
                f"[red]Calibration vectors contain NaN or Inf values.[/red]\n"
                f"  Check your source/target vector files for corrupt data."
            )
        elif "Need at least 2" in msg:
            console.print(f"[red]{msg}[/red]")
        elif "zero samples" in msg.lower():
            console.print(f"[red]{msg}[/red]")
        else:
            console.print(f"[red]Calibration failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Calibration failed: {exc}[/red]")
        raise typer.Exit(1) from exc

    # Fit score recalibrator from holdout calibration data
    from isotrieve.mapping.base import l2_normalize
    from isotrieve.recalibration import ScoreRecalibrator

    recal = ScoreRecalibrator()
    try:
        mapping.validation_report()
        # Use holdout indices from the mapping's internal split
        # Reconstruct: use the calibration data as both query and doc set
        # (K-pair similarity matrix covers the score range adequately)
        mapped_X = mapping.transform(X)
        mapped_n = l2_normalize(mapped_X)
        target_n = l2_normalize(Y)
        # K × K similarity matrix
        sim_mapped = mapped_n @ target_n.T
        sim_ceiling = target_n @ target_n.T
        # Sample pairs: all top-20 + random subset
        rng_cal = np.random.default_rng(seed)
        n = len(X)
        pairs_m, pairs_c = [], []
        # Top-20 neighbors per row
        for i in range(n):
            top = np.argsort(-sim_mapped[i])[:20]
            for j in top:
                pairs_m.append(float(sim_mapped[i, j]))
                pairs_c.append(float(sim_ceiling[i, j]))
        # Random pairs
        n_random = min(50_000, n * n)
        ri = rng_cal.integers(0, n, size=n_random)
        rj = rng_cal.integers(0, n, size=n_random)
        for i, j in zip(ri, rj):
            pairs_m.append(float(sim_mapped[i, j]))
            pairs_c.append(float(sim_ceiling[i, j]))
        recal.fit(np.array(pairs_m), np.array(pairs_c))
        mapping._recalibrator = recal
    except Exception:
        pass  # Skip recalibration if it fails; mapping still works

    meta: dict = {
        "source_model_id": src_id,
        "target_model_id": tgt_id,
        "calibration_k": int(X.shape[0]),
        "seed": seed,
    }
    if texts is not None:
        meta["calibration_checksum"] = corpus_checksum(texts)
        meta["calibration_corpus_id"] = "user-texts" if texts_file else "builtin-demo"
    mapping.set_meta(**meta)
    mapping.save(output)
    report = mapping.validation_report()

    if as_json:
        _print_json(
            {"output": str(output), "validation": report.to_dict(), "meta": meta}
        )
        return

    table = Table(title=f"Calibrated → {output}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Holdout cosine mean", f"{report.holdout_cosine_mean:.4f}")
    table.add_row("Holdout cosine p5", f"{report.holdout_cosine_p5:.4f}")
    table.add_row("Top-1 retention", f"{report.top1_retention:.4f}")
    table.add_row("Top-10 retention", f"{report.top10_retention:.4f}")
    table.add_row("Train / holdout", f"{report.n_train} / {report.n_holdout}")
    table.add_row("Alpha", f"{report.alpha}")
    if mapping.has_recalibrator:
        rr = mapping._recalibrator.report
        if rr:
            table.add_row("Score recalibrator", f"trained ({rr.n_pairs} pairs)")
            if rr.threshold_agreement:
                tau_8 = rr.threshold_agreement.get(0.8, 0)
                table.add_row("Threshold agreement @0.8", f"{tau_8:.1%}")
    console.print(table)
    console.print(
        "Cosine alone is misleading — prefer top-k retention. "
        "Next: transform a file store, or run a quality gate on a fresh sample."
    )


@app.command("transform")
def transform_cmd(
    mapping_path: Path = typer.Option(..., "--mapping"),
    source_dir: Path = typer.Option(..., "--source-dir", help="NumpyFileStore dir"),
    target_dir: Path = typer.Option(
        ..., "--target-dir", help="New output dir (never overwrite in place)"
    ),
    batch_size: int = typer.Option(2048, "--batch-size"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Stream-transform a NumpyFileStore into a new directory."""
    from isotrieve.mapping.registry import load_mapping

    if not mapping_path.exists():
        console.print(
            f"[red]Mapping file not found: {mapping_path}[/red]\n"
            f"  Run [bold]isotrieve calibrate[/bold] first to create a mapping."
        )
        raise typer.Exit(1)

    try:
        mapping = load_mapping(mapping_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Failed to load mapping: {exc}[/red]")
        raise typer.Exit(1) from exc

    if not source_dir.exists():
        console.print(
            f"[red]Source directory not found: {source_dir}[/red]\n"
            f"  Provide a directory containing vectors.npy and ids.npy."
        )
        raise typer.Exit(1)

    try:
        src = NumpyFileStore(source_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]Source store invalid: {exc}[/red]")
        raise typer.Exit(1) from exc

    dst = NumpyFileStore(target_dir, create=True)

    def gen():
        for batch in src.iter_vectors(batch_size=batch_size):
            vecs = np.stack([r.vector for r in batch], axis=0)
            try:
                mapped = mapping.transform(vecs)
            except ValueError as exc:
                msg = str(exc)
                if "Dimension" in msg or "dim" in msg.lower():
                    console.print(f"[red]Transform failed: {msg}[/red]")
                elif "NaN" in msg or "Inf" in msg:
                    console.print(
                        f"[red]Stored vectors contain NaN or Inf values.[/red]"
                    )
                else:
                    console.print(f"[red]Transform failed: {exc}[/red]")
                raise typer.Exit(1) from exc
            yield [
                VectorRecord(
                    id=batch[i].id,
                    vector=mapped[i],
                    text=batch[i].text,
                    payload=batch[i].payload,
                )
                for i in range(len(batch))
            ]

    try:
        n = dst.write_vectors(gen(), batch_size=batch_size)
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[red]Write failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    if as_json:
        _print_json({"written": n, "target_dir": str(target_dir)})
        return
    console.print(f"Wrote {n:,} vectors → {target_dir}")


@app.command("inspect")
def inspect_cmd(
    mapping_path: Path = typer.Argument(..., help="Path to .isotrieve file"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Pretty-print the header of a ``.isotrieve`` mapping file."""
    if not mapping_path.exists():
        console.print(
            f"[red]File not found: {mapping_path}[/red]\n"
            f"  Provide a valid .isotrieve mapping file.\n"
            f"  Create one with: [bold]isotrieve calibrate[/bold]"
        )
        raise typer.Exit(1)

    try:
        header = read_isotrieve_header(mapping_path)
    except ValueError as exc:
        console.print(f"[red]Invalid mapping file: {exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Failed to read mapping: {exc}[/red]")
        raise typer.Exit(1) from exc

    if as_json:
        _print_json(header)
        return
    console.print_json(json.dumps(header, indent=2, default=str))


if __name__ == "__main__":
    app()
