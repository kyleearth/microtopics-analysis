"""Stage 2 — Dimensionality Reduction and Clustering.

Loads embedding shards produced by Stage 1, applies UMAP dimensionality
reduction on the GPU, and clusters the reduced space with HDBSCAN.
Hyperparameters are drawn deterministically from a configurable grid so
that different ``config_id`` values explore distinct parameter
combinations — ideal for SLURM array jobs.

Requires a RAPIDS-enabled environment (cuML, CuPy).

Usage — grid search::

    python clustering.py embeddings/ output/clusters/ 0

Final run (writes labelled parquet)::

    python clustering.py embeddings/ output/clusters/ 0 --final-run

With a YAML config::

    python clustering.py embeddings/ output/clusters/ 0 \\
        --config configs/cluster_config.yaml --final-run
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────
# These grids are used when no external YAML config is supplied.  Each key
# maps to a list of candidate values; ``select_params`` picks one per key
# using a seeded RNG keyed on the config_id.

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "embedding_dims": [128, 256],
    "umap_components": [10, 15],
    "umap_neighbors": [10, 15, 20],
    "umap_min_dist": [0.05, 0.1],
    "min_cluster_size": [200, 300, 400, 500],
    "min_samples": [25, 50, 100, 150],
    "cluster_selection_epsilon": [0.0, 0.05],
}

DEFAULT_FINAL_PARAMS: dict[str, list[Any]] = {
    "embedding_dims": [128],
    "umap_components": [15],
    "umap_neighbors": [20],
    "umap_min_dist": [0.05, 0.1],
    "min_cluster_size": [300, 500],
    "min_samples": [50, 100],
    "cluster_selection_epsilon": [0],
}

DEFAULTS: dict[str, Any] = {
    "text_column": "record.media.merge.clean",
    "post_id_column": "cid",
    "embedding_column": "embedding",
    "metrics_subsample": 30_000,
}


# ── GPU dependency loading ───────────────────────────────────────────────────


def _import_gpu_deps() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import RAPIDS / CuPy libraries.

    Raises:
        RuntimeError: If the required GPU packages are not installed.

    Returns:
        A tuple of ``(cupy, HDBSCAN, UMAP, trustworthiness, silhouette_score)``.
    """
    try:
        import cupy as cp
        from cuml.cluster import HDBSCAN
        from cuml.manifold import UMAP
        from cuml.metrics import trustworthiness as cuml_trustworthiness
        from cuml.metrics.cluster import silhouette_score
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies are missing. Install RAPIDS + CuPy in your "
            "environment before running clustering."
        ) from exc
    return cp, HDBSCAN, UMAP, cuml_trustworthiness, silhouette_score


# ── Parameter selection ──────────────────────────────────────────────────────


def select_params(config_id: int, grid: dict[str, list[Any]]) -> dict[str, Any]:
    """Deterministically pick one value per grid key.

    Uses a ``random.Random`` instance seeded with *config_id* so that the
    same ID always produces identical hyperparameters.

    Args:
        config_id: Seed value (typically the SLURM array task ID).
        grid: Mapping of parameter names to candidate value lists.

    Returns:
        A flat dictionary of selected hyperparameters.
    """
    rng = random.Random(config_id)
    return {key: rng.choice(values) for key, values in grid.items()}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_embeddings(pattern: str) -> pl.DataFrame:
    """Read all parquet shards matching a glob *pattern*.

    Args:
        pattern: Glob expression (e.g. ``"embeddings/*.parquet"``).

    Returns:
        A concatenated polars DataFrame.
    """
    log.info("Loading embedding shards from %s", pattern)
    df = pl.read_parquet(pattern)
    log.info("Loaded %s rows", f"{len(df):,}")
    return df


def to_gpu_array(df: pl.DataFrame, dims: int, cp: Any) -> Any:
    """Move the ``embedding`` column to a CuPy array on the GPU.

    Args:
        df: DataFrame that must contain an ``embedding`` column.
        dims: Number of leading dimensions to keep.
        cp: The ``cupy`` module (passed to avoid a top-level import).

    Returns:
        A CuPy 2-D float32 array of shape ``(n_rows, dims)``.
    """
    arr = df["embedding"].to_numpy()
    if arr.dtype == object:
        arr = np.stack(arr)
    arr = arr.astype(np.float32)[:, :dims]
    return cp.asarray(arr)


# ── UMAP + HDBSCAN ──────────────────────────────────────────────────────────


def run_umap(
    embeddings: Any,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    UMAP: Any,
) -> Any:
    """Fit UMAP and return the reduced embedding matrix.

    Args:
        embeddings: GPU array of shape ``(n, d)``.
        n_components: Target dimensionality.
        n_neighbors: UMAP neighbourhood size.
        min_dist: Minimum distance parameter for UMAP.
        UMAP: The ``cuml.manifold.UMAP`` class.

    Returns:
        Reduced GPU array of shape ``(n, n_components)``.
    """
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        init="random",
        build_algo="nn_descent",
        build_kwds={"knn_n_clusters": 4},
        verbose=6,
    )
    reduced = reducer.fit_transform(embeddings)
    log.info("UMAP done -> %s", reduced.shape)
    del reducer
    return reduced


def run_hdbscan(
    embeddings: Any,
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    cp: Any,
    HDBSCAN: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HDBSCAN on a GPU embedding array.

    Args:
        embeddings: Reduced GPU array from UMAP.
        min_cluster_size: Smallest group size considered a cluster.
        min_samples: Core-distance smoothing parameter.
        cluster_selection_epsilon: Merge threshold for the cluster tree.
        cp: The ``cupy`` module.
        HDBSCAN: The ``cuml.cluster.HDBSCAN`` class.

    Returns:
        A ``(labels, probabilities)`` tuple of NumPy arrays.
    """
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        verbose=6,
    )
    hdb.fit(embeddings)

    labels = cp.asnumpy(hdb.labels_) if isinstance(hdb.labels_, cp.ndarray) else np.asarray(hdb.labels_)
    probs = (
        cp.asnumpy(hdb.probabilities_)
        if isinstance(hdb.probabilities_, cp.ndarray)
        else np.asarray(hdb.probabilities_)
    )
    return labels, probs


# ── Evaluation ───────────────────────────────────────────────────────────────


def compute_metrics(
    embeddings_original: Any,
    embeddings_reduced: Any,
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    subsample: int,
    cp: Any,
    cuml_trustworthiness: Any,
    silhouette_score: Any,
) -> dict[str, Any] | None:
    """Compute clustering quality metrics on a subsample.

    Calculates silhouette score, UMAP trustworthiness, and average
    cluster membership probability.

    Args:
        embeddings_original: Full-dimensional GPU embeddings.
        embeddings_reduced: UMAP-reduced GPU embeddings.
        labels: HDBSCAN cluster labels (``-1`` = noise).
        probs: HDBSCAN membership probabilities.
        subsample: Maximum rows to evaluate (for speed).
        cp: The ``cupy`` module.
        cuml_trustworthiness: The ``cuml.metrics.trustworthiness`` function.
        silhouette_score: The ``cuml.metrics.cluster.silhouette_score`` function.

    Returns:
        A metrics dictionary, or ``None`` if fewer than two clusters
        were found.
    """
    n_total = len(labels)
    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) < 2:
        return None

    idx = np.random.default_rng(42).choice(n_total, size=min(subsample, n_total), replace=False)
    sub_original = embeddings_original[idx]
    sub_reduced = embeddings_reduced[idx]
    sub_labels = labels[idx]

    sub_mask = sub_labels != -1
    if sub_mask.sum() < 2 or len(np.unique(sub_labels[sub_mask])) < 2:
        return None

    sil = float(silhouette_score(sub_reduced[sub_mask], cp.asarray(sub_labels[sub_mask])))
    trust = float(cuml_trustworthiness(sub_original, sub_reduced))

    full_mask = labels != -1
    avg_prob = float(probs[full_mask].mean()) if full_mask.any() else 0.0

    return {
        "n_clusters": int(len(unique_labels)),
        "n_noise": int((labels == -1).sum()),
        "noise_pct": round(float((labels == -1).sum()) / n_total * 100, 4),
        "silhouette": round(sil, 4),
        "trustworthiness": round(trust, 4),
        "avg_prob": round(avg_prob, 4),
    }


# ── Results I/O ──────────────────────────────────────────────────────────────


def save_metrics(result: dict[str, Any], output_path: Path) -> None:
    """Write a metrics dictionary to a JSON file.

    Args:
        result: Flat dictionary of parameters and metric values.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log.info("Wrote metrics -> %s", output_path)


def save_labelled_parquet(
    df: pl.DataFrame,
    labels: np.ndarray,
    *,
    post_id_column: str,
    text_column: str,
    output_path: Path,
) -> None:
    """Write a parquet containing post IDs, text, and cluster labels.

    Args:
        df: Source dataframe (must contain *post_id_column* and *text_column*).
        labels: HDBSCAN cluster label array aligned with *df*.
        post_id_column: Name of the unique-identifier column.
        text_column: Name of the text column to carry forward.
        output_path: Destination ``.parquet`` file path.
    """
    missing = [name for name in [post_id_column, text_column] if name not in df.columns]
    if missing:
        raise ValueError(f"Cannot write labels; missing columns: {missing}")

    df_out = df.select(post_id_column, text_column).with_columns(
        pl.Series("cluster", labels),
    )
    df_out.write_parquet(output_path)
    log.info("Wrote labelled parquet -> %s", output_path)


# ── Config loading ───────────────────────────────────────────────────────────


def load_config(config_path: str | None) -> dict[str, Any]:
    """Load a YAML config or fall back to built-in defaults.

    The YAML is expected to have top-level keys ``params``,
    ``final_params``, ``metrics_subsample``, and optionally ``columns``.

    Args:
        config_path: Path to a YAML file, or ``None`` to use defaults.

    Returns:
        A configuration dictionary.
    """
    if config_path is None:
        return {
            "metrics_subsample": DEFAULTS["metrics_subsample"],
            "params": DEFAULT_PARAM_GRID,
            "final_params": DEFAULT_FINAL_PARAMS,
            "columns": {"embedding": DEFAULTS["embedding_column"]},
        }
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Pipeline entry point ─────────────────────────────────────────────────────


def run_single(
    *,
    input_dir: str,
    output_dir: str,
    config_id: int,
    config: dict[str, Any],
    text_column: str = DEFAULTS["text_column"],
    post_id_column: str = DEFAULTS["post_id_column"],
    final_run: bool = False,
    test: bool = False,
) -> None:
    """Execute one UMAP + HDBSCAN configuration.

    Loads embedding shards from *input_dir*, selects hyperparameters
    deterministically from the grid using *config_id*, runs
    dimensionality reduction and clustering, evaluates quality metrics,
    and writes results to *output_dir*.

    When *final_run* is ``True`` the ``final_params`` grid is used and a
    labelled parquet (with a ``cluster`` column) is saved alongside the
    metrics JSON so that Stage 3 can consume it directly.

    Args:
        input_dir: Directory containing embedding parquet shards.
        output_dir: Directory for clustering output files.
        config_id: Deterministic seed for parameter selection.
        config: Configuration dictionary (see :func:`load_config`).
        text_column: Name of the text column in the shards.
        post_id_column: Name of the unique-ID column in the shards.
        final_run: If ``True``, use ``final_params`` and write a
            labelled parquet.
        test: If ``True``, subsample to 5 M rows for faster iteration.
    """
    cp, HDBSCAN, UMAP, cuml_trustworthiness, silhouette_score = _import_gpu_deps()

    metrics_subsample = int(config.get("metrics_subsample", DEFAULTS["metrics_subsample"]))
    embedding_col: str = config.get("columns", {}).get("embedding", DEFAULTS["embedding_column"])

    grid = DEFAULT_FINAL_PARAMS if final_run else config.get("params")
    if not grid:
        raise ValueError("Parameter grid is empty; check config `params` / `final_params`.")
    params = select_params(config_id, grid)

    out_dir = Path(output_dir)
    out_path = out_dir / f"config_{config_id:04d}.json"

    if out_path.exists():
        log.info("config %d already complete, skipping (%s)", config_id, out_path)
        return

    # ── Load & clean embeddings ──────────────────────────────────────────
    pattern = str(Path(input_dir) / "*.parquet")
    df = load_embeddings(pattern)
    if embedding_col not in df.columns:
        raise ValueError(f"Embedding column '{embedding_col}' not found in shards: {pattern}")

    n_before = len(df)
    df = df.filter(~pl.col(embedding_col).arr.contains(float("nan")))
    if dropped := n_before - len(df):
        log.info("Dropped %s rows with NaN embeddings", f"{dropped:,}")

    if test:
        df = df.sample(n=min(5_000_000, len(df)), seed=42)
        log.info("--test enabled, sampled down to %s rows", f"{len(df):,}")

    df = df.rename({embedding_col: "embedding"})
    embeddings_gpu = to_gpu_array(df, int(params["embedding_dims"]), cp)
    log.info("Embedding matrix shape: %s", embeddings_gpu.shape)

    # ── UMAP ─────────────────────────────────────────────────────────────
    reduced = run_umap(
        embeddings_gpu,
        n_components=int(params["umap_components"]),
        n_neighbors=int(params["umap_neighbors"]),
        min_dist=float(params["umap_min_dist"]),
        UMAP=UMAP,
    )

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ── HDBSCAN ──────────────────────────────────────────────────────────
    labels, probs = run_hdbscan(
        reduced,
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
        cp=cp,
        HDBSCAN=HDBSCAN,
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(
        embeddings_gpu,
        reduced,
        labels,
        probs,
        subsample=metrics_subsample,
        cp=cp,
        cuml_trustworthiness=cuml_trustworthiness,
        silhouette_score=silhouette_score,
    )

    if metrics is None:
        metrics = {
            "n_clusters": 0,
            "n_noise": int((labels == -1).sum()),
            "noise_pct": 100.0,
            "silhouette": None,
            "trustworthiness": None,
            "avg_prob": None,
        }

    result = {"config_id": config_id, **params, **metrics}
    save_metrics(result, out_path)

    if final_run:
        pq_path = out_dir / f"config_{config_id:04d}.parquet"
        save_labelled_parquet(
            df, labels,
            post_id_column=post_id_column,
            text_column=text_column,
            output_path=pq_path,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    p = argparse.ArgumentParser(
        description="Stage 2: Run one UMAP + HDBSCAN config sampled from a grid.",
    )
    p.add_argument("input_dir", help="Directory containing embedding parquet shards.")
    p.add_argument("output_dir", help="Directory for clustering outputs.")
    p.add_argument("config_id", type=int, help="Config seed for parameter selection.")
    p.add_argument(
        "--text-column",
        default=DEFAULTS["text_column"],
        help="Name of the text column (default: %(default)s).",
    )
    p.add_argument(
        "--post-id-column",
        default=DEFAULTS["post_id_column"],
        help="Name of the post / row ID column (default: %(default)s).",
    )
    p.add_argument(
        "--config", default=None,
        help="YAML config path.  If omitted, built-in defaults are used.",
    )
    p.add_argument(
        "--final-run", action="store_true",
        help="Use final_params and write labelled parquets.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Subsample to 5 M rows for faster iteration.",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run the clustering pipeline."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_single(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_id=args.config_id,
        config=config,
        text_column=args.text_column,
        post_id_column=args.post_id_column,
        final_run=args.final_run,
        test=args.test,
    )


if __name__ == "__main__":
    main()
