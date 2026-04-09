"""Stage 1 — Embedding Generation.

Loads text data from a parquet file and produces dense vector embeddings
using a sentence-transformer model.  Supports sharding so large datasets
can be processed incrementally across multiple SLURM array jobs.

Usage::

    python embedding.py data/posts.parquet output/embeddings/ 0

Override defaults::

    python embedding.py data/posts.parquet output/embeddings/ 0 \\
        --model-name intfloat/e5-large-v2 \\
        --batch-size 128 --target-dim 128
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────
# Adapt these when switching to a new dataset, model, or compute environment.

DEFAULTS: dict[str, Any] = {
    "text_column": "clean_merged_text",
    "batch_size": 64,
    "shard_size": 1_000_000,
    "target_dim": 256,
    "model_name": "Qwen/Qwen3-Embedding-8B",
    "cache_dir": "/m/cs/scratch/ecanet/bluesky_datapool/vllm_cache/",
    "device": "cuda",
    "dtype": "float16",
    "max_seq_len": 512,
}


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(
    model_name: str,
    *,
    cache_dir: str | None = None,
    device: str = "cuda",
    dtype: str = "float16",
    max_seq_len: int = 512,
) -> SentenceTransformer:
    """Instantiate and configure a SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier or local path.
        cache_dir: Directory used to cache downloaded model weights.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        dtype: Model weight dtype forwarded via ``model_kwargs``.
        max_seq_len: Maximum token sequence length the model will use.

    Returns:
        A ready-to-use SentenceTransformer instance.
    """
    model = SentenceTransformer(
        model_name_or_path=model_name,
        cache_folder=cache_dir,
        device=device,
        model_kwargs={"dtype": dtype},
    )
    model.max_seq_length = max_seq_len
    return model


# ── Data I/O ─────────────────────────────────────────────────────────────────


def load_shard(
    input_parquet: str | Path,
    shard: int,
    shard_size: int,
) -> pl.DataFrame | None:
    """Read a single shard slice from a parquet file.

    Args:
        input_parquet: Path to the source parquet file.
        shard: Zero-based shard index.
        shard_size: Number of rows per shard.

    Returns:
        A polars DataFrame for the requested shard, or ``None`` if the
        shard index exceeds the dataset.
    """
    total_rows: int = pl.scan_parquet(input_parquet).select(pl.len()).collect().item()
    n_shards = (total_rows + shard_size - 1) // shard_size

    if shard >= n_shards:
        log.info("[shard %d] out of range (n_shards=%d), skipping", shard, n_shards)
        return None

    offset = shard * shard_size
    length = min(shard_size, total_rows - offset)
    log.info(
        "[shard %d/%d] reading rows %s..%s",
        shard, n_shards, f"{offset:,}", f"{offset + length - 1:,}",
    )
    return pl.scan_parquet(input_parquet).slice(offset, length).collect()


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int = 64,
    target_dim: int | None = None,
) -> np.ndarray:
    """Encode a list of strings into dense embeddings.

    Args:
        model: A loaded SentenceTransformer model.
        texts: Raw text strings to embed.
        batch_size: Mini-batch size for the encoder.
        target_dim: If set, truncate embeddings to this many dimensions.

    Returns:
        A 2-D numpy array of shape ``(len(texts), dim)``.
    """
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    if target_dim is not None:
        embeddings = embeddings[:, :target_dim]
    return embeddings


def save_shard(df: pl.DataFrame, embeddings: np.ndarray, output_path: Path) -> None:
    """Attach an embedding column and write the dataframe to parquet.

    Args:
        df: Source dataframe (one row per text).
        embeddings: 2-D float array aligned row-wise with *df*.
        output_path: Destination ``.parquet`` file path.
    """
    dim = embeddings.shape[1]
    df = df.with_columns(
        pl.Series("embedding", embeddings).cast(pl.Array(pl.Float32, dim)),
    )
    df.write_parquet(str(output_path))
    log.info("Wrote %s -> %s", df.shape, output_path)


# ── Pipeline entry point ─────────────────────────────────────────────────────


def run_embedding(
    input_parquet: str | Path,
    output_dir: str | Path,
    shard: int,
    *,
    text_column: str = DEFAULTS["text_column"],
    batch_size: int = DEFAULTS["batch_size"],
    shard_size: int = DEFAULTS["shard_size"],
    target_dim: int | None = DEFAULTS["target_dim"],
    model_name: str = DEFAULTS["model_name"],
    cache_dir: str | None = DEFAULTS["cache_dir"],
    device: str = DEFAULTS["device"],
    dtype: str = DEFAULTS["dtype"],
    max_seq_len: int = DEFAULTS["max_seq_len"],
) -> None:
    """Run the full embedding pipeline for a single shard.

    Loads a slice of the input parquet, encodes the text column, and
    writes the result (original columns + ``embedding``) back to parquet.
    Shards that already exist on disk are silently skipped, making reruns
    safe and idempotent.

    Args:
        input_parquet: Path to the source parquet file.
        output_dir: Directory where shard parquets are written.
        shard: Zero-based shard index.
        text_column: Name of the column containing raw text.
        batch_size: Mini-batch size for the encoder.
        shard_size: Number of rows per shard.
        target_dim: Truncate embeddings to this dimensionality.  ``None``
            keeps the original model dimension.
        model_name: HuggingFace model identifier.
        cache_dir: Model weight cache directory.
        device: Torch device string.
        dtype: Model weight dtype.
        max_seq_len: Maximum token sequence length.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"shard_{shard:04d}.parquet"
    if out_path.exists():
        log.info("[shard %d] already exists, skipping", shard)
        return

    df = load_shard(input_parquet, shard, shard_size)
    if df is None:
        return

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {input_parquet}.")

    model = load_model(
        model_name,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )

    texts: list[str] = df[text_column].to_list()
    embeddings = encode_texts(model, texts, batch_size=batch_size, target_dim=target_dim)
    save_shard(df, embeddings, out_path)
    log.info("[shard %d] complete", shard)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    p = argparse.ArgumentParser(
        description="Stage 1: Generate embedding shards from a text parquet.",
    )
    p.add_argument("input", help="Path to the input parquet file.")
    p.add_argument("output_dir", help="Directory for embedding shard outputs.")
    p.add_argument("shard", type=int, help="Zero-based shard index to process.")
    p.add_argument(
        "--text-column",
        default=DEFAULTS["text_column"],
        help="Name of the text column (default: %(default)s).",
    )
    p.add_argument(
        "--batch-size", type=int,
        default=DEFAULTS["batch_size"],
        help="Encoder batch size (default: %(default)s).",
    )
    p.add_argument(
        "--shard-size", type=int,
        default=DEFAULTS["shard_size"],
        help="Rows per shard (default: %(default)s).",
    )
    p.add_argument(
        "--target-dim", type=int,
        default=DEFAULTS["target_dim"],
        help="Truncation dimensionality (default: %(default)s).",
    )
    p.add_argument(
        "--model-name",
        default=DEFAULTS["model_name"],
        help="HuggingFace model ID (default: %(default)s).",
    )
    p.add_argument(
        "--cache-dir",
        default=DEFAULTS["cache_dir"],
        help="Model cache directory (default: %(default)s).",
    )
    p.add_argument(
        "--device",
        default=DEFAULTS["device"],
        help="Torch device (default: %(default)s).",
    )
    p.add_argument(
        "--dtype",
        default=DEFAULTS["dtype"],
        help="Model weight dtype (default: %(default)s).",
    )
    p.add_argument(
        "--max-seq-len", type=int,
        default=DEFAULTS["max_seq_len"],
        help="Max token sequence length (default: %(default)s).",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run the embedding pipeline."""
    args = build_parser().parse_args()
    run_embedding(
        args.input,
        args.output_dir,
        args.shard,
        text_column=args.text_column,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        target_dim=args.target_dim,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
