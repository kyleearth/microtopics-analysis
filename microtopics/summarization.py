"""Stage 3 — Cluster Insights (keywords + LLM summaries).

Reads a labelled parquet produced by Stage 2 (must contain ``cluster``
and a text column).  For every non-noise cluster it:

1. Extracts the most frequent terms via ``CountVectorizer``.
2. Sends a sample of posts to a vLLM-hosted language model and asks
   for a short title and description (returned as structured JSON).

The two result sets are merged into a single analysis JSON.  An optional
validation JSON containing the sampled posts can also be written.

Usage — YAML controls paths and LLM::

    python clustering_insights.py --config configs/insights_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, ValidationError
from sklearn.feature_extraction.text import CountVectorizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────
# When running without a YAML config, these values are used.

DEFAULTS: dict[str, Any] = {
    "text_column": "record.media.merge.clean",
    "n_keywords": 10,
    "sample_size": 100,
    "max_tokens": 256,
    "temperature": 0.3,
}


# ── Schema ───────────────────────────────────────────────────────────────────


class ClusterDescription(BaseModel):
    """Structured output expected from the LLM for each cluster."""

    title: str
    description: str


# ── Config helpers ───────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        A dictionary representing the parsed YAML contents.
    """
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_paths(paths: dict[str, str], config_path: Path) -> dict[str, str]:
    """Resolve relative paths against the config file's parent directory.

    Absolute paths are kept as-is; relative paths are resolved relative
    to the directory that contains *config_path*.

    Args:
        paths: Raw path strings from the YAML ``paths`` section.
        config_path: Path to the YAML file itself.

    Returns:
        A new dictionary with all paths resolved to absolute strings.
    """
    root = config_path.parent
    out: dict[str, str] = {}
    for key, val in paths.items():
        p = Path(val)
        out[key] = str(p if p.is_absolute() else (root / p).resolve())
    return out


# ── LLM construction ────────────────────────────────────────────────────────


def build_llm(llm_cfg: dict[str, Any]) -> LLM:
    """Instantiate a vLLM engine from a configuration block.

    Args:
        llm_cfg: Dictionary with at least a ``model`` key.  Optional keys
            include ``download_dir``, ``tokenizer_mode``, ``config_format``,
            and ``load_format``.

    Returns:
        A ready-to-use ``vllm.LLM`` instance.
    """
    kwargs: dict[str, Any] = {
        "model": llm_cfg["model"],
        "download_dir": llm_cfg.get("download_dir"),
        "tokenizer_mode": llm_cfg.get("tokenizer_mode", "auto"),
    }
    if kwargs["tokenizer_mode"] == "mistral":
        kwargs["config_format"] = llm_cfg.get("config_format", "mistral")
        kwargs["load_format"] = llm_cfg.get("load_format", "mistral")
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    log.info("LLM init kwargs: %s", kwargs)
    return LLM(**kwargs)


# ── Keyword extraction ───────────────────────────────────────────────────────


def keywords_per_cluster(
    df: pl.DataFrame,
    *,
    text_column: str = DEFAULTS["text_column"],
    n_keywords: int = DEFAULTS["n_keywords"],
) -> dict[int, dict[str, Any]]:
    """Extract the most frequent terms for each non-noise cluster.

    Groups posts by cluster label (skipping ``-1``), fits a
    ``CountVectorizer`` per group, and returns the top-*n_keywords*
    terms ranked by raw frequency.

    Args:
        df: Labelled dataframe with ``cluster`` and *text_column*.
        text_column: Name of the column containing post text.
        n_keywords: Number of top keywords to retain per cluster.

    Returns:
        Mapping of cluster ID to ``{"size": int, "keywords": [...]}``.
    """
    out: dict[int, dict[str, Any]] = {}

    grouped = (
        df.filter(pl.col("cluster") != -1)
        .group_by("cluster")
        .agg(pl.col(text_column))
        .sort("cluster")
    )

    for row in grouped.iter_rows(named=True):
        cid: int = row["cluster"]
        texts: list[str] = row[text_column]

        if len(texts) < 2:
            log.warning("cluster %s: <2 texts, skipping keywords", cid)
            continue

        try:
            vec = CountVectorizer(max_features=100, stop_words="english")
            mat = vec.fit_transform(texts)
            names = vec.get_feature_names_out()

            freqs = mat.sum(axis=0).A1
            top_idx = freqs.argsort()[-n_keywords:][::-1]

            out[int(cid)] = {
                "size": len(texts),
                "keywords": [
                    {"term": names[i], "score": float(freqs[i])} for i in top_idx
                ],
            }
        except ValueError as exc:
            log.warning("cluster %s: keyword extraction failed (%s)", cid, exc)
            out[int(cid)] = {"size": len(texts), "keywords": []}

    return out


# ── LLM summaries ────────────────────────────────────────────────────────────


def parse_description(raw: str) -> ClusterDescription | None:
    """Try to parse a JSON string into a ``ClusterDescription``.

    Args:
        raw: Raw LLM output text.

    Returns:
        A validated ``ClusterDescription``, or ``None`` on failure.
    """
    try:
        return ClusterDescription.model_validate_json(raw.strip())
    except (json.JSONDecodeError, ValidationError) as exc:
        log.debug("JSON parse: %s", exc)
        return None


def summarize_clusters(
    df: pl.DataFrame,
    llm: LLM,
    *,
    system_prompt: str,
    text_column: str = DEFAULTS["text_column"],
    sample_size: int = DEFAULTS["sample_size"],
    max_tokens: int = DEFAULTS["max_tokens"],
    temperature: float = DEFAULTS["temperature"],
) -> tuple[dict[int, dict[str, str]], dict[int, list[str]]]:
    """Generate LLM titles and descriptions for every non-noise cluster.

    A random sample of up to *sample_size* posts per cluster is sent to
    the model in a single batched ``llm.chat`` call.

    Args:
        df: Labelled dataframe with ``cluster`` and *text_column*.
        llm: An initialised vLLM engine.
        system_prompt: System-role prompt instructing the model.
        text_column: Name of the column containing post text.
        sample_size: Maximum posts per cluster to include in the prompt.
        max_tokens: Token budget for the LLM response.
        temperature: Sampling temperature.

    Returns:
        A tuple ``(summaries, sampled_posts)`` where *summaries* maps
        cluster IDs to ``{"title": ..., "description": ...}`` and
        *sampled_posts* maps cluster IDs to the text lists that were
        sent to the model.
    """
    df_sample = (
        df.filter(pl.col("cluster") != -1)
        .group_by("cluster")
        .agg(pl.col(text_column).shuffle(seed=42).head(sample_size))
        .sort("cluster")
    )

    n_clusters = len(df_sample)
    log.info("LLM: sampling up to %d posts x %d clusters", sample_size, n_clusters)

    cluster_ids: list[int] = df_sample["cluster"].to_list()
    posts_lists: list[list[str]] = df_sample[text_column].to_list()

    sampled_posts: dict[int, list[str]] = {
        int(cid): posts for cid, posts in zip(cluster_ids, posts_lists)
    }

    prompts = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Posts:\n\n" + "\n\n".join(posts)},
        ]
        for posts in posts_lists
    ]

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        structured_outputs=StructuredOutputsParams(
            json=ClusterDescription.model_json_schema(),
        ),
    )

    log.info("Generating summaries...")
    outputs = llm.chat(prompts, params, use_tqdm=True)

    summaries: dict[int, dict[str, str]] = {}
    fails = 0

    for cid, out in zip(cluster_ids, outputs):
        text = out.outputs[0].text.strip()
        parsed = parse_description(text)
        if parsed is None:
            fails += 1
            log.warning("cluster %s: parse failed", cid)
        else:
            summaries[int(cid)] = {
                "title": parsed.title,
                "description": parsed.description,
            }

    log.info("Parsed %d summaries, %d failures", len(summaries), fails)
    return summaries, sampled_posts


# ── Merging & output ─────────────────────────────────────────────────────────


def merge_results(
    top_keywords: dict[int, dict[str, Any]],
    llm_summaries: dict[int, dict[str, str]],
) -> dict[int, dict[str, Any]]:
    """Combine keyword and LLM results into a unified dictionary.

    Args:
        top_keywords: Output of :func:`keywords_per_cluster`.
        llm_summaries: Output of :func:`summarize_clusters` (first element).

    Returns:
        Merged dictionary keyed by cluster ID.
    """
    merged: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(top_keywords) | set(llm_summaries)):
        merged[cid] = {
            "cluster_id": cid,
            "size": top_keywords.get(cid, {}).get("size", 0),
            "top_keywords": top_keywords.get(cid, {}).get("keywords", []),
            "llm_summary": llm_summaries.get(cid, {}),
        }
    return merged


def write_json(
    payload: dict[int, dict[str, Any]],
    *,
    input_parquet: Path,
    output_dir: Path,
    prompt_file: Path,
) -> Path:
    """Write the merged analysis to a JSON file.

    The file is placed under ``<output_dir>/<prompt_stem>/``.

    Args:
        payload: Combined keyword + LLM results.
        input_parquet: Path to the source parquet (used to derive the
            output filename).
        output_dir: Top-level output directory.
        prompt_file: Path to the system-prompt file (its stem is used as
            a subdirectory tag).

    Returns:
        The path to the written JSON file.
    """
    tag = prompt_file.stem
    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cluster_analysis_{input_parquet.stem}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s", out_path)
    return out_path


def write_validation_json(
    sampled_posts: dict[int, list[str]],
    summaries: dict[int, dict[str, str]],
    *,
    input_parquet: Path,
    output_dir: Path,
    prompt_file: Path,
) -> Path:
    """Write sampled posts alongside summaries for manual validation.

    Args:
        sampled_posts: Cluster-ID-to-text-list mapping.
        summaries: LLM-generated titles and descriptions per cluster.
        input_parquet: Source parquet path (for filename derivation).
        output_dir: Top-level output directory.
        prompt_file: System-prompt file (stem used as subdirectory tag).

    Returns:
        The path to the written validation JSON file.
    """
    tag = prompt_file.stem
    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        str(cid): {
            **summaries.get(cid, {}),
            "sampled_posts": posts,
        }
        for cid, posts in sorted(sampled_posts.items())
    }
    out_path = out_dir / f"validation_posts_{input_parquet.stem}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Wrote %s", out_path)
    return out_path


def print_compact(results: dict[int, dict[str, Any]]) -> None:
    """Print a human-readable summary of cluster insights to stdout.

    Args:
        results: Merged results dictionary from :func:`merge_results`.
    """
    line = "\u2500" * 72
    print(f"\n{line}")
    print("CLUSTER INSIGHTS")
    print(line)
    for cid in sorted(results.keys()):
        row = results[cid]
        llm_out = row.get("llm_summary") or {}
        print(f"\nCluster {row['cluster_id']}  (n={row['size']})")
        if llm_out:
            print(f"  {llm_out.get('title', '')}")
            desc = llm_out.get("description", "")
            if desc:
                print(f"  {desc}")
        kws = row.get("top_keywords") or []
        if kws:
            top = ", ".join(f"{k['term']} ({k['score']:.3f})" for k in kws[:5])
            print(f"  keywords: {top}")
    print(f"{line}\n")


# ── Pipeline entry point ─────────────────────────────────────────────────────


def run(
    config_path: Path,
    *,
    sample_size_override: int | None = None,
    quiet: bool = False,
    validation: bool = False,
) -> dict[int, dict[str, Any]]:
    """Orchestrate keyword extraction and LLM summarisation.

    Reads a YAML config that specifies file paths, LLM settings, and
    runtime parameters.  CLI overrides (sample size) take precedence
    when provided.

    Args:
        config_path: Path to the YAML configuration file.
        sample_size_override: If given, replaces ``run.sample_size``.
        quiet: Suppress the stdout summary table.
        validation: Write a validation JSON with the sampled posts.

    Returns:
        The merged results dictionary.
    """
    raw = load_yaml(config_path)
    paths = resolve_paths(raw["paths"], config_path)
    llm_cfg: dict[str, Any] = raw["llm"]
    run_cfg: dict[str, Any] = raw["run"]

    text_column: str = run_cfg.get("text_column", DEFAULTS["text_column"])

    input_parquet = Path(paths["input_parquet"])

    sample_sz = (
        sample_size_override
        if sample_size_override is not None
        else int(run_cfg.get("sample_size", DEFAULTS["sample_size"]))
    )

    log.info("Reading %s", input_parquet)
    df = pl.read_parquet(input_parquet)
    n_clust = df["cluster"].n_unique()
    log.info("%s rows, %s unique cluster labels", f"{len(df):,}", n_clust)

    with open(paths["prompt_file"], encoding="utf-8") as fh:
        system_prompt = fh.read()

    log.info("Extracting top keywords...")
    top_keywords = keywords_per_cluster(df, text_column=text_column)

    llm = build_llm(llm_cfg)
    llm_out, sampled_posts = summarize_clusters(
        df,
        llm,
        system_prompt=system_prompt,
        text_column=text_column,
        sample_size=sample_sz,
        max_tokens=int(run_cfg.get("max_tokens", DEFAULTS["max_tokens"])),
        temperature=float(run_cfg.get("temperature", DEFAULTS["temperature"])),
    )

    combined = merge_results(top_keywords, llm_out)

    out_dir = Path(paths["output_dir"])
    prompt_path = Path(paths["prompt_file"])
    write_json(
        combined,
        input_parquet=input_parquet,
        output_dir=out_dir,
        prompt_file=prompt_path,
    )

    if validation:
        write_validation_json(
            sampled_posts,
            llm_out,
            input_parquet=input_parquet,
            output_dir=out_dir,
            prompt_file=prompt_path,
        )

    if not quiet:
        print_compact(combined)

    return combined


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    p = argparse.ArgumentParser(
        description="Stage 3: Cluster keywords + LLM summaries.",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="YAML with paths, llm, and run sections.",
    )
    p.add_argument(
        "--sample-size", type=int, default=None,
        help="Posts per cluster for LLM (default: run.sample_size or %(default)s).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Skip the stdout summary table.",
    )
    p.add_argument(
        "--validation", action="store_true",
        help="Save sampled posts per cluster as a separate validation JSON.",
    )
    return p


def main() -> None:
    """Parse CLI arguments and run the insights pipeline."""
    args = build_parser().parse_args()
    run(
        args.config,
        sample_size_override=args.sample_size,
        quiet=args.quiet,
        validation=args.validation,
    )


if __name__ == "__main__":
    main()
