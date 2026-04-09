# Microtopics Analysis

A data-driven pipeline for discovering fine-grained **microtopics** in large
text corpora.  It embeds text into dense vectors, clusters the embedding space
on the GPU, and generates human-readable labels for every discovered topic.

The pipeline is designed for HPC environments (SLURM) and scales to tens of
millions of documents by sharding and leveraging NVIDIA RAPIDS for
GPU-accelerated dimensionality reduction and clustering.

---

## Pipeline Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Stage 1     │     │  Stage 2      │     │  Stage 3       │
│  Embed       │────▶│  Cluster      │────▶│  Summarize     │
│              │     │               │     │                │
│ Sentence-    │     │ UMAP (cuML)   │     │ CountVectorizer│
│ Transformer  │     │ + HDBSCAN     │     │ + vLLM (LLM)   │
│ → parquet    │     │ → labels JSON │     │ → analysis JSON │
└─────────────┘     └──────────────┘     └───────────────┘
```

| Stage | Input | Output | Script |
|-------|-------|--------|--------|
| **1 — Embed** | Raw text parquet | Sharded parquets with `embedding` column | `microtopics/embedding.py` |
| **2 — Cluster** | Embedding shards | Metrics JSON + (optional) labelled parquet | `microtopics/clustering.py` |
| **3 — Summarize** | Labelled parquet | Cluster analysis JSON (keywords + LLM titles) | `microtopics/summarization.py` |

---

## Repository Structure

```
microtopics-analysis/
├── README.md
├── LICENSE                              # Apache 2.0
├── .gitignore
│
├── microtopics/                         # Core Python package
│   ├── __init__.py
│   ├── embedding.py                     # Stage 1 — text vectorisation
│   ├── clustering.py                    # Stage 2 — UMAP + HDBSCAN
│   └── summarization.py                # Stage 3 — keywords + LLM summaries
│
├── notebooks/                           # Jupyter diagnostics & visualisation
│   └── clustering_diagnostics.ipynb
│
├── configs/                             # Runtime configuration files
├── slurm_scripts/                       # SLURM worker job templates
├── logs/                                # SLURM stdout / stderr logs
└── data/                                # Raw and processed datasets
```

> **Note:** `configs/`, `slurm_scripts/`, `logs/`, and `data/` are kept as
> empty placeholder directories in the repository.  Their contents are
> gitignored to prevent user data, credentials, or cluster-specific paths
> from being committed.

---

## Script Registry

### Core Pipeline

| Script | Description |
|--------|-------------|
| `microtopics/embedding.py` | Loads a text parquet, encodes rows with a SentenceTransformer model (default: `Qwen/Qwen3-Embedding-8B`), and writes sharded parquets with a dense `embedding` column. Supports configurable batch size, target dimensionality, and shard size. |
| `microtopics/clustering.py` | Reads embedding shards, runs GPU-accelerated UMAP (cuML) for dimensionality reduction, then HDBSCAN for density-based clustering. In **grid-search mode** each `config_id` deterministically samples a unique hyperparameter combination. In **final-run mode** (`--final-run`) it exports a labelled parquet for Stage 3. Computes silhouette score, UMAP trustworthiness, and noise statistics. |
| `microtopics/summarization.py` | Reads a labelled parquet from Stage 2. Extracts top-*k* keywords per cluster via `CountVectorizer`, then batches a sample of posts per cluster to a vLLM-hosted language model for structured JSON summaries (title + description). Merges both result sets into a single analysis JSON. Optionally writes a validation JSON with the sampled posts. |

### Diagnostics & Optimisation

| Artefact | Description |
|----------|-------------|
| `notebooks/clustering_diagnostics.ipynb` | Dataset-agnostic Jupyter notebook that loads grid-search metrics JSONs and produces: correlation heatmaps, pairwise metric scatter plots, marginal-effect box plots, HDBSCAN parameter interaction heatmaps, cluster-count vs. silhouette bubble charts, and a ranked table of best configurations. Change a single `OUTPUT_DIR` variable to analyse any dataset. |

---

## Diagnostics & Hyperparameter Tuning

The clustering stage supports automatic hyperparameter exploration via SLURM
array jobs.  Each array task receives a different `config_id`, which seeds a
deterministic RNG that samples one value per grid axis:

| Parameter | Default Candidates |
|-----------|--------------------|
| `embedding_dims` | 128, 256 |
| `umap_components` | 10, 15 |
| `umap_neighbors` | 10, 15, 20 |
| `umap_min_dist` | 0.05, 0.1 |
| `min_cluster_size` | 200, 300, 400, 500 |
| `min_samples` | 25, 50, 100, 150 |
| `cluster_selection_epsilon` | 0.0, 0.05 |

After a sweep completes, open the diagnostics notebook:

```bash
jupyter notebook notebooks/clustering_diagnostics.ipynb
```

Set `OUTPUT_DIR` to the directory containing the `config_*.json` files and
re-run all cells.  The notebook highlights optimal parameter ranges and
trade-offs between cluster count, silhouette score, and noise ratio.

---

## Usage

### Stage 1 — Embedding

```bash
python -m microtopics.embedding \
    data/posts.parquet \
    output/embeddings/ \
    0 \
    --text-column "clean_merged_text" \
    --batch-size 64 \
    --target-dim 256
```

The `shard` argument (here `0`) selects which slice of the input to process.
Run multiple shards in parallel via SLURM array jobs for large datasets.

### Stage 2 — Clustering (grid search)

```bash
python -m microtopics.clustering \
    output/embeddings/ \
    output/clusters/ \
    0
```

Each `config_id` explores a different hyperparameter combination.  Use
`--test` to subsample to 5 M rows for faster iteration.

### Stage 2 — Clustering (final run)

```bash
python -m microtopics.clustering \
    output/embeddings/ \
    output/clusters_final/ \
    0 \
    --final-run
```

Writes a labelled parquet (`config_XXXX.parquet`) alongside the metrics JSON.

### Stage 3 — Summarisation

```bash
python -m microtopics.summarization \
    --config configs/clustering_insights.yaml \
    --validation
```

Reads the labelled parquet specified in the YAML config, extracts keywords,
queries the LLM, and writes the merged analysis JSON.  Pass `--validation`
to also save the sampled posts for manual review.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
