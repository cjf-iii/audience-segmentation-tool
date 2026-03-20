#!/usr/bin/env python3
"""
CLI Entry Point — Audience Segmentation Tool.

Usage:
    python segment.py --input data/prospects.csv --clusters auto --output results/

This script orchestrates the full pipeline:
  1. Load prospect data from CSV.
  2. Run the NLP feature extraction pipeline (TF-IDF + UMAP/PCA + scaling).
  3. Determine optimal k (or use user-specified k).
  4. Run K-Means clustering and generate segment profiles.
  5. Optionally run DBSCAN for comparison.
  6. Write outputs:
       - segments.csv          (original data + cluster assignments)
       - segment_profiles.json (profile summaries per cluster)
       - visualization.html    (interactive Plotly scatter plot)
       - report.md             (markdown report with segment descriptions)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `pipeline` is importable
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.segmentation import (
    find_optimal_k,
    generate_cluster_profiles,
    run_dbscan,
    run_kmeans,
)
from pipeline.text_processor import TextFeatureProcessor

# ---------------------------------------------------------------------------
# Logging configuration — INFO to console, DEBUG available via --verbose
# ---------------------------------------------------------------------------
logger = logging.getLogger("segment")


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------


def _write_segments_csv(
    df: pd.DataFrame,
    labels: np.ndarray,
    profiles: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Write the original prospect data augmented with cluster ID and
    segment name to a CSV file.
    """
    out = df.copy()
    out["cluster_id"] = labels

    # Map cluster_id -> segment_name for readability
    name_map = {p["cluster_id"]: p["segment_name"] for p in profiles}
    out["segment_name"] = out["cluster_id"].map(name_map)

    path = output_dir / "segments.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(out))
    return path


def _write_profiles_json(
    profiles: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write cluster profiles to a JSON file."""
    path = output_dir / "segment_profiles.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, default=str)
    logger.info("Wrote %s (%d profiles)", path, len(profiles))
    return path


def _write_visualization(
    df: pd.DataFrame,
    labels: np.ndarray,
    profiles: list[dict[str, Any]],
    projection_2d: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Create an interactive Plotly scatter plot of the 2-D UMAP/PCA
    projection, coloured by cluster assignment, and save as standalone
    HTML.

    Each point shows the company name, industry, and segment name on
    hover — making the visualisation immediately useful for exploration.
    """
    import plotly.express as px

    # Build a DataFrame for Plotly
    name_map = {p["cluster_id"]: p["segment_name"] for p in profiles}
    viz_df = pd.DataFrame(
        {
            "x": projection_2d[:, 0],
            "y": projection_2d[:, 1],
            "Cluster": [name_map.get(int(l), f"Cluster {l}") for l in labels],
            "Company": df["company_name"].values,
            "Industry": df["industry"].values,
            "Revenue": df["annual_revenue"].values,
            "Ad Spend": df["current_ad_spend_estimate"].values,
            "Social Score": df["social_presence_score"].values,
        }
    )

    fig = px.scatter(
        viz_df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["Company", "Industry", "Revenue", "Ad Spend", "Social Score"],
        title="Audience Segments — 2-D Projection (UMAP / PCA)",
        labels={"x": "Component 1", "y": "Component 2"},
        template="plotly_white",
        width=1100,
        height=700,
    )

    # Improve marker styling for readability
    fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="white")))
    fig.update_layout(
        legend_title_text="Segment",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        title_font_size=18,
    )

    path = output_dir / "visualization.html"
    fig.write_html(str(path), include_plotlyjs=True)
    logger.info("Wrote %s", path)
    return path


def _write_report(
    profiles: list[dict[str, Any]],
    n_total: int,
    optimal_k: int,
    output_dir: Path,
) -> Path:
    """
    Generate a Markdown report summarising each segment with
    statistics, themes, and recommended outreach strategies.
    """
    lines: list[str] = []
    lines.append("# Audience Segmentation Report\n")
    lines.append(f"**Total prospects analysed:** {n_total}  ")
    lines.append(f"**Number of segments:** {optimal_k}  ")
    lines.append(f"**Algorithm:** K-Means (elbow method)\n")
    lines.append("---\n")

    for p in profiles:
        lines.append(f"## Segment {p['cluster_id']}: {p['segment_name']}\n")
        lines.append(f"**Size:** {p['size']} companies ({p['percentage']}% of total)\n")

        lines.append("### Key Metrics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Median Revenue | ${p['median_revenue']:,.0f} |")
        lines.append(f"| Mean Revenue | ${p['mean_revenue']:,.0f} |")
        lines.append(f"| Median Employees | {p['median_employees']:,.0f} |")
        lines.append(f"| Median Social Score | {p['median_social_score']:.0f} / 100 |")
        lines.append(f"| Median Ad Spend | ${p['median_ad_spend']:,.0f} |")
        lines.append(f"| Mean Ad Spend | ${p['mean_ad_spend']:,.0f} |")
        lines.append("")

        lines.append("### Industry Mix\n")
        for industry, pct in p["industry_distribution"].items():
            lines.append(f"- **{industry}**: {pct}%")
        lines.append("")

        if p["campaign_themes"]:
            lines.append("### Campaign Themes\n")
            for theme in p["campaign_themes"]:
                lines.append(f"- {theme}")
            lines.append("")

        if p["top_demographics"]:
            lines.append("### Target Demographics\n")
            for demo in p["top_demographics"]:
                lines.append(f"- {demo}")
            lines.append("")

        lines.append("### Recommended Outreach Strategy\n")
        lines.append(f"{p['outreach_strategy']}\n")
        lines.append("---\n")

    # Footer
    lines.append(
        "*Generated by the Audience Segmentation Tool — "
        "a portfolio project by CJ Fleming demonstrating NLP, "
        "unsupervised learning, and production Python.*\n"
    )

    path = output_dir / "report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_dir: str,
    n_clusters: str = "auto",
    use_umap: bool = True,
    run_dbscan_comparison: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Execute the full segmentation pipeline end-to-end.

    This function is the single orchestration point used by both the CLI
    and the Streamlit app, ensuring consistent behaviour regardless of
    the interface.

    Parameters
    ----------
    input_path : str
        Path to the prospects CSV file.
    output_dir : str
        Directory where output files will be written.
    n_clusters : str
        Number of clusters: "auto" for elbow method, or an integer string.
    use_umap : bool
        Whether to use UMAP (True) or PCA (False) for dimensionality
        reduction.
    run_dbscan_comparison : bool
        Whether to also run DBSCAN and include its results in the
        profiles JSON.
    verbose : bool
        Enable debug-level logging.

    Returns
    -------
    dict with keys:
        "df"         — the original DataFrame
        "labels"     — K-Means cluster labels
        "profiles"   — list of cluster profile dicts
        "projection" — 2-D projection array
        "optimal_k"  — selected number of clusters
        "output_dir" — Path to the output directory
    """
    _setup_logging(verbose)

    # --- 1. Load data ----------------------------------------------------
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)

    df = pd.read_csv(input_file)
    logger.info("Loaded %d prospects from %s", len(df), input_file)

    # --- 2. Feature extraction -------------------------------------------
    processor = TextFeatureProcessor(
        n_text_components=10,
        max_tfidf_features=500,
        use_umap=use_umap,
        random_state=42,
    )
    features = processor.fit_transform(df)
    logger.info("Feature matrix: %s", features.shape)

    # --- 3. Determine number of clusters ---------------------------------
    if n_clusters == "auto":
        optimal_k, inertias = find_optimal_k(features)
        logger.info("Auto-selected k=%d via elbow method", optimal_k)
    else:
        optimal_k = int(n_clusters)
        logger.info("User-specified k=%d", optimal_k)

    # --- 4. K-Means clustering -------------------------------------------
    labels, km_model = run_kmeans(features, n_clusters=optimal_k)

    # --- 5. Generate profiles --------------------------------------------
    profiles = generate_cluster_profiles(df, labels, algorithm_name="kmeans")

    # --- 6. DBSCAN comparison (optional) ---------------------------------
    dbscan_profiles: list[dict[str, Any]] = []
    if run_dbscan_comparison:
        dbscan_labels = run_dbscan(features, eps=2.0, min_samples=5)
        dbscan_profiles = generate_cluster_profiles(
            df, dbscan_labels, algorithm_name="dbscan"
        )

    # --- 7. 2-D projection for visualisation -----------------------------
    projection_2d = processor.get_2d_projection(df)

    # --- 8. Write outputs ------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _write_segments_csv(df, labels, profiles, out_path)
    _write_visualization(df, labels, profiles, projection_2d, out_path)
    _write_report(profiles, len(df), optimal_k, out_path)

    # Combine K-Means and DBSCAN profiles into one JSON for comparison
    all_profiles = {
        "kmeans": profiles,
    }
    if dbscan_profiles:
        all_profiles["dbscan"] = dbscan_profiles

    profiles_path = out_path / "segment_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, default=str)
    logger.info("Wrote %s", profiles_path)

    logger.info("Pipeline complete. Results in %s/", out_path)

    return {
        "df": df,
        "labels": labels,
        "profiles": profiles,
        "dbscan_profiles": dbscan_profiles,
        "projection": projection_2d,
        "optimal_k": optimal_k,
        "output_dir": out_path,
    }


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="segment",
        description=(
            "Audience Segmentation Tool — segment advertising prospects "
            "using NLP and unsupervised learning."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/prospects.csv",
        help="Path to the prospects CSV file (default: data/prospects.csv)",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default="auto",
        help=(
            "Number of clusters: 'auto' to use the elbow method, or an "
            "integer (default: auto)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Use PCA instead of UMAP for dimensionality reduction",
    )
    parser.add_argument(
        "--no-dbscan",
        action="store_true",
        help="Skip DBSCAN comparison clustering",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point — parse args and run the pipeline."""
    args = _parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        n_clusters=args.clusters,
        use_umap=not args.no_umap,
        run_dbscan_comparison=not args.no_dbscan,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
