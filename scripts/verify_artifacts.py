#!/usr/bin/env python3
"""Validate the generated portfolio artifacts and their data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ARTIFACT_NAMES = {
    "segments.csv",
    "segment_profiles.json",
    "visualization.html",
    "report.md",
}


def verify(data_dir: Path, results_dir: Path) -> dict[str, int]:
    missing = sorted(name for name in ARTIFACT_NAMES if not (results_dir / name).is_file())
    if missing:
        raise RuntimeError(f"missing result artifacts: {', '.join(missing)}")

    prospects = pd.read_csv(data_dir / "prospects.csv")
    deals = pd.read_csv(data_dir / "deal_history.csv")
    segments = pd.read_csv(results_dir / "segments.csv")
    profiles = json.loads((results_dir / "segment_profiles.json").read_text(encoding="utf-8"))

    if len(prospects) != 500 or len(segments) != 500:
        raise RuntimeError("expected 500 prospect and segment rows")
    if len(deals) != 200:
        raise RuntimeError("expected 200 deal-history rows")

    required_columns = {
        "deal_count",
        "mean_deal_value",
        "win_rate",
        "mean_sales_cycle_days",
        "cluster_id",
        "segment_name",
        "projection_x",
        "projection_y",
    }
    missing_columns = sorted(required_columns - set(segments.columns))
    if missing_columns:
        raise RuntimeError(f"segments.csv is missing columns: {', '.join(missing_columns)}")
    if int(segments["deal_count"].sum()) != len(deals):
        raise RuntimeError("deal-history rows were not fully represented in deal_count")

    kmeans_profiles = profiles.get("kmeans", [])
    if not kmeans_profiles or any(not profile.get("segment_name") for profile in kmeans_profiles):
        raise RuntimeError("K-Means profiles must contain non-empty segment names")

    html = (results_dir / "visualization.html").read_text(encoding="utf-8")
    if 'id="audience-segments"' not in html:
        raise RuntimeError("visualization.html lacks the deterministic Plotly element id")

    return {
        "prospects": len(prospects),
        "deals": len(deals),
        "segments": len(segments),
        "kmeans_profiles": len(kmeans_profiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
