from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import SEED, _generate_deals, _generate_prospects
from pipeline.segmentation import generate_cluster_profiles, run_dbscan, run_kmeans
from pipeline.text_processor import TextFeatureProcessor
from segment import _merge_deal_history, run_pipeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AudienceSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prospects = _generate_prospects(90, seed=SEED)
        cls.frame = pd.DataFrame(cls.prospects)

    def test_generators_are_seeded_and_validate_inputs(self) -> None:
        prospects_again = _generate_prospects(90, seed=SEED)
        deals = _generate_deals(self.prospects, n=40, seed=SEED + 1)
        deals_again = _generate_deals(prospects_again, n=40, seed=SEED + 1)

        self.assertEqual(self.prospects, prospects_again)
        self.assertEqual(deals, deals_again)
        self.assertEqual(len({row["company_name"] for row in self.prospects}), 90)
        with self.assertRaises(ValueError):
            _generate_prospects(0)
        with self.assertRaises(ValueError):
            _generate_deals([], n=1)

    def test_deal_history_becomes_model_features(self) -> None:
        deals = pd.DataFrame(_generate_deals(self.prospects, n=60, seed=SEED + 1))
        enriched = _merge_deal_history(self.frame, deals)

        self.assertEqual(len(enriched), 90)
        self.assertEqual(int(enriched["deal_count"].sum()), 60)
        self.assertTrue(enriched["win_rate"].between(0, 1).all())
        self.assertGreater(int((enriched["deal_count"] > 0).sum()), 0)

    def test_nlp_features_cluster_into_named_actionable_segments(self) -> None:
        processor = TextFeatureProcessor(
            n_text_components=3,
            max_tfidf_features=100,
            use_umap=False,
            random_state=42,
        )
        features = processor.fit_transform(self.frame)
        labels, _ = run_kmeans(features, n_clusters=5, random_state=42)
        profiles = generate_cluster_profiles(self.frame, labels)

        self.assertEqual(features.shape, (90, 14))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(labels), 90)
        self.assertEqual(len(profiles), 5)
        self.assertTrue(all(profile["segment_name"] for profile in profiles))
        self.assertTrue(all(profile["outreach_strategy"] for profile in profiles))

    def test_pipeline_writes_repeatable_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prospects_path = root / "prospects.csv"
            deals_path = root / "deal_history.csv"
            self.frame.to_csv(prospects_path, index=False)
            pd.DataFrame(_generate_deals(self.prospects, n=60, seed=SEED + 1)).to_csv(
                deals_path, index=False
            )

            artifact_names = {
                "segments.csv",
                "segment_profiles.json",
                "visualization.html",
                "report.md",
            }
            output_a = root / "results-a"
            output_b = root / "results-b"
            for output in (output_a, output_b):
                run_pipeline(
                    input_path=str(prospects_path),
                    deals_path=str(deals_path),
                    output_dir=str(output),
                    n_clusters="5",
                    use_umap=False,
                    run_dbscan_comparison=True,
                )

            hashes_a = {name: _sha256(output_a / name) for name in artifact_names}
            hashes_b = {name: _sha256(output_b / name) for name in artifact_names}
            self.assertEqual(hashes_a, hashes_b)

            segments = pd.read_csv(output_a / "segments.csv")
            self.assertEqual(len(segments), 90)
            self.assertTrue(
                {
                    "deal_count",
                    "mean_deal_value",
                    "win_rate",
                    "mean_sales_cycle_days",
                    "cluster_id",
                    "segment_name",
                    "projection_x",
                    "projection_y",
                }.issubset(segments.columns)
            )

    def test_dbscan_returns_one_label_per_prospect(self) -> None:
        points = np.vstack(
            [
                np.random.default_rng(42).normal(0, 0.05, (12, 2)),
                np.random.default_rng(43).normal(2, 0.05, (12, 2)),
            ]
        )
        labels = run_dbscan(points, eps=0.25, min_samples=3)
        self.assertEqual(len(labels), 24)
        self.assertGreaterEqual(len(set(labels) - {-1}), 2)


if __name__ == "__main__":
    unittest.main()
