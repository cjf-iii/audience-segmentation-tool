from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from generate_data import SEED, _generate_deals, _generate_prospects
from pipeline.segmentation import generate_cluster_profiles, run_dbscan, run_kmeans
from pipeline.text_processor import TextFeatureProcessor


class AudienceSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        random.seed(SEED)
        cls.prospects = _generate_prospects(90)
        cls.frame = pd.DataFrame(cls.prospects)

    def test_generators_produce_requested_record_counts(self) -> None:
        deals = _generate_deals(self.prospects, n=40)
        self.assertEqual(len(self.prospects), 90)
        self.assertEqual(len(deals), 40)
        self.assertEqual(len({row["company_name"] for row in self.prospects}), 90)

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

        self.assertEqual(features.shape, (90, 10))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(labels), 90)
        self.assertEqual(len(profiles), 5)
        self.assertTrue(all(profile["segment_name"] for profile in profiles))
        self.assertTrue(all(profile["outreach_strategy"] for profile in profiles))

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
