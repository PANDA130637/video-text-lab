"""Checks for the transparent quality metrics and perturbation controls."""
import unittest

import numpy as np

from vtlab.quality import evaluate_quality, perturbation_diagnostics


def _records(n=4):
    return [{"id": f"v{i}", "group_id": f"g{i}", "split": "test",
             "semantic_id": "action", "text": "action", "video_path": f"v{i}.avi"}
            for i in range(n)]


class QualityTests(unittest.TestCase):
    def test_perfect_alignment_and_smoothness_score_high(self):
        videos = np.zeros((4, 4, 3), dtype=np.float32)
        videos[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        texts = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1))
        result = evaluate_quality(videos, texts, _records())
        self.assertAlmostEqual(result["summary"]["alignment_mean"], 1.0)
        self.assertAlmostEqual(result["summary"]["temporal_adjacent_cosine"], 1.0)
        self.assertGreater(result["summary"]["quality_score_mean"], 99.0)
        self.assertEqual(result["summary"]["quality_score_ci95"], [100.0, 100.0])
        self.assertIn("action", result["by_semantic_id"])

    def test_flicker_and_shuffle_controls_are_reproducible(self):
        rng = np.random.default_rng(7)
        videos = rng.normal(size=(4, 6, 8)).astype(np.float32)
        texts = videos.mean(axis=1)
        records = _records()
        first = perturbation_diagnostics(videos, texts, records, seed=11)
        second = perturbation_diagnostics(videos, texts, records, seed=11)
        self.assertEqual(first, second)
        self.assertIn("shuffled_frames", first)
        self.assertIn("cross_video_flicker", first)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_quality(np.ones((1, 2, 3)), np.ones((1, 3)), _records(1), flicker_threshold=0)
        with self.assertRaises(ValueError):
            evaluate_quality(np.ones((2, 2, 3)), np.ones((2, 3)), _records(2), bootstrap_samples=99)


if __name__ == "__main__":
    unittest.main()
