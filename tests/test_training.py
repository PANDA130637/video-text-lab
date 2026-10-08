"""Regressions for data leakage, metric ties and checkpoint reproducibility."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vtlab.metrics import retrieval_metrics
from vtlab.models import build_model
from vtlab.training import (evaluate, load_cache, load_model, multi_positive_loss,
                            train, validate_cache_compatibility)


def records_for(labels, split="test"):
    return [dict(id=f"{split}_{i}", group_id=f"{split}_group_{i // 2}",
                 pair_id=f"{split}_pair_{i // 2}", split=split,
                 video_path=f"{split}/video_{i}", text=f"caption {label}",
                 semantic_id=str(label)) for i, label in enumerate(labels)]


class MetricsTests(unittest.TestCase):
    def test_tied_retrieval_is_not_changed_by_candidate_order(self):
        records = records_for(["a", "b", "a", "b"])
        video = np.ones((4, 2), dtype=np.float32)
        text = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
        before = retrieval_metrics(video, text, records)
        order = [1, 0, 3, 2]
        after = retrieval_metrics(video[order], text[order], [records[i] for i in order])
        self.assertEqual(before["metrics"], after["metrics"])
        self.assertEqual(before["counts"]["text_candidates"], 2)
        self.assertAlmostEqual(before["metrics"]["mean_recall_at_1"], 0.5)
        self.assertAlmostEqual(before["metrics"]["order_query_accuracy"], 0.5)
        self.assertAlmostEqual(before["metrics"]["order_pair_accuracy"], 0.5)

    def test_recall_at_k_ties_are_fractional(self):
        labels = [str(i) for i in range(10)]
        embeddings = np.ones((10, 2), dtype=np.float32)
        metrics = retrieval_metrics(embeddings, embeddings, records_for(labels))["metrics"]
        self.assertAlmostEqual(metrics["video_to_text"]["recall_at_1"], 0.1)
        self.assertAlmostEqual(metrics["video_to_text"]["recall_at_5"], 0.5)
        self.assertAlmostEqual(metrics["video_to_text"]["recall_at_10"], 1.0)

    def test_order_success_failure_and_near_ties(self):
        records = records_for(["a", "b"])
        embeddings = np.eye(2, dtype=np.float32)
        good = retrieval_metrics(embeddings, embeddings, records)["metrics"]
        bad = retrieval_metrics(embeddings[::-1].copy(), embeddings, records)["metrics"]
        ties = retrieval_metrics(np.array([[1, 1 + 1e-7], [1, 1]], dtype=np.float32),
                                 embeddings, records)["metrics"]
        self.assertEqual(good["order_pair_accuracy"], 1.0)
        self.assertEqual(bad["order_query_accuracy"], 0.0)
        self.assertEqual(ties["order_pair_accuracy"], 0.5)

    def test_source_group_without_pair_annotation_has_no_order_metric(self):
        records = records_for(["a", "b"])
        for record in records:
            record.pop("pair_id")
        embeddings = np.eye(2, dtype=np.float32)
        result = retrieval_metrics(embeddings, embeddings, records)
        self.assertIsNone(result["metrics"]["order_query_accuracy"])
        self.assertIsNone(result["metrics"]["order_pair_accuracy"])
        self.assertEqual(result["counts"]["order_pairs"], 0)

    def test_multiple_correct_captions_are_positive(self):
        values = torch.eye(2).repeat_interleave(2, dim=0)
        labels = torch.tensor([0, 0, 1, 1])
        self.assertLess(float(multi_positive_loss(values, values, labels)), 1e-5)
        self.assertEqual(float(multi_positive_loss(values, values, torch.zeros(4))), 0.0)

    def test_mean_model_cannot_distinguish_frame_permutation(self):
        torch.manual_seed(13)
        model = build_model("mean", 4, 3)
        frames = torch.randn(2, 8, 4)
        torch.testing.assert_close(model.encode_video(frames), model.encode_video(frames.flip(1)))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(11)
        self.records = sum([records_for(["a", "b", "a", "b"], split)
                            for split in ("train", "val", "test")], [])
        self.video = rng.normal(size=(12, 4, 3)).astype(np.float32)
        self.text = np.tile(np.eye(2, dtype=np.float32), (6, 1))
        self.cache = self.root / "cache.npz"
        self.write_cache()

    def tearDown(self):
        self.temporary.cleanup()

    def write_cache(self):
        np.savez(self.cache, video_features=self.video, text_features=self.text,
                 metadata=np.array(json.dumps(self.records)),
                 provenance=np.array(json.dumps(dict(synthetic=True, encoder="fixture", frames=4))))

    def test_test_features_do_not_affect_checkpoint(self):
        report = train(self.cache, self.root / "first", epochs=2, batch_size=4, threads=1)
        first_model, first = load_model(Path(report["checkpoint"]))
        self.video[8:] = self.video[8:] * 100 + 99
        self.write_cache()
        second_report = train(self.cache, self.root / "second", epochs=2, batch_size=4, threads=1)
        second_model, second = load_model(Path(second_report["checkpoint"]))
        for key, value in first_model.state_dict().items():
            self.assertTrue(torch.equal(value, second_model.state_dict()[key]), key)
        self.assertEqual(first["val_metrics"], second["val_metrics"])
        self.assertEqual(first["selection_split"], "val")
        self.assertFalse(report["test_used_for_selection"])
        evaluated = evaluate(self.cache, Path(report["checkpoint"]))
        self.assertEqual(evaluated["counts"]["videos"], 4)
        self.assertTrue(evaluated["provenance"]["synthetic"])
        json.dumps(evaluated, allow_nan=False)

    def test_cache_rejects_cross_split_group(self):
        self.records[4]["group_id"] = self.records[0]["group_id"]
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_cache(self.cache)

    def test_cache_rejects_nonfinite_features(self):
        self.video[0, 0, 0] = np.nan
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_cache(self.cache)

    def test_compatibility_allows_new_data_not_new_encoder(self):
        cache = load_cache(self.cache)
        checkpoint = dict(config=dict(video_dim=3, text_dim=2),
                          provenance=dict(encoder="fixture", frames=4, fingerprint="old"))
        cache["provenance"]["fingerprint"] = "new"
        validate_cache_compatibility(cache, checkpoint)
        cache["provenance"]["encoder"] = "other"
        with self.assertRaisesRegex(ValueError, "encoder"):
            validate_cache_compatibility(cache, checkpoint)


if __name__ == "__main__":
    unittest.main()
