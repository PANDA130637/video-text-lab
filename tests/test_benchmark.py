"""Research controls: model capacity, invariance, and subject separation."""
import unittest
from unittest.mock import MagicMock, patch

import torch

from vtlab.kth import record_for, SUBJECT_SPLITS, ACTIONS, HTTPRangeFile
from vtlab.models import build_model


class BenchmarkTests(unittest.TestCase):
    def test_capacity_control_remains_order_invariant(self):
        model = build_model("mlp", 512, 512)
        gru = build_model("gru", 512, 512)
        counts = [sum(p.numel() for p in m.parameters()) for m in (model, gru)]
        self.assertLess(abs(counts[0] - counts[1]) / counts[1], 0.01)
        frames = torch.randn(2, 8, 512)
        torch.testing.assert_close(model.encode_video(frames), model.encode_video(frames.flip(1)))

    def test_official_subject_split_and_no_invented_order_pairs(self):
        rows = [record_for(f"person{person:02d}_{action}_d1_uncomp.avi", action)
                for person in range(1, 26) for action in ACTIONS]
        self.assertEqual(len(rows), 75)
        for name, subjects in SUBJECT_SPLITS.items():
            self.assertEqual(sum(r["split"] == name for r in rows), len(subjects) * 3)
        groups = {s: {r["group_id"] for r in rows if r["split"] == s} for s in SUBJECT_SPLITS}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertTrue(all(r["pair_id"] == "" for r in rows))

    def test_range_reader_refuses_full_response(self):
        session = MagicMock()
        session.head.return_value.headers = {"Content-Length": "1000000"}
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        response.headers = {"Content-Length": "1000000"}
        with patch("vtlab.kth.requests.Session", return_value=session):
            with HTTPRangeFile("https://example.invalid/archive.zip") as remote:
                with self.assertRaisesRegex(RuntimeError, "honor byte range"):
                    remote.read(64)


if __name__ == "__main__":
    unittest.main()
