"""Checks for the real-data-facing entry points, without network downloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vtlab.cli import dispatch, make_parser
from vtlab.data import load_video_frames


class EntryPointTests(unittest.TestCase):
    def test_frozen_clip_baseline_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "features.npz"
            records = [dict(id=str(i), group_id=f"scene{i}", split="test",
                            video_path=f"video{i}.mp4", text=f"caption{i}", semantic_id=str(i))
                       for i in range(3)]
            vectors = np.eye(3, dtype=np.float32)
            np.savez(cache, video_features=np.repeat(vectors[:, None, :], 4, axis=1),
                     text_features=vectors,
                     metadata=np.array(json.dumps(records)),
                     provenance=np.array(json.dumps({"encoder": "clip"})))
            args = make_parser().parse_args(["baseline", "--cache", str(cache)])
            result = dispatch(args)
            self.assertEqual(result["metrics"]["video_to_text"]["recall_at_1"], 1.0)
            self.assertEqual(result["metrics"]["text_to_video"]["recall_at_1"], 1.0)
            self.assertFalse(result["trained"])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Optional OpenCV is not installed")
    def test_video_file_decoding_preserves_sampling_order(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 64))
            if not writer.isOpened():
                self.skipTest("This OpenCV build cannot write MJPG AVI")
            try:
                for value in (20, 60, 120, 200):
                    writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            frames = load_video_frames(path, frames=2)
            self.assertEqual(len(frames), 2)
            means = [np.asarray(frame).mean() for frame in frames]
            self.assertLess(means[0], 30)
            self.assertGreater(means[1], 190)


if __name__ == "__main__":
    unittest.main()
