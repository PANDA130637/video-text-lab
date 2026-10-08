import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from vtlab.data import extract_features, make_synthetic, validate_manifest
from vtlab.encoders import CAPTIONS, encode_text_queries


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = make_synthetic(self.root / "data", groups=12, frames=4)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite_manifest(self, records):
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def test_group_split_leakage_rejected(self):
        records = validate_manifest(self.manifest)
        records[1]["split"] = "test" if records[0]["split"] != "test" else "train"
        self.rewrite_manifest(records)
        with self.assertRaisesRegex(ValueError, "Group split leakage"):
            validate_manifest(self.manifest)

    def test_duplicate_ids_rejected(self):
        records = validate_manifest(self.manifest)
        records[1]["id"] = records[0]["id"]
        self.rewrite_manifest(records)
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            validate_manifest(self.manifest)

    def test_all_classes_in_all_splits(self):
        records = validate_manifest(self.manifest)
        for split in ("train", "val", "test"):
            self.assertEqual({r["text"] for r in records if r["split"] == split}, set(CAPTIONS))

    def test_reverse_pairs_have_identical_means_and_cache_invalidates(self):
        cache = self.root / "features.npz"
        first = extract_features(self.manifest, cache, frames=4)
        self.assertFalse(first["reused"])
        with np.load(cache, allow_pickle=False) as archive:
            video = archive["video_features"]
            self.assertEqual(video.dtype, np.float32)
            self.assertEqual(video.shape, (24, 4, 6))
            self.assertEqual(archive["text_features"].shape, (24, 8))
            for index in range(0, len(video), 2):
                np.testing.assert_array_equal(video[index], video[index + 1, ::-1])
                np.testing.assert_array_equal(video[index].mean(0), video[index + 1].mean(0))
            self.assertEqual(json.loads(str(archive["provenance"].item()))["source"], "synthetic_control")
        second = extract_features(self.manifest, cache, frames=4)
        self.assertTrue(second["reused"])
        record = validate_manifest(self.manifest)[0]
        changed_frame = self.manifest.parent / record["video_path"] / "frame_0000.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(changed_frame)
        third = extract_features(self.manifest, cache, frames=4)
        self.assertFalse(third["reused"])
        self.assertNotEqual(first["fingerprint"], third["fingerprint"])

    def test_color_rejects_real_data_and_unknown_queries(self):
        records = validate_manifest(self.manifest)
        records[0]["source"] = "real"
        self.rewrite_manifest(records)
        with self.assertRaisesRegex(ValueError, "restricted"):
            extract_features(self.manifest, self.root / "features.npz")
        with self.assertRaisesRegex(ValueError, "only accepts"):
            encode_text_queries(["a person drinks water"], encoder="color")

    def test_synthetic_does_not_overwrite_existing_directory(self):
        occupied = self.root / "occupied"
        occupied.mkdir()
        (occupied / "important.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            make_synthetic(occupied)
        self.assertEqual((occupied / "important.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
