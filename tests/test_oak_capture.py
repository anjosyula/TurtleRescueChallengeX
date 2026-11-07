import unittest
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from collect_oak import STREAMS, metric_depth, timestamp_span_ms, write_json
from validate_oak import validate


class CaptureContractTests(unittest.TestCase):
    def test_depth_units_and_unknowns(self):
        raw = np.array([[0, 1, 1000, 65535]], dtype=np.uint16)
        depth, valid = metric_depth(raw)
        np.testing.assert_allclose(depth, [[0, 0.001, 1, 65.535]], rtol=1e-6)
        np.testing.assert_array_equal(valid, [[False, True, True, True]])
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())

    def test_wrong_units_or_nonfinite_input_are_not_silently_accepted(self):
        for raw in (np.array([[float("nan")]], dtype=np.float32),
                    np.array([[1.5]], dtype=np.float32), np.zeros((2, 2, 3), np.uint16)):
            with self.assertRaises(ValueError):
                metric_depth(raw)

    def test_synchronization_uses_entire_group(self):
        self.assertAlmostEqual(timestamp_span_ms({"rgb": 10, "left": 10.01, "depth": 9.995}), 15)
        self.assertGreater(timestamp_span_ms({"rgb": 10, "depth": 10.1}), 20)

    def test_nonfinite_timestamps_are_rejected(self):
        for timestamps in ({}, {"rgb": float("nan")}, {"rgb": float("inf")}):
            with self.assertRaises(ValueError):
                timestamp_span_ms(timestamps)


class DatasetValidationTests(unittest.TestCase):
    """Small synthetic files exercise corruption handling, not real camera accuracy."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_json(self.root / "session.json", {"session_id": "fixture", "settings": {"sync_ms": 20}})
        write_json(self.root / "calibration.json", {"synthetic_fixture": True})
        write_json(self.root / "summary.json", {"samples": 2, "status": "complete"})
        self.records = []
        for i in range(2):
            sid = f"{i:06d}"
            folder = self.root / "samples" / sid
            folder.mkdir(parents=True)
            paths = {}
            for name in STREAMS:
                if name == "depth":
                    continue
                shape = (3, 4, 3) if name == "rgb" else (3, 4)
                path = folder / f"{name}.png"
                cv2.imwrite(str(path), np.zeros(shape, np.uint8))
                paths[name] = str(path.relative_to(self.root))
            depth = np.ones((3, 4), np.float32)
            depth[0, 0] = 0
            np.save(folder / "depth_m.npy", depth)
            cv2.imwrite(str(folder / "validity.png"), (depth > 0).astype(np.uint8) * 255)
            paths.update(depth=f"samples/{sid}/depth_m.npy", validity=f"samples/{sid}/validity.png")
            self.records.append({
                "sample_id": sid, "session_id": "fixture", "depth_units": "meters",
                "depth_kind": "camera_axis_z", "depth_alignment": "rgb", "sync_span_ms": 0,
                "streams": {n: {"device_monotonic_s": float(i + 1), "geometry": None} for n in STREAMS},
                "files": paths, "depth_valid_fraction": 11 / 12,
            })
        self.commit()

    def commit(self):
        for record in self.records:
            write_json(self.root / "samples" / record["sample_id"] / "sample.json", record)
        (self.root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_valid_fixture(self):
        self.assertEqual(validate(self.root)["samples"], 2)

    def test_alignment_must_match_session(self):
        self.records[0]["depth_alignment"] = "rectified_left"
        self.commit()
        with self.assertRaisesRegex(ValueError, "alignment differs"):
            validate(self.root)

    def test_annotation_coordinates_and_status(self):
        folder = self.root / "annotations/rgb"
        folder.mkdir(parents=True)
        label = {"session_id": "fixture", "sample_id": "000000", "view": "rgb",
                 "image": self.records[0]["files"]["rgb"], "size_wh": [4, 3],
                 "class": "turtle", "class_id": 0, "boxes_xyxy": [[0, 0, 4, 3]],
                 "annotation_status": "reviewed_turtle"}
        write_json(folder / "000000.json", label)
        self.assertEqual(validate(self.root)["validation"], "passed")
        label["boxes_xyxy"] = [[0, 0, 5, 3]]
        write_json(folder / "000000.json", label)
        with self.assertRaisesRegex(ValueError, "outside image"):
            validate(self.root)
        label["boxes_xyxy"] = []
        write_json(folder / "000000.json", label)
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            validate(self.root)

    def test_nonfinite_depth_is_rejected(self):
        np.save(self.root / "samples/000000/depth_m.npy", np.full((3, 4), np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "bad depth values"):
            validate(self.root)

    def test_false_valid_depth_is_rejected(self):
        cv2.imwrite(str(self.root / "samples/000000/validity.png"), np.full((3, 4), 255, np.uint8))
        with self.assertRaisesRegex(ValueError, "validity disagrees"):
            validate(self.root)

    def test_stale_modality_is_rejected(self):
        self.records[1]["streams"]["mono_left"]["device_monotonic_s"] = 1.0
        self.commit()
        with self.assertRaisesRegex(ValueError, "out of sync"):
            validate(self.root)

    def test_repeated_group_is_rejected(self):
        for stream in self.records[1]["streams"].values():
            stream["device_monotonic_s"] = 1.0
        self.commit()
        with self.assertRaisesRegex(ValueError, "stale"):
            validate(self.root)

    def test_partial_write_is_rejected(self):
        (self.root / "samples/.000002.partial").mkdir()
        with self.assertRaisesRegex(ValueError, "Uncommitted"):
            validate(self.root)

    def test_missing_image_is_rejected(self):
        (self.root / "samples/000000/rgb.png").unlink()
        with self.assertRaisesRegex(ValueError, "unreadable rgb"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
