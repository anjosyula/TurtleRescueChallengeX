#!/usr/bin/env python3
"""Validate a TurtleHero OAK capture offline, without connecting a camera."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from collect_oak import STREAMS, timestamp_span_ms


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(session):
    session = Path(session).resolve()
    config = json.loads((session / "session.json").read_text())
    calibration = json.loads((session / "calibration.json").read_text())
    check(bool(calibration), "Empty calibration")
    records = [json.loads(line) for line in (session / "manifest.jsonl").read_text().splitlines() if line]
    check(bool(records), "No samples in manifest")
    seen = set()
    last_timestamps = {}
    valid_fractions, spans = [], []
    for sample in records:
        sid = sample["sample_id"]
        check(isinstance(sid, str) and sid.isascii() and sid.isdigit(), "Invalid sample ID")
        check(sid not in seen, f"Duplicate sample {sid}")
        seen.add(sid)
        check(sample["session_id"] == config["session_id"], f"{sid}: wrong session ID")
        check(sample["depth_units"] == "meters", f"{sid}: wrong depth units")
        check(sample["depth_kind"] == "camera_axis_z", f"{sid}: wrong depth convention")
        check(set(sample["streams"]) == set(STREAMS), f"{sid}: missing streams")
        timestamps = {n: m["device_monotonic_s"] for n, m in sample["streams"].items()}
        span = timestamp_span_ms(timestamps)
        check(span <= config["settings"]["sync_ms"] + 0.001, f"{sid}: timestamps out of sync")
        check(abs(span - sample["sync_span_ms"]) < 0.001, f"{sid}: inconsistent sync metadata")
        for stream, timestamp in timestamps.items():
            check(timestamp > last_timestamps.get(stream, -float("inf")),
                  f"{sid}: repeated or stale {stream} timestamp")
        last_timestamps = timestamps
        paths = {n: (session / p).resolve() for n, p in sample["files"].items()}
        check(set(paths) == set(STREAMS) | {"validity"}, f"{sid}: missing files")
        check(all(p.is_relative_to(session) for p in paths.values()), f"{sid}: path escapes session")
        depth = np.load(paths["depth"], allow_pickle=False)
        validity = cv2.imread(str(paths["validity"]), cv2.IMREAD_UNCHANGED)
        check(depth.dtype == np.float32 and depth.ndim == 2, f"{sid}: wrong depth dtype/shape")
        check(np.isfinite(depth).all() and (depth >= 0).all(), f"{sid}: bad depth values")
        check(validity is not None and validity.shape == depth.shape and validity.dtype == np.uint8,
              f"{sid}: bad validity shape/dtype")
        check(np.isin(validity, [0, 255]).all(), f"{sid}: validity must be binary")
        check(np.array_equal(validity != 0, depth > 0), f"{sid}: validity disagrees with depth")
        target = sample["depth_alignment"]
        check(target in ("rgb", "rectified_left"), f"{sid}: unknown alignment target")
        expected_target = "rectified_left" if config["settings"].get("align") == "left" else "rgb"
        check(target == expected_target, f"{sid}: alignment differs from session settings")
        for name in STREAMS:
            if name == "depth":
                continue
            img = cv2.imread(str(paths[name]), cv2.IMREAD_UNCHANGED)
            check(img is not None and img.dtype == np.uint8, f"{sid}: unreadable {name}")
            check(img.ndim == (3 if name == "rgb" else 2), f"{sid}: unexpected channels in {name}")
            if name == "rgb":
                check(img.shape[2] == 3, f"{sid}: RGB must have three channels")
            if name == target:
                check(img.shape[:2] == depth.shape, f"{sid}: alignment shape mismatch")
            geometry = sample["streams"][name]["geometry"]
            if geometry:
                check(list(img.shape[1::-1]) == geometry["size_wh"], f"{sid}: wrong {name} geometry size")
                k = np.array(geometry["intrinsics"])
                check(k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0,
                      f"{sid}: invalid {name} intrinsics")
            label_path = session / "annotations" / name / (sid + ".json")
            if name in ("rgb", "rectified_left") and label_path.exists():
                validate_annotation(json.loads(label_path.read_text()), sample, name, img.shape[1::-1])
        fraction = float((depth > 0).mean())
        check(abs(fraction - sample["depth_valid_fraction"]) < 1e-6, f"{sid}: wrong valid fraction")
        saved = json.loads((session / "samples" / sid / "sample.json").read_text())
        check(saved == sample, f"{sid}: manifest differs from sample metadata")
        valid_fractions.append(fraction)
        spans.append(span)
    directories = {p.name for p in (session / "samples").iterdir() if p.is_dir()}
    check(directories == seen, "Uncommitted/orphan sample directories; inspect before use")
    summary = json.loads((session / "summary.json").read_text())
    check(summary["samples"] == len(records), "Summary count differs from manifest")
    return {"session": str(session), "samples": len(records), "status": summary["status"],
            "max_sync_span_ms": max(spans), "mean_valid_depth_fraction": float(np.mean(valid_fractions)),
            "validation": "passed", "limitations": "Structural validation only; no depth accuracy or turtle-label validation"}


def validate_annotation(label, sample, view, size_wh):
    """Check annotation coordinates and provenance, without judging ground truth."""
    check(label["session_id"] == sample["session_id"] and label["sample_id"] == sample["sample_id"],
          "Annotation belongs to a different sample")
    check(label["view"] == view and label["image"] == sample["files"][view], "Annotation view mismatch")
    check(label["size_wh"] == list(size_wh), "Annotation image size mismatch")
    check(label["class"] == "turtle" and label["class_id"] == 0, "Unknown annotation class")
    boxes = label["boxes_xyxy"]
    check(isinstance(boxes, list), "Annotation boxes must be a list")
    expected_status = "reviewed_turtle" if boxes else "reviewed_background"
    check(label["annotation_status"] == expected_status, "Annotation status disagrees with boxes")
    width, height = size_wh
    for box in boxes:
        check(isinstance(box, list) and len(box) == 4, "Invalid annotation box")
        check(all(isinstance(v, (int, float)) and np.isfinite(v) for v in box), "Invalid box coordinates")
        x1, y1, x2, y2 = box
        check(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "Annotation box outside image")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(validate(args.session), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
