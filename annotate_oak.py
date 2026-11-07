#!/usr/bin/env python3
"""Review saved OAK images and draw turtle boxes locally with OpenCV."""
import argparse
import json
from pathlib import Path

import cv2

from collect_oak import utc_now, write_json
from validate_oak import validate, validate_annotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", type=Path)
    p.add_argument("--view", choices=("rgb", "rectified_left"), default="rgb")
    p.add_argument("--proposals", type=Path, help="Verified proposal file; a explicitly accepts nonempty boxes")
    args = p.parse_args()
    validate(args.session)
    records = [json.loads(line) for line in (args.session / "manifest.jsonl").read_text().splitlines() if line]
    proposals = None
    if args.proposals:
        from oak_review import load_proposals
        proposals = load_proposals(args.proposals, args.session, records, args.view)
        selected = set(proposals['selected_sample_ids'])
        records = [r for r in records if r['sample_id'] in selected]
    labels = args.session / "annotations" / args.view
    labels.mkdir(parents=True, exist_ok=True)
    index = 0
    title = "Turtle review: b=boxes a=accept proposals e=empty n=skip p=previous q=quit"
    print("b: draw boxes, Enter/Space accepts each, Esc finishes; e: confirm no visible turtles.")
    print("n skips without changing labels. Existing labels are shown; b replaces them.")
    print("Amber boxes are unreviewed proposals. a explicitly accepts all shown proposals; use b to replace incorrect boxes.")
    try:
        while 0 <= index < len(records):
            record = records[index]
            source = args.session / record["files"][args.view]
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Cannot read {source}")
            label_path = labels / (record["sample_id"] + ".json")
            previous = json.loads(label_path.read_text()) if label_path.exists() else None
            display = image.copy()
            proposed = proposals['samples'][record['sample_id']]['boxes_xyxy'] if proposals else []
            if not previous:
                for x1,y1,x2,y2 in proposed:
                    cv2.rectangle(display,(round(x1),round(y1)),(round(x2),round(y2)),(0,180,255),2)
            if previous:
                validate_annotation(previous, record, args.view, image.shape[1::-1])
                for x1, y1, x2, y2 in previous["boxes_xyxy"]:
                    cv2.rectangle(display, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), (0, 255, 0), 2)
            status = previous["annotation_status"] if previous else "unreviewed"
            cv2.putText(display, f"{index + 1}/{len(records)}: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(title, display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                index = max(0, index - 1)
                continue
            if key == ord("n"):
                index += 1
                continue
            if key not in (ord("b"), ord("e"), ord("a")):
                continue
            boxes = []
            if key == ord('a'):
                if previous or not proposed:
                    continue  # Never accept unseen proposals or infer a background label.
                boxes = proposed
            if key == ord("b"):
                rois = cv2.selectROIs("Draw visible turtle boxes; Esc finishes", image, fromCenter=False)
                cv2.destroyWindow("Draw visible turtle boxes; Esc finishes")
                boxes = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in rois if w > 0 and h > 0]
                if not boxes:
                    continue  # Canceling must not silently label an image as background.
            write_json(label_path, {
                "schema_version": 1, "session_id": record["session_id"],
                "sample_id": record["sample_id"], "view": args.view,
                "image": record["files"][args.view], "size_wh": list(image.shape[1::-1]),
                "annotation_status": "reviewed_turtle" if boxes else "reviewed_background",
                "class": "turtle", "class_id": 0, "boxes_xyxy": boxes,
                "coordinate_convention": "original image pixels, xyxy, exclusive upper bounds",
                "reviewed_utc": utc_now(),
                "review_method": "explicit_proposal_acceptance" if key == ord('a') else "manual_boxes" if boxes else "explicit_background",
                "proposal_checkpoint_sha256": proposals['checkpoint_sha256'] if key == ord('a') else None,
            })
            index += 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
