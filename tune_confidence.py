"""Select confidence on validation detections using exact IoU=0.5 matching."""
import json
import numpy as np
from train import ROOT
from ultralytics import YOLO


def iou(box, targets):
    lo = np.maximum(box[:2], targets[:, :2])
    hi = np.minimum(box[2:], targets[:, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(targets[:, 2:] - targets[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def main():
    model = YOLO(str(ROOT / 'runs/detect/turtle_yolo26n/weights/best.pt'))
    samples = []
    total = 0
    for result in model.predict(str(ROOT / 'datasets/turtle-hero/images/val'), conf=0.001,
                                imgsz=640, device='mps', stream=True, verbose=False):
        label = ROOT / 'datasets/turtle-hero/labels/val' / (Path(result.path).stem + '.txt')
        targets = np.array([[float(v) for v in line.split()] for line in label.read_text().splitlines()])
        height, width = result.orig_shape
        xywh = targets[:, 1:] * np.array([width, height, width, height])
        truth = np.column_stack((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2))
        total += len(truth)
        used = set()
        for box in sorted(result.boxes.cpu().data.numpy(), key=lambda b: -b[4]):
            overlaps = iou(box[:4], truth)
            candidates = [i for i in np.argsort(-overlaps) if i not in used and overlaps[i] >= .5 and targets[i, 0] == box[5]]
            matched = bool(candidates)
            if matched:
                used.add(candidates[0])
            samples.append((float(box[4]), int(matched)))
    samples.sort(reverse=True)
    scores = np.array([x[0] for x in samples])
    tp = np.cumsum([x[1] for x in samples])
    counts = np.arange(1, len(samples) + 1)
    indices = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    def summarize(index):
        precision = tp[index] / counts[index]
        recall = tp[index] / total
        return dict(confidence=float(scores[index]), precision=float(precision), recall=float(recall),
                    F1=float(2 * tp[index] / (counts[index] + total)),
                    F2=float(5 * tp[index] / (counts[index] + 4 * total)),
                    true_positives=int(tp[index]), false_positives=int(counts[index] - tp[index]),
                    false_negatives=int(total - tp[index]))
    best_f1 = int(indices[np.argmax(2 * tp[indices] / (counts[indices] + total))])
    best_f2 = int(indices[np.argmax(5 * tp[indices] / (counts[indices] + 4 * total))])
    old_index = int(np.count_nonzero(scores >= .016) - 1)
    report = {'method':'Confidence-ranked greedy matching at IoU >= 0.5; F2 emphasizes recall. Tuned on 50 validation images, not an independent test.',
              'previous':summarize(old_index), 'best_F1':summarize(best_f1), 'best_F2':summarize(best_f2)}
    report['previous']['confidence'] = 0.016
    threshold = report['best_F2']['confidence'] - 1e-9  # Include the boundary detection.
    report['selected_confidence'] = threshold
    output = ROOT / 'runs/new_inaturalist'
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    backup = output / 'manifest_conf_0.016.json'
    if not backup.exists():
        backup.write_text(manifest_path.read_text())
    manifest['confidence'] = threshold
    manifest['threshold_method'] = 'Maximum validation F2 at IoU 0.5'
    for row in manifest['images']:
        result = model.predict(str(ROOT / row['original']), conf=threshold, imgsz=640, device='mps', verbose=False)[0]
        result.save(filename=str(ROOT / row['annotated']))
        row['detections'] = json.loads(result.to_json())
        print(f"{row['observation_id']}: {len(result.boxes)} detections", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'confidence_tuning.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    from pathlib import Path
    main()
