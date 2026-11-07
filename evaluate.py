"""Evaluate checkpoints and record validation-tuned confidence thresholds."""
import json
import numpy as np
from train import ROOT
from ultralytics import YOLO

if __name__ == '__main__':
    summaries = {}
    for checkpoint in ('best', 'last'):
        path = ROOT / 'runs/detect/turtle_yolo26n/weights' / f'{checkpoint}.pt'
        model = YOLO(str(path))
        metrics = model.val(data=str(ROOT / 'datasets/turtle-hero/data.yaml'),
                            device='mps', imgsz=640, batch=8, workers=0,
                            project=str(ROOT / 'runs/evaluate'), name=checkpoint,
                            plots=True)
        box = metrics.box
        index = int(np.argmax(box.f1_curve.mean(axis=0)))
        threshold = float(box.px[index])
        summaries[checkpoint] = {
            'weights': str(path.relative_to(ROOT)),
            'mAP50': float(box.map50), 'mAP50_95': float(box.map),
            'validation_tuned_confidence': threshold,
            'precision_at_threshold': float(box.p_curve[:, index].mean()),
            'recall_at_threshold': float(box.r_curve[:, index].mean()),
            'F1_at_threshold': float(box.f1_curve[:, index].mean()),
            'note': 'Threshold selected on validation data; not an independent test result.',
        }
        results = model.predict(source=str(ROOT / 'datasets/turtle-hero/images/val'),
                                conf=threshold, imgsz=640, device='mps',
                                save=True, project=str(ROOT / 'runs/predict'),
                                name=checkpoint, verbose=False)
        summaries[checkpoint]['validation_images_with_predictions'] = sum(len(r.boxes) > 0 for r in results)
        print(json.dumps(summaries[checkpoint], indent=2), flush=True)
    (ROOT / 'runs/detect/turtle_yolo26n/evaluation.json').write_text(json.dumps(summaries, indent=2) + '\n')
