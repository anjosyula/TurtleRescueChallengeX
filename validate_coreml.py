import json
from train import ROOT
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'runs/detect/turtle_yolo26n/weights/best.mlpackage'), task='detect')
    metrics = model.val(data=str(ROOT / 'datasets/turtle-hero/data.yaml'), imgsz=640,
                        batch=1, device='cpu', workers=0, conf=0.001,
                        project=str(ROOT / 'runs/evaluate'), name='coreml', plots=False)
    results = model.predict(str(ROOT / 'runs/new_inaturalist/originals'), imgsz=640,
                            conf=0.016388704, device='cpu', save=True,
                            project=str(ROOT / 'runs/predict'), name='coreml')
    report = {'metrics':metrics.results_dict,
              'predictions':{r.path:json.loads(r.to_json()) for r in results}}
    (ROOT / 'runs/evaluate/coreml_report.json').write_text(json.dumps(report, indent=2))
