import json
import shutil
from train import ROOT
from ultralytics import YOLO

if __name__ == '__main__':
    weights = ROOT / 'runs/detect/turtle_yolo26n/weights'
    model = YOLO(str(weights / 'best.pt'))
    metrics = model.val(data=str(ROOT / 'datasets/turtle-hero/data.yaml'), imgsz=640, rect=False,
                        batch=1, device='cpu', workers=0, conf=.001,
                        project=str(ROOT / 'runs/evaluate'), name='pytorch_square', plots=False)
    shutil.copy2(weights / 'best.pt', weights / 'turtle_fp32.pt')
    exported = YOLO(str(weights / 'turtle_fp32.pt')).export(format='coreml', imgsz=640, batch=1, nms=False, quantize=None)
    coreml = YOLO(exported, task='detect')
    converted = coreml.val(data=str(ROOT / 'datasets/turtle-hero/data.yaml'), imgsz=640,
                          batch=1, device='cpu', workers=0, conf=.001,
                          project=str(ROOT / 'runs/evaluate'), name='coreml_fp32', plots=False)
    report = {'pytorch_square':metrics.results_dict, 'coreml_fp32':converted.results_dict}
    (ROOT / 'runs/evaluate/coreml_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
