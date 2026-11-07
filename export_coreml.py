from train import ROOT
from ultralytics import YOLO

if __name__ == '__main__':
    YOLO(str(ROOT / 'runs/detect/turtle_yolo26n/weights/best.pt')).export(
        format='coreml', imgsz=640, batch=1, nms=None, quantize=16,
    )
