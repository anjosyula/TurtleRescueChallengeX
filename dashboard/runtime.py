"""Inference runtime kept separate from optional training and research dependencies."""
import json
import os
from pathlib import Path
import numpy as np


def local_runtime():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    path = Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics/settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    settings = json.loads(path.read_text()) if path.exists() else {}
    for key in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb"):
        settings[key] = False
    settings["api_key"] = ""
    path.write_text(json.dumps(settings, indent=2) + "\n")


def rtdetr_decode(model,image,transform,w,h):
    raw=model(image)[0][0].detach().cpu().numpy()
    raw=raw[raw[:,4]>=.001]
    xywh=raw[:,:4]*640
    boxes=np.column_stack((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2))
    boxes=transform.boxes(boxes,inverse=True)
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
    good=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    return np.column_stack((boxes,raw[:,4:6]))[good].astype(float).tolist()
