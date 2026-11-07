import numpy as np
import torch
import coremltools as ct
from PIL import Image
from train import ROOT
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2

p = ROOT / 'runs/new_inaturalist/originals/401705937.jpg'
im = LetterBox(new_shape=(640,640), auto=False)(image=cv2.imread(str(p)))[:,:,::-1].copy()
x = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()/255
model = YOLO(str(ROOT/'runs/detect/turtle_yolo26n/weights/best.pt')).model.eval().float()
with torch.no_grad():
    y = model(x)[0].numpy()
print('PyTorch top', y[0,:3], flush=True)
for name in ['best', 'turtle_fp32']:
    for unit in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL]:
        m=ct.models.MLModel(str(ROOT/f'runs/detect/turtle_yolo26n/weights/{name}.mlpackage'),compute_units=unit)
        output = next(iter(m.predict({'image':Image.fromarray(im)}).values()))
        print(name,unit,output[0,:3],flush=True)
