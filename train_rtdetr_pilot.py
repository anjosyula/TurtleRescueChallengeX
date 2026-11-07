"""Local, bounded RT-DETR-L turtle pilot using the existing audited RGB split."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import yaml
from train_challenge import ROOT, local_runtime, sha256, write_json
from turtle_geometry import prepare_rgb_geometry
from evaluate_fusion import truth_boxes


def prepare(destination):
    rows = json.loads((ROOT/'datasets/turtle-challenge-v1/manifest.json').read_text())
    lists = {'train': [], 'val': []}
    for row in rows:
        if sha256(ROOT/row['image']) != row['sha256'] or sha256(ROOT/row['label']) != row['label_sha256']:
            raise ValueError('Audited input changed')
        rgb = cv2.cvtColor(cv2.imread(str(ROOT/row['image'])), cv2.COLOR_BGR2RGB)
        image, _, transform = prepare_rgb_geometry(rgb)
        # RT-DETR normally stretches. Materialize square letterboxes so neither
        # its training loader nor predictor distorts the original aspect ratio.
        name = Path(row['image']).stem
        image_path = destination/'images'/row['split']/(name+'.png')
        label_path = destination/'labels'/row['split']/(name+'.txt')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.rint(image.transpose(1,2,0)*255).astype(np.uint8)
        if not cv2.imwrite(str(image_path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
            raise OSError(image_path)
        boxes = transform.boxes(np.asarray(truth_boxes(row)).reshape(-1,4))
        labels = []
        for x1,y1,x2,y2 in boxes:
            labels.append(f'0 {(x1+x2)/1280:.9f} {(y1+y2)/1280:.9f} {(x2-x1)/640:.9f} {(y2-y1)/640:.9f}')
        label_path.write_text('\n'.join(labels)+'\n')
        lists[row['split']].append(str(image_path.resolve()))
    for split, paths in lists.items():
        (destination/f'{split}.txt').write_text('\n'.join(paths)+'\n')
    (destination/'data.yaml').write_text(yaml.safe_dump(dict(path=str(destination.resolve()), train='train.txt',val='val.txt',names={0:'turtle'})))
    write_json(destination/'source_manifest.json', rows)
    return destination/'data.yaml'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',default='mps')
    args=p.parse_args()
    if args.epochs<1: p.error('Positive epochs required')
    local_runtime()
    import torch
    from ultralytics import RTDETR
    if importlib.metadata.version('ultralytics')!='8.4.157': raise RuntimeError('Pinned Ultralytics required')
    torch.set_num_threads(4)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    data=prepare(output/'data')
    initial=ROOT/'rtdetr-l.pt'
    if not initial.is_file(): raise FileNotFoundError('Download official Ultralytics rtdetr-l.pt first')
    model=RTDETR(str(initial))
    settings=dict(data=str(data),epochs=args.epochs,patience=0,imgsz=640,batch=2,nbs=8,
        device=args.device,workers=0,seed=42,deterministic=True,optimizer='AdamW',
        lr0=.0001,lrf=.1,cos_lr=True,warmup_epochs=1,warmup_bias_lr=0.,weight_decay=.0001,
        freeze=10,mosaic=0.,mixup=0.,cutmix=0.,degrees=5.,translate=.05,scale=.2,
        fliplr=.5,flipud=0.,hsv_h=.015,hsv_s=.4,hsv_v=.3,amp=False,plots=False,
        project=str(output),name='fit',exist_ok=False,save=True)
    provenance=dict(initial_sha256=sha256(initial),source='https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt',
        settings=settings,versions={k:importlib.metadata.version(k) for k in ('ultralytics','torch','numpy')},
        manifest_sha256=sha256(ROOT/'datasets/turtle-challenge-v1/manifest.json'),
        script_sha256=sha256(__file__),uploads=False,implementation='Ultralytics RT-DETR-L, not NVIDIA TAO execution',
        limitations=['Short frozen-backbone pilot, not a matched training-budget architecture comparison',
                    'Existing 50-image validation is model-selection data, not independent encounters',
                    'No real negative encounters or depth inputs'])
    write_json(output/'provenance.json',provenance)
    shutil.copy2(__file__,output/Path(__file__).name)
    def progress(trainer):
        write_json(output/'progress.json',dict(epoch=trainer.epoch+1,metrics=trainer.metrics))
    model.add_callback('on_fit_epoch_end',progress)
    start=time.perf_counter()
    model.train(**settings)
    write_json(output/'completed.json',dict(best=str(model.trainer.best),seconds=time.perf_counter()-start,
        epochs=args.epochs,metrics=model.metrics.results_dict,best_sha256=sha256(model.trainer.best)))


if __name__=='__main__': main()
