"""Compare local turtle checkpoints with a fixed, square-letterbox protocol."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

from train_challenge import ROOT, local_runtime, sha256, write_json


def overlaps(box, targets):
    low = np.maximum(box[:2], targets[:, :2])
    high = np.minimum(box[2:4], targets[:, 2:4])
    intersection = np.maximum(high-low, 0).prod(axis=1)
    area = np.maximum(box[2:4]-box[:2], 0).prod()
    areas = np.maximum(targets[:, 2:4]-targets[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area+areas-intersection, 1e-9)


def match_predictions(predictions, truth, threshold=0.0):
    """Confidence-ranked, one-to-one matching at IoU .5; supports empty images."""
    truth = np.asarray(truth, dtype=float).reshape(-1,4)
    predictions = np.asarray(predictions, dtype=float).reshape(-1,6)
    used, outcomes = set(), []
    for box in predictions[np.argsort(-predictions[:,4], kind='stable')]:
        if box[4] < threshold:
            continue
        candidates = []
        if len(truth) and box[5] == 0:
            ious = overlaps(box, truth)
            candidates = [int(i) for i in np.argsort(-ious) if i not in used and ious[i] >= .5]
        index = candidates[0] if candidates else None
        if index is not None:
            used.add(index)
        outcomes.append((float(box[4]), int(index is not None)))
    return outcomes, used


def operating_points(samples):
    total = sum(len(s['truth']) for s in samples)
    ranked = sorted([row for s in samples for row in match_predictions(s['predictions'],s['truth'])[0]],
                    key=lambda row:-row[0])
    if not ranked:
        zero={'confidence':1.,'precision':0.,'recall':0.,'F1':0.,'F2':0.,'tp':0,'fp':0,'fn':total}
        return {'best_F1':zero,'best_F2':zero}
    scores=np.array([r[0] for r in ranked]);tp=np.cumsum([r[1] for r in ranked]);count=np.arange(1,len(ranked)+1)
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    def summary(i):
        return {'confidence':float(scores[i]),'precision':float(tp[i]/count[i]),
                'recall':float(tp[i]/max(total,1)),'F1':float(2*tp[i]/max(count[i]+total,1)),
                'F2':float(5*tp[i]/max(count[i]+4*total,1)),
                'tp':int(tp[i]),'fp':int(count[i]-tp[i]),'fn':int(total-tp[i])}
    return {'best_F1':summary(ends[np.argmax(2*tp[ends]/(count[ends]+total))]),
            'best_F2':summary(ends[np.argmax(5*tp[ends]/(count[ends]+4*total))])}


def coco_metrics(samples, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt={'info':{},'images':[],'annotations':[],'categories':[{'id':1,'name':'turtle'}]}
    detections=[]; aid=1; sizes=Counter()
    for image_id,s in enumerate(samples,1):
        gt['images'].append({'id':image_id,'file_name':s['image'],'height':s['height'],'width':s['width']})
        for x1,y1,x2,y2 in s['truth']:
            area=(x2-x1)*(y2-y1)
            sizes['small' if area<32**2 else 'medium' if area<96**2 else 'large']+=1
            gt['annotations'].append({'id':aid,'image_id':image_id,'category_id':1,
                'bbox':[x1,y1,x2-x1,y2-y1],'area':area,'iscrowd':0});aid+=1
        for x1,y1,x2,y2,score,cls in s['predictions']:
            if int(cls)!=0:raise ValueError('Expected one turtle class')
            detections.append({'image_id':image_id,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
    write_json(output/'coco_ground_truth.json',gt);write_json(output/'coco_predictions.json',detections)
    coco=COCO();coco.dataset=gt;coco.createIndex()
    if detections:
        result=coco.loadRes(detections)
    else:
        result=COCO();result.dataset={**gt,'annotations':[]};result.createIndex()
    evaluator=COCOeval(coco,result,'bbox');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    keys=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    return {**{k:float(v) if v>=0 else None for k,v in zip(keys,evaluator.stats)},
            'object_counts_original_pixel_area':dict(sizes),
            'size_definition':'COCO areas in original-image pixels; small <32^2, medium 32^2..96^2, large >96^2'}


def breakdown(samples, threshold):
    sizes={k:{'objects':0,'matched':0} for k in ('small','medium','large')}
    per_image=[]
    for s in samples:
        outcomes,used=match_predictions(s['predictions'],s['truth'],threshold)
        tp=len(used);per_image.append([tp,len(outcomes)-tp,len(s['truth'])-tp])
        scale=640/max(s['width'],s['height'])
        for i,(x1,y1,x2,y2) in enumerate(s['truth']):
            area=(x2-x1)*(y2-y1)*scale**2
            size='small' if area<32**2 else 'medium' if area<96**2 else 'large'
            sizes[size]['objects']+=1;sizes[size]['matched']+=int(i in used)
    for v in sizes.values():v['recall']=v['matched']/v['objects'] if v['objects'] else None
    counts=np.array(per_image);rng=np.random.default_rng(42);values=[]
    for _ in range(1000):
        tp,fp,fn=counts[rng.integers(len(counts),size=len(counts))].sum(axis=0)
        values.append([tp/max(tp+fp,1),tp/max(tp+fn,1)])
    intervals=np.quantile(values,[.025,.975],axis=0)
    return {'recall_by_size_at_640':sizes,
            'conditional_image_bootstrap_95pct':{'precision':intervals[:,0].tolist(),'recall':intervals[:,1].tolist()},
            'uncertainty_note':'Conditional on this validation set and selected threshold; does not include selection bias, encounter correlation, or retraining uncertainty.'}


def evaluate(checkpoint, destination, device):
    from ultralytics import YOLO
    import torch
    manifest=json.loads((ROOT/'datasets/turtle-challenge-v1/manifest.json').read_text())
    rows=[r for r in manifest if r['split']=='val']
    model=YOLO(str(checkpoint))
    samples=[];latencies=[]
    destination.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(rows):
        # Explicit square letterbox matches deployment, independent of batch aspect ratios.
        start=time.perf_counter()
        result=model.predict(str(ROOT/row['image']),imgsz=640,rect=False,conf=.001,iou=.7,
                             max_det=300,device=device,nms=None,verbose=False)[0]
        predictions=result.boxes.cpu().data.numpy().astype(float).tolist()
        elapsed=(time.perf_counter()-start)*1000
        if index>=3:latencies.append(elapsed)
        boxes=np.loadtxt(ROOT/row['label'],ndmin=2).reshape(-1,5)[:,1:]
        xywh=boxes*np.array([row['width'],row['height'],row['width'],row['height']])
        truth=np.column_stack((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2)).tolist()
        samples.append({'image':row['image'],'width':row['width'],'height':row['height'],
                        'truth':truth,'predictions':predictions})
    write_json(destination/'predictions.json',samples)
    points=operating_points(samples)
    threshold=points['best_F2']['confidence']
    report={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'images':len(samples),
            'objects':sum(len(s['truth']) for s in samples),'inference':'one-to-many + NMS; 640 square; batch=1; conf=.001; IoU NMS=.7; max_det=300',
            'coco':coco_metrics(samples,destination),'operating_points_IoU50':points,
            'selected_confidence':max(0,float(np.nextafter(np.float32(threshold),np.float32(-np.inf)))),
            'threshold_selection':'Maximum validation F2, favors recall; not a field-safety threshold',
            **breakdown(samples,threshold),
            'host_latency_ms':{'median':float(np.median(latencies)),'p95':float(np.quantile(latencies,.95)),
                               'scope':'file read + preprocessing + inference + NMS + CPU transfer; first three images excluded; host only'},
            'parameters':sum(p.numel() for p in model.model.parameters()),'checkpoint_bytes':checkpoint.stat().st_size,
            'limitations':['Validation set used for model and threshold selection; not independent testing',
                           'No turtle-free negatives: false alarms per hour cannot be measured',
                           'No encounter timestamps, species, occlusion, or site labels',
                           'RGB only; no trained depth fusion or IR model']}
    write_json(destination/'evaluation.json',report)
    # Save selected-threshold evidence without another inference pass.
    for index in [0,10,20,30,40]:
        import cv2
        s=samples[index];im=cv2.imread(str(ROOT/s['image']))
        for x1,y1,x2,y2,score,cls in s['predictions']:
            if score>=threshold:
                cv2.rectangle(im,(round(x1),round(y1)),(round(x2),round(y2)),(0,200,0),2)
                cv2.putText(im,f'turtle {score:.3f}',(max(0,round(x1)),max(20,round(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,200,0),2)
        cv2.imwrite(str(destination/f'prediction_{index:02d}.jpg'),im)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints',type=Path,nargs='+');p.add_argument('--device',default='mps')
    p.add_argument('--output',type=Path,default=ROOT/'runs/challenge/comparison_20261019')
    args=p.parse_args();local_runtime()
    comparison=args.output/'comparison.json'
    reports=json.loads(comparison.read_text()) if comparison.exists() else {}
    for checkpoint in args.checkpoints:
        name=checkpoint.parent.parent.name
        reports[name]=evaluate(checkpoint.resolve(),args.output/name,args.device)
        print(name,json.dumps(reports[name]['coco']),flush=True)
    write_json(args.output/'comparison.json',reports)


if __name__=='__main__':main()
