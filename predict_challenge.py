"""Run the selected turtle detector locally on images or a saved OAK session."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from train_challenge import ROOT,local_runtime,write_json


def associate_depth(depth,box,min_support=.3,min_pixels=20,max_relative_iqr=.2):
    """Conservative interior-box heuristic, not a calibrated range confidence."""
    h,w=depth.shape;x1,y1,x2,y2=box
    dx,dy=(x2-x1)*.25,(y2-y1)*.25
    left,right=max(0,int(np.ceil(x1+dx))),min(w,int(np.floor(x2-dx)))
    top,bottom=max(0,int(np.ceil(y1+dy))),min(h,int(np.floor(y2-dy)))
    region=depth[top:bottom,left:right] if right>left and bottom>top else np.empty((0,0))
    values=region[np.isfinite(region)&(region>0)]
    fraction=len(values)/region.size if region.size else 0.
    result={'distance_meters':None,'distance_quality':'unknown','valid_support_fraction':fraction,
            'policy':'central 50% of box; median camera-axis depth, minimum support and IQR checks; uncalibrated'}
    if len(values)<min_pixels or fraction<min_support:return result
    q1,median,q3=np.quantile(values,[.25,.5,.75])
    result['relative_iqr']=float((q3-q1)/median)
    if result['relative_iqr']<=max_relative_iqr:
        result.update(distance_meters=float(median),distance_quality='low')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--model-card',type=Path,default=ROOT/'models/challenge-20261019/model.json')
    p.add_argument('--device',default='mps');p.add_argument('--conf',type=float)
    p.add_argument('--output',type=Path,default=ROOT/'runs/challenge/predictions')
    p.add_argument('--save-every',type=int,default=10,help='Save every Nth annotated frame; all predictions retained')
    args=p.parse_args();local_runtime()
    from ultralytics import YOLO
    card=json.loads(args.model_card.read_text())
    model=YOLO(str(args.model_card.parent/card['checkpoint']))
    conf=args.conf if args.conf is not None else card['selected_confidence']
    if not 0<=conf<=1 or args.save_every<1:raise ValueError('Invalid confidence or save interval')
    if args.output.exists():raise FileExistsError('Choose a new --output directory')
    manifest=args.source/'manifest.jsonl'
    records=None
    if manifest.exists():
        from validate_oak import validate
        validate(args.source)
        records=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        sources=[args.source/r['files']['rgb'] for r in records]
    elif args.source.is_file():sources=[args.source]
    else:sources=sorted(p for p in args.source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if not sources:raise ValueError('No input images')
    args.output.mkdir(parents=True)
    all_predictions=[]
    for index,source in enumerate(sources):
        result=model.predict(str(source),imgsz=640,rect=False,conf=conf,iou=.7,nms=None,
                             device=args.device,verbose=False)[0]
        record=records[index] if records else None
        depth=np.load(args.source/record['files']['depth'],allow_pickle=False) if record and record['depth_alignment']=='rgb' else None
        boxes=[]
        for x1,y1,x2,y2,score,cls in result.boxes.cpu().data.numpy():
            if int(cls)!=0:raise ValueError('Unexpected class')
            box=[float(x1),float(y1),float(x2),float(y2)]
            boxes.append({'bounding_box_original_image':box,'turtle_score':float(score),'class':'turtle',
                'alert':'possible turtle; human review required','track_id':None,
                **(associate_depth(depth,box) if depth is not None else {'distance_meters':None,'distance_quality':'unknown'})})
        output={'image':str(source),'sample_id':record['sample_id'] if record else None,
                'timestamp':record['streams']['rgb']['device_monotonic_s'] if record else None,
                'timestamp_clock':'device_monotonic_seconds' if record else None,
                'sensor_mode':'oak_rgb_with_posthoc_depth' if depth is not None else 'rgb',
                'detections':boxes,'not_site_clearance':True}
        all_predictions.append(output)
        if index%args.save_every==0 or len(sources)==1:
            result.save(filename=str(args.output/f'{index:06d}.jpg'))
    write_json(args.output/'predictions.json',all_predictions)
    write_json(args.output/'summary.json',{'model_card':str(args.model_card),'confidence':conf,
        'images':len(sources),'images_with_detections':sum(bool(s['detections']) for s in all_predictions),
        'detections':sum(len(s['detections']) for s in all_predictions),
        'note':'Predictions, not reviewed annotations; no detection does not establish site clearance.'})
    print(args.output/'predictions.json')


if __name__=='__main__':main()
