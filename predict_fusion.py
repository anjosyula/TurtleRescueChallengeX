"""Run the trained RGB-depth model on a local image or validated saved OAK session."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from train_challenge import ROOT, local_runtime, write_json, sha256
from train_fusion import load_fused
from turtle_geometry import prepare_rgb_geometry
from evaluate_fusion import decode
from predict_challenge import associate_depth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--model-card',type=Path,default=ROOT/'models/fused-experimental-20261019/model.json')
    p.add_argument('--depth',type=Path,help='For a single image: already aligned camera-axis float32 meters .npy')
    p.add_argument('--device',default='mps')
    p.add_argument('--conf',type=float)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-every',type=int,default=10)
    args=p.parse_args();local_runtime();torch.set_num_threads(4)
    card=json.loads(args.model_card.read_text())
    checkpoint=args.model_card.parent/card['checkpoint']
    if sha256(checkpoint)!=card['checkpoint_sha256']:raise ValueError('Checkpoint hash differs from model card')
    model,_=load_fused(checkpoint,args.device)
    threshold=args.conf if args.conf is not None else card['selected_confidence']
    if not 0<=threshold<=1 or args.save_every<1:raise ValueError('Invalid confidence or save interval')
    records=None;validation=None
    if (args.source/'manifest.jsonl').exists():
        if args.depth:raise ValueError('--depth is only for a single image')
        from validate_oak import validate
        validation=validate(args.source)
        records=[json.loads(s) for s in (args.source/'manifest.jsonl').read_text().splitlines() if s.strip()]
        sources=[args.source/r['files']['rgb'] for r in records]
    elif args.source.is_file():sources=[args.source]
    else:raise ValueError('Expected an image or recorded OAK session')
    args.output.mkdir(parents=True,exist_ok=False)
    predictions=[]
    with torch.inference_mode():
        for index,source in enumerate(sources):
            bgr=cv2.imread(str(source))
            if bgr is None:raise ValueError(f'Cannot read image: {source}')
            row=records[index] if records else None
            depth=validity=None
            if row and row['depth_alignment']=='rgb':
                depth=np.load(args.source/row['files']['depth'],allow_pickle=False)
                validity=cv2.imread(str(args.source/row['files']['validity']),cv2.IMREAD_GRAYSCALE)
                if validity is None:raise ValueError('Missing recorded validity')
            elif args.depth:
                depth=np.load(args.depth,allow_pickle=False)
            # Raw stereo confidence is NOT RGB-aligned. Use explicitly uncalibrated binary validity.
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            image,geometry,transform=prepare_rgb_geometry(rgb,depth,validity)
            image=torch.from_numpy(image[None]).to(args.device)
            actual=model(image,torch.from_numpy(geometry[None]).to(args.device))[0]
            fallback=model(image,torch.zeros_like(image))[0]
            depth_effect=float((actual[:,4:]-fallback[:,4:]).abs().max())
            h,w=rgb.shape[:2];boxes=[]
            for x1,y1,x2,y2,score,cls in decode(actual,transform,w,h,confidence=threshold):
                if score<threshold:continue
                box=[x1,y1,x2,y2]
                boxes.append({'bounding_box_original_image':box,'turtle_score':score,'class':'turtle',
                    'alert':'possible turtle; human review required','track_id':None,
                    **(associate_depth(np.where(validity>0,depth,0) if validity is not None else depth,box)
                       if depth is not None else {'distance_meters':None,'distance_quality':'unknown'})})
                cv2.rectangle(bgr,(round(x1),round(y1)),(round(x2),round(y2)),(0,220,255),2)
                cv2.putText(bgr,f'turtle? {score:.2f}',(round(x1),max(18,round(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
            predictions.append({'image':str(source.resolve()),'sample_id':row['sample_id'] if row else None,
                'timestamp':row['streams']['rgb']['device_monotonic_s'] if row else None,
                'timestamp_clock':'device_monotonic_seconds' if row else None,
                'sensor_mode':'rgb_depth_learned_fusion' if depth is not None else 'rgb_zero_depth_fallback',
                'geometry_valid_fraction':float(geometry[1].mean()),
                'quality_policy':'binary validity; uncalibrated; raw stereo confidence unused',
                'max_score_change_from_zero_depth':depth_effect,'detections':boxes,'not_site_clearance':True})
            if index%args.save_every==0:
                cv2.putText(bgr,'Experimental RGB-depth fusion',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
                cv2.imwrite(str(args.output/f'{index:06d}.jpg'),bgr)
    write_json(args.output/'predictions.json',predictions)
    write_json(args.output/'summary.json',{'checkpoint_sha256':sha256(checkpoint),'confidence':threshold,
        'images':len(predictions),'images_with_detections':sum(bool(r['detections']) for r in predictions),
        'detections':sum(len(r['detections']) for r in predictions),
        'images_where_depth_changes_scores':sum(r['max_score_change_from_zero_depth']>0 for r in predictions),
        'max_score_change_from_zero_depth':max(r['max_score_change_from_zero_depth'] for r in predictions),
        'real_paired_training':False,'note':'Inference smoke test only; unreviewed recordings cannot establish accuracy.'})
    if validation is not None:write_json(args.output/'validation.json',validation)
    print(args.output/'predictions.json')


if __name__=='__main__':main()
