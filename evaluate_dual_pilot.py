"""Compare a local RT-DETR pilot and independent RGB/depth OR, without relabeling data."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from train_challenge import ROOT,local_runtime,sha256,write_json
from train_depth_pilot import load_depth
from train_fusion import load_fused,PairedShellDataset
from turtle_geometry import prepare_rgb_geometry
from turtle_late_fusion import late_or
from evaluate_fusion import decode,truth_boxes
from evaluate_challenge import coco_metrics,operating_points,match_predictions,breakdown


def counts(samples,threshold):
    tp=fp=fn=0
    for sample in samples:
        outcomes,used=match_predictions(sample['predictions'],sample['truth'],threshold)
        tp+=len(used);fp+=len(outcomes)-len(used);fn+=len(sample['truth'])-len(used)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


def at_fp_budget(samples,budget):
    ranked=sorted([p for s in samples for p in match_predictions(s['predictions'],s['truth'])[0]],reverse=True)
    total=sum(len(s['truth']) for s in samples)
    points=[dict(threshold=1.,tp=0,fp=0,fn=total,precision=0.,recall=0.)]
    tp=fp=0
    for i,(score,hit) in enumerate(ranked):
        tp+=hit;fp+=1-hit
        if i+1<len(ranked) and ranked[i+1][0]==score:continue
        if fp<=budget:points.append(dict(threshold=score,tp=tp,fp=fp,fn=total-tp,
            precision=tp/max(tp+fp,1),recall=tp/max(total,1)))
    return max(points,key=lambda p:(p['tp'],-p['fp']))


def merged_predictions(result):
    # Used only for fixed-threshold diagnostics, never score-based AP ranking.
    # Constant scores retain deterministic RGB-first matching; incomparable
    # branch confidence scales are not combined into one probability.
    return [d['box']+[1.,0.] for d in result['detections']]


def report(samples,destination):
    destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'predictions.json',samples)
    points=operating_points(samples)
    threshold=points['best_F2']['confidence']
    result=dict(coco=coco_metrics(samples,destination),operating_points=points,
                at_selected_F2=counts(samples,threshold),**breakdown(samples,threshold))
    write_json(destination/'evaluation.json',result)
    return result


def rtdetr_decode(model,image,transform,w,h):
    raw=model(image)[0][0].detach().cpu().numpy()
    raw=raw[raw[:,4]>=.001]
    xywh=raw[:,:4]*640
    boxes=np.column_stack((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2))
    boxes=transform.boxes(boxes,inverse=True)
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
    good=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    return np.column_stack((boxes,raw[:,4:6]))[good].astype(float).tolist()


def filtered_boxes(preds):
    return [p for p in preds if p[2]>p[0] and p[3]>p[1]]


def overlay(bgr,alerts,label):
    out=bgr.copy()
    for d in alerts['detections']:
        x1,y1,x2,y2=map(round,d['box'])
        color=(0,210,255) if 'rgb' in d['branches'] else (255,180,0)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        text='/'.join(f'{k}:{v:.2f}' for k,v in d['branches'].items())
        cv2.putText(out,text,(max(0,x1),max(45,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    cv2.putText(out,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Fresh evaluation folder; defaults to EXPERIMENT/evaluation')
    p.add_argument('--device',default='mps')
    args=p.parse_args();local_runtime();torch.set_num_threads(4)
    from ultralytics import YOLO,RTDETR
    from validate_oak import validate
    root=args.experiment.resolve();out=args.output.resolve() if args.output else root/'evaluation';out.mkdir(parents=True,exist_ok=False)
    rgb_path=ROOT/'models/challenge-20261019/turtle.pt'
    depth_path=root/'depth/depth.pt';rt_path=root/'rtdetr/fit/weights/best.pt'
    fused_path=ROOT/'models/fused-scan-splat-40ep-20261026/turtle_fused.pt'
    card=json.loads((ROOT/'models/challenge-20261019/model.json').read_text())
    rgb_threshold=card['selected_confidence'];depth_threshold=.25
    rgb=YOLO(str(rgb_path)).model.float().to(args.device).eval();rgb.model[-1].end2end=False
    depth,_=load_depth(depth_path,args.device)
    fused,_=load_fused(fused_path,args.device)
    rt=RTDETR(str(rt_path)).model.float().to(args.device).eval()
    write_json(out/'provenance.json',dict(checkpoints={str(p):sha256(p) for p in (rgb_path,depth_path,rt_path,fused_path)},
        parameters={k:sum(p.numel() for p in model.parameters()) for k,model in [('rgb',rgb),('depth',depth),('rtdetr',rt),('fused',fused)]},
        rgb_threshold=rgb_threshold,depth_threshold=depth_threshold,
        depth_threshold_policy='Fixed experimental .25, not calibrated on real data',
        depth_availability_policy='Invoke only if preprocessed valid depth contains at least one point; this is availability, not a quality guarantee',
        source_sha256=sha256(__file__),uploads=False,
        latency_scope='Host, batch 1, preprocessed tensor through decoding/CPU transfer; excludes acquisition and preprocessing',
        data_roles={'rgb':'Existing model-selection validation, 50 photos',
                    'synthetic':'All 192 scan derivatives remain training diagnostics, never held-out',
                    'oak':'Unreviewed replay; no accuracy or false-alert-rate claims'}))
    def tensor(value):return torch.from_numpy(value[None]).to(args.device)
    def infer_yolo(model,value,transform,w,h):
        return filtered_boxes(decode(model(tensor(value))[0],transform,w,h))
    rows=[r for r in json.loads((ROOT/'datasets/turtle-challenge-v1/manifest.json').read_text()) if r['split']=='val']
    real={'rgb':[],'rtdetr':[]};latencies={k:[] for k in real}
    with torch.inference_mode():
        for index,row in enumerate(rows):
            bgr=cv2.imread(str(ROOT/row['image']));image,_,transform=prepare_rgb_geometry(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            x=tensor(image)
            base={k:row[k] for k in ('image','width','height')};base['truth']=truth_boxes(row)
            for key,model in [('rgb',rgb),('rtdetr',rt)]:
                if args.device=='mps':torch.mps.synchronize()
                start=time.perf_counter()
                pred=rtdetr_decode(model,x,transform,row['width'],row['height']) if key=='rtdetr' else filtered_boxes(decode(model(x)[0],transform,row['width'],row['height']))
                if index>=3:latencies[key].append((time.perf_counter()-start)*1000)
                real[key].append(dict(base,predictions=pred))
            if index%10==0:print(f'RGB comparison {index+1}/{len(rows)}',flush=True)
        real_reports={key:report(value,out/f'real_{key}') for key,value in real.items()}
        for key,values in latencies.items():
            real_reports[key]['latency_ms']=dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)))
        # A matched-count diagnostic is possible; false alerts/minute is not.
        target_fp=counts(real['rgb'],rgb_threshold)['fp']
        real_reports['rtdetr_at_rgb_fp_budget']=at_fp_budget(real['rtdetr'],target_fp)
        real_reports['rgb_at_existing_threshold']=counts(real['rgb'],rgb_threshold)
        write_json(out/'rgb_comparison.json',real_reports)
        rt_threshold=real_reports['rtdetr']['operating_points']['best_F2']['confidence']

        data=PairedShellDataset(ROOT/'datasets/turtle-gaussian-v1/manifest.json',augment=False)
        synthetic={profile:{key:[] for key in ('rgb','depth','depth_reversed','fused')} for profile in ('oak','iphone')}
        all_or={profile:{str(t):[] for t in (.1,.25,.5)} for profile in synthetic}
        for index,row in enumerate(data.rows):
            image,_,_=data.get(index,'oak')
            _,_,transform=prepare_rgb_geometry(np.zeros((640,640,3),np.uint8))
            base={k:row[k] for k in ('image','width','height')};base['truth']=truth_boxes(row)
            r=infer_yolo(rgb,image,transform,640,640)
            for profile in synthetic:
                _,geometry,_=data.get(index,profile)
                available=bool(geometry[1].any())
                d=infer_yolo(depth,geometry,transform,640,640) if available else []
                reversed_geometry=geometry.copy()
                support=reversed_geometry[1]>0
                reversed_geometry[0,support]=1-reversed_geometry[0,support]
                reversed_pred=infer_yolo(depth,reversed_geometry,transform,640,640) if available else []
                f=filtered_boxes(decode(fused(tensor(image),tensor(geometry))[0],transform,640,640))
                for key,pred in [('rgb',r),('depth',d),('depth_reversed',reversed_pred),('fused',f)]:synthetic[profile][key].append(dict(base,predictions=pred))
                for threshold in (.1,.25,.5):
                    combined=late_or(r,d,rgb_threshold,threshold,available)
                    all_or[profile][str(threshold)].append(dict(base,predictions=merged_predictions(combined),alert=combined))
            if index%24==0:print(f'Synthetic diagnostic {index+1}/{len(data)}',flush=True)
        synth_reports={}
        for profile,modes in synthetic.items():
            synth_reports[profile]={key:report(samples,out/f'training_{profile}_{key}') for key,samples in modes.items()}
            synth_reports[profile]['fixed_threshold_counts']={key:counts(samples,depth_threshold if key.startswith('depth') else rgb_threshold) for key,samples in modes.items()}
            for threshold,samples in all_or[profile].items():
                write_json(out/f'training_{profile}_or_{threshold}.json',samples)
                synth_reports[profile][f'OR_depth_{threshold}']=counts(samples,0.)
        write_json(out/'synthetic_diagnostics.json',synth_reports)
        # All original camera data remains unmodified and unreviewed.
        replay_reports={}
        for session_name in ('20261016T205218_98151fbf','20261016T205810_f6a2ab4a'):
            session=ROOT/'recordings/oak'/session_name
            validation=validate(session)
            records=[json.loads(s) for s in (session/'manifest.jsonl').read_text().splitlines() if s.strip()]
            folder=out/session_name;folder.mkdir();write_json(folder/'validation.json',validation)
            predictions=[];tiles=[]
            for index,row in enumerate(records):
                bgr=cv2.imread(str(session/row['files']['rgb']))
                z=valid=None
                if row['depth_alignment']=='rgb':
                    z=np.load(session/row['files']['depth'],allow_pickle=False)
                    valid=cv2.imread(str(session/row['files']['validity']),cv2.IMREAD_GRAYSCALE)
                    if valid is None:raise ValueError('Recorded validity missing')
                image,geometry,transform=prepare_rgb_geometry(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),z,valid)
                h,w=bgr.shape[:2];available=bool(geometry[1].any())
                r=infer_yolo(rgb,image,transform,w,h)
                d=infer_yolo(depth,geometry,transform,w,h) if available else []
                f=filtered_boxes(decode(fused(tensor(image),tensor(geometry))[0],transform,w,h))
                rt_pred=rtdetr_decode(rt,tensor(image),transform,w,h)
                combined=late_or(r,d,rgb_threshold,depth_threshold,available)
                # Explicit missing-depth replay checks decision behavior on every frame.
                missing=late_or(r,None,rgb_threshold,depth_threshold,False)
                if missing['possible_turtle']!=combined['rgb_hit']:raise AssertionError('Missing depth vetoed RGB')
                predictions.append(dict(sample_id=row['sample_id'],timestamp=row['streams']['rgb']['device_monotonic_s'],
                    geometry_valid_fraction=float(geometry[1].mean()),rgb=r,depth=d,fused=f,rtdetr=rt_pred,
                    **combined,missing_depth_possible_turtle=missing['possible_turtle'],labels_reviewed=False))
                if index%10==0:
                    frame=overlay(bgr,combined,f'{session_name} / {row["sample_id"]} - experimental OR, unreviewed')
                    cv2.imwrite(str(folder/f'{row["sample_id"]}.jpg'),frame)
                    tiles.append(cv2.resize(frame,(480,360)))
                if index%20==0:print(f'OAK replay {session_name}: {index+1}/{len(records)}',flush=True)
            write_json(folder/'predictions.json',predictions)
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(folder/'contact.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
            replay_reports[session_name]=dict(frames=len(predictions),
                rgb_alert_frames=sum(p['rgb_hit'] for p in predictions),
                depth_alert_frames=sum(p['depth_hit'] is True for p in predictions),
                or_alert_frames=sum(p['possible_turtle'] for p in predictions),
                depth_only_added_alert_frames=sum(p['depth_hit'] is True and not p['rgb_hit'] for p in predictions),
                fused_alert_frames=sum(any(b[4]>=rgb_threshold for b in p['fused']) for p in predictions),
                rtdetr_alert_frames=sum(any(b[4]>=rt_threshold for b in p['rtdetr']) for p in predictions),
                missing_depth_preserves_rgb=True,accuracy=None,
                note='Alert counts are not correct detection counts; frames are correlated and unreviewed')
        write_json(out/'oak_replay_summary.json',replay_reports)
    write_json(out/'completed.json',dict(status='completed',rgb_images=len(rows),synthetic_scenes=len(data),
        oak_frames=sum(r['frames'] for r in replay_reports.values()),rgb_threshold=rgb_threshold,
        depth_threshold=depth_threshold,rtdetr_threshold=rt_threshold,default_model_changed=False))


if __name__=='__main__':main()
