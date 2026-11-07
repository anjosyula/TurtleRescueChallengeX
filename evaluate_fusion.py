"""Measure real zero-depth fallback and training-only simulated depth ablations."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from train_challenge import ROOT, local_runtime, write_json, sha256
from train_fusion import load_fused, PairedShellDataset, shift_geometry
from turtle_geometry import prepare_rgb_geometry
from evaluate_challenge import coco_metrics, operating_points, breakdown


def decode(raw, transform, width, height, confidence=.001):
    from ultralytics.utils.nms import non_max_suppression
    # Host NMS avoids thousands of Metal scalar synchronizations in suppression.
    import torchvision.ops
    predictions=non_max_suppression(raw.detach().cpu().clone(),conf_thres=confidence,iou_thres=.7,max_det=300,nc=1,
                                   end2end=False,max_time_img=10.)[0].cpu().numpy()
    predictions[:,:4]=transform.boxes(predictions[:,:4],inverse=True)
    predictions[:,[0,2]]=predictions[:,[0,2]].clip(0,width)
    predictions[:,[1,3]]=predictions[:,[1,3]].clip(0,height)
    return predictions.astype(float).tolist()


def truth_boxes(row):
    text=(ROOT/row['label']).read_text().strip()
    labels=np.asarray([list(map(float,s.split())) for s in text.splitlines()],float).reshape(-1,5)[:,1:]
    xywh=labels*np.array([row['width'],row['height'],row['width'],row['height']])
    return np.column_stack((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2)).tolist()


def summarize(samples,folder,role,threshold):
    folder.mkdir(parents=True)
    write_json(folder/'predictions.json',samples)
    report={'role':role,'images':len(samples),'coco':coco_metrics(samples,folder),
            'operating_points':operating_points(samples),
            'at_inherited_rgb_threshold':{'confidence':threshold,**breakdown(samples,threshold)}}
    write_json(folder/'evaluation.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='mps')
    p.add_argument('--manifest',type=Path,default=ROOT/'datasets/turtle-real-synthetic-v1/manifest.json',
                   help='Synthetic diagnostic inputs; these remain training scenes, not held-out validation')
    args=p.parse_args();local_runtime();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    model,payload=load_fused(args.checkpoint,args.device)
    threshold=json.loads((ROOT/'models/challenge-20261019/model.json').read_text())['selected_confidence']
    rows=[r for r in json.loads((ROOT/'datasets/turtle-challenge-v1/manifest.json').read_text()) if r['split']=='val']
    samples=[];differences=[];times=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            rgb=cv2.cvtColor(cv2.imread(str(ROOT/row['image'])),cv2.COLOR_BGR2RGB)
            image,geometry,transform=prepare_rgb_geometry(rgb)
            image=torch.from_numpy(image[None]).to(args.device);geometry=torch.from_numpy(geometry[None]).to(args.device)
            start=time.perf_counter();actual=model(image,geometry)[0]
            # The detector is the exactly frozen original RGB checkpoint (training completion verifies this).
            reference=model.detector(image)[0]
            difference=float((actual-reference).abs().max());differences.append(difference)
            if difference!=0:raise AssertionError('Zero-depth fallback changed RGB outputs')
            predictions=decode(actual,transform,row['width'],row['height'])
            if i>=3:times.append((time.perf_counter()-start)*1000)
            samples.append({**{k:row[k] for k in ('image','width','height')},'truth':truth_boxes(row),'predictions':predictions})
            if (i+1)%10==0:print(f'Real fallback checks {i+1}/{len(rows)}',flush=True)
    reports={'real_zero_depth':summarize(samples,args.output/'real_zero_depth',
        '50 real validation photos, zero geometry only; no test of fusion benefit',threshold)}
    reports['real_zero_depth']['max_raw_difference_from_frozen_rgb']=max(differences)
    data=PairedShellDataset(args.manifest,augment=False)
    modes=('zero','oak','iphone','oak_shift_12px','oak_depth_reversed')
    sensitivity={m:[] for m in modes if m!='zero'}
    mode_samples={m:[] for m in modes}
    with torch.inference_mode():
        for i,row in enumerate(data.rows):
            image,oak,_=data.get(i,'oak');_,phone,_=data.get(i,'iphone')
            image=torch.from_numpy(image[None]).to(args.device)
            reverse=oak.copy();valid=reverse[1]>0
            reverse[0,valid]=1-reverse[0,valid]  # Preserve quality/support while perturbing metric-depth content.
            geometries={'zero':np.zeros_like(oak),'oak':oak,'iphone':phone,
                        'oak_shift_12px':shift_geometry(oak,12,0),'oak_depth_reversed':reverse}
            zero=model(image,torch.zeros_like(image))[0]
            _,_,transform=prepare_rgb_geometry(np.zeros((640,640,3),np.uint8))
            for mode,geometry in geometries.items():
                actual=zero if mode=='zero' else model(image,torch.from_numpy(geometry[None]).to(args.device))[0]
                if mode!='zero':sensitivity[mode].append(float((actual[:,4:]-zero[:,4:]).abs().max()))
                predictions=decode(actual,transform,640,640)
                mode_samples[mode].append({**{k:row[k] for k in ('image','width','height')},
                    'truth':truth_boxes(row),'predictions':predictions})
            if (i+1)%16==0:print(f'Synthetic diagnostics {i+1}/{len(data)}',flush=True)
    for mode in modes:
        reports['synthetic_'+mode]=summarize(mode_samples[mode],args.output/('synthetic_'+mode),
            'TRAINING scenes, same scan and renders; diagnostic only, not validation or field accuracy',threshold)
    domains={}
    for i,row in enumerate(data.rows):
        domain='gaussian' if row['source'].startswith('RGB-D initialized') else 'blender'
        domains.setdefault(domain,[]).append(i)
    domain_reports={}
    if len(domains)>1:
        for domain,indices in domains.items():
            domain_reports[domain]={}
            for mode in modes:
                subset=[mode_samples[mode][i] for i in indices]
                domain_reports[domain][mode]=summarize(subset,args.output/'by_domain'/domain/mode,
                    'Correlated synthetic diagnostic subset, NOT independent validation',threshold)
    report={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),
        'reports':reports,'raw_geometry_score_effect':{k:{'max':max(v),'median':float(np.median(v))} for k,v in sensitivity.items()},
        'diagnostic_manifest':str(args.manifest.resolve()),'diagnostic_manifest_sha256':sha256(args.manifest),
        'diagnostic_domain_counts':{k:len(v) for k,v in domains.items()},'by_domain':domain_reports,
        'paired_real_validation_available':False,'checkpoint_selection':'final fixed epoch; no selection on synthetic diagnostic AP',
        'zero_geometry_exact_rgb_parity_images':len(rows),'zero_geometry_max_raw_error':max(differences),
        'timing_note':'No hardware throughput claim; evaluation includes duplicate reference inference and host postprocessing'}
    write_json(args.output/'comparison.json',report)
    print(json.dumps({k:v['coco']['AP'] for k,v in reports.items()},indent=2),flush=True)


if __name__=='__main__':main()
