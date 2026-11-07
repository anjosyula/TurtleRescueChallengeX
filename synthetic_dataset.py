"""Validate Blender shell renders and assemble a local real-plus-synthetic split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from train_challenge import ROOT,sha256,write_json


def simulated_depth(depth, profile, seed, focal_px=None):
    """Uncalibrated sensor-like corruption; never a measured sensor depth map."""
    if profile not in ('oak','iphone'):raise ValueError('Unknown simulation profile')
    rng=np.random.default_rng(seed);h,w=depth.shape
    focal_px=float(w if focal_px is None else focal_px)
    divisor=2 if profile=='oak' else 4
    low=cv2.resize(depth,(max(1,w//divisor),max(1,h//divisor)),interpolation=cv2.INTER_NEAREST)
    valid=np.isfinite(low)&(low>=.4 if profile=='oak' else low>=.2)&(low<=(10 if profile=='oak' else 5))
    # Invalid measurements remain unknown through noise, quantization, and upsampling.
    low=np.where(valid,low,0).astype(np.float32)
    if profile=='oak':
        disparity=np.zeros_like(low);disparity[valid]=(.075*focal_px/divisor)/low[valid]
        disparity[valid]+=rng.normal(0,.15,int(valid.sum()))
        disparity=np.round(disparity*8)/8;valid &= disparity>0
        noisy=np.zeros_like(low);noisy[valid]=(.075*focal_px/divisor)/disparity[valid]
    else:
        noisy=low+rng.normal(0,.005+.003*low**2,low.shape).astype(np.float32)
    maximum=cv2.dilate(low,np.ones((3,3),np.uint8));minimum=cv2.erode(low,np.ones((3,3),np.uint8))
    edge=(maximum-minimum)>.05
    valid &= ~edge
    valid &= rng.random(low.shape)>(.18 if profile=='oak' else .10)
    valid &= np.isfinite(noisy)&(noisy>0)
    noisy[~valid]=0
    result=cv2.resize(noisy,(w,h),interpolation=cv2.INTER_NEAREST)
    shift=int(rng.integers(-1,2)); shifted=np.zeros_like(result)
    if shift>0:shifted[:,shift:]=result[:,:-shift]
    elif shift<0:shifted[:,:shift]=result[:,-shift:]
    else:shifted=result
    # Do not fill a pixel whose ideal depth was unknown, including after perturbation.
    shifted[~(np.isfinite(depth)&(depth>0))]=0
    return shifted,{'profile':profile,'simulated':True,'calibrated_sensor_model':False,
                    'spatial_divisor':divisor,'depth_shift_pixels_xy':[shift,0],
                    'focal_px':focal_px,'assumed_stereo_baseline_m':.075 if profile=='oak' else None,
                    'corruption_parameters':{'dropout_probability':.18 if profile=='oak' else .10,
                        'edge_rejection_m':.05,'range_m':[.4,10] if profile=='oak' else [.2,5],
                        'noise':'0.15 low-resolution disparity pixels; 1/8 px quantization' if profile=='oak' else 'sigma=0.005+0.003*Z^2 meters'},
                    'quality_policy':'binary validity, not calibrated sensor confidence',
                    'valid_fraction':float((shifted>0).mean())}


def validate_render(root, record):
    paths={k:(root/v).resolve() for k,v in record['files'].items()}
    if not all(p.is_relative_to(root.resolve()) for p in paths.values()):raise ValueError('Render path escapes root')
    rgb=cv2.imread(str(paths['rgb']));z=np.load(paths['depth'],allow_pickle=False)
    mask=np.load(paths['mask'],allow_pickle=False);valid=np.load(paths['validity'],allow_pickle=False)
    if rgb is None or list(rgb.shape[1::-1])!=record['image_size_wh']:raise ValueError('Image size mismatch')
    if z.shape!=rgb.shape[:2] or z.dtype!=np.float32 or not np.isfinite(z).all() or (z<0).any():raise ValueError('Invalid depth')
    if mask.shape!=z.shape or valid.shape!=z.shape or not np.isin(mask,[0,1]).all() or not np.isin(valid,[0,1]).all():raise ValueError('Invalid masks')
    if not np.array_equal(z>0,valid>0) or ((mask>0)&(z<=0)).any():raise ValueError('Depth/mask inconsistency')
    yy,xx=np.where(mask>0);expected=[[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]] if len(xx)>=8 else []
    if record['boxes_xyxy']!=expected:raise ValueError('Visible box does not match mask')
    if record['negative_scene'] and len(xx):raise ValueError('Negative scene contains shell pixels')
    if record['accepted'] != (record['negative_scene'] or bool(expected)):raise ValueError('Wrong acceptance state')
    if record['visible_mask_pixels']!=int(mask.sum()):raise ValueError('Wrong visibility count')
    k=np.asarray(record['K']);transform=np.asarray(record['world_to_camera_cv'])
    if k.shape!=(3,3) or transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid calibration')
    if not np.allclose(transform[:3,:3]@transform[:3,:3].T,np.eye(3),atol=1e-5):raise ValueError('Invalid camera rotation')
    if record['split']!='train' or not record['synthetic'] or record['scale_measured']:raise ValueError('Synthetic provenance lost')
    return rgb,z,mask,paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renders',type=Path,default=ROOT/'datasets/blender-shell-v1')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/turtle-real-synthetic-v1')
    args=p.parse_args()
    args.renders=args.renders.resolve();args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError('Choose a new dataset output')
    asset=json.loads((args.renders/'asset.json').read_text())
    if sha256(asset['source'])!=asset['source_sha256']:raise ValueError('Source scan changed')
    records=[json.loads(s) for s in (args.renders/'manifest.jsonl').read_text().splitlines() if s]
    completed=json.loads((args.renders/'completed.json').read_text())
    if len(records)!=completed['samples'] or len({r['sample_id'] for r in records})!=len(records):raise ValueError('Incomplete/duplicate render manifest')
    original=json.loads((ROOT/'datasets/turtle-challenge-v1/manifest.json').read_text())
    manifest=list(original);counts=Counter();sizes=Counter();simulated=[];tiles=[]
    for sub in ('images/train','labels/train','geometry'):(args.output/sub).mkdir(parents=True,exist_ok=True)
    for index,r in enumerate(records):
        rgb,z,mask,paths=validate_render(args.renders,r)
        if not r['accepted']:counts['excluded_invisible_or_tiny_fragment']+=1;continue
        name='blender_'+r['sample_id'];image=args.output/'images/train'/f'{name}.png';label=args.output/'labels/train'/f'{name}.txt'
        blur=index%7==3
        if blur:rgb=cv2.GaussianBlur(rgb,(5,5),.8)
        if not cv2.imwrite(str(image),rgb):raise OSError('Cannot save training image')
        h,w=rgb.shape[:2];lines=[]
        for x1,y1,x2,y2 in r['boxes_xyxy']:
            lines.append(f'0 {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
            area=(x2-x1)*(y2-y1);sizes['small' if area<32**2 else 'medium' if area<96**2 else 'large']+=1
        label.write_text('\n'.join(lines)+('\n' if lines else ''))
        geometry=args.output/'geometry'/name;geometry.mkdir()
        profiles={}
        for profile in ('oak','iphone'):
            degraded,meta=simulated_depth(z,profile,20261019+index,r['K'][0][0])
            np.save(geometry/f'{profile}_depth_z_m.npy',degraded)
            cv2.imwrite(str(geometry/f'{profile}_validity.png'),((degraded>0)*255).astype(np.uint8))
            profiles[profile]=meta|{'depth':str((geometry/f'{profile}_depth_z_m.npy').relative_to(ROOT))}
        write_json(geometry/'simulation.json',profiles)
        manifest.append({'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),
            'split':'train','sha256':sha256(image),'label_sha256':sha256(label),'boxes':len(lines),'width':w,'height':h,
            'source':'Blender shell-proxy simulation','synthetic':True,'group_id':asset['group_id'],
            'license':None,'rights_note':'User-provided scan; procedural materials; no external distribution permission inferred',
            'render_record':str((args.renders/'samples'/r['sample_id']/'sample.json').relative_to(ROOT)),
            'appearance_blur_sigma_pixels':.8 if blur else 0,'geometry_profiles':profiles})
        simulated.append(r);counts['synthetic_negative' if not lines else 'synthetic_positive']+=1
        if index%4==0 and len(tiles)<24:
            display=rgb.copy()
            for x1,y1,x2,y2 in r['boxes_xyxy']:cv2.rectangle(display,(x1,y1),(x2-1,y2-1),(0,210,255),2)
            display=cv2.resize(display,(240,240));cv2.putText(display,name,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(display)
    for split in ('train','val'):
        (args.output/f'{split}.txt').write_text(''.join(str(ROOT/r['image'])+'\n' for r in manifest if r['split']==split))
    (args.output/'data.yaml').write_text(yaml.safe_dump({'path':str(args.output.resolve()),'train':'train.txt','val':'val.txt','names':{0:'turtle'}}))
    write_json(args.output/'manifest.json',manifest)
    audit={'images':dict(Counter(r['split'] for r in manifest)),'synthetic_counts':dict(counts),'synthetic_box_sizes_at_640':dict(sizes),
        'source_manifest_sha256':sha256(ROOT/'datasets/turtle-challenge-v1/manifest.json'),
        'render_manifest_sha256':sha256(args.renders/'manifest.jsonl'),'synthetic_group':asset['group_id'],
        'all_scan_derivatives_train_only':True,'validation_unchanged_real_images':True,
        'depth_training_in_this_run':False,'limitations':['One incomplete shell proxy, no species diversity claim',
        f"Assumed {asset['inspection']['assumed_longest_dimension_m']} m simulation scale, not a measurement",'Stylized procedural materials and environments',
        'Depth profiles are uncalibrated simulations, not measured OAK/iPhone data',
        'Synthetic negatives do not replace real turtle-free field recordings']}
    write_json(args.output/'audit.json',audit)
    if tiles:
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.output/'contact-sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
