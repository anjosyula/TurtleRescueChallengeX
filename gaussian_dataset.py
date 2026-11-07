"""Generate training-only RGB-depth views by Metal 3D Gaussian splatting of labeled RGB-D renders.

This is geometry-initialized splat augmentation, not a fitted multiview reconstruction.
No unlabeled real capture is assigned synthetic ground truth.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
import yaml

from train_challenge import ROOT, sha256, write_json
from synthetic_dataset import simulated_depth, validate_render

METALSPLAT_COMMIT='84e8c3a433f71c107e9387d3f72a765436c11f6e'


def seed_cloud(rgb, depth, mask, K, stride=2):
    """Backproject source pixel centers into its CV camera frame, without depth completion."""
    h,w=depth.shape;K=np.asarray(K,dtype=np.float32)
    yy,xx=np.mgrid[0:h:stride,0:w:stride]
    z=depth[yy,xx];good=np.isfinite(z)&(z>0)
    yy,xx,z=yy[good],xx[good],z[good]
    rays=np.linalg.solve(K,np.stack((xx+.5,yy+.5,np.ones(len(xx)))))
    means=(rays*z).T.astype(np.float32)
    # Small camera-facing ellipsoids; their thin depth extent does not fill unseen surfaces.
    sx=z/K[0,0]*stride*.8;sy=z/K[1,1]*stride*.8
    scales=np.stack((sx,sy,np.minimum(sx,sy)*.1),1).astype(np.float32)
    quats=np.zeros((len(z),4),np.float32);quats[:,0]=1
    return {'means':means,'scales':scales,'quats':quats,
        'opacities':np.full(len(z),.98,np.float32),
        'colors':rgb[yy,xx].astype(np.float32)/255,
        'shell':mask[yy,xx].astype(np.float32)}


def camera_delta(index, median_depth):
    """Small, deterministic camera poses; translations are in source-camera meters."""
    angles=[(-.65,.3),(.65,-.3),(-.3,-.65),(.3,.65)]
    yaw,pitch=np.deg2rad(angles[index%4])
    ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    delta=np.eye(4,dtype=np.float32);delta[:3,:3]=rx@ry
    delta[:3,3]=np.array([(-1 if index%2 else 1)*.002,(-1 if index//2 else 1)*.001,0])*median_depth
    return delta


def render_cloud(cloud,K,pose,width,height,alpha_min=.95,depth_cv_max=.02):
    """Same alpha-composited geometry produces RGB, visibility, depth first/second moments.

    The pinned Metal kernel returns an accumulated depth SUM, despite an ambiguous
    'expected depth' docstring. Normalize exactly once by alpha (verified by fixture).
    """
    from metalsplat.ops.project import project_gaussians
    from metalsplat.ops.rasterize import rasterize_gaussians
    t=lambda x:torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device='mps')
    K=np.asarray(K);pose=np.asarray(pose)
    with torch.no_grad():
        means=t(cloud['means']);scales=t(cloud['scales']);quats=t(cloud['quats'])
        projected=project_gaussians(means,scales,quats,t(pose[:3,:3]),t(pose[:3,3]),
            float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]),width,height,near=.01,eps2d=.3)
        xy,z,conic,radii,valid,compensation=projected
        opacity=t(cloud['opacities'])*compensation
        def raster(colors,background):
            return rasterize_gaussians(xy,z,conic,opacity,colors,radii,valid,width,height,
                background=t(background),return_aux=True)
        rgb,z_sum,transmittance=raster(t(cloud['colors']),[114/255]*3)
        # Unbounded feature channels are supported by the inspected kernel; no sigmoid or clamp.
        features=torch.stack((t(cloud['shell']),z,z*z),dim=1).contiguous()
        moments,_,feature_T=raster(features,[0,0,0])
        if not torch.allclose(feature_T,transmittance,atol=1e-6,rtol=0):
            raise AssertionError('Semantic and RGB passes differ in visibility')
        rgb=rgb.cpu().numpy();z_sum=z_sum.cpu().numpy()
        alpha=(1-transmittance).cpu().numpy();moments=moments.cpu().numpy()
    denominator=np.maximum(alpha,1e-8)
    z=z_sum/denominator
    if not np.allclose(moments[:,:,1],z_sum,atol=1e-4,rtol=1e-5):raise AssertionError('Depth channel mismatch')
    variance=np.maximum(moments[:,:,2]/denominator-z*z,0)
    std=np.sqrt(variance)
    valid=(alpha>=alpha_min)&np.isfinite(z)&(z>0)&(std<=np.maximum(.005,depth_cv_max*z))
    # RGB visibility labels are independent of depth reliability: missing depth never removes a turtle label.
    shell=((moments[:,:,0]/denominator)>=.5)&(alpha>=alpha_min)
    return {'rgb':np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),
            'depth':np.where(valid,z,0).astype(np.float32),'validity':valid.astype(np.uint8),
            'mask':shell.astype(np.uint8),'alpha':alpha.astype(np.float32),
            'depth_std':std.astype(np.float32)}


def visible_boxes(mask):
    yy,xx=np.where(mask>0)
    return [[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]] if len(xx)>=8 else []


def select_sources(records,count):
    good=[r for r in records if r['accepted']]
    positive=[r for r in good if not r['negative_scene']];negative=[r for r in good if r['negative_scene']]
    nneg=max(1,round(count/8));npos=count-nneg
    if len(positive)<npos or len(negative)<nneg:raise ValueError('Insufficient distinct labeled sources')
    selected=[group[i] for group,n in [(positive,npos),(negative,nneg)]
              for i in np.linspace(0,len(group)-1,n,dtype=int)]
    return sorted(selected,key=lambda r:r['sample_id'])


def write_sample(output, name, result, source, pose, source_checksums, alpha_min):
    sample=output/'samples'/name;sample.mkdir(parents=True)
    h,w=result['mask'].shape;boxes=visible_boxes(result['mask'])
    image=output/'images/train'/f'{name}.png';label=output/'labels/train'/f'{name}.txt'
    if not cv2.imwrite(str(image),cv2.cvtColor(result['rgb'],cv2.COLOR_RGB2BGR)):raise OSError(image)
    label.write_text(''.join(f'0 {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}\n'
                              for x1,y1,x2,y2 in boxes))
    for key in ('depth','validity','mask'):np.save(sample/f'{key}.npy',result[key])
    np.savez_compressed(sample/'render_quality.npz',alpha=result['alpha'],depth_std_m=result['depth_std'])
    profiles={}
    for profile in ('oak','iphone'):
        degraded,meta=simulated_depth(result['depth'],profile,20261026+int(source['sample_id'])*100+int(name[-2:]),source['K'][0][0])
        path=sample/f'{profile}_depth_z_m.npy';np.save(path,degraded)
        profiles[profile]=meta|{'depth':str(path.relative_to(ROOT)),
            'parent_depth_kind':'Gaussian alpha-normalized camera-axis Z; ambiguous pixels invalidated'}
    write_json(sample/'simulation.json',profiles)
    metadata={'sample_id':name,'source_sample_id':source['sample_id'],'synthetic':True,'split':'train',
        'group_id':source['group_id'],'renderer':'MetalSplat 3DGS','gaussians_optimized':False,
        'depth_kind':'synthetic Gaussian expected camera_axis_z','depth_units':'meters',
        'quality_policy':f'alpha >= {alpha_min}, depth std <= max(0.005 m, 2% Z); binary validity, not sensor confidence',
        'K':source['K'],'world_to_camera_cv':(pose@np.asarray(source['world_to_camera_cv'])).tolist(),
        'source_camera_to_target_camera':pose.tolist(),'source_checksums':source_checksums,
        'image_size_wh':[w,h],'boxes_xyxy':boxes,'negative_scene':source['negative_scene'],
        'label_semantics':'visible source-shell contribution >=50% at supported RGB pixels; exclusive xyxy',
        'mask_pixels':int(result['mask'].sum()),'rgb_supported_fraction':float((result['alpha']>=alpha_min).mean()),
        'depth_valid_fraction':float(result['validity'].mean()),'scale_measured':False,
        'assumed_shell_length_m':source['assumed_shell_length_m'],
        'files':{'rgb':str(image.relative_to(output)),'depth':str((sample/'depth.npy').relative_to(output)),
                 'mask':str((sample/'mask.npy').relative_to(output)),'validity':str((sample/'validity.npy').relative_to(output))}}
    write_json(sample/'sample.json',metadata)
    return {'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),
        'sha256':sha256(image),'label_sha256':sha256(label),'width':w,'height':h,'boxes':len(boxes),
        'split':'train','synthetic':True,'source':'RGB-D initialized Gaussian splat augmentation of Blender scene',
        'group_id':source['group_id'],'source_sample_id':source['sample_id'],
        'render_record':str((sample/'sample.json').relative_to(ROOT)),'geometry_profiles':profiles,
        'license':None,'rights_note':'User-provided scan and procedural scene; no new external image assets'},metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'datasets/blender-shell-v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-count',type=int,default=24)
    p.add_argument('--views',type=int,default=4)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--min-free-gb',type=float,default=1.5)
    args=p.parse_args();args.output=args.output.resolve();args.source=args.source.resolve()
    if args.source_count<2 or not 1<=args.views<=4 or args.stride not in (1,2):p.error('Invalid generation settings')
    if not args.output.is_relative_to(ROOT):p.error('Output must be inside this workspace')
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    if not torch.backends.mps.is_available():raise RuntimeError('This pinned backend requires Apple Silicon MPS')
    if importlib.metadata.version('metalsplat')!='0.2.0':raise RuntimeError('Use requirements-gaussian.txt')
    dist=importlib.metadata.distribution('metalsplat');origin=json.loads(dist.read_text('direct_url.json') or '{}')
    if origin.get('vcs_info',{}).get('commit_id')!=METALSPLAT_COMMIT:raise RuntimeError('MetalSplat commit differs from tested pin')
    records=[json.loads(s) for s in (args.source/'manifest.jsonl').read_text().splitlines() if s.strip()]
    selected=select_sources(records,args.source_count)
    if any(s['image_size_wh']!=[640,640] for s in selected):
        raise ValueError('This pipeline requires 640-square source renders; do not silently change camera intrinsics')
    needed=args.source_count*args.views*9_000_000+args.source_count*5_000_000
    if shutil.disk_usage(ROOT).free<needed+args.min_free_gb*1e9:raise OSError('Insufficient disk space for requested batch plus reserve')
    for sub in ('images/train','labels/train','samples','splats','source'):(args.output/sub).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'datasets/turtle-real-synthetic-v1/manifest.json').read_text())
    # Existing real split and Blender derivatives are referenced, not copied or re-split.
    original=list(manifest);stats=[];rejected=[];tiles=[];started=time.perf_counter()
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'backend_commit':METALSPLAT_COMMIT,
        'versions':{n:importlib.metadata.version(n) for n in ('metalsplat','torch','numpy','opencv-python')},
        'source_manifest_sha256':sha256(args.source/'manifest.jsonl'),'base_manifest_sha256':sha256(ROOT/'datasets/turtle-real-synthetic-v1/manifest.json'),
        'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'initialization':'one 3D Gaussian per valid sampled RGB-D pixel; camera-facing anisotropic scale; no optimization',
        'pixel_centers':'u+0.5,v+0.5; matches Metal rasterizer and Blender camera',
        'view_restriction':'small local transforms; unseen backsides are not reconstructed',
        'color_space':'display RGB from source render, no relighting or SH view dependence',
        'label_policy':'source masks carried through visibility-aware alpha compositing; never detector pseudo-labels',
        'depth_policy':'rendered camera-axis expected depth, normalized once; mixed surfaces/low alpha remain unknown',
        'uploads':False,'training_started':False,'source_group_split':'all scan derivatives train-only',
        'limitations':['Not a multiview-fitted 3DGS reconstruction','No new turtle shape/species or independent observations',
        'Original 20 cm scale is assumed','Source shading baked in; local-view augmentation only',
        'Synthetic depth and OAK/iPhone degradation are not measured or calibrated sensor data']}
    write_json(args.output/'provenance.json',provenance)
    for filename in ('gaussian_dataset.py','synthetic_dataset.py'):
        shutil.copy2(ROOT/filename,args.output/'source'/filename)
    for s in selected:
        bgr,z,mask,paths=validate_render(args.source,s)
        checksums={k:sha256(path) for k,path in paths.items()}
        cloud=seed_cloud(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),z,mask,s['K'],args.stride)
        np.savez_compressed(args.output/'splats'/f"{s['sample_id']}.npz",**cloud)
        # One portable standard 3DGS PLY example; all sources retain lossless primitive arrays.
        if s['sample_id']==selected[0]['sample_id']:
            from metalsplat import GaussianModel,save_ply
            model=GaussianModel(**{k:torch.from_numpy(v) for k,v in cloud.items() if k!='shell'})
            save_ply(model,args.output/'splats/example.ply')
        for view in range(args.views):
            if shutil.disk_usage(ROOT).free<args.min_free_gb*1e9:raise OSError('Free-space reserve reached; partial generation retained')
            name=f"splat_{s['sample_id']}_{view:02d}"
            pose=camera_delta(view,float(np.median(z[z>0])))
            r=render_cloud(cloud,s['K'],pose,640,640)
            support=float((r['alpha']>=.95).mean());boxes=visible_boxes(r['mask'])
            if support<.95 or (not s['negative_scene'] and not boxes) or (s['negative_scene'] and boxes):
                rejected.append({'name':name,'rgb_support':support,'boxes':boxes});continue
            row,meta=write_sample(args.output,name,r,s,pose,checksums,.95)
            manifest.append(row);stats.append(meta)
            if len(tiles)<24 and view==0:
                display=cv2.cvtColor(r['rgb'],cv2.COLOR_RGB2BGR)
                for x1,y1,x2,y2 in boxes:cv2.rectangle(display,(x1,y1),(x2-1,y2-1),(0,220,255),2)
                display=cv2.resize(display,(240,240));cv2.putText(display,name,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                tiles.append(display)
            write_json(args.output/'progress.json',{'status':'generating','accepted':len(stats),'rejected':len(rejected),'last':name})
        print(f"Source {s['sample_id']}: {len(cloud['means'])} splats; {len(stats)} accepted / {len(rejected)} rejected",flush=True)
        torch.mps.empty_cache()
    if not stats:raise RuntimeError('No acceptable splat views; inspect source support')
    write_json(args.output/'manifest.json',manifest)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in stats))
    for split in ('train','val'):
        (args.output/f'{split}.txt').write_text(''.join(str(ROOT/r['image'])+'\n' for r in manifest if r['split']==split))
    (args.output/'data.yaml').write_text(yaml.safe_dump({'path':str(args.output),'train':'train.txt','val':'val.txt','names':{0:'turtle'}}))
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(args.output/'contact-sheet.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    audit={'images':dict(Counter(r['split'] for r in manifest)),
        'base_images':len(original),'gaussian_views':len(stats),'distinct_gaussian_source_scenes':len(selected),
        'positive_views':sum(bool(s['boxes_xyxy']) for s in stats),'negative_views':sum(not s['boxes_xyxy'] for s in stats),
        'rejected':rejected,'all_derivatives_train_only':True,'validation_unchanged':True,
        'rgb_supported_fraction_min':min(s['rgb_supported_fraction'] for s in stats),
        'mean_valid_depth_fraction':float(np.mean([s['depth_valid_fraction'] for s in stats])),
        'elapsed_seconds':time.perf_counter()-started,'training_started':False,
        'depth_is_measured':False,'depth_kind':'Gaussian expected camera-axis Z, simulated',
        'limitations':provenance['limitations']}
    write_json(args.output/'audit.json',audit);write_json(args.output/'completed.json',audit)
    write_json(args.output/'progress.json',{'status':'completed',**audit})
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
