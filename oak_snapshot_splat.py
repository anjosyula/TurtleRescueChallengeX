"""A single recorded OAK RGB-D snapshot as metric Gaussians; no motion fusion or hole completion."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from gaussian_dataset import seed_cloud, render_cloud
from train_challenge import sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--sample',default='000050')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sample=json.loads((a.session/'samples'/a.sample/'sample.json').read_text())
    if sample['depth_alignment']!='rgb' or sample['depth_kind']!='camera_axis_z' or sample['depth_units']!='meters':
        raise ValueError('Requires camera-axis meters aligned to RGB')
    geometry=sample['streams']['rgb']['geometry']
    K=np.array(geometry['intrinsics'],np.float32)
    if np.any(np.abs(geometry['distortion_coefficients'])>1e-7):
        raise ValueError('This snapshot adapter requires the recorded undistorted RGB output')
    rgb=cv2.cvtColor(cv2.imread(str(a.session/sample['files']['rgb'])),cv2.COLOR_BGR2RGB)
    depth=np.load(a.session/sample['files']['depth'],allow_pickle=False)
    valid=cv2.imread(str(a.session/sample['files']['validity']),0)>0
    if depth.shape!=rgb.shape[:2] or not np.array_equal(valid,depth>0):raise ValueError('Depth/validity mismatch')
    accepted=valid&np.isfinite(depth)&(depth>=.2)&(depth<=3.)
    z=np.where(accepted,depth,0).astype(np.float32)
    # Recorded intrinsics use integer pixel centers. The Metal rasterizer samples +0.5.
    raster_k=K.copy();raster_k[0,2]+=.5;raster_k[1,2]+=.5
    cloud=seed_cloud(rgb,z,np.zeros_like(valid),raster_k,stride=2)
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'measured_cloud.npz',**cloud)
    from metalsplat import GaussianModel,save_ply
    model=GaussianModel(torch.from_numpy(cloud['means']),scales=torch.from_numpy(cloud['scales']),
        colors=torch.from_numpy(cloud['colors']),opacities=torch.from_numpy(cloud['opacities']))
    save_ply(model,a.output/'oak_snapshot.ply')
    render_k=K.copy();render_k[:2]/=2;render_k[0,2]+=.5;render_k[1,2]+=.5
    w,h=rgb.shape[1]//2,rgb.shape[0]//2
    images=[Image.fromarray(rgb[::2,::2])]
    for name,dx in [('same_view',0),('shifted_2cm',.02)]:
        pose=np.eye(4,dtype=np.float32);pose[0,3]=-dx
        rendered=render_cloud(cloud,render_k,pose,w,h)
        image=Image.fromarray(rendered['rgb']);image.save(a.output/f'{name}.png');images.append(image)
        np.save(a.output/f'{name}_rendered_z_m.npy',rendered['depth'])
        np.save(a.output/f'{name}_render_support.npy',rendered['validity'])
    sheet=Image.new('RGB',(w*3,h+45),'white');draw=ImageDraw.Draw(sheet)
    for i,(im,label) in enumerate(zip(images,['Recorded RGB','Measured-depth Gaussian snapshot','Virtual camera shifted 2 cm'])):
        sheet.paste(im,(w*i,0));draw.text((w*i+10,h+12),label,fill='black')
    sheet.save(a.output/'comparison.jpg',quality=92)
    write_json(a.output/'provenance.json',dict(session=str(a.session),sample=a.sample,
        source_checksums={k:sha256(a.session/sample['files'][k]) for k in ('rgb','depth','validity')},
        calibration_sha256=sha256(a.session/'calibration.json'),sample_metadata=sample,
        gaussians=len(cloud['means']),source_valid_fraction=float(valid.mean()),accepted_fraction=float(accepted.mean()),
        measured_range_m=[.2,3.],units='meters',K_recorded=K.tolist(),K_render=render_k.tolist(),
        source_pixel_center='integer coordinates',raster_pixel_center='half-integer; principal point explicitly adjusted',
        point_sampling_stride=2,raw_confidence_used=False,unknown_depth_completed=False,
        multi_view_optimized=False,motion_fused=False,
        limitations=['Single-view observed surfaces only; not a complete turtle or scene',
                    'Gaussian surface footprints and new-view depths are rendered estimates, not additional measurements',
                    'Moving turtle and people cannot be fused as a static object without motion handling']))
    print(a.output/'comparison.jpg')


if __name__=='__main__':main()
