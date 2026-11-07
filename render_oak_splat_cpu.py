"""Render a saved measured OAK Gaussian snapshot without a Metal device."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from train_challenge import sha256, write_json

CPP = r'''
#include <cmath>
#include <algorithm>
extern "C" void splat(int W,int H,int N,const float* g,const float* color,float* out,float* trans){
 for(int i=0;i<N;i++){
  const float* p=g+7*i; float x=p[0],y=p[1],A=p[2],B=p[3],C=p[4],r=p[5],opacity=p[6];
  for(int v=std::max(0,(int)std::floor(y-r));v<=std::min(H-1,(int)std::ceil(y+r));v++){
   for(int u=std::max(0,(int)std::floor(x-r));u<=std::min(W-1,(int)std::ceil(x+r));u++){
    int j=v*W+u; if(trans[j]<.001f)continue;
    float dx=u+.5f-x,dy=v+.5f-y,q=A*dx*dx+2*B*dx*dy+C*dy*dy;
    if(q>9.f)continue;float a=std::min(.99f,opacity*std::exp(-.5f*q)),w=trans[j]*a;
    for(int c=0;c<3;c++)out[3*j+c]+=w*color[3*i+c];trans[j]*=1-a;
   }
  }
 }
}
'''


def render(cloud, K, R, T, width, height, kernel):
    means=cloud['means'];scales=cloud['scales']
    q=means@R.T+T;z=q[:,2]
    valid=z>.01
    q=q[valid];z=z[valid];scales=scales[valid]
    uv=(q@K.T)[:,:2]/z[:,None]
    J=np.zeros((len(z),2,3),np.float32)
    J[:,0,0]=K[0,0]/z;J[:,1,1]=K[1,1]/z
    J[:,0,2]=-K[0,0]*q[:,0]/z**2;J[:,1,2]=-K[1,1]*q[:,1]/z**2
    # Saved snapshot quaternions are identity; preserve anisotropic scales.
    covariance=np.einsum('ai,ni,bi->nab',R,scales**2,R)
    projected=np.einsum('nai,nij,nbj->nab',J,covariance,J)
    projected+=np.eye(2,dtype=np.float32)[None]*.3
    conic=np.linalg.inv(projected)
    radius=3*np.sqrt(np.linalg.eigvalsh(projected)[:,1])
    keep=(radius<100)&(uv[:,0]+radius>=0)&(uv[:,0]-radius<width)&(uv[:,1]+radius>=0)&(uv[:,1]-radius<height)
    order=np.flatnonzero(keep);order=order[np.argsort(z[order])]
    g=np.ascontiguousarray(np.column_stack((uv,conic[:,0,0],conic[:,0,1],conic[:,1,1],radius,cloud['opacities'][valid]))[order],dtype=np.float32)
    colors=np.ascontiguousarray(cloud['colors'][valid][order],dtype=np.float32)
    out=np.zeros((height,width,3),np.float32);trans=np.ones((height,width),np.float32)
    kernel(width,height,len(order),*[x.ctypes.data for x in (g,colors,out,trans)])
    return np.rint(np.clip(out,0,1)*255).astype(np.uint8),1-trans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--sample',default='000085')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    row=json.loads((args.session/'samples'/args.sample/'sample.json').read_text())
    k=np.array(row['streams']['rgb']['geometry']['intrinsics'],np.float32)
    rgb=np.asarray(Image.open(args.session/row['files']['rgb']).convert('RGB'))
    if row['depth_alignment']!='rgb' or row['depth_kind']!='camera_axis_z' or row['depth_units']!='meters':
        raise ValueError('Requires measured camera-axis depth aligned to RGB')
    if np.any(np.abs(row['streams']['rgb']['geometry']['distortion_coefficients'])>1e-7):
        raise ValueError('Requires undistorted recorded RGB')
    provenance_path=out/'provenance.json'
    if provenance_path.exists():
        previous=json.loads(provenance_path.read_text())
        if previous['sample']!=args.sample or previous['source_checksums']['rgb']!=sha256(args.session/row['files']['rgb']):
            raise ValueError('Use a new output directory for a different source')
    if not (out/'measured_cloud.npz').exists():
        import torch
        from gaussian_dataset import seed_cloud
        from metalsplat import GaussianModel,save_ply
        depth=np.load(args.session/row['files']['depth'],allow_pickle=False)
        valid=np.asarray(Image.open(args.session/row['files']['validity']))>0
        if depth.shape!=rgb.shape[:2] or not np.array_equal(valid,depth>0):
            raise ValueError('Depth/validity mismatch')
        z=np.where(valid&np.isfinite(depth)&(depth>=.2)&(depth<=3),depth,0).astype(np.float32)
        seed_k=k.copy();seed_k[0,2]+=.5;seed_k[1,2]+=.5
        cloud=seed_cloud(rgb,z,np.zeros_like(valid),seed_k,stride=2)
        np.savez_compressed(out/'measured_cloud.npz',**cloud)
        model=GaussianModel(torch.from_numpy(cloud['means']),scales=torch.from_numpy(cloud['scales']),
            colors=torch.from_numpy(cloud['colors']),opacities=torch.from_numpy(cloud['opacities']))
        save_ply(model,out/'oak_snapshot.ply')
    cloud=dict(np.load(out/'measured_cloud.npz'))
    if not np.allclose(cloud['quats'],np.array([1,0,0,0])):raise ValueError('Expected camera-facing snapshot Gaussians')
    source=out/'raster.cpp';source.write_text(CPP)
    library=out/'raster.dylib'
    subprocess.run(['clang++','-O3','-dynamiclib',str(source),'-o',str(library)],check=True)
    kernel=ctypes.CDLL(str(library.resolve())).splat
    kernel.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*4
    kernel.restype=None
    w,h=640,480
    k[:2]/=2;k[0,2]+=.5;k[1,2]+=.5
    rgb_image=Image.fromarray(rgb).resize((w,h),Image.Resampling.LANCZOS)
    rgb_image.save(out/'recorded.jpg',quality=94)
    images=[]
    for name,dx in [('same_view',0),('shifted_2cm',.02)]:
        image,support=render(cloud,k,np.eye(3),np.array([-dx,0,0]),w,h,kernel)
        Image.fromarray(image).save(out/(name+'.png'))
        np.save(out/(name+'_alpha.npy'),support)
        images.append(Image.fromarray(image))
    with (out/'oak_snapshot.ply').open('rb') as file:
        header=file.read(1000).split(b'end_header')[0].decode()
    count=int(next(line.split()[-1] for line in header.splitlines() if line.startswith('element vertex')))
    if count!=len(cloud['means']):raise AssertionError('PLY/cloud vertex count mismatch')
    frame_dir=out/'orbit';frame_dir.mkdir(exist_ok=True)
    for i,dx in enumerate(.025*np.sin(np.linspace(0,2*np.pi,48,endpoint=False))):
        image,_=render(cloud,k,np.eye(3),np.array([-dx,0,0]),w,h,kernel)
        Image.fromarray(image).save(frame_dir/f'{i:04d}.jpg',quality=92)
    subprocess.run(['ffmpeg','-v','error','-y','-framerate','12','-i',str(frame_dir/'%04d.jpg'),
        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'fence-splat.mp4')],check=True)
    write_json(out/'provenance.json',dict(session=str(args.session),sample=args.sample,gaussians=count,
        source_checksums={key:sha256(args.session/row['files'][key]) for key in ('rgb','depth','validity')},
        calibration_sha256=sha256(args.session/'calibration.json'),sample_metadata=row,
        representation='3D Gaussians backprojected from one aligned RGB-depth frame',
        renderer='CPU covariance projection and front-to-back alpha compositing',
        geometry_units='meters',accepted_depth_m=[.2,3.],sampling_stride=2,opacity=.98,
        novel_camera_translation_m=.025,source_pixel_center='integer; raster K shifted by 0.5',
        source_valid_fraction=row['depth_valid_fraction'],unknown_depth_completed=False,
        multi_view_optimized=False,scene='Construction perimeter fence with fabric screen and rocks',
        limitations=['Single-view visible surfaces only','Blank areas have no accepted measured depth',
                    'No claim that this fence is a silt fence','Rendered novel views are estimates, not measurements']))
    print(json.dumps({'gaussians':count,'preview':str(out/'shifted_2cm.png'),'video':str(out/'fence-splat.mp4')}))


if __name__=='__main__':main()
