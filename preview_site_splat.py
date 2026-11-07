"""Render a fitted site's recorded camera path and compare photometric held-out frames."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw
import torch

from metalsplat import render
from metalsplat.export import load_ply
from metalsplat.data.colmap import load_colmap_scene
from train_challenge import write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconstruction',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True)
    a=p.parse_args()
    scene=load_colmap_scene(a.reconstruction/'undistorted',sparse_subdir='sparse',device='mps',downscale=2.)
    model=load_ply(a.fit/'garden_best.ply',device='mps')
    w,h=scene.cameras[0].img_width,scene.cameras[0].img_height
    width=w*2;height=(h+41)//2*2
    command=['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
             '-r','8','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(a.fit/'preview.mp4')]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    snapshots=[];scores=[]
    for i,camera in enumerate(scene.cameras):
        with torch.no_grad():pred=render(model,camera).clamp(0,1)
        target=scene.images[i]
        if i%8==0:scores.append(float(-10*torch.log10((pred-target).square().mean())))
        rgb=(pred.cpu().numpy()*255).astype(np.uint8)
        real=(target.cpu().numpy()*255).astype(np.uint8)
        canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(real),(0,0));canvas.paste(Image.fromarray(rgb),(w,0))
        draw.text((8,h+5),'Recorded frame',fill='black')
        draw.text((w+8,h+5),'Fitted 3D Gaussian render',fill='black')
        draw.text((8,h+21),f'{scene.image_names[i]} | rough reconstruction; scale unknown',fill='black')
        process.stdin.write(np.asarray(canvas).tobytes())
        if i in (0,16,32):snapshots.append(canvas)
    process.stdin.close()
    if process.wait()!=0:raise RuntimeError('Preview encoding failed')
    sheet=Image.new('RGB',(width,height*len(snapshots)),'white')
    for i,canvas in enumerate(snapshots):sheet.paste(canvas,(0,i*height))
    sheet.save(a.fit/'comparison.jpg',quality=92)
    snapshots[0].save(a.fit/'preview.jpg',quality=92)
    write_json(a.fit/'preview-validation.json',dict(gaussians=model.num_points,frames=len(scene.cameras),
        photometric_holdout_psnr_after_ply_reload_db=scores,mean_psnr_db=sum(scores)/len(scores),
        ply_sha256=sha256(a.fit/'garden_best.ply'),video_sha256=sha256(a.fit/'preview.mp4'),
        view_path='Recovered recorded camera poses; not an independently captured novel-view test',metric_scale=None))


if __name__=='__main__':main()
