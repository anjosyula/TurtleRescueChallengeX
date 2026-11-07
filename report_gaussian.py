"""Static visual QA: original render, two Gaussian viewpoints, depth and visibility."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_challenge import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/turtle-gaussian-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/gaussian-20261026/render-qa.png')
    args=parser.parse_args()
    fig,axs=plt.subplots(2,5,figsize=(16,6),layout='constrained')
    for j,sid in enumerate(('000004','000080')):
        source=ROOT/'datasets/blender-shell-v1/samples'/sid
        sample=args.dataset/'samples'/f'splat_{sid}_00'
        paths=[source/'rgb.png',args.dataset/'images/train'/f'splat_{sid}_00.png',
               args.dataset/'images/train'/f'splat_{sid}_01.png']
        for i,path in enumerate(paths):
            axs[j,i].imshow(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB))
        z=np.load(sample/'depth.npy')
        depth_plot=axs[j,3].imshow(np.ma.masked_equal(z,0),cmap='viridis')
        fig.colorbar(depth_plot,ax=axs[j,3],fraction=.045,pad=.02,label='meters')
        valid=np.load(sample/'validity.npy');mask=np.load(sample/'mask.npy')
        overlay=np.stack((mask,valid*(1-mask),mask),axis=2)*255
        axs[j,4].imshow(overlay)
        for i,title in enumerate(('Blender source','Gaussian view A','Gaussian view B',
                                 'Rendered Z (white = unknown)','Shell: magenta / valid depth: green')):
            axs[j,i].set_title(title,fontsize=10);axs[j,i].axis('off')
    fig.suptitle('Gaussian-splat augmentation: nearby views, synthetic depth, visibility-derived labels')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150);plt.close(fig)


if __name__=='__main__':main()
