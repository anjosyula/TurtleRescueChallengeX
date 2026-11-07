"""Run a candidate through COLMAP then Splatfacto; requires a separate Nerfstudio image."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--container',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=30000)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Container must have exactly one allocated CUDA GPU')
    if importlib.metadata.version('nerfstudio')!='1.1.3':raise RuntimeError('Expected Nerfstudio 1.1.3 container')
    scene=json.loads((args.scene/'scene.json').read_text())
    for row in scene['frames']:
        path=(args.scene/row['file_path']).resolve()
        if not path.is_relative_to(args.scene.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('Candidate frames changed or escape scene directory')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'input_scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    (args.output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (args.output/'container.sha256').write_text(sha(args.container)+'\n')
    processed=args.output/'processed'
    commands=[]
    def run(cmd):
        commands.append(cmd)
        (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        subprocess.run(cmd,check=True)
    run(['ns-process-data','images','--data',str(args.scene/'images'),'--output-dir',str(processed),
         '--matching-method','exhaustive'])
    poses=json.loads((processed/'transforms.json').read_text())
    registered=len(poses['frames']);fraction=registered/len(scene['frames'])
    report=dict(registered=registered,input_frames=len(scene['frames']),registered_fraction=fraction,
                gate='At least 20 registered frames and 80% registration: engineering screening threshold, not geometry validation',
                metric_scale=None,measured_depth=False)
    (args.output/'sfm_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if registered<20 or fraction<.8:
        raise RuntimeError('Insufficient registration; inspect overlap/motion before training a splat')
    run(['ns-train','splatfacto','--data',str(processed),'--output-dir',str(args.output/'models'),
         '--experiment-name','site','--vis','tensorboard','--max-num-iterations',str(args.iterations)])
    configs=list((args.output/'models').glob('site/splatfacto/*/config.yml'))
    if len(configs)!=1:raise RuntimeError('Expected one completed Nerfstudio config')
    run(['ns-export','gaussian-splat','--load-config',str(configs[0]),'--output-dir',str(args.output/'export')])
    (args.output/'completed.json').write_text(json.dumps(dict(status='completed',config=str(configs[0]),
        output=str(args.output/'export'),field_validated=False,metric_scale=None,
        note='Inspect novel views and geometry; reconstructed depth is not measured sensor depth'),indent=2)+'\n')


if __name__=='__main__':main()
