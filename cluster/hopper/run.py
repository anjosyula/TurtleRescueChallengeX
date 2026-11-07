"""Allocated-node preflight and dispatch. Nothing downloads or uploads datasets."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from hopper import ROOT, prepare


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--action',choices=('train','smoke','resume'),required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('A Slurm allocation is required')
    import torch
    import torchvision
    import cv2
    if importlib.metadata.version('ultralytics')!='8.4.157':raise RuntimeError('Wrong Ultralytics version')
    if torch.__version__!='2.14.0+cu126' or torchvision.__version__!='0.29.0+cu126':
        raise RuntimeError('Use the pinned Hopper CUDA environment from setup.sh')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check driver compatibility with CUDA 12.6 wheels')
    if torch.cuda.device_count()!=1:raise RuntimeError('Request exactly one GPU')
    # Exercise both forward/backward CUDA and compiled torchvision ops before a long run.
    x=torch.randn(32,32,device='cuda',requires_grad=True)
    (x@x).square().mean().backward()
    torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device='cuda'),torch.ones(1,device='cuda'),.5)
    torch.cuda.synchronize()
    config=json.loads(args.config.read_text())
    if Path(config['name']).name!=config['name'] or config['name'] in ('','.','..'):
        raise ValueError('Invalid run name')
    name=config['name']+('-smoke-'+os.environ['SLURM_JOB_ID'] if args.action=='smoke' else '')
    locks=args.scratch/'locks';locks.mkdir(parents=True,exist_ok=True)
    lock=(locks/(name+'.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Held until this process exits.
    prepared=args.scratch/'prepared'/name
    audit=prepare(ROOT,ROOT/config['manifest'],prepared)
    if config['mode']=='fusion':
        sys.path.insert(0,str(ROOT))
        from train_fusion import PairedShellDataset
        dataset=PairedShellDataset(ROOT/config['manifest'],augment=False)
        # Validate all paired input dimensions/labels, not only the first batch.
        for i in range(len(dataset)):dataset.get(i,'oak')
    evidence=args.scratch/'runtime'/os.environ['SLURM_JOB_ID']
    evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (evidence/'preflight.json').write_text(json.dumps(dict(audit=audit,gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,cuda=torch.version.cuda,opencv=cv2.__version__),indent=2)+'\n')
    mode=config['mode']
    if mode not in ('rgb','fusion'):raise ValueError('Unknown training mode')
    project=args.scratch/'runs'/mode
    common=['--name',name,'--project',str(project),'--device','cuda:0' if mode=='fusion' else '0',
            '--epochs',str(1 if args.action=='smoke' else config['epochs']),
            '--batch',str(config['batch']),'--workers',str(config['workers']),
            '--learning-rate',str(config['learning_rate'])]
    if mode=='fusion':
        cmd=[sys.executable,str(ROOT/'train_fusion.py'),*common,'--manifest',str(ROOT/config['manifest']),
             '--checkpoint',str(ROOT/config['checkpoint'])]
        if args.action=='smoke':cmd += ['--max-batches','2']
        if args.action=='resume':cmd += ['--resume']
    else:
        cmd=[sys.executable,str(ROOT/'train_challenge.py'),*common,'--variant',config['variant'],
             '--init-checkpoint',str(ROOT/config['checkpoint']),'--data',str(prepared/'data.yaml')]
        if args.action=='resume':cmd += ['--resume',str(project/name/'weights/last.pt')]
    (evidence/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('Preflight passed:',json.dumps(audit),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
