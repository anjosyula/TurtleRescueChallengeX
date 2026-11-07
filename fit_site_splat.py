"""Fit a small real-scene 3D Gaussian model on Apple Metal using pinned upstream code."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time

import torch

from train_challenge import ROOT, sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconstruction',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2500)
    a=p.parse_args()
    report=json.loads((a.reconstruction/'sfm_report.json').read_text())
    if not report['screening_passed']:raise ValueError('Reconstruction failed geometry screening')
    if a.steps<500:raise ValueError('Use at least 500 optimization steps')
    a.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'vendor/metalsplat_site/train_garden.py'
    spec=importlib.util.spec_from_file_location('site_trainer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_loader=module.load_colmap_scene
    module.load_colmap_scene=lambda *args,**kwargs:original_loader(*args,sparse_subdir='sparse',**kwargs)
    settings=dict(DATA_ROOT=a.reconstruction.resolve()/'undistorted',OUT_DIR=a.output.resolve(),
        RESOLUTION_DOWNSCALE=2.,NUM_ITERS=a.steps,EVAL_EVERY=250,
        DENSIFY_START=250,DENSIFY_STOP=a.steps-500,DENSIFY_INTERVAL=100,DENSIFY_MAX_POINTS=80000,
        SEED_START=100,SEED_STOP=min(1250,a.steps//2),SEED_INTERVAL=100,
        OPACITY_RESET_INTERVAL=1000,OPACITY_RESET_STOP=1000,PRUNE_STOP=a.steps,
        SH_DEGREE=3,SH_DEGREE_INTERVAL=500)
    for key,value in settings.items():setattr(module,key,value)
    torch.set_num_threads(4);torch.manual_seed(42)
    write_json(a.output/'provenance.json',dict(source_commit='84e8c3a433f71c107e9387d3f72a765436c11f6e',
        source_sha256=sha256(source),settings={k:str(v) if isinstance(v,Path) else v for k,v in settings.items()},
        seed=42,torch=str(torch.__version__),sfm=report,
        validation='Every eighth frame withheld from photometric optimization; all frames were used in SfM. Correlated same-clip diagnostics, not independent validation.',
        metric_scale=None,measured_depth=False,
        background_seeding='Upstream photometric residual seeding borrows an estimated depth; it is a modeling hypothesis, never sensor depth',
        limitations=['Short preliminary fit','Moving excavator may produce ghosting','Unknown absolute scale','Do not treat splat surface depth as measured or physically validated']))
    shutil.copy2(source,a.output/'upstream_train_garden.py')
    shutil.copy2(ROOT/'vendor/metalsplat_site/LICENSE',a.output/'LICENSE.metalsplat')
    started=time.perf_counter()
    with (a.output/'training.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log):
        module.main()
    log=(a.output/'training.log').read_text()
    psnr=[float(v) for v in re.findall(r'eval PSNR .*?: ([\d.]+) dB',log)]
    write_json(a.output/'completed.json',dict(status='completed',seconds=time.perf_counter()-started,
        steps=a.steps,photometric_holdout_psnr_db=psnr,best_ply=str(a.output/'garden_best.ply'),
        best_ply_sha256=sha256(a.output/'garden_best.ply'),metric_scale=None,field_validated=False))
    print(a.output/'garden_best.ply')


if __name__=='__main__':main()
