"""Local stage-one fusion training: frozen RGB detector, supervised simulated depth."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_challenge import ROOT, local_runtime, sha256, write_json
from turtle_geometry import prepare_rgb_geometry
from turtle_fusion import RGBDepthP2


def shift_geometry(geometry, dx, dy):
    """Perturb alignment without wrapping measurements across frame boundaries."""
    out = np.zeros_like(geometry)
    _, h, w = geometry.shape
    if abs(dx) < w and abs(dy) < h:
        out[:, max(0,dy):min(h,h+dy), max(0,dx):min(w,w+dx)] = geometry[
            :, max(0,-dy):min(h,h-dy), max(0,-dx):min(w,w-dx)]
    return out


def flip_pair(image, geometry, labels):
    labels = labels.copy()
    labels[:, 1] = 1 - labels[:, 1]
    return image[:, :, ::-1].copy(), geometry[:, :, ::-1].copy(), labels


class PairedShellDataset(Dataset):
    def __init__(self, manifest, seed=42, augment=True, dropout=.2, holes=.3, shift=.2):
        self.rows = [r for r in json.loads(Path(manifest).read_text()) if r.get('synthetic')]
        if not self.rows or any(r['split'] != 'train' for r in self.rows):
            raise ValueError('All derivatives of this scan must remain training-only')
        self.seed, self.epoch, self.augment = seed, 0, augment
        self.dropout, self.holes, self.shift = dropout, holes, shift
        self.hashes = {}
        for row in self.rows:
            for key, digest in [('image','sha256'), ('label','label_sha256')]:
                path = ROOT / row[key]
                if sha256(path) != row[digest]:
                    raise ValueError(f'Changed dataset file: {path}')
                self.hashes[str(path.relative_to(ROOT))] = row[digest]
            for profile in ('oak', 'iphone'):
                record = row['geometry_profiles'][profile]
                if not record['simulated'] or record['calibrated_sensor_model']:
                    raise ValueError('This adapter requires explicitly uncalibrated simulation')
                path = ROOT / record['depth']
                depth = np.load(path, allow_pickle=False)
                if depth.dtype != np.float32 or depth.shape != (row['height'], row['width']):
                    raise ValueError(f'Invalid depth shape/dtype: {path}')
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError(f'Invalid stored simulation depth: {path}')
                self.hashes[str(path.relative_to(ROOT))] = sha256(path)

    def __len__(self):
        return len(self.rows)

    def get(self, index, profile):
        row = self.rows[index]
        rgb = cv2.cvtColor(cv2.imread(str(ROOT / row['image'])), cv2.COLOR_BGR2RGB)
        depth = np.load(ROOT / row['geometry_profiles'][profile]['depth'], allow_pickle=False)
        image, geometry, transform = prepare_rgb_geometry(rgb, depth, depth > 0)
        text = (ROOT / row['label']).read_text().strip()
        labels = np.asarray([list(map(float,s.split())) for s in text.splitlines()], np.float32).reshape(-1,5)
        # Current renders are square 640. Fail instead of silently scaling labels incorrectly.
        if transform.source_wh != (640,640) or transform.scale != 1:
            raise ValueError('Paired-shell labels require original 640-square renders')
        return image, geometry, labels

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index)
        # Exactly balanced profiles each epoch; each scene alternates sensor profile.
        profile = ('oak','iphone')[(index+self.epoch)%2]
        image, geometry, labels = self.get(index, profile)
        if self.augment:
            if rng.random() < .5:
                image, geometry, labels = flip_pair(image,geometry,labels)
            image = np.clip(image * rng.uniform(.8,1.2) + rng.uniform(-.04,.04),0,1)
            if rng.random() < self.holes:
                x,y = rng.integers(0,512,size=2); w,h = rng.integers(32,129,size=2)
                geometry[:, y:y+h, x:x+w] = 0
            if rng.random() < self.shift:
                dx,dy = rng.integers(-3,4,size=2)
                geometry = shift_geometry(geometry,int(dx),int(dy))
            if rng.random() < self.dropout:
                geometry.fill(0)
        return torch.from_numpy(image), torch.from_numpy(geometry), torch.from_numpy(labels)


def collate_pairs(samples):
    images, geometries, labels = zip(*samples)
    return {'img':torch.stack(images), 'geometry':torch.stack(geometries),
            'cls':torch.cat([r[:,:1] for r in labels]),
            'bboxes':torch.cat([r[:,1:] for r in labels]),
            'batch_idx':torch.cat([torch.full((len(r),),i,dtype=torch.float32) for i,r in enumerate(labels)])}


def set_fusion_train(model):
    model.train()
    model.detector.eval()  # Frozen BatchNorm statistics are part of the fallback guarantee.
    model.detector.model[-1].training = True  # Raw loss outputs; leave child BatchNorm in eval.


def load_fused(path, device='cpu'):
    from ultralytics.nn.tasks import DetectionModel
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload['format'] != 'turtle-rgb-depth-p2-v1':
        raise ValueError('Unsupported fusion checkpoint')
    detector = DetectionModel(payload['detector_yaml'],ch=3,nc=1,verbose=False)
    model = RGBDepthP2(detector)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.detector.model[-1].end2end = False
    return model.float().to(device).eval(), payload


def save_training_state(path, model, optimizer, scheduler, generator, epoch, history,
                        provenance, initial_fusion):
    """Atomic epoch-boundary restart, separate from the deployable inference file."""
    payload = dict(format='turtle-fusion-training-v1', epoch=epoch, history=history,
        provenance=provenance, initial_fusion=initial_fusion,
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training_state(payload, model, optimizer, scheduler, generator):
    if payload['format'] != 'turtle-fusion-training-v1':
        raise ValueError('Resume requires training_state.pt, not an inference checkpoint')
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    generator.set_state(payload['loader_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['cuda_rng'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--project',type=Path,default=ROOT/'runs/fusion')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--resume',action='store_true',help='Restart this run at its last completed epoch')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/challenge-20261019/turtle.pt')
    p.add_argument('--manifest',type=Path,default=ROOT/'datasets/turtle-real-synthetic-v1/manifest.json')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='mps')
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--depth-dropout',type=float,default=.2)
    p.add_argument('--local-holes',type=float,default=.3)
    p.add_argument('--alignment-perturbation',type=float,default=.2)
    p.add_argument('--max-batches',type=int,help='Smoke test only; recorded in provenance')
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.','..') or args.workers < 0 or min(args.epochs,args.batch) < 1 or args.learning_rate <= 0:
        p.error('Invalid name or training parameters')
    if any(not 0<=x<=1 for x in (args.depth_dropout,args.local_holes,args.alignment_perturbation)):
        p.error('Augmentation probabilities must be in [0,1]')
    if args.max_batches is not None and args.max_batches < 1:p.error('max-batches must be positive')
    local_runtime()
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    if importlib.metadata.version('ultralytics') != '8.4.157':
        raise RuntimeError('Requires pinned ultralytics==8.4.157')
    torch.set_num_threads(4); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    run = args.project.resolve()/args.name
    if args.resume and not (run/'training_state.pt').is_file():
        p.error('No restart state exists for this run')
    run.mkdir(parents=True,exist_ok=args.resume)
    data = PairedShellDataset(args.manifest,dropout=args.depth_dropout,holes=args.local_holes,
                             shift=args.alignment_perturbation)
    detector = YOLO(str(args.checkpoint)).model.float()
    detector.args = get_cfg(overrides={'epochs':args.epochs,'box':7.5,'cls':.5,'dfl':1.5})
    model = RGBDepthP2(detector).to(args.device)
    for param in detector.parameters():param.requires_grad_(False)
    frozen = {k:v.detach().cpu().clone() for k,v in detector.state_dict().items()}
    initial_fusion = {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('detector.')}
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.learning_rate*.1)
    criterion = v8DetectionLoss(detector,tal_topk=10)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(data,batch_size=args.batch,shuffle=True,num_workers=args.workers,
                        collate_fn=collate_pairs, generator=generator)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    provenance = {'created_utc':datetime.now(timezone.utc).isoformat(),'settings':settings,
        'versions':{n:importlib.metadata.version(n) for n in ('torch','ultralytics','numpy','opencv-python')},
        'initial_checkpoint_sha256':sha256(args.checkpoint),'manifest_sha256':sha256(args.manifest),
        'source_checksums':{n:sha256(ROOT/n) for n in ('train_fusion.py','turtle_fusion.py','turtle_geometry.py')},
        'paired_training_images':len(data),'paired_training_domain':'simulated only',
        'real_paired_training_images':0,'real_rgb_initialization_images':198,
        'trainable_parameters':sum(v.numel() for v in parameters),
        'frozen_detector':True,'frozen_batchnorm':True,'selection':'final fixed epoch, no paired validation available',
        'loss':'Pinned v8DetectionLoss, one-to-many TAL topk=10, YOLO26 STAL and L1 box loss; fixed RGB head',
        'loss_note':'One-to-one branch is detached from fusion and frozen. Its loss/progressive weighting is omitted; inference uses one-to-many + NMS.',
        'quality_policy':'binary validity, not calibrated confidence','normalization_m':[.2,10.],
        'paired_augmentation':'shared horizontal flip; image-only brightness; complete depth dropout, local holes, no-wrap geometry shifts; no mosaic/mixup',
        'uploads':False,'limitations':['Single scan, all derivatives train-only; no independent paired validation',
        'Assumed 20 cm shell scale; stylized proxy with no head/limbs','Sensor degradation is simulated, not calibrated',
        'Real OAK recordings unreviewed and excluded from training','Not field validated or deployed']}
    history=[]; start_epoch=0
    if args.resume:
        restart=torch.load(run/'training_state.pt',map_location='cpu',weights_only=True)
        old=restart['provenance']
        for key in ('initial_checkpoint_sha256','manifest_sha256'):
            if old[key]!=provenance[key]:raise ValueError(f'Resume input changed: {key}')
        if old['source_checksums']!=provenance['source_checksums']:
            raise ValueError('Resume source code changed')
        if old['versions']!=provenance['versions']:
            raise ValueError('Resume training environment changed')
        for key in ('epochs','batch','learning_rate','depth_dropout','local_holes','alignment_perturbation','max_batches'):
            if old['settings'][key]!=settings[key]:raise ValueError(f'Resume recipe changed: {key}')
        if json.loads((run/'input_checksums.json').read_text())!=data.hashes:
            raise ValueError('Resume dataset contents changed')
        restore_training_state(restart,model,optimizer,scheduler,generator)
        start_epoch=restart['epoch'];history=restart['history']
        initial_fusion=restart['initial_fusion'];provenance=old
        if (run/'completed.json').exists():raise ValueError('This run is already complete; choose a new name')
    write_json(run/'provenance.json',provenance)
    write_json(run/'input_checksums.json',data.hashes)
    write_json(run/'dataset_manifest.json',data.rows)
    (run/'source').mkdir(exist_ok=args.resume)
    for name in ('train_fusion.py','turtle_fusion.py','turtle_geometry.py'):
        if not args.resume:shutil.copy2(ROOT/name,run/'source'/name)
    started=time.perf_counter()
    for epoch in range(start_epoch,args.epochs):
        data.epoch=epoch;set_fusion_train(model);total=0.;count=0;components=np.zeros(3);nonzero=0
        for step,batch in enumerate(loader):
            batch={k:v.to(args.device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            predictions=model(batch['img'],batch['geometry'])
            loss,items=criterion.loss(predictions['one2many'],batch)
            objective=loss.sum()/len(batch['img'])
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite training loss')
            objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            optimizer.step()
            total+=float(objective.detach());count+=1
            components+=np.array([float(v) for v in items.values()])
            nonzero+=int((batch['geometry'][:,1].sum((1,2))>0).sum())
            write_json(run/'progress.json',{'status':'training','epoch':epoch+1,'epochs':args.epochs,
                'batch':step+1,'batches':min(len(loader),args.max_batches or len(loader)),
                'mean_loss':total/count,'elapsed_seconds':time.perf_counter()-started})
            if args.max_batches and step+1>=args.max_batches:break
        row={'epoch':epoch+1,'loss':total/count,'loss_components':dict(zip(criterion.loss_names,(components/count).tolist())),
             'learning_rate':optimizer.param_groups[0]['lr'],'last_gradient_norm':float(norm),
             'nonzero_geometry_samples':nonzero,'seconds_elapsed':time.perf_counter()-started}
        history.append(row);write_json(run/'history.json',history)
        print(json.dumps(row),flush=True);scheduler.step()
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        payload={'format':'turtle-rgb-depth-p2-v1','detector_yaml':detector.yaml,'state_dict':state,
                 'epoch':epoch+1,'provenance':provenance}
        torch.save(payload,run/'last.tmp');(run/'last.tmp').replace(run/'turtle_fused.pt')
        save_training_state(run/'training_state.pt',model,optimizer,scheduler,generator,
                            epoch+1,history,provenance,initial_fusion)
    if any(not torch.equal(v,detector.state_dict()[k].cpu()) for k,v in frozen.items()):
        raise AssertionError('Frozen RGB weights or buffers changed')
    deltas={k:float((model.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in initial_fusion.items()}
    if not any(v>0 for k,v in deltas.items() if k.startswith('encoder.')):
        raise AssertionError('Depth encoder did not learn; inspect gradient path')
    completed={'status':'completed','epochs':args.epochs,'checkpoint':str(run/'turtle_fused.pt'),
        'checkpoint_sha256':sha256(run/'turtle_fused.pt'),'frozen_detector_exactly_unchanged':True,
        'fusion_parameter_max_updates':deltas,'seconds':time.perf_counter()-started}
    write_json(run/'completed.json',completed);write_json(run/'progress.json',completed)
    print('Completed:',run/'turtle_fused.pt',flush=True)


if __name__=='__main__':main()
