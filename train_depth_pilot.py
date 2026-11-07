"""Train a separate geometry-only P2 detector on existing simulated shell pairs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_challenge import ROOT,local_runtime,initialize_p2,sha256,write_json
from train_fusion import PairedShellDataset,collate_pairs


def load_depth(path,device='cpu'):
    from ultralytics.nn.tasks import DetectionModel
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='turtle-depth-only-p2-v1': raise ValueError('Wrong checkpoint type')
    model=DetectionModel(payload['yaml'],ch=3,nc=1,verbose=False)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.model[-1].end2end=False
    return model.float().to(device).eval(),payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='mps')
    p.add_argument('--manifest',type=Path,default=ROOT/'datasets/turtle-gaussian-v1/manifest.json')
    args=p.parse_args()
    if min(args.epochs,args.batch)<1:p.error('Positive epochs/batch required')
    local_runtime()
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    if importlib.metadata.version('ultralytics')!='8.4.157':raise RuntimeError('Pinned version required')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    data=PairedShellDataset(args.manifest,augment=True,dropout=0,holes=.3,shift=.2)
    model=initialize_p2(output).model.float().to(args.device)
    model.model[-1].end2end=False
    # Train only the one-to-many inference path; independent RGB model is untouched.
    for name,param in model.named_parameters():
        if 'one2one_' in name:param.requires_grad_(False)
    model.args=get_cfg(overrides={'epochs':args.epochs,'box':7.5,'cls':.5,'dfl':1.5})
    criterion=v8DetectionLoss(model,tal_topk=10)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00003)
    loader=DataLoader(data,batch_size=args.batch,shuffle=True,num_workers=0,collate_fn=collate_pairs,
        generator=torch.Generator().manual_seed(42))
    provenance=dict(epochs=args.epochs,batch=args.batch,seed=42,device=args.device,
        initial_sha256=sha256(ROOT/'yolo26n.pt'),manifest_sha256=sha256(args.manifest),
        versions={k:importlib.metadata.version(k) for k in ('torch','ultralytics','numpy')},
        source_sha256=sha256(__file__),samples=len(data),real_paired_training=0,
        input='Float32 normalized log camera-Z [0.2,10m], binary validity, binary validity fallback quality',
        appearance_input=False,selection='Final fixed epoch; no independent depth validation available',
        augmentation='Shared flip, geometry holes, small no-wrap alignment shifts; alternating simulated sensor profiles',
        optimizer='AdamW lr .0003 cosine to .00003; clip norm 10; one-to-many v8DetectionLoss',
        limitations=['One scan, all derivatives training-only','Synthetic sensor corruption is uncalibrated',
                    'No real paired labels; diagnostic replay is not accuracy validation'],uploads=False)
    write_json(output/'provenance.json',provenance);write_json(output/'input_checksums.json',data.hashes)
    write_json(output/'dataset_manifest.json',data.rows)
    shutil.copy2(__file__,output/Path(__file__).name)
    history=[];started=time.perf_counter()
    for epoch in range(args.epochs):
        data.epoch=epoch;model.train();total=0
        for step,batch in enumerate(loader):
            # RGB is never passed into this network, including during training.
            batch={k:v.to(args.device) for k,v in batch.items() if k!='img'}
            batch['img']=batch.pop('geometry')
            optimizer.zero_grad(set_to_none=True)
            raw=model(batch['img'])
            loss,items=criterion.loss(raw['one2many'],batch)
            objective=loss.sum()/len(batch['img'])
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite depth loss')
            objective.backward();torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True)
            optimizer.step();total+=float(objective.detach())
            write_json(output/'progress.json',dict(epoch=epoch+1,batch=step+1,batches=len(loader),loss=total/(step+1)))
        history.append(dict(epoch=epoch+1,loss=total/len(loader),seconds=time.perf_counter()-started))
        print(json.dumps(history[-1]),flush=True);schedule.step()
        write_json(output/'history.json',history)
        payload=dict(format='turtle-depth-only-p2-v1',yaml=model.yaml,
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},provenance=provenance,epoch=epoch+1)
        torch.save(payload,output/'depth.tmp');(output/'depth.tmp').replace(output/'depth.pt')
    write_json(output/'completed.json',dict(epochs=args.epochs,seconds=time.perf_counter()-started,
        checkpoint_sha256=sha256(output/'depth.pt'),status='completed'))


if __name__=='__main__':main()
