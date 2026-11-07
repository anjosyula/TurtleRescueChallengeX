"""Reproducible local RGB / P2 training on the audited turtle-photo split."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

from train import ROOT


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_runtime():
    """Disable analytics and cloud loggers before loading any training callbacks."""
    path = Path(os.environ['YOLO_CONFIG_DIR']) / 'Ultralytics/settings.json'
    settings = json.loads(path.read_text()) if path.exists() else {}
    for key in ('sync', 'clearml', 'comet', 'dvc', 'mlflow', 'raytune', 'tensorboard', 'wandb'):
        settings[key] = False
    settings['api_key'] = ''
    write_json(path, settings)
    os.environ.pop('ULTRALYTICS_API_KEY', None)
    os.environ['WANDB_MODE'] = 'disabled'


def prepare_audited():
    """Freeze a local-only view. Never label unreviewed camera captures as negatives."""
    import numpy as np
    from PIL import Image
    import yaml

    source = ROOT / 'datasets/turtle-hero'
    dataset = ROOT / 'datasets/turtle-challenge-v1'
    rows = [json.loads(s) for s in (ROOT / 'turtle-hero.ndjson').read_text().splitlines() if s.strip()]
    excluded = {'6516879.jpg': 'Visually confirmed near-duplicate of validation image 10066100.jpg; pHash distance 2/63.'}
    manifest, hashes = [], {}
    image_lists = {'train': [], 'val': []}
    for row in rows[1:]:
        split, name = row['split'], row['file']
        if split not in image_lists or (split == 'train' and name in excluded):
            continue
        if Path(name).name != name:
            raise ValueError('Image name must be a basename')
        image = source / 'images' / split / name
        label = source / 'labels' / split / (Path(name).stem + '.txt')
        with Image.open(image) as im:
            if im.size != (row['width'], row['height']):
                raise ValueError(f'Image dimensions changed: {image}')
            im.verify()
        boxes = np.asarray(row['annotations']['boxes'], dtype=float).reshape(-1, 5)
        for cls, x, y, w, h in boxes:
            if not (cls == 0 and np.isfinite([x,y,w,h]).all() and w > 0 and h > 0
                    and min(x-w/2,y-h/2) >= -1e-4 and max(x+w/2,y+h/2) <= 1.0001):
                raise ValueError(f'Invalid box: {name}')
        saved = np.loadtxt(label, ndmin=2).reshape(-1, 5)
        if not np.allclose(saved, boxes, atol=1e-7):
            raise ValueError(f'Labels differ from source: {name}')
        digest = sha256(image)
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f'Exact duplicate crosses split: {name}')
        hashes[digest] = split
        image_lists[split].append(str(image))
        manifest.append({'image': str(image.relative_to(ROOT)), 'label': str(label.relative_to(ROOT)),
                         'split': split, 'sha256': digest, 'label_sha256': sha256(label),
                         'boxes': len(boxes), 'width': row['width'], 'height': row['height'],
                         'source': 'user-provided turtle-hero.ndjson',
                         'license': None, 'observation_id': None, 'individual_id': None})
    dataset.mkdir(parents=True, exist_ok=True)
    for split, images in image_lists.items():
        (dataset / f'{split}.txt').write_text('\n'.join(images) + '\n')
    (dataset / 'data.yaml').write_text(yaml.safe_dump({'path': str(dataset), 'train': 'train.txt',
                                                     'val': 'val.txt', 'names': {0:'turtle'}}))
    write_json(dataset / 'manifest.json', manifest)
    write_json(dataset / 'audit.json', {
        'source_sha256': sha256(ROOT / 'turtle-hero.ndjson'), 'excluded_training_images': excluded,
        'images': dict(Counter(r['split'] for r in manifest)),
        'boxes': {s: sum(r['boxes'] for r in manifest if r['split'] == s) for s in image_lists},
        'validation_role': 'model/threshold selection; no independent test set',
        'limitations': ['No labeled turtle-free images', 'Source licenses and encounter/individual groups absent from export',
                       'Near-duplicate screening is not proof of independent observations',
                       'OAK captures excluded: no reviewed boxes', 'No paired iPhone or IR training data']})
    return dataset / 'data.yaml'


def initialize_p2(destination):
    """Transfer semantically corresponding blocks; P2 shifts layer and scale indices."""
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO

    cfg_path = Path(ultralytics.__file__).parent / 'cfg/models/26/yolo26-p2.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(nc=1, scale='n')
    config_path = destination / 'yolo26n-p2-turtle.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    model = YOLO(str(config_path))
    model.model.names = {0: 'turtle'}
    source = YOLO(str(ROOT / 'yolo26n.pt')).model.float().state_dict()
    target = model.model.state_dict()
    blocks = {**{i:i for i in range(17)}, 17:23, 18:24, 19:25, 20:26, 21:27, 22:28}
    transferred, mapping = {}, {}
    for key, value in source.items():
        parts = key.split('.')
        old_index = int(parts[1])
        if old_index in blocks:
            parts[1] = str(blocks[old_index])
        elif old_index == 23 and parts[2] in ('cv2', 'cv3', 'one2one_cv2', 'one2one_cv3'):
            parts[1], parts[3] = '29', str(int(parts[3]) + 1)
        else:
            continue
        new_key = '.'.join(parts)
        if new_key in target and target[new_key].shape == value.shape:
            transferred[new_key], mapping[new_key] = value, key
    model.model.load_state_dict(transferred, strict=False)
    assert model.model.stride.tolist() == [4.,8.,16.,32.]
    assert torch.equal(model.model.state_dict()['model.0.conv.weight'], source['model.0.conv.weight'])
    write_json(destination / 'p2_transfer.json', {'source': str(ROOT/'yolo26n.pt'),
        'source_sha256': sha256(ROOT/'yolo26n.pt'), 'tensor_mapping': mapping,
        'transferred_tensors': len(mapping), 'total_tensors': len(target)})
    path = destination / 'p2_initial.pt'
    model.save(str(path))
    return YOLO(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['rgb','p2'], default='rgb')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--name', required=True)
    p.add_argument('--project', type=Path, default=ROOT/'runs/challenge')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--resume', type=Path, help='Trusted local Ultralytics last.pt; resumes its original recipe')
    p.add_argument('--init-checkpoint', type=Path, help='Local turtle checkpoint for a new fine-tuning run')
    p.add_argument('--scale-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Explicit image-scale gain bounds; defaults to the original 0.7..1.3 recipe')
    p.add_argument('--learning-rate', type=float, default=0.0003)
    p.add_argument('--data', type=Path, help='Prepared local dataset YAML with manifest/audit and split lists')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.scale_range and not 0 < args.scale_range[0] <= args.scale_range[1]:
        p.error('--scale-range must have positive, ordered bounds')
    if args.learning_rate <= 0 or min(args.epochs,args.batch) < 1 or args.patience < 0:
        p.error('Invalid training configuration')
    if args.init_checkpoint and not args.init_checkpoint.is_file():
        p.error('--init-checkpoint must be an existing local file')
    local_runtime()
    if args.workers < 0 or Path(args.name).name != args.name or args.name in ('.','..'):
        p.error('Invalid workers or run name')
    if args.resume:
        from ultralytics import YOLO
        if importlib.metadata.version('ultralytics') != '8.4.157':
            raise RuntimeError('Requires pinned ultralytics==8.4.157')
        if not args.resume.is_file():p.error('Resume checkpoint does not exist')
        original = args.project.resolve()/args.name
        if args.resume.resolve() != original/'weights/last.pt':
            p.error('Resume must use this project/name/weights/last.pt')
        old = json.loads((original/'provenance.json').read_text())
        if not args.data or sha256(args.data.parent/'manifest.json') != old['dataset_manifest_sha256']:
            p.error('Resume requires the unchanged prepared --data manifest')
        model=YOLO(str(args.resume))
        model.add_callback('on_fit_epoch_end',lambda trainer: write_json(original/'progress.json',
            {'epoch':trainer.epoch+1,'metrics':trainer.metrics,'best_fitness':trainer.best_fitness}))
        model.train(resume=True, device=args.device, workers=args.workers)
        with (original/'results.csv').open() as stream:
            epochs_completed=len(list(csv.DictReader(stream)))
        completed={'status':'completed','best':str(model.trainer.best),'last':str(model.trainer.last),
                   'epochs_completed':epochs_completed,'metrics':model.metrics.results_dict,'resumed':True}
        write_json(original/'completed.json',completed)
        write_json(original/'progress.json',completed)
        return
    data = args.data.resolve() if args.data else prepare_audited()
    if args.data:
        import yaml
        cfg=yaml.safe_load(data.read_text())
        if cfg.get('names')!={0:'turtle'} or cfg.get('download'):
            p.error('Expected a local one-class turtle dataset without download hooks')
        for name in ('manifest.json','audit.json','train.txt','val.txt'):
            if not (data.parent/name).is_file():p.error(f'Missing dataset contract: {name}')
        if Path(cfg['path']).resolve()!=data.parent or cfg['train']!='train.txt' or cfg['val']!='val.txt':
            p.error('Dataset YAML must use its adjacent audited split lists')
        for split in ('train','val'):
            if not all(Path(s).is_file() for s in (data.parent/f'{split}.txt').read_text().splitlines()):
                p.error('Missing local dataset image')
    if args.prepare_only:
        print(data)
        return
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    if ultralytics.__version__ != '8.4.157':
        raise RuntimeError('This experiment is pinned to ultralytics==8.4.157')
    init_seeds(42, deterministic=True)
    project = args.project.resolve()
    run = project / args.name
    if run.exists():
        raise FileExistsError(f'Choose a new run name: {run}')
    run.mkdir(parents=True)
    initial = args.init_checkpoint.resolve() if args.init_checkpoint else ROOT/'yolo26n.pt'
    if args.init_checkpoint:
        model = YOLO(str(initial))
        expected = [4.,8.,16.,32.] if args.variant == 'p2' else [8.,16.,32.]
        if model.model.stride.tolist() != expected or model.names != {0:'turtle'}:
            raise ValueError('Fine-tuning checkpoint must match the requested variant and turtle class')
    else:
        model = initialize_p2(run) if args.variant == 'p2' else YOLO(str(initial))
    settings = dict(data=str(data), epochs=args.epochs, patience=args.patience, imgsz=640,
        batch=args.batch, device=args.device, workers=args.workers, seed=42, deterministic=True,
        optimizer='AdamW', lr0=args.learning_rate, lrf=0.1, cos_lr=True, warmup_epochs=3,
        warmup_bias_lr=0.0, weight_decay=0.0005, mosaic=0.3, close_mosaic=10,
        mixup=0.0, cutmix=0.0, scale=tuple(args.scale_range) if args.scale_range else 0.3,
        degrees=10.0, fliplr=0.5, flipud=0.1,
        hsv_s=0.4, hsv_v=0.3, amp=False, plots=True, save=True,
        project=str(project), name=args.name, exist_ok=True)
    write_json(run/'provenance.json', {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'variant':args.variant,
        'python':platform.python_version(), 'platform':platform.platform(),
        'versions':{n:importlib.metadata.version(n) for n in ['ultralytics','torch','torchvision','numpy','opencv-python']},
        'initial_checkpoint':str(initial), 'initial_checkpoint_sha256':sha256(initial),
        'fine_tuning':bool(args.init_checkpoint),
        'dataset_manifest_sha256':sha256(data.parent/'manifest.json'),
        'training_script_sha256':sha256(__file__), 'settings':settings,
        'inference':'one-to-many branch with NMS; nms=None; square letterbox 640 for comparisons',
        'loss':'Pinned Ultralytics E2ELoss with progressive one-to-many/one-to-one weighting and small-box TAL expansion',
        'uploads':False})
    for filename in ('audit.json','manifest.json','data.yaml','train.txt','val.txt'):
        shutil.copy2(data.parent/filename,run/('dataset_'+filename))
    def report(trainer):
        write_json(run/'progress.json',{'epoch':trainer.epoch+1, 'metrics':trainer.metrics,
                                       'best_fitness':trainer.best_fitness})
    model.add_callback('on_fit_epoch_end',report)
    model.train(**settings)
    with (run/'results.csv').open() as stream:
        epochs_completed = len(list(csv.DictReader(stream)))
    write_json(run/'completed.json',{'best':str(model.trainer.best),'last':str(model.trainer.last),
                                   'epochs_completed':epochs_completed,'metrics':model.metrics.results_dict})
    write_json(run/'progress.json',{'status':'completed','epochs_completed':epochs_completed,
                                   'best_checkpoint':str(model.trainer.best),
                                   'selected_checkpoint_metrics':model.metrics.results_dict})
    print('Completed:',model.trainer.best,flush=True)


if __name__ == '__main__':
    main()
