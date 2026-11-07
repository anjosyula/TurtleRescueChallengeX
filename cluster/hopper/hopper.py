#!/usr/bin/env python3
"""Portable dataset preparation, explicit transfer inventory, and Slurm submission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\n' in value or '\r' in value:
        raise ValueError(f'Expected a project-relative path: {value}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f'Missing file or path outside project: {value}')
    return resolved


def audit(root, manifest):
    """Check immutable inputs and grouped splits without inventing labels or depth."""
    from PIL import Image
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty manifest list')
    files, groups, seen = {}, {}, set()
    counts = dict(train=0, val=0, test=0)
    missing_groups = 0
    for row in rows:
        split = row['split']
        if split not in counts:raise ValueError(f'Invalid split: {split}')
        if row.get('review_only') or row.get('accepted_for_training') is False:
            raise ValueError('Review-only samples cannot be training inputs')
        if row.get('reviewed') is False:raise ValueError('Unreviewed labels cannot be training inputs')
        if row.get('synthetic') and split != 'train':
            raise ValueError('Current synthetic derivatives must remain train-only')
        if row['image'] in seen:raise ValueError('Duplicate image entry')
        seen.add(row['image']); counts[split] += 1
        for key, hash_key in [('image','sha256'), ('label','label_sha256')]:
            path = local_path(root, row[key])
            actual = digest(path)
            if actual != row[hash_key]:raise ValueError(f'Checksum mismatch: {row[key]}')
            files[row[key]] = actual
        image, label = root / row['image'], root / row['label']
        # Ultralytics derives labels from the images path; a wrong layout silently loses labels.
        expected = Path(str(image).replace('/images/', '/labels/')).with_suffix('.txt')
        if '/images/' not in str(image) or expected != label:
            raise ValueError(f'Use parallel images/ and labels/ directories: {image}')
        with Image.open(image) as im:
            if im.size != (row['width'],row['height']):raise ValueError('Image dimensions changed')
            im.verify()
        labels = label.read_text().splitlines()
        for line in labels:
            values = list(map(float, line.split()))
            if (len(values) != 5 or values[0] != 0 or not all(math.isfinite(x) for x in values)
                or not all(0 <= x <= 1 for x in values[1:]) or min(values[3:]) <= 0):
                raise ValueError(f'Invalid one-class YOLO label: {label}')
        if len(labels) != row['boxes']:raise ValueError(f'Box count mismatch: {label}')
        identities = [('image_sha256',row['sha256'])]
        identities += [(k,str(row[k])) for k in ('group_id','observation_id','individual_id','session_id') if row.get(k)]
        if len(identities) == 1:missing_groups += 1
        for identity in identities:
            if identity in groups and groups[identity] != split:
                raise ValueError(f'Train/validation/test leakage: {identity}')
            groups[identity] = split
        for record in row.get('geometry_profiles',{}).values():
            path = local_path(root,record['depth'])
            files[record['depth']] = digest(path)
        if row.get('render_record'):
            path = local_path(root,row['render_record'])
            files[row['render_record']] = digest(path)
    if not counts['train'] or not counts['val']:raise ValueError('Need train and validation images')
    report = dict(counts=counts,files=len(files),manifest_sha256=digest(manifest),
        rows_without_group_metadata=missing_groups,
        note='Missing group IDs cannot be checked for individual/session leakage; validation is not an independent test.')
    return rows, files, report


def prepare(root, manifest, output):
    rows, files, report = audit(root,manifest)
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'input_checksums.json').write_text(json.dumps(files,indent=2)+'\n')
    for split in ('train','val','test'):
        (output/f'{split}.txt').write_text(''.join(str((root/r['image']).resolve())+'\n' for r in rows if r['split']==split))
    # JSON string quoting is valid YAML and preserves spaces in paths.
    (output/'data.yaml').write_text(f'path: {json.dumps(str(output.resolve()))}\ntrain: train.txt\nval: val.txt\nnames:\n  0: turtle\n')
    return report


def inventory(root, manifest, checkpoint, output):
    _, files, report = audit(root,manifest)
    for path in (manifest,checkpoint):
        rel=path.resolve().relative_to(root.resolve()).as_posix()
        files[rel]=digest(local_path(root,rel))
    # Explicit source allowlist. Never traverse a parent Git root, virtualenv, or all recordings.
    source = list(root.glob('*.py')) + list(root.glob('requirements-*.txt'))
    source += [root/n for n in ('AGENTS.md','instructions.md','OAK_CAPTURE.md','README.md','TRAINING.md','docs/shell-morphology-sources.json')]
    source += list((root/'cluster/hopper').glob('*')) + list((root/'tests/training').glob('*.py'))
    for path in source:
        if path.is_file() and path.suffix in ('.py','.sh','.json','.txt','.md'):
            rel=path.relative_to(root).as_posix()
            files[rel]=digest(local_path(root,rel))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(name+'\n' for name in sorted(files)))
    report.update(bytes=sum((root/name).stat().st_size for name in files),transfer_files=len(files),sha256=files)
    output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='sha256'}


def scene_inventory(root, scenes, output):
    files={}
    for scene in scenes:
        scene=scene.resolve()
        manifest=scene/'scene.json'
        data=json.loads(manifest.read_text())
        if data['format']!='turtlehero-scene-candidate-v1':raise ValueError('Unknown scene format')
        paths=[manifest]
        for row in data['frames']:
            path=local_path(root,(scene/row['file_path']).relative_to(root).as_posix())
            if digest(path)!=row['sha256']:raise ValueError('Scene frame changed')
            paths.append(path)
        for path in paths:files[path.relative_to(root).as_posix()]=digest(path)
    for path in (root/'cluster/hopper').glob('*'):
        if path.is_file():files[path.relative_to(root).as_posix()]=digest(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(name+'\n' for name in sorted(files)))
    report=dict(files=len(files),bytes=sum((root/n).stat().st_size for n in files),sha256=files,
                note='Selected scene frames and provenance only; no raw videos, detector datasets or checkpoints')
    output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='sha256'}


def submit(config_path, execute=False, smoke=False, resume=False):
    config=json.loads(config_path.read_text())
    name=config['name']
    if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('Unsafe run name')
    if config['mode'] not in ('rgb','fusion'):raise ValueError('mode must be rgb or fusion')
    if smoke and resume:raise ValueError('Smoke and resume are separate actions')
    for key in ('manifest','checkpoint'):local_path(ROOT,config[key])
    cluster=config['slurm']
    if not cluster['gres'].endswith(':1'):raise ValueError('Only single-GPU jobs are supported')
    scratch=Path(os.path.expandvars(config['scratch'])).expanduser().resolve()
    env=Path(os.path.expandvars(config['venv'])).expanduser().resolve()
    cmd=['sbatch','--parsable','--nodes=1','--ntasks=1',f'--job-name=turtle-{name}',
         f'--partition={cluster["partition"]}',f'--qos={cluster["qos"]}',f'--gres={cluster["gres"]}',
         f'--cpus-per-task={cluster["cpus"]}',f'--mem={cluster["memory"]}',
         f'--time={"00:15:00" if smoke else cluster["time"]}',
         f'--output={scratch}/logs/%x-%j.out',f'--chdir={ROOT}']
    if cluster.get('account'):cmd.append(f'--account={cluster["account"]}')
    cmd += [str(ROOT/'cluster/hopper/job.sh'),str(ROOT),str(config_path.resolve()),str(env),str(scratch),
            'smoke' if smoke else 'resume' if resume else 'train']
    print(shlex.join(cmd),flush=True)
    if execute:
        if not env.joinpath('bin/python').is_file():raise ValueError('Run setup.sh on Hopper first')
        # Full local audit before consuming a GPU; repeated inside the job after queueing.
        audit(ROOT,ROOT/config['manifest'])
        (scratch/'logs').mkdir(parents=True,exist_ok=True)
        subprocess.run(cmd,check=True)
    return cmd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','inventory'):
        p=sub.add_parser(command)
        p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command=='inventory':p.add_argument('--checkpoint',type=Path,required=True)
    p=sub.add_parser('submit')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Actually submit to Slurm; default only prints command')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('scene-inventory')
    p.add_argument('--scene',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='submit':submit(args.config,args.execute,args.smoke,args.resume)
    elif args.command=='scene-inventory':print(json.dumps(scene_inventory(ROOT,args.scene,args.output),indent=2))
    elif args.command=='prepare':print(json.dumps(prepare(ROOT,args.manifest,args.output),indent=2))
    else:print(json.dumps(inventory(ROOT,args.manifest,args.checkpoint,args.output),indent=2))


if __name__=='__main__':main()
