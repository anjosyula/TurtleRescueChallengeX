"""Fine-tune RT-DETR with user-confirmed USB backgrounds and the audited turtle photos."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from train_challenge import ROOT, local_runtime, sha256, write_json
from turtle_geometry import prepare_rgb_geometry
from train_rtdetr_pilot import prepare

SESSION = ROOT / 'recordings/oak/20261031T151948_a6eff4b9'
BASE = ROOT / 'models/rtdetr-baseline/turtle.pt'


def prepare_usb(session, destination):
    """Copy RGB only; retain original identities, hashes and the user's label provenance."""
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'images').mkdir()
    for filename in ('session.json', 'calibration.json', 'manifest.jsonl', 'summary.json'):
        shutil.copy2(session / filename, destination / ('source_' + filename))
    records = [json.loads(line) for line in (session / 'manifest.jsonl').read_text().splitlines() if line]
    selected = set(range(3, len(records), 4))
    manifest = []
    for i, row in enumerate(records):
        source = (session / row['files']['rgb']).resolve()
        if not source.is_relative_to(session.resolve()):
            raise ValueError('Source image escapes session')
        target = destination / 'images' / (row['sample_id'] + '.png')
        shutil.copy2(source, target)
        checksum = sha256(source)
        if sha256(target) != checksum:
            raise ValueError('USB image copy changed')
        image = cv2.imread(str(target))
        if image is None:
            raise ValueError('Unreadable RGB image')
        manifest.append(dict(sample_id=row['sample_id'], session_id=row['session_id'],
            image=str(target.relative_to(ROOT)), source=str(source), sha256=checksum,
            width=image.shape[1], height=image.shape[0], truth=[], split='train',
            selected_for_finetune=i in selected,
            source_group=row['session_id'], label_method='User confirmed entire recording contains no turtles',
            label_confirmation_date='2026-11-06',
            role='Training background; all views of this session excluded from independent evaluation'))
    write_json(destination / 'manifest.json', manifest)
    return manifest


def build_data(output, backgrounds):
    data = output / 'data'
    yaml_path = prepare(data)
    with (data / 'train.txt').open('a') as listing:
        for row in backgrounds:
            if not row['selected_for_finetune']:
                continue
            source = ROOT / row['image']
            if sha256(source) != row['sha256']:
                raise ValueError('Background image changed')
            rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
            image, _, _ = prepare_rgb_geometry(rgb)
            name = 'usb_' + row['sample_id']
            target = data / 'images/train' / (name + '.png')
            pixels = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
            if not cv2.imwrite(str(target), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
                raise OSError(target)
            (data / 'labels/train' / (name + '.txt')).write_text('')
            listing.write(str(target.resolve()) + '\n')
    write_json(data / 'usb_manifest.json', backgrounds)
    return yaml_path


def train(output, backgrounds, epochs, device):
    local_runtime()
    import torch
    from ultralytics import RTDETR
    if importlib.metadata.version('ultralytics') != '8.4.157':
        raise RuntimeError('Use the existing pinned Ultralytics environment')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    data = build_data(output, backgrounds)
    settings = dict(data=str(data), epochs=epochs, patience=0, imgsz=640, batch=2, nbs=8,
        device=device, workers=0, seed=42, deterministic=True, optimizer='AdamW',
        lr0=.00002, lrf=.1, cos_lr=True, warmup_epochs=1, warmup_bias_lr=0., weight_decay=.0001,
        freeze=10, mosaic=0., mixup=0., cutmix=0., degrees=5., translate=.05, scale=.2,
        fliplr=.5, flipud=0., hsv_h=.015, hsv_s=.4, hsv_v=.3, amp=False, plots=False,
        project=str(output), name='fit', exist_ok=False, save=True)
    write_json(output / 'provenance.json', dict(initial_checkpoint=str(BASE), initial_sha256=sha256(BASE),
        settings=settings, usb_frames=len(backgrounds), usb_training_frames=sum(r['selected_for_finetune'] for r in backgrounds),
        original_training_photos=198, original_validation_photos=50,
        validation_policy='Original photo validation unchanged; entire USB encounter is training only',
        user_confirmation='No turtles — background recording', source_script_sha256=sha256(__file__),
        versions={k: importlib.metadata.version(k) for k in ('torch','ultralytics','numpy')},
        uploads=False, new_camera_recording=False))
    shutil.copy2(__file__, output / Path(__file__).name)
    model = RTDETR(str(BASE))
    model.add_callback('on_fit_epoch_end', lambda trainer: write_json(output / 'progress.json',
        dict(epoch=trainer.epoch + 1, epochs=epochs, metrics=trainer.metrics)))
    start = time.perf_counter()
    model.train(**settings)
    write_json(output / 'trained.json', dict(epochs=epochs, seconds=time.perf_counter()-start,
        best=str(model.trainer.best), sha256=sha256(model.trainer.best), metrics=model.metrics.results_dict))


def evaluate(output, backgrounds, device, package):
    local_runtime()
    import torch
    from ultralytics import RTDETR
    from evaluate_dual_pilot import rtdetr_decode, report, counts, at_fp_budget
    from evaluate_fusion import truth_boxes
    torch.set_num_threads(4)
    if package.exists():
        raise FileExistsError('Choose a fresh --model-output directory')
    folder = output / 'evaluation'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'provenance.json', dict(device=device,
        evaluator_sha256=sha256(__file__), baseline_sha256=sha256(BASE),
        adapted_sha256=sha256(output / 'fit/weights/best.pt'),
        validation_manifest_sha256=sha256(ROOT / 'datasets/turtle-challenge-v1/manifest.json'),
        validation_role='Existing model-selection set', usb_role='Training encounter diagnostic'))
    source = json.loads((ROOT / 'datasets/turtle-challenge-v1/manifest.json').read_text())
    real = [r for r in source if r['split'] == 'val']
    results = {}
    threshold = .6331660747528076
    for name, weights in [('baseline', BASE), ('adapted', output / 'fit/weights/best.pt')]:
        model = RTDETR(str(weights)).model.float().to(device).eval()
        samples, negatives = [], []
        with torch.inference_mode():
            for rows, target, kind in [(real, samples, 'validation'), (backgrounds, negatives, 'USB training diagnostic')]:
                for i, row in enumerate(rows):
                    rgb = cv2.cvtColor(cv2.imread(str(ROOT / row['image'])), cv2.COLOR_BGR2RGB)
                    image, _, transform = prepare_rgb_geometry(rgb)
                    predictions = rtdetr_decode(model, torch.from_numpy(image[None]).to(device), transform, row['width'], row['height'])
                    target.append(dict(image=row['image'], width=row['width'], height=row['height'],
                        truth=truth_boxes(row) if kind == 'validation' else [], predictions=predictions))
                    if i % 20 == 0:
                        print(name, kind, i + 1, '/', len(rows), flush=True)
        measured = report(samples, folder / (name + '_real'))
        write_json(folder / (name + '_usb_predictions.json'), negatives)
        results[name] = dict(coco=measured['coco'], fixed_threshold=counts(samples, threshold),
            at_nine_false_boxes=at_fp_budget(samples, 9), selected=measured['operating_points']['best_F2'],
            usb_alert_frames=sum(any(p[4] >= threshold for p in r['predictions']) for r in negatives),
            usb_total_frames=len(negatives))
        del model
        if device == 'mps': torch.mps.empty_cache()
    recommended = (results['adapted']['coco']['AP'] >= results['baseline']['coco']['AP'] and
                   results['adapted']['at_nine_false_boxes']['tp'] >= results['baseline']['at_nine_false_boxes']['tp'])
    results.update(recommend_default=recommended, fixed_confidence=threshold,
        usb_role='Training encounter diagnostic only; no held-out USB site',
        caveat='Same 50-photo model-selection set; no claim of independent field accuracy or false alerts per hour')
    write_json(output / 'comparison.json', results)
    package.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output / 'fit/weights/best.pt', package / 'turtle.pt')
    write_json(package / 'model.json', dict(model='RT-DETR-L',weights='turtle.pt',
        selected_confidence=results['adapted']['selected']['confidence'],
        recommended_as_default=recommended, experiment=str(output),
        checkpoint_sha256=sha256(package / 'turtle.pt'), comparison=results,
        preprocessing='Existing prepare_rgb_geometry square letterbox and rtdetr_decode'))
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=SESSION)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/usb-background-20261106')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/usb-background-20261106')
    parser.add_argument('--model-output', type=Path, default=ROOT / 'models/rtdetr-usb-background-20261106')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--phase', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    if args.epochs < 1: parser.error('Positive epoch count required')
    import torch
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable in this environment; use --device cpu')
    if args.phase in ('prepare', 'all'):
        from validate_oak import validate
        validation = validate(args.session)
        backgrounds = prepare_usb(args.session, args.dataset)
        write_json(args.dataset / 'validation.json', validation)
    else:
        backgrounds = json.loads((args.dataset / 'manifest.json').read_text())
    if args.phase in ('train', 'all'): train(args.output, backgrounds, args.epochs, args.device)
    if args.phase in ('evaluate', 'all'): evaluate(args.output, backgrounds, args.device, args.model_output)


if __name__ == '__main__':
    main()
