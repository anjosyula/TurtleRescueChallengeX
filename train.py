"""Prepare the exported turtle dataset and fine-tune YOLO26 nano locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
(Path(os.environ['YOLO_CONFIG_DIR']) / 'Ultralytics').mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)


def prepare():
    from PIL import Image
    import yaml

    source = ROOT / 'turtle-hero.ndjson'
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert records[0]['task'] == 'detect'
    names = {int(k): v for k, v in records[0]['class_names'].items()}
    dataset = ROOT / 'datasets' / 'turtle-hero'
    seen = set()
    for row in records[1:]:
        assert row['split'] in ('train', 'val', 'test')
        assert Path(row['file']).name == row['file']
        assert row['file'] not in seen, f"Duplicate image: {row['file']}"
        seen.add(row['file'])
        assert 'boxes' in row.get('annotations', {}), f"Missing annotations: {row['file']}"
        for box in row['annotations']['boxes']:
            assert len(box) == 5 and box[0] in names
            assert all(0 <= v <= 1 for v in box[1:]) and box[3] > 0 and box[4] > 0

    def materialize(row):
        image_path = dataset / 'images' / row['split'] / row['file']
        label_path = dataset / 'labels' / row['split'] / (Path(row['file']).stem + '.txt')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            for attempt in range(4):
                try:
                    with urlopen(row['url'], timeout=60) as response:
                        content = response.read()
                    temporary = image_path.with_suffix('.part')
                    temporary.write_bytes(content)
                    with Image.open(temporary) as image:
                        image.verify()
                    temporary.replace(image_path)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise RuntimeError(f"Failed to download {row['file']}: {type(exc).__name__}") from None
                    time.sleep(2 ** attempt)
        with Image.open(image_path) as image:
            assert image.size == (row['width'], row['height']), f"Size mismatch: {row['file']}"
            image.verify()
        label_path.write_text(''.join(' '.join(map(str, [int(b[0]), *b[1:]])) + '\n'
                                          for b in row['annotations']['boxes']))
        return row['split'], hashlib.sha256(image_path.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(materialize, records[1:]))
    hashes = {}
    for split, digest in results:
        assert digest not in hashes or hashes[digest] == split, 'Duplicate image crosses dataset splits'
        hashes[digest] = split
    config = dataset / 'data.yaml'
    config.write_text(yaml.safe_dump({'path': str(dataset), 'train': 'images/train',
                                     'val': 'images/val', 'names': names}, sort_keys=False))
    (dataset / 'source.sha256').write_text(hashlib.sha256(source.read_bytes()).hexdigest() + '\n')
    from collections import Counter
    print(f'Dataset verified: {dict(Counter(s for s, _ in results))}', flush=True)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default=None)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    data = prepare()
    if args.prepare_only:
        return
    import torch
    from ultralytics import YOLO
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Training YOLO26 nano on {device}', flush=True)
    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True, device=device)
    else:
        model = YOLO(str(ROOT / 'yolo26n.pt'))
        model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=args.batch,
                    device=device, workers=0, patience=20, seed=42,
                    project=str(ROOT / 'runs' / 'detect'), name='turtle_yolo26n',
                    plots=True, save=True)
    print(f'Best weights: {model.trainer.best}', flush=True)


if __name__ == '__main__':
    main()
