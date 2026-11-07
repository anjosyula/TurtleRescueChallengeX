"""Fetch five unseen, alive-annotated turtles and save model predictions."""
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from train import ROOT
from PIL import Image
from ultralytics import YOLO


def fetch(url):
    for attempt in range(4):
        try:
            time.sleep(1)
            with urlopen(Request(url, headers={'User-Agent': 'TurtleHero/1.0'}), timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    output = ROOT / 'runs' / 'new_inaturalist'
    originals = output / 'originals'
    annotated = output / 'annotated'
    originals.mkdir(parents=True, exist_ok=True)
    annotated.mkdir(parents=True, exist_ok=True)
    excluded = {Path(json.loads(line).get('file', '')).stem
                for line in (ROOT / 'turtle-hero.ndjson').read_text().splitlines() if line.strip()}
    photo_ids = set()
    for manifest in (ROOT / 'images').rglob('manifest.json'):
        for row in json.loads(manifest.read_text()):
            excluded.add(str(row['observation_id']))
            photo_ids.add(row.get('photo_id'))
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'datasets/turtle-hero/images').rglob('*') if p.is_file()}
    selected = []
    for grade in ('research', 'needs_id', 'casual'):
        page = 1
        while len(selected) < 5:
            params = dict(taxon_id=39682, photos='true', term_id=17, term_value_id=18,
                          quality_grade=grade, per_page=100, page=page, order_by='id', order='desc')
            observations = json.loads(fetch('https://api.inaturalist.org/v1/observations?' + urlencode(params)))['results']
            if not observations:
                break
            page += 1
            for obs in observations:
                if str(obs['id']) in excluded:
                    continue
                terms = [a for a in obs.get('annotations', []) if a.get('controlled_attribute_id') == 17]
                if not any(a.get('controlled_value_id') == 18 and (a.get('vote_score') or 0) >= 0 for a in terms):
                    continue
                if any(a.get('controlled_value_id') in (19, 20) for a in terms):
                    continue
                for photo in obs.get('photos', []):
                    if photo.get('id') in photo_ids or not photo.get('url'):
                        continue
                    parts = urlsplit(photo['url'])
                    path = Path(parts.path)
                    url = urlunsplit(parts._replace(path=str(path.with_name('large' + path.suffix))))
                    content = fetch(url)
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in hashes:
                        continue
                    destination = originals / f"{obs['id']}{path.suffix}"
                    destination.write_bytes(content)
                    with Image.open(destination) as image:
                        image.verify()
                    selected.append(dict(observation_id=obs['id'],
                                         observation_url=f"https://www.inaturalist.org/observations/{obs['id']}",
                                         photo_id=photo['id'], photo_url=url,
                                         license_code=photo.get('license_code'), attribution=photo.get('attribution'),
                                         quality_grade=grade, alive_annotation=True, original=str(destination.relative_to(ROOT))))
                    excluded.add(str(obs['id']))
                    hashes.add(digest)
                    print(f"Downloaded {len(selected)}/5: {destination.name}", flush=True)
                    break
                if len(selected) == 5:
                    break
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise RuntimeError(f'Only found {len(selected)} new eligible images')
    tuning = output / 'confidence_tuning.json'
    confidence = json.loads(tuning.read_text())['selected_confidence'] if tuning.exists() else 0.016
    model = YOLO(str(ROOT / 'runs/detect/turtle_yolo26n/weights/best.pt'))
    for row in selected:
        result = model.predict(str(ROOT / row['original']), conf=confidence, imgsz=640, device='mps', verbose=False)[0]
        destination = annotated / (Path(row['original']).stem + '.jpg')
        result.save(filename=str(destination))
        row['annotated'] = str(destination.relative_to(ROOT))
        row['detections'] = json.loads(result.to_json())
        print(f"Saved {destination.name}: {len(result.boxes)} detections", flush=True)
    (output / 'manifest.json').write_text(json.dumps({'taxon_id':39682, 'confidence':confidence, 'images':selected}, indent=2) + '\n')


if __name__ == '__main__':
    main()
