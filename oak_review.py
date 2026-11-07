"""Build and verify local detector proposals for explicit OAK annotation review."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from collect_oak import write_json
from validate_oak import check, validate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_proposals(path, session, records, view):
    data = json.loads(Path(path).read_text())
    check(data['annotation_status'] == 'unreviewed_proposals', 'Expected unreviewed proposals')
    check(data['view'] == view, 'Proposal camera view differs')
    indexed = {r['sample_id']: r for r in records}
    ids = data['selected_sample_ids']
    check(bool(ids) and len(ids) == len(set(ids)), 'Invalid review sample selection')
    check(set(ids) <= set(data['samples']) <= set(indexed), 'Unknown proposal sample')
    for sid, proposal in data['samples'].items():
        record = indexed[sid]
        check(data['session_id'] == record['session_id'], 'Wrong proposal session')
        check(proposal['image'] == record['files'][view], 'Wrong proposal image')
        image_path = (Path(session)/proposal['image']).resolve()
        check(image_path.is_relative_to(Path(session).resolve()), 'Proposal image escapes session')
        check(digest(image_path) == proposal['image_sha256'], 'Proposal image changed')
        im = cv2.imread(str(image_path))
        check(im is not None and list(im.shape[1::-1]) == proposal['size_wh'], 'Proposal image size differs')
        w, h = proposal['size_wh']
        boxes, scores = proposal['boxes_xyxy'], proposal['scores']
        check(len(boxes) == len(scores), 'Proposal scores do not match boxes')
        for box, score in zip(boxes, scores):
            check(len(box) == 4 and np.isfinite(box).all(), 'Invalid proposal box')
            x1, y1, x2, y2 = box
            check(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, 'Proposal box outside image')
            check(np.isfinite(score) and 0 <= score <= 1, 'Invalid proposal score')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--model-card', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new review output directory')
    validation = validate(args.session)
    records = [json.loads(s) for s in (args.session/'manifest.jsonl').read_text().splitlines() if s]
    predictions = json.loads(args.predictions.read_text())
    by_id = {r['sample_id']: r for r in predictions}
    check(len(by_id) == len(predictions), 'Duplicate prediction IDs')
    card = json.loads(args.model_card.read_text())
    checkpoint = args.model_card.parent/card['checkpoint']
    check(digest(checkpoint) == card['checkpoint_sha256'], 'Model differs from its card')
    samples = {}
    for r in records:
        sid = r['sample_id']; pred = by_id[sid]
        source = (args.session/r['files']['rgb']).resolve()
        check(Path(pred['image']).resolve() == source, 'Predictions belong to a different image/session')
        im = cv2.imread(str(source))
        samples[sid] = {'image': r['files']['rgb'], 'image_sha256': digest(source),
                        'size_wh': list(im.shape[1::-1]),
                        'boxes_xyxy': [d['bounding_box_original_image'] for d in pred['detections']],
                        'scores': [d['turtle_score'] for d in pred['detections']]}
    # Temporal coverage plus cases with multiple/no detections or low scores.
    selected = set(np.linspace(0, len(records)-1, min(16,len(records)),dtype=int).tolist())
    def priority(i):
        scores = samples[records[i]['sample_id']]['scores']
        return (len(scores) != 1, 1-min(scores) if scores else 1)
    for i in sorted(range(len(records)), key=priority, reverse=True):
        if len(selected) >= min(24,len(records)): break
        selected.add(i)
    ids = [records[i]['sample_id'] for i in sorted(selected)]
    data = {'schema_version': 1, 'annotation_status': 'unreviewed_proposals',
            'session_id': records[0]['session_id'], 'view': 'rgb',
            'checkpoint_sha256': card['checkpoint_sha256'],
            'predictions_sha256': digest(args.predictions),
            'selection_policy': '16 evenly spaced frames plus up to 8 multiple/empty/low-score cases; not a calibrated uncertainty estimate',
            'selected_sample_ids': ids, 'samples': samples}
    args.output.mkdir(parents=True)
    write_json(args.output/'proposals.json', data)
    load_proposals(args.output/'proposals.json',args.session,records,'rgb')
    write_json(args.output/'session_validation.json',validation)
    tiles = []
    for sid in ids:
        sample = samples[sid]; im = cv2.imread(str(args.session/sample['image']))
        for box, score in zip(sample['boxes_xyxy'],sample['scores']):
            x1,y1,x2,y2 = map(round,box)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,180,255),3)
        im = cv2.resize(im,(320,240),interpolation=cv2.INTER_AREA)
        tile = np.full((270,320,3),30,dtype=np.uint8);tile[30:]=im
        cv2.putText(tile,f"{sid}: {len(sample['scores'])} proposals",(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        tiles.append(tile)
    while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    cv2.imwrite(str(args.output/'contact-sheet.jpg'),sheet)
    print(f'{len(ids)} selected review frames; {len(samples)} proposal records; no reviewed labels written.')


if __name__ == '__main__': main()
