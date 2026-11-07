"""Extract a bounded, timestamped video segment for a later SfM/splat feasibility test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('datasets/what-about-turtles/media_manifest.json'))
    p.add_argument('--video-id',required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--end',type=float,required=True)
    p.add_argument('--fps',type=float,default=2)
    p.add_argument('--max-width',type=int,default=1280,help='Maximum dimension of an upright frame')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    row=next(r for r in json.loads(args.manifest.read_text()) if r['id']==args.video_id)
    if row['kind']!='video' or not 0<=args.start<args.end<=row['duration_seconds']:
        raise ValueError('Choose a valid video time interval')
    if args.fps<=0 or (args.end-args.start)*args.fps>200 or args.max_width<64:
        raise ValueError('Use positive fps, at most 200 candidate frames, and max-width >=64')
    source=args.manifest.parent/row['path']
    if sha(source)!=row['sha256']:raise ValueError('Original media changed')
    args.output.mkdir(parents=True,exist_ok=False)
    images=args.output/'images';images.mkdir()
    # Decode original PTS and select frames against them, preserving the actual source time.
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(source)]))
    times=[float(r['best_effort_timestamp_time']) for r in probe['frames']]
    capture=cv2.VideoCapture(str(source))
    rows=[];next_time=args.start
    for index,t in enumerate(times):
        ok,frame=capture.read()
        if not ok:break
        if t>=args.end:break
        if t+1e-6<next_time:continue
        h,w=frame.shape[:2];scale=min(1,args.max_width/max(w,h))
        frame=cv2.resize(frame,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        filename=f'frame_{len(rows):05d}.jpg';path=images/filename
        if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('Frame write failed')
        rows.append(dict(file_path='images/'+filename,source_frame_index=index,source_pts_seconds=t,
            sha256=sha(path),width=frame.shape[1],height=frame.shape[0],
            laplacian_variance=float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())))
        next_time+=1/args.fps
    capture.release()
    if len(rows)<20:raise ValueError('Fewer than 20 frames: choose a longer useful segment')
    result=dict(format='turtlehero-scene-candidate-v1',source=row,frames=rows,
        settings=vars(args)|{'manifest':str(args.manifest),'output':str(args.output)},
        camera_model=row['camera_model'],intrinsics_status='Unknown for decoded video; estimate with SfM, do not reuse OAK or still-photo intrinsics',
        metric_scale=None,measured_depth=False,poses_recovered=False,
        accepted_for_training=False,review_only=True,
        limitations=['Feasibility candidate: image overlap/parallax and static geometry unverified',
                    'Moving people, machinery, vegetation and water need exclusion/masking',
                    'Nearby video frames are correlated; keep this encounter in one downstream split'])
    (args.output/'scene.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'scene':str(args.output),'frames':len(rows),'source':row['original_member']},indent=2))


if __name__=='__main__':main()
