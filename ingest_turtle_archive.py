"""Preserve a user-provided media ZIP and prepare metadata/contact sheets, not labels."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageOps
from pillow_heif import register_heif_opener


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def text_value(value):
    return str(value).strip('\x00') if value is not None else None


def sheet(items,path):
    cols=4;cw,ch=320,270
    canvas=Image.new('RGB',(cols*cw,((len(items)+cols-1)//cols)*ch),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(image,label) in enumerate(items):
        x,y=(i%cols)*cw,(i//cols)*ch
        image.thumbnail((cw-12,220))
        canvas.paste(image,(x+(cw-image.width)//2,y))
        draw.text((x+5,y+222),label[:115],fill='black')
    canvas.save(path,quality=85)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=Path('datasets/what-about-turtles'))
    args=p.parse_args()
    register_heif_opener()
    root=args.output.resolve();raw=root/'raw';previews=root/'previews'
    raw.mkdir(parents=True,exist_ok=True);previews.mkdir(exist_ok=True)
    rows=[];pictures=[];video_pictures=[]
    with zipfile.ZipFile(args.archive) as archive:
        entries=[x for x in archive.infolist() if not x.is_dir()]
        if shutil.disk_usage(root).free < sum(x.file_size for x in entries)+1024**3:
            raise RuntimeError('Require expanded archive size plus 1 GiB free')
        for i,entry in enumerate(entries):
            relative=Path(entry.filename)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe ZIP path')
            destination=raw/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            payload=archive.read(entry)
            checksum=hashlib.sha256(payload).hexdigest()
            if destination.exists():
                if sha(destination)!=checksum:raise ValueError(f'Existing original changed: {destination}')
            else:destination.write_bytes(payload)
            row=dict(id=f'{i:04d}',original_member=entry.filename,path=str(destination.relative_to(root)),
                sha256=checksum,bytes=entry.file_size,group_hint=str(relative.parent),
                reviewed=False,accepted_for_training=False,labels=None,license=None,
                camera_model=None,capture_time=None)
            extension=destination.suffix.lower()
            if extension in ('.heic','.jpg','.jpeg','.png'):
                row['kind']='image'
                with Image.open(io.BytesIO(payload)) as image:
                    exif=image.getexif();details=exif.get_ifd(34665)
                    row.update(camera_make=text_value(exif.get(271)),camera_model=text_value(exif.get(272)),
                        capture_time=text_value(details.get(36867) or exif.get(306)),
                        lens_model=text_value(details.get(42036)),
                        focal_length_mm=text_value(details.get(37386)),
                        focal_length_35mm=text_value(details.get(41989)),
                        orientation=exif.get(274),original_wh=list(image.size))
                    upright=ImageOps.exif_transpose(image).convert('RGB')
                    row['upright_wh']=list(upright.size)
                    thumb=upright.copy();thumb.thumbnail((640,480))
                    thumb.save(previews/(row['id']+'.jpg'),quality=85)
                    pictures.append((thumb,f"{row['id']} {destination.name}\n{relative.parent.name[:42]}\n{row['camera_model'] or 'camera unknown'}"))
            elif extension in ('.mov','.mp4'):
                row['kind']='video'
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(destination)]))
                tags=probe.get('format',{}).get('tags',{})
                stream=next(s for s in probe['streams'] if s['codec_type']=='video')
                duration=float(probe['format']['duration'])
                row.update(camera_model=tags.get('com.apple.quicktime.model'),
                    camera_make=tags.get('com.apple.quicktime.make'),
                    capture_time=tags.get('com.apple.quicktime.creationdate') or tags.get('creation_time'),
                    duration_seconds=duration,encoded_wh=[stream['width'],stream['height']],
                    frame_rate=stream.get('avg_frame_rate'),codec=stream.get('codec_name'))
                # Six review frames only: avoid treating these as reconstructed camera poses.
                frames=[]
                for j in range(6):
                    seconds=duration*(j+.5)/6
                    output=previews/f"{row['id']}-frame-{j}.jpg"
                    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(seconds),'-i',str(destination),
                        '-frames:v','1','-vf','scale=640:480:force_original_aspect_ratio=decrease',str(output)],check=True)
                    with Image.open(output) as image:thumb=image.convert('RGB').copy()
                    frames.append((thumb,f"{row['id']} {destination.name} @ {seconds:.1f}s\n{relative.parent.name[:42]}"))
                sheet(frames,previews/(row['id']+'-video-sheet.jpg'))
                video_pictures.append(frames[2])
            else:row['kind']='document'
            rows.append(row)
    for i in range(0,len(pictures),32):sheet(pictures[i:i+32],root/f'images-{i//32+1}.jpg')
    if video_pictures:sheet(video_pictures,root/'videos.jpg')
    summary=dict(archive=args.archive.name,archive_sha256=sha(args.archive),files=len(rows),
        kinds=dict(Counter(r['kind'] for r in rows)),
        cameras=dict(Counter(r['camera_model'] or 'unknown' for r in rows if r['kind']!='document')),
        original_bytes=sum(r['bytes'] for r in rows),
        annotation_status='Unreviewed media only; no detection labels or train/test split assigned',
        group_policy='Folder is a capture-group hint only; verify dates, individuals and encounters before splitting',
        reconstruction_status='No poses, metric scale, reconstruction or Gaussian optimization yet')
    (root/'media_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
