"""Audit generated Gaussian views and their real/synthetic training integration."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from train_challenge import ROOT,sha256,write_json
from gaussian_dataset import visible_boxes


def validate(output,checkpoint=None):
    output=Path(output).resolve()
    manifest=json.loads((output/'manifest.json').read_text())
    records=[json.loads(s) for s in (output/'manifest.jsonl').read_text().splitlines() if s.strip()]
    audit=json.loads((output/'audit.json').read_text())
    if len(records)!=audit['gaussian_views']:raise ValueError('Incomplete generation')
    base=json.loads((ROOT/'datasets/turtle-real-synthetic-v1/manifest.json').read_text())
    if manifest[:len(base)]!=base:raise ValueError('Existing split/labels changed')
    if (output/'val.txt').read_bytes()!=(ROOT/'datasets/turtle-real-synthetic-v1/val.txt').read_bytes():
        raise ValueError('Real validation split changed')
    rows=manifest[len(base):]
    if len(rows)!=len(records):raise ValueError('Generated manifest rows differ')
    if len({r['image'] for r in manifest})!=len(manifest):raise ValueError('Duplicate images')
    for row in manifest:
        for path_key,digest_key in [('image','sha256'),('label','label_sha256')]:
            if sha256(ROOT/row[path_key])!=row[digest_key]:raise ValueError('Combined dataset source changed')
    source_root=Path(json.loads((output/'provenance.json').read_text())['settings']['source'])
    source_records={r['sample_id']:r for r in map(json.loads,(source_root/'manifest.jsonl').read_text().splitlines())}
    checksums={};totals={'positive':0,'negative':0,'depth_unknown_under_shell_pixels':0}
    for row,record in zip(rows,records):
        if row['split']!='train' or not row['synthetic'] or record['scale_measured']:
            raise ValueError('Synthetic/split provenance lost')
        source=source_records[record['source_sample_id']]
        if row['group_id']!=source['group_id'] or record['group_id']!=source['group_id']:
            raise ValueError('Source scan group changed')
        if record['negative_scene']!=source['negative_scene']:raise ValueError('Cannot invent background labels')
        if json.loads((ROOT/row['render_record']).read_text())!=record:raise ValueError('Sample metadata differs')
        for key,digest in record['source_checksums'].items():
            if sha256(source_root/source['files'][key])!=digest:raise ValueError('Source render changed')
        for key,digest in [('image','sha256'),('label','label_sha256')]:
            if sha256(ROOT/row[key])!=row[digest]:raise ValueError('Changed image/label')
        paths={k:(output/v).resolve() for k,v in record['files'].items()}
        if any(not v.is_relative_to(output) for v in paths.values()):raise ValueError('Path escapes dataset')
        rgb=cv2.imread(str(paths['rgb']));z=np.load(paths['depth'],allow_pickle=False)
        valid=np.load(paths['validity'],allow_pickle=False);mask=np.load(paths['mask'],allow_pickle=False)
        if rgb is None or rgb.shape!=(640,640,3) or z.shape!=(640,640) or z.dtype!=np.float32:
            raise ValueError('Invalid image/depth dimensions or type')
        if not np.isfinite(z).all() or (z<0).any():raise ValueError('Invalid depth')
        if valid.shape!=z.shape or mask.shape!=z.shape or not np.isin(valid,[0,1]).all() or not np.isin(mask,[0,1]).all():
            raise ValueError('Invalid masks')
        if not np.array_equal(z>0,valid>0):raise ValueError('Unknown depth was filled or validity lost')
        quality=np.load(paths['depth'].parent/'render_quality.npz',allow_pickle=False)
        alpha=quality['alpha'];std=quality['depth_std_m']
        if not np.isfinite(alpha).all() or not np.isfinite(std).all() or np.any((alpha<0)|(alpha>1.00001)):
            raise ValueError('Invalid compositing evidence')
        if np.any((valid>0)&((alpha<.95)|(std>np.maximum(.005,.02*z)+1e-6))):
            raise ValueError('Unreliable rendered depth labeled valid')
        if (alpha>=.95).mean()<.95:raise ValueError('Insufficient RGB support')
        expected=visible_boxes(mask)
        if record['boxes_xyxy']!=expected or (record['negative_scene'] and mask.any()):
            raise ValueError('Visibility labels inconsistent')
        text=(ROOT/row['label']).read_text().strip()
        labels=np.asarray([list(map(float,s.split())) for s in text.splitlines()]).reshape(-1,5)
        target=np.array([[0,(x1+x2)/1280,(y1+y2)/1280,(x2-x1)/640,(y2-y1)/640] for x1,y1,x2,y2 in expected]).reshape(-1,5)
        if not np.allclose(labels,target,atol=1e-8):raise ValueError('YOLO boxes differ from visible mask')
        pose=np.array(record['world_to_camera_cv']);delta=np.array(record['source_camera_to_target_camera'])
        if not np.isfinite(pose).all() or not np.allclose(pose,delta@np.array(source['world_to_camera_cv'])):
            raise ValueError('Camera pose composition differs')
        if not np.allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-5) or np.linalg.det(pose[:3,:3])<.999:
            raise ValueError('Camera rotation invalid')
        if record['K']!=source['K']:raise ValueError('Intrinsics changed without a resize')
        for profile in ('oak','iphone'):
            meta=row['geometry_profiles'][profile]
            d=np.load(ROOT/meta['depth'],allow_pickle=False)
            if not meta['simulated'] or meta['calibrated_sensor_model'] or d.dtype!=np.float32 or d.shape!=z.shape:
                raise ValueError('Sensor simulation provenance/type lost')
            if not np.isfinite(d).all() or (d<0).any() or np.any((z==0)&(d>0)):
                raise ValueError('Simulated sensor filled unknown depth')
        totals['positive' if expected else 'negative']+=1
        totals['depth_unknown_under_shell_pixels']+=int(((mask>0)&(valid==0)).sum())
    for path in output.rglob('*'):
        if path.is_file() and path.name not in ('validation.json','artifact_checksums.json'):
            checksums[str(path.relative_to(output))]=sha256(path)
    # Exercise both profile adapters through the production paired training loader, no optimizer/training.
    from train_fusion import PairedShellDataset,collate_pairs
    data=PairedShellDataset(output/'manifest.json',augment=False)
    new_indices=[i for i,r in enumerate(data.rows) if r['source'].startswith('RGB-D initialized')]
    support=[]
    for i in new_indices:
        for profile in ('oak','iphone'):
            image,geometry,labels=data.get(i,profile)
            if image.shape!=(3,640,640) or geometry.shape!=image.shape or not np.isfinite(geometry).all():
                raise ValueError('Fusion loader contract failed')
            if np.any((geometry[1]==0)&(geometry[0]!=0)):raise ValueError('Invalid geometry in loader')
            support.append(float(geometry[1].mean()))
    batch=collate_pairs([data[new_indices[0]],data[new_indices[-1]]])
    if batch['img'].shape!=(2,3,640,640):raise ValueError('Batch collator failed')
    result={'validation':'passed','new_gaussian_views':len(records),**totals,
        'fusion_samples_in_combined_manifest':len(data),'profile_load_checks':len(support),
        'mean_degraded_depth_valid_fraction':float(np.mean(support)),
        'validation_images':50,'validation_split_byte_identical':True,'all_scan_derivatives_train_only':True,
        'training_started':False,'new_model_created':False,'artifact_files_hashed':len(checksums),
        'limitations':'Structural/numerical validity and loader compatibility; no detector improvement or field realism established'}
    if checkpoint:
        from train_fusion import load_fused,set_fusion_train
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.loss import v8DetectionLoss
        torch.set_num_threads(4)
        model,_=load_fused(checkpoint,'mps')
        batch={k:v.to('mps') for k,v in batch.items()}
        with torch.no_grad():
            output_tensor=model(batch['img'],batch['geometry'])[0]
            if output_tensor.shape!=(2,5,34000) or not torch.isfinite(output_tensor).all():
                raise ValueError('Fused model inference rejected generated batch')
            set_fusion_train(model)
            model.detector.args=get_cfg(overrides={'box':7.5,'cls':.5,'dfl':1.5})
            criterion=v8DetectionLoss(model.detector,tal_topk=10)
            predictions=model(batch['img'],batch['geometry'])
            loss,_=criterion.loss(predictions['one2many'],batch)
            if not torch.isfinite(loss).all():raise ValueError('Nonfinite training loss on generated batch')
        result['model_compatibility']={'checkpoint_sha256':sha256(checkpoint),'raw_output_shape':[2,5,34000],
            'one_to_many_loss_sum':float(loss.sum()),'optimizer_steps':0,'weights_saved':False}
    write_json(output/'artifact_checksums.json',checksums);write_json(output/'validation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    p.add_argument('--check-model',type=Path)
    args=p.parse_args();print(json.dumps(validate(args.dataset,args.check_model),indent=2))
