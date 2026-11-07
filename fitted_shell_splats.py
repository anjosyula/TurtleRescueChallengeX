"""Turn fitted-shell variants into local review-only Gaussian RGB-depth examples."""
import argparse,json,shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from gaussian_dataset import seed_cloud,render_cloud,visible_boxes
from synthetic_dataset import simulated_depth

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    validation=json.loads((args.dataset/'validation.json').read_text())
    if validation['status']!='passed':raise ValueError('Validate actual Blender geometry first')
    records=[]
    for i,row in enumerate(json.loads((args.dataset/'manifest.json').read_text())):
        if shutil.disk_usage(args.output).free<1.5e9:raise OSError('Preserve 1.5 GB disk reserve')
        source=args.dataset/row['sample_id'];out=args.output/row['sample_id'];out.mkdir()
        rgb=cv2.cvtColor(cv2.imread(str(source/'rgb.png')),cv2.COLOR_BGR2RGB)
        z=np.load(source/'depth_z_m.npy');mask=np.load(source/'shell_mask.npy')
        cloud=seed_cloud(rgb,z,mask,row['K'],stride=2)
        # Small new viewpoint; no inferred unseen backsides.
        angle=np.deg2rad(.08);pose=np.eye(4,dtype=np.float32)
        pose[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        result=render_cloud(cloud,row['K'],pose,640,640)
        if not cv2.imwrite(str(out/'rgb.png'),cv2.cvtColor(result['rgb'],cv2.COLOR_RGB2BGR)):raise OSError(out)
        for key in ('depth','validity','mask'):np.save(out/f'{key}.npy',result[key])
        profiles={}
        for profile in ('oak','iphone'):
            depth,metadata=simulated_depth(result['depth'],profile,20261026+i,row['K'][0][0])
            assert np.all(depth[result['validity']==0]==0)
            np.save(out/f'{profile}_depth_z_m.npy',depth);profiles[profile]=metadata
        info={**row,'renderer':'MetalSplat, RGB-D initialized, not multiview-fitted',
            'source_camera_to_target_camera':pose.tolist(),
            'world_to_camera_cv':(pose@np.array(row['world_to_camera_cv'])).tolist(),
            'boxes_xyxy':visible_boxes(result['mask']),
            'rgb_supported_fraction':float((result['alpha']>=.95).mean()),
            'geometry_profiles':profiles,'depth_kind':'synthetic Gaussian expected camera_axis_z',
            'review_only':True,'accepted_for_training':False}
        (out/'sample.json').write_text(json.dumps(info,indent=2)+'\n');records.append(info)
        torch.mps.empty_cache();print(row['sample_id'],info['rgb_supported_fraction'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Completed',len(records),'review-only examples')

if __name__=='__main__':main()
