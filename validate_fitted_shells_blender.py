"""Check actual rendered positions, intrinsics, depth, and texture support."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from blender_synthetic import read_pass

p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);rows=json.loads((args.dataset/'manifest.json').read_text())
checks=[]
for m in rows:
    folder=args.dataset/m['sample_id'];world=read_pass(folder,'position',640)
    K=np.array(m['K']);M=np.array(m['world_to_camera_cv'])
    z=np.load(folder/'depth_z_m.npy');valid=np.load(folder/'validity.npy')>0
    mask=np.load(folder/'shell_mask.npy')>0;support=np.load(folder/'texture_support.npy')>0
    assert np.isfinite(z).all() and np.all(z[~valid]==0) and np.all(z[valid]>0)
    assert not np.any(support&~mask)
    cam=world@M[:3,:3].T+M[:3,3];q=cam@K.T;uv=q[:,:,:2]/q[:,:,2:3]
    yy,xx=np.indices(z.shape);error=np.linalg.norm(uv-np.stack((xx+.5,yy+.5),-1),axis=-1)
    px=float(error[valid].max());depth_error=float(np.max(np.abs(cam[:,:,2][valid]-z[valid])))
    assert px<1.5,(m['sample_id'],px)
    assert depth_error<1e-4,(m['sample_id'],depth_error)
    assert m['review_only'] and not m['accepted_for_training']
    y,x=np.where(mask);assert m['boxes_xyxy']==[[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]]
    checks.append({'sample':m['sample_id'],'projection_max_error_px':px,'depth_max_error_m':depth_error,
        'supported_shell_fraction':float(support.sum()/mask.sum())})
report={'status':'passed','samples':len(checks),'checks':checks,'scope':'Numerical/structural checks only; not anatomical or field validation'}
(args.dataset/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
