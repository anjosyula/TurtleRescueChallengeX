"""Fit a rough scan silhouette under camera/size hypotheses, not recover true anatomy."""
from pathlib import Path
import argparse
import json
import hashlib
import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent

def source_evidence(source):
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    actual=digest(ROOT/source['image'])
    if actual!=source['image_sha256']:raise ValueError('Source image changed')
    record={'image_sha256':actual}
    if source.get('oak_sample'):
        folder=ROOT/source['oak_sample'];m=json.loads((folder/'sample.json').read_text())
        record.update(source_sample_sha256=digest(folder/'sample.json'),
            depth_sha256=digest(folder/'depth_m.npy'),validity_sha256=digest(folder/'validity.png'),
            calibration_sha256=digest(folder.parent.parent/'calibration.json'),
            rgb_device_monotonic_s=m['streams']['rgb']['device_monotonic_s'],
            depth_device_monotonic_s=m['streams']['depth']['device_monotonic_s'],
            source_sync_span_ms=m['sync_span_ms'],
            raw_confidence_policy='Not consumed or reinterpreted; binary validity only')
    return record

def canonical(vertices):
    lo,hi=vertices.min(0),vertices.max(0)
    q=(vertices-lo)/(hi-lo)
    q[:,:2]-=.5
    return q

def deform(q,length,width_ratio,height_ratio,taper=0.):
    v=q.copy();v[:,1]*=1+taper*2*v[:,0]
    v*=np.array([length,length*width_ratio,length*height_ratio])
    return v

def rotation(angle,tilt):
    a,t=np.deg2rad([angle,180-tilt])
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
    return rz@rx

def project(vertices,K,R,t):
    p=vertices@R.T+t
    if np.any(p[:,2]<=0):raise ValueError('Geometry is behind camera')
    h=p@K.T
    return h[:,:2]/h[:,2,None],p[:,2]

def resample_polygon(poly,count=96):
    p=np.asarray(poly,float);p=np.vstack((p,p[0]))
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1);cum=np.r_[0,np.cumsum(lengths)]
    s=np.linspace(0,cum[-1],count,endpoint=False)
    return np.column_stack([np.interp(s,cum,p[:,j]) for j in range(2)])

def pose_from_parameters(q,p,K,fixed_length=None):
    angle,tilt,width,height,taper,z,u,v=p[:8]
    length=p[8] if fixed_length is None else fixed_length
    verts=deform(q,length,width,height,taper)
    R=rotation(angle,tilt)
    t=np.linalg.solve(K,np.array([u,v,1.]))*z
    return verts,R,t

def fit(q,polygon,K,profile,depth_hint=None,seed=42):
    target=resample_polygon(polygon);tree=cKDTree(target)
    center=np.mean(target,0);extent=np.ptp(target,0);diameter=np.linalg.norm(extent)
    # Fitting a convex envelope deliberately does not pretend to repair holes in the scan.
    indices=np.unique(np.linspace(0,len(q)-1,min(len(q),3000),dtype=int));sub=q[indices]
    fixed=profile['nominal_length_m'] if depth_hint is None else None
    z0=(fixed or .12)*K[0,0]/max(extent)
    if depth_hint is not None:z0=depth_hint+.03
    bounds=[(-180,180),(0,65),tuple(profile['width_ratio_bounds']),tuple(profile['height_ratio_bounds']),
            (-.15,.15),(max(.15,z0*.65),z0*1.65),
            (center[0]-extent[0]*.25,center[0]+extent[0]*.25),
            (center[1]-extent[1]*.25,center[1]+extent[1]*.25)]
    if fixed is None:bounds.append(tuple(profile['length_bounds_m']))
    def cost(p):
        verts,R,t=pose_from_parameters(sub,p,K,fixed)
        try:uv,z=project(verts,K,R,t)
        except ValueError:return 1e6
        hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
        boundary=resample_polygon(hull)
        d1=tree.query(boundary)[0];d2=cKDTree(boundary).query(target)[0]
        value=(np.mean(d1*d1)+np.mean(d2*d2))/diameter**2
        # Weak shape priors: modeling constraints, not measured species ratios.
        value+=.0004*((p[2]-profile['width_ratio'])/.15)**2
        value+=.0004*((p[3]-profile['height_ratio'])/.12)**2+.0002*(p[4]/.15)**2
        if depth_hint is not None:
            value+=.2*((np.quantile(z,.15)-depth_hint)/depth_hint)**2
        return float(value)
    result=differential_evolution(cost,bounds,seed=seed,maxiter=65,popsize=7,polish=False,tol=.0001)
    refined=minimize(cost,result.x,method='Powell',bounds=bounds,options={'maxiter':160,'xtol':1e-5,'ftol':1e-7})
    p=refined.x if refined.fun<result.fun else result.x
    verts,R,t=pose_from_parameters(q,p,K,fixed);uv,z=project(verts,K,R,t)
    hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
    border=resample_polygon(hull)
    rms=float(np.sqrt((np.mean(tree.query(border)[0]**2)+np.mean(cKDTree(border).query(target)[0]**2))/2))
    return {'parameters':p.tolist(),'objective':cost(p),'contour_rms_px':rms,
        'length_m':float(fixed or p[8]),'width_ratio':float(p[2]),'height_ratio':float(p[3]),
        'taper':float(p[4]),'R':R.tolist(),'t':t.tolist(),'K':K.tolist(),
        'estimated_front_surface_z_m':float(np.quantile(z,.15)),
        'metric_scale':'rough stereo-constrained estimate' if depth_hint else 'assumed species-size scenario; RGB scale is ambiguous'}

def fit_sources(template,config,output):
    output=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    data=np.load(template);q=canonical(data['vertices']);faces=data['faces']
    results=[]
    for source in config['sources']:
        evidence=source_evidence(source)
        image=cv2.imread(str(ROOT/source['image']));h,w=image.shape[:2]
        polygon=np.array(source['shell_polygon_xy'],np.int32)
        mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[polygon],255)
        folder=output/source['id'];folder.mkdir(exist_ok=False)
        cv2.imwrite(str(folder/'source-mask.png'),mask)
        hint=None;depth_stats=None
        if source.get('oak_sample'):
            sample=ROOT/source['oak_sample'];meta=json.loads((sample/'sample.json').read_text())
            if meta['depth_alignment']!='rgb' or meta['depth_units']!='meters':raise ValueError('Invalid OAK depth contract')
            geometry=meta['streams']['rgb']['geometry']
            if any(geometry['distortion_coefficients']):raise ValueError('Undistort before fitting')
            K=np.array(geometry['intrinsics']);cameras=[K]
            depth=np.load(sample/'depth_m.npy');valid=cv2.imread(str(sample/'validity.png'),0)>0
            inner=cv2.erode(mask,np.ones((9,9),np.uint8))>0
            good=inner&valid&np.isfinite(depth)&(depth>0)
            values=depth[good]
            depth_stats={'interior_pixels':int(inner.sum()),'valid_pixels':len(values),
                'valid_fraction':float(good.sum()/max(inner.sum(),1))}
            if len(values)>=50 and depth_stats['valid_fraction']>=.3:
                quantiles=np.quantile(values,[.1,.5,.9]);depth_stats['quantiles_m']=quantiles.tolist()
                if (quantiles[2]-quantiles[0])/quantiles[1]<.15:hint=float(quantiles[1])
            camera_kind='measured OAK per-frame intrinsics; undistorted RGB'
        else:
            cameras=[np.array([[s*w,0,w/2],[0,s*w,h/2],[0,0,1.]]) for s in [.8,1.2,1.8]]
            camera_kind='uncalibrated focal hypotheses; centered principal point, distortion unknown'
        trials=[]
        for i,K in enumerate(cameras):
            result=fit(q,polygon,K,config['profiles'][source['profile']],hint,seed=42+i)
            trials.append(result);print(source['id'],i,result['contour_rms_px'],flush=True)
        result=min(trials,key=lambda r:r['objective'])
        verts=deform(q,result['length_m'],result['width_ratio'],result['height_ratio'],result['taper'])
        uv,z=project(verts,np.array(result['K']),np.array(result['R']),np.array(result['t']))
        np.savez_compressed(folder/'fit.npz',vertices=verts,faces=faces,uv_pixels=uv,source_depth=z)
        preview=image.copy();cv2.polylines(preview,[polygon],True,(0,255,255),2)
        cv2.polylines(preview,[cv2.convexHull(uv.astype(np.float32)).astype(np.int32)],True,(255,255,0),2)
        cv2.imwrite(str(folder/'fit-overlay.png'),preview)
        report={**source,**result,'source_evidence':evidence,'camera_kind':camera_kind,'depth_statistics':depth_stats,
            'focal_hypotheses':trials,'profile':config['profiles'][source['profile']],
            'image_size_wh':[w,h],'mask_review':'agent-traced approximate shell outline; not human-reviewed ground truth',
            'fit_method':'regularized convex-silhouette fit of deformed scan; sparse stereo summary is a weak scale constraint',
            'scale_and_shape_uncertainty':'Single-view fit is underdetermined; low contour error is not 3D accuracy',
            'fit_folder':str(folder.relative_to(ROOT))}
        (folder/'fit.json').write_text(json.dumps(report,indent=2)+'\n');results.append(report)
    (output/'fits.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    fit_sources(args.template,json.loads(args.config.read_text()),args.output)
