"""Measured-depth preprocessing. No monocular completion or hole filling."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    left: int
    top: int
    source_wh: tuple[int,int]
    target_wh: tuple[int,int]

    def boxes(self, xyxy, inverse=False):
        boxes=np.asarray(xyxy,dtype=np.float32).copy()
        offset=np.array([self.left,self.top,self.left,self.top],np.float32)
        return (boxes-offset)/self.scale if inverse else boxes*self.scale+offset


def prepare_rgb_geometry(rgb, depth_m=None, validity=None, aligned_confidence=None,
                         target_wh=(640,640), min_m=.2, max_m=10.):
    """RGB uint8 + camera-axis meters -> image/geometry CHW floats and transform.

    Confidence must already share RGB pixel coordinates, with 1 best and 0 worst.
    Never pass the OAK raw stereo confidence map here. When aligned confidence is
    unavailable, binary measured validity is an explicit uncalibrated quality gate.
    Missing geometry produces three zero channels.
    """
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Expected uint8 RGB HWC image')
    if not 0<min_m<max_m or not np.isfinite([min_m,max_m]).all():
        raise ValueError('Depth normalization bounds must be finite and positive')
    h,w=rgb.shape[:2];tw,th=target_wh
    if min(tw,th)<=0 or tw%32 or th%32:
        raise ValueError('Target dimensions must be positive multiples of 32')
    scale=min(tw/w,th/h);nw,nh=round(w*scale),round(h*scale)
    left,top=round((tw-nw)/2-.1),round((th-nh)/2-.1)
    transform=LetterboxTransform(scale,left,top,(w,h),target_wh)
    image=np.full((th,tw,3),114,np.uint8)
    image[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
    geometry=np.zeros((th,tw,3),np.float32)
    if depth_m is not None:
        d=np.asarray(depth_m,dtype=np.float32)
        if d.shape!=(h,w):raise ValueError('Depth must be aligned to RGB before preprocessing')
        valid=np.isfinite(d)&(d>0)&(d>=min_m)&(d<=max_m)
        if validity is not None:
            if np.shape(validity)!=(h,w):raise ValueError('Validity dimensions differ')
            valid &= np.isfinite(validity)&(np.asarray(validity)>0)
        if aligned_confidence is None:
            confidence=valid.astype(np.float32)
        else:
            c=np.asarray(aligned_confidence,dtype=np.float32)
            if c.shape!=(h,w):raise ValueError('Confidence must be explicitly aligned to RGB')
            if np.any(np.isfinite(c)&((c<0)|(c>1))):raise ValueError('Expected normalized confidence in 0..1')
            confidence=np.where(valid&np.isfinite(c),c,0)
        normalized=np.zeros_like(d)
        normalized[valid]=np.log(d[valid]/min_m)/np.log(max_m/min_m)
        source=np.stack((normalized,valid.astype(np.float32),confidence),axis=-1)
        # A nearest sample preserves holes and foreground/background discontinuities.
        geometry[top:top+nh,left:left+nw]=cv2.resize(source,(nw,nh),interpolation=cv2.INTER_NEAREST)
    elif validity is not None or aligned_confidence is not None:
        raise ValueError('Validity/confidence without depth is not measured geometry')
    return np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)/255, np.ascontiguousarray(geometry.transpose(2,0,1)), transform


def reproject_depth(depth_m, source_k, target_k, source_to_target_m, target_wh):
    """Calibrated pinhole reprojection with a nearest-Z visibility buffer.

    Inputs must already be undistorted. Translation is meters; output is target
    camera-axis Z. Unobserved pixels remain zero. This does not densify depth.
    """
    depth=np.asarray(depth_m,dtype=np.float64)
    if depth.ndim!=2:raise ValueError('Depth must be 2-D')
    ks=np.asarray(source_k,dtype=float);kt=np.asarray(target_k,dtype=float);pose=np.asarray(source_to_target_m,dtype=float)
    if ks.shape!=(3,3) or kt.shape!=(3,3) or pose.shape!=(4,4) or not all(np.isfinite(a).all() for a in (ks,kt,pose)):
        raise ValueError('Invalid calibration matrices')
    yy,xx=np.nonzero(np.isfinite(depth)&(depth>0))
    rays=np.linalg.solve(ks,np.stack((xx,yy,np.ones_like(xx))))
    xyz=pose[:3,:3]@(rays*depth[yy,xx])+pose[:3,3:4]
    good=np.isfinite(xyz).all(axis=0)&(xyz[2]>0)
    xyz=xyz[:,good];uv=kt@xyz
    uv=np.rint(uv[:2]/uv[2]).astype(np.int64)
    tw,th=target_wh;inside=(uv[0]>=0)&(uv[0]<tw)&(uv[1]>=0)&(uv[1]<th)
    output=np.full(tw*th,np.inf,np.float64)
    np.minimum.at(output,uv[1,inside]*tw+uv[0,inside],xyz[2,inside])
    output[~np.isfinite(output)]=0
    return output.reshape(th,tw).astype(np.float32)
