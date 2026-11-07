"""Experimental independent RGB/depth alerts; missing depth never vetoes RGB."""
import numpy as np
from evaluate_challenge import overlaps


def checked_predictions(value):
    a=np.asarray(value,dtype=float).reshape(-1,6)
    if not np.isfinite(a).all() or np.any(a[:,4]<0) or np.any(a[:,4]>1):
        raise ValueError('Predictions must have finite coordinates and scores in [0,1]')
    if np.any(a[:,2:4]<=a[:,:2]) or np.any(a[:,5]!=0):
        raise ValueError('Expected positive-area turtle boxes')
    return a


def late_or(rgb, depth, rgb_threshold, depth_threshold, depth_available, duplicate_iou=.7):
    """Inputs use identical original-image coordinates, [x1,y1,x2,y2,score,class].

    RGB boxes retain their coordinates and scores. A matching depth detection
    contributes provenance; unmatched depth boxes remain separate alerts. Scores
    from independently trained branches are never averaged or multiplied.
    """
    if not all(np.isfinite(x) and 0<=x<=1 for x in (rgb_threshold,depth_threshold,duplicate_iou)):
        raise ValueError('Thresholds must lie in [0,1]')
    r=checked_predictions(rgb)
    d=checked_predictions(depth) if depth_available and depth is not None else np.empty((0,6))
    if depth_available and depth is None: raise ValueError('Available depth branch needs predictions')
    r=r[r[:,4]>=rgb_threshold]; d=d[d[:,4]>=depth_threshold]
    detections=[dict(box=b[:4].tolist(),branches={'rgb':float(b[4])},members={'rgb':b.tolist()}) for b in r]
    for b in d[np.argsort(-d[:,4],kind='stable')]:
        ious=overlaps(b,r[:,:4]) if len(r) else np.empty(0)
        if len(ious) and ious.max()>=duplicate_iou:
            hit=detections[int(ious.argmax())]
            if 'depth' not in hit['branches']:
                hit['branches']['depth']=float(b[4]);hit['members']['depth']=b.tolist()
        else:
            detections.append(dict(box=b[:4].tolist(),branches={'depth':float(b[4])},members={'depth':b.tolist()}))
    return dict(possible_turtle=bool(len(r) or len(d)),rgb_hit=bool(len(r)),
        depth_hit=bool(len(d)) if depth_available else None,
        depth_status='available' if depth_available else 'unavailable',detections=detections)
