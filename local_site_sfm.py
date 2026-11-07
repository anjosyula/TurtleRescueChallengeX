"""Local CPU COLMAP feasibility test of checksum-verified site video frames."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pycolmap

from train_challenge import sha256, write_json


def reconstruct(scene, output):
    metadata=json.loads((scene/'scene.json').read_text())
    for row in metadata['frames']:
        if sha256(scene/row['file_path'])!=row['sha256']:raise ValueError('Changed input frame')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_scene.json',metadata)
    database=output/'database.db'
    extraction=pycolmap.FeatureExtractionOptions()
    extraction.num_threads=4;extraction.max_image_size=1280;extraction.sift.max_num_features=4096
    matching=pycolmap.FeatureMatchingOptions();matching.num_threads=4
    options=pycolmap.IncrementalPipelineOptions()
    options.num_threads=4;options.random_seed=42;options.max_runtime_seconds=240
    options.min_model_size=10;options.max_num_models=3
    options.structure_less_registration_fallback=False
    options.mapper.init_min_tri_angle=4.0
    options.min_focal_length_ratio=.5;options.max_focal_length_ratio=3.0
    write_json(output/'settings.json',dict(pycolmap=pycolmap.__version__,extraction=extraction.summary(),
        matching=matching.summary(),mapper_summary=options.summary(),
        calibration='Shared SIMPLE_RADIAL camera estimated from video, not measured',metric_scale=None))
    start=time.perf_counter()
    pycolmap.extract_features(database,scene/'images',camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction,device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database,matching_options=matching,device=pycolmap.Device.cpu)
    models=pycolmap.incremental_mapping(database,scene/'images',output/'sparse',options=options)
    summaries=[]
    for index,model in models.items():
        angles=[]
        for point in model.points3D.values():
            centers=np.array([model.images[t.image_id].projection_center() for t in point.track.elements])
            vectors=centers-point.xyz
            vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            if len(vectors)>1:angles.append(float(np.rad2deg(np.arccos(np.clip((vectors@vectors.T).min(),-1,1)))))
        summaries.append(dict(model=index,registered=model.num_reg_images(),points=model.num_points3D(),
            mean_reprojection_error_px=model.compute_mean_reprojection_error(),
            mean_track_length=model.compute_mean_track_length(),
            median_max_triangulation_angle_deg=float(np.median(angles)) if angles else 0,
            cameras=[dict(id=k,model=v.model_name,width=v.width,height=v.height,params=v.params.tolist()) for k,v in model.cameras.items()]))
    best=max(summaries,key=lambda r:r['registered'],default=None)
    passed=bool(best and best['registered']>=.8*len(metadata['frames']) and best['points']>=300
                and best['median_max_triangulation_angle_deg']>=2 and best['mean_reprojection_error_px']<2)
    report=dict(status='geometry_candidate' if passed else 'insufficient_geometry',models=summaries,
        selected_model=best['model'] if best else None,input_frames=len(metadata['frames']),
        seconds=time.perf_counter()-start,screening_passed=passed,metric_scale=None,
        screening='>=80% registration, >=300 points, median maximum triangulation angle >=2 degrees, mean reprojection error <2 pixels; engineering screening, not accuracy certification',
        limitations=['No measured scale or depth','Dynamic objects and video stabilization can bias geometry'])
    write_json(output/'sfm_report.json',report)
    if passed:
        pycolmap.undistort_images(output/'undistorted',output/'sparse'/str(best['model']),scene/'images',num_threads=4)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reconstruct(a.scene.resolve(),a.output.resolve())
