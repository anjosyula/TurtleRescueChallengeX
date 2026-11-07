"""Export a fixed 640-square, two-input fusion ONNX graph and verify CPU parity."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from train_challenge import ROOT,local_runtime,write_json,sha256
from train_fusion import load_fused,PairedShellDataset
from turtle_geometry import prepare_rgb_geometry


class ExportFusion(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,image,geometry):
        return self.model(image,geometry)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'datasets/turtle-real-synthetic-v1/manifest.json')
    args=p.parse_args();local_runtime();torch.set_num_threads(2)
    import onnx
    import onnxruntime as ort
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'turtle_fused.onnx'
    if target.exists():raise FileExistsError(target)
    model,_=load_fused(args.checkpoint,'cpu');wrapper=ExportFusion(model).eval()
    data=PairedShellDataset(args.manifest,augment=False)
    fixtures=[]
    for index,profile in [(0,'oak'),(14,'iphone'),(20,'oak')]:
        image,geometry,_=data.get(index,profile)
        fixtures.append((f'synthetic_{index}_{profile}',image[None],geometry[None]))
    gaussian=[i for i,row in enumerate(data.rows) if row['source'].startswith('RGB-D initialized')]
    if gaussian:
        for index,profile in [(gaussian[0],'oak'),(gaussian[len(gaussian)//2],'iphone'),(gaussian[-1],'oak')]:
            image,geometry,_=data.get(index,profile)
            fixtures.append((f'gaussian_{index}_{profile}',image[None],geometry[None]))
    image_path=Path((ROOT/'datasets/turtle-challenge-v1/val.txt').read_text().splitlines()[0])
    image,geometry,_=prepare_rgb_geometry(cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB))
    fixtures.append(('real_zero_depth',image[None],geometry[None]))
    args0=tuple(torch.from_numpy(v) for v in fixtures[0][1:])
    with torch.inference_mode():
        wrapper(*args0)  # Initialize fixed detection grids before tracing.
        torch.onnx.export(wrapper,args0,str(target),input_names=['image','geometry'],output_names=['detections'],
                          opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(target))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    runtime=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    checks=[]
    with torch.inference_mode():
        for name,image,geometry in fixtures:
            expected=wrapper(torch.from_numpy(image),torch.from_numpy(geometry)).numpy()
            actual=runtime.run(None,{'image':image,'geometry':geometry})[0]
            if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Invalid exported output')
            error=np.abs(actual-expected)
            result={'fixture':name,'box_max_abs_px':float(error[:,:4].max()),'score_max_abs':float(error[:,4:].max())}
            checks.append(result)
            if result['box_max_abs_px']>.1 or result['score_max_abs']>.001:
                write_json(args.output/'export_report.json',{'status':'failed','parity':checks})
                raise ValueError(f'Export parity outside numerical tolerances: {result}')
    report={'status':'passed','artifact':str(target),'sha256':sha256(target),
        'checkpoint_sha256':sha256(args.checkpoint),'inputs':{'image':[1,3,640,640],'geometry':[1,3,640,640]},
        'fixture_manifest_sha256':sha256(args.manifest),
        'output':[1,5,34000],'output_semantics':'xywh pixels and turtle score; host NMS required',
        'parity':checks,'tolerance':{'box_px':.1,'score':.001},
        'versions':{n:importlib.metadata.version(n) for n in ('torch','ultralytics','onnx','onnxruntime')},
        'target_runtime':'ONNX Runtime CPU on host; no RVC2 compilation or phone deployment claimed'}
    write_json(args.output/'export_report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
