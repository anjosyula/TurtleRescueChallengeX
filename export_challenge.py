"""Export the selected RGB detector and compare raw outputs locally."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback
import cv2
import numpy as np
from train_challenge import ROOT,local_runtime,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-card',type=Path,default=ROOT/'models/challenge-20261019/model.json')
    p.add_argument('--formats',nargs='+',choices=['onnx','coreml'],default=['onnx','coreml'])
    p.add_argument('--coreml-precision',type=int,choices=[16,32],default=32)
    args=p.parse_args();local_runtime()
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    torch.set_num_threads(2)
    card=json.loads(args.model_card.read_text());folder=args.model_card.parent
    checkpoint=folder/card['checkpoint']
    files=[Path(s) for s in (ROOT/'datasets/turtle-challenge-v1/val.txt').read_text().splitlines()][:3]
    inputs=[]
    reference=YOLO(str(checkpoint)).model.float().eval()
    for path in files:
        rgb=LetterBox(new_shape=(640,640),auto=False)(image=cv2.imread(str(path)))[:,:,::-1].copy()
        x=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255
        with torch.no_grad(): y=reference(torch.from_numpy(x))[0].numpy()
        inputs.append((path,rgb,x,y))
    report_path=folder/'export_report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'checkpoint':str(checkpoint),'images':[str(x[0]) for x in inputs],
            'head':'one-to-many; raw xywh and turtle score; host applies NMS',
            'raw_output_shape':list(inputs[0][3].shape),'exports':{}}
    for kind in args.formats:
        if kind in report['exports']:
            report.setdefault('previous_attempts',[]).append({kind:report['exports'][kind]})
        try:
            kwargs={'opset':17,'simplify':False,'dynamic':False} if kind=='onnx' else {'quantize':args.coreml_precision}
            artifact=YOLO(str(checkpoint)).export(format=kind,imgsz=640,batch=1,device='cpu',nms=None,**kwargs)
            differences=[]
            if kind=='onnx':
                import onnxruntime as ort
                options=ort.SessionOptions();options.intra_op_num_threads=2
                runtime=ort.InferenceSession(str(artifact),sess_options=options,providers=['CPUExecutionProvider'])
            else:
                import coremltools as ct
                from PIL import Image
                runtime=ct.models.MLModel(str(artifact),compute_units=ct.ComputeUnit.CPU_ONLY)
            for path,rgb,x,expected in inputs:
                if kind=='onnx':actual=runtime.run(None,{runtime.get_inputs()[0].name:x})[0]
                else:actual=next(iter(runtime.predict({'image':Image.fromarray(rgb)}).values()))
                if actual.shape!=expected.shape:raise ValueError(f'Output shape differs: {actual.shape} vs {expected.shape}')
                if not np.isfinite(actual).all():raise ValueError('Nonfinite exported output')
                box_diff=np.abs(actual[:,:4]-expected[:,:4]);score_diff=np.abs(actual[:,4:]-expected[:,4:])
                differences.append({'image':str(path),'box_max_abs_px':float(box_diff.max()),
                    'box_mean_abs_px':float(box_diff.mean()),'score_max_abs':float(score_diff.max())})
                # Numerical conversion tolerances only, not detector safety criteria.
                box_tol,score_tol=(.1,.001) if kind=='onnx' or args.coreml_precision==32 else (5.,.02)
                if box_diff.max()>box_tol or score_diff.max()>score_tol:
                    raise ValueError(f'Raw parity outside {box_tol}px / {score_tol} score tolerance: {differences[-1]}')
            report['exports'][kind]={'status':'passed','artifact':str(artifact),'parity':differences,
                'precision':args.coreml_precision if kind=='coreml' else 32,
                'box_max_tolerance_px':box_tol,'score_max_tolerance':score_tol,
                'note':'Three-image raw-output parity only; exported-model task metrics and target-device latency remain separate checks.'}
        except Exception as exc:
            report['exports'][kind]={'status':'failed','error':str(exc),'traceback':traceback.format_exc()}
        write_json(folder/'export_report.json',report)
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    write_json(folder/'export_environment.json',versions)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
