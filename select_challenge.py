"""Package the best new RGB experiment by validation COCO AP50-95."""
import argparse
import json
from pathlib import Path
import shutil
from train_challenge import ROOT,local_runtime,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,default=ROOT/'runs/challenge/comparison_20261019/comparison.json')
    p.add_argument('--output',type=Path,default=ROOT/'models/challenge-20261019')
    args=p.parse_args();local_runtime()
    reports=json.loads(args.comparison.read_text())
    if len(reports)<2:raise ValueError('Complete both RGB and P2 evaluations before model selection')
    selected=max(reports,key=lambda key:reports[key]['coco']['AP'])
    result=reports[selected]
    if args.output.exists():raise FileExistsError('Choose a new output path; existing models are preserved')
    args.output.mkdir(parents=True)
    shutil.copy2(result['checkpoint'],args.output/'turtle.pt')
    from ultralytics import YOLO
    model=YOLO(str(args.output/'turtle.pt'))
    source_run=Path(result['checkpoint']).parent.parent
    dataset=json.loads((source_run/'dataset_manifest.json').read_text())
    train_rows=[r for r in dataset if r['split']=='train']
    synthetic_count=sum(bool(r.get('synthetic')) for r in train_rows)
    audit_path=source_run/'dataset_audit.json'
    audit=json.loads(audit_path.read_text())
    card={'checkpoint':'turtle.pt','checkpoint_sha256':sha256(args.output/'turtle.pt'),
          'selected_experiment':selected,'selection_metric':'validation COCO AP50-95',
          'selected_confidence':result['selected_confidence'],'threshold_policy':result['threshold_selection'],
          'class_names':{0:'turtle'},'input':'RGB, 640x640 centered letterbox, float32/255',
          'detection_strides':model.model.stride.tolist(),'inference_head':'one-to-many with NMS, nms=None',
          'nms_iou':.7,'validation_metrics':result['coco'],
          'validation_operating_point':result['operating_points_IoU50']['best_F2'],
          'training_data':f'{len(train_rows)-synthetic_count} real RGB photos and {synthetic_count} synthetic shell-proxy scenes',
          'training_source_counts':{'real':len(train_rows)-synthetic_count,'synthetic':synthetic_count},
          'validation_data':f"{result['images']} images, {result['objects']} turtles; used for model and threshold selection, not independent testing",
          'depth_fusion_trained':False,'synthetic_scan_training':bool(synthetic_count),'infrared_trained':False,
          'dataset_limitations':audit.get('limitations',[]),
          'limitations':result['limitations']+['No construction-site acceptance testing or target-device runtime validation',
                       'Seeded MPS runs can vary because some reductions are not deterministic'],
          'comparison':str(args.comparison.resolve())}
    write_json(args.output/'model.json',card)
    shutil.copy2(args.comparison,args.output/'comparison.json')
    shutil.copy2(audit_path,args.output/'dataset_audit.json')
    shutil.copy2(ROOT/'requirements-training.txt',args.output/'requirements-training.txt')
    print(json.dumps(card,indent=2))


if __name__=='__main__':main()
