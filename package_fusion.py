"""Package a completed local fusion experiment with its actual data and evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_challenge import ROOT,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-evaluation',type=Path)
    args=p.parse_args()
    done=json.loads((args.run/'completed.json').read_text())
    if done['status']!='completed' or not done['frozen_detector_exactly_unchanged']:
        raise ValueError('Training not successfully completed')
    checkpoint=args.run/'turtle_fused.pt'
    if sha256(checkpoint)!=done['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    evaluation=json.loads((args.run/'evaluation/comparison.json').read_text())
    if evaluation['checkpoint_sha256']!=done['checkpoint_sha256']:raise ValueError('Evaluation/checkpoint mismatch')
    provenance=json.loads((args.run/'provenance.json').read_text())
    manifest=json.loads((args.run/'dataset_manifest.json').read_text())
    counts=Counter('gaussian' if r['source'].startswith('RGB-D initialized') else 'blender' for r in manifest)
    base=json.loads((ROOT/'models/challenge-20261019/model.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('turtle_fused.pt','provenance.json','dataset_manifest.json','input_checksums.json','history.json','completed.json'):
        shutil.copy2(args.run/name,args.output/name)
    shutil.copy2(args.run/'evaluation/comparison.json',args.output/'evaluation.json')
    shutil.copy2(ROOT/'requirements-training.txt',args.output/'requirements-training.txt')
    card={'checkpoint':'turtle_fused.pt','checkpoint_sha256':done['checkpoint_sha256'],
        'architecture':'YOLO26n P2 + depth encoder + confidence-gated residual fusion at strides 4/8/16',
        'depth_fusion_trained':True,'real_paired_depth_training':False,'infrared_trained':False,
        'recommended_default':False,'run':str(args.run.resolve()),
        'training':{'epochs':done['epochs'],'paired_images':len(manifest),'domains':dict(counts),
            'positive_images':sum(r['boxes']>0 for r in manifest),'background_images':sum(r['boxes']==0 for r in manifest),
            'initialization':'Original RGB P2 detector; new zero-initialized fusion branch',
            'initial_checkpoint_sha256':provenance['initial_checkpoint_sha256'],'frozen_rgb_weights_and_buffers':True},
        'trainable_parameters':provenance['trainable_parameters'],
        'inputs':base_inputs(),
        'geometry_quality':'Binary validity fallback, uncalibrated; raw OAK confidence is not RGB-aligned and is not consumed',
        'missing_depth':'All three geometry channels zero; exactly frozen RGB fallback',
        'selected_confidence':base['selected_confidence'],
        'threshold_policy':'Inherited RGB F2 threshold; not calibrated for fused outputs',
        'nms_iou':.7,'max_detections':300,'inference_head':'one-to-many + host NMS',
        'checkpoint_selection':'Final fixed epoch; no paired validation set',
        'real_zero_depth_validation':evaluation['reports']['real_zero_depth'],
        'zero_geometry_exact_rgb_parity_images':evaluation['zero_geometry_exact_rgb_parity_images'],
        'zero_geometry_max_raw_error':evaluation['zero_geometry_max_raw_error'],
        'synthetic_training_diagnostics':{k:v['coco'] for k,v in evaluation['reports'].items() if k.startswith('synthetic')},
        'status':'trained experimental; evaluated locally; export tracked separately in export_report.json',
        'limitations':['No reviewed real paired depth training/validation; no field improvement established',
            'All synthetic derivatives share one incomplete shell scan; Gaussian views are correlated augmentations',
            'Assumed 20 cm scale and uncalibrated sensor profiles','Synthetic diagnostics are training-scene results',
            'Training duration and data size can differ between runs; score differences do not isolate augmentation benefit',
            'RGB fallback AP does not measure depth benefit','No phone deployment or on-camera compilation verified']}
    export_dir=args.run/'export'
    if (export_dir/'export_report.json').exists():
        export=json.loads((export_dir/'export_report.json').read_text())
        if export['status']!='passed' or export['checkpoint_sha256']!=done['checkpoint_sha256']:
            raise ValueError('Export does not match completed experiment')
        artifact=export_dir/'turtle_fused.onnx'
        if sha256(artifact)!=export['sha256']:raise ValueError('Export artifact changed')
        shutil.copy2(artifact,args.output/artifact.name)
        export['validated_artifact_path']=export['artifact']
        export['artifact']=str((args.output/artifact.name).resolve())
        write_json(args.output/'export_report.json',export)
        card.update(onnx_export=artifact.name,export_report='export_report.json',
                    status='trained experimental; local evaluation and ONNX parity completed')
    write_json(args.output/'model.json',card)
    history=json.loads((args.run/'history.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axs[0].plot([r['epoch'] for r in history],[r['loss'] for r in history],marker='o')
    axs[0].set(xlabel='Epoch',ylabel='Augmented one-to-many loss',title=f'{len(manifest)} paired synthetic training examples')
    modes=['zero','oak','iphone','oak_shift_12px','oak_depth_reversed']
    labels=['No depth','OAK-like','iPhone-like','Shift 12 px','Depth reversed']
    values=[evaluation['reports']['synthetic_'+m]['coco']['AP']*100 for m in modes]
    bars=axs[1].barh(labels,values,color=['#88949a','#337ea9','#337ea9','#ba9343','#ba9343'])
    axs[1].bar_label(bars,fmt='%.1f%%',padding=3);axs[1].invert_yaxis()
    axs[1].set(xlabel='AP50–95 (%)',xlim=(0,max(values)+12),title='TRAINING-scene diagnostics, not field accuracy')
    real_ap=evaluation['reports']['real_zero_depth']['coco']['AP']*100
    fig.suptitle(f'Gaussian-augmented RGB–depth fusion • Real zero-depth AP: {real_ap:.1f}%')
    fig.savefig(args.output/'comparison.png',dpi=170);plt.close(fig)
    if args.prior_evaluation:
        prior=json.loads(args.prior_evaluation.read_text())
        if prior.get('diagnostic_manifest_sha256')!=evaluation.get('diagnostic_manifest_sha256'):
            raise ValueError('Prior comparison must use the identical diagnostic manifest')
        comparison={'scope':'Same inputs; training-scene diagnostics only, not independent testing or equal-compute ablation',
            'previous_checkpoint':prior['checkpoint_sha256'],'new_checkpoint':evaluation['checkpoint_sha256'],
            'new_training_domains':dict(counts),
            'AP':{k:{'previous':prior['reports'][k]['coco']['AP'],'new':v['coco']['AP']}
                  for k,v in evaluation['reports'].items()},
            'by_domain_AP':{d:{m:{'previous':prior['by_domain'][d][m]['coco']['AP'],'new':v['coco']['AP']}
                              for m,v in reports.items()} for d,reports in evaluation['by_domain'].items()}}
        write_json(args.output/'previous_model_comparison.json',comparison)
    print(args.output.resolve())


def base_inputs():
    return {'image':'B,3,640,640 RGB float32 0..1, centered letterbox',
            'geometry':'B,3,640,640: log camera-axis Z normalized over [0.2,10] meters, validity, quality'}


if __name__=='__main__':main()
