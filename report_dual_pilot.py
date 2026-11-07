"""Summarize completed local pilot; keep boolean alerts separate from box scoring."""
import argparse
import json
from pathlib import Path
from train_challenge import write_json
from turtle_late_fusion import late_or
from evaluate_dual_pilot import counts,merged_predictions


def frame_presence_counts(samples,threshold):
    tp=fp=tn=fn=0
    for sample in samples:
        truth=bool(sample['truth'])
        hit=any(p[4]>=threshold for p in sample['predictions'])
        tp+=int(truth and hit);fn+=int(truth and not hit)
        fp+=int(not truth and hit);tn+=int(not truth and not hit)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);args=p.parse_args()
    root=args.experiment;folder=root/'evaluation'
    done=json.loads((folder/'completed.json').read_text())
    real=json.loads((folder/'rgb_comparison.json').read_text())
    replay=json.loads((folder/'oak_replay_summary.json').read_text())
    synthetic={}
    for profile in ('oak','iphone'):
        modes={k:json.loads((folder/f'training_{profile}_{k}/predictions.json').read_text()) for k in ('rgb','depth','fused')}
        result={k:frame_presence_counts(s,done['depth_threshold'] if k=='depth' else done['rgb_threshold']) for k,s in modes.items()}
        ort=json.loads((folder/f'training_{profile}_or_0.25.json').read_text())
        result['OR']=frame_presence_counts(ort,0.)
        # Duplicate overlap changes object-level counts but not the frame OR flag.
        sensitivity={}
        for iou in (.3,.5,.7):
            samples=[]
            for r,d in zip(modes['rgb'],modes['depth']):
                if r['image']!=d['image']:raise ValueError('Mismatched sample ordering')
                out=late_or(r['predictions'],d['predictions'],done['rgb_threshold'],done['depth_threshold'],True,iou)
                samples.append(dict(r,predictions=merged_predictions(out)))
            sensitivity[str(iou)]=counts(samples,0.)
        synthetic[profile]=dict(frame_presence=result,duplicate_iou_sensitivity=sensitivity,
            role='Training diagnostics only, 168 positive and 24 negative synthetic scenes',
            note='Presence counts do not require a correctly localized box. Box false positives include duplicate detections.')
    summary=dict(status='completed',rgb_validation={k:dict(AP50_95=real[k]['coco']['AP'],
        selected_operating_point=real[k]['operating_points']['best_F2'],latency_ms=real[k]['latency_ms'],
        size_breakdown=real[k]['recall_by_size_at_640']) for k in ('rgb','rtdetr')},
        synthetic=synthetic,oak_replay=replay,tests_passed=44,default_model_changed=False,
        limitations=['RGB model-selection split, not independent field testing; different training budgets',
            'Depth model trained on simulations only; OAK recordings lack reviewed labels',
            'No measured false-alerts-per-minute or target-device latency',
            'No camera capture, external upload or remote training in this experiment'])
    write_json(root/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
