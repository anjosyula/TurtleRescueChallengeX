"""Render the measured RGB/P2 comparison as a local, standalone figure."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train_challenge import ROOT
from evaluate_challenge import match_predictions


def experiment_label(name):
    if 'blender_control' in name or name.endswith('_control'):
        return 'Real-only fine-tune'
    if 'blender_synthetic' in name or name.endswith('_synthetic'):
        return 'Real + Blender'
    if name.startswith('p2_small'):
        return 'P2 small-scale fine-tune'
    return 'RGB + P2' if name.startswith('p2') else 'Stock RGB'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path,
                   default=ROOT/'runs/challenge/comparison_20261019/comparison.json')
    args = p.parse_args()
    reports = json.loads(args.comparison.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#247ba0', '#c46628', '#51883f', '#9455a6']
    for (name, r), color in zip(reports.items(), colors):
        label = experiment_label(name)
        results = Path(r['checkpoint']).parent.parent/'results.csv'
        with results.open() as stream:
            rows = list(csv.DictReader(stream))
        axes[0, 0].plot([int(x['epoch']) for x in rows],
                       [float(x['metrics/mAP50-95(B)']) for x in rows], label=label, color=color)
        samples = json.loads((args.comparison.parent/name/'predictions.json').read_text())
        ranked = sorted([v for s in samples for v in match_predictions(s['predictions'], s['truth'])[0]],
                        key=lambda x: -x[0])
        scores = np.array([s for s, _ in ranked])
        tp = np.cumsum([v for _, v in ranked])
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
        recall = tp[ends]/r['objects']
        precision = tp[ends]/(ends+1)
        axes[0, 1].plot(recall, precision, label=label, color=color)
        point = r['operating_points_IoU50']['best_F2']
        axes[0, 1].scatter(point['recall'], point['precision'], color=color, s=45, zorder=3)
    names = list(reports)
    labels = [experiment_label(n) for n in names]
    x = np.arange(len(names))
    for offset, key, label in [(-.18, 'AP', 'AP50–95'), (.18, 'AP50', 'AP50')]:
        bars = axes[1, 0].bar(x+offset, [reports[n]['coco'][key] for n in names], .34, label=label)
        axes[1, 0].bar_label(bars, fmt='%.3f', padding=3)
    axes[1, 0].set_xticks(x, labels)
    sizes = ['small', 'medium', 'large']
    width=.8/len(names)
    for i, (name, color) in enumerate(zip(names, colors)):
        data = reports[name]['recall_by_size_at_640']
        bars = axes[1, 1].bar(np.arange(3)+(i-(len(names)-1)/2)*width,
                             [data[k]['recall'] or 0 for k in sizes], width, label=labels[i], color=color)
        axes[1, 1].bar_label(bars, labels=[f"{data[k]['matched']}/{data[k]['objects']}" for k in sizes], padding=3)
    axes[1, 1].set_xticks(np.arange(3), ['Small (<32²)', 'Medium (32²–96²)', 'Large (>96²)'])
    titles = ['Training-time validation AP50–95', 'Fixed evaluation: precision / recall at IoU 0.5',
              'Fixed evaluation: COCO AP', 'Recall at selected F2 threshold, by area at 640 input']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_xlim(0, 1.02)
    fig.suptitle('TurtleHero — local RGB model comparison', fontsize=17)
    fig.text(.5, .025, '50 validation images / 51 turtles; used for model and threshold selection. No independent test or turtle-free scenes.\n'
             'Only 2 small targets at 640 input. Training-time and fixed evaluation use different inference protocols.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .07, 1, .95))
    output = args.comparison.parent/'comparison.png'
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
