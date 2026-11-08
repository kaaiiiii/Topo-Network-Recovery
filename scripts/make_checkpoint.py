"""Make the ridge checkpoint figure and bootstrap statistics from saved results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_summary(metrics_path):
    with metrics_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    draws = np.random.default_rng(5420).integers(0, 10, size=(10000, 10))
    result = []
    for length in [100, 300, 1000]:
        selected = sorted((r for r in rows if r['method']=='ridge' and
                           float(r['noise'])==.25 and int(r['length'])==length), key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in selected] == list(range(1001,1011))
        summary = dict(length=length, selected_alphas=[float(r['alpha']) for r in selected],
                       cv_rmse_mean=float(np.mean([float(r['cv_rmse']) for r in selected])),
                       cv_fold_sd_mean=float(np.mean([float(r['cv_fold_sd']) for r in selected])))
        for metric in ['test_rmse', 'average_precision']:
            values = np.array([float(r[metric]) for r in selected])
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
            summary[metric] = dict(mean=float(values.mean()), low=float(low), high=float(high),
                                   values=values.tolist())
        result.append(summary)
    return result


def plot(summary, output):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1,2,figsize=(11.5,5.3))
    x = np.arange(3)
    panels = [('test_rmse','Next-state prediction','Test RMSE (state units)','#146b81'),
              ('average_precision','Directed-edge recovery','Average precision','#6c4893')]
    for ax, (metric, title, ylabel, color) in zip(axes,panels):
        means = np.array([r[metric]['mean'] for r in summary])
        low = np.array([r[metric]['low'] for r in summary])
        high = np.array([r[metric]['high'] for r in summary])
        values = np.array([r[metric]['values'] for r in summary])
        for seed in range(10):
            ax.plot(x, values[:,seed],color=color,alpha=.16,lw=.9)
        ax.errorbar(x,means,yerr=np.array([means-low,high-means]),fmt='o-',color=color,
                    linewidth=2.5,markersize=7,capsize=6,capthick=1.8,label='Mean and 95% bootstrap CI')
        for i, value in enumerate(means):
            offset = (0,15) if metric=='test_rmse' else (0,-25)
            ax.annotate(f'{value:.3f}',(i,value),xytext=offset,textcoords='offset points',
                        ha='center',fontsize=11,color=color,fontweight='bold')
        ax.set(title=title,ylabel=ylabel,xlabel='Training transitions per network',
               xticks=x,xticklabels=['100','300','1000'],xlim=(-.18,2.18))
        ax.grid(axis='y',alpha=.16)
        if metric=='test_rmse':
            ax.set_ylim(1.075,1.24)
            ax.legend(loc='upper right',frameon=False,fontsize=9)
        else:
            ax.set_ylim(0,1.04)
            ax.axhline(300/9900,color='#777',ls='--',lw=1)
            ax.text(.02,.055,'Edge prevalence = 0.0303',transform=ax.transAxes,color='#666',fontsize=9)
    fig.suptitle('Ridge baseline: more observations improve prediction and network recovery',fontsize=14,y=.98)
    fig.text(.5,.90,'100 nodes, 300 directed edges | measurement-noise SD / stationary SD = 0.25',
             ha='center',fontsize=10,color='#444')
    fig.text(.5,.035,'10 independent network / trajectory draws; 10,000 bootstrap resamples of their mean.\n'
             'Five chronological CV folds select ridge penalties; prediction uses separate 1,000-transition test trajectories.',
             ha='center',fontsize=9,color='#444')
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.78,wspace=.28)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/checkpoint/baseline.png')
    parser.add_argument('--metrics',type=Path,default=ROOT/'reference_results/baseline/metrics.csv')
    args = parser.parse_args()
    summary = checkpoint_summary(args.metrics)
    plot(summary,args.output)
    print(json.dumps(summary[-1],indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
