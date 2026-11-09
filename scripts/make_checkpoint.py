"""Make the checkpoint figure and bootstrap statistics from saved results.

Panels: test RMSE and edge-recovery AP versus training length for the ridge
baseline and the Lasso comparison, and precision-recall curves at 1000
transitions. Metrics come from --metrics (published snapshot by default); PR
curves need the per-fit arrays from a local run of network_recovery.py (--runs).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NOISE = .25
LENGTHS = [100, 300, 1000]
SEEDS = list(range(1001, 1011))
METHODS = {'ridge': dict(label='Ridge (baseline)', color='#2a78d6', marker='o'),
           'lasso': dict(label='Lasso (comparison)', color='#eb6834', marker='s')}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
PREVALENCE = 300 / 9900


def checkpoint_summary(metrics_path, method='ridge'):
    with metrics_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    draws = np.random.default_rng(5420).integers(0, 10, size=(10000, 10))
    result = []
    for length in LENGTHS:
        selected = sorted((r for r in rows if r['method'] == method and
                           float(r['noise']) == NOISE and int(r['length']) == length), key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in selected] == SEEDS
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


def pr_curves(runs, method, grid):
    """Interpolated precision (max precision at recall >= r) per network, on a recall grid."""
    curves = []
    for seed in SEEDS:
        path = runs / f'seed_{seed}' / f'{method}_T{LENGTHS[-1]}_noise{NOISE:g}.npz'
        if not path.exists():
            raise SystemExit(f'Missing {path}. Run scripts/network_recovery.py first (PR curves need per-fit arrays).')
        saved = np.load(path)
        precision, recall = saved['precision'], saved['recall']   # recall is non-increasing
        envelope = np.maximum.accumulate(precision)               # max precision over recall >= current
        index = np.searchsorted(-recall, -grid, side='right') - 1  # last point with recall >= r
        curves.append(envelope[np.clip(index, 0, None)])
    return np.array(curves)


def style(ax, title, ylabel, xlabel):
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax.tick_params(colors=MUTED, length=3)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#b8b7b0')


def plot(summaries, curves, grid, output):
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.axisbelow': True})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    x = np.arange(len(LENGTHS))
    panels = [('test_rmse', 'Next-state prediction (lower is better)', 'Test RMSE (state units)'),
              ('average_precision', 'Directed-edge recovery (higher is better)', 'Average precision')]
    for ax, (metric, title, ylabel) in zip(axes[:2], panels):
        for k, (method, spec) in enumerate(METHODS.items()):
            summary = summaries[method]
            means = np.array([r[metric]['mean'] for r in summary])
            low = np.array([r[metric]['low'] for r in summary])
            high = np.array([r[metric]['high'] for r in summary])
            shift = (k - .5) * .06   # small dodge so error bars do not overlap
            ax.errorbar(x + shift, means, yerr=[means - low, high - means], color=spec['color'], marker=spec['marker'],
                        lw=2, ms=7, capsize=4, capthick=1.5, mec='white', mew=1.2, label=spec['label'], zorder=3)
            offset = (10, 9 if k == 0 else -9)   # ridge label above, Lasso below
            ax.annotate(f'{means[-1]:.3f}', (x[-1] + shift, means[-1]), xytext=offset, textcoords='offset points',
                        va='center', fontsize=10, color=INK)
        style(ax, title, ylabel, 'Training transitions per network')
        ax.set(xticks=x, xticklabels=[str(n) for n in LENGTHS], xlim=(-.25, 2.45))
        if metric == 'average_precision':
            ax.set_ylim(0, 1.02)
            ax.axhline(PREVALENCE, color=MUTED, ls='--', lw=1)
            ax.text(-.2, PREVALENCE + .025, f'Random ranking = {PREVALENCE:.3f}', color=MUTED, fontsize=9)
    axes[0].legend(loc='upper right', frameon=False)

    ax = axes[2]
    for method, spec in METHODS.items():
        c = curves[method]
        ap = summaries[method][-1]['average_precision']['mean']
        ax.fill_between(grid, c.min(axis=0), c.max(axis=0), color=spec['color'], alpha=.16, lw=0)
        ax.plot(grid, c.mean(axis=0), color=spec['color'], lw=2, label=f"{spec['label']}, AP {ap:.3f}")
    ax.axhline(PREVALENCE, color=MUTED, ls='--', lw=1)
    ax.text(.02, PREVALENCE + .025, f'Random ranking = {PREVALENCE:.3f}', color=MUTED, fontsize=9)
    style(ax, 'Precision-recall at 1000 transitions', 'Precision', 'Recall (fraction of true edges found)')
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(axis='x', color=GRID, lw=.8)
    ax.legend(loc='center left', frameon=False)

    # Setup, uncertainty and the Lasso tie at recall ~0.92 are described in the caption, not on the figure.
    fig.subplots_adjust(left=.06, right=.985, bottom=.13, top=.9, wspace=.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/checkpoint/baseline.png')
    parser.add_argument('--metrics', type=Path, default=ROOT / 'reference_results/baseline/metrics.csv')
    parser.add_argument('--runs', type=Path, default=ROOT / 'results/baseline')
    args = parser.parse_args()
    summaries = {m: checkpoint_summary(args.metrics, m) for m in METHODS}
    grid = np.linspace(0, 1, 201)
    curves = {m: pr_curves(args.runs, m, grid) for m in METHODS}
    plot(summaries, curves, grid, args.output)
    print(json.dumps(summaries['ridge'][-1], indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
