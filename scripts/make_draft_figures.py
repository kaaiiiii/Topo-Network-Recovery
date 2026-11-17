"""Figures and LaTeX tables for the early paper draft.

Reads saved results only (no fitting):
  results/baseline/metrics.csv           seven-value grids (checkpoint protocol)
  results/tuning_and_ties/               dense grids and complete Lasso rankings
  results/email_eu_core/dept_*/          real email topologies and their controls
Writes results/draft/: fig_length_noise.pdf/.png and tables.tex.
All intervals bootstrap the 10 network draws (10,000 resamples, seed 5420).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from length_sweep import t_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'draft'
SEEDS = list(range(1001, 1011))
DRAWS = np.random.default_rng(5420).integers(0, 10, size=(10000, 10))
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
SERIES = {'ridge': dict(label='Ridge (baseline)', color='#2a78d6', marker='o'),
          'lasso': dict(label='Lasso, CV-selected model', color='#eb6834', marker='s'),
          'lasso_tiebreak': dict(label='Lasso, complete ranking', color='#1baf7a', marker='^')}
DEPARTMENTS = [4, 14, 1, 21, 15, 7]


def ci(values, log=False):
    values = np.asarray(values, float)
    v = np.log(values) if log else values
    boot = v[DRAWS].mean(axis=1)
    mean, low, high = v.mean(), np.quantile(boot, .025), np.quantile(boot, .975)
    if log:
        mean, low, high = np.exp(mean), np.exp(low), np.exp(high)
    return float(mean), float(low), float(high)


def fmt(m, lo, hi, d=3, sign=False):
    s = '+' if sign else ''
    num = lambda v: f'{v:{s}.{d}f}'.replace('-', '$-$', 1)   # typographic minus in LaTeX
    return f'{num(m)} [{num(lo)}, {num(hi)}]'


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def figure(rows):
    lengths = sorted({int(r['length']) for r in rows})
    noise_levels = sorted({float(r['noise']) for r in rows})
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.axisbelow': True})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, level in zip(axes, noise_levels):
        for k, (score, spec) in enumerate(SERIES.items()):
            stats = []
            for length in lengths:
                group = sorted((r for r in rows if float(r['noise']) == level and int(r['length']) == length),
                               key=lambda r: int(r['seed']))
                stats.append(ci([float(r[f'ap_{score}']) for r in group]))
            m, lo, hi = map(np.array, zip(*stats))
            x = np.array(lengths) * (1 + .035 * (k - 1))      # small dodge on the log axis
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=spec['color'], marker=spec['marker'], ms=4, lw=1.4,
                        capsize=2, mec='white', mew=.6, label=spec['label'], zorder=3)
        ax.axhline(.9, color=MUTED, ls=':', lw=1)
        ax.axhline(300 / 9900, color=MUTED, ls='--', lw=.8)
        ax.set_xscale('log')
        ax.set_xticks([100, 300, 1000, 3000]); ax.set_xticklabels(['100', '300', '1000', '3000'])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_title(f'Noise ratio {level:g}', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('Training transitions $T$', color=MUTED)
        ax.grid(axis='y', color=GRID, lw=.6)
        ax.tick_params(colors=MUTED, length=2)
    axes[0].set_ylabel('Edge-recovery AP', color=MUTED)
    axes[0].text(110, .915, 'AP = 0.9', color=MUTED, fontsize=7.5)
    axes[0].text(110, .05, 'random ranking', color=MUTED, fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.subplots_adjust(left=.08, right=.99, top=.9, bottom=.3, wspace=.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'fig_length_noise.pdf')
    fig.savefig(OUT / 'fig_length_noise.png', dpi=220)
    plt.close(fig)


def main_table(baseline, fine):
    """T=1000, noise 0.25: checkpoint grid versus dense grid, with paired differences against ridge."""
    pick = lambda rows, method: sorted((r for r in rows if r['method'] == method and float(r['noise']) == .25
                                        and int(r['length']) == 1000), key=lambda r: int(r['seed']))
    ridge0, lasso0 = pick(baseline, 'ridge'), pick(baseline, 'lasso')
    dense = sorted((r for r in fine if float(r['noise']) == .25 and int(r['length']) == 1000), key=lambda r: int(r['seed']))
    ap = lambda rows, key='average_precision': [float(r[key]) for r in rows]
    lines = []
    lines.append(('Ridge (baseline)', 'seven values', fmt(*ci(ap(ridge0))), '---',
                  fmt(*ci(ap(ridge0, 'test_rmse')), d=4)))
    lines.append(('Lasso', 'seven values', fmt(*ci(ap(lasso0))),
                  fmt(*ci(np.subtract(ap(lasso0), ap(ridge0))), sign=True), fmt(*ci(ap(lasso0, 'test_rmse')), d=4)))
    r_d = [float(r['ap_ridge']) for r in dense]
    for name, key in [('Ridge (baseline)', 'ap_ridge'), ('Lasso', 'ap_lasso'), ('Lasso, complete ranking', 'ap_lasso_tiebreak')]:
        v = [float(r[key]) for r in dense]
        diff = '---' if key == 'ap_ridge' else fmt(*ci(np.subtract(v, r_d)), sign=True)
        lines.append((name, 'four per decade', fmt(*ci(v)), diff, 'to run'))
    body = '\n'.join(f'{a} & {b} & {c} & {d} & {e} \\\\' + (' \\midrule' if i == 1 else '')
                     for i, (a, b, c, d, e) in enumerate(lines))
    return ('\\begin{tabular}{llccc}\n\\toprule\nMethod & Penalty grid & AP & $\\Delta$AP vs.\\ ridge (paired) & Test RMSE \\\\\n'
            '\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}')


def t90_table(fine):
    """T90 per method, plus the paired ratio T90(Lasso)/T90(ridge): geometric mean of per-network ratios
    with a paired bootstrap interval. An interval containing 1 is a tie."""
    lengths = sorted({int(r['length']) for r in fine})
    out = []
    for level in sorted({float(r['noise']) for r in fine}):
        t90 = {}
        for score in SERIES:
            t = []
            for seed in SEEDS:
                rows = sorted((r for r in fine if int(r['seed']) == seed and float(r['noise']) == level),
                              key=lambda r: int(r['length']))
                t.append(t_target(lengths, [float(r[f'ap_{score}']) for r in rows], .9))
            t90[score] = np.array(t)
        cells = [f'{m:.0f} [{lo:.0f}, {hi:.0f}]' for m, lo, hi in (ci(t90[s], log=True) for s in SERIES)]
        for score in ['lasso', 'lasso_tiebreak']:
            m, lo, hi = ci(t90[score] / t90['ridge'], log=True)
            cells.append(f'{m:.3f} [{lo:.3f}, {hi:.3f}]' + ('$^\\dagger$' if lo <= 1 <= hi else ''))
        out.append(f'{level:g} & ' + ' & '.join(cells) + ' \\\\')
    return ('\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{$T_{90}$ (transitions)} & '
            '\\multicolumn{2}{c}{Paired ratio to ridge} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n'
            'Noise ratio & Ridge (baseline) & Lasso & Lasso, complete ranking & Lasso & Lasso, complete ranking \\\\\n'
            '\\midrule\n' + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}')


def email_table():
    out = []
    for dept in DEPARTMENTS:
        folder = ROOT / 'results' / 'email_eu_core' / f'dept_{dept}'
        graph = json.loads((folder / 'graph.json').read_text())
        rows = load_csv(folder / 'metrics.csv')
        pick = lambda topo: sorted((r for r in rows if r['topology'] == topo and r['method'] == 'ridge'
                                    and int(r['length']) == 1000), key=lambda r: int(r['seed']))
        email, degree, random = (np.array([float(r['average_precision']) for r in pick(t)]) for t in ('email', 'degree', 'random'))
        chance = float(pick('email')[0]['chance_precision'])
        out.append(f"{dept} & {graph['nodes']} & {graph['edges']} & {chance:.3f} & {fmt(*ci(email))} & "
                   f"{fmt(*ci(email - random), sign=True)} & {fmt(*ci(email - degree), sign=True)} \\\\")
    return ('\\begin{tabular}{rrrcccc}\n\\toprule\nDept. & Nodes & Edges & Chance AP & AP (email) & '
            'Email $-$ random & Email $-$ degree-preserving \\\\\n\\midrule\n' + '\n'.join(out) +
            '\n\\bottomrule\n\\end{tabular}')


SIS_SERIES = {'ridge': dict(label='Ridge (baseline)', color='#2a78d6', marker='o', key='average_precision'),
              'lasso': dict(label='Lasso, complete ranking', color='#eb6834', marker='s', key='ap_complete_ranking'),
              'model_based': dict(label='Model-based Lasso', color='#1baf7a', marker='^', key='average_precision'),
              'local_poly': dict(label='Local poly. + model-based', color='#eda100', marker='D', key='average_precision')}
SIS_K = [5, 10, 20, 40, 80]


def sis_sweep_rows():
    return load_csv(ROOT / 'results' / 'sis_data_sweep' / 'metrics.csv')


def sis_curve(rows, beta, noise, method):
    spec = SIS_SERIES[method]
    per_seed = []
    for seed in SEEDS:
        v = sorted((r for r in rows if int(r['seed']) == seed and float(r['beta']) == beta and float(r['noise_sd']) == noise
                    and r['method'] == method), key=lambda r: int(r['trajectories']))
        assert [int(r['trajectories']) for r in v] == SIS_K, (seed, beta, noise, method)
        per_seed.append([float(r[spec['key']]) for r in v])
    return np.array(per_seed)            # (seeds, sizes)


def sis_figure(rows, beta=1.5):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.axisbelow': True})
    levels = [0., .005, .01, .02]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    for ax, noise in zip(axes, levels):
        for k, (method, spec) in enumerate(SIS_SERIES.items()):
            values = sis_curve(rows, beta, noise, method)
            stats = [ci(values[:, j]) for j in range(len(SIS_K))]
            m, lo, hi = map(np.array, zip(*stats))
            x = np.array(SIS_K) * (1 + .04 * (k - 1.5))
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=spec['color'], marker=spec['marker'], ms=3.5, lw=1.3,
                        capsize=2, mec='white', mew=.5, label=spec['label'], zorder=3)
        ax.axhline(.9, color=MUTED, ls=':', lw=1)
        ax.axhline(300 / 9900, color=MUTED, ls='--', lw=.8)
        ax.set_xscale('log'); ax.set_xticks(SIS_K); ax.set_xticklabels([str(k) for k in SIS_K]); ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_title(f'Noise SD {noise:g}', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('Training trajectories', color=MUTED)
        ax.grid(axis='y', color=GRID, lw=.6)
        ax.tick_params(colors=MUTED, length=2)
    axes[0].set_ylabel('Edge-recovery AP', color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, -.02))
    fig.subplots_adjust(left=.08, right=.99, top=.9, bottom=.32, wspace=.08)
    fig.savefig(OUT / 'fig_sis_data.pdf')
    fig.savefig(OUT / 'fig_sis_data.png', dpi=220)
    plt.close(fig)


def sis_n90_table(rows):
    """Training trajectories needed for AP 0.9, per spreading strength and noise."""
    out = []
    for beta in [.8, 1.5, 3.]:
        for noise in [0., .005, .01, .02]:
            cells = []
            for method in SIS_SERIES:
                values = sis_curve(rows, beta, noise, method)
                n90 = np.array([t_target(SIS_K, v, .9) for v in values])
                reached = np.isfinite(n90)
                if reached.all() and np.allclose(n90, SIS_K[0]):
                    cells.append(f'$\\le${SIS_K[0]}')
                elif reached.all():
                    m, lo, hi = ci(n90, log=True)
                    cells.append(f'{m:.0f} [{lo:.0f}, {hi:.0f}]')
                elif not reached.any():
                    cells.append(f'$>$80 (AP {values[:, -1].mean():.2f})')
                else:
                    cells.append(f'{reached.sum()}/10 reach')
            out.append(f'{beta:g} & {noise:g} & ' + ' & '.join(cells) + ' \\\\' + (' \\midrule' if noise == .02 and beta < 3 else ''))
    return ('\\begin{tabular}{llcccc}\n\\toprule\n$\\beta$ & Noise SD & Ridge (baseline) & Lasso, complete ranking & '
            'Model-based Lasso & Local poly.\\ + model-based \\\\\n\\midrule\n' + '\n'.join(out) +
            '\n\\bottomrule\n\\end{tabular}')


def sis_table():
    """SIS edge-recovery AP by spreading strength and noise, from the published snapshots."""
    ref = ROOT / 'reference_results'
    rows = (load_csv(ref / 'sis' / 'metrics.csv') + load_csv(ref / 'sis_aware' / 'metrics.csv') +
            load_csv(ref / 'sis_local_poly' / 'metrics.csv') +
            [r for r in load_csv(ref / 'neural_granger' / 'metrics.csv') if r['topology'] == 'random'])
    methods = ['ridge', 'lasso', 'sis_aware', 'local_poly', 'cmlp']
    out = []
    for noise in ['0.0', '0.02']:
        for beta in ['0.8', '1.5', '3.0']:
            cells = []
            for method in methods:
                group = sorted((r for r in rows if r['method'] == method and float(r['beta']) == float(beta)
                                and float(r['noise_sd']) == float(noise)), key=lambda r: int(r['seed']))
                if not group:
                    cells.append('to run')
                    continue
                assert [int(r['seed']) for r in group] == SEEDS, (method, beta, noise)
                cells.append(fmt(*ci([float(r['average_precision']) for r in group])))
            out.append(f'{float(noise):g} & {float(beta):g} & ' + ' & '.join(cells) + ' \\\\')
    return ('\\begin{tabular}{llccccc}\n\\toprule\nNoise SD & $\\beta$ & Ridge & Lasso & Model-based Lasso & '
            'Local poly.\\ + model-based & cMLP \\\\\n\\midrule\n' + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}')


def main():
    baseline = load_csv(ROOT / 'results' / 'baseline' / 'metrics.csv')
    fine = load_csv(ROOT / 'results' / 'tuning_and_ties' / 'metrics.csv')
    OUT.mkdir(parents=True, exist_ok=True)
    figure(fine)
    sis_rows = sis_sweep_rows()
    sis_figure(sis_rows)
    tex = ['% Generated by scripts/make_draft_figures.py; do not edit by hand.',
           '\\newcommand{\\TableMain}{' + main_table(baseline, fine) + '}',
           '\\newcommand{\\TableTninety}{' + t90_table(fine) + '}',
           '\\newcommand{\\TableEmail}{' + email_table() + '}',
           '\\newcommand{\\TableSIS}{' + sis_table() + '}',
           '\\newcommand{\\TableSISdata}{' + sis_n90_table(sis_rows) + '}']
    (OUT / 'tables.tex').write_text('\n\n'.join(tex) + '\n', encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
