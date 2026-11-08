"""Verify local-polynomial SIS fits and write comparison figures and tables."""
import hashlib
import json
from collections import Counter

import numpy as np
from scipy.signal import savgol_filter
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits

from network_recovery import ROOT, dump, write_csv
from sis_recovery import CONFIG as SIS, transitions
from sis_aware import CONFIG as SOLVER
from sis_local_poly import CONFIG, regression_samples
from make_report import independent_ap

OUT = ROOT/'results/sis_local_poly'
SOURCE = ROOT/'results/sis'
METHODS = ['ridge', 'lasso', 'sis_aware', 'local_poly']
LABELS = {'ridge':'Ridge', 'lasso':'Raw Lasso', 'sis_aware':'Model-based Lasso', 'local_poly':'Local polynomial +\nmodel-based Lasso'}
COLORS = {'ridge':'#8d99a7', 'lasso':'#b86829', 'sis_aware':'#167f83', 'local_poly':'#754ba1'}


def fit_folder(method, seed):
    folder = OUT if method == 'local_poly' else ROOT/'results/sis_aware' if method == 'sis_aware' else SOURCE
    return folder/f'seed_{seed}'


def verify_and_load():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for name, digest in protocol['script_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() == digest
    for name, digest in protocol['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    rows, windows, max_kkt = [], [], 0.
    for seed in SIS['seeds']:
        data = SOURCE/f'seed_{seed}'
        system = np.load(data/'system.npz')
        truth = system['truth']
        mask = ~np.eye(len(truth), dtype=bool)
        for beta in SIS['beta']:
            trajectories = np.load(data/f'trajectories_beta{beta:g}.npz')
            for noise in SIS['noise_sd']:
                tx, ty, _ = transitions(trajectories['test_clean']+noise*system['test_measurement_draw'])
                for method in METHODS:
                    name = f'{method}_beta{beta:g}_noise{noise:g}'
                    folder = fit_folder(method, seed)
                    record = json.loads((folder/f'{name}.json').read_text())
                    row = record['metrics']
                    fitted = np.load(folder/f'{name}.npz')
                    if method in ['local_poly', 'sis_aware']:
                        weights = fitted['b_hat']
                        predicted = (1-SIS['dt']*SIS['delta'])*tx + SIS['dt']*beta*(1-tx)*(tx@weights.T)
                    else:
                        weights = fitted['a_hat']
                        predicted = tx@weights.T+fitted['intercept']
                    ap = independent_ap(truth[mask], abs(weights[mask]))
                    rmse = float(np.sqrt(np.mean((predicted-ty)**2)))
                    np.testing.assert_allclose(ap, row['average_precision'], atol=1e-12)
                    np.testing.assert_allclose(rmse, row['test_rmse'], atol=1e-12)
                    if method == 'local_poly':
                        assert weights.min() >= -1e-12 and np.all(np.diag(weights) == 0)
                        assert len(record['trials']) == len(CONFIG['windows'])*len(CONFIG['fractions'])
                        best = min(record['trials'], key=lambda trial:trial['cv_mse'])
                        assert (best['window'], best['fraction']) == (row['window'], row['fraction'])
                        np.testing.assert_allclose(np.sqrt(best['cv_mse']), row['cv_rmse'])
                        observed = trajectories['clean']+noise*system['measurement_draw']
                        x, y = regression_samples(observed, row['window'], SIS['dt'])
                        assert len(x) == row['training_samples']
                        for node in range(len(weights)):
                            keep = np.arange(len(weights)) != node
                            design = SIS['dt']*beta*(1-x[:,node,None])*x[:,keep]
                            target = y[:,node]-(1-SIS['dt']*SIS['delta'])*x[:,node]
                            sx = np.sqrt(np.mean(design**2, axis=0)); sx[sx<=1e-14] = 1.
                            sy = max(float(np.sqrt(np.mean(target**2))), 1e-14)
                            xs, ys = design/sx, target/sy
                            coef = weights[node,keep]*sx/sy
                            alpha = row['fraction']*max(0., float((xs.T@ys/len(xs)).max()))
                            grad = xs.T@(xs@coef-ys)/len(xs)+alpha
                            residual = float(np.where(coef>1e-10, abs(grad), np.maximum(-grad,0)).max())
                            assert residual <= SOLVER['kkt_tolerance']+1e-12
                            max_kkt = max(max_kkt, residual)
                        for window in CONFIG['windows']:
                            trial = min((r for r in record['trials'] if r['window']==window), key=lambda r:r['cv_mse'])
                            windows.append(dict(seed=seed, beta=beta, noise_sd=noise, window=window,
                                                cv_rmse=float(np.sqrt(trial['cv_mse'])), fraction=trial['fraction'],
                                                selected=int(window==row['window'])))
                    precision, recall = fitted['precision'][::-1], fitted['recall'][::-1]
                    grid = np.arange(SIS['edges']+1)/SIS['edges']
                    curve = precision[np.searchsorted(recall, grid-1e-12, side='left').clip(max=len(precision)-1)]
                    np.testing.assert_allclose(curve[1:].mean(), ap, atol=1e-12)
                    rows.append(dict(**row, curve=curve))
    assert len(rows) == 240
    local = [r for r in rows if r['method']=='local_poly']
    dump(OUT/'verification.json', dict(compared_fits=240, new_fits=60, checked_AP=240,
         checked_test_RMSE=240, checked_PR_areas=240, checked_nodewise_KKT=6000,
         max_selected_KKT_residual=max_kkt, max_candidate_KKT_residual=max(r['max_kkt_residual'] for r in local),
         solver_warnings=sum(r['solver_warnings'] for r in local),
         polished_candidates=sum(r['polished_candidates'] for r in local)))
    write_csv(OUT/'window_cv.csv', windows)
    return rows, windows


def aggregate(rows):
    summary, paired = [], []
    draws = np.random.default_rng(85420).integers(0, 10, size=(5000, 10))
    for noise in SIS['noise_sd']:
        for beta in SIS['beta']:
            groups = {m:sorted([r for r in rows if r['method']==m and r['beta']==beta and r['noise_sd']==noise],
                               key=lambda r:r['seed']) for m in METHODS}
            for method, group in groups.items():
                result = dict(noise_sd=noise, beta=beta, method=method, graphs=len(group))
                for metric in ['average_precision', 'test_rmse', 'rmse_ratio']:
                    values = np.array([r[metric] for r in group])
                    result[metric+'_mean'] = float(values.mean())
                    result[metric+'_sd'] = float(values.std(ddof=1))
                result['selected_windows'] = json.dumps(dict(sorted(Counter(r['window'] for r in group).items()))) if method=='local_poly' else ''
                summary.append(result)
            for method in ['ridge', 'lasso', 'sis_aware']:
                for metric in ['average_precision', 'test_rmse']:
                    diff = np.array([a[metric]-b[metric] for a,b in zip(groups['local_poly'], groups[method])])
                    low, high = np.quantile(diff[draws].mean(axis=1), [.025,.975])
                    paired.append(dict(noise_sd=noise, beta=beta, reference=method, metric=metric,
                                       local_poly_minus_reference=float(diff.mean()), ci_low=float(low), ci_high=float(high)))
    write_csv(OUT/'summary.csv', summary)
    write_csv(OUT/'paired_differences.csv', paired)
    return summary, paired


def plots(rows, summary, windows):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 4, figsize=(15,4.5), layout='constrained')
    for ax, method in zip(axes, METHODS):
        group = [r for r in summary if r['method']==method]
        values = [[next(r['average_precision_mean'] for r in group if r['beta']==b and r['noise_sd']==n)
                   for n in SIS['noise_sd']] for b in SIS['beta']]
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for i, beta in enumerate(SIS['beta']):
            for j, noise in enumerate(SIS['noise_sd']):
                r = next(r for r in group if r['beta']==beta and r['noise_sd']==noise)
                ax.text(j,i,f"{r['average_precision_mean']:.3f}\n± {r['average_precision_sd']:.3f}",ha='center',va='center',
                        color='white' if values[i][j]<.55 else '#172433')
        ax.set(title=LABELS[method], xticks=[0,1], xticklabels=['0','0.02'], yticks=range(3),
               yticklabels=['0.8','1.5','3.0'], xlabel='Measurement-noise SD', ylabel='Spreading strength β')
    fig.colorbar(im, ax=axes, label='Edge average precision')
    fig.suptitle('SIS dynamics | local polynomial preprocessing | mean ± SD across 10 graphs')
    fig.savefig(OUT/'recovery_comparison.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13,4), layout='constrained')
    for ax, beta in zip(axes, SIS['beta']):
        for method in METHODS:
            curves = np.stack([r['curve'] for r in rows if r['method']==method and r['beta']==beta and r['noise_sd']==.02])
            ax.step(np.arange(301)/300,curves.mean(axis=0),where='pre',label=LABELS[method],color=COLORS[method])
        ax.axhline(300/9900,color='#aaa',ls='--',lw=1)
        ax.set(title=f'β = {beta:g}',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    axes[0].legend(loc='upper right',fontsize=8)
    fig.suptitle('SIS dynamics | edge recovery with measurement-noise SD 0.02')
    fig.savefig(OUT/'precision_recall_noisy.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13,4), layout='constrained')
    for ax, beta in zip(axes, SIS['beta']):
        values = np.array([[r['cv_rmse'] for r in windows if r['noise_sd']==.02 and r['beta']==beta and r['window']==w]
                           for w in CONFIG['windows']])
        values = values-values[0]
        ax.errorbar(range(5),values.mean(axis=1),yerr=values.std(axis=1,ddof=1),fmt='o-',capsize=4,color=COLORS['local_poly'])
        ax.axhline(0,color='#aaa',ls='--',lw=1)
        ax.set(title=f'β = {beta:g}',xticks=range(5),xticklabels=['Raw','5','9','15','21'],
               xlabel='Local polynomial window',ylabel='Validation RMSE change vs raw (mean ± SD)')
    fig.suptitle('SIS dynamics | window selection from training trajectories | noise SD 0.02')
    fig.savefig(OUT/'window_selection.png',dpi=180); plt.close(fig)


def network_picture():
    seed, beta, noise = 1001, 3., .02
    truth = np.load(SOURCE/f'seed_{seed}/system.npz')['truth']
    edges = {(int(j), int(i)) for i,j in zip(*np.where(truth))}
    visible = {e for e in edges if max(e)<25}
    graph = nx.DiGraph(); graph.add_nodes_from(range(25)); graph.add_edges_from(sorted(visible))
    pos = nx.circular_layout(graph)
    fig, axes = plt.subplots(1,4,figsize=(17,5))
    for ax, method in zip(axes, ['truth','lasso','sis_aware','local_poly']):
        nx.draw_networkx_nodes(graph,pos,node_size=145,node_color='#e9f1f5',edgecolors='#466379',linewidths=.8,ax=ax)
        nx.draw_networkx_labels(graph,pos,font_size=8,ax=ax)
        if method=='truth':
            selections = [(visible,'#466379','solid')]
            title = f'Truth\n{len(visible)} edges in this view'
        else:
            folder = fit_folder(method,seed); name = f'{method}_beta{beta:g}_noise{noise:g}'
            fitted = np.load(folder/f'{name}.npz')
            scores = abs(fitted['a_hat'] if method=='lasso' else fitted['b_hat']).copy()
            np.fill_diagonal(scores,-np.inf)
            indices = np.argsort(-scores.ravel(),kind='stable')[:300]
            chosen = {(int(k%100),int(k//100)) for k in indices if scores.ravel()[k]>0}
            chosen = {e for e in chosen if max(e)<25}
            tp, fn, fp = chosen&visible, visible-chosen, chosen-visible
            selections = [(tp,'#138777','solid'),(fn,'#db9226','dashed'),(fp,'#c84972','solid')]
            ap = json.loads((folder/f'{name}.json').read_text())['metrics']['average_precision']
            title = f'{LABELS[method]} | AP = {ap:.3f}\n{len(tp)} correct, {len(fn)} missed, {len(fp)} false'
        for chosen,color,style in selections:
            nx.draw_networkx_edges(graph,pos,edgelist=sorted(chosen),edge_color=color,style=style,
                                   node_size=145,arrowsize=10,alpha=.8,ax=ax)
        ax.set_title(title,fontsize=10,pad=12)
        ax.set(xlim=(-1.18,1.18),ylim=(-1.18,1.18),aspect='equal'); ax.axis('off')
    fig.legend(handles=[Line2D([],[],color=c,ls=s,label=l) for c,s,l in
                [('#138777','-','Recovered true edge'),('#db9226','--','Missed edge'),('#c84972','-','False edge')]],
               loc='lower center',ncol=3,frameon=False)
    fig.suptitle('SIS: seed 1001, β = 3, measurement-noise SD = 0.02\n'
                 '25-node induced view; top 300 nonzero absolute scores on the full graph',fontsize=12)
    fig.tight_layout(rect=(0,.07,1,.86))
    fig.savefig(OUT/'network_example_noisy.png',dpi=180); plt.close(fig)


def preprocessing_picture():
    seed, beta, noise = 1001, 3., .02
    source = SOURCE/f'seed_{seed}'
    system = np.load(source/'system.npz')
    clean = np.load(source/f'trajectories_beta{beta:g}.npz')['clean'][0]
    observed = clean+noise*system['measurement_draw'][0]
    record = json.loads((OUT/f'seed_{seed}/local_poly_beta{beta:g}_noise{noise:g}.json').read_text())
    window = record['metrics']['window']
    times = np.arange(len(clean))*SIS['dt']
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,node in zip(axes,range(3)):
        ax.scatter(times,observed[:,node],s=14,alpha=.5,color='#8d99a7',label='Noisy observation')
        ax.plot(times,clean[:,node],color='#203547',lw=2,label='Latent state')
        if window:
            radius = window//2
            smooth = savgol_filter(observed[:,node],window,CONFIG['degree'])
            ax.plot(times[radius:-radius],smooth[radius:-radius],color=COLORS['local_poly'],lw=2,label='Local polynomial')
        ax.set(title=f'Node {node}',xlabel='Time',ylabel='Infection probability')
    axes[-1].legend(fontsize=8)
    fig.suptitle(f'SIS training trajectory preprocessing | seed 1001, β = 3, noise SD 0.02, selected window {window}')
    fig.savefig(OUT/'preprocessing_example.png',dpi=180); plt.close(fig)


def main():
    with threadpool_limits(limits=1):
        rows, windows = verify_and_load()
        summary, paired = aggregate(rows)
        plots(rows,summary,windows)
        network_picture()
        preprocessing_picture()
    for r in summary:
        if r['method']=='local_poly':
            print(f"noise={r['noise_sd']:g} beta={r['beta']:g}: AP={r['average_precision_mean']:.4f} "
                  f"+/- {r['average_precision_sd']:.4f}; windows={r['selected_windows']}")
    print(OUT)


if __name__ == '__main__':
    main()
