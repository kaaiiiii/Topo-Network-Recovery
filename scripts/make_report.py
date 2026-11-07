"""Plots and independent reconciliation of Topic 7 baseline outputs."""
import csv
import json
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from network_recovery import ROOT, CONFIG, dump, write_csv, cv_splits

OUT = ROOT / 'results' / 'baseline'


def independent_ap(truth, scores):
    order = np.argsort(-scores, kind='stable')
    labels = truth[order]; ranked = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked)-1]
    tp = np.cumsum(labels)[ends]
    precision = tp / (ends + 1)
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def verify_and_load():
    rows = []
    selected_cv_checks = 0
    for seed in CONFIG['seeds']:
        folder = OUT / f'seed_{seed}'
        system = np.load(folder / 'system.npz')
        a, truth = system['a'], system['truth']
        mask = ~np.eye(len(a), dtype=bool)
        assert truth.sum() == CONFIG['edges'] and not np.diag(truth).any()
        graph = nx.gnm_random_graph(CONFIG['nodes'], CONFIG['edges'], seed=seed, directed=True)
        expected_edges = {(j,i) for i,j in graph.edges}
        assert set(zip(*np.where(truth))) == expected_edges
        np.testing.assert_allclose(max(abs(np.linalg.eigvals(a))), .9, atol=1e-10)
        for length in CONFIG['lengths']:
            for noise in CONFIG['noise_ratios']:
                observed = system['test_clean'] + noise * system['test_measurement_draw']
                for method in ['ridge', 'lasso']:
                    name = f'{method}_T{length}_noise{noise:g}'
                    record = json.loads((folder / f'{name}.json').read_text())
                    row = record['metrics']; result = np.load(folder / f'{name}.npz')
                    estimated = result['a_hat']
                    ap = independent_ap(truth[mask], abs(estimated[mask]))
                    np.testing.assert_allclose(ap, row['average_precision'], atol=1e-12)
                    prediction = observed[:-1] @ estimated.T + result['intercept']
                    np.testing.assert_allclose(np.sqrt(np.mean((prediction-observed[1:])**2)), row['test_rmse'], atol=1e-12)
                    trials = record['trials']
                    selected = min(trials, key=lambda x: x['cv_mse'])
                    assert selected['alpha'] == row['alpha'] and len(trials) == 7
                    if method == 'lasso' and row['convergence_warnings']:
                        training = (system['clean'] + noise * system['measurement_draw'])[:length+1]
                        x, y = training[:-1], training[1:]
                        for k, (it, iv) in enumerate(cv_splits(length)):
                            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always', ConvergenceWarning)
                                model = make_pipeline(StandardScaler(), Lasso(alpha=row['alpha'], precompute=True,
                                                      max_iter=CONFIG['lasso_max_iter'], tol=CONFIG['lasso_tol']))
                                model.fit(x[it], y[it])
                                rmse = np.sqrt(np.mean((model.predict(x[iv])-y[iv])**2))
                            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                            np.testing.assert_allclose(rmse, selected['fold_rmse'][k], atol=1e-9)
                            selected_cv_checks += 1
                    rows.append(row)
    assert len(rows) == 180
    dump(OUT / 'verification.json', dict(independent_graphs=10, fitted_models=len(rows),
          directed_support_checks=10, independent_AP_checks=len(rows), test_RMSE_checks=len(rows),
          penalty_selection_checks=len(rows), convergence_warnings=sum(r['convergence_warnings'] for r in rows),
          selected_CV_fits_rechecked_after_warnings=selected_cv_checks, selected_CV_warning_count=0,
          selected_lasso_iteration_caps=sum(r['method']=='lasso' and r['max_selected_iterations'] >= CONFIG['lasso_max_iter'] for r in rows)))
    return rows


def aggregate(rows):
    summary, differences = [], []
    draws = np.random.default_rng(5420).integers(0, len(CONFIG['seeds']), size=(5000, len(CONFIG['seeds'])))
    for noise in CONFIG['noise_ratios']:
        for length in CONFIG['lengths']:
            paired = {}
            for method in ['ridge', 'lasso']:
                part = sorted([r for r in rows if r['noise']==noise and r['length']==length and r['method']==method], key=lambda r:r['seed'])
                values = np.array([r['average_precision'] for r in part])
                paired[method] = values
                summary.append(dict(noise=noise, length=length, method=method, n_graphs=len(part),
                                    AP_mean=float(values.mean()), AP_sd=float(values.std(ddof=1)),
                                    test_RMSE_mean=float(np.mean([r['test_rmse'] for r in part]))))
            delta = paired['lasso'] - paired['ridge']
            low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            differences.append(dict(noise=noise, length=length, mean_AP_gain=float(delta.mean()), lower95=float(low), upper95=float(high)))
    write_csv(OUT / 'summary.csv', summary); write_csv(OUT / 'paired_AP_differences.csv', differences)
    return summary, differences


def heatmap(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for ax, method in zip(axes, ['ridge', 'lasso']):
        grid = np.array([[next(r['AP_mean'] for r in summary if r['method']==method and r['noise']==noise and r['length']==length)
                          for length in CONFIG['lengths']] for noise in CONFIG['noise_ratios']])
        im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        for i, noise in enumerate(CONFIG['noise_ratios']):
            for j, length in enumerate(CONFIG['lengths']):
                row = next(r for r in summary if r['method']==method and r['noise']==noise and r['length']==length)
                ax.text(j,i,f"{row['AP_mean']:.3f}\n+/- {row['AP_sd']:.3f}", ha='center',va='center',fontsize=11,
                        color='white' if row['AP_mean'] < .55 else '#142d37')
        ax.set(xticks=range(3), xticklabels=CONFIG['lengths'], yticks=range(3), yticklabels=CONFIG['noise_ratios'],
               xlabel='Observed transitions', ylabel='Measurement noise SD / clean stationary SD', title=method.capitalize())
    fig.colorbar(im, ax=axes, label='Edge-recovery average precision', shrink=.85)
    fig.suptitle('Recovering 300 directed edges among 9,900 possible connections\nMean +/- sample SD across 10 independent graph/trajectory draws', fontsize=13)
    fig.savefig(OUT / 'recovery_heatmap.png', dpi=170); plt.close(fig)


def precision_recall():
    colors = ['#245caa','#d67b1a','#16836b']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
    # Exact attainable recall grid for the fixed 300 true edges. Use PR steps,
    # not straight-line interpolation across the large tie of zero Lasso scores.
    recall_grid = np.arange(CONFIG['edges'] + 1) / CONFIG['edges']
    for ax, noise in zip(axes, CONFIG['noise_ratios']):
        for method, linestyle in [('lasso','-'), ('ridge','--')]:
            for length, color in zip(CONFIG['lengths'], colors):
                curves = []
                for seed in CONFIG['seeds']:
                    data = np.load(OUT / f'seed_{seed}' / f'{method}_T{length}_noise{noise:g}.npz')
                    r, p = data['recall'][::-1], data['precision'][::-1]
                    values = p[np.minimum(np.searchsorted(r, recall_grid-1e-12, side='left'),len(r)-1)]
                    metrics = json.loads((OUT / f'seed_{seed}' / f'{method}_T{length}_noise{noise:g}.json').read_text())['metrics']
                    np.testing.assert_allclose(values[1:].mean(), metrics['average_precision'], atol=1e-12)
                    curves.append(values)
                values = np.array(curves)
                ax.step(recall_grid, values.mean(axis=0), where='pre', color=color, linestyle=linestyle, lw=1.7)
                if method == 'lasso':
                    ax.fill_between(recall_grid, np.maximum(0, values.mean(axis=0)-values.std(axis=0,ddof=1)),
                                    np.minimum(1, values.mean(axis=0)+values.std(axis=0,ddof=1)), color=color, alpha=.09, step='pre')
        ax.axhline(300/9900, color='#777', lw=1, linestyle=':')
        ax.set(title=f'Measurement noise ratio = {noise:g}', xlabel='Recall', xlim=(0,1), ylim=(0,1.02))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Precision')
    handles = [Line2D([0],[0],color=c,label=f'T = {t}') for c,t in zip(colors,CONFIG['lengths'])]
    handles += [Line2D([0],[0],color='#333',ls=s,label=m) for m,s in [('Lasso','-'),('Ridge','--')]]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False)
    fig.suptitle('Directed-edge precision-recall: more observations versus more measurement noise')
    fig.tight_layout(rect=(0,.08,1,.94)); fig.savefig(OUT / 'precision_recall.png', dpi=170); plt.close(fig)


def network_picture():
    # Fixed seed and condition, selected before looking at recovery scores.
    seed, length, noise = CONFIG['seeds'][0], 300, .25
    system = np.load(OUT / f'seed_{seed}' / 'system.npz')
    truth = system['truth']; display_nodes = set(range(25))
    edges = {(j,i) for i,j in zip(*np.where(truth))}
    g = nx.DiGraph(); g.add_nodes_from(display_nodes)
    visible_truth = {e for e in edges if set(e) <= display_nodes}; g.add_edges_from(visible_truth)
    pos = nx.circular_layout(g)
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.6))
    for ax, method in zip(axes, ['truth','ridge','lasso']):
        nx.draw_networkx_nodes(g,pos,node_size=130,node_color='#e9f1f5',edgecolors='#466379',linewidths=.7,ax=ax)
        nx.draw_networkx_labels(g,pos,font_size=7,ax=ax)
        if method=='truth':
            nx.draw_networkx_edges(g,pos,edgelist=list(visible_truth),edge_color='#466379',arrowsize=9,alpha=.65,ax=ax)
        else:
            a_hat = np.load(OUT / f'seed_{seed}' / f'{method}_T{length}_noise{noise:g}.npz')['a_hat']
            scores = abs(a_hat).copy(); np.fill_diagonal(scores, -1)
            indices = np.argsort(-scores.ravel(),kind='stable')[:CONFIG['edges']]
            chosen = {(int(k%len(truth)),int(k//len(truth))) for k in indices if scores.ravel()[k]>0}
            chosen = {e for e in chosen if set(e)<=display_nodes}
            for selected,color,style in [(chosen & visible_truth,'#138777','solid'),
                                          (visible_truth-chosen,'#db9226','dashed'),(chosen-visible_truth,'#c84972','solid')]:
                nx.draw_networkx_edges(g,pos,edgelist=list(selected),edge_color=color,style=style,arrowsize=9,alpha=.8,ax=ax)
        ax.set_title(method.capitalize()); ax.set_xlim(-1.18,1.18); ax.set_ylim(-1.18,1.18); ax.set_aspect('equal'); ax.axis('off')
    handles = [Line2D([0],[0],color=c,ls=s,label=t) for c,s,t in [('#138777','-','Recovered true edge'),('#db9226','--','Missed edge'),('#c84972','-','False edge')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Fixed example: seed 1001, T = 300, noise ratio = 0.25\n25-node induced view; up to top 300 positive edges ranked on the full graph',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.9)); fig.savefig(OUT / 'network_example.png',dpi=170); plt.close(fig)




def main():
    rows = verify_and_load()
    summary, differences = aggregate(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    heatmap(summary); precision_recall(); network_picture()
    for row in summary:
        print(row)
    print('Figures, tables and verification complete:', OUT)


if __name__ == '__main__':
    main()
