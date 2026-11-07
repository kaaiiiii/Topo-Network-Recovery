"""Reconcile SIS outputs and create a self-contained scientific report."""
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from network_recovery import ROOT, dump, write_csv
from sis_recovery import CONFIG, SOURCES, transitions, simulate_sis, episode_folds
from make_report import independent_ap

OUT = ROOT / 'results' / 'sis'
COLORS = ['#2876a5', '#cb7127', '#994e99']


def verify_and_load():
    rows, pr_checks = [], 0
    selected_kkt_checks = 0
    grid = np.arange(CONFIG['edges']+1)/CONFIG['edges']
    for seed in CONFIG['seeds']:
        folder = OUT / f'seed_{seed}'
        system = np.load(folder / 'system.npz')
        b, truth = system['b'], system['truth']
        mask = ~np.eye(len(b), dtype=bool)
        graph = nx.gnm_random_graph(CONFIG['nodes'], CONFIG['edges'], seed=seed, directed=True)
        assert set(zip(*np.where(truth))) == {(j, i) for i, j in graph.edges}
        np.testing.assert_array_equal(b != 0, truth)
        np.testing.assert_allclose(max(abs(np.linalg.eigvals(b))), 1., atol=1e-10)
        baseline = np.load(ROOT / 'results' / 'baseline' / f'seed_{seed}' / 'system.npz')
        np.testing.assert_array_equal(truth, baseline['truth'])
        np.testing.assert_allclose(.3*np.eye(len(b))+.6*b, baseline['a'], atol=1e-12)
        for beta in CONFIG['beta']:
            trajectories = np.load(folder / f'trajectories_beta{beta:g}.npz')
            clean, test_clean = trajectories['clean'], trajectories['test_clean']
            np.testing.assert_array_equal(clean, simulate_sis(b, system['initial'], beta,
                                          CONFIG['steps'], CONFIG['delta'], CONFIG['dt']))
            np.testing.assert_array_equal(test_clean, simulate_sis(b, system['test_initial'], beta,
                                          CONFIG['steps'], CONFIG['delta'], CONFIG['dt']))
            _, _, groups = transitions(clean)
            episode_folds(groups)
            for sd in CONFIG['noise_sd']:
                x, y, _ = transitions(test_clean + sd*system['test_measurement_draw'])
                for method in ['ridge', 'lasso']:
                    name = f'{method}_beta{beta:g}_noise{sd:g}'
                    record = json.loads((folder / f'{name}.json').read_text())
                    row = record['metrics']
                    result = np.load(folder / f'{name}.npz')
                    if method == 'lasso':
                        train_x, train_y, _ = transitions(clean + sd*system['measurement_draw'])
                        xmean, ymean = train_x.mean(axis=0), train_y.mean(axis=0)
                        xscale, yscale = train_x.std(axis=0), train_y.std(axis=0)
                        xs, ys = (train_x-xmean)/xscale, (train_y-ymean)/yscale
                        coef = result['a_hat']*xscale[None,:]/yscale[:,None]
                        gradient = (xs.T@(xs@coef.T-ys)/len(xs)).T
                        active = abs(coef)>1e-10
                        kkt = np.where(active,abs(gradient+row['alpha']*np.sign(coef)),
                                       np.maximum(abs(gradient)-row['alpha'],0))
                        assert kkt.max() <= CONFIG['kkt_tolerance']
                        np.testing.assert_allclose(result['intercept'],ymean-result['a_hat']@xmean,atol=1e-10)
                        selected_kkt_checks += 1
                    ap = independent_ap(truth[mask], abs(result['a_hat'][mask]))
                    np.testing.assert_allclose(ap, row['average_precision'], atol=1e-12)
                    rmse = np.sqrt(np.mean((x@result['a_hat'].T+result['intercept']-y)**2))
                    persistence = np.sqrt(np.mean((x-y)**2))
                    np.testing.assert_allclose([rmse, persistence, rmse/persistence],
                                              [row['test_rmse'], row['persistence_rmse'], row['rmse_ratio']], atol=1e-12)
                    selected = min(record['trials'], key=lambda v: v['cv_mse'])
                    assert selected['alpha'] == row['alpha']
                    assert len(record['trials']) == 15
                    assert all(not t['convergence_warnings'] for t in record['trials'])
                    assert not selected['convergence_warnings'] and not row['selected_convergence_warnings']
                    assert row['max_selected_iterations'] < CONFIG['lars_max_iter']
                    assert row['max_kkt_residual'] <= CONFIG['kkt_tolerance']
                    p, r = result['precision'][::-1], result['recall'][::-1]
                    values = p[np.searchsorted(r, grid-1e-12, side='left').clip(max=len(p)-1)]
                    np.testing.assert_allclose(values[1:].mean(), ap, atol=1e-12)
                    rows.append(dict(**row, pr_values=values))
                    pr_checks += 1
    assert len(rows) == 120
    dump(OUT / 'verification.json', dict(models=len(rows), directed_graph_checks=10,
         matched_baseline_graphs=10, reproduced_trajectory_arrays=60, independent_AP_checks=120,
         test_RMSE_checks=120, PR_step_area_checks=pr_checks, penalty_selection_checks=120,
         selected_convergence_warnings=0,
         candidate_convergence_warnings=sum(r['convergence_warnings'] for r in rows),
         resolved_convergence_warnings=sum(r['resolved_convergence_warnings'] for r in rows),
         solver_warnings=sum(r['solver_warnings'] for r in rows),
         independently_recomputed_selected_KKT_checks=selected_kkt_checks,
         max_lasso_KKT_residual=max(r['max_kkt_residual'] for r in rows)))
    return rows


def aggregate(rows):
    summary, differences = [], []
    draws = np.random.default_rng(7420).integers(0, 10, size=(5000, 10))
    for sd in CONFIG['noise_sd']:
        for beta in CONFIG['beta']:
            paired = {}
            for method in ['ridge', 'lasso']:
                group = sorted([r for r in rows if r['noise_sd']==sd and r['beta']==beta and r['method']==method],
                               key=lambda r: r['seed'])
                row = dict(noise_sd=sd, beta=beta, method=method, graphs=len(group))
                for metric in ['average_precision', 'test_rmse', 'persistence_rmse', 'rmse_ratio',
                               'final_prevalence', 'nonlinear_suppression']:
                    values = np.array([r[metric] for r in group])
                    row[metric+'_mean'], row[metric+'_sd'] = float(values.mean()), float(values.std(ddof=1))
                paired[method] = np.array([r['average_precision'] for r in group])
                low, high = np.quantile(paired[method][draws].mean(axis=1), [.025, .975])
                row.update(ap_ci_low=float(low), ap_ci_high=float(high))
                row['selected_alpha_at_grid_boundary'] = sum(r['alpha'] in
                        [CONFIG[method+'_alphas'][0], CONFIG[method+'_alphas'][-1]] for r in group)
                summary.append(row)
            diff = paired['lasso']-paired['ridge']
            ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
            differences.append(dict(noise_sd=sd, beta=beta, lasso_minus_ridge_mean=float(diff.mean()),
                                    paired_bootstrap_low=float(ci[0]), paired_bootstrap_high=float(ci[1])))
    write_csv(OUT/'summary.csv', summary)
    write_csv(OUT/'paired_AP_differences.csv', differences)
    noise_effects = []
    for beta in CONFIG['beta']:
        for method in ['ridge', 'lasso']:
            paired = {}
            for sd in CONFIG['noise_sd']:
                group = sorted([r for r in rows if r['beta']==beta and r['method']==method and r['noise_sd']==sd],
                               key=lambda r:r['seed'])
                paired[sd] = np.array([r['average_precision'] for r in group])
            diff = paired[.02]-paired[0.]
            ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
            noise_effects.append(dict(beta=beta, method=method, noisy_minus_clean_AP=float(diff.mean()),
                                     paired_bootstrap_low=float(ci[0]), paired_bootstrap_high=float(ci[1])))
    write_csv(OUT/'paired_noise_effects.csv', noise_effects)
    return summary, differences


def plots(rows, summary):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for ax, method in zip(axes, ['ridge', 'lasso']):
        matrix = np.array([[next(r['average_precision_mean'] for r in summary if
                          r['method']==method and r['beta']==b and r['noise_sd']==sd)
                          for sd in CONFIG['noise_sd']] for b in CONFIG['beta']])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for i,b in enumerate(CONFIG['beta']):
            for j,sd in enumerate(CONFIG['noise_sd']):
                row = next(r for r in summary if r['method']==method and r['beta']==b and r['noise_sd']==sd)
                ax.text(j,i,f"{row['average_precision_mean']:.3f}\n± {row['average_precision_sd']:.3f}",
                        ha='center',va='center',color='white' if matrix[i,j]<.55 else '#172433',fontsize=12)
        ax.set(title=method.title(), xticks=[0,1], xticklabels=['0', '0.02'],
               yticks=range(3), yticklabels=['0.8','1.5','3.0'], xlabel='Measurement-noise SD', ylabel='Spreading strength β')
    fig.colorbar(im, ax=axes, label='Directed-edge average precision')
    fig.suptitle('SIS network recovery | mean ± SD over 10 graphs')
    fig.savefig(OUT/'recovery_heatmap.png',dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11,4.6), sharey=True, layout='constrained')
    recall = np.arange(301)/300
    for ax, sd in zip(axes,CONFIG['noise_sd']):
        for color, beta in zip(COLORS, CONFIG['beta']):
            for method, style in [('lasso','-'), ('ridge','--')]:
                curves = np.stack([r['pr_values'] for r in rows if r['noise_sd']==sd and r['beta']==beta and r['method']==method])
                ax.step(recall,curves.mean(axis=0),where='pre',color=color,ls=style,lw=1.8)
        ax.axhline(300/9900,color='#777777',ls=':',lw=1)
        ax.set(xlabel='Recall',title=f'Measurement-noise SD = {sd:g}',xlim=(0,1),ylim=(0,1.03))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Precision')
    handles=[Line2D([],[],color=c,label=f'β = {b:g}') for c,b in zip(COLORS,CONFIG['beta'])]
    handles += [Line2D([],[],color='black',ls=s,label=m) for m,s in [('Lasso','-'),('Ridge','--')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Mean precision–recall curves | 10 paired graphs')
    fig.savefig(OUT/'precision_recall.png',dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    t=np.arange(CONFIG['steps']+1)*CONFIG['dt']
    for color,beta in zip(COLORS,CONFIG['beta']):
        means=[]
        for seed in CONFIG['seeds']:
            clean=np.load(OUT/f'seed_{seed}'/f'trajectories_beta{beta:g}.npz')['clean']
            means.append(clean.mean(axis=(0,2)))
        means=np.stack(means)
        mean,sd=means.mean(axis=0),means.std(axis=0,ddof=1)
        axes[0].plot(t,mean,color=color,label=f'β = {beta:g}')
        axes[0].fill_between(t,mean-sd,mean+sd,color=color,alpha=.15)
    axes[0].set(xlabel='Time (recovery rate δ = 1)',ylabel='Mean infection probability',title='Nonlinear propagation: mean ± graph SD')
    axes[0].legend(frameon=False)
    for color,beta in zip(COLORS,CONFIG['beta']):
        for method,marker in [('ridge','s'),('lasso','o')]:
            group=sorted([r for r in summary if r['beta']==beta and r['method']==method],key=lambda r:r['noise_sd'])
            axes[1].plot([r['rmse_ratio_mean'] for r in group],[r['average_precision_mean'] for r in group],
                         color=color,marker=marker,ls='--' if method=='ridge' else '-',alpha=.85)
            for row in group:
                axes[1].errorbar(row['rmse_ratio_mean'],row['average_precision_mean'],
                    xerr=row['rmse_ratio_sd'],yerr=row['average_precision_sd'],color=color,fmt='none',capsize=3,alpha=.6)
    axes[1].axvline(1,color='#777777',ls=':',lw=1)
    axes[1].set(xlabel='Test RMSE / persistence RMSE (lower is better)',ylabel='Edge AP (higher is better)',
                title='Prediction versus recovery',ylim=(0,1.03))
    axes[1].legend(handles=[Line2D([],[],color='black',marker=m,ls=s,label=n) for n,m,s in
                           [('Ridge','s','--'),('Lasso','o','-')]],frameon=False,loc='lower left')
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(OUT/'dynamics_and_prediction.png',dpi=180); plt.close(fig)


def network_pictures():
    """Fixed first-seed illustration, same nodes/layout/cutoff as the linear figure."""
    seed, beta = CONFIG['seeds'][0], 3.
    folder = OUT/f'seed_{seed}'
    truth = np.load(folder/'system.npz')['truth']
    n = len(truth)
    edges = {(int(source),int(target)) for target,source in zip(*np.where(truth))}
    nodes = list(range(25))
    visible_truth = {e for e in edges if e[0]<25 and e[1]<25}
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(sorted(visible_truth))
    pos = nx.circular_layout(graph)
    audit = []
    for noise in CONFIG['noise_sd']:
        fig, axes = plt.subplots(1,3,figsize=(13.5,4.9))
        for ax, method in zip(axes,['truth','ridge','lasso']):
            nx.draw_networkx_nodes(graph,pos,node_size=145,node_color='#e9f1f5',
                                   edgecolors='#466379',linewidths=.8,ax=ax)
            nx.draw_networkx_labels(graph,pos,font_size=8,ax=ax)
            if method=='truth':
                nx.draw_networkx_edges(graph,pos,edgelist=sorted(visible_truth),edge_color='#466379',
                                       node_size=145,arrowsize=10,alpha=.7,ax=ax)
                title=f'Truth\n{len(visible_truth)} edges in this view'
            else:
                name=f'{method}_beta{beta:g}_noise{noise:g}'
                coeff=np.load(folder/f'{name}.npz')['a_hat']
                row=json.loads((folder/f'{name}.json').read_text())['metrics']
                scores=abs(coeff).copy()
                np.fill_diagonal(scores,-np.inf)
                indices=np.argsort(-scores.ravel(),kind='stable')[:CONFIG['edges']]
                chosen_global={(int(k%n),int(k//n)) for k in indices if scores.ravel()[k]>0}
                chosen={e for e in chosen_global if e[0]<25 and e[1]<25}
                tp,fn,fp=chosen & visible_truth,visible_truth-chosen,chosen-visible_truth
                assert tp|fn == visible_truth and tp|fp == chosen
                assert tp.isdisjoint(fp) and tp.isdisjoint(fn)
                for selected,color,style in [(tp,'#138777','solid'),(fn,'#db9226','dashed'),(fp,'#c84972','solid')]:
                    nx.draw_networkx_edges(graph,pos,edgelist=sorted(selected),edge_color=color,style=style,
                                           node_size=145,arrowsize=10,alpha=.85,ax=ax)
                title=(f"{method.title()} | full-graph AP = {row['average_precision']:.3f}\n"
                       f"View: {len(tp)} recovered, {len(fn)} missed, {len(fp)} false")
                audit.append(dict(seed=seed,beta=beta,noise_sd=noise,method=method,
                                  displayed_nodes=nodes,visible_true_edges=len(visible_truth),
                                  full_graph_selected_edges=len(chosen_global),
                                  recovered=sorted(tp),missed=sorted(fn),false=sorted(fp)))
            ax.set_title(title,fontsize=11,pad=12)
            ax.set(xlim=(-1.18,1.18),ylim=(-1.18,1.18),aspect='equal')
            ax.axis('off')
        handles=[Line2D([],[],color=c,ls=s,label=t) for c,s,t in
                 [('#138777','-','Recovered true edge'),('#db9226','--','Missed edge'),('#c84972','-','False edge')]]
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=11)
        fig.suptitle(f'SIS: seed {seed}, β = {beta:g}, measurement-noise SD = {noise:g}\n'
                     'Same 25-node view; up to 300 highest nonzero absolute scores on the full graph',
                     fontsize=12)
        fig.tight_layout(rect=(0,.07,1,.86))
        name='network_example.png' if noise==0 else 'network_example_noisy.png'
        fig.savefig(OUT/name,dpi=180)
        plt.close(fig)
    dump(OUT/'network_example_details.json',audit)




def main():
    with threadpool_limits(limits=1):
        rows=verify_and_load()
        summary,differences=aggregate(rows)
        plots(rows,summary)
        network_pictures()
    for row in summary:
        print(f"noise={row['noise_sd']:g} beta={row['beta']:g} {row['method']}: "
              f"AP={row['average_precision_mean']:.3f} +/- {row['average_precision_sd']:.3f}; "
              f"RMSE/persistence={row['rmse_ratio_mean']:.3f}")
    print(OUT)


if __name__=='__main__':
    main()
