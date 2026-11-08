"""Verify community-network experiments and compare topology and bridge recovery."""
import argparse
import hashlib
import json

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score

from network_recovery import ROOT, dump, write_csv
from sis_recovery import CONFIG as SIS, simulate_sis, transitions
from sis_aware import CONFIG as AWARE
from sis_local_poly import regression_samples
from community_recovery import CONFIG, community_system, write_mixed_csv
from make_report import independent_ap

METHODS = CONFIG['methods']
LABELS = {'ridge':'Ridge','lasso':'Raw Lasso','sis_aware':'Model-based Lasso','local_poly':'Local polynomial +\nmodel-based Lasso'}


def verify(out, seeds, pilot):
    protocol = json.loads((out/'protocol.json').read_text())
    for name, digest in protocol['script_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    rows, diagnostics, cv_rows = [], [], []
    max_kkt = 0.
    source = ROOT/'results'/('sis_pilot' if pilot else 'sis')
    for seed in seeds:
        folder = out/f'seed_{seed}'
        data = np.load(folder/'system.npz')
        b,truth,labels = community_system(seed)
        for name, expected in [('b',b),('truth',truth),('labels',labels)]:
            np.testing.assert_array_equal(data[name],expected)
        original = np.load(source/f'seed_{seed}/system.npz')
        for key in ['initial','test_initial','measurement_draw','test_measurement_draw']:
            np.testing.assert_array_equal(data[key],original[key])
        for key,initial in [('clean',data['initial']),('test_clean',data['test_initial'])]:
            np.testing.assert_allclose(data[key],simulate_sis(b,initial,3.,SIS['steps'],SIS['delta'],SIS['dt']),atol=1e-14)
        graph = nx.from_numpy_array(truth.T,create_using=nx.DiGraph)
        diagnostics.append(dict(seed=seed,nodes=len(b),edges=int(truth.sum()),
                                internal_edges=int(truth[labels[:,None]==labels[None,:]].sum()),
                                bridge_edges=int(truth[labels[:,None]!=labels[None,:]].sum()),
                                largest_strong_component=max(map(len,nx.strongly_connected_components(graph))),
                                max_in_degree=max(dict(graph.in_degree()).values()),
                                max_out_degree=max(dict(graph.out_degree()).values()),
                                step_probability_bound=float(.05*3*b.sum(axis=1).max())))
        mask = ~np.eye(len(b),dtype=bool)
        for noise in CONFIG['noise_sd']:
            tx,ty,_ = transitions(data['test_clean']+noise*data['test_measurement_draw'])
            for method in METHODS:
                name = f'{method}_noise{noise:g}'
                record = json.loads((folder/f'{name}.json').read_text())
                row = record['metrics']
                for trial in record['trials']:
                    cv_rows.append(dict(seed=seed,noise_sd=noise,method=method,
                                        window=trial.get('window',''),alpha=trial.get('alpha',''),
                                        fraction=trial.get('fraction',''),cv_mse=trial['cv_mse'],
                                        **{f'fold_{k+1}_rmse':v for k,v in enumerate(trial['fold_rmse'])}))
                fitted = np.load(folder/f'{name}.npz')
                weights = fitted['weights']
                ap = independent_ap(truth[mask],abs(weights[mask]))
                np.testing.assert_allclose(ap,row['average_precision'],atol=1e-12)
                if method in ['ridge','lasso']:
                    predicted = tx@weights.T+fitted['intercept']
                    best = min(record['trials'],key=lambda t:t['cv_mse'])
                    assert best['alpha']==row['alpha']
                else:
                    predicted = (1-.05)*tx+.05*3*(1-tx)*(tx@weights.T)
                    best = min(record['trials'],key=lambda t:t['cv_mse'])
                    assert best['fraction']==row['fraction']
                    window = row['window'] if method=='local_poly' else 0
                    if method=='local_poly': assert best['window']==window
                    x,y = regression_samples(data['clean']+noise*data['measurement_draw'],window)
                    assert weights.min()>=-1e-12 and np.all(np.diag(weights)==0)
                    for node in range(len(b)):
                        keep = np.arange(len(b))!=node
                        design = .05*3*(1-x[:,node,None])*x[:,keep]
                        target = y[:,node]-.95*x[:,node]
                        sx = np.sqrt(np.mean(design**2,axis=0));sx[sx<=1e-14]=1.
                        sy = max(float(np.sqrt(np.mean(target**2))),1e-14)
                        xs,ys = design/sx,target/sy
                        coef = weights[node,keep]*sx/sy
                        alpha = row['fraction']*max(0.,float((xs.T@ys/len(xs)).max()))
                        grad = xs.T@(xs@coef-ys)/len(xs)+alpha
                        residual = float(np.where(coef>1e-10,abs(grad),np.maximum(-grad,0.)).max())
                        assert residual<=AWARE['kkt_tolerance']+1e-12
                        max_kkt = max(max_kkt,residual)
                np.testing.assert_allclose(np.sqrt(np.mean((predicted-ty)**2)),row['test_rmse'],atol=1e-12)
                scores = abs(weights).copy();scores[~mask]=-np.inf
                ranked = np.argsort(-scores.ravel(),kind='stable')[:300]
                chosen = np.zeros_like(truth);chosen.ravel()[ranked[scores.ravel()[ranked]>0]]=True
                for kind,subset in [('within',mask&(labels[:,None]==labels[None,:])),
                                    ('bridge',mask&(labels[:,None]!=labels[None,:]))]:
                    np.testing.assert_allclose(average_precision_score(truth[subset],scores[subset]),row[kind+'_ap'])
                    np.testing.assert_allclose((chosen&truth&subset).sum()/truth[subset].sum(),row[kind+'_recall_at_300'])
                rows.append(row)
                original_folder = source if method in ['ridge','lasso'] else ROOT/'results'/(
                    ('sis_aware_pilot' if pilot else 'sis_aware') if method=='sis_aware' else
                    ('sis_local_poly_pilot' if pilot else 'sis_local_poly'))
                record = json.loads((original_folder/f'seed_{seed}/{method}_beta3_noise{noise:g}.json').read_text())
                rows.append(dict(record['metrics'],topology='random'))
    write_csv(out/'graph_diagnostics.csv',diagnostics)
    write_csv(out/'cv_trials.csv',cv_rows)
    write_mixed_csv(out/'comparison_metrics.csv',rows)
    dump(out/'verification.json',dict(community_fits=len(seeds)*8,random_reference_fits=len(seeds)*8,
         matched_initial_and_noise_arrays=len(seeds)*4,reproduced_trajectories=len(seeds)*30,
         checked_community_AP=len(seeds)*8,checked_community_test_RMSE=len(seeds)*8,
         checked_edge_stratum_scores=len(seeds)*8*4,checked_nodewise_KKT=len(seeds)*4*100,
         max_selected_KKT_residual=max_kkt))
    return rows


def aggregate(rows,out,seeds):
    draws = np.random.default_rng(85420).integers(0,len(seeds),size=(5000,len(seeds)))
    summary,paired = [], []
    for noise in CONFIG['noise_sd']:
        for topology in ['random','community']:
            for method in METHODS:
                group = sorted([r for r in rows if r['topology']==topology and r['noise_sd']==noise and r['method']==method],
                               key=lambda r:r['seed'])
                metrics = ['average_precision','test_rmse']
                if topology=='community': metrics += ['within_ap','bridge_ap','within_recall_at_300','bridge_recall_at_300']
                for metric in metrics:
                    values = np.array([r[metric] for r in group])
                    low,high = np.quantile(values[draws].mean(axis=1),[.025,.975])
                    summary.append(dict(topology=topology,noise_sd=noise,method=method,metric=metric,
                                        graphs=len(seeds),mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(seeds)>1 else 0.,
                                        ci_low=float(low),ci_high=float(high)))
        for method in METHODS:
            values = {top:np.array([r['average_precision'] for r in sorted(rows,key=lambda r:r['seed'])
                      if r['topology']==top and r['noise_sd']==noise and r['method']==method]) for top in ['random','community']}
            diff = values['community']-values['random']
            low,high = np.quantile(diff[draws].mean(axis=1),[.025,.975])
            paired.append(dict(comparison='community_minus_random',noise_sd=noise,method=method,reference=method,
                               metric='average_precision',mean=float(diff.mean()),ci_low=float(low),ci_high=float(high)))
        groups = {method:sorted([r for r in rows if r['topology']=='community' and r['noise_sd']==noise and r['method']==method],
                               key=lambda r:r['seed']) for method in METHODS}
        for method in METHODS:
            diff = np.array([r['bridge_recall_at_300']-r['within_recall_at_300'] for r in groups[method]])
            low,high = np.quantile(diff[draws].mean(axis=1),[.025,.975])
            paired.append(dict(comparison='bridge_minus_within',noise_sd=noise,method=method,reference=method,
                               metric='recall_at_300',mean=float(diff.mean()),ci_low=float(low),ci_high=float(high)))
        for method in ['ridge','lasso','sis_aware']:
            for metric in ['average_precision','within_recall_at_300','bridge_recall_at_300']:
                diff = np.array([a[metric]-b[metric] for a,b in zip(groups['local_poly'],groups[method])])
                low,high = np.quantile(diff[draws].mean(axis=1),[.025,.975])
                paired.append(dict(comparison='local_poly_minus_reference',noise_sd=noise,method='local_poly',reference=method,
                                   metric=metric,mean=float(diff.mean()),ci_low=float(low),ci_high=float(high)))
    write_csv(out/'summary.csv',summary)
    write_csv(out/'paired_differences.csv',paired)
    selections = [dict(seed=r['seed'],noise_sd=r['noise_sd'],window=r['window'],fraction=r['fraction'])
                  for r in rows if r['topology']=='community' and r['method']=='local_poly']
    write_csv(out/'selected_windows.csv',selections)
    return summary


def figures(summary,out,n):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    def bars(ax,topology,noise,metric,offset,color,label):
        group = [next(r for r in summary if r['topology']==topology and r['noise_sd']==noise and
                      r['method']==method and r['metric']==metric) for method in METHODS]
        means = np.array([r['mean'] for r in group])
        errors = np.array([[r['mean']-r['ci_low'] for r in group],[r['ci_high']-r['mean'] for r in group]])
        ax.bar(np.arange(4)+offset,means,width=.34,color=color,label=label,yerr=errors if n>1 else None,capsize=4)
        for i,value in enumerate(means): ax.text(i+offset,group[i]['ci_high']+.025,f'{value:.3f}',ha='center',fontsize=9)
    fig,axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,noise in zip(axes,CONFIG['noise_sd']):
        bars(ax,'random',noise,'average_precision',-.18,'#9aa7b5','Random graph')
        bars(ax,'community',noise,'average_precision',.18,'#754ba1','Four communities')
        ax.set(title=f'Measurement-noise SD = {noise:g}',ylabel='Edge average precision',ylim=(0,1.14),
               xticks=range(4),xticklabels=['Ridge','Raw Lasso','Model-based\nLasso','Local polynomial +\nmodel-based Lasso'])
    axes[0].legend(loc='upper left',frameon=False,fontsize=9)
    interval_note = f'{n} networks per topology; mean and 95% graph-bootstrap intervals' if n>1 else 'Single-network pilot'
    fig.suptitle(f'Does community structure change network recovery? | SIS β = 3\n'+interval_note)
    fig.savefig(out/'topology_comparison.png',dpi=200);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,noise in zip(axes,CONFIG['noise_sd']):
        bars(ax,'community',noise,'within_recall_at_300',-.18,'#167f83','Within-community edges (240)')
        bars(ax,'community',noise,'bridge_recall_at_300',.18,'#c78539','Bridge edges (60)')
        ax.set(title=f'Measurement-noise SD = {noise:g}',ylabel='Recall among top 300 positive edge scores',ylim=(0,1.14),
               xticks=range(4),xticklabels=['Ridge','Raw Lasso','Model-based\nLasso','Local polynomial +\nmodel-based Lasso'])
    axes[0].legend(loc='upper left',frameon=False,fontsize=8)
    interval_note = f'Mean and 95% bootstrap intervals across {n} networks' if n>1 else 'Single-network pilot'
    fig.suptitle(f'Which connections are recovered? | four communities, SIS β = 3\n'+interval_note)
    fig.savefig(out/'bridge_recovery.png',dpi=200);plt.close(fig)


def network_picture(out,seed):
    folder = out/f'seed_{seed}'
    data = np.load(folder/'system.npz');truth,labels=data['truth'],data['labels']
    all_edges = {(int(j),int(i)) for i,j in zip(*np.where(truth))}
    centers = [(-1.35,1.35),(1.35,1.35),(-1.35,-1.35),(1.35,-1.35)]
    palette = ['#8fc7ce','#edc28a','#b5abd5','#b8cf9a']
    pos = {node:np.array(centers[labels[node]])+.8*np.array([np.cos(2*np.pi*(node%25)/25),
                                                          np.sin(2*np.pi*(node%25)/25)]) for node in range(100)}
    graph = nx.DiGraph();graph.add_nodes_from(range(100));graph.add_edges_from(sorted(all_edges))
    details=[]
    for noise in CONFIG['noise_sd']:
        fig,axes = plt.subplots(1,4,figsize=(18,6))
        for ax,method in zip(axes,['truth','lasso','sis_aware','local_poly']):
            if method=='truth':
                within={e for e in all_edges if labels[e[0]]==labels[e[1]]}
                selections=[(within,'#7e8f9d','solid',.4),(all_edges-within,'#254a67','solid',.9)]
                title='Truth\n240 internal edges; 60 bridges'
            else:
                weights=np.load(folder/f'{method}_noise{noise:g}.npz')['weights']
                scores=abs(weights).copy();np.fill_diagonal(scores,-np.inf)
                ranked=np.argsort(-scores.ravel(),kind='stable')[:300]
                selected={(int(i%100),int(i//100)) for i in ranked if scores.ravel()[i]>0}
                tp,fn,fp=selected&all_edges,all_edges-selected,selected-all_edges
                selections=[(fn,'#d9993d','dashed',.45),(fp,'#d05a7c','solid',.45),(tp,'#138777','solid',.7)]
                row=json.loads((folder/f'{method}_noise{noise:g}.json').read_text())['metrics']
                title=f"{LABELS[method]} | AP = {row['average_precision']:.3f}\n"+\
                      f"Internal recall {row['within_recall_at_300']:.0%}; bridge recall {row['bridge_recall_at_300']:.0%}"
                details.append(dict(seed=seed,noise_sd=noise,method=method,recovered=len(tp),missed=len(fn),false=len(fp)))
            for edges,color,style,alpha in selections:
                nx.draw_networkx_edges(graph,pos,edgelist=sorted(edges),edge_color=color,style=style,alpha=alpha,
                                       width=.7,arrowsize=5,node_size=28,connectionstyle='arc3,rad=0.05',ax=ax)
            nx.draw_networkx_nodes(graph,pos,node_size=28,node_color=[palette[g] for g in labels],
                                   edgecolors='#fff',linewidths=.35,ax=ax)
            for i,(cx,cy) in enumerate(centers): ax.text(cx,cy,f'C{i+1}',ha='center',va='center',fontsize=10,color='#384958')
            ax.set_title(title,fontsize=10,pad=12)
            ax.set(xlim=(-2.35,2.35),ylim=(-2.35,2.35),aspect='equal');ax.axis('off')
        fig.legend(handles=[Line2D([],[],color=c,ls=ls,label=label) for c,ls,label in
                    [('#138777','-','Recovered true edge'),('#d9993d','--','Missed edge'),('#d05a7c','-','False edge')]],
                   loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'Four communities | seed {seed}, SIS β = 3, measurement-noise SD = {noise:g}\n'
                     'All 100 nodes; same positions; top 300 positive edge scores',fontsize=13)
        fig.tight_layout(rect=(0,.06,1,.85))
        name='network_example.png' if noise==0 else 'network_example_noisy.png'
        fig.savefig(out/name,dpi=200);plt.close(fig)
    dump(out/'network_example_details.json',details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results'/('community_pilot' if args.pilot else 'community')
    seeds=[5420] if args.pilot else CONFIG['seeds']
    with threadpool_limits(limits=1):
        rows=verify(out,seeds,args.pilot)
        summary=aggregate(rows,out,seeds)
        figures(summary,out,len(seeds))
        network_picture(out,seeds[0])
    dump(out/'report_protocol.json',dict(bootstrap_seed=85420,bootstrap_resamples=5000,
         bootstrap_unit='Seed-paired networks; percentile 95% intervals for means and mean differences',
         edge_ranking='Top 300 positive absolute off-diagonal scores, shared globally across edge strata',
         network_example_seed=seeds[0],
         report_sha256=hashlib.sha256((ROOT/'scripts/community_report.py').read_bytes()).hexdigest()))
    for row in summary:
        if row['topology']=='community' and row['metric']=='average_precision':
            print(f"noise={row['noise_sd']:g} {row['method']}: AP={row['mean']:.4f} "
                  f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")


if __name__=='__main__':
    main()
