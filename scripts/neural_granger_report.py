"""Verify cMLP predictions and compare neural and regression network recovery."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from network_recovery import ROOT,dump,write_csv
from make_report import independent_ap
from sis_recovery import transitions
from community_recovery import write_mixed_csv,edge_strata
from neural_granger import CONFIG,BatchedCMLP,load_case,predict

METHODS=['ridge','lasso','cmlp','sis_aware','local_poly']
LABELS={'ridge':'Ridge','lasso':'Raw Lasso','cmlp':'cMLP',
        'sis_aware':'Model-based\nLasso','local_poly':'Local polynomial +\nmodel-based Lasso'}
COLORS={'ridge':'#8996a4','lasso':'#bd783c','cmlp':'#377bb5','sis_aware':'#16877e','local_poly':'#754ba1'}


def verify(out,seeds,pilot):
    protocol=json.loads((out/'protocol.json').read_text())
    for name,digest in protocol['script_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    for name,digest in protocol['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in protocol['upstream']['sha256'].items():
        assert hashlib.sha256((ROOT/'third_party/neural_gc'/name).read_bytes()).hexdigest()==digest
    rows,trials,selections=[],[],[]
    max_difference=0.
    for seed in seeds:
        for topology in CONFIG['topologies']:
            data,_=load_case(seed,topology,pilot)
            for noise in CONFIG['noise_sd']:
                name=f'{topology}_noise{noise:g}'
                folder=out/f'seed_{seed}'
                record=json.loads((folder/f'{name}.json').read_text())
                row=record['metrics'];arrays=dict(np.load(folder/f'{name}.npz'))
                x,y,_=transitions(data['clean']+noise*data['measurement_draw'])
                sx,sy=StandardScaler().fit(x),StandardScaler().fit(y)
                for key,value in [('x_mean',sx.mean_),('x_scale',sx.scale_),('y_mean',sy.mean_),('y_scale',sy.scale_)]:
                    np.testing.assert_array_equal(arrays[key],value)
                mask=~np.eye(len(data['truth']),dtype=bool)
                scores=np.sqrt(np.square(arrays['w1']).sum(axis=1))
                np.testing.assert_array_equal(scores,arrays['scores'])
                np.testing.assert_allclose(independent_ap(data['truth'][mask],scores[mask]),row['average_precision'],atol=1e-12)
                tx,ty,_=transitions(data['test_clean']+noise*data['test_measurement_draw'])
                predicted=predict(arrays,tx)
                np.testing.assert_allclose(np.sqrt(np.mean((predicted-ty)**2)),row['test_rmse'],atol=1e-12)
                model=BatchedCMLP(tx.shape[-1],CONFIG['hidden'],[0],dtype=torch.float64)
                with torch.no_grad():
                    for key,param in model.named_parameters():param.copy_(torch.tensor(arrays[key][None],dtype=torch.float64))
                    direct=model(torch.tensor(((tx-sx.mean_)/sx.scale_)[None],dtype=torch.float64)).numpy()[0]
                direct=direct*sy.scale_+sy.mean_
                difference=float(abs(direct-predicted).max())
                np.testing.assert_allclose(direct,predicted,atol=1e-10)
                max_difference=max(max_difference,difference)
                best=min(record['trials'],key=lambda r:r['cv_mse'])
                assert (best['alpha'],best['epoch'])==(row['alpha'],row['epoch'])
                assert len(record['trials'])==len(CONFIG['lambdas'])*len(CONFIG['epochs'])
                for trial in record['trials']:
                    np.testing.assert_allclose(np.mean(np.square(trial['fold_rmse'])),trial['cv_mse'],atol=1e-14)
                    trials.append(dict(seed=seed,topology=topology,noise_sd=noise,alpha=trial['alpha'],
                        optimizer_steps=trial['epoch'],cv_mse=trial['cv_mse'],
                        **{f'fold_{i+1}_rmse':r for i,r in enumerate(trial['fold_rmse'])}))
                if topology=='community':
                    for key,value in edge_strata(data['truth'],scores,data['labels']).items():
                        np.testing.assert_allclose(value,row[key])
                rows.append(row)
                selections.append(dict(seed=seed,topology=topology,noise_sd=noise,alpha=row['alpha'],
                    optimizer_steps=row['epoch'],cv_rmse=row['cv_rmse'],cv_fold_sd=row['cv_fold_sd'],
                    final_objective=record['training_history']['final'][-1]['objective'][0]))
                for method in ['ridge','lasso','sis_aware','local_poly']:
                    if topology=='community':
                        old=ROOT/'results'/('community_pilot' if pilot else 'community')/f'seed_{seed}'/f'{method}_noise{noise:g}.json'
                    else:
                        base='sis' if method in ['ridge','lasso'] else 'sis_aware' if method=='sis_aware' else 'sis_local_poly'
                        old=ROOT/'results'/(base+'_pilot' if pilot else base)/f'seed_{seed}'/f'{method}_beta3_noise{noise:g}.json'
                    rows.append(dict(json.loads(old.read_text())['metrics'],topology=topology))
    write_mixed_csv(out/'comparison_metrics.csv',rows)
    write_csv(out/'cv_trials.csv',trials)
    write_csv(out/'selected_settings.csv',selections)
    dump(out/'verification.json',dict(neural_fits=len(seeds)*4,comparison_fits=len(rows),
        independent_AP_checks=len(seeds)*4,independent_RMSE_checks=len(seeds)*4,
        independent_torch_prediction_checks=len(seeds)*4,max_torch_numpy_difference=max_difference,
        selected_CV_settings_checked=len(seeds)*4,scaler_arrays_checked=len(seeds)*16,
        upstream_source_files_checked=3))
    return rows,trials


def aggregate(rows,out,seeds):
    draws=np.random.default_rng(85420).integers(0,len(seeds),size=(5000,len(seeds)))
    summary,paired=[],[]
    for topology in CONFIG['topologies']:
        for noise in CONFIG['noise_sd']:
            groups={method:sorted([r for r in rows if r['topology']==topology and r['noise_sd']==noise and r['method']==method],
                                  key=lambda r:r['seed']) for method in METHODS}
            metrics=['average_precision','test_rmse']
            if topology=='community':metrics+=['within_recall_at_300','bridge_recall_at_300']
            for method,group in groups.items():
                assert [r['seed'] for r in group]==seeds
                for metric in metrics:
                    values=np.array([r[metric] for r in group])
                    low,high=np.quantile(values[draws].mean(axis=1),[.025,.975])
                    summary.append(dict(topology=topology,noise_sd=noise,method=method,metric=metric,
                        mean=float(values.mean()),sd=float(values.std(ddof=1)) if len(seeds)>1 else 0.,
                        ci_low=float(low),ci_high=float(high)))
            for method in ['ridge','lasso','sis_aware','local_poly']:
                for metric in metrics:
                    values=np.array([a[metric]-b[metric] for a,b in zip(groups['cmlp'],groups[method])])
                    low,high=np.quantile(values[draws].mean(axis=1),[.025,.975])
                    paired.append(dict(topology=topology,noise_sd=noise,reference=method,metric=metric,
                        cmlp_minus_reference=float(values.mean()),ci_low=float(low),ci_high=float(high)))
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_differences.csv',paired)
    return summary


def plots(summary,trials,out,n):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'figure.facecolor':'white','savefig.facecolor':'white'})
    for metric,ylabel,file in [('average_precision','Directed-edge average precision','recovery_comparison.png'),
                               ('test_rmse','Held-out one-step RMSE','prediction_comparison.png')]:
        fig,axes=plt.subplots(2,2,figsize=(14,8.5),layout='constrained')
        for i,topology in enumerate(CONFIG['topologies']):
            for j,noise in enumerate(CONFIG['noise_sd']):
                ax=axes[i,j]
                group=[next(r for r in summary if r['topology']==topology and r['noise_sd']==noise and
                     r['method']==m and r['metric']==metric) for m in METHODS]
                means=np.array([r['mean'] for r in group])
                error=np.array([[r['mean']-r['ci_low'] for r in group],[r['ci_high']-r['mean'] for r in group]])
                ax.bar(range(5),means,color=[COLORS[m] for m in METHODS],width=.65,
                       yerr=error if n>1 else None,capsize=4)
                upper=max(r['ci_high'] for r in group)
                for k,r in enumerate(group):
                    ax.text(k,r['ci_high']+(.02 if metric=='average_precision' else upper*.04),
                            f"{r['mean']:.3f}" if metric=='average_precision' else f"{r['mean']:.5f}",ha='center',fontsize=9)
                ax.set(xticks=range(5),xticklabels=[LABELS[m] for m in METHODS],ylabel=ylabel,
                       title=f'{"Random network" if topology=="random" else "Four communities"} | noise SD = {noise:g}',
                       ylim=(0,1.13 if metric=='average_precision' else upper*1.23))
        note=f'{n} networks; means and 95% graph-bootstrap intervals' if n>1 else 'Independent pilot network'
        fig.suptitle('SIS dynamics, β = 3 | '+note)
        fig.savefig(out/file,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,topology in enumerate(CONFIG['topologies']):
        for j,noise in enumerate(CONFIG['noise_sd']):
            ax=axes[i,j]
            for penalty in CONFIG['lambdas']:
                values=[np.mean([np.sqrt(r['cv_mse']) for r in trials if r['topology']==topology and
                    r['noise_sd']==noise and r['alpha']==penalty and r['optimizer_steps']==step]) for step in CONFIG['epochs']]
                ax.plot(CONFIG['epochs'],values,'o-',label=f'λ={penalty:g}',markersize=4)
            ax.set(title=f'{topology.capitalize()} | noise SD = {noise:g}',xlabel='Optimizer steps',ylabel='Mean CV RMSE')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('cMLP training duration and regularization | five whole-trajectory folds')
    fig.savefig(out/'validation_curves.png',dpi=180);plt.close(fig)


def network_picture(out,seed,pilot):
    data,_=load_case(seed,'community',pilot)
    truth,labels=data['truth'],data['labels']
    edges={(int(j),int(i)) for i,j in zip(*np.where(truth))}
    centers=[(-1.35,1.35),(1.35,1.35),(-1.35,-1.35),(1.35,-1.35)]
    pos={i:np.array(centers[labels[i]])+.8*np.array([np.cos(2*np.pi*(i%25)/25),np.sin(2*np.pi*(i%25)/25)]) for i in range(100)}
    graph=nx.DiGraph();graph.add_nodes_from(range(100));graph.add_edges_from(sorted(edges))
    source=ROOT/'results'/('community_pilot' if pilot else 'community')/f'seed_{seed}'
    fig,axes=plt.subplots(1,4,figsize=(18,6))
    palette=['#8fc7ce','#edc28a','#b5abd5','#b8cf9a']
    for ax,method in zip(axes,['truth','lasso','cmlp','local_poly']):
        if method=='truth':
            inside={e for e in edges if labels[e[0]]==labels[e[1]]}
            selections=[(inside,'#7e8f9d','solid',.4),(edges-inside,'#254a67','solid',.9)]
            title='Truth\n240 internal edges; 60 bridges'
        else:
            folder,name=(out/f'seed_{seed}','community_noise0.02') if method=='cmlp' else (source,f'{method}_noise0.02')
            fitted=np.load(folder/f'{name}.npz')
            scores=fitted['scores'].copy() if method=='cmlp' else abs(fitted['weights']).copy()
            np.fill_diagonal(scores,-np.inf)
            ranked=np.argsort(-scores.ravel(),kind='stable')[:300]
            chosen={(int(k%100),int(k//100)) for k in ranked if scores.ravel()[k]>0}
            tp,fn,fp=chosen&edges,edges-chosen,chosen-edges
            selections=[(fn,'#d9993d','dashed',.45),(fp,'#d05a7c','solid',.45),(tp,'#138777','solid',.7)]
            row=json.loads((folder/f'{name}.json').read_text())['metrics']
            title=f"{LABELS[method]} | AP = {row['average_precision']:.3f}\n"+\
                  f"Internal recall {row['within_recall_at_300']:.0%}; bridge recall {row['bridge_recall_at_300']:.0%}"
        for selected,color,style,alpha in selections:
            nx.draw_networkx_edges(graph,pos,edgelist=sorted(selected),edge_color=color,style=style,alpha=alpha,
                                   width=.7,arrowsize=5,node_size=28,connectionstyle='arc3,rad=0.05',ax=ax)
        nx.draw_networkx_nodes(graph,pos,node_size=28,node_color=[palette[g] for g in labels],edgecolors='white',linewidths=.35,ax=ax)
        for i,(cx,cy) in enumerate(centers):ax.text(cx,cy,f'C{i+1}',ha='center',va='center',fontsize=10,color='#384958')
        ax.set_title(title,fontsize=10,pad=12)
        ax.set(xlim=(-2.35,2.35),ylim=(-2.35,2.35),aspect='equal');ax.axis('off')
    fig.legend(handles=[Line2D([],[],color=c,ls=s,label=l) for c,s,l in
          [('#138777','-','Recovered true edge'),('#d9993d','--','Missed edge'),('#d05a7c','-','False edge')]],
          loc='lower center',ncol=3,frameon=False)
    fig.suptitle(f'Four communities | seed {seed}, SIS β = 3, noise SD = 0.02\nAll 100 nodes; top 300 positive edge scores')
    fig.tight_layout(rect=(0,.06,1,.85));fig.savefig(out/'network_example_noisy.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    out=ROOT/'results'/('neural_granger_pilot' if args.pilot else 'neural_granger')
    seeds=[5420] if args.pilot else CONFIG['seeds']
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        rows,trials=verify(out,seeds,args.pilot)
        summary=aggregate(rows,out,seeds)
        plots(summary,trials,out,len(seeds));network_picture(out,seeds[0],args.pilot)
    dump(out/'report_protocol.json',dict(bootstrap_seed=85420,bootstrap_resamples=5000,
        bootstrap_unit='Paired network seeds; percentile intervals for means and cMLP-minus-reference differences',
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for row in summary:
        if row['method']=='cmlp' and row['metric'] in ['average_precision','test_rmse']:print(row)


if __name__=='__main__':main()
