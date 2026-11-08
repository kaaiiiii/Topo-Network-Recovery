"""Independent checks and comparison of Model-based Lasso and raw-state regressions."""
import csv
import hashlib
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from network_recovery import ROOT, dump, write_csv
from sis_recovery import CONFIG as SIS, transitions
from sis_aware import CONFIG
from make_report import independent_ap

OUT=ROOT/'results/sis_aware'
SOURCE=ROOT/'results/sis'
METHODS=['ridge','lasso','sis_aware']
LABELS={'ridge':'Ridge','lasso':'Raw-state Lasso','sis_aware':'Model-based Lasso'}
COLORS={'ridge':'#8d99a7','lasso':'#b86829','sis_aware':'#107f83'}


def verify_and_load():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'scripts/sis_aware.py').read_bytes()).hexdigest()==protocol['script_sha256']
    rows=[]
    max_kkt=0.
    for seed in SIS['seeds']:
        system=np.load(SOURCE/f'seed_{seed}'/'system.npz')
        truth,b=system['truth'],system['b']
        mask=~np.eye(len(b),dtype=bool)
        for beta in SIS['beta']:
            data=np.load(SOURCE/f'seed_{seed}'/f'trajectories_beta{beta:g}.npz')
            for noise in SIS['noise_sd']:
                test=data['test_clean']+noise*system['test_measurement_draw']
                tx,ty,_=transitions(test)
                for method in METHODS:
                    folder=(OUT if method=='sis_aware' else SOURCE)/f'seed_{seed}'
                    name=f'{method}_beta{beta:g}_noise{noise:g}'
                    record=json.loads((folder/f'{name}.json').read_text())
                    row=record['metrics']
                    fitted=np.load(folder/f'{name}.npz')
                    if method=='sis_aware':
                        estimate=fitted['b_hat']
                        assert estimate.min()>=0 and np.all(np.diag(estimate)==0)
                        prediction=(1-SIS['dt']*SIS['delta'])*tx+SIS['dt']*beta*(1-tx)*(tx@estimate.T)
                        x,y,_=transitions(data['clean']+noise*system['measurement_draw'])
                        for node in range(len(b)):
                            keep=np.arange(len(b))!=node
                            design=SIS['dt']*beta*(1-x[:,node,None])*x[:,keep]
                            target=y[:,node]-(1-SIS['dt']*SIS['delta'])*x[:,node]
                            sx=np.sqrt(np.mean(design**2,axis=0)); sx[sx<=1e-14]=1.
                            sy=float(np.sqrt(np.mean(target**2)))
                            sy=sy if sy>1e-14 else 1.
                            xs,ys=design/sx,target/sy
                            coef=estimate[node,keep]*sx/sy
                            alpha=row['fraction']*max(0.,float((xs.T@ys/len(xs)).max()))
                            grad=xs.T@(xs@coef-ys)/len(xs)+alpha
                            residual=float(np.where(coef>1e-10,abs(grad),np.maximum(-grad,0)).max())
                            assert residual<=CONFIG['kkt_tolerance']+1e-12
                            max_kkt=max(max_kkt,residual)
                        best=min(record['trials'],key=lambda r:r['cv_mse'])
                        assert best['fraction']==row['fraction'] and len(record['trials'])==15
                        np.testing.assert_allclose(np.linalg.norm(estimate-b)/np.linalg.norm(b),row['weight_relative_error'])
                        assert row['max_kkt_residual']<=CONFIG['kkt_tolerance']
                    else:
                        estimate=fitted['a_hat']
                        prediction=tx@estimate.T+fitted['intercept']
                    ap=independent_ap(truth[mask],abs(estimate[mask]))
                    np.testing.assert_allclose(ap,row['average_precision'],atol=1e-12)
                    rmse=float(np.sqrt(np.mean((prediction-ty)**2)))
                    np.testing.assert_allclose(rmse,row['test_rmse'],atol=1e-12)
                    np.testing.assert_allclose(np.sqrt(np.mean((tx-ty)**2)),row['persistence_rmse'],atol=1e-12)
                    precision,recall=fitted['precision'][::-1],fitted['recall'][::-1]
                    grid=np.arange(301)/300
                    curve=precision[np.searchsorted(recall,grid-1e-12,side='left').clip(max=len(precision)-1)]
                    np.testing.assert_allclose(curve[1:].mean(),ap,atol=1e-12)
                    rows.append(dict(**row,curve=curve))
    assert len(rows)==180
    aware=[r for r in rows if r['method']=='sis_aware']
    dump(OUT/'verification.json',dict(total_compared_fits=180,new_fits=60,source_hash_checks=len(protocol['input_sha256']),
         independent_AP_checks=180,independent_test_RMSE_checks=180,PR_area_checks=180,
         independently_checked_nodewise_KKT_conditions=6000,max_selected_KKT_residual=max_kkt,
         max_all_candidate_KKT_residual=max(r['max_kkt_residual'] for r in aware),
         solver_warnings=sum(r['solver_warnings'] for r in aware),
         coordinate_descent_polished_candidates=sum(r['polished_candidates'] for r in aware)))
    return rows


def aggregate(rows):
    summary,paired=[],[]
    draws=np.random.default_rng(85420).integers(0,10,size=(5000,10))
    for noise in SIS['noise_sd']:
        for beta in SIS['beta']:
            groups={m:sorted([r for r in rows if r['method']==m and r['noise_sd']==noise and r['beta']==beta],
                             key=lambda r:r['seed']) for m in METHODS}
            for method,group in groups.items():
                row=dict(noise_sd=noise,beta=beta,method=method,graphs=10)
                for metric in ['average_precision','test_rmse','rmse_ratio']:
                    values=np.array([r[metric] for r in group])
                    row[metric+'_mean']=float(values.mean())
                    row[metric+'_sd']=float(values.std(ddof=1))
                if method=='sis_aware':
                    row['zero_penalty_selections']=sum(r['fraction']==0 for r in group)
                    row['weight_relative_error_mean']=float(np.mean([r['weight_relative_error'] for r in group]))
                else:
                    row['zero_penalty_selections']=''
                    row['weight_relative_error_mean']=''
                summary.append(row)
            for reference in ['lasso','ridge']:
                for metric in ['average_precision','test_rmse']:
                    diff=np.array([a[metric]-b[metric] for a,b in zip(groups['sis_aware'],groups[reference])])
                    lo,hi=np.quantile(diff[draws].mean(axis=1),[.025,.975])
                    paired.append(dict(noise_sd=noise,beta=beta,reference=reference,metric=metric,
                                       aware_minus_reference=float(diff.mean()),ci_low=float(lo),ci_high=float(hi)))
    write_csv(OUT/'summary.csv',summary)
    write_csv(OUT/'paired_differences.csv',paired)
    return summary,paired


def plots(rows,summary):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        matrix=np.array([[next(r['average_precision_mean'] for r in summary if
                             r['method']==method and r['noise_sd']==noise and r['beta']==beta)
                          for noise in SIS['noise_sd']] for beta in SIS['beta']])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i,beta in enumerate(SIS['beta']):
            for j,noise in enumerate(SIS['noise_sd']):
                r=next(r for r in summary if r['method']==method and r['noise_sd']==noise and r['beta']==beta)
                ax.text(j,i,f"{r['average_precision_mean']:.3f}\n± {r['average_precision_sd']:.3f}",
                        ha='center',va='center',color='white' if matrix[i,j]<.55 else '#172433')
        ax.set(title=LABELS[method],xticks=[0,1],xticklabels=['0','0.02'],yticks=range(3),yticklabels=['0.8','1.5','3.0'],
               xlabel='Measurement-noise SD',ylabel='Spreading strength β')
    fig.colorbar(im,ax=axes,label='Directed-edge average precision')
    fig.suptitle('Does using the SIS equation improve recovery? | mean ± SD across 10 graphs')
    fig.savefig(OUT/'recovery_comparison.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained',sharex=True,sharey=True)
    for i,noise in enumerate(SIS['noise_sd']):
        for j,beta in enumerate(SIS['beta']):
            ax=axes[i,j]
            for method in METHODS:
                curves=np.stack([r['curve'] for r in rows if r['method']==method and r['noise_sd']==noise and r['beta']==beta])
                ax.step(np.arange(301)/300,curves.mean(axis=0),where='pre',color=COLORS[method],label=LABELS[method])
            ax.axhline(300/9900,color='#999999',ls=':',lw=1)
            ax.set(title=f'β = {beta:g}, noise SD = {noise:g}',xlim=(0,1),ylim=(0,1.03))
            ax.grid(alpha=.15)
            if i==1: ax.set_xlabel('Recall')
            if j==0: ax.set_ylabel('Precision')
    fig.legend(handles=[Line2D([],[],color=COLORS[m],label=LABELS[m]) for m in METHODS],
               loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('SIS dynamics | precision–recall on paired observations')
    fig.savefig(OUT/'precision_recall.png',dpi=180); plt.close(fig)


def network_pictures():
    seed,beta=1001,3.
    truth=np.load(SOURCE/f'seed_{seed}'/'system.npz')['truth']
    n=len(truth)
    edges={(int(j),int(i)) for i,j in zip(*np.where(truth))}
    visible={e for e in edges if max(e)<25}
    graph=nx.DiGraph(); graph.add_nodes_from(range(25)); graph.add_edges_from(sorted(visible))
    pos=nx.circular_layout(graph)
    details=[]
    for noise in SIS['noise_sd']:
        fig,axes=plt.subplots(1,4,figsize=(17,5))
        for ax,method in zip(axes,['truth']+METHODS):
            nx.draw_networkx_nodes(graph,pos,node_size=145,node_color='#e9f1f5',edgecolors='#466379',linewidths=.8,ax=ax)
            nx.draw_networkx_labels(graph,pos,font_size=8,ax=ax)
            if method=='truth':
                selections=[(visible,'#466379','solid')]
                title=f'Truth\n{len(visible)} edges in this view'
            else:
                folder=(OUT if method=='sis_aware' else SOURCE)/f'seed_{seed}'
                name=f'{method}_beta{beta:g}_noise{noise:g}'
                fitted=np.load(folder/f'{name}.npz')
                estimate=fitted['b_hat'] if method=='sis_aware' else fitted['a_hat']
                ap=json.loads((folder/f'{name}.json').read_text())['metrics']['average_precision']
                scores=abs(estimate).copy(); np.fill_diagonal(scores,-np.inf)
                indices=np.argsort(-scores.ravel(),kind='stable')[:300]
                chosen={(int(k%n),int(k//n)) for k in indices if scores.ravel()[k]>0}
                chosen={e for e in chosen if max(e)<25}
                tp,fn,fp=chosen&visible,visible-chosen,chosen-visible
                assert (tp|fn)==visible and (tp|fp)==chosen
                selections=[(tp,'#138777','solid'),(fn,'#db9226','dashed'),(fp,'#c84972','solid')]
                title=f'{LABELS[method]} | AP = {ap:.3f}\n{len(tp)} correct, {len(fn)} missed, {len(fp)} false'
                details.append(dict(seed=seed,beta=beta,noise_sd=noise,method=method,
                                    recovered=sorted(tp),missed=sorted(fn),false=sorted(fp)))
            for chosen,color,style in selections:
                nx.draw_networkx_edges(graph,pos,edgelist=sorted(chosen),edge_color=color,style=style,
                                       node_size=145,arrowsize=10,alpha=.8,ax=ax)
            ax.set_title(title,fontsize=10,pad=12)
            ax.set(xlim=(-1.18,1.18),ylim=(-1.18,1.18),aspect='equal'); ax.axis('off')
        fig.legend(handles=[Line2D([],[],color=c,ls=s,label=l) for c,s,l in
                    [('#138777','-','Recovered true edge'),('#db9226','--','Missed edge'),('#c84972','-','False edge')]],
                   loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'SIS: seed {seed}, β = {beta:g}, measurement-noise SD = {noise:g}\n'
                     'Same 25 nodes; up to top 300 nonzero absolute scores on the full graph',fontsize=12)
        fig.tight_layout(rect=(0,.07,1,.86))
        name='network_example.png' if noise==0 else 'network_example_noisy.png'
        fig.savefig(OUT/name,dpi=180); plt.close(fig)
    dump(OUT/'network_example_details.json',details)




def main():
    with threadpool_limits(limits=1):
        rows=verify_and_load()
        summary,paired=aggregate(rows)
        plots(rows,summary)
        network_pictures()
    for r in summary:
        print(f"noise={r['noise_sd']:g} beta={r['beta']:g} {r['method']}: "
              f"AP={r['average_precision_mean']:.4f} +/- {r['average_precision_sd']:.4f}; RMSE={r['test_rmse_mean']:.6f}")
    print(OUT)


if __name__=='__main__':
    main()
