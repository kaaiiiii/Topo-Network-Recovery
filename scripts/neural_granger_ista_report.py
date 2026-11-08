"""Verify and plot the independent ISTA optimizer/width pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from network_recovery import ROOT,dump,write_csv
from community_recovery import write_mixed_csv
from make_report import independent_ap
from neural_granger import load_case,predict
from neural_granger_ista_pilot import BASE
from sis_recovery import transitions


def main():
    out=ROOT/'results/neural_granger_ista_pilot'
    protocol=json.loads((out/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
    for field in ['input_sha256','adam_reference_sha256']:
        for name,digest in protocol[field].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    data,_=load_case(5420,'random',True)
    attempts=[];records={};checks=0
    for path in sorted(out.glob('hidden*.json')):
        record=json.loads(path.read_text());row=record['metrics'];attempts.append(row)
        if row['status']!='completed':continue
        arrays=dict(np.load(path.with_suffix('.npz')))
        noise=row['noise_sd'];x,y,_=transitions(data['clean']+noise*data['measurement_draw'])
        sx,sy=StandardScaler().fit(x),StandardScaler().fit(y)
        for name,value in [('x_mean',sx.mean_),('x_scale',sx.scale_),('y_mean',sy.mean_),('y_scale',sy.scale_)]:
            np.testing.assert_array_equal(arrays[name],value)
        mask=~np.eye(100,dtype=bool)
        score=np.linalg.norm(arrays['w1'],axis=1)
        np.testing.assert_array_equal(score,arrays['scores'])
        np.testing.assert_allclose(independent_ap(data['truth'][mask],score[mask]),row['average_precision'],atol=1e-12)
        tx,ty,_=transitions(data['test_clean']+noise*data['test_measurement_draw'])
        np.testing.assert_allclose(np.sqrt(np.mean((predict(arrays,tx)-ty)**2)),row['test_rmse'],atol=1e-12)
        history=record['training_history'];best=min(history,key=lambda r:r['objective'])
        assert best['step']==row['best_step']
        assert np.isfinite([r['objective'] for r in history]).all()
        checks+=1;records[(row['hidden'],noise,row['learning_rate'])]=record
    write_mixed_csv(out/'attempts.csv',attempts)
    rows=[];selected={}
    for noise in [0.,.02]:
        reference=json.loads((BASE/f'seed_5420/random_noise{noise:g}.json').read_text())
        rows.append(dict(method='Adam H=10',noise_sd=noise,
                         average_precision=reference['metrics']['average_precision'],test_rmse=reference['metrics']['test_rmse']))
        for hidden in [10,100]:
            # Choose the first stable learning rate, independent of graph/test scores.
            candidates=[records[(hidden,noise,lr)] for lr in [.05,.01,.005] if (hidden,noise,lr) in records]
            assert candidates,f'No stable fit for hidden={hidden}, noise={noise}'
            chosen=candidates[0];selected[(hidden,noise)]=chosen
            rows.append(dict(method=f'ISTA H={hidden}',noise_sd=noise,
                 **{k:chosen['metrics'][k] for k in ['average_precision','test_rmse','learning_rate','best_step','stop_reason','nonzero_edges']}))
        for method,base,label in [('lasso','sis_pilot','Raw Lasso'),('local_poly','sis_local_poly_pilot','Local polynomial +\nmodel-based Lasso')]:
            record=json.loads((ROOT/f'results/{base}/seed_5420/{method}_beta3_noise{noise:g}.json').read_text())
            rows.append(dict(method=label,noise_sd=noise,average_precision=record['metrics']['average_precision'],test_rmse=record['metrics']['test_rmse']))
    write_mixed_csv(out/'comparison.csv',rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['Raw Lasso','Adam H=10','ISTA H=10','ISTA H=100','Local polynomial +\nmodel-based Lasso']
    colors=['#bd783c','#377bb5','#138777','#52b5a5','#754ba1']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,noise in enumerate([0.,.02]):
        for rowidx,metric in enumerate(['average_precision','test_rmse']):
            values=[next(r[metric] for r in rows if r['noise_sd']==noise and r['method']==label) for label in labels]
            ax=axes[rowidx,col];ax.bar(range(5),values,color=colors,width=.65)
            upper=max(values)
            for i,value in enumerate(values):ax.text(i,value+upper*.035,f'{value:.3f}' if rowidx==0 else f'{value:.5f}',ha='center',fontsize=9)
            ax.set(xticks=range(5),xticklabels=[label.replace(' H=','\nH=') for label in labels],title=f'Noise SD = {noise:g}',
                   ylabel='Edge-recovery AP' if rowidx==0 else 'Held-out one-step RMSE',ylim=(0,upper*1.2))
    fig.suptitle('Independent random-network pilot | seed 5420, SIS beta = 3')
    fig.savefig(out/'optimizer_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,noise in zip(axes,[0.,.02]):
        old=json.loads((BASE/f'seed_5420/random_noise{noise:g}.json').read_text())
        history=old['training_history']['final']
        ax.plot([r['epoch'] for r in history],[r['objective'][0] for r in history],label='Adam H=10, batch=256',color=colors[1])
        for hidden,color in [(10,colors[2]),(100,colors[3])]:
            record=selected[(hidden,noise)];history=record['training_history']
            ax.plot([r['step'] for r in history],[r['objective'] for r in history],
                    label=f"ISTA H={hidden}, full batch, lr={record['metrics']['learning_rate']:g}",color=color)
        ax.set(xscale='log',yscale='log',xlabel='Optimizer updates',ylabel='Mean regularized training objective',title=f'Noise SD = {noise:g}')
        ax.legend(fontsize=8)
    fig.suptitle('Training objective | same input/target scaling and fixed group penalty')
    fig.savefig(out/'training_comparison.png',dpi=180);plt.close(fig)
    dump(out/'verification.json',dict(completed_fits_checked=checks,scaler_arrays_checked=4*checks,
         independent_AP_checks=checks,independent_RMSE_checks=checks,best_checkpoint_checks=checks,
         stable_rate_rule='First completed rate in [0.05, 0.01, 0.005]',
         report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for row in rows:print(row)


if __name__=='__main__':main()
