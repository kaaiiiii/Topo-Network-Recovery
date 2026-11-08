"""Fixed-penalty pilot comparing author ISTA training with the saved Adam fit."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from network_recovery import ROOT, dump, edge_scores, write_csv
from neural_granger import BatchedCMLP, load_case, predict, seed_for
from sis_recovery import transitions
from make_report import independent_ap

SETTINGS=dict(seed=5420,topology='random',hidden=[10,100],lag=1,noise_sd=[0.,.02],
              learning_rates=[.05,.01,.005],max_iter=50000,check_every=100,lookback=5,
              output_ridge=.01,penalty='GL (equivalent to H at lag one)',
              lambda_source='Saved five-fold Adam pilot selection; held fixed for optimizer comparison',
              optimizer='Full-batch ISTA with group soft thresholding and best-training-objective restoration')
BASE=ROOT/'reference_results/neural_granger_ista_pilot/adam_reference'


def smooth_loss(model,x,y,ridge):
    return (model(x)-y).square().mean(dim=1).sum()+ridge*model.w2.square().sum()


def ista_update(model,x,y,lam,lr,ridge):
    model.zero_grad(set_to_none=True)
    smooth_loss(model,x,y,ridge).backward()
    with torch.no_grad():
        for parameter in model.parameters():parameter.add_(parameter.grad,alpha=-lr)
        norm=torch.linalg.vector_norm(model.w1,dim=2,keepdim=True)
        model.w1.copy_(model.w1/norm.clamp(min=lr*lam)*(norm-lr*lam).clamp(min=0.))


def train_ista(model,x,y,lam,lr=.05,ridge=.01,max_iter=50000,check_every=100,
               lookback=5,capture=False,verbose=True):
    """Author train_model_ista update and stopping rule, with batched node arithmetic."""
    best_loss=float('inf');best_step=0;best=None;trace=[];graph=None
    started=time.perf_counter()
    for step in range(1,max_iter+1):
        if capture and step==4:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):ista_update(model,x,y,lam,lr,ridge)
        if graph is None:ista_update(model,x,y,lam,lr,ridge)
        else:graph.replay()
        if step%check_every==0:
            with torch.no_grad():
                mse=(model(x)-y).square().mean()
                value=(smooth_loss(model,x,y,ridge)+lam*model.scores().sum())/model.w1.shape[1]
                loss=float(value);usage=float((model.scores()>0).float().mean())
            if not np.isfinite(loss):
                return dict(status='diverged',stopped_step=step,learning_rate=lr,
                            seconds=time.perf_counter()-started),trace
            trace.append(dict(step=step,objective=loss,scaled_mse=float(mse),nonzero_fraction=usage))
            if loss<best_loss:
                best_loss=loss;best_step=step
                best={k:v.detach().clone() for k,v in model.named_parameters()}
            elif step-best_step==lookback*check_every:break
            if verbose and (step==check_every or step%5000==0):
                print(f'  step={step}, objective={loss:.7f}, nonzero={usage:.3f}, seconds={time.perf_counter()-started:.1f}',flush=True)
    with torch.no_grad():
        for key,parameter in model.named_parameters():parameter.copy_(best[key])
    # Gradient-mapping norm measures the change made by one further ISTA update.
    ista_update(model,x,y,lam,lr,ridge)
    with torch.no_grad():
        residual=float(torch.sqrt(sum((v-best[k]).square().sum() for k,v in model.named_parameters()))/lr)
    with torch.no_grad():
        for key,parameter in model.named_parameters():parameter.copy_(best[key])
    return dict(status='completed',learning_rate=lr,best_step=best_step,stopped_step=step,best_objective=best_loss,
                stop_reason='patience' if step<max_iter else 'iteration_limit',
                gradient_mapping_norm=residual,seconds=time.perf_counter()-started),trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden',type=int,choices=[10,100],required=True)
    parser.add_argument('--noise',type=float,choices=[0.,.02],required=True)
    parser.add_argument('--lr',type=float,choices=SETTINGS['learning_rates'],default=.05)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'results/neural_granger_ista_pilot';out.mkdir(exist_ok=True)
    data,inputs=load_case(5420,'random',True)
    protocol=dict(settings=SETTINGS,torch_version=torch.__version__,device=torch.cuda.get_device_name(0),
        source_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                       for name in ['neural_granger_ista_pilot.py','neural_granger.py','sis_recovery.py']},
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        adam_reference_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((BASE/'seed_5420').glob('*.json'))},
        upstream=json.loads((ROOT/'third_party/neural_gc/source.json').read_text()))
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:dump(protocol_path,protocol)
    name=f'hidden{args.hidden}_noise{args.noise:g}_lr{args.lr:g}'
    if (out/f'{name}.json').exists():print('Already saved:',name);return
    baseline_path=BASE/f'seed_5420/random_noise{args.noise:g}.json'
    baseline=json.loads(baseline_path.read_text())
    lam=baseline['metrics']['alpha']
    x,y,_=transitions(data['clean']+args.noise*data['measurement_draw'])
    sx,sy=StandardScaler().fit(x),StandardScaler().fit(y)
    model=BatchedCMLP(100,args.hidden,[seed_for(5420,5)],device='cuda')
    tx,ty=[torch.tensor(v[None],dtype=torch.float32,device='cuda') for v in [sx.transform(x),sy.transform(y)]]
    print(f'Start {name}, lambda={lam}, lr={args.lr:g}, max_iter=50000',flush=True)
    with threadpool_limits(limits=1):
        info,trace=train_ista(model,tx,ty,lam,lr=args.lr,capture=True)
        if info['status']=='diverged':
            dump(out/f'{name}.json',dict(metrics=dict(method='ista',hidden=args.hidden,
                 noise_sd=args.noise,alpha=lam,**info),training_history=trace))
            print(info,flush=True)
            return
        arrays={key:value.detach().cpu().numpy()[0] for key,value in model.named_parameters()}
        arrays.update(x_mean=sx.mean_,x_scale=sx.scale_,y_mean=sy.mean_,y_scale=sy.scale_)
        scores=np.linalg.norm(arrays['w1'],axis=1)
        ap,precision,recall=edge_scores(data['truth'],scores)
        ex,ey,_=transitions(data['test_clean']+args.noise*data['test_measurement_draw'])
        predicted=predict(arrays,ex)
        rmse=float(np.sqrt(np.mean((predicted-ey)**2)))
        model=model.cpu().double()
        with torch.no_grad():
            direct=model(torch.tensor(sx.transform(ex)[None],dtype=torch.float64)).numpy()[0]*sy.scale_+sy.mean_
        np.testing.assert_allclose(direct,predicted,rtol=0,atol=1e-10)
        mask=~np.eye(100,dtype=bool)
        np.testing.assert_allclose(independent_ap(data['truth'][mask],scores[mask]),ap,atol=1e-12)
    row=dict(method='ista',hidden=args.hidden,noise_sd=args.noise,alpha=lam,
             average_precision=ap,test_rmse=rmse,nonzero_edges=int((scores[mask]>0).sum()),**info)
    np.savez_compressed(out/f'{name}.npz',**arrays,scores=scores,precision=precision,recall=recall)
    dump(out/f'{name}.json',dict(metrics=row,training_history=trace,adam_reference=baseline['metrics'],
         baseline_record_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
         verification=dict(independent_AP=True,independent_torch_prediction=True,
                           max_prediction_difference=float(abs(predicted-direct).max()))))
    write_csv(out/f'{name}_trace.csv',trace)
    print(row,flush=True)


if __name__=='__main__':main()
