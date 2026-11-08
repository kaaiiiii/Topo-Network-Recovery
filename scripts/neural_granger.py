"""Component-wise MLP with input group Lasso, following Tank et al. (2022).

The batched network has the same one-hidden-layer architecture and regularized
Adam objective as iancovert/Neural-GC; the author implementation is vendored
under third_party/neural_gc for numerical equivalence checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from network_recovery import ROOT, dump, edge_scores
from sis_recovery import transitions, episode_folds
from community_recovery import edge_strata, write_mixed_csv

CONFIG = dict(hidden=10, lag=1, activation='relu',
              lambdas=[0., .001, .003, .01, .03, .1, .3],
              epochs=[1500, 4000, 10000], learning_rate=.001, output_ridge=.01,
              batch_size=256,
              optimizer='Minibatch Adam on per-node MSE + input group Lasso + output-weight ridge',
              epoch_unit='Optimizer steps; 256 within-trajectory transitions sampled with replacement per step',
              scaling='Separate predictor and target StandardScaler, fitted within training folds',
              selection='Mean raw validation MSE across five whole-trajectory folds; jointly select lambda and epoch',
              edge_score='L2 norm of each input group, as author GC(threshold=False); diagonal excluded',
              beta=3., noise_sd=[0., .02], topologies=['random','community'],
              seeds=list(range(1001,1011)), training_seed_stream=8100,
              dtype='float32', deterministic=True, cuda_graph_refit=True)


class BatchedCMLP(nn.Module):
    """Independent models indexed by batch, with one MLP per output node."""
    def __init__(self, nodes, hidden, seeds, device='cpu', dtype=torch.float32):
        super().__init__()
        values = [[] for _ in range(4)]
        for seed in seeds:
            generator = torch.Generator().manual_seed(int(seed))
            for dest,shape,bound in zip(values,
                    [(nodes,hidden,nodes),(nodes,hidden),(nodes,hidden),(nodes,)],
                    [nodes**-.5,nodes**-.5,hidden**-.5,hidden**-.5]):
                dest.append(torch.empty(shape,dtype=dtype).uniform_(-bound,bound,generator=generator))
        self.w1,self.b1,self.w2,self.b2 = [nn.Parameter(torch.stack(v).to(device)) for v in values]

    def forward(self,x):
        batch,nodes,hidden,_ = self.w1.shape
        z = torch.bmm(x,self.w1.reshape(batch,nodes*hidden,nodes).transpose(1,2))
        h = torch.relu(z.reshape(batch,-1,nodes,hidden)+self.b1[:,None])
        return (h*self.w2[:,None]).sum(-1)+self.b2[:,None]

    def scores(self):
        return torch.linalg.vector_norm(self.w1,dim=2)


def objective(model,x,y,lambdas):
    # Sum of independent per-node objectives, exactly as the author Adam code.
    mse = (model(x)-y).square().mean(dim=1)
    ridge = CONFIG['output_ridge']*model.w2.square().sum(dim=-1)
    group = lambdas[:,None]*model.scores().sum(dim=-1)
    return (mse+ridge+group).sum(), (mse+ridge+group).mean(dim=1)


def seed_for(seed,fold):
    return int(np.random.SeedSequence([seed,CONFIG['training_seed_stream'],fold]).generate_state(1)[0])


def train_batch(xs,ys,seeds,lambdas,checkpoints,device,callback,cuda_graph=None):
    model = BatchedCMLP(xs.shape[-1],CONFIG['hidden'],seeds,device)
    x,y = [torch.as_tensor(a,dtype=torch.float32,device=device) for a in [xs,ys]]
    penalties = torch.tensor(lambdas,dtype=torch.float32,device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],capturable=device=='cuda')
    indices=torch.tensor(np.random.default_rng(seeds[0]).integers(0,len(xs[0]),
                         size=(max(checkpoints),CONFIG['batch_size'])),device=device)
    history=[]
    started=time.perf_counter()
    use_graph=(device=='cuda' and len(seeds)==1) if cuda_graph is None else cuda_graph
    static_index=indices[0].clone()
    graph=None
    def update():
        optimizer.zero_grad(set_to_none=True)
        loss,_=objective(model,x[:,static_index],y[:,static_index],penalties)
        loss.backward()
        optimizer.step()
    for epoch in range(1,max(checkpoints)+1):
        static_index.copy_(indices[epoch-1])
        if use_graph and epoch==4:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):update()
        if graph is not None:graph.replay()
        else:update()
        if epoch%100==0 or epoch in checkpoints:
            with torch.no_grad(): _,per_model=objective(model,x,y,penalties)
            values=per_model.cpu().numpy()
            if not np.isfinite(values).all(): raise RuntimeError('Nonfinite cMLP objective')
            history.append(dict(epoch=epoch,objective=values.tolist()))
        if epoch in checkpoints:
            callback(epoch,model)
            print(f'  epoch={epoch}, models={len(seeds)}, seconds={time.perf_counter()-started:.1f}',flush=True)
    return model,history


def fit_model(observed,seed,device):
    """Only observed training trajectories and initialization seed enter fitting."""
    started=time.perf_counter()
    train_x,train_y,valid_x,valid_y,scalers=[],[],[],[],[]
    seeds,penalties=[],[]
    folds=episode_folds(np.arange(len(observed)))
    for fold,(train,valid) in enumerate(folds):
        x,y,_=transitions(observed[train]);vx,vy,_=transitions(observed[valid])
        sx,sy=StandardScaler().fit(x),StandardScaler().fit(y)
        scalers.append((sx,sy))
        for penalty in CONFIG['lambdas']:
            train_x.append(sx.transform(x));train_y.append(sy.transform(y))
            valid_x.append(sx.transform(vx));valid_y.append(vy)
            seeds.append(seed_for(seed,fold));penalties.append(penalty)
    vx=torch.tensor(np.array(valid_x),dtype=torch.float32,device=device)
    vy=np.array(valid_y)
    scales=np.array([sy.scale_ for _,sy in scalers for _ in CONFIG['lambdas']])
    means=np.array([sy.mean_ for _,sy in scalers for _ in CONFIG['lambdas']])
    trials=[]
    def evaluate(epoch,model):
        with torch.no_grad(): predicted=model(vx).cpu().numpy()*scales[:,None]+means[:,None]
        errors=((predicted-vy)**2).mean(axis=(1,2)).reshape(5,-1)
        for i,penalty in enumerate(CONFIG['lambdas']):
            trials.append(dict(epoch=epoch,alpha=penalty,cv_mse=float(errors[:,i].mean()),
                               fold_rmse=np.sqrt(errors[:,i]).tolist()))
    _,history=train_batch(np.array(train_x),np.array(train_y),seeds,penalties,CONFIG['epochs'],device,evaluate)
    selected=min(trials,key=lambda r:r['cv_mse'])
    x,y,_=transitions(observed)
    sx,sy=StandardScaler().fit(x),StandardScaler().fit(y)
    model,final_history=train_batch(sx.transform(x)[None],sy.transform(y)[None],
         [seed_for(seed,5)],[selected['alpha']],[selected['epoch']],device,lambda epoch,model:None)
    info=dict(alpha=selected['alpha'],epoch=selected['epoch'],
              cv_rmse=float(np.sqrt(selected['cv_mse'])),cv_fold_sd=float(np.std(selected['fold_rmse'],ddof=1)),
              seconds=time.perf_counter()-started,initialization_seed=seed_for(seed,5))
    arrays={name:param.detach().cpu().numpy()[0] for name,param in model.named_parameters()}
    arrays.update(x_mean=sx.mean_,x_scale=sx.scale_,y_mean=sy.mean_,y_scale=sy.scale_)
    return arrays,info,trials,dict(cv=history,final=final_history)


def predict(arrays,x):
    """Independent NumPy prediction used for saved-model evaluation."""
    xs=(x-arrays['x_mean'])/arrays['x_scale']
    hidden=np.maximum(np.einsum('nj,ihj->nih',xs,arrays['w1'])+arrays['b1'],0.)
    scaled=np.einsum('nih,ih->ni',hidden,arrays['w2'])+arrays['b2']
    return scaled*arrays['y_scale']+arrays['y_mean']


def load_case(seed,topology,pilot):
    if topology=='community':
        folder=ROOT/'results'/('community_pilot' if pilot else 'community')/f'seed_{seed}'
        data=dict(np.load(folder/'system.npz'))
        inputs=[folder/'system.npz']
    else:
        folder=ROOT/'results'/('sis_pilot' if pilot else 'sis')/f'seed_{seed}'
        data=dict(np.load(folder/'system.npz'))
        trajectory=folder/'trajectories_beta3.npz'
        data.update(dict(np.load(trajectory)))
        inputs=[folder/'system.npz',trajectory]
    return data,inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--case',choices=['all','random_clean','random_noisy','community_clean','community_noisy'],default='all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    out=ROOT/'results'/('neural_granger_pilot' if args.pilot else 'neural_granger')
    out.mkdir(parents=True,exist_ok=True)
    seeds=[5420] if args.pilot else CONFIG['seeds']
    inputs=[]
    for seed in seeds:
        for topology in CONFIG['topologies']: inputs.extend(load_case(seed,topology,args.pilot)[1])
    scripts=['neural_granger.py','sis_recovery.py','network_recovery.py','community_recovery.py']
    protocol=dict(config=CONFIG,pilot=args.pilot,torch_version=torch.__version__,device=args.device,
        device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
        upstream=json.loads((ROOT/'third_party/neural_gc/source.json').read_text()),
        script_sha256={s:hashlib.sha256((ROOT/'scripts'/s).read_bytes()).hexdigest() for s in scripts},
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise RuntimeError('Saved protocol differs')
    dump(path,protocol)
    rows=[]
    with threadpool_limits(limits=1):
        for seed in seeds:
            for topology in CONFIG['topologies']:
                data,_=load_case(seed,topology,args.pilot)
                for noise in CONFIG['noise_sd']:
                    case=topology+('_clean' if noise==0 else '_noisy')
                    if args.case!='all' and args.case!=case: continue
                    folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
                    name=f'{topology}_noise{noise:g}'
                    record=folder/f'{name}.json'
                    if record.exists():
                        rows.append(json.loads(record.read_text())['metrics']);continue
                    print(f'Start seed={seed} topology={topology} noise={noise:g}',flush=True)
                    arrays,info,trials,history=fit_model(data['clean']+noise*data['measurement_draw'],seed,args.device)
                    scores=np.linalg.norm(arrays['w1'],axis=1)
                    ap,precision,recall=edge_scores(data['truth'],scores)
                    tx,ty,_=transitions(data['test_clean']+noise*data['test_measurement_draw'])
                    rmse=float(np.sqrt(np.mean((predict(arrays,tx)-ty)**2)))
                    row=dict(seed=seed,topology=topology,beta=3.,noise_sd=noise,method='cmlp',
                        average_precision=ap,test_rmse=rmse,**info)
                    if topology=='community':row.update(edge_strata(data['truth'],scores,data['labels']))
                    np.savez_compressed(folder/f'{name}.npz',**arrays,scores=scores,precision=precision,recall=recall)
                    dump(record,dict(metrics=row,trials=trials,training_history=history))
                    rows.append(row)
                    print(f'Finished AP={ap:.4f}, test RMSE={rmse:.6f}, alpha={info["alpha"]:g}, epoch={info["epoch"]}',flush=True)
                    if args.device=='cuda':torch.cuda.empty_cache()
    write_mixed_csv(out/('metrics.csv' if args.case=='all' else f'metrics_{args.case}.csv'),rows)


if __name__=='__main__':
    main()
