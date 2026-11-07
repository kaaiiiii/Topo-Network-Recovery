"""Known-dynamics SIS network identification by nonnegative sparse regression.

Uses saved observations from the SIS experiment. This adapts model-based network
identification to our weighted directed Euler SIS map.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, LassoLars
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
from network_recovery import ROOT, dump, write_csv, edge_scores
from sis_recovery import CONFIG as SIS_CONFIG, transitions, episode_folds, sis_step

CONFIG = dict(fractions=[0.]+np.logspace(-8,0,14).tolist(), kkt_tolerance=1e-7,
              lars_max_iter=5000, polish_max_iter=100000, polish_tol=1e-12,
              known_rates=True, center=False, intercept=False, smoothing=False,
              scaling='Training-fold RMS of each feature and target; no centering',
              penalty='Common fraction of each node/fold alpha_max; 15 candidates including NNLS')
SOURCES = ['https://www.ieeespmb.org/2016/papers/l04_03.pdf',
           'https://arxiv.org/abs/1811.06741']


def node_design(x, y, node, beta, delta=1., dt=.05):
    sources = np.arange(x.shape[1]) != node
    design = dt*beta*(1-x[:,node,None])*x[:,sources]
    target = y[:,node]-(1-dt*delta)*x[:,node]
    return design, target, sources


def positive_kkt(gram, xy, coef, alpha, samples):
    gradient = (gram@coef-xy)/samples+alpha
    # At a positive coefficient the gradient is zero; at the boundary it is nonnegative.
    residual = np.where(coef>1e-10,abs(gradient),np.maximum(-gradient,0.))
    return float(max(residual.max(),max(0.,-coef.min())))


def positive_path(design, target, fractions):
    sx = np.sqrt(np.mean(design**2,axis=0))
    sy = float(np.sqrt(np.mean(target**2)))
    sx = np.where(sx>1e-14,sx,1.)
    sy = sy if sy>1e-14 else 1.
    x, y = design/sx, target/sy
    gram, xy = x.T@x, x.T@y
    alpha_max = float(max(0.,xy.max()/len(x)))
    warnings_count, polished = 0, 0
    candidates = {}
    if alpha_max>0 and any(f>0 for f in fractions):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            path = LassoLars(alpha=min(f for f in fractions if f>0)*alpha_max,
                             positive=True,fit_intercept=False,precompute=gram,
                             max_iter=CONFIG['lars_max_iter']).fit(x,y)
        warnings_count = len(caught)
        assert path.n_iter_ < CONFIG['lars_max_iter']
    max_kkt = 0.
    for fraction in fractions:
        alpha = fraction*alpha_max
        if alpha_max==0:
            coef = np.zeros(design.shape[1])
        elif fraction==0:
            coef = nnls(x,y,maxiter=5000)[0]
        else:
            coef = np.array([np.interp(alpha,path.alphas_[::-1],c[::-1]) for c in path.coef_path_])
        residual = positive_kkt(gram,xy,coef,alpha,len(x))
        if residual>CONFIG['kkt_tolerance'] and fraction>0:
            # sklearn documents a limitation of positive LARS near the end of its path.
            # Polish that candidate with standard coordinate descent, keeping the same objective.
            model = Lasso(alpha=alpha,fit_intercept=False,positive=True,warm_start=True,
                          precompute=gram,max_iter=CONFIG['polish_max_iter'],tol=CONFIG['polish_tol'])
            model.coef_ = coef.copy()
            for attempt in range(3):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(x,y)
                warnings_count += len(caught)
                coef = model.coef_.copy()
                residual = positive_kkt(gram,xy,coef,alpha,len(x))
                if residual<=CONFIG['kkt_tolerance']:
                    break
            polished += 1
        if residual>CONFIG['kkt_tolerance']:
            raise RuntimeError(f'Nonnegative Lasso failed KKT check: fraction={fraction:g}, residual={residual:g}')
        assert coef.min() >= -1e-12
        candidates[fraction] = np.maximum(coef,0.)*sy/sx
        max_kkt = max(max_kkt,residual)
    return candidates, dict(max_kkt_residual=max_kkt,solver_warnings=warnings_count,
                            polished_candidates=polished,alpha_max=alpha_max)


def fit_path(x,y,beta,fractions,delta=1.,dt=.05):
    nodes = x.shape[1]
    weights = {f:np.zeros((nodes,nodes)) for f in fractions}
    info = dict(max_kkt_residual=0.,solver_warnings=0,polished_candidates=0)
    for node in range(nodes):
        design,target,sources = node_design(x,y,node,beta,delta,dt)
        path,diagnostics = positive_path(design,target,fractions)
        for fraction,coef in path.items():
            weights[fraction][node,sources] = coef
        info['max_kkt_residual'] = max(info['max_kkt_residual'],diagnostics['max_kkt_residual'])
        info['solver_warnings'] += diagnostics['solver_warnings']
        info['polished_candidates'] += diagnostics['polished_candidates']
    return weights,info


def fit_model(observed,beta,delta=1.,dt=.05):
    """Graph truth, clean latent states, and test observations are not inputs."""
    started = time.perf_counter()
    x,y,groups = transitions(observed)
    errors = {f:[] for f in CONFIG['fractions']}
    diagnostics = []
    for train,valid in episode_folds(groups):
        path,info = fit_path(x[train],y[train],beta,CONFIG['fractions'],delta,dt)
        diagnostics.append(info)
        for fraction,weights in path.items():
            pred = sis_step(x[valid],weights,beta,delta,dt)
            errors[fraction].append(float(np.mean((pred-y[valid])**2)))
    trials = [dict(fraction=f,cv_mse=float(np.mean(e)),fold_rmse=np.sqrt(e).tolist())
              for f,e in errors.items()]
    selected = min(trials,key=lambda r:r['cv_mse'])
    path,info = fit_path(x,y,beta,[selected['fraction']],delta,dt)
    diagnostics.append(info)
    info = dict(fraction=selected['fraction'],cv_rmse=float(np.sqrt(selected['cv_mse'])),
                cv_fold_sd=float(np.std(selected['fold_rmse'],ddof=1)),
                max_kkt_residual=max(d['max_kkt_residual'] for d in diagnostics),
                solver_warnings=sum(d['solver_warnings'] for d in diagnostics),
                polished_candidates=sum(d['polished_candidates'] for d in diagnostics),
                seconds=time.perf_counter()-started)
    return path[selected['fraction']],info,trials


def run_seed(seed,source,out):
    folder = out/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    data = source/f'seed_{seed}'
    system = np.load(data/'system.npz')
    rows = []
    with threadpool_limits(limits=1):
        for beta in SIS_CONFIG['beta']:
            trajectories = np.load(data/f'trajectories_beta{beta:g}.npz')
            for noise in SIS_CONFIG['noise_sd']:
                name = f'sis_aware_beta{beta:g}_noise{noise:g}'
                saved = folder/f'{name}.json'
                if saved.exists():
                    row = json.loads(saved.read_text())['metrics']
                else:
                    observed = trajectories['clean']+noise*system['measurement_draw']
                    bhat,info,trials = fit_model(observed,beta,SIS_CONFIG['delta'],SIS_CONFIG['dt'])
                    # Ground truth and held-out test data are used only after fitting.
                    ap,precision,recall = edge_scores(system['truth'],bhat)
                    tx,ty,_ = transitions(trajectories['test_clean']+noise*system['test_measurement_draw'])
                    pred = sis_step(tx,bhat,beta,SIS_CONFIG['delta'],SIS_CONFIG['dt'])
                    rmse = float(np.sqrt(np.mean((pred-ty)**2)))
                    persistence = float(np.sqrt(np.mean((tx-ty)**2)))
                    row = dict(seed=seed,beta=beta,noise_sd=noise,method='sis_aware',average_precision=ap,
                               test_rmse=rmse,persistence_rmse=persistence,rmse_ratio=rmse/persistence,
                               weight_relative_error=float(np.linalg.norm(bhat-system['b'])/np.linalg.norm(system['b'])),
                               **info)
                    np.savez_compressed(folder/f'{name}.npz',b_hat=bhat,precision=precision,recall=recall)
                    dump(saved,dict(metrics=row,trials=trials))
                rows.append(row)
                print(f"seed={seed} beta={beta:g} noise={noise:g}: AP={row['average_precision']:.4f}, "
                      f"RMSE/persistence={row['rmse_ratio']:.3f}, fraction={row['fraction']:.2g}, {row['seconds']:.1f}s",flush=True)
    write_csv(folder/'metrics.csv',rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    source = ROOT/'results'/('sis_pilot' if args.pilot else 'sis')
    out = ROOT/'results'/('sis_aware_pilot' if args.pilot else 'sis_aware')
    out.mkdir(parents=True,exist_ok=True)
    seeds = [5420] if args.pilot else SIS_CONFIG['seeds']
    inputs = [source/'protocol.json',source/'metrics.csv']
    for seed in seeds:
        inputs += [source/f'seed_{seed}'/'system.npz']
        inputs += [source/f'seed_{seed}'/f'trajectories_beta{b:g}.npz' for b in SIS_CONFIG['beta']]
    protocol = dict(config=CONFIG,sis_config=SIS_CONFIG,pilot=args.pilot,sources=SOURCES,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    sis_script_sha256=hashlib.sha256(Path(__file__).with_name('sis_recovery.py').read_bytes()).hexdigest(),
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    method='Known-rate SIS features, nonnegative L1 regression, zero diagonal and intercept',
                    literature_scope='Model-based identification for weighted directed SIS dynamics')
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise RuntimeError('Frozen protocol differs; preserve old outputs before revising')
    dump(protocol_path,protocol)
    with parallel_backend('loky',inner_max_num_threads=1):
        batches = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s,source,out) for s in seeds)
    write_csv(out/'metrics.csv',[r for batch in batches for r in batch])


if __name__=='__main__':
    main()
