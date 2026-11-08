"""Compare SIS network recovery on four communities with fixed edge counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
import osqp
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score

from network_recovery import ROOT, dump, write_csv, edge_scores
from sis_recovery import CONFIG as SIS, simulate_sis, transitions, sis_step, fit_model as fit_raw
from sis_aware import CONFIG as AWARE, fit_model as fit_aware
from sis_local_poly import CONFIG as LOCAL, fit_model as fit_local

CONFIG = dict(nodes=100, groups=4, group_size=25, internal_edges_per_group=60,
              bridge_edges=60, edges=300, beta=3., noise_sd=[0.,.02],
              seeds=list(range(1001,1011)), methods=['ridge','lasso','sis_aware','local_poly'],
              weights='Uniform(0.2,1), then normalize spectral radius to one',
              topology='Fixed-count planted partition: 60 directed edges inside each block and 60 across blocks',
              graph_seed_stream=910, weight_seed_stream=10,
              state_and_noise_streams='Same as original SIS: seed streams 710 and 720',
              selection_budget=300)


def community_system(seed):
    labels = np.repeat(np.arange(CONFIG['groups']), CONFIG['group_size'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, CONFIG['graph_seed_stream']]))
    truth = np.zeros((CONFIG['nodes'],CONFIG['nodes']), dtype=bool)
    for group in range(CONFIG['groups']):
        candidates = np.argwhere((labels[:,None]==group)&(labels[None,:]==group)&~np.eye(len(labels),dtype=bool))
        selected = candidates[rng.choice(len(candidates),CONFIG['internal_edges_per_group'],replace=False)]
        truth[selected[:,0],selected[:,1]] = True
    candidates = np.argwhere(labels[:,None]!=labels[None,:])
    selected = candidates[rng.choice(len(candidates),CONFIG['bridge_edges'],replace=False)]
    truth[selected[:,0],selected[:,1]] = True
    weights = np.zeros(truth.shape)
    wrng = np.random.default_rng(np.random.SeedSequence([seed, CONFIG['weight_seed_stream']]))
    for source,target in sorted((int(j),int(i)) for i,j in zip(*np.where(truth))):
        weights[target,source] = wrng.uniform(.2,1.)
    radius = float(np.max(abs(np.linalg.eigvals(weights))))
    if radius<=1e-10:
        raise ValueError('Graph has zero spectral radius')
    return weights/radius, truth, labels


def edge_strata(truth, weights, labels):
    mask = ~np.eye(len(truth),dtype=bool)
    scores = abs(weights).copy()
    scores[~mask] = -np.inf
    ranked = np.argsort(-scores.ravel(),kind='stable')[:CONFIG['selection_budget']]
    chosen = np.zeros_like(truth)
    chosen.ravel()[ranked[scores.ravel()[ranked]>0]] = True
    result = dict(selected_edges=int(chosen.sum()))
    for name, subset in [('within',mask&(labels[:,None]==labels[None,:])),
                         ('bridge',mask&(labels[:,None]!=labels[None,:]))]:
        positives = int(truth[subset].sum())
        result[name+'_ap'] = float(average_precision_score(truth[subset],scores[subset]))
        result[name+'_recall_at_300'] = float((chosen&truth&subset).sum()/positives)
        result[name+'_true_edges'] = positives
        result[name+'_candidate_edges'] = int(subset.sum())
    return result


def run_seed(seed, out):
    folder = out/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        b, truth, labels = community_system(seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed,710]))
        test_rng = np.random.default_rng(np.random.SeedSequence([seed,720]))
        initial = rng.uniform(SIS['initial_low'],SIS['initial_high'],size=(SIS['training_episodes'],len(b)))
        test_initial = test_rng.uniform(SIS['initial_low'],SIS['initial_high'],size=(SIS['test_episodes'],len(b)))
        noise = rng.normal(size=(len(initial),SIS['steps']+1,len(b)))
        test_noise = test_rng.normal(size=(len(test_initial),SIS['steps']+1,len(b)))
        clean = simulate_sis(b,initial,CONFIG['beta'],SIS['steps'],SIS['delta'],SIS['dt'])
        test_clean = simulate_sis(b,test_initial,CONFIG['beta'],SIS['steps'],SIS['delta'],SIS['dt'])
        system_path = folder/'system.npz'
        if not system_path.exists():
            np.savez_compressed(system_path,b=b,truth=truth,labels=labels,initial=initial,test_initial=test_initial,
                                measurement_draw=noise,test_measurement_draw=test_noise,clean=clean,test_clean=test_clean)
        rows = []
        for sd in CONFIG['noise_sd']:
            observed = clean+sd*noise
            tx,ty,_ = transitions(test_clean+sd*test_noise)
            persistence = float(np.sqrt(np.mean((tx-ty)**2)))
            for method in CONFIG['methods']:
                name = f'{method}_noise{sd:g}'
                record_path = folder/f'{name}.json'
                if record_path.exists():
                    row = json.loads(record_path.read_text())['metrics']
                else:
                    if method in ['ridge','lasso']:
                        weights,intercept,info,trials = fit_raw(observed,method)
                        predicted = tx@weights.T+intercept
                    else:
                        fitter = fit_aware if method=='sis_aware' else fit_local
                        weights,info,trials = fitter(observed,CONFIG['beta'],SIS['delta'],SIS['dt'])
                        intercept = np.zeros(len(b))
                        predicted = sis_step(tx,weights,CONFIG['beta'],SIS['delta'],SIS['dt'])
                    ap,precision,recall = edge_scores(truth,weights)
                    rmse = float(np.sqrt(np.mean((predicted-ty)**2)))
                    row = dict(seed=seed,topology='community',beta=CONFIG['beta'],noise_sd=sd,method=method,
                               average_precision=ap,test_rmse=rmse,persistence_rmse=persistence,
                               rmse_ratio=rmse/persistence,**edge_strata(truth,weights,labels),**info)
                    np.savez_compressed(folder/f'{name}.npz',weights=weights,intercept=intercept,
                                        precision=precision,recall=recall)
                    dump(record_path,dict(metrics=row,trials=trials))
                rows.append(row)
                print(f"seed={seed} noise={sd:g} {method}: AP={row['average_precision']:.4f}, "
                      f"within recall={row['within_recall_at_300']:.3f}, bridge recall={row['bridge_recall_at_300']:.3f}",flush=True)
    return rows


def write_mixed_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path,[{key:row.get(key,'') for key in keys} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    out = ROOT/'results'/('community_pilot' if args.pilot else 'community')
    out.mkdir(parents=True,exist_ok=True)
    names = ['community_recovery.py','sis_recovery.py','sis_aware.py','sis_local_poly.py','network_recovery.py']
    protocol = dict(config=CONFIG,sis_config=SIS,aware_config=AWARE,local_poly_config=LOCAL,pilot=args.pilot,
                    versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,osqp=osqp.__version__),
                    script_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise RuntimeError('Saved experiment protocol differs from current code')
    dump(protocol_path,protocol)
    seeds = [5420] if args.pilot else CONFIG['seeds']
    with parallel_backend('loky',inner_max_num_threads=1):
        batches = Parallel(n_jobs=args.jobs)(delayed(run_seed)(seed,out) for seed in seeds)
    write_mixed_csv(out/'metrics.csv',[r for rows in batches for r in rows])


if __name__=='__main__':
    main()
