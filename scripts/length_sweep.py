"""How much data does recovery take? A dense training-length sweep.

Same graphs, dynamics, noise, CV and scoring as network_recovery.py, at more
training lengths. Trajectories are simulated once at the longest length; shorter
lengths are prefixes, so lengths 100, 300 and 1000 reproduce the original runs.

For each network, T90 is the training length at which AP first reaches 0.9,
interpolated linearly in log(T) between neighbouring lengths. Intervals bootstrap
the 10 networks (10,000 resamples, seed 5420).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_recovery import CONFIG, dump, edge_scores, fit_model, graph_system, simulate, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETUP = dict(lengths=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000],
             noise_ratios=CONFIG['noise_ratios'], methods=['ridge', 'lasso'], seeds=CONFIG['seeds'], target_ap=.9)


def run_seed(seed):
    rows = []
    with threadpool_limits(limits=1):
        a, truth, covariance = graph_system(seed, CONFIG['nodes'], CONFIG['edges'])
        clean = simulate(a, covariance, max(SETUP['lengths']), np.random.SeedSequence([seed, 20]))
        test_clean = simulate(a, covariance, CONFIG['test_steps'], np.random.SeedSequence([seed, 30]))
        sigma = np.sqrt(np.diag(covariance))
        noise = np.random.default_rng(np.random.SeedSequence([seed, 40])).normal(size=clean.shape) * sigma
        test_noise = np.random.default_rng(np.random.SeedSequence([seed, 50])).normal(size=test_clean.shape) * sigma
        mask = ~np.eye(len(a), dtype=bool)
        for level in SETUP['noise_ratios']:
            test = test_clean + level * test_noise
            for length in SETUP['lengths']:
                observed = (clean + level * noise)[:length + 1]
                for method in SETUP['methods']:
                    a_hat, intercept, info = fit_model(observed, method)
                    ap, _, _ = edge_scores(truth, a_hat)
                    prediction = test[:-1] @ a_hat.T + intercept
                    rows.append(dict(seed=seed, noise=level, length=length, method=method, average_precision=ap,
                                     test_rmse=float(np.sqrt(mean_squared_error(test[1:], prediction))),
                                     edge_weight_relative_error=float(np.linalg.norm((a_hat - a)[mask]) / np.linalg.norm(a[mask])),
                                     alpha=info['alpha'], cv_rmse=info['cv_rmse'],
                                     convergence_warnings=info['convergence_warnings']))
        print(f'seed {seed} done', flush=True)
    return rows


def t_target(lengths, ap, target):
    """First length where AP reaches target, log-linear interpolation; inf if never reached."""
    lengths, ap = np.asarray(lengths, float), np.asarray(ap, float)
    above = np.nonzero(ap >= target)[0]
    if len(above) == 0:
        return float('inf')
    i = above[0]
    if i == 0:
        return float(lengths[0])
    x0, x1, y0, y1 = np.log(lengths[i - 1]), np.log(lengths[i]), ap[i - 1], ap[i]
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))


def summarize(rows):
    draws = np.random.default_rng(5420).integers(0, len(SETUP['seeds']), size=(10000, len(SETUP['seeds'])))
    def ci(values):
        values = np.asarray(values, float)
        boot = values[draws].mean(axis=1)
        return dict(mean=float(values.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)))
    out = dict(curves=[], t90=[], alphas_at_grid_edge=[])
    for level in SETUP['noise_ratios']:
        for method in SETUP['methods']:
            per_seed = {s: sorted((r for r in rows if r['seed'] == s and r['noise'] == level and r['method'] == method),
                                  key=lambda r: r['length']) for s in SETUP['seeds']}
            for k, length in enumerate(SETUP['lengths']):
                vals = [per_seed[s][k] for s in SETUP['seeds']]
                out['curves'].append(dict(noise=level, method=method, length=length,
                                          average_precision=ci([v['average_precision'] for v in vals]),
                                          test_rmse=ci([v['test_rmse'] for v in vals]),
                                          alphas=sorted({v['alpha'] for v in vals})))
            t90 = [t_target(SETUP['lengths'], [r['average_precision'] for r in per_seed[s]], SETUP['target_ap'])
                   for s in SETUP['seeds']]
            finite = all(np.isfinite(t90))
            entry = dict(noise=level, method=method, per_network=t90, reached_by_all=finite)
            if finite:
                entry.update(ci(np.log(t90)))   # bootstrap the geometric mean
                entry = {**entry, **{k: float(np.exp(entry[k])) for k in ('mean', 'low', 'high')}}
            out['t90'].append(entry)
            grid = CONFIG[f'{method}_alphas']
            edge = sum(1 for r in rows if r['noise'] == level and r['method'] == method and r['alpha'] in (grid[0], grid[-1]))
            out['alphas_at_grid_edge'].append(dict(noise=level, method=method, fits=edge))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    out = ROOT / 'results' / 'length_sweep'
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'protocol.json', dict(setup=SETUP, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with parallel_backend('loky', inner_max_num_threads=1):
        pieces = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s) for s in SETUP['seeds'])
    rows = [r for piece in pieces for r in piece]
    write_csv(out / 'metrics.csv', rows)
    dump(out / 'summary.json', summarize(rows))
    print(f'Complete: {len(rows)} fits saved to {out}', flush=True)


if __name__ == '__main__':
    main()
