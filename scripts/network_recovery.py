"""Topic 7: known sparse directed graph -> noisy dynamics -> nodewise lasso.

Run --pilot for one timing/sanity case, then run the frozen ten-graph experiment.
No adjacency labels enter preprocessing, regression, or penalty selection.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.linalg import solve_discrete_lyapunov
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(nodes=100, edges=300, self_memory=.3, coupling_radius=.6,
              lengths=[100, 300, 1000], noise_ratios=[0., .25, .75], test_steps=1000,
              seeds=list(range(1001, 1011)), cv_splits=5, cv_gap=1,
              lasso_alphas=np.logspace(-3, 0, 7).tolist(),
              ridge_alphas=np.logspace(-1, 4, 7).tolist(),
              lasso_max_iter=10000, lasso_tol=1e-4)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def graph_system(seed, nodes=100, edges=300):
    graph = nx.gnm_random_graph(nodes, edges, seed=seed, directed=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    w = np.zeros((nodes, nodes))
    for source, target in sorted(graph.edges):
        w[target, source] = rng.uniform(.2, 1.)
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    if radius <= 1e-10:
        raise ValueError('Acyclic/degenerate graph: cannot apply the specified spectral scaling')
    a = CONFIG['self_memory'] * np.eye(nodes) + CONFIG['coupling_radius'] * w / radius
    covariance = solve_discrete_lyapunov(a, np.eye(nodes))
    covariance = (covariance + covariance.T) / 2
    assert np.max(np.abs(np.linalg.eigvals(a))) < 1
    return a, w != 0, covariance


def simulate(a, covariance, steps, seed):
    rng = np.random.default_rng(seed)
    x = np.empty((steps + 1, len(a)))
    # Exact stationary initialization; process innovations have fixed covariance I.
    x[0] = np.linalg.cholesky(covariance) @ rng.normal(size=len(a))
    innovations = rng.normal(size=(steps, len(a)))
    for t in range(steps):
        x[t + 1] = a @ x[t] + innovations[t]
    return x


def cv_splits(n):
    # First training block is about half the available transitions; expand forward.
    splitter = TimeSeriesSplit(n_splits=CONFIG['cv_splits'], test_size=n // 10, gap=CONFIG['cv_gap'])
    folds = list(splitter.split(np.empty(n)))
    for train, valid in folds:
        # A transition uses raw timestamps t and t+1. No raw timestamp is shared.
        assert train[-1] + 1 < valid[0]
    return folds


def fit_model(observed, method):
    """Takes observations only: graph support is deliberately not an argument."""
    x, y = observed[:-1], observed[1:]
    if method == 'lasso':
        estimator = Lasso(max_iter=CONFIG['lasso_max_iter'], tol=CONFIG['lasso_tol'],
                          precompute=True, selection='cyclic')
    else:
        estimator = Ridge(solver='cholesky')
    pipe = make_pipeline(StandardScaler(), estimator)
    grid = {f'{method}__alpha': CONFIG[f'{method}_alphas']}
    search = GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=cv_splits(len(x)),
                          n_jobs=1, refit=True, error_score='raise', return_train_score=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        search.fit(x, y)
    best = search.best_estimator_
    scale = best.named_steps['standardscaler']
    model = best.named_steps[method]
    # sklearn stores [target, source]. Undo predictor scaling for edge ranking.
    a_hat = model.coef_ / scale.scale_[None, :]
    intercept = model.intercept_ - a_hat @ scale.mean_
    np.testing.assert_allclose(best.predict(x), x @ a_hat.T + intercept, atol=1e-9)
    trials = []
    for k, params in enumerate(search.cv_results_['params']):
        split_rmse = [float(np.sqrt(-search.cv_results_[f'split{i}_test_score'][k])) for i in range(CONFIG['cv_splits'])]
        trials.append(dict(alpha=params[f'{method}__alpha'], cv_mse=float(-search.cv_results_['mean_test_score'][k]),
                           fold_rmse=split_rmse))
    info = dict(alpha=float(model.alpha), cv_rmse=float(np.sqrt(-search.best_score_)),
                cv_fold_sd=float(np.std(trials[search.best_index_]['fold_rmse'], ddof=1)),
                seconds=time.perf_counter() - started,
                convergence_warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                max_selected_iterations=int(np.max(model.n_iter_)) if method == 'lasso' else 0,
                trials=trials)
    return a_hat, intercept, info


def edge_scores(truth, a_hat):
    mask = ~np.eye(len(truth), dtype=bool)
    labels = truth[mask].astype(int)
    scores = np.abs(a_hat[mask])
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(average_precision_score(labels, scores)), precision, recall


def run_seed(seed, out, pilot=False):
    folder = out / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        a, truth, covariance = graph_system(seed, CONFIG['nodes'], CONFIG['edges'])
        clean = simulate(a, covariance, max(CONFIG['lengths']), np.random.SeedSequence([seed, 20]))
        test_clean = simulate(a, covariance, CONFIG['test_steps'], np.random.SeedSequence([seed, 30]))
        sigma = np.sqrt(np.diag(covariance))
        noise = np.random.default_rng(np.random.SeedSequence([seed, 40])).normal(size=clean.shape) * sigma
        test_noise = np.random.default_rng(np.random.SeedSequence([seed, 50])).normal(size=test_clean.shape) * sigma
        # Same latent trajectory/noise draws across conditions; different test trajectory.
        np.savez_compressed(folder / 'system.npz', a=a, truth=truth, covariance=covariance,
                            clean=clean, test_clean=test_clean, measurement_draw=noise, test_measurement_draw=test_noise)
        lengths = [300] if pilot else CONFIG['lengths']
        levels = [.25] if pilot else CONFIG['noise_ratios']
        rows = []
        for length in lengths:
            for level in levels:
                observed = (clean + level * noise)[:length + 1]
                test = test_clean + level * test_noise
                for method in ['ridge', 'lasso']:
                    name = f'{method}_T{length}_noise{level:g}'
                    saved = folder / f'{name}.json'
                    if saved.exists():
                        row = json.loads(saved.read_text())['metrics']
                    else:
                        a_hat, intercept, info = fit_model(observed, method)
                        ap, precision, recall = edge_scores(truth, a_hat)
                        mask = ~np.eye(len(a), dtype=bool)
                        prediction = test[:-1] @ a_hat.T + intercept
                        row = dict(seed=seed, length=length, noise=level, method=method, average_precision=ap,
                                   edge_weight_relative_error=float(np.linalg.norm((a_hat-a)[mask])/np.linalg.norm(a[mask])),
                                   test_rmse=float(np.sqrt(mean_squared_error(test[1:], prediction))),
                                   persistence_rmse=float(np.sqrt(mean_squared_error(test[1:], test[:-1]))),
                                   chance_precision=float(truth[mask].mean()),
                                   **{k: v for k, v in info.items() if k != 'trials'})
                        np.savez_compressed(folder / f'{name}.npz', a_hat=a_hat, intercept=intercept,
                                            precision=precision, recall=recall)
                        dump(saved, dict(metrics=row, trials=info['trials']))
                    rows.append(row)
                    print(f"seed={seed} T={length} noise={level:g} {method}: AP={row['average_precision']:.3f}, "
                          f"RMSE={row['test_rmse']:.3f}, alpha={row['alpha']:.3g}, {row['seconds']:.1f}s", flush=True)
        write_csv(folder / 'metrics.csv', rows)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    out = ROOT / 'results' / ('pilot' if args.pilot else 'baseline')
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(configuration=CONFIG, pilot=args.pilot,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, networkx=nx.__version__),
                conventions='A[target,source]; diagonal excluded; common alpha across separate nodewise regressions',
                noise='process covariance I; measurement SD = ratio * population stationary SD per node',
                validation='5 expanding time splits with one-transition gap; penalty chosen by prediction MSE only')
    plan = out / 'protocol.json'
    if plan.exists():
        assert json.loads(plan.read_text()) == spec, 'Protocol changed; use a separate result directory'
    else:
        dump(plan, spec)
    seeds = [5420] if args.pilot else CONFIG['seeds']
    with parallel_backend('loky', inner_max_num_threads=1):
        pieces = Parallel(n_jobs=min(args.jobs, len(seeds)))(delayed(run_seed)(seed, out, args.pilot) for seed in seeds)
    rows = [row for piece in pieces for row in piece]
    write_csv(out / 'metrics.csv', rows)
    print(f'Complete: {len(rows)} fitted models saved to {out}', flush=True)


if __name__ == '__main__':
    main()
