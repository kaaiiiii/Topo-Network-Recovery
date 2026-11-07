"""Topic 7 extension: linear network recovery under nonlinear SIS dynamics.

The simulated model is an explicit Euler, individual-based mean-field SIS map.
States are infection probabilities, not stochastic binary infection events.
"""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_backend
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge, LassoLars
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from network_recovery import ROOT, graph_system, edge_scores, dump, write_csv

CONFIG = dict(nodes=100, edges=300, seeds=list(range(1001, 1011)),
              beta=[.8, 1.5, 3.], delta=1., dt=.05, steps=50,
              training_episodes=20, test_episodes=10, initial_low=.02, initial_high=.2,
              noise_sd=[0., .02], cv_splits=5, cv_seed=5420,
              lasso_alphas=np.logspace(-6, 0, 15).tolist(),
              ridge_alphas=np.logspace(-12, 4, 15).tolist(),
              max_iter=100000, tol=1e-5, lars_max_iter=5000, kkt_tolerance=1e-6,
              lasso_solver='LassoLars path with independently checked KKT residual')
SOURCES = ['https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.SIS_individual_based.html',
           'https://arxiv.org/abs/1505.00768']


def sis_step(p, b, beta, delta=1., dt=.05):
    return p + dt * (beta * (1 - p) * (p @ b.T) - delta * p)


def simulate_sis(b, initial, beta, steps=50, delta=1., dt=.05):
    if np.any(b < 0) or beta < 0 or delta < 0 or dt <= 0:
        raise ValueError('SIS rates and weights must be nonnegative; dt must be positive')
    # Sufficient invariance conditions for the Euler map on [0,1]^n.
    if dt * delta > 1 or dt * beta * np.max(b.sum(axis=1)) > 1:
        raise ValueError('Time step does not preserve probability bounds')
    if np.any(initial < 0) or np.any(initial > 1):
        raise ValueError('Initial states must be probabilities')
    states = np.empty((len(initial), steps + 1, len(b)))
    states[:, 0] = initial
    for t in range(steps):
        states[:, t + 1] = sis_step(states[:, t], b, beta, delta, dt)
    assert states.min() >= 0 and states.max() <= 1
    return states


def transitions(episodes):
    """Construct transitions within episodes, never across restart boundaries."""
    count, times, nodes = episodes.shape
    x = episodes[:, :-1].reshape(-1, nodes)
    y = episodes[:, 1:].reshape(-1, nodes)
    return x, y, np.repeat(np.arange(count), times - 1)


def episode_folds(groups):
    splitter = GroupKFold(n_splits=CONFIG['cv_splits'], shuffle=True, random_state=CONFIG['cv_seed'])
    folds = list(splitter.split(np.empty(len(groups)), groups=groups))
    for train, valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
    return folds


def estimator(method, alpha):
    if method == 'lasso':
        model = Lasso(alpha=alpha, precompute=True, max_iter=CONFIG['max_iter'], tol=CONFIG['tol'])
    elif method == 'ridge':
        model = Ridge(alpha=alpha, solver='svd')
    else:
        raise ValueError(method)
    # Both predictor and response scaling are fitted exclusively to the fold's training episodes.
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model),
                                      transformer=StandardScaler())


def checked_fit(x, y, method, alpha):
    fitted = estimator(method, alpha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        fitted.fit(x, y)
    count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return fitted, count


def fit_path(x, y, method, alphas):
    """Standard LARS Lasso path; scaling and paths stay within each training fold."""
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
    xs, ys = sx.transform(x), sy.transform(y)
    results = {}
    if method == 'lasso':
        gram, xy = xs.T @ xs, xs.T @ ys
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = LassoLars(alpha=min(alphas), fit_intercept=False, precompute=gram,
                              max_iter=CONFIG['lars_max_iter']).fit(xs, ys)
        assert np.max(model.n_iter_) < CONFIG['lars_max_iter']
        for alpha in alphas:
            # The Lasso solution is piecewise linear between LARS knots.
            coefficients = np.stack([np.array([np.interp(alpha, knots[::-1], c[::-1]) for c in path])
                                     for knots, path in zip(model.alphas_, model.coef_path_)])
            gradient = (coefficients @ gram - xy.T)/len(xs)
            active = abs(coefficients) > 1e-10
            residual = np.where(active, abs(gradient+alpha*np.sign(coefficients)),
                                np.maximum(abs(gradient)-alpha, 0))
            kkt = float(residual.max())
            if kkt > CONFIG['kkt_tolerance']:
                raise RuntimeError(f'Lasso alpha={alpha:g}: KKT residual {kkt:g} exceeds tolerance')
            ahat = sy.scale_[:,None]*coefficients/sx.scale_[None,:]
            intercept = sy.mean_-ahat@sx.mean_
            np.testing.assert_allclose(x@ahat.T+intercept, sy.inverse_transform(xs@coefficients.T), atol=1e-10)
            results[alpha] = dict(a_hat=ahat, intercept=intercept, retried_warnings=0,
                                  solver_warnings=len(caught), kkt_residual=kkt,
                                  iterations=int(np.max(model.n_iter_)))
        return results
    model = Ridge(solver='svd')
    for alpha in sorted(alphas, reverse=True):
        model.set_params(alpha=alpha)
        model.fit(xs, ys)
        ahat = sy.scale_[:, None] * model.coef_ / sx.scale_[None, :]
        intercept = sy.mean_ + sy.scale_ * model.intercept_ - ahat @ sx.mean_
        np.testing.assert_allclose(x @ ahat.T + intercept,
                                   sy.inverse_transform(model.predict(xs)), atol=1e-10)
        results[alpha] = dict(a_hat=ahat, intercept=intercept, retried_warnings=0,
                             solver_warnings=0, kkt_residual=0., iterations=0)
    return results


def fit_model(observed, method):
    """Only observations enter the learner. All tuning minimizes prediction MSE."""
    x, y, groups = transitions(observed)
    alphas = CONFIG[f'{method}_alphas']
    errors = {alpha: [] for alpha in alphas}
    retries = {alpha: 0 for alpha in alphas}
    kkt = {alpha: 0. for alpha in alphas}
    solver_warnings = 0
    start = time.perf_counter()
    for train, valid in episode_folds(groups):
        path = fit_path(x[train], y[train], method, alphas)
        solver_warnings += max(f['solver_warnings'] for f in path.values())
        for alpha, fitted in path.items():
            pred = x[valid] @ fitted['a_hat'].T + fitted['intercept']
            errors[alpha].append(float(np.mean((pred-y[valid])**2)))
            retries[alpha] += fitted['retried_warnings']
            kkt[alpha] = max(kkt[alpha], fitted['kkt_residual'])
    trials = [dict(alpha=alpha, cv_mse=float(np.mean(errors[alpha])),
                   fold_rmse=np.sqrt(errors[alpha]).tolist(), convergence_warnings=0,
                   resolved_convergence_warnings=retries[alpha], max_kkt_residual=kkt[alpha]) for alpha in alphas]
    selected = min(trials, key=lambda trial: trial['cv_mse'])
    alpha = selected['alpha']
    fitted = fit_path(x, y, method, [a for a in alphas if a >= alpha])[alpha]
    info = dict(alpha=alpha, cv_rmse=float(np.sqrt(selected['cv_mse'])),
                cv_fold_sd=float(np.std(selected['fold_rmse'], ddof=1)),
                convergence_warnings=0, selected_convergence_warnings=0,
                resolved_convergence_warnings=sum(retries.values())+fitted['retried_warnings'],
                max_kkt_residual=max(max(kkt.values()), fitted['kkt_residual']),
                solver_warnings=solver_warnings+fitted['solver_warnings'],
                max_selected_iterations=fitted['iterations'], seconds=time.perf_counter()-start)
    return fitted['a_hat'], fitted['intercept'], info, trials


def run_seed(seed, out, pilot=False):
    folder = out / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        a, truth, _ = graph_system(seed)
        b = (a - .3 * np.eye(CONFIG['nodes'])) / .6
        rng = np.random.default_rng(np.random.SeedSequence([seed, 710]))
        test_rng = np.random.default_rng(np.random.SeedSequence([seed, 720]))
        shape = (CONFIG['training_episodes'], CONFIG['nodes'])
        test_shape = (CONFIG['test_episodes'], CONFIG['nodes'])
        initial = rng.uniform(CONFIG['initial_low'], CONFIG['initial_high'], size=shape)
        test_initial = test_rng.uniform(CONFIG['initial_low'], CONFIG['initial_high'], size=test_shape)
        noise = rng.normal(size=(shape[0], CONFIG['steps']+1, shape[1]))
        test_noise = test_rng.normal(size=(test_shape[0], CONFIG['steps']+1, test_shape[1]))
        np.savez_compressed(folder / 'system.npz', b=b, truth=truth, initial=initial,
                            test_initial=test_initial, measurement_draw=noise, test_measurement_draw=test_noise)
        rows = []
        for beta in CONFIG['beta']:
            clean = simulate_sis(b, initial, beta, CONFIG['steps'], CONFIG['delta'], CONFIG['dt'])
            test_clean = simulate_sis(b, test_initial, beta, CONFIG['steps'], CONFIG['delta'], CONFIG['dt'])
            np.savez_compressed(folder / f'trajectories_beta{beta:g}.npz', clean=clean, test_clean=test_clean)
            cx, cy, _ = transitions(clean)
            force = cx @ b.T
            diagnostics = dict(mean_prevalence=float(clean.mean()),
                               final_prevalence=float(clean[:, -1].mean()),
                               nonlinear_suppression=float(np.linalg.norm(cx*force)/np.linalg.norm(force)))
            for sd in CONFIG['noise_sd']:
                observed, test = clean + sd * noise, test_clean + sd * test_noise
                tx, ty, _ = transitions(test)
                persistence = float(np.sqrt(np.mean((ty-tx)**2)))
                for method in ['ridge', 'lasso']:
                    name = f'{method}_beta{beta:g}_noise{sd:g}'
                    saved = folder / f'{name}.json'
                    if saved.exists():
                        row = json.loads(saved.read_text())['metrics']
                    else:
                        ahat, intercept, info, trials = fit_model(observed, method)
                        ap, precision, recall = edge_scores(truth, ahat)
                        rmse = float(np.sqrt(np.mean((tx @ ahat.T + intercept-ty)**2)))
                        row = dict(seed=seed, beta=beta, noise_sd=sd, method=method,
                                   average_precision=ap, test_rmse=rmse, persistence_rmse=persistence,
                                   rmse_ratio=rmse/persistence, **diagnostics, **info)
                        np.savez_compressed(folder / f'{name}.npz', a_hat=ahat, intercept=intercept,
                                            precision=precision, recall=recall)
                        dump(saved, dict(metrics=row, trials=trials))
                    rows.append(row)
                    print(f"seed={seed} beta={beta:g} noise={sd:g} {method}: AP={row['average_precision']:.3f}, "
                          f"RMSE/persistence={row['rmse_ratio']:.3f}, {row['seconds']:.1f}s", flush=True)
        write_csv(folder / 'metrics.csv', rows)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    out = ROOT / 'results' / ('sis_pilot' if args.pilot else 'sis')
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=CONFIG, pilot=args.pilot, sklearn=sklearn.__version__, sources=SOURCES,
                    revision='Expanded boundary-limited ridge grid. Standard LARS solves the same Lasso objective; all candidates pass KKT checks. Earlier incomplete coordinate-descent runs retained for audit.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    graph_script_sha256=hashlib.sha256(Path(__file__).with_name('network_recovery.py').read_bytes()).hexdigest(),
                    model='Forward-Euler SIS mean field, continuous probability states, observation noise only',
                    validation='Five folds of whole independent training episodes; separate test episodes')
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Frozen protocol differs; retain previous results and choose a new output directory')
    dump(protocol_path, protocol)
    seeds = [5420] if args.pilot else CONFIG['seeds']
    with parallel_backend('loky', inner_max_num_threads=1):
        batches = Parallel(n_jobs=args.jobs)(delayed(run_seed)(seed, out, args.pilot) for seed in seeds)
    write_csv(out / 'metrics.csv', [row for batch in batches for row in batch])


if __name__ == '__main__':
    main()
