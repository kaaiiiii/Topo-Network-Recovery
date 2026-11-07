"""SIS network recovery with local polynomial states and derivative estimates."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, eye
from sklearn.linear_model import LassoLars
import osqp
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

from network_recovery import ROOT, dump, write_csv, edge_scores
from sis_recovery import CONFIG as SIS, transitions, episode_folds, sis_step
from sis_aware import CONFIG as SOLVER, node_design, positive_kkt

CONFIG = dict(windows=[0, 5, 9, 15, 21], degree=2, fractions=SOLVER['fractions'],
              validation='Five whole-trajectory folds; raw one-step prediction MSE',
              boundaries='Use centered complete windows within each training trajectory',
              window_zero='Original SIS-aware fit on raw transitions',
              derivative='Savitzky-Golay first derivative, equivalent to Becker eqs. 3-4 at interior points',
              state='Savitzky-Golay degree-2 state estimate using the same window',
              solver_fallback='OSQP, eps_abs=eps_rel=1e-10, max_iter=200000, polishing=True',
              numerical_zero=1e-9,
              reference='https://www.ieeespmb.org/2016/papers/l04_03.pdf')


def checked_path(design, target, fractions):
    sx = np.sqrt(np.mean(design**2, axis=0))
    sx[sx <= 1e-14] = 1.
    sy = max(float(np.sqrt(np.mean(target**2))), 1e-14)
    x, y = design/sx, target/sy
    gram, xy = x.T@x, x.T@y
    alpha_max = max(0., float(xy.max()/len(x)))
    if alpha_max > 0 and any(f > 0 for f in fractions):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            path = LassoLars(alpha=min(f for f in fractions if f > 0)*alpha_max,
                             positive=True, fit_intercept=False, precompute=gram,
                             max_iter=SOLVER['lars_max_iter']).fit(x, y)
        warning_count = len(caught)
    else:
        warning_count = 0
    candidates, maximum, polished, qp = {}, 0., 0, None
    for fraction in sorted(fractions, reverse=True):
        alpha = fraction*alpha_max
        if alpha_max == 0:
            coef = np.zeros(design.shape[1])
        elif fraction == 0:
            coef = nnls(x, y, maxiter=10000)[0]
        else:
            coef = np.array([np.interp(alpha, path.alphas_[::-1], c[::-1]) for c in path.coef_path_])
        residual = positive_kkt(gram, xy, coef, alpha, len(x))
        if residual > SOLVER['kkt_tolerance']:
            if qp is None:
                qp = osqp.OSQP()
                qp.setup(P=csc_matrix(gram/len(x)), q=alpha-xy/len(x),
                         A=eye(len(coef), format='csc'), l=np.zeros(len(coef)),
                         u=np.full(len(coef), np.inf), verbose=False,
                         eps_abs=1e-10, eps_rel=1e-10, max_iter=200000,
                         polishing=True, adaptive_rho_interval=50)
            else:
                qp.update(q=alpha-xy/len(x))
            qp.warm_start(x=coef)
            result = qp.solve(raise_error=True)
            coef = np.maximum(result.x, 0.)
            coef[coef < CONFIG['numerical_zero']] = 0.
            polished += 1
            residual = positive_kkt(gram, xy, coef, alpha, len(x))
        if residual > SOLVER['kkt_tolerance']:
            raise RuntimeError(f'QP KKT residual {residual:g}, fraction {fraction:g}')
        maximum = max(maximum, residual)
        candidates[fraction] = coef*sy/sx
    return candidates, dict(max_kkt_residual=maximum, solver_warnings=warning_count,
                            polished_candidates=polished)


def fit_path(x, y, beta, fractions, delta, dt):
    weights = {f:np.zeros((x.shape[1], x.shape[1])) for f in fractions}
    info = dict(max_kkt_residual=0., solver_warnings=0, polished_candidates=0)
    for node in range(x.shape[1]):
        design, target, sources = node_design(x, y, node, beta, delta, dt)
        path, diagnostics = checked_path(design, target, fractions)
        for fraction, coef in path.items():
            weights[fraction][node, sources] = coef
        info['max_kkt_residual'] = max(info['max_kkt_residual'], diagnostics['max_kkt_residual'])
        info['solver_warnings'] += diagnostics['solver_warnings']
        info['polished_candidates'] += diagnostics['polished_candidates']
    return weights, info


def regression_samples(observed, window, dt=.05):
    """Construct local state and pseudo-next-state samples within each episode."""
    if window == 0:
        return transitions(observed)[:2]
    if window <= CONFIG['degree'] or window % 2 != 1 or window > observed.shape[1]:
        raise ValueError('Window must be odd, exceed polynomial degree and fit within a trajectory')
    radius = window // 2
    state = savgol_filter(observed, window, CONFIG['degree'], axis=1)
    derivative = savgol_filter(observed, window, CONFIG['degree'], deriv=1, delta=dt, axis=1)
    state = state[:, radius:-radius].reshape(-1, observed.shape[-1])
    derivative = derivative[:, radius:-radius].reshape(state.shape)
    # The SIS-aware solver then fits dt*(derivative + delta*state).
    return state, state + dt*derivative


def fit_model(observed, beta, delta=1., dt=.05):
    started = time.perf_counter()
    folds = episode_folds(np.arange(len(observed)))
    trials, diagnostics = [], []
    for window in CONFIG['windows']:
        errors = {fraction: [] for fraction in CONFIG['fractions']}
        for train, valid in folds:
            x, y = regression_samples(observed[train], window, dt)
            vx, vy, _ = transitions(observed[valid])
            path, info = fit_path(x, y, beta, CONFIG['fractions'], delta, dt)
            diagnostics.append(info)
            for fraction, weights in path.items():
                predicted = sis_step(vx, weights, beta, delta, dt)
                errors[fraction].append(float(np.mean((predicted-vy)**2)))
        trials.extend(dict(window=window, fraction=fraction, cv_mse=float(np.mean(values)),
                           fold_rmse=np.sqrt(values).tolist()) for fraction, values in errors.items())
    selected = min(trials, key=lambda row: row['cv_mse'])
    x, y = regression_samples(observed, selected['window'], dt)
    path, info = fit_path(x, y, beta, [selected['fraction']], delta, dt)
    diagnostics.append(info)
    info = dict(window=selected['window'], fraction=selected['fraction'],
                cv_rmse=float(np.sqrt(selected['cv_mse'])),
                cv_fold_sd=float(np.std(selected['fold_rmse'], ddof=1)),
                training_samples=len(x),
                max_kkt_residual=max(d['max_kkt_residual'] for d in diagnostics),
                solver_warnings=sum(d['solver_warnings'] for d in diagnostics),
                polished_candidates=sum(d['polished_candidates'] for d in diagnostics),
                seconds=time.perf_counter()-started)
    return path[selected['fraction']], info, trials


def run_seed(seed, source, out):
    folder = out/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    data = source/f'seed_{seed}'
    system = np.load(data/'system.npz')
    rows = []
    with threadpool_limits(limits=1):
        for beta in SIS['beta']:
            trajectories = np.load(data/f'trajectories_beta{beta:g}.npz')
            for noise in SIS['noise_sd']:
                name = f'local_poly_beta{beta:g}_noise{noise:g}'
                saved = folder/f'{name}.json'
                if saved.exists():
                    row = json.loads(saved.read_text())['metrics']
                else:
                    observed = trajectories['clean'] + noise*system['measurement_draw']
                    weights, info, trials = fit_model(observed, beta, SIS['delta'], SIS['dt'])
                    ap, precision, recall = edge_scores(system['truth'], weights)
                    tx, ty, _ = transitions(trajectories['test_clean'] + noise*system['test_measurement_draw'])
                    predicted = sis_step(tx, weights, beta, SIS['delta'], SIS['dt'])
                    rmse = float(np.sqrt(np.mean((predicted-ty)**2)))
                    persistence = float(np.sqrt(np.mean((tx-ty)**2)))
                    row = dict(seed=seed, beta=beta, noise_sd=noise, method='local_poly',
                               average_precision=ap, test_rmse=rmse, persistence_rmse=persistence,
                               rmse_ratio=rmse/persistence,
                               weight_relative_error=float(np.linalg.norm(weights-system['b'])/np.linalg.norm(system['b'])),
                               **info)
                    np.savez_compressed(folder/f'{name}.npz', b_hat=weights, precision=precision, recall=recall)
                    dump(saved, dict(metrics=row, trials=trials))
                rows.append(row)
                print(f"seed={seed} beta={beta:g} noise={noise:g}: AP={row['average_precision']:.4f}, "
                      f"window={row['window']}, fraction={row['fraction']:.3g}, {row['seconds']:.1f}s", flush=True)
    write_csv(folder/'metrics.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    source = ROOT/'results'/('sis_pilot' if args.pilot else 'sis')
    out = ROOT/'results'/('sis_local_poly_pilot' if args.pilot else 'sis_local_poly')
    out.mkdir(parents=True, exist_ok=True)
    seeds = [5420] if args.pilot else SIS['seeds']
    inputs = [source/'protocol.json', source/'metrics.csv']
    for seed in seeds:
        inputs += [source/f'seed_{seed}'/'system.npz']
        inputs += [source/f'seed_{seed}'/f'trajectories_beta{beta:g}.npz' for beta in SIS['beta']]
    scripts = ['sis_local_poly.py', 'sis_aware.py', 'sis_recovery.py', 'network_recovery.py']
    protocol = dict(config=CONFIG, sis_config=SIS, solver_config=SOLVER, pilot=args.pilot,
                    numpy_version=np.__version__, scipy_version=scipy.__version__, osqp_version=osqp.__version__,
                    script_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in scripts},
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Frozen protocol differs; preserve old outputs before revising')
    dump(protocol_path, protocol)
    with parallel_backend('loky', inner_max_num_threads=1):
        batches = Parallel(n_jobs=args.jobs)(delayed(run_seed)(seed, source, out) for seed in seeds)
    write_csv(out/'metrics.csv', [row for batch in batches for row in batch])


if __name__ == '__main__':
    main()
