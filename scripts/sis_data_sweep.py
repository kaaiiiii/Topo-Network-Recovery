"""How much data does SIS recovery take? Sweep the number of training trajectories.

Same graphs, SIS map, test trajectories and estimators as sis_recovery.py,
sis_aware.py and sis_local_poly.py; their fitting functions are reused unchanged.
New training trajectories (stream 730, up to 80 per network) are simulated once;
smaller training sets are their first k trajectories. Folds hold out whole
trajectories. Ridge and Lasso get four penalties per decade; the Lasso is also
scored with a complete ranking of its zero coefficients (entry penalty on the
Lasso path, as in tuning_and_ties.py). Graph truth never enters fitting.
Every finished condition is saved, so an interrupted run resumes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import osqp
from scipy.optimize import nnls
from scipy.sparse import csc_matrix, eye
from sklearn.linear_model import LassoLars, lasso_path
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sis_recovery as sr  # noqa: E402
import sis_aware as sa  # noqa: E402
import sis_local_poly as slp  # noqa: E402
from network_recovery import ROOT, dump, edge_scores  # noqa: E402
from community_recovery import write_mixed_csv  # noqa: E402

SETUP = dict(trajectories=[5, 10, 20, 40, 80], noise_sd=[0., .005, .01, .02], beta=[.8, 1.5, 3.],
             seeds=sr.CONFIG['seeds'], training_stream=730,
             ridge_alphas=np.logspace(-12, 4, 65).tolist(), lasso_alphas=np.logspace(-6, 0, 25).tolist(),
             methods=['ridge', 'lasso', 'model_based', 'local_poly'])


def entry_scores(x, y):
    """Entry penalty of each source on each target's Lasso path, divided by the input scale."""
    scaler = StandardScaler().fit(x)
    xs, yc = scaler.transform(x), y - y.mean(axis=0)
    entry = np.zeros((y.shape[1], x.shape[1]))
    for target in range(y.shape[1]):
        alphas, coefs, _ = lasso_path(xs, yc[:, target], eps=1e-4, alphas=200)
        active = coefs != 0
        first = np.where(active.any(axis=1), active.argmax(axis=1), -1)
        entry[target] = np.where(first >= 0, alphas[np.clip(first, 0, None)], 0.)
    return entry / scaler.scale_[None, :]


def tiebreak(a_hat, entry):
    magnitude = np.abs(a_hat)
    nonzero = magnitude > 0
    floor = magnitude[nonzero].min() if nonzero.any() else 1.
    scaled = entry / entry.max() if entry.max() > 0 else entry
    return np.where(nonzero, magnitude, .5 * floor * scaled)


RELAXED = dict(count=0, violations=0, fallback_nodes=0, tolerance=1e-5, eps=1e-8, max_iter=20000)


def refine_active_set(gram, xy, coef, alpha, samples, rounds=20):
    """Exact nonnegative-Lasso solution on OSQP's active set: solve gram_AA w = xy_A - n*alpha,
    drop nonpositive weights, and repeat. Keeps whichever of the two solutions has the smaller KKT residual."""
    best, best_res = coef, sa.positive_kkt(gram, xy, coef, alpha, samples)
    active = coef > 0
    for _ in range(rounds):
        if not active.any():
            break
        trial = np.zeros_like(coef)
        trial[active] = np.linalg.lstsq(gram[np.ix_(active, active)], xy[active] - samples * alpha, rcond=None)[0]
        if (trial[active] <= 0).any():
            active &= trial > 0
            continue
        res = sa.positive_kkt(gram, xy, trial, alpha, samples)
        if res < best_res:
            best, best_res = trial, res
        break
    return best


def relaxed_checked_path(design, target, fractions):
    """Same as sis_local_poly.checked_path (positive LARS path, NNLS at zero penalty,
    KKT check), except when a candidate fails the 1e-7 KKT check: then OSQP runs at
    1e-8 for at most 20,000 iterations and an exact active-set solve refines it. With
    very little data the original's 1e-10 / 200,000-iteration QP can end 'solved
    inaccurate' or take minutes per node. Refined candidates above 1e-7 are counted;
    above 1e-5 they are kept but counted as violations."""
    sx = np.sqrt(np.mean(design ** 2, axis=0)); sx[sx <= 1e-14] = 1.
    sy = max(float(np.sqrt(np.mean(target ** 2))), 1e-14)
    x, y = design / sx, target / sy
    gram, xy = x.T @ x, x.T @ y
    n = len(x)
    alpha_max = max(0., float(xy.max() / n))
    warning_count = 0
    if alpha_max > 0 and any(f > 0 for f in fractions):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            path = LassoLars(alpha=min(f for f in fractions if f > 0) * alpha_max, positive=True,
                             fit_intercept=False, precompute=gram, max_iter=sa.CONFIG['lars_max_iter']).fit(x, y)
        warning_count = len(caught)
    candidates, maximum, polished, qp = {}, 0., 0, None
    for fraction in sorted(fractions, reverse=True):
        alpha = fraction * alpha_max
        if alpha_max == 0:
            coef = np.zeros(design.shape[1])
        elif fraction == 0:
            coef = nnls(x, y, maxiter=10000)[0]
        else:
            coef = np.array([np.interp(alpha, path.alphas_[::-1], c[::-1]) for c in path.coef_path_])
        residual = sa.positive_kkt(gram, xy, coef, alpha, n)
        if residual > sa.CONFIG['kkt_tolerance']:
            if qp is None:
                qp = osqp.OSQP()
                qp.setup(P=csc_matrix(gram / n), q=alpha - xy / n, A=eye(len(coef), format='csc'),
                         l=np.zeros(len(coef)), u=np.full(len(coef), np.inf), verbose=False,
                         eps_abs=RELAXED['eps'], eps_rel=RELAXED['eps'], max_iter=RELAXED['max_iter'],
                         polishing=True, adaptive_rho_interval=50)
            else:
                qp.update(q=alpha - xy / n)
            qp.warm_start(x=coef)
            coef = np.maximum(qp.solve(raise_error=False).x, 0.)
            coef[coef < slp.CONFIG['numerical_zero']] = 0.
            coef = refine_active_set(gram, xy, coef, alpha, n)
            polished += 1
            residual = sa.positive_kkt(gram, xy, coef, alpha, n)
            if residual > sa.CONFIG['kkt_tolerance']:
                RELAXED['count'] += 1
            if residual > RELAXED['tolerance']:
                RELAXED['violations'] += 1          # kept and reported, not silently dropped
        maximum = max(maximum, residual)
        candidates[fraction] = coef * sy / sx
    return candidates, dict(max_kkt_residual=maximum, solver_warnings=warning_count, polished_candidates=polished)


_original_positive_path = sa.positive_path


def relaxed_positive_path(design, target, fractions):
    """Model-based Lasso: the original solver (LARS + coordinate-descent polish, 1e-7 KKT
    check) first; only if it raises, re-solve the same scaled nonnegative Lasso with
    relaxed_checked_path and count the node."""
    try:
        return _original_positive_path(design, target, fractions)
    except (RuntimeError, AssertionError):
        RELAXED['fallback_nodes'] += 1
    candidates, info = relaxed_checked_path(design, target, fractions)
    return candidates, info | dict(alpha_max=None)


def fit_one(method, observed, beta, truth, tx, ty):
    row = {}
    if method in ('ridge', 'lasso'):
        a_hat, intercept, info, _ = sr.fit_model(observed, method)
        row['average_precision'] = edge_scores(truth, a_hat)[0]
        row['test_rmse'] = float(np.sqrt(np.mean((tx @ a_hat.T + intercept - ty) ** 2)))
        if method == 'lasso':
            x, y, _ = sr.transitions(observed)
            row['ap_complete_ranking'] = edge_scores(truth, tiebreak(a_hat, entry_scores(x, y)))[0]
            row['true_edges_zeroed'] = float(np.mean(a_hat[truth.astype(bool)] == 0))
        row.update(alpha=info['alpha'], alpha_at_grid_edge=info['alpha'] in
                   (SETUP[f'{method}_alphas'][0], SETUP[f'{method}_alphas'][-1]), seconds=info['seconds'])
    else:
        fit = sa.fit_model if method == 'model_based' else slp.fit_model
        RELAXED['count'] = RELAXED['violations'] = RELAXED['fallback_nodes'] = 0
        weights, info, _ = fit(observed, beta, sr.CONFIG['delta'], sr.CONFIG['dt'])
        row.update(relaxed_qp_solves=RELAXED['count'], fallback_nodes=RELAXED['fallback_nodes'],
                   max_kkt_residual=info['max_kkt_residual'])
        row['kkt_violations_above_1e-5'] = RELAXED['violations']
        row['average_precision'] = edge_scores(truth, weights)[0]
        predicted = sr.sis_step(tx, weights, beta, sr.CONFIG['delta'], sr.CONFIG['dt'])
        row['test_rmse'] = float(np.sqrt(np.mean((predicted - ty) ** 2)))
        row.update(fraction=info['fraction'], window=info.get('window', ''), seconds=info['seconds'])
    return row


def run_case(seed, beta, out, pilot=False, noise_levels=None):
    sr.CONFIG['ridge_alphas'], sr.CONFIG['lasso_alphas'] = SETUP['ridge_alphas'], SETUP['lasso_alphas']
    slp.checked_path = relaxed_checked_path      # set inside the worker process
    sa.positive_path = relaxed_positive_path
    folder = out / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    system = np.load(ROOT / 'results' / 'sis' / f'seed_{seed}' / 'system.npz')
    b, truth = system['b'], system['truth']
    rng = np.random.default_rng(np.random.SeedSequence([seed, SETUP['training_stream']]))
    k_max, nodes = max(SETUP['trajectories']), len(b)
    initial = rng.uniform(sr.CONFIG['initial_low'], sr.CONFIG['initial_high'], size=(k_max, nodes))
    draw = rng.normal(size=(k_max, sr.CONFIG['steps'] + 1, nodes))
    rows = []
    with threadpool_limits(limits=1):
        clean = sr.simulate_sis(b, initial, beta, sr.CONFIG['steps'], sr.CONFIG['delta'], sr.CONFIG['dt'])
        test_clean = sr.simulate_sis(b, system['test_initial'], beta, sr.CONFIG['steps'], sr.CONFIG['delta'], sr.CONFIG['dt'])
        levels = [.02] if pilot else (noise_levels or SETUP['noise_sd'])
        sizes = [max(SETUP['trajectories'])] if pilot else SETUP['trajectories']
        for noise in levels:
            tx, ty, _ = sr.transitions(test_clean + noise * system['test_measurement_draw'])
            for k in sizes:
                observed = (clean + noise * draw)[:k]
                for method in SETUP['methods']:
                    record = folder / f'beta{beta:g}_noise{noise:g}_k{k}_{method}.json'
                    if record.exists():
                        rows.append(json.loads(record.read_text())); continue
                    row = dict(seed=seed, beta=beta, noise_sd=noise, trajectories=k, method=method)
                    try:
                        row.update(fit_one(method, observed, beta, truth, tx, ty))
                    except Exception as error:   # record and continue; a rerun retries unsaved fits
                        with open(out / 'failures.log', 'a', encoding='utf-8') as log:
                            log.write(f'{record.name}\t{type(error).__name__}: {error}\n')
                        print(f'FAILED {record.name}: {type(error).__name__}: {error}', flush=True)
                        continue
                    row['persistence_rmse'] = float(np.sqrt(np.mean((tx - ty) ** 2)))
                    dump(record, row)
                    rows.append(row)
                    print(f"seed={seed} beta={beta:g} noise={noise:g} k={k} {method}: "
                          f"AP={row['average_precision']:.3f} {row['seconds']:.0f}s", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--pilot', action='store_true', help='one seed, beta 3, noise 0.02, 80 trajectories')
    args = parser.parse_args()
    out = ROOT / 'results' / ('sis_data_sweep_pilot' if args.pilot else 'sis_data_sweep')
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'protocol.json', dict(setup=SETUP, sis=sr.CONFIG | dict(ridge_alphas=None, lasso_alphas=None),
                                     code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    # One task per (seed, beta, noise) keeps every worker busy; the same graph and training
    # trajectories are re-simulated deterministically inside each task.
    cases = [(1001, 3., None)] if args.pilot else [(s, b, [n]) for n in SETUP['noise_sd'][::-1]
                                                   for s in SETUP['seeds'] for b in SETUP['beta']]
    with parallel_backend('loky', inner_max_num_threads=1):
        pieces = Parallel(n_jobs=args.jobs)(delayed(run_case)(s, b, out, args.pilot, n) for s, b, n in cases)
    rows = [r for piece in pieces for r in piece]
    write_mixed_csv(out / 'metrics.csv', rows)
    print(f'Complete: {len(rows)} fits saved to {out}', flush=True)


if __name__ == '__main__':
    main()
