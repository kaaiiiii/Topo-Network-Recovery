"""Is the ridge-vs-Lasso recovery comparison fair? Two checks.

1. Penalty grid. The original grids have seven values a factor ~7 apart, and some
   fits select a grid endpoint. Here both methods get four values per decade over a
   wider range (ridge 1e-2..1e6, Lasso 1e-4..1e1), still chosen by time-ordered
   CV on prediction MSE only.
2. Lasso ties. Lasso sets most coefficients exactly to zero, and tied zero scores
   cannot be ranked, which lowers AP. Two complete rankings are added:
     tiebreak  nonzero coefficients by magnitude, then zeros by the penalty at which
               they enter the Lasso path (selected model unchanged);
     path      every edge by its entry penalty on the Lasso path.
   Entry penalties are divided by the predictor scale, like the coefficients, so
   scores are in original units across target regressions.

Graphs, dynamics, noise, folds and scoring are imported from network_recovery.py.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network_recovery as nr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRIDS = dict(ridge_alphas=np.logspace(-2, 6, 33).tolist(), lasso_alphas=np.logspace(-4, 1, 21).tolist())
SETUP = dict(lengths=[100, 200, 300, 500, 700, 1000, 1500, 2000, 3000], noise_ratios=nr.CONFIG['noise_ratios'],
             seeds=nr.CONFIG['seeds'], path_eps=1e-4, path_alphas=200, grids=GRIDS)
SCORES = ['ridge', 'lasso', 'lasso_tiebreak', 'lasso_path']


def entry_scores(observed):
    """Entry penalty of each source on each target's Lasso path, in original units."""
    x, y = observed[:-1], observed[1:]
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    yc = y - y.mean(axis=0)
    entry = np.zeros((y.shape[1], x.shape[1]))
    for target in range(y.shape[1]):
        alphas, coefs, _ = lasso_path(xs, yc[:, target], eps=SETUP['path_eps'], alphas=SETUP['path_alphas'])
        active = coefs != 0                                   # (sources, alphas), alphas decreasing
        first = np.where(active.any(axis=1), active.argmax(axis=1), -1)
        entry[target] = np.where(first >= 0, alphas[np.clip(first, 0, None)], 0.)
    return entry / scaler.scale_[None, :]


def tiebreak(a_hat, entry):
    """Keep nonzero magnitudes; order zeros below them by entry penalty."""
    magnitude = np.abs(a_hat)
    nonzero = magnitude > 0
    floor = magnitude[nonzero].min() if nonzero.any() else 1.
    scaled = entry / entry.max() if entry.max() > 0 else entry
    return np.where(nonzero, magnitude, .5 * floor * scaled)


def run_seed(seed):
    nr.CONFIG.update(GRIDS)          # set inside the worker process
    rows = []
    with threadpool_limits(limits=1):
        a, truth, covariance = nr.graph_system(seed, nr.CONFIG['nodes'], nr.CONFIG['edges'])
        clean = nr.simulate(a, covariance, max(SETUP['lengths']), np.random.SeedSequence([seed, 20]))
        sigma = np.sqrt(np.diag(covariance))
        noise = np.random.default_rng(np.random.SeedSequence([seed, 40])).normal(size=clean.shape) * sigma
        mask = ~np.eye(len(a), dtype=bool)
        labels = truth[mask].astype(int)
        for level in SETUP['noise_ratios']:
            for length in SETUP['lengths']:
                observed = (clean + level * noise)[:length + 1]
                row = dict(seed=seed, noise=level, length=length)
                for method in ['ridge', 'lasso']:
                    a_hat, _, info = nr.fit_model(observed, method)
                    grid = GRIDS[f'{method}_alphas']
                    row[f'{method}_alpha'] = info['alpha']
                    row[f'{method}_alpha_at_edge'] = info['alpha'] in (grid[0], grid[-1])
                    row[f'{method}_convergence_warnings'] = info['convergence_warnings']
                    row[f'ap_{method}'] = float(average_precision_score(labels, np.abs(a_hat[mask])))
                    if method == 'lasso':
                        entry = entry_scores(observed)
                        row['lasso_zero_fraction'] = float(np.mean(a_hat[mask] == 0))
                        row['lasso_true_edges_zeroed'] = float(np.mean(a_hat[mask][labels == 1] == 0))
                        row['ap_lasso_tiebreak'] = float(average_precision_score(labels, tiebreak(a_hat, entry)[mask]))
                        row['ap_lasso_path'] = float(average_precision_score(labels, entry[mask]))
                rows.append(row)
        print(f'seed {seed} done', flush=True)
    return rows


def summarize(rows):
    draws = np.random.default_rng(5420).integers(0, len(SETUP['seeds']), size=(10000, len(SETUP['seeds'])))
    def ci(values):
        values = np.asarray(values, float)
        boot = values[draws].mean(axis=1)
        return dict(mean=float(values.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)))
    out = []
    for level in SETUP['noise_ratios']:
        for length in SETUP['lengths']:
            group = sorted((r for r in rows if r['noise'] == level and r['length'] == length), key=lambda r: r['seed'])
            entry = dict(noise=level, length=length)
            for score in SCORES:
                entry[f'ap_{score}'] = ci([r[f'ap_{score}'] for r in group])
                if score != 'ridge':
                    entry[f'{score}_minus_ridge'] = ci([r[f'ap_{score}'] - r['ap_ridge'] for r in group])
            entry['ridge_alphas'] = sorted({r['ridge_alpha'] for r in group})
            entry['lasso_alphas'] = sorted({r['lasso_alpha'] for r in group})
            entry['fits_at_grid_edge'] = sum(r['ridge_alpha_at_edge'] + r['lasso_alpha_at_edge'] for r in group)
            entry['lasso_true_edges_zeroed'] = float(np.mean([r['lasso_true_edges_zeroed'] for r in group]))
            out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    out = ROOT / 'results' / 'tuning_and_ties'
    out.mkdir(parents=True, exist_ok=True)
    nr.dump(out / 'protocol.json', dict(setup=SETUP, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with parallel_backend('loky', inner_max_num_threads=1):
        pieces = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s) for s in SETUP['seeds'])
    rows = [r for piece in pieces for r in piece]
    nr.write_csv(out / 'metrics.csv', rows)
    nr.dump(out / 'summary.json', summarize(rows))
    print(f'Complete: {len(rows)} conditions saved to {out}', flush=True)


if __name__ == '__main__':
    main()
