"""cMLP (neural Granger causality) at every SIS spreading strength and noise level.

neural_granger.py ran cMLP at beta=3 only. This driver reuses its fitting code
unchanged (same architecture, penalties, training lengths, folds, seeds and
deterministic GPU settings) on the saved SIS trajectories from sis_recovery.py
for beta 0.8, 1.5 and 3. beta=3 is refit as a reproduction check.
Per-fit records are saved as they finish, so an interrupted run resumes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_granger as ng  # noqa: E402
from network_recovery import ROOT, dump, edge_scores, write_csv  # noqa: E402
from sis_recovery import transitions  # noqa: E402

SETUP = dict(beta=[.8, 1.5, 3.], noise_sd=[0., .02], seeds=ng.CONFIG['seeds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT / 'results' / 'cmlp_all_beta'
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'protocol.json', dict(setup=SETUP, cmlp=ng.CONFIG, torch=torch.__version__, device=args.device))
    rows = []
    with threadpool_limits(limits=1):
        for seed in SETUP['seeds']:
            source = ROOT / 'results' / 'sis' / f'seed_{seed}'
            system = np.load(source / 'system.npz')
            for beta in SETUP['beta']:
                traj = np.load(source / f'trajectories_beta{beta:g}.npz')
                for noise in SETUP['noise_sd']:
                    record = out / f'seed{seed}_beta{beta:g}_noise{noise:g}.json'
                    if record.exists():
                        rows.append(json.loads(record.read_text())); continue
                    observed = traj['clean'] + noise * system['measurement_draw']
                    arrays, info, _, _ = ng.fit_model(observed, seed, args.device)
                    scores = np.linalg.norm(arrays['w1'], axis=1)
                    ap, _, _ = edge_scores(system['truth'], scores)
                    tx, ty, _ = transitions(traj['test_clean'] + noise * system['test_measurement_draw'])
                    rmse = float(np.sqrt(np.mean((ng.predict(arrays, tx) - ty) ** 2)))
                    row = dict(seed=seed, beta=beta, noise_sd=noise, method='cmlp', average_precision=ap,
                               test_rmse=rmse, persistence_rmse=float(np.sqrt(np.mean((tx - ty) ** 2))), **info)
                    dump(record, row)
                    rows.append(row)
                    print(f'seed={seed} beta={beta:g} noise={noise:g}: AP={ap:.4f} '
                          f'alpha={info["alpha"]:g} epoch={info["epoch"]} {info["seconds"]:.0f}s', flush=True)
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
    write_csv(out / 'metrics.csv', rows)
    print(f'Complete: {len(rows)} fits', flush=True)


if __name__ == '__main__':
    main()
