"""Real topology test: linear diffusion on a SNAP email-Eu-core department.

The directed email network among members of one department (SNAP email-Eu-core,
department labels file) replaces the random G(n,m) graph. Two controls separate
the causes of any difference:
  random  G(n,m) with the same node and edge counts (removes hubs and reciprocity);
  degree  directed edge swaps of the email graph, which keep every node's in- and
          out-degree (keeps hubs) but randomize which pairs connect (removes most
          reciprocal edges).
Dynamics, noise, cross-validation and scoring are imported unchanged from
network_recovery.py. Graph truth never enters fitting or penalty selection.

Data: https://snap.stanford.edu/data/email-Eu-core.html
(Leskovec, Kleinberg and Faloutsos, 2007; Yin, Benson, Leskovec and Gleich, 2017).
"""
import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_recovery import CONFIG, dump, edge_scores, fit_model, simulate, write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'email-Eu-core'
SOURCES = {
    'email-Eu-core.txt.gz': '4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f',
    'email-Eu-core-department-labels.txt.gz': 'e5abe5b4581a480032a63adcf2576c161785f45692642c6ebb0b1276f0f33669',
}
SETUP = dict(department=4, topologies=['email', 'degree', 'random'], lengths=[100, 300, 1000], noise=.25,
             seeds=CONFIG['seeds'], methods=['ridge', 'lasso'])


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCES.items():
        path = DATA / name
        if not path.exists():
            urllib.request.urlretrieve(f'https://snap.stanford.edu/data/{name}', path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'{name}: checksum mismatch'


def department_graph(department):
    """Directed email edges among one department's members; self-loops removed."""
    with gzip.open(DATA / 'email-Eu-core-department-labels.txt.gz', 'rt') as f:
        members = sorted(int(node) for node, dept in (line.split() for line in f) if int(dept) == department)
    index = {node: i for i, node in enumerate(members)}
    edges = set()
    with gzip.open(DATA / 'email-Eu-core.txt.gz', 'rt') as f:
        for line in f:
            source, target = map(int, line.split())
            if source != target and source in index and target in index:
                edges.add((index[source], index[target]))
    return len(members), sorted(edges)


def degree_preserving(edges, nodes, seed):
    graph = nx.DiGraph(); graph.add_nodes_from(range(nodes)); graph.add_edges_from(edges)
    shuffled = nx.directed_edge_swap(graph.copy(), nswap=10 * len(edges), max_tries=10**7, seed=seed)
    assert nx.number_of_selfloops(shuffled) == 0
    assert dict(shuffled.in_degree()) == dict(graph.in_degree())
    assert dict(shuffled.out_degree()) == dict(graph.out_degree())
    return sorted(shuffled.edges)


def topology_stats(edges, nodes):
    graph = nx.DiGraph(); graph.add_nodes_from(range(nodes)); graph.add_edges_from(edges)
    w = nx.to_numpy_array(graph, nodelist=range(nodes)).T
    return dict(reciprocity=nx.reciprocity(graph), binary_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(w)))),
                max_in_degree=int(w.sum(1).max()), max_out_degree=int(w.sum(0).max()))


def system(edges, nodes, seed):
    """Same construction as network_recovery.graph_system, for a given edge list."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    w = np.zeros((nodes, nodes))
    for source, target in edges:
        w[target, source] = rng.uniform(.2, 1.)
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    a = CONFIG['self_memory'] * np.eye(nodes) + CONFIG['coupling_radius'] * w / radius
    covariance = solve_discrete_lyapunov(a, np.eye(nodes))
    covariance = (covariance + covariance.T) / 2
    assert np.max(np.abs(np.linalg.eigvals(a))) < 1
    return a, w != 0, covariance


def run_seed(seed, nodes, email_edges, out):
    rows = []
    graphs = dict(email=email_edges, degree=degree_preserving(email_edges, nodes, seed),
                  random=sorted(nx.gnm_random_graph(nodes, len(email_edges), seed=seed, directed=True).edges))
    with threadpool_limits(limits=1):
        for topology in SETUP['topologies']:
            edges = graphs[topology]
            stats = topology_stats(edges, nodes)
            a, truth, covariance = system(edges, nodes, seed)
            mean_edge_weight = float(a[truth].mean())
            clean = simulate(a, covariance, max(SETUP['lengths']), np.random.SeedSequence([seed, 20]))
            test_clean = simulate(a, covariance, CONFIG['test_steps'], np.random.SeedSequence([seed, 30]))
            sigma = np.sqrt(np.diag(covariance))
            noise = np.random.default_rng(np.random.SeedSequence([seed, 40])).normal(size=clean.shape) * sigma
            test_noise = np.random.default_rng(np.random.SeedSequence([seed, 50])).normal(size=test_clean.shape) * sigma
            test = test_clean + SETUP['noise'] * test_noise
            mask = ~np.eye(nodes, dtype=bool)
            for length in SETUP['lengths']:
                observed = (clean + SETUP['noise'] * noise)[:length + 1]
                for method in SETUP['methods']:
                    a_hat, intercept, info = fit_model(observed, method)
                    ap, _, _ = edge_scores(truth, a_hat)
                    prediction = test[:-1] @ a_hat.T + intercept
                    row = dict(seed=seed, topology=topology, length=length, method=method, average_precision=ap,
                               edge_weight_relative_error=float(np.linalg.norm((a_hat - a)[mask]) / np.linalg.norm(a[mask])),
                               test_rmse=float(np.sqrt(mean_squared_error(test[1:], prediction))),
                               persistence_rmse=float(np.sqrt(mean_squared_error(test[1:], test[:-1]))),
                               chance_precision=float(truth[mask].mean()), alpha=info['alpha'],
                               cv_rmse=info['cv_rmse'], convergence_warnings=info['convergence_warnings'],
                               mean_edge_weight=mean_edge_weight, **stats)
                    rows.append(row)
                    print(f"seed={seed} {topology} T={length} {method}: AP={ap:.3f} "
                          f"RMSE={row['test_rmse']:.3f} alpha={info['alpha']:.3g}", flush=True)
    return rows


def summarize(rows):
    draws = np.random.default_rng(5420).integers(0, len(SETUP['seeds']), size=(10000, len(SETUP['seeds'])))
    def ci(values):
        values = np.asarray(values)
        low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
        return dict(mean=float(values.mean()), low=float(low), high=float(high))
    pick = lambda **k: sorted((r for r in rows if all(r[a] == b for a, b in k.items())), key=lambda r: r['seed'])
    summary = []
    for length in SETUP['lengths']:
        for method in SETUP['methods']:
            groups = {t: pick(topology=t, length=length, method=method) for t in SETUP['topologies']}
            entry = dict(length=length, method=method)
            for metric in ['average_precision', 'test_rmse', 'edge_weight_relative_error']:
                for t, group in groups.items():
                    entry[f'{metric}_{t}'] = ci([r[metric] for r in group])
                    if t != 'email':
                        entry[f'{metric}_email_minus_{t}'] = ci([e[metric] - r[metric] for e, r in zip(groups['email'], group)])
            for t, group in groups.items():
                entry[f'alphas_{t}'] = sorted({r['alpha'] for r in group})
            summary.append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    fetch()
    nodes, edges = department_graph(SETUP['department'])
    out = ROOT / 'results' / 'email_eu_core'
    out.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph(); graph.add_nodes_from(range(nodes)); graph.add_edges_from(edges)
    in_deg, out_deg = np.array([d for _, d in graph.in_degree()]), np.array([d for _, d in graph.out_degree()])
    info = dict(setup=SETUP, nodes=nodes, edges=len(edges), density=len(edges) / (nodes * (nodes - 1)),
                reciprocity=nx.reciprocity(graph), max_in_degree=int(in_deg.max()), max_out_degree=int(out_deg.max()),
                isolated_nodes=int(np.sum(in_deg + out_deg == 0)), sources=SOURCES,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out / 'graph.json', info)
    print(json.dumps({k: v for k, v in info.items() if k not in ('sources', 'setup')}), flush=True)
    with parallel_backend('loky', inner_max_num_threads=1):
        pieces = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s, nodes, edges, out) for s in SETUP['seeds'])
    rows = [r for piece in pieces for r in piece]
    write_csv(out / 'metrics.csv', rows)
    dump(out / 'summary.json', summarize(rows))
    print(f'Complete: {len(rows)} fits saved to {out}', flush=True)


if __name__ == '__main__':
    main()
