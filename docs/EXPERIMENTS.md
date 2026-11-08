# Detailed experiment notes

Paths below refer to generated local output under `results/`. Published figures,
and summary tables are available in [reference_results](../reference_results/).
The complete reproduction order and environment setup are in the [README](../README.md).
Exploratory intermediate runs are retained only in the ignored local archive;
they are not included in the repository or required to reproduce the final runs.

This implements the topic guide's first experiment: simulate noisy linear dynamics
on a known sparse directed graph, regress each node's next state on every current
node state with Lasso, and evaluate directed edges with precision-recall curves.
Ridge is the reference comparison.

Figures and tables are in **results/baseline/**.

At measurement-noise ratio 0.25, Lasso's mean edge-recovery AP rises from 0.261
at 100 transitions to 0.752 at 300 and 0.917 at 1000 (ten independent graph draws).
At 1000 transitions, ridge has higher AP (0.942), although Lasso predicts the next
observed state better (RMSE 1.070 versus 1.095). Prediction and structure recovery
therefore produce different rankings in this experiment.

All 180 AP values, independent-trajectory RMSE values and penalty selections were
reconciled. There were 52 convergence warnings among nonselected CV candidates;
the selected penalties' CV fits in affected searches were rerun and converged,
and selected full-training fits stayed below the iteration limit.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u scripts/network_recovery.py --pilot
.\.venv\Scripts\python.exe -u scripts/network_recovery.py --jobs 4
.\.venv\Scripts\python.exe scripts/make_report.py
```

The pilot uses seed 5420 and a single condition. The full run uses ten distinct
seeds (1001 through 1010), with no changes to model settings after the pilot.
Completed fits resume only when the saved protocol matches the current script.

## Fixed first experiment

- 100 nodes, 300 directed off-diagonal edges; NetworkX directed G(n,m).
- Positive edge weights U(0.2,1); A = 0.3 I + 0.6 W / spectral_radius(W).
- A[target,source] describes source -> target; spectral radius is 0.9.
- Process innovations are N(0,I); initialize from the exact stationary covariance.
- 100, 300, and 1000 observed transitions.
- Independent measurement-noise SD ratios 0, 0.25, and 0.75 relative to each
  node's population stationary SD. This measurement noise is added to the process innovations. Clean trajectories and noise draws are paired across conditions.
- Independent 1000-transition evaluation trajectories from the same graph.
- Five expanding time splits, approximately half the sequence in the first
  training block, and a one-transition gap to avoid overlapping raw timestamps.
- Fold-local StandardScaler and seven penalties per method. A common alpha is
  chosen by mean prediction MSE across separate node regressions. The true graph
  never enters fitting or penalty selection.
- Rank unscaled absolute coefficients; exclude diagonal entries. AP is computed
  per graph, with mean and sample SD across graphs. Paired bootstrap intervals
  resample the ten graph/trajectory draws.

## Outputs

- `protocol.json`: frozen parameters, versions, and code hash.
- `metrics.csv`, `summary.csv`, `paired_AP_differences.csv`: numeric results.
- `seed_*/system.npz`: exact systems, trajectories, and measurement-noise draws.
- Per-fit JSON/NPZ: every candidate CV score, chosen penalty, coefficients, and PR curve.
- `verification.json`: independent AP, RMSE, edge-direction, and selection checks.
- `recovery_heatmap.png`, `precision_recall.png`, `network_example.png`: exportable figures.

The network illustration uses a fixed seed and condition, and displays the first
25 nodes. The display selects up to 300 positive-score edges on the full graph.
AP evaluates the full off-diagonal coefficient ranking.

Method references are listed in the README.

## Nonlinear extension: SIS

The guide's “Go further” experiment now uses a forward-Euler SIS mean-field map:

`p_next = p + 0.05 * (beta * (1-p) * (B @ p) - p)`.

Here multiplication by `(1-p)` is elementwise. States are infection probabilities.
The model is a discrete numerical version of the standard individual-based
mean-field equation. See the
[EoN equation documentation](https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.SIS_individual_based.html)
and [Nowzari, Preciado and Pappas](https://arxiv.org/abs/1505.00768).

- Reuse the same ten weighted directed graphs, with spectral radius of B equal to one.
- Spreading strength beta: 0.8, 1.5, 3.0; recovery rate: 1.
- 20 training and 10 independent test trajectories, 50 transitions each.
- Independent initial probabilities U(0.02,0.2); all nodes observed.
- Measurement-noise SD 0 or 0.02 in absolute probability units; no process noise.
- Five-fold CV over whole training trajectories; no transitions across restarts.
- Fold-local predictor and response scaling. Fifteen penalties per method:
  Lasso from 1e-6 to 1, ridge from 1e-12 to 1e4. Lasso uses the standard
  [LARS path solver](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html);
  ridge uses SVD. Every Lasso candidate passes an independent KKT optimality
  check with maximum residual at most 1e-6 in standardized units.
- Rank unscaled absolute off-diagonal coefficients. Report AP mean and graph SD,
  paired graph-bootstrap intervals, and test RMSE relative to persistence.

```powershell
.\.venv\Scripts\python.exe -u scripts/sis_recovery.py --pilot --jobs 1
.\.venv\Scripts\python.exe -u scripts/sis_recovery.py --jobs 4
.\.venv\Scripts\python.exe scripts/sis_report.py
```

Outputs live in `results/sis/`. The pilot uses
seed 5420 and all six conditions; it is excluded from final summaries.

The completed 120-fit experiment gives the following Lasso edge AP values
(mean ± sample SD across ten graphs):

| Spreading strength beta | No measurement noise | Measurement-noise SD 0.02 |
| --- | --- | --- |
| 0.8 | 0.9997 ± 0.0007 | 0.038 ± 0.003 |
| 1.5 | 0.970 ± 0.016 | 0.083 ± 0.015 |
| 3.0 | 0.438 ± 0.095 | 0.144 ± 0.024 |

Thus a nonlinear state map does not automatically prevent linear recovery.
In this setup, recovery deteriorates at stronger spreading and with measurement
noise. At beta 3 without noise, Lasso's RMSE is only 0.153 times persistence RMSE,
despite edge AP of 0.438: successful prediction does not guarantee network recovery.
The outputs include ridge, PR curves, paired method comparisons and confidence
intervals. `paired_noise_effects.csv` holds the paired measurement-noise contrasts.
All 120 AP/RMSE/selection records and 60 selected Lasso optimality conditions were
independently checked; no final-run solver warnings occurred. Maximum candidate
KKT residual was 1.20e-7, below the 1e-6 tolerance. Eleven tests cover the linear
and SIS baselines; the later SIS-aware extension brings the total to fourteen.

The initial, incomplete run is retained locally in `.local_archive/results/sis_initial_grid/` with its
source and protocol. Its ridge CV optimum reached the grid's lower boundary,
and cold-start Lasso fits left some candidates unconverged. A warm-start revision
was also slow on strongly correlated noisy data and is retained in
`.local_archive/results/sis_warmstart_partial/`. The final run expands the grid and solves the
same Lasso objective via LARS, checked against coordinate descent in the tests;
its summaries use the completed LARS runs.

SIS uses independent transient trajectories, noise in probability units and
trajectory-level validation. Increasing beta changes coupling and the observed
trajectory distribution.

## SIS-aware network recovery

`scripts/sis_aware.py` adds a third estimator using the known SIS functional form,
recovery rate, spreading rate and sampling step. It regresses
`p_next[i] - (1-h*delta)*p[i]` on `h*beta*(1-p[i])*p[j]` for `j != i`,
with nonnegative weights, zero diagonal, no intercept and an L1 penalty.
Training-fold RMS scaling preserves the physical zero intercept. A common fraction
of each node/fold's alpha-max is selected using the same five trajectory folds.
Fifteen candidates include zero penalty (NNLS) and fourteen logarithmic fractions
from 1e-8 to 1. All candidates pass nonnegative KKT checks; coordinate descent
refines positive LARS solutions when necessary.

The estimator reads the existing SIS trajectories.
The observed arrays, test trajectories and input hashes are paired with the
original ridge/Lasso comparison. The estimator combines known dynamics with
nonnegative edge weights and fits the observed states directly.

```powershell
.\.venv\Scripts\python.exe -u scripts/sis_aware.py --pilot --jobs 1
.\.venv\Scripts\python.exe -u scripts/sis_aware.py --jobs 4
.\.venv\Scripts\python.exe scripts/sis_aware_report.py
```

Open `results/sis_aware/` for the three-method comparison, paired AP
and test-RMSE intervals, and four-panel true/recovered network illustrations.

All 60 new fits completed and passed independent checks. Without measurement
noise, all 30 fits achieve AP=1; at beta 3 the AP gain over raw Lasso is 0.562
with paired 95% interval [0.508, 0.619]. With noise SD 0.02, SIS-aware AP is
0.039, 0.062 and 0.160 across beta 0.8, 1.5 and 3, compared with raw Lasso's
0.038, 0.083 and 0.144. The beta-3 gain interval [-0.004, 0.033] includes zero,
and noisy test RMSE is worse than raw Lasso in every condition. Known dynamics
solve the noiseless model mismatch here; this first estimator does not reliably
resolve measurement error. Fourteen tests pass; 6000 selected node solutions
pass independently recomputed KKT checks. Twelve intermediate solver warnings
were recorded, with every final candidate checked against the 1e-7 KKT tolerance.

This is a SIS adaptation of the model-based identification principle discussed in
[Becker et al. (2016)](https://www.ieeespmb.org/2016/papers/l04_03.pdf), with
[Prasse and Van Mieghem (2018)](https://arxiv.org/abs/1811.06741) as a directly
relevant NIMFA reconstruction reference. The initial partial pilot, which exposed
an insufficient coordinate-descent tolerance, is retained in
`.local_archive/results/sis_aware_pilot_initial/` and excluded from summaries and the repository.

## Local polynomial preprocessing for SIS

`scripts/sis_local_poly.py` estimates states and time derivatives with a degree-2
Savitzky-Golay fit within each training trajectory. The centered derivative
coefficients match equations (3)-(4) of
[Becker et al. (2016)](https://www.ieeespmb.org/2016/papers/l04_03.pdf).
The regression is

`h * (estimated_derivative[i] + delta * smoothed_state[i])`

on `h * beta * (1-smoothed_state[i]) * smoothed_state[j]`, for `j != i`.
Weights are nonnegative, with zero diagonal and an L1 penalty. Training-fold
RMS scaling keeps the zero intercept. This uses the same known rates and
sampling step as the SIS-aware estimator.

Five whole-trajectory folds jointly select the window and regularization:

- Windows: 5, 9, 15 and 21 samples, plus the original raw-transition fit (window 0).
- Fifteen penalty fractions per window, including zero penalty: 75 candidates.
- Complete centered windows supply the training samples; each trajectory is processed separately.
- Validation predicts the next raw observation from the current raw observation.
- Final test RMSE uses the same raw one-step task and independent test trajectories as the earlier methods.

The positive LARS path supplies candidate weights. OSQP refines candidates whose
nonnegative KKT residual exceeds 1e-7; the zero-penalty candidate uses NNLS.
Every candidate is checked against the same 1e-7 tolerance. Five added tests cover
polynomial reproduction, the paper's derivative coefficients, trajectory separation,
the original raw-transition option, and the QP refinement against an analytic solution.

```powershell
.\.venv\Scripts\python.exe -u scripts/sis_local_poly.py --pilot --jobs 1
.\.venv\Scripts\python.exe -u scripts/sis_local_poly.py --jobs 4
.\.venv\Scripts\python.exe scripts/sis_local_poly_report.py
```

Outputs are saved under `results/sis_local_poly/`: four-method AP comparisons,
noisy precision-recall curves, window-validation scores, a fixed-network figure,
and paired bootstrap differences across the ten graphs. The seed-5420 pilot
is stored under `results/sis_local_poly_pilot/`.

All 60 local-polynomial fits completed. All 30 noiseless conditions select window 0
and achieve AP=1. With measurement-noise SD 0.02:

| SIS spreading strength | Raw-state Lasso | SIS-aware | Local polynomial + SIS |
| --- | ---: | ---: | ---: |
| 0.8 | 0.038 | 0.039 | **0.390** |
| 1.5 | 0.083 | 0.062 | **0.637** |
| 3 | 0.144 | 0.160 | **0.710** |

The comparison includes 240 fitted models, with independently checked AP, test RMSE,
precision-recall areas and 6000 selected local-polynomial nodewise KKT conditions.

Noisy test RMSE (same raw one-step prediction task):

| β | Raw Lasso | SIS-aware | Local polynomial + SIS |
| --- | ---: | ---: | ---: |
| 0.8 | 0.026074 | 0.027519 | 0.027507 |
| 1.5 | 0.025975 | 0.027430 | 0.027406 |
| 3 | 0.025545 | 0.026903 | 0.026868 |

## Four-community SIS networks

`scripts/community_recovery.py` compares the existing four estimators on a directed
fixed-count planted partition: 100 nodes in four groups of 25, 60 internal edges
per group, and 60 edges between groups. Both the community and original random
networks have 300 edges. Edge weights are sampled uniformly from [0.2, 1] and
normalized to spectral radius one. The community labels are used for evaluation
and plotting; the estimators fit all off-diagonal candidate edges.

The experiment uses beta=3 and measurement-noise SD 0 or 0.02, with ten network
seeds (1001-1010). It keeps the original SIS sampling step, recovery rate,
20 training trajectories, 10 independent test trajectories, five whole-trajectory
CV folds, and each method's existing parameter grid. Initial states and noise
draws are identical to the random-network experiment for each seed.

Overall edge AP measures recovery. Within-community and bridge recall use the
same global budget of the top 300 positive absolute edge scores. Separate AP
scores are also saved for each edge type; their positive fractions are 240/2400
and 60/7500, respectively. Mean scores and paired differences use 5000 bootstrap
resamples of the ten network seeds (bootstrap seed 85420).

```powershell
.\.venv\Scripts\python.exe -u scripts/community_recovery.py --pilot --jobs 1
.\.venv\Scripts\python.exe scripts/community_report.py --pilot
.\.venv\Scripts\python.exe -u scripts/community_recovery.py --jobs 4
.\.venv\Scripts\python.exe scripts/community_report.py
```

`results/community/` contains topology comparisons, within-community versus bridge
recovery, and fixed-seed network diagrams showing all 100 nodes. The report checks
the source hashes, shared initial states and noise draws, simulated trajectories,
edge scores, held-out prediction errors, and selected nonnegative solutions.

All 80 community fits completed. The 80 AP scores, 80 test RMSE values, 320
edge-stratum scores, and 4000 selected nodewise KKT conditions passed independent
checks. The maximum selected KKT residual was 9.44e-8 (tolerance 1e-7).

With measurement-noise SD 0.02:

| Method | Edge AP (95% bootstrap interval) | Internal recall at 300 | Bridge recall at 300 | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 0.108 [0.098, 0.119] | 0.152 | 0.093 | 0.028548 |
| Raw Lasso | 0.189 [0.173, 0.204] | 0.290 | 0.197 | 0.025449 |
| SIS-aware | 0.186 [0.171, 0.199] | 0.285 | 0.255 | 0.026893 |
| Local polynomial + SIS | **0.773 [0.755, 0.794]** | **0.734** | **0.667** | 0.026854 |

For local polynomial + SIS, community-network AP exceeds random-network AP by
0.062, with paired 95% interval [0.034, 0.088]. Its bridge recall is 0.0675 below
internal recall, with paired interval [-0.0938, -0.0392] for bridge minus internal.
Its AP gain over raw Lasso is 0.584 [0.566, 0.603]. Raw Lasso has the lowest noisy
one-step test RMSE. Without measurement noise, both SIS-aware estimators recover
all true edges at the 300-edge budget on all ten graphs, with AP=1.

Published figures, per-run metrics, all CV candidate scores, paired intervals,
and verification records are in `reference_results/community/`.
