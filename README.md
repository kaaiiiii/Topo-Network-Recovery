# Recovering networks from dynamics

ESE 5420 final project, Topic 7: recover a sparse directed network from observed
node-state time series, and measure how observation length, noise and knowledge
of the dynamics affect recovery.

Three completed experiments use ten independently generated 100-node,
300-edge networks:

| Experiment | Estimators | Completed fits | Results |
| --- | --- | ---: | --- |
| Noisy linear diffusion | Lasso and ridge | 180 | [Linear baseline](reference_results/baseline/) |
| Nonlinear SIS dynamics | Raw-state Lasso and ridge | 120 | [SIS comparison](reference_results/sis/) |
| Known SIS dynamics | Nonnegative sparse regression | 60 additional | [SIS-aware comparison](reference_results/sis_aware/) |

PNG figures and CSV tables can be viewed directly on GitHub.

![True and recovered networks, without measurement noise](reference_results/sis_aware/network_example.png)

## Findings

- **More observations help the linear baseline.** At measurement-noise ratio
  0.25, Lasso's mean edge average precision (AP) increases from 0.261 at 100
  transitions to 0.752 at 300 and 0.917 at 1000.
- **Good predictions do not guarantee network recovery.** For noiseless SIS
  at spreading strength 3, raw-state Lasso's mean AP is 0.438, while its test
  RMSE is only 0.153 times the persistence predictor's RMSE.
- **Knowing the SIS equation helps in the noiseless setting.** The SIS-aware
  estimator achieves AP=1 in all 30 noiseless conditions. With measurement noise,
  it does not consistently outperform raw-state Lasso.

Mean AP across ten graphs, with measurement-noise SD 0.02:

| SIS spreading strength | Raw-state Lasso | SIS-aware |
| --- | ---: | ---: |
| 0.8 | 0.038 | 0.039 |
| 1.5 | 0.083 | 0.062 |
| 3.0 | 0.144 | 0.160 |

At strength 3, the paired 95% interval on the AP gain is [-0.004, 0.033].
These intervals resample graph/trajectory realizations.
The SIS-aware estimator knows the dynamics and rates and imposes nonnegative
weights. SIS states are mean-field infection probabilities.

## Reproduce

The reference environment is Python 3.13.13 with the versions in
[requirements.txt](requirements.txt). Run commands from the repository root.
No external dataset download is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# 1. Generate and evaluate the linear experiment.
python scripts/network_recovery.py --jobs 4
python scripts/make_report.py

# 2. Generate SIS observations and fit the raw-state baselines.
python scripts/sis_recovery.py --jobs 4
python scripts/sis_report.py

# 3. Reuse those SIS observations for the model-aware estimator.
python scripts/sis_aware.py --jobs 4
python scripts/sis_aware_report.py
```

On macOS/Linux, activate with `source .venv/bin/activate`; the remaining commands
are identical. Use `--jobs 1` to limit parallel CPU use. Fourteen unit tests cover
simulation, edge direction, validation separation, solver equivalence and the
SIS-aware regression algebra. Figure/table scripts also reconcile saved scores and
solver optimality conditions.

Optional pilot runs use seed 5420, excluded from final summaries: add `--pilot`
to the fitting commands. Run the SIS pilot before the SIS-aware pilot.

**Output policy:** generated trajectories, fitted coefficients and runtime
protocols go in the ignored `results/` directory. The repository includes
curated, read-only reference snapshots in `reference_results/`. On a fresh clone,
run the experiments above to regenerate the raw files needed by report scripts.
Do not copy reference protocols into `results/` before a fresh run: they record
the original environment and exact file hashes. Existing runtime runs resume
only when their frozen protocol matches; archive that output directory before
changing its experiment configuration or environment.

## Repository contents

```text
scripts/             Experiment runners and figure/table generators
tests/               Unit tests
reference_results/   Reference figures, metrics and provenance
docs/EXPERIMENTS.md   Detailed methods and experimental results
requirements.txt     Pinned numerical dependencies
```

`.gitattributes` keeps LF line endings so recorded code hashes survive a Windows checkout. Local
environments, raw runs, logs and exploratory archives are excluded by
`.gitignore`.

The course manuscript and proposal draft are maintained in a separate Overleaf
repository. They are not part of this code repository.

## Methods and references

Read [the detailed experiment notes](docs/EXPERIMENTS.md) for parameters,
validation, uncertainty and convergence checks. The initial
experiment follows the course guide's Lasso exercise. The SIS-aware extension
uses model-based identification with nonnegative sparse regression.

- [Shahrampour and Preciado: topology identification via power spectral analysis](https://arxiv.org/abs/1308.2248).
- [Becker et al.: Wilson–Cowan network identification by inverse transformations](https://www.ieeespmb.org/2016/papers/l04_03.pdf).
- [Prasse and Van Mieghem: network reconstruction for NIMFA processes](https://arxiv.org/abs/1811.06741).
