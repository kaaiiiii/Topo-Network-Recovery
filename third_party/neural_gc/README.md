# Neural Granger Causality reference implementation

Author: Ian Covert and the Neural-GC authors.
Repository: https://github.com/iancovert/Neural-GC
Paper: https://doi.org/10.1109/TPAMI.2021.3065601

The unmodified author files and MIT license are retained here. `source.json`
records the upstream commit and SHA256 hashes. The project uses a batched cMLP
implementation in `scripts/neural_granger.py`; its forward predictions,
regularized objective, gradients, edge scores and Adam update are checked against
these author files in `tests/test_neural_granger.py`.
