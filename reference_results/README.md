# Reference results

Curated snapshots of the completed experiments:

- [Linear diffusion](baseline/): 180 Lasso/ridge fits.
- [SIS with raw-state regressions](sis/): 120 fits.
- [SIS-aware comparison](sis_aware/): 60 additional fits, compared with the 120 SIS baselines.

- [Local polynomial SIS recovery](sis_local_poly/): 60 additional fits with window and penalty selection.

Each folder contains figures, aggregate and per-fit metric tables, the recorded
protocol and a verification summary. PNG and CSV files can be inspected on GitHub.

Raw trajectories, per-fit coefficient arrays, complete CV-search records and
exploratory runs are generated under the ignored `results/` directory and are not
part of these snapshots. Follow the [reproduction commands](../README.md#reproduce)
to regenerate them. Run all four experiment stages in order before rebuilding
the final comparison figures and tables.

The protocols record experiment settings and source and input hashes.
`sis_aware/editorial_revision.json` records the source hashes before and after
the prose cleanup; the computation was verified unchanged.
`manifest.json` hashes the included snapshot files. Snapshots are preserved
byte-for-byte by `.gitattributes`; fresh runs create their own runtime protocols.
Do not seed a new runtime directory with these historical protocol files.
