# Changelog

## 0.1.0 (2026-09)

First public release of the code accompanying the JASA article.

- Packaged the implementation as an installable Python package (`pip install .` from `pywarpu/`),
  with a PEP 517 build configuration.
- Bridge-sampling estimators are now importable as `warpu.bridge` (same code as
  `simulation/estimator/bridge.py`, which is kept so the paper's scripts run unchanged).
- Added `examples/quickstart.py`.
- Updated two unit tests to the current return signature of `forward_info_batch`; tests now run against the installed package.
- Added continuous-integration tests, license, and citation metadata.
- Reproduction instructions for the paper moved to `REPRODUCIBILITY.md`.
