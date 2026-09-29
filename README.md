# warpU — Warp-U sampling and stochastic bridge sampling for multimodal distributions

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

`warpu` is a Python/Cython implementation of the **Warp-U sampler** and the **stochastic Warp-U
bridge sampling estimator** introduced in

> Fei Ding, Shiyuan He, David E. Jones & Xiao-Li Meng (2026). Channeling Multimodality Through a
> Unimodalizing Transport: Warp-U Sampler and Stochastic Bridge Sampling Estimator.
> *Journal of the American Statistical Association*.
> [doi:10.1080/01621459.2026.2691322](https://doi.org/10.1080/01621459.2026.2691322)

Multimodal posterior distributions are hard for standard MCMC: a chain started near one mode can
stay there indefinitely, and normalizing constants (Bayesian evidence) estimated from such output
can be badly wrong. A Warp-U transformation maps a multimodal density into a unimodal one using a
mixture approximation `Phi_mix` placed at the mode locations; the Warp-U sampler moves between
modes by applying the transformation and then inverting it with injected stochasticity. The same
construction yields a bridge sampling estimator of the normalizing constant with higher asymptotic
precision per CPU second than the original Warp-U bridge estimator of Wang, Jones & Meng (2022),
under the conditions stated in the paper.

## Installation

Requires Python ≥ 3.9 and a C compiler.

```bash
git clone https://github.com/feidingDS/warpU.git
cd warpU/pywarpu
pip install .
```

To reproduce the paper's full environment (including PyTorch for the neural-ODE `Phi_mix`, and R
for the real-data analysis), see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and `environment.yml`.

## Quick start

```bash
python examples/quickstart.py
```

The example targets a two-dimensional mixture of two well-separated Gaussians with a known
normalizing constant of 3, and starts every chain in the left-hand mode. Output on our machine:

```
Fraction of draws in the right-hand mode (target: 0.50)
  Warp-U sampler        : 0.508
  same kernel, no Warp-U: 0.000
Mean of first coordinate (target: 0.00)
  Warp-U sampler        : +0.038
  same kernel, no Warp-U: -3.968

Normalizing constant C (true value 3.000), Phi_mix shifted by 0.3 and scaled by 1.4
  bridge sampling estimator : mean 3.029, sd over 10 runs 0.045
  stochastic Warp-U bridge  : mean 3.026, sd over 10 runs 0.039
```

The core pattern is:

```python
from warpu.gaussmixfull import GaussMixtureFull
from warpu.WarpUSampling import WarpUSampling
from warpu.bridge import one_kernel_seq

phimix = GaussMixtureFull(K, d)                 # mixture placed at the mode locations
for k in range(K):
    phimix.set_parameter(k, mu[k], prec_chol[k], weight[k])
phimix.setlogq(logq); phimix.setlogq_grad(logq_grad)

sampler = WarpUSampling(d, rw_sd, True, hmc_step, hmc_steps, hmc_sd, p_random_walk)
sampler.set_phimix(phimix)
sampler.setpi0(init_fn); sampler.setlogq(logq); sampler.setlogqgrad(logq_grad)
sampler.run_classical(n_iter)
draws = sampler.get_theta1All()

estimates = one_kernel_seq(draws, phimix, logq, [len(draws)], [1, 1, 0])
```

## What is in this repository

| Path | Contents |
|---|---|
| `pywarpu/warpu/` | The package: Warp-U sampler (`WarpUSampling`), base MCMC kernels (`coupledMH`, `coupledGradMCMC`), mixture families for `Phi_mix` (Gaussian with full/diagonal covariance, skew-t, Gaussian–gamma, Gaussian–skew), parallel tempering (`tempering`), and bridge estimators (`bridge`) |
| `pywarpu/tests/` | Unit tests (`pytest pywarpu/tests`) |
| `examples/` | Self-contained usage examples |
| `simulation/` | Code for the simulation study in the paper |
| `realdata/` | Code and data for the exoplanet (radial-velocity) analysis in the paper, in R |
| `REPRODUCIBILITY.md` | Step-by-step instructions for reproducing every figure and table in the paper |

## Scope and limitations

As discussed in the paper, in high dimensions the method relies on information about the mode
locations, and the efficiency comparison for the stochastic bridge estimator is derived under an
independent-draw assumption that the sampler itself does not satisfy; extending that analysis to
dependent draws is open work. Warp-U sampling does not universally outperform parallel tempering;
it is offered as a competitive alternative.

## Roadmap

Planned development toward a documented, general-purpose package:

- user-facing API with documented parameters and defaults, and API reference documentation;
- automatic construction of `Phi_mix` from mode-finding or variational fits;
- diagnostics for Warp-U chains and bridge estimates under Markov-chain dependence;
- packaged releases (PyPI).

Issues and pull requests are welcome.

## Citation

If you use this software, please cite the paper above (see also [`CITATION.cff`](CITATION.cff)).

## Authors

Fei Ding, Shiyuan He, David E. Jones and Xiao-Li Meng. Repository maintained by Fei Ding.

## License

MIT — see [`LICENSE`](LICENSE).
