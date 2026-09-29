"""
Quick start: Warp-U sampling on a well-separated bimodal target.

Target (unnormalized):  q(theta) = C * [0.5 N(theta; -m, I) + 0.5 N(theta; +m, I)],  C = 3,
so the true normalizing constant is 3. The two modes are far apart, so a standard
random-walk / HMC chain started in one mode rarely visits the other. The Warp-U
sampler uses a Gaussian mixture Phi_mix placed at the (known) mode locations to
move between modes.

Part 2 estimates the normalizing constant C from the Warp-U draws with the
bridge sampling estimator and the stochastic Warp-U bridge sampling estimator,
using a deliberately imperfect Phi_mix (shifted means, inflated scale).

Run:  python examples/quickstart.py
"""
import numpy as np
from scipy.special import logsumexp

from warpu.gaussmixfull import GaussMixtureFull
from warpu.WarpUSampling import WarpUSampling
from warpu.coupledGradMCMC import coupledGradMCMC
from warpu.bridge import one_kernel_seq

rng_seed = 2026
d = 2
m = np.array([4.0, 4.0])
modes = np.vstack([-m, m])
logC = np.log(3.0)


def logq(theta):
    """log of the unnormalized target; accepts shape (d,) or (n, d)."""
    th = np.atleast_2d(theta)
    comp = np.stack([-0.5 * np.sum((th - mu) ** 2, axis=1) for mu in modes])  # (2, n)
    val = logC + np.log(0.5) - 0.5 * d * np.log(2 * np.pi) + logsumexp(comp, axis=0)
    return val[0] if np.ndim(theta) == 1 else val


def logq_grad(theta):
    comp = np.array([-0.5 * np.sum((theta - mu) ** 2) for mu in modes])
    w = np.exp(comp - logsumexp(comp))
    return sum(w[k] * (modes[k] - theta) for k in range(2))


def build_phimix(shift=0.0, scale=1.0):
    """Phi_mix: a two-component Gaussian mixture located at (or near) the modes."""
    phimix = GaussMixtureFull(2, d)
    prec_chol = np.linalg.cholesky(np.eye(d) / scale**2)
    for k in range(2):
        phimix.set_parameter(k, modes[k] + shift, prec_chol, 0.5)
    phimix.setlogq(logq)
    phimix.setlogq_grad(logq_grad)
    phimix.set_backward_rep(100)
    return phimix


def run(sampler, n_iter):
    sampler.setpi0(lambda: modes[0] + np.random.normal(0, 1, d))   # start in the left mode
    sampler.setlogq(logq)
    sampler.setlogqgrad(logq_grad)
    sampler.run_classical(n_iter)
    return sampler.get_theta1All()


if __name__ == "__main__":
    n_iter = 5000
    np.random.seed(rng_seed)
    hmc_args = (True, 0.1, 10, 1.0, 0.5)   # useHMC, step size, leapfrog steps, momentum sd, P(random-walk kernel)

    warpu = WarpUSampling(d, 0.5, *hmc_args)
    warpu.set_phimix(build_phimix())
    x_warpu = run(warpu, n_iter)

    np.random.seed(rng_seed)
    baseline = coupledGradMCMC(d, 0.5, *hmc_args)
    x_base = run(baseline, n_iter)

    frac = lambda x: np.mean(x[:, 0] > 0)
    print(f"Fraction of draws in the right-hand mode (target: 0.50)")
    print(f"  Warp-U sampler        : {frac(x_warpu):.3f}")
    print(f"  same kernel, no Warp-U: {frac(x_base):.3f}")
    print(f"Mean of first coordinate (target: 0.00)")
    print(f"  Warp-U sampler        : {x_warpu[:, 0].mean():+.3f}")
    print(f"  same kernel, no Warp-U: {x_base[:, 0].mean():+.3f}")

    # ---- Part 2: normalizing constant (true value 3) ----
    print("\nNormalizing constant C (true value 3.000), Phi_mix shifted by 0.3 and scaled by 1.4")
    est = []
    for rep in range(10):
        np.random.seed(100 + rep)
        s = WarpUSampling(d, 0.5, *hmc_args)
        s.set_phimix(build_phimix(0.3, 1.4))
        draws = run(s, 3000)[500:]
        r = one_kernel_seq(draws, build_phimix(0.3, 1.4), logq, [len(draws)], [1, 1, 0])
        est.append(r[0, :2])
    est = np.array(est)
    for j, name in enumerate(["bridge sampling estimator", "stochastic Warp-U bridge"]):
        print(f"  {name:<26}: mean {est[:, j].mean():.3f}, sd over 10 runs {est[:, j].std(ddof=1):.3f}")
