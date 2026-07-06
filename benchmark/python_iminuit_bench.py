#!/usr/bin/env python3
"""
Python iminuit wall-time benchmark: MIGRAD + MINOS + MNCONTOUR.

Mirrors the NativeMinuit + C++ Minuit2 benchmark FCNs (rosenbrock_2d,
rosenbrock_10d, quad_4d, gauss_ll_2_100, gauss_ll_10_1000) so the
three implementations can be compared head-to-head.

Run via: `python3 benchmark/python_iminuit_bench.py > /tmp/iminuit_bench.json`
"""

import json
import sys
import time
from statistics import median

import numpy as np
from iminuit import Minuit


# Same RNG seeds as the C++/Julia benches → identical Gaussian datasets
RNG = np.random.default_rng(0xCAFE_F00D)


def rosenbrock_2(p):
    return (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2


def quad_nd(p):
    return float(np.sum(np.asarray(p)**2))


def rosenbrock_nd(p):
    p = np.asarray(p)
    return float(np.sum(100 * (p[1:] - p[:-1]**2)**2 + (1 - p[:-1])**2))


def gauss_nll_factory(n_events, true_mu, true_sigma):
    """N(μ, σ²) negative log-likelihood, two free params (μ, σ)."""
    data = RNG.normal(true_mu, true_sigma, n_events)
    def f(p):
        mu, sigma = p[0], p[1]
        if sigma <= 0:
            return 1e30
        d = data - mu
        return float(np.sum(np.log(sigma) + 0.5 * d**2 / sigma**2))
    return f


def gauss_nll_nd_factory(n_pars, n_events):
    """n-dim Gaussian product NLL with all-independent μ_i."""
    n_per = max(1, n_events // n_pars)
    truths = RNG.normal(0.0, 1.0, n_pars)
    data = [t + RNG.normal(0.0, 1.0, n_per) for t in truths]
    def f(p):
        s = 0.0
        for i in range(n_pars):
            d = data[i] - p[i]
            s += 0.5 * float(np.sum(d * d))
        return s
    return f


def time_op(setup_fn, op_fn, n_samples=30):
    """Run setup→op n_samples times, return median wall-time in ns + diagnostic."""
    times = []
    last_result = None
    for _ in range(n_samples):
        m = setup_fn()
        t0 = time.perf_counter_ns()
        last_result = op_fn(m)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return float(median(times)), last_result


def main():
    bench_cases = [
        ("rosenbrock_2d",
         lambda: Minuit(rosenbrock_2, (-1.2, 1.0),
                        name=("p0", "p1")),
         {"errordef": 1.0, "name_pars": ["p0", "p1"]}),
        ("rosenbrock_10d",
         lambda: Minuit(rosenbrock_nd,
                        np.array([(-1.2, 1.0)[i & 1] for i in range(10)]),
                        name=tuple(f"p{i}" for i in range(10))),
         {"errordef": 1.0, "name_pars": [f"p{i}" for i in range(10)]}),
        ("quad_4d",
         lambda: Minuit(quad_nd, (1.0, 1.0, 1.0, 1.0),
                        name=("p0", "p1", "p2", "p3")),
         {"errordef": 1.0, "name_pars": ["p0", "p1", "p2", "p3"]}),
        ("gauss_ll_2_100",
         lambda: Minuit(gauss_nll_factory(100, 2.0, 1.0), (1.0, 2.0),
                        name=("mu", "sigma")),
         {"errordef": 0.5, "name_pars": ["mu", "sigma"]}),
        ("gauss_ll_10_1000",
         lambda: Minuit(gauss_nll_nd_factory(10, 1000),
                        np.zeros(10),
                        name=tuple(f"p{i}" for i in range(10))),
         {"errordef": 0.5, "name_pars": [f"p{i}" for i in range(10)]}),
    ]

    results = []
    for name, factory, cfg in bench_cases:
        # ── MIGRAD (Strategy 0) ─────────────────────────────────────
        def setup_migrad():
            m = factory()
            m.errordef = cfg["errordef"]
            m.errors = [0.1] * len(cfg["name_pars"])
            m.strategy = 0
            m.print_level = 0
            return m

        def do_migrad(m):
            m.migrad()
            return (float(m.fval), int(m.nfcn))

        migrad_ns, migrad_diag = time_op(setup_migrad, do_migrad)

        # ── MINOS on parameter 0 (after MIGRAD) ─────────────────────
        def setup_minos():
            m = factory()
            m.errordef = cfg["errordef"]
            m.errors = [0.1] * len(cfg["name_pars"])
            m.strategy = 0
            m.print_level = 0
            m.migrad()
            return m

        def do_minos(m):
            try:
                m.minos(cfg["name_pars"][0])
                return float(m.merrors[cfg["name_pars"][0]].upper)
            except Exception as e:
                return float("nan")

        minos_ns, minos_diag = time_op(setup_minos, do_minos, n_samples=20)

        # ── MNCONTOUR (par 0 × par 1, 30 boundary points) ───────────
        def do_mncontour(m):
            try:
                pts = m.mncontour(cfg["name_pars"][0], cfg["name_pars"][1],
                                   size=30, cl=0.68)
                return len(pts)
            except Exception as e:
                return -1

        # Reuse the MIGRAD-converged setup_minos for MNCONTOUR
        mnct_ns, mnct_diag = time_op(setup_minos, do_mncontour, n_samples=10)

        results.append({
            "name": name,
            "migrad_ns": migrad_ns,
            "minos_ns": minos_ns,
            "mncontour_ns": mnct_ns,
            "migrad_fval": migrad_diag[0],
            "migrad_nfcn": migrad_diag[1],
            "minos_upper": minos_diag,
            "mncontour_npts": mnct_diag,
        })

    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
