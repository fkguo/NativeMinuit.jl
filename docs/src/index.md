# NativeMinuit.jl

A native-Julia port of [CERN ROOT Minuit2](https://root.cern.ch/doc/master/Minuit2Page.html),
the workhorse function-minimization library used throughout high-energy
physics for χ² and likelihood fits.

## Why?

[iminuit](https://github.com/scikit-hep/iminuit) (Python) wraps the upstream C++
Minuit2 library, and [IMinuit.jl](https://github.com/fkguo/IMinuit.jl) (Julia, by
the same lead author) in turn wraps `iminuit` through PyCall — so it carries both
a Python and a C++ dependency. **NativeMinuit.jl is a clean-room Julia port** of the
same algorithms, with **no C++ or Python dependency and no PyCall / FFI** — plus
full access to Julia tooling (ForwardDiff, threads, broadcasted FCN evaluation).
On the benchmark corpus it runs in the **0.15–0.89× C++ wall-time** range, i.e.
comparable to or faster than C++ Minuit2 — see
[`benchmark/`](https://github.com/fkguo/NativeMinuit.jl/tree/main/benchmark).

## Quick example

```julia
using NativeMinuit

# χ² with a 4-parameter quadratic
cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]))

# Initial parameter values + step sizes
fm = migrad(cf, [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1])   # a FunctionMinimum

show(stdout, MIME"text/plain"(), fm)
```

```
┌───────────────────────────────────────────────────────────────────────┐
│                                Migrad                                 │
├───────────────────────────────────┬───────────────────────────────────┤
│ FCN = 2.257e-18                   │             Nfcn = 26             │
│ EDM = 2.257e-18 (Goal: 0.002)     │                                   │
├───────────────────────────────────┼───────────────────────────────────┤
│           Valid Minimum           │  Below EDM threshold (goal x 10)  │
├───────────────────────────────────┼───────────────────────────────────┤
│      No parameters at limit       │         Below call limit          │
├───────────────────────────────────┼───────────────────────────────────┤
│             Hesse OK              │        Covariance accurate        │
└───────────────────────────────────┴───────────────────────────────────┘
```

Or the iminuit / IMinuit.jl-style front end, with named parameters, limits,
and asymmetric MINOS errors:

```julia
m = Minuit(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]), zeros(4);
           names = ["a", "b", "c", "d"])
migrad!(m)
minos!(m)
m.values        # ≈ [1, 2, 3, 4]
m.merrors       # asymmetric ±σ per parameter (name-keyed Dict)
```

## What's included

- **Minuit2 algorithms** — MIGRAD, HESSE, MINOS, MnContours, Simplex and
  Scan; bounds, fixed parameters, and Strategy levels 0/1/2, ported with
  line-by-line C++ fidelity and iminuit-matching defaults.
- **iminuit / IMinuit.jl-compatible front end** — `m.values`, `m.errors`,
  `migrad!`, `minos!`, `mncontour`, named-parameter access, per-parameter
  `fix!`/`set_limits!`, and Jupyter-first rich output. `Fit`/`ArrayFit` are
  exported aliases of [`Minuit`](@ref).
- **[Cost functions](cost_functions.md)** — a Julia-native family
  (`LeastSquares`, `UnbinnedNLL`, `BinnedNLL`, …) composable with `CostSum`.
- **[Error analysis](error_analysis.md) beyond HESSE/MINOS** — derived-quantity
  intervals & profile bands (`extremize`/`profile_band`), Monte-Carlo Δχ²
  regions, likelihood-ensemble MCMC, a non-mutating **Bayesian posterior bridge**
  (`bayesian`/`posterior_sample` — priors, credible intervals & limits),
  bootstrap, jackknife, and multi-modal solution detection, for the flat or
  strongly non-Gaussian likelihoods where MINOS struggles.
- **AD & threaded gradients** — a ForwardDiff extension and an opt-in
  threaded numerical gradient — plus an `Optim.jl` alternative-minimizer
  bridge (`optim`).

## Tutorials & reference

**Tutorials**

* [Quickstart](tutorials/quickstart.md) — a hands-on first fit.
* [Bounded parameters](tutorials/bounded.md) — parameter limits and fixed
  parameters.
* [MINOS errors & contours](tutorials/minos_contours.md) — asymmetric error
  bars and 2-D confidence contours.

**Guides**

* [Cost functions](cost_functions.md) — the Julia-native cost family.
* [Gradients: AD & threading](guides/gradients.md) — ForwardDiff gradients and
  the threaded / `:auto` numerical gradient.
* [Alternative minimizers](guides/optim.md) — the `Optim.jl` bridge (`optim`).
* [Error analysis](error_analysis.md) — which uncertainty method to use, when.
* [Plotting & rich output](guides/plotting.md) — plot recipes, `draw_*`
  helpers, and LaTeX / ASCII tables.
* [Migrating from iminuit / IMinuit.jl](guides/migration.md) — the drop-in
  mapping.

Full [API reference](api.md) and [internals](internals.md).

## Citation & references

If you use NativeMinuit.jl in a publication, please cite **both** NativeMinuit.jl and
the upstream Minuit algorithms it ports:

> F.-K. Guo, *NativeMinuit.jl: a native-Julia port of Minuit2*,
> <https://github.com/fkguo/NativeMinuit.jl> (2026). A
> [`CITATION.cff`](https://github.com/fkguo/NativeMinuit.jl/blob/main/CITATION.cff)
> is provided (GitHub's "Cite this repository" → APA / BibTeX).

> F. James and M. Roos, "MINUIT: A system for function minimization and
> analysis of the parameter errors and correlations", Comput. Phys. Commun.
> **10** (1975) 343–367. [doi:10.1016/0010-4655(75)90039-9](https://doi.org/10.1016/0010-4655(75)90039-9)

Further Minuit documentation:

- F. James, *MINUIT function minimization and error analysis: Reference manual
  version 94.1*, CERN-D-506 (1994).
- F. James and M. Winkler, *MINUIT user's guide*, CERN (2004).

## License

LGPL-2.1-or-later — matches upstream Minuit2 (the same algorithms,
ported to Julia). See [`docs/UPSTREAM.md`](https://github.com/fkguo/NativeMinuit.jl/blob/main/docs/UPSTREAM.md)
for provenance and attribution.
