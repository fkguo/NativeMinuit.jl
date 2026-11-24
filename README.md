# NativeMinuit.jl

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fkguo.github.io/NativeMinuit.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24475297.svg)](https://doi.org/10.5281/zenodo.24475297)
[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL%20v2.1%2B-blue.svg)](LICENSE)
[![Julia General](https://img.shields.io/badge/Julia%20General-registered-9558B2.svg)](https://github.com/JuliaRegistries/General/tree/master/N/NativeMinuit)
[![Pkg requests](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fpackages%2FNativeMinuit%2Fstats%3Ftimeframe%3Dall%26user_data%3Dtrue%26ci_data%3Dtrue%26missing_data%3Dtrue&query=timeframe_requests&label=Pkg%20requests&color=brightgreen)](https://juliapkgstats.com/pkg/NativeMinuit?timeframe=all&trendingPeriod=14d&userData=true&ciData=true&missingData=true)
[![Binder: IAM fit](https://img.shields.io/badge/Binder-IAM%20fit-F37626?logo=jupyter)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FIAM_2Pformfactor%2Fiamfit.ipynb)
[![Binder: X(3872) dip](https://img.shields.io/badge/Binder-X%283872%29%20dip-F37626?logo=jupyter)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FX3872_dip%2FXdip_published.ipynb)

NativeMinuit.jl ports the C++ [Minuit2](https://root.cern.ch/doc/master/Minuit2Page.html)
function-minimization library to Julia. It provides an iminuit-style API for
fitting and error analysis, including profile-likelihood contours, Monte Carlo
confidence regions, likelihood-ensemble sampling, Bayesian posterior analysis,
bootstrap and jackknife estimates, and detection of multiple minima.

License: **LGPL 2.1 or later** (mirrors upstream Minuit2). This is a derivative
work of C++ Minuit2 — see [`LICENSE`](LICENSE) and [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

## Installation

```julia
using Pkg
Pkg.add("NativeMinuit")
```

NativeMinuit requires **Julia 1.11 or later**. Install and load the packages
listed below to enable the corresponding integrations.

| Capability | Install and load |
|---|---|
| Automatic differentiation | `ForwardDiff` |
| Alternative minimizers | `Optim` |
| Plots / drawing helpers | `Plots` |
| DataFrame output | `DataFrames` |
| No-U-Turn sampler (NUTS) | `AdvancedHMC`, `LogDensityProblems`, `LogDensityProblemsAD`, `TransformVariables`, `ForwardDiff` |

For automatic differentiation (AD), install `ForwardDiff` and pass a gradient
with `grad=` to `Minuit`. Load `NativeMinuit` and `ForwardDiff` together to use
the `CostFunctionAD` constructor with the low-level API.

## Quick start

```julia
using NativeMinuit

# iminuit-style API
m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
           [0.0, 0.0];
           names  = ["a", "b"],
           errors = [0.1, 0.1],
           limits = [(-5.0, 5.0), nothing])
migrad!(m; tol = 1e-6)    # explicit EDM tolerance for this quadratic example
hesse!(m)
minos!(m)

println(m.values)        # ≈ [1.0, 2.0]
println(m.errors)        # parabolic (HESSE) 1σ errors
println(m.fval)          # ≈ 0.0
println(m.minos_errors)  # asymmetric ±σ per parameter
m                        # rich table (HTML in Jupyter, text in the REPL)
```

`Fit` and `ArrayFit` are exported as IMinuit.jl-compatible aliases, so existing
IMinuit.jl scripts largely work unchanged.

## Beyond C++ Minuit2

NativeMinuit implements MIGRAD, HESSE, MINOS, MnContours, Simplex and Scan with
bounds, fixed parameters and strategy levels. The Julia interface also provides:

- **iminuit / IMinuit.jl-style front end** — `Minuit`, `migrad!`, `minos!` and the fit properties `m.values`, `m.errors` and `m.merrors`.
- **Julia-native cost functions** — `LeastSquares`, `UnbinnedNLL`, `BinnedNLL`, `ExtendedUnbinnedNLL`, `ExtendedBinnedNLL`, and their `CostSum`.
- **Automatic-differentiation gradients** (ForwardDiff) and a **threaded numerical gradient**.
- **Derived-quantity intervals & error bands** — `extremize` / `profile_band` find the range of a derived quantity over a confidence region.
- **Monte Carlo confidence regions** — `get_contours_samples` samples the joint region defined by the objective threshold.
- **Likelihood-ensemble MCMC** — `mcmc_sample` and `quantile_band` provide marginal intervals and curve bands.
- **Bayesian posterior analysis** — `bayesian` / `posterior_sample`: priors, credible intervals & limits, and three samplers (random-walk, affine-invariant ensemble, NUTS).
- **Bootstrap & jackknife** — `bootstrap` / `jackknife` (data-resampling errors).
- **Multi-modal solution detection** — `find_solution_modes` (cluster statistically distinct minima).
- **Basin escape** — `find_deeper_minimum` (basin-hopping to a deeper minimum).
- **Alternative minimizers** — `optim` / `minimize_with` (Optim.jl integration).
- **Rich output** — `to_latex` publication tables, Plots recipes, Unicode/HTML auto-display, ASCII plots.

## Performance

NativeMinuit **0.8.0** and C++ Minuit2 **6.24.0** were measured on 23 November
2026 on an Apple M3 MacBook running macOS 15.6.1. The toolchain
was Julia 1.13.2 and Apple clang 17 (`-O3 -DNDEBUG`).
The benchmark uses one Julia thread and one BLAS thread. Both implementations
use the same fixed inputs, numerical gradients, `Strategy(0)`, tolerance 0.1,
a limit of 10,000 objective calls and storage level 0.

Each timed fit creates a fresh callback and solver state, runs core MIGRAD,
and returns parameters, errors, covariance and fit status. Timing starts after
compilation and data preparation. The C++ entry point is
`VariableMetricMinimizer::Minimize`.

For the two-parameter Gaussian likelihood, both implementations compute
$\log(\sigma)$ and $\sigma^2$ once per objective call, then reuse those values
for every observation. Here $\sigma$ is the Gaussian width.

The benchmark cases use a four-parameter quadratic (`quad_4d`), Rosenbrock
functions with two and ten parameters (`rosenbrock_2d`, `rosenbrock_10d`),
and Gaussian negative log-likelihoods with two parameters and 100 observations
(`gauss_ll_2_100`) and ten parameters and 1,000 observations
(`gauss_ll_10_1000`).

The table reports medians over 10 pairs of fresh Julia and C++ processes.
Each process contributes the median of six timing blocks after warmup. For
each pair, divide the Julia time by the C++ time. The last two columns show
the median and the range of those 10 ratios. Values below 1 indicate shorter
Julia times.

| Benchmark | Julia (μs) | C++ (μs) | Median Julia/C++ | Observed ratio range |
|---|---:|---:|---:|---:|
| `quad_4d` | 0.93 | 4.2 | 0.22 | 0.21–0.24 |
| `rosenbrock_2d` | 8.3 | 48 | 0.17 | 0.17–0.19 |
| `rosenbrock_10d` | 54 | 200 | 0.27 | 0.24–0.30 |
| `gauss_ll_2_100` | 5.8 | 16 | 0.35 | 0.31–0.44 |
| `gauss_ll_10_1000` | 31 | 38 | 0.82 | 0.80–0.84 |

The [measurement record](benchmark/results/readme_080_macos.json) contains the
per-pair timings, fit results and source versions. The
[benchmark protocol](benchmark/readme_comparison.md) gives reproduction steps.
See [Real-world physics fits](#real-world-physics-fits) for examples with physics
data.

## Features

### Minuit2 algorithms (audited against C++ Minuit2)

- **MIGRAD** — Variable-Metric (DFP) with a central-difference numerical
  gradient, with optional user-supplied or AD gradients.
- **HESSE** — full numerical Hessian + Bunch–Kaufman inversion +
  positive-definite enforcement.
- **MINOS** — asymmetric ±σ errors via `MnFunctionCross` parabolic root-find
  with inner re-minimization.
- **MnContours** — profiled two-parameter confidence contours. Use `profile`
  and `mnprofile` for one-dimensional scans.
- **Simplex** and **Scan** minimizers.
- **Bounds, fixed parameters, and strategy levels 0/1/2** — Minuit2's bounded
  parameter transforms. The objective receives parameters in physical
  coordinates. The default strategy is `Strategy(1)`.

### iminuit / IMinuit.jl-compatible front end

`m.values`, `m.errors`, `m.covariance`, `m.merrors`, `migrad!`, `hesse!`,
`minos!`, `mncontour`, per-parameter `fix!`/`set_limits!`/…, named-parameter
access, and Jupyter-first rich output (`to_latex`, HTML tables, plot recipes).

### Julia-native cost functions

`LeastSquares`, `UnbinnedNLL`, `BinnedNLL`, `ExtendedUnbinnedNLL`,
`ExtendedBinnedNLL` compose with `CostSum` (`+`). Each cost supplies the
error definition (`errordef`) used by HESSE and MINOS. The costs interoperate
with IMinuit.jl's `chisq` and `Data` helpers.

### Error analysis beyond HESSE and MINOS

- **Derived-quantity intervals and profile bands** (`extremize`, `profile_band`)
  find the range of a scalar quantity over a confidence region. Use
  `profile_band` to compute that range pointwise for a model curve.
  `mode=:directional` uses a linearized search direction. The default
  `:full` mode searches the objective boundary.
- **Monte Carlo confidence regions** (`get_contours_samples`) sample the region
  defined by the objective threshold and capture non-Gaussian shapes and
  correlations across parameters.
- **Likelihood-ensemble MCMC** (`mcmc_sample`) samples using the fitted
  objective and applies parameter limits by rejection. Use `quantiles` and
  `quantile_band` for marginal intervals and pointwise bands of derived
  quantities. Save and reload an ensemble with `save_ensemble` and
  `load_ensemble`.
- **Bayesian posterior analysis** (`bayesian`, `posterior_sample`) combines the
  likelihood with a prior and leaves the fit state intact. Use
  `credible_interval`, `upper_limit` and `lower_limit` to summarize posterior
  samples. Multi-chain diagnostics include split R-hat and effective sample
  size (ESS). Choose random-walk Metropolis (`:metropolis`), affine-invariant
  ensemble sampling (`:stretch`, the default), or NUTS (`:nuts`) for smooth
  objectives.
- **Bootstrap and jackknife** (`bootstrap`, `jackknife`) estimate uncertainties
  and covariance by resampling the data.
- **Multiple minima** (`find_solution_modes`) clusters accepted samples in
  scaled parameter coordinates and can refit each cluster.
- **Search for a lower minimum** (`find_deeper_minimum`) refits from perturbed
  parameters or resampled data and retains valid fits with lower objective
  values.

For an objective with several minima, first search for lower valid fits with
`find_deeper_minimum` or multiple starting points and
`find_solution_modes(…; refine=true)`. Then evaluate uncertainties around the
selected minimum with HESSE, MINOS or `get_contours_samples`. A bootstrap or
jackknife refit can move to a different minimum. Its spread then includes
differences between minima.

See the [error-analysis guide](docs/src/error_analysis.md) and the worked
examples for [multi-basin IAM fits](BenchmarkExamples/IAM_2Pformfactor/error_crosscheck.jl),
[searching for lower minima](BenchmarkExamples/IAM_2Pformfactor/find_deeper_minimum_demo.jl)
and the [X(3872) line shape](BenchmarkExamples/X3872_dip/error_crosscheck.jl).

### Alternative minimizers

`optim(m)` bridges to any [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
optimizer (LBFGS/BFGS/NelderMead/Newton/…) and writes the result back into the
`Minuit` so you can follow with `hesse!`/`minos!` — the Julia-native analogue of
iminuit's `Minuit.scipy()`. Loads on `using Optim` (package extension).

## Migrating from iminuit / IMinuit.jl

| iminuit / IMinuit.jl | NativeMinuit |
|---|---|
| `Minuit(fcn, x0; ...)` | same — `Minuit(fcn, x0; names, errors, limits, ...)` |
| `m.migrad()` / `migrad(m)` | `migrad!(m)` |
| `m.hesse()` / `m.minos()` | `hesse!(m)` / `minos!(m)` |
| `m.values`, `m.errors`, `m.covariance` | same |
| `m.covariance.correlation()` / `matrix(f; correlation=true)` | `correlation(m)` or `matrix(m; correlation=true)` |
| `m.mncontour(a, b)` | `mncontour(m, a, b)` |
| IMinuit.jl `Fit`, `ArrayFit` | exported aliases of `Minuit` |
| IMinuit.jl `chisq`, `Data` | exported, same signatures |
| `m.scipy(method=...)` | `optim(m; method=...)` (needs `using Optim`) |

## Gradient options

### Automatic differentiation with ForwardDiff

Pass a gradient through `grad=` so `Minuit` uses it during the fit.

```julia
using NativeMinuit, ForwardDiff

chi2(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
x0 = [0.0, 0.0]
m = Minuit(chi2, x0; grad = x -> ForwardDiff.gradient(chi2, x))
migrad!(m)
```

Write the objective to accept ForwardDiff's parameter element type. Let Julia
infer the element type of complex values from their components and allocate
scratch with `similar(x)` or an explicit type based on `eltype(x)`. For the
low-level API, use `CostFunctionAD(chi2)` to construct the objective and its
gradient together.

### Threaded numerical gradients

Start Julia with multiple threads (for example, `julia -t 4`) and choose
`threaded_gradient` when constructing a fit with numerical gradients.
Threaded evaluation requires an objective that supports concurrent calls.

| `threaded_gradient` | Behavior |
|---------------------|----------|
| `false` (default) | Evaluate the numerical gradient serially. |
| `true` | Evaluate coordinate differences concurrently. With `verify_threading=true` (the default), compare serial and threaded gradients at the starting point and raise `ThreadSafetyError` if they differ. |
| `:auto` | Compare the gradients at the starting point and cache the result for this fit. Agreement within the verification tolerance selects threading. A mismatch issues a warning and selects serial evaluation. Other errors from the objective propagate. |

For the same objective and starting point, select threaded numerical gradients
as follows.

```julia
m = Minuit(chi2, x0; threaded_gradient = true)
migrad!(m)
```

**Thread-safety contract.** Give each objective call its own mutable scratch,
or keep reusable scratch private to its task. A task may yield and resume on a
different thread. Give each concurrent fit its own `CostFunction` or `Minuit`
object. Use `NativeMinuit.is_thread_safe(cf, x0)` to compare serial and
threaded gradients at a chosen starting point.

For an objective with expensive evaluations, compare the elapsed time for a
complete fit with and without threading on the intended inputs.

## Real-world physics fits

The examples below were benchmarked on 31 May 2026 at commit `41226f5`,
using Julia 1.12 on macOS with eight Julia threads and one BLAS thread.
The [benchmark report](BenchmarkExamples/RESULTS.md) gives the configurations
and timings.

- **X(3872) dip line shape.** The fit has three parameters and uses coupled
  channels. Median MIGRAD plus HESSE times were 4.7 ms for NativeMinuit with
  ForwardDiff and 7.4 ms for iminuit. Adding MINOS for all three parameters
  gave median totals of 72.8 ms and 154.7 ms, respectively.
  [Published analysis](https://doi.org/10.1103/PhysRevD.109.L111501)
- **IAM phase-shift fit.** The inverse-amplitude-method example has seven free
  parameters and several minima. In the recorded cold-start comparison with
  `Strategy(1)` and retries, NativeMinuit returned a valid fit and iminuit
  returned an invalid fit. The example also shows how to diagnose a shared
  workspace in a function used for a threaded gradient.
  [IAM analysis](https://doi.org/10.1007/JHEP04(2021)086)

## Reliability

- **Package and numerical tests** include Aqua checks, selected `@inferred`
  checks and regression cases. Run `julia --project=. -e 'using Pkg; Pkg.test()'`.
- **Compiler diagnostics** run in a pinned JET environment.
  [JET diagnostics](docs/dev/JET.md)
- **C++ Minuit2 reference cases** cover unbounded quadratic and Rosenbrock
  objectives, bounded parameter transforms and fixed parameters.
  [Reference tests](test/test_cpp_oracle.jl)
- **Minuit2 source comparison.** The audit records how the port implements
  minimization and error analysis from Minuit2 6.24.0.
  [Audit](docs/dev/CPP_FIDELITY_AUDIT.md)

## Documentation

- **[Manual](https://fkguo.github.io/NativeMinuit.jl/dev)** — tutorials (quickstart,
  bounded parameters, MINOS & contours), error analysis, cost functions, and the
  full API reference.
- **[Error-analysis guide](docs/src/error_analysis.md)** — which uncertainty method
  to use, when, and why (HESSE / MINOS / MC-Δχ² / MCMC ensemble / Bayesian
  posterior / bootstrap / jackknife / multi-modal).
- **[`docs/dev/`](docs/dev/)** — development-history archive: design notes, the
  C++-fidelity audit, the original roadmap, and the deferred-features list — a
  point-in-time snapshot from the v0.3 era, kept for provenance (see the
  [CHANGELOG](CHANGELOG.md) for everything since).
- **[`docs/UPSTREAM.md`](docs/UPSTREAM.md)** — upstream provenance and LGPL
  attribution.

## Using NativeMinuit with an AI coding agent (agent skill)

This repository ships an **agent skill** that teaches an AI coding agent the
NativeMinuit API — the `Minuit` / `migrad!` / `minos!` workflow, the Julia-native cost
functions, bounds and fixed parameters, AD & threaded gradients, and the
error-analysis tools (`mncontour`, `get_contours_samples`, `mcmc_sample` /
`quantile_band`, `bayesian` / `posterior_sample`, `bootstrap` / `jackknife`,
`find_deeper_minimum`, …).
The skill supplies API examples and conventions for fitting and error analysis,
with links to the docstrings and manual for details. Give your coding agent
the [NativeMinuit skill](skills/nativeminuit-usage/SKILL.md).

To use the skill with Codex, create a symlink in its documented user skill
directory. Run this from the NativeMinuit repository root:

```sh
skill_source="$PWD/skills/nativeminuit-usage"
skill_target="$HOME/.agents/skills/nativeminuit-usage"
if [ -f "$skill_source/SKILL.md" ] && [ ! -e "$skill_target" ] && [ ! -L "$skill_target" ]; then
    mkdir -p "$HOME/.agents/skills" && ln -s "$skill_source" "$skill_target"
fi
```

The command creates the link at an unused destination. Request
`$nativeminuit-usage` in Codex to use the skill for a task.
[Codex skill discovery and invocation](https://learn.chatgpt.com/docs/build-skills)

## Citation

Please cite the NativeMinuit.jl release you used and the original MINUIT paper.
The [`CITATION.cff`](CITATION.cff) file supplies the software citation for
GitHub's **Cite this repository** button in APA and BibTeX.

> F.-K. Guo (2026). *NativeMinuit.jl: a native-Julia port of Minuit2*
> (v0.8.0). Zenodo. [doi:10.5281/zenodo.24475298](https://doi.org/10.5281/zenodo.24475298)
>
> F. James and M. Roos, "MINUIT: A system for function minimization and analysis
> of the parameter errors and correlations", Comput. Phys. Commun. **10** (1975)
> 343–367. [doi:10.1016/0010-4655(75)90039-9](https://doi.org/10.1016/0010-4655(75)90039-9)

## Acknowledgements

NativeMinuit.jl is developed by Feng-Kun Guo and
[contributors](https://github.com/fkguo/NativeMinuit.jl/graphs/contributors).

- **Mikhail Mikhasenko** contributed support for structured parameter vectors
  and HESSE errors.
- **C++ Minuit2** by M. Winkler, F. James, L. Moneta, A. Zsenei (CERN PH/SFT,
  2003–) — the algorithmic basis. This Julia port is a derivative work.
- **[IMinuit.jl](https://github.com/fkguo/IMinuit.jl)** (Feng-Kun Guo, Yu Zhang)
  — the Julia wrapper this package complements and can replace.
- **[iminuit](https://github.com/scikit-hep/iminuit)** (Hans Dembinski,
  scikit-hep) — the Python wrapper whose API NativeMinuit mirrors.
- **AI coding agents** assisted with drafting and review of the port, its
  tests and its documentation.
