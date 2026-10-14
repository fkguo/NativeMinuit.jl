# NativeMinuit.jl

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fkguo.github.io/NativeMinuit.jl/dev)
[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL%20v2.1%2B-blue.svg)](LICENSE)
[![Julia General](https://img.shields.io/badge/Julia%20General-registered-9558B2.svg)](https://github.com/JuliaRegistries/General/tree/master/N/NativeMinuit)
[![Pkg requests](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fpackages%2FNativeMinuit%2Fstats%3Ftimeframe%3Dall%26user_data%3Dtrue%26ci_data%3Dtrue%26missing_data%3Dtrue&query=timeframe_requests&label=Pkg%20requests&color=brightgreen)](https://juliapkgstats.com/pkg/NativeMinuit?timeframe=all&trendingPeriod=14d&userData=true&ciData=true&missingData=true)
[![Binder: IAM fit](https://img.shields.io/badge/Binder-IAM%20fit-F37626?logo=jupyter)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FIAM_2Pformfactor%2Fiamfit.ipynb)
[![Binder: X(3872) dip](https://img.shields.io/badge/Binder-X%283872%29%20dip-F37626?logo=jupyter)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FX3872_dip%2FXdip_published.ipynb)

Native-Julia port of the C++ [Minuit2](https://root.cern.ch/doc/master/Minuit2Page.html)
function-minimization library — the workhorse of every HEP fit. NativeMinuit is a
drop-in replacement for [IMinuit.jl](https://github.com/fkguo/IMinuit.jl) (the
Julia wrapper of the Python [iminuit](https://github.com/scikit-hep/iminuit)),
with an iminuit-style
API and error-analysis tools that
go well beyond either — profile-likelihood confidence contours, Monte-Carlo Δχ² regions, a
likelihood-ensemble MCMC, **Bayesian posterior analysis** (priors & credible
intervals, three samplers), bootstrap / jackknife, and multi-modal detection.

License: **LGPL 2.1 or later** (mirrors upstream Minuit2). This is a derivative
work of C++ Minuit2 — see [`LICENSE`](LICENSE) and [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

## Installation

```julia
using Pkg
Pkg.add("NativeMinuit")
```

NativeMinuit requires **Julia 1.11 or later**. The minimizer is implemented in
Julia and does not require a Python or C++ Minuit2 installation.
Optional integrations are installed separately and activate when loaded:

| Capability | Install and load |
|---|---|
| AD factory | `ForwardDiff` |
| Alternative minimizers | `Optim` |
| Plots / drawing helpers | `Plots` |
| DataFrame output | `DataFrames` |
| NUTS posterior sampler | `AdvancedHMC`, `LogDensityProblems`, `LogDensityProblemsAD`, `TransformVariables`, `ForwardDiff` (all five) |

For example, run `Pkg.add("ForwardDiff")` and then
`using NativeMinuit, ForwardDiff` to enable `CostFunctionAD`. Loading it does
not automatically change a `Minuit` fit to AD; supply `grad=` or use the factory.

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

C++ Minuit2 gives you MIGRAD / HESSE / MINOS / MnContours / Simplex / Scan with
bounds, fixed parameters, and Strategy levels — all ported here. On top of that,
NativeMinuit adds, in pure Julia:

- **iminuit / IMinuit.jl-style front end** — `Minuit`, `migrad!`, `minos!`, `m.values` / `m.errors` / `m.merrors`, … (drop-in for IMinuit.jl).
- **Julia-native cost functions** — `LeastSquares`, `UnbinnedNLL`, `BinnedNLL`, `ExtendedUnbinnedNLL`, `ExtendedBinnedNLL`, and their `CostSum`.
- **Automatic-differentiation gradients** (ForwardDiff) and a **threaded numerical gradient**.
- **Derived-quantity intervals & error bands** — `extremize` / `profile_band`: MINOS for any scalar `f(θ)`, with a fast `:directional` mode.
- **Monte-Carlo Δχ² confidence regions** — `get_contours_samples` (the true non-Gaussian / joint region, not just the ellipse).
- **Likelihood-ensemble MCMC** — `mcmc_sample` + `quantile_band` (marginal intervals & curve bands on the exact FCN).
- **Bayesian posterior analysis** — `bayesian` / `posterior_sample`: priors, credible intervals & limits, and three samplers (random-walk, affine-invariant ensemble, NUTS).
- **Bootstrap & jackknife** — `bootstrap` / `jackknife` (data-resampling errors).
- **Multi-modal solution detection** — `find_solution_modes` (cluster statistically distinct minima).
- **Basin escape** — `find_deeper_minimum` (basin-hopping to a deeper minimum).
- **Alternative minimizers** — `optim` / `minimize_with` (Optim.jl integration).
- **Rich output** — `to_latex` publication tables, Plots recipes, Unicode/HTML auto-display, ASCII plots.

Each is detailed in the sections below.

## Performance

Measured on **2026-10-14**, using NativeMinuit **0.7.5** at
[`68de553c`](https://github.com/fkguo/NativeMinuit.jl/tree/68de553cbccf1dd4f83949cea7b9e6d83df982fb)
and C++ Minuit2 **6.24.0**: Apple M3, macOS 15.6.1, Julia 1.12.6,
OpenBLAS 0.3.33, Apple clang 17 (`-O3 -DNDEBUG`), one Julia thread and one
actual BLAS thread. Both use the same fixed inputs, numerical gradients,
`Strategy(0)`, tolerance 0.1, at most 10000 calls, and storage level 0.

The timed operation constructs a fresh callback and solver state, runs full
**core MIGRAD**, and materializes parameters, errors, covariance and status.
Immutable data preparation, compilation and optional global correlations are
excluded on both sides. This compares Julia `migrad` with C++
`VariableMetricMinimizer::Minimize`, rather than the additional work performed
by the C++ `MnMigrad` application.

Five alternating process pairs each collect 50 warmed samples per objective.
Times below are medians of the five process medians; the ratio is the median
of the five **paired** ratios, so it need not equal the quotient of the displayed
times. The range preserves the smallest and largest paired ratios observed.

| Benchmark | Julia (μs) | C++ (μs) | Paired Julia / C++ | Observed range |
|---|---:|---:|---:|---:|
| `quad_4d` | 1.666 | 6.792 | 0.242 | 0.177–0.250 |
| `rosenbrock_2d` | — | — | Not admitted | Strict output comparison failed |
| `rosenbrock_10d` | — | — | Not admitted | Strict output comparison failed |
| `gauss_ll_10_1000` | 50.834 | 64.334 | 0.790 | 0.688–0.793 |
| `gauss_ll_2_100` | 31.667 | 27.605 | 1.150 | 0.874–1.855 |

Only cases passing the predeclared strict parameter, objective, EDM, error and
covariance checks receive timing ratios. Both Rosenbrock results are valid and
pass the existing diagnostic checks, but their terminal outputs fail the strict
comparison; all raw results are retained. The small Gaussian case's range crosses
1, so these measurements do not support a stable faster/slower conclusion for it.
Ranges describe this single-machine measurement, not confidence intervals or
general speed guarantees. See the [protocol and reproduction commands](benchmark/readme_comparison.md)
and [numerical results, raw timings and environment](benchmark/results/readme_075.json).
The previous table used unmatched Gaussian inputs and timing boundaries and has
been replaced rather than reused as a current comparison.

The objectives (the user FCN that MIGRAD minimizes — these are standard
optimization test problems, not Minuit operations) are:

- `quad_4d` — a 4-parameter quadratic (smooth and convex — the easy baseline).
- `rosenbrock_2d` / `rosenbrock_10d` — the Rosenbrock "banana" function (a
  curved, narrow valley; the classic hard test for a minimizer) in 2 and 10
  dimensions.
- `gauss_ll_<npar>_<ndata>` — a Gaussian negative-log-likelihood fit with
  `<npar>` free parameters over `<ndata>` data points (a realistic
  maximum-likelihood fit; e.g. `gauss_ll_10_1000` = 10 parameters, 1000 points).

The parametric Julia cost function can specialize objective calls. The table
alone does not isolate this mechanism or establish the cause of a timing
difference. Benchmarking requires the same objective, requested outputs,
strategy, limits, dependencies and machine, with numerical agreement checked
separately. The sources live in [`benchmark/`](benchmark/) and
[`scripts/`](scripts/).

(Real-world HEP-fit benchmarks vs `iminuit` are further down, under
[Real-world physics fits](#real-world-physics-fits).)

## Features

### Minuit2 algorithms (audited against C++ Minuit2)

- **MIGRAD** — Variable-Metric (DFP) with a central-difference numerical
  gradient, with optional user-supplied or AD gradients.
- **HESSE** — full numerical Hessian + Bunch–Kaufman inversion +
  positive-definite enforcement.
- **MINOS** — asymmetric ±σ errors via `MnFunctionCross` parabolic root-find
  with inner re-minimization.
- **MnContours** — profiled two-parameter confidence contours (not just the
  HESSE ellipse), plus `profile` / `mnprofile`.
- **Simplex** and **Scan** minimizers.
- **Bounds, fixed parameters, and Strategy levels 0/1/2** — the same sin/√
  parameter transforms as C++ Minuit2; the user FCN always sees external
  (physical) coordinates. Defaults match iminuit (`Strategy(1)`, `4·ε`
  machine precision).

### iminuit / IMinuit.jl-compatible front end

`m.values`, `m.errors`, `m.covariance`, `m.merrors`, `migrad!`, `hesse!`,
`minos!`, `mncontour`, per-parameter `fix!`/`set_limits!`/…, named-parameter
access, and Jupyter-first rich output (`to_latex`, HTML tables, plot recipes).

### Julia-native cost functions

`LeastSquares`, `UnbinnedNLL`, `BinnedNLL`, `ExtendedUnbinnedNLL`,
`ExtendedBinnedNLL`, composable with `CostSum` (`+`). Each carries the right
`errordef`, so MINOS/HESSE scaling is automatic. Interoperates with IMinuit.jl's
`chisq` / `Data` helpers.

### Error analysis beyond HESSE and MINOS

When MINOS can't close a contour (flat or strongly non-Gaussian likelihoods —
common in coupled-channel / amplitude fits), NativeMinuit adds:

- **Derived-quantity intervals & profile bands** (`extremize`, `profile_band`)
  — MINOS for an arbitrary scalar `f(θ)` (a peak position, an amplitude ratio,
  a model curve at one energy): a numerical search for `[min, max]` of `f` over the
  `Δχ² ≤ delta_chisq(cl, 1)` region, and the pointwise error band of a whole
  curve family. `mode = :directional` adds a fast linear-Gaussian boundary
  crossing for expensive FCNs; cost and applicability depend on the objective;
  `:full` handles non-linear / multi-corridor regions. The first leg of the
  profile-extremization ↔ ensemble-quantiles ↔ MINOS triangulation.
- **Monte-Carlo Δχ² regions** (`get_contours_samples`) — sample the true
  `Δχ² ≤ delta_chisq(cl, ndof)` region; captures non-Gaussian and joint
  multi-parameter shapes. Over-coverage-aware (inflation, adaptive widening,
  covariance-free box proposal).
- **Likelihood-ensemble MCMC** (`mcmc_sample`) — a Metropolis chain on the
  exact FCN (no Gaussian surrogate; parameter limits by rejection), feeding
  marginal **quantile intervals and pointwise bands of derived quantities**
  (`quantiles`, `quantile_band`) and reusable on-disk error sets
  (`save_ensemble` / `load_ensemble`). The second leg of the
  profile-extremization ↔ ensemble-quantiles ↔ MINOS triangulation; iminuit
  has no native analogue (Python users bolt on emcee).
- **Bayesian posterior analysis** (`bayesian`, `posterior_sample`) — a
  non-mutating layer over the same kernel: `prior × exp(−fcn/(2·up))` sampled in
  external coordinates (`flat`/`normal`/`uniform`/`half_normal` priors), with
  **credible intervals**, one-sided **credible limits** (`credible_interval`,
  `upper_limit`/`lower_limit`), multi-chain split-R̂ / ESS, and a flat prior that
  (single chain, same seed) reproduces the likelihood path byte-for-byte. Three
  samplers (`:stretch` is the default): a random-walk Metropolis (`:metropolis`);
  a gradient-free, affine-invariant **ensemble** (`:stretch`, the Goodman–Weare/emcee kernel —
  works on any FCN, including non-differentiable complex-χ², and handles strongly
  correlated posteriors); and **NUTS** (`:nuts`, via an AdvancedHMC extension —
  gradient-based, for smooth higher-dimensional posteriors, with proper
  unconstraining transforms + log-Jacobian). Credible, not confidence — and it
  never writes into `m.errors`/`m.covariance`/`m.nfcn`.
- **Bootstrap** and **jackknife** (`bootstrap`, `jackknife`) — data-resampling
  errors that don't trust the quoted `σ`; with full covariance + `correlation`.
- **Multi-modal solution detection** (`find_solution_modes`) — cluster the
  accepted samples (in whitened coordinates — Mahalanobis for fit-scale clouds,
  robust cloud-MAD for wide multi-basin scans, picked automatically) into
  **statistically distinct solutions**, with optional budgeted per-mode re-fit
  and a "deeper-than-global" flag. Detects when a fit has several physically
  different solutions of comparable χ² that a single error bar would hide.
- **Escaping a local basin** (`find_deeper_minimum`) — a basin-hopping search
  that climbs out of the basin a single MIGRAD lands in, by **parameter
  perturbation** (any objective) or **data resampling** (bootstrap-driven —
  stronger on hard multi-basin data fits, automating the multi-start +
  `find_solution_modes(…; refine=true)` loop in one call). Adopts any **deeper**
  valid minimum until no round improves. A heuristic — it finds *a* deeper
  minimum, not a *certified global* one — but a useful step toward the true
  minimum before quoting errors.

**On multi-basin surfaces** (ill-conditioned coupled-channel fits), do two things
in order. **(1) Search for deeper valid minima** — a single MIGRAD only reaches the basin
its start drains into, so use `find_deeper_minimum`, or multi-start +
`find_solution_modes(…; refine=true)` (whose `new_min` flag marks a re-fit
*deeper* than the current best). **(2) At the selected valid minimum, inspect the _local_ error
methods** (HESSE / MINOS / `get_contours_samples`). Naive bootstrap/jackknife are
**unreliable** here — each resample re-fits into a possibly-different basin, so
the spread measures the distance *between* basins, not the 1σ error, and there is
no clean "basin-selection" rescue. Worked examples:
[`IAM_2Pformfactor/error_crosscheck.jl`](BenchmarkExamples/IAM_2Pformfactor/error_crosscheck.jl)
(the full multi-basin error study) and
[`find_deeper_minimum_demo.jl`](BenchmarkExamples/IAM_2Pformfactor/find_deeper_minimum_demo.jl)
(`find_deeper_minimum` dropping a cold χ²≈379 fit to χ²≈255, a Δχ²≈124 descent, in
one call); contrast the clean single-basin
[`X3872_dip`](BenchmarkExamples/X3872_dip/error_crosscheck.jl).

See the [error-analysis guide](docs/src/error_analysis.md) for the full comparison
table (which method, when) and worked examples.

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

The API mirrors iminuit where it makes sense and leans on Julia's strengths
(generic FCNs, multiple dispatch, package extensions) where that is better.

## Julia-only gradient options (AD & threading)

Generic Julia objectives can carry ForwardDiff dual numbers and run on Julia
tasks, which is useful for expensive objectives or complex-valued intermediates.
C++ Minuit2 also supports user-supplied gradients through `FCNGradientBase`;
its `double`-valued objective interface is different from ForwardDiff’s direct
evaluation of a Julia objective on dual-number parameters.

### 1. AD gradients via the ForwardDiff extension

Load `ForwardDiff` (or any AD that returns `Vector{Float64}`) and the gradient
routes through AD end-to-end — MIGRAD, MINOS, and contour boundaries all use it.

```julia
using NativeMinuit, ForwardDiff     # extension auto-activates

function chi2(par)
    mass, coupling, width = par
    χ² = 0.0
    for (sᵢ, yᵢ) in data
        amp   = coupling / (sᵢ - mass^2 - im * mass * width)  # complex BW
        model = abs2(amp)
        χ²   += (model - yᵢ)^2
    end
    return χ²
end

m = Minuit(chi2, x0; error = errs, grad = x -> ForwardDiff.gradient(chi2, x))
migrad!(m)
```

> **Your FCN must be generic on element type** for AD to work: write `f(x)` not
> `f(x::Vector{Float64})`, use `complex(...)` rather than `Complex{Float64}`
> literals, and allocate scratch as `similar(x, eltype(x))`. If it can't be made
> generic (mutates Float64 buffers, calls C libraries), use the threaded option.

### 2. Threaded numerical gradient

Start Julia with `julia -t N` and pass `threaded_gradient=true`; the
per-coordinate gradient loop runs in parallel. Works on **any thread-safe FCN**.

`threaded_gradient` is a 3-way switch:

| value | behaviour |
|-------|-----------|
| `false` *(default)* | serial numerical gradient without a parallel safety probe. |
| `true` | force the threaded gradient; with verification enabled, compare serial and threaded gradients at the entry seed and raise `ThreadSafetyError` on a detected mismatch. |
| `:auto` | probe at the seed until a result is obtained, then reuse that result (memoized on the fit); use the threaded gradient if the probe passes, otherwise warn and fall back to serial. Exceptions other than `ThreadSafetyError` propagate. Neither mode proves safety at every input or task interleaving. On single-thread Julia (`-t 1`) no probe runs; AD (`grad=`) fits do not parallelize the gradient. |

The default stays `false` because threading only pays off for expensive
FCNs at higher `n`; `:auto` enables threading when the seed probe passes.

```julia
m = Minuit(my_chi2, x0; error = errs, threaded_gradient = true)
migrad!(m)             # threading propagates through MINOS / contours too
```

The win scales with FCN cost and parameter count: an expensive FCN at higher
`n` can benefit from parallel evaluation. Measure the actual workload; a
seed-point probe can detect a race but cannot establish universal thread safety.

> **⚠ Thread-safety contract.** Your FCN must not share mutable state across
> threads (module-level scratch buffers, RNG, file I/O). The classic HEP
> anti-pattern is a `const T_BUF = zeros(ComplexF64, …)` mutated inside the FCN:
> parallel calls race on it and MIGRAD silently converges to the **wrong**
> minimum. With verification enabled, `threaded_gradient=true` compares
> the threaded gradient against the sequential one at the entry seed (raises
> `ThreadSafetyError` on a detected mismatch), and `NativeMinuit.is_thread_safe(cf, x0)`
> probes it standalone. NativeMinuit's own scratch belongs to the work unit or invocation
> using it; your FCN must also keep mutable scratch private. See the manual for the full treatment and the worked failure
> case (`BenchmarkExamples/IAM_2Pformfactor/`).

**Fixing a thread-unsafe FCN — use private scratch throughout each call.**
Allocate scratch inside the FCN, for example
`Matrix{Complex{eltype(par)}}(undef, 3, 3)` to remain AD-generic. Reusable
scratch must belong exclusively to the calling task or work unit. A pool
indexed by `Threads.threadid()` is unsafe if the objective yields: another
task can overwrite the same slot and tasks can migrate. NativeMinuit uses
dynamic scheduling and supports worker and nested-thread entry.

Use separate `CostFunction`/`Minuit` objects for concurrent fits. Confirm an
objective with `NativeMinuit.is_thread_safe(cf, x0)` or the default
`threaded_gradient=true` verification; the seed-point comparison is a diagnostic
and does not prove safety at every input or task interleaving.

### When to choose which

Use serial numerical gradients as the default. Try ForwardDiff when the
objective is generic over its parameter element type; use threaded numerical
gradients when calls are expensive and simultaneous objective evaluations own
their mutable scratch. Measure both compilation and repeated execution costs
for the objective you actually use; neither parameter count nor call cost alone
fixes the best choice.

## Real-world physics fits

The X(3872) and IAM comparisons below are historical measurements from
2026-05-31 at source commit `41226f5`, on macOS / Julia 1.12,
`julia -t 8` and single-threaded BLAS. Stage medians use three rounds after warmup
(see [`BenchmarkExamples/RESULTS.md`](BenchmarkExamples/RESULTS.md)).
They do not establish the current release’s speed or universal agreement.

- **X(3872) dip line shape** (3 params, J/ψρ + DD̄* coupled channels;
  [arXiv:2404.12003](https://arxiv.org/abs/2404.12003),
  [Phys. Rev. D DOI](https://doi.org/10.1103/PhysRevD.109.L111501)) —
  NativeMinuit with AD gradients runs migrad+HESSE **1.6× faster than iminuit**
  (4.7 vs 7.4 ms) and MINOS **2.1×** faster (72.8 vs 154.7 ms); the numerical
  path is ~1.2× faster too. All schemes reach the published `fval = 0.0174`.
- **The IAM fit** ([fkguo/IAMfit](https://github.com/fkguo/IAMfit);
  [arXiv:2011.00921](https://arxiv.org/abs/2011.00921),
  [JHEP DOI](https://doi.org/10.1007/JHEP04(2021)086)) — an
  **ill-conditioned**, multi-basin Inverse-Amplitude-Method fit (7 free LECs,
  paper-faithful) and the worked **thread-safety** case study above (the
  shared-buffer race). A **robustness stress-test**, not a speed/fval showcase:
  NativeMinuit and iminuit are numerically *identical* near a minimum (agree to
  ~10⁻⁹), but on a cold start which basin each reaches is path-sensitive on this
  chaotic surface. Here NativeMinuit happens to converge to a *valid* minimum at its
  default (and runs MINOS/contours, which iminuit refuses on its invalid one)
  (details in [`BenchmarkExamples/RESULTS.md`](BenchmarkExamples/RESULTS.md)).
Large, expensive amplitude fits should be compared by both objective-call
count and completed wall time, using the same objective and termination
criteria. The cost of a multi-second objective can dominate optimizer overhead.
This is a workload consideration, not a measured current-version speed claim.

## Reliability

- **Package and numerical tests** include Aqua quality checks, selected
  `@inferred` signatures and regression/oracle cases. Test coverage is specific
  to those inputs and is not a proof for every public argument combination. Run
  `julia --project=. -e 'using Pkg; Pkg.test()'`.
- **Compiler diagnostics** run separately in a compatible pinned JET environment;
  an unavailable JET check is not a passing diagnostic. See
  [JET diagnostics](docs/dev/JET.md).
- **C++ JSON oracle parity** — reference cases generated by a C++ Minuit2
  harness are asserted in `test/test_cpp_oracle.jl`: unbounded Rosenbrock/Quad,
  bounded sin/upper/lower transforms, fixed parameters.
- **C++ Minuit2 fidelity audit** — the ported minimization and error-analysis
  routines were compared against upstream Minuit2 v6.24.0 and reviewed against
  the source; the audit trail and resolved findings are archived in
  [`docs/dev/`](docs/dev/).

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
The skill supplies API examples and conventions for fitting and error analysis.
Its use does not replace checking the objective, fit validity or statistical assumptions. The
skill is a concise quick-reference; its authoritative source is the package's own
docstrings and [`docs/`](docs/), which it points to for depth.

It is a plain `SKILL.md` agent skill — **not tied to one tool**: it works with any
coding agent that can read a skill ([Claude Code](https://claude.com/claude-code),
OpenAI Codex, Gemini CLI, …). It lives at
[`skills/nativeminuit-usage/SKILL.md`](skills/nativeminuit-usage/SKILL.md).

To use the skill with Codex, create a symlink in its documented user skill
directory. Run this from the NativeMinuit repository root:

```sh
skill_source="$PWD/skills/nativeminuit-usage"
skill_target="$HOME/.agents/skills/nativeminuit-usage"
if [ -f "$skill_source/SKILL.md" ] && [ ! -e "$skill_target" ] && [ ! -L "$skill_target" ]; then
    mkdir -p "$HOME/.agents/skills" && ln -s "$skill_source" "$skill_target"
fi
```

If that destination already exists, the commands leave it unchanged; inspect
it before replacing an installed skill. Codex can select discovered skills when
the task matches their description, or you can request `$nativeminuit-usage`
explicitly. See [Codex skill discovery and invocation](https://learn.chatgpt.com/docs/build-skills).
For other agents, use their documented skill installation mechanism or explicitly
ask them to read `skills/nativeminuit-usage/SKILL.md`.

## Citation

If you use NativeMinuit.jl in a publication, please cite **both** NativeMinuit.jl and the
upstream Minuit algorithms it ports. NativeMinuit.jl ships a
[`CITATION.cff`](CITATION.cff) — use GitHub's **"Cite this repository"** button
for APA / BibTeX, or:

> F.-K. Guo, *NativeMinuit.jl: a native-Julia port of Minuit2*,
> [NativeMinuit.jl](https://github.com/fkguo/NativeMinuit.jl) (2026).
>
> F. James and M. Roos, "MINUIT: A system for function minimization and analysis
> of the parameter errors and correlations", Comput. Phys. Commun. **10** (1975)
> 343–367. [doi:10.1016/0010-4655(75)90039-9](https://doi.org/10.1016/0010-4655(75)90039-9)

## Acknowledgements

- **C++ Minuit2** by M. Winkler, F. James, L. Moneta, A. Zsenei (CERN PH/SFT,
  2003–) — the algorithmic basis. This Julia port is a derivative work.
- **[IMinuit.jl](https://github.com/fkguo/IMinuit.jl)** (Feng-Kun Guo, Yu Zhang)
  — the Julia wrapper this package complements and can replace.
- **[iminuit](https://github.com/scikit-hep/iminuit)** (Hans Dembinski,
  scikit-hep) — the Python wrapper whose API NativeMinuit mirrors.
- **AI coding agents** — portions of the port, its tests, and the documentation
  were drafted and adversarially cross-reviewed with the help of multiple AI
  coding agents. The algorithmic and C++-fidelity decisions, and final
  responsibility for the code, remain the authors'.
