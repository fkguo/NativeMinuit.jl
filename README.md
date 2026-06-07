# JuMinuit.jl

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://fkguo.github.io/JuMinuit.jl/dev)
[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL%20v2.1%2B-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FJuMinuit&query=total_requests&suffix=%2Fmonth&label=downloads&color=brightgreen)](https://juliapkgstats.com/pkg/JuMinuit)

Native-Julia port of the C++ [Minuit2](https://root.cern.ch/doc/master/Minuit2Page.html)
function-minimization library — the workhorse of every HEP fit. JuMinuit is a
drop-in replacement for [IMinuit.jl](https://github.com/fkguo/IMinuit.jl) (the
Julia wrapper of the Python [iminuit](https://github.com/scikit-hep/iminuit)),
with an iminuit-style
API, **C++-comparable (often better) performance**, and error-analysis tools
that go beyond what either offers.

License: **LGPL 2.1 or later** (mirrors upstream Minuit2). This is a derivative
work of C++ Minuit2 — see [`LICENSE`](LICENSE) and [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

[![IAM fit (Binder)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fkguo/JuMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FIAM_2Pformfactor%2Fiamfit.ipynb)
[![X(3872) dip (Binder)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fkguo/JuMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FX3872_dip%2FXdip_published.ipynb)

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/fkguo/JuMinuit.jl")   # until registered in General
```

JuMinuit needs no compiled dependencies — it is pure Julia.

## Quick start

```julia
using JuMinuit

# iminuit-style API
m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
           [0.0, 0.0];
           names  = ["a", "b"],
           errors = [0.1, 0.1],
           limits = [(-5.0, 5.0), nothing])
migrad!(m)
minos!(m)

println(m.values)        # ≈ [1.0, 2.0]
println(m.errors)        # parabolic (HESSE) 1σ errors
println(m.fval)          # ≈ 0.0
println(m.minos_errors)  # asymmetric ±σ per parameter
m                        # rich table (HTML in Jupyter, text in the REPL)
```

`Fit` and `ArrayFit` are exported as IMinuit.jl-compatible aliases, so existing
IMinuit.jl scripts largely work unchanged.

## Features

### Minuit2 algorithms (ported with line-by-line C++ fidelity)

- **MIGRAD** — Variable-Metric (DFP) with a central-difference numerical
  gradient. Faster than C++ Minuit2 on every benchmark in the test corpus.
- **HESSE** — full numerical Hessian + Bunch–Kaufman inversion +
  positive-definite enforcement.
- **MINOS** — asymmetric ±σ errors via `MnFunctionCross` parabolic root-find
  with inner re-minimization.
- **MnContours** — exact multi-parameter confidence contours (not just the
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
common in coupled-channel / amplitude fits), JuMinuit adds:

- **Monte-Carlo Δχ² regions** (`get_contours_samples`) — sample the true
  `Δχ² ≤ delta_chisq(cl, ndof)` region; captures non-Gaussian and joint
  multi-parameter shapes. Over-coverage-aware (inflation, adaptive widening,
  covariance-free box proposal).
- **Bootstrap** and **jackknife** (`bootstrap`, `jackknife`) — data-resampling
  errors that don't trust the quoted `σ`; with full covariance + `correlation`.
- **Multi-modal solution detection** (`find_solution_modes`) — cluster the
  accepted samples (in whitened/Mahalanobis coordinates) into **statistically
  distinct solutions**, with optional per-mode re-fit and a "deeper-than-global"
  flag. Detects when a fit has several physically different solutions of
  comparable χ² that a single error bar would hide.
- **Escaping a local basin** (`find_deeper_minimum`) — a basin-hopping search
  that climbs out of the basin a single MIGRAD lands in, by **parameter
  perturbation** (any objective) or **data resampling** (bootstrap-driven —
  stronger on hard multi-basin data fits, automating the multi-start +
  `find_solution_modes(…; refine=true)` loop in one call). Adopts any **deeper**
  valid minimum until no round improves. A heuristic — it finds *a* deeper
  minimum, not a *certified global* one — but a useful step toward the true
  minimum before quoting errors.

**On multi-basin surfaces** (ill-conditioned coupled-channel fits), do two things
in order. **(1) Find the true minimum** — a single MIGRAD only reaches the basin
its start drains into, so use `find_deeper_minimum`, or multi-start +
`find_solution_modes(…; refine=true)` (whose `new_min` flag marks a re-fit
*deeper* than the current best). **(2) At that minimum, trust the _local_ error
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

| iminuit / IMinuit.jl | JuMinuit |
|---|---|
| `Minuit(fcn, x0; ...)` | same — `Minuit(fcn, x0; names, errors, limits, ...)` |
| `m.migrad()` / `migrad(m)` | `migrad!(m)` |
| `m.hesse()` / `m.minos()` | `hesse!(m)` / `minos!(m)` |
| `m.values`, `m.errors`, `m.covariance` | same |
| `m.mncontour(a, b)` | `mncontour(m, a, b)` |
| IMinuit.jl `Fit`, `ArrayFit` | exported aliases of `Minuit` |
| IMinuit.jl `chisq`, `Data` | exported, same signatures |
| `m.scipy(method=...)` | `optim(m; method=...)` (needs `using Optim`) |

The API mirrors iminuit where it makes sense and leans on Julia's strengths
(generic FCNs, multiple dispatch, package extensions) where that is better.

## Beyond C++ Minuit2 — Julia-only gradient options

Two capabilities flow directly from Julia's generic-function dispatch and
lightweight threading — useful when the FCN is expensive or contains
complex-valued intermediates (amplitudes, propagators). C++ Minuit2's
`MnFcn::operator()` is a **virtual** function locked to `double`, so neither
generic AD nor zero-copy threading is possible there.

### 1. AD gradients via the ForwardDiff extension

Load `ForwardDiff` (or any AD that returns `Vector{Float64}`) and the gradient
routes through AD end-to-end — MIGRAD, MINOS, and contour boundaries all use it.

```julia
using JuMinuit, ForwardDiff     # extension auto-activates

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
| `false` *(default)* | serial gradient — always safe, zero overhead. |
| `true` | force the threaded gradient; on the first call it auto-verifies thread-safety and raises `ThreadSafetyError` if the FCN is not thread-safe (see the contract below). |
| `:auto` | probe thread-safety **once** at the seed (memoized on the fit); if the probe passes, use the threaded gradient, otherwise emit a single `@warn` and fall back to the serial gradient. Never throws. The probe is a best-effort single-point check — it reliably catches the common shared-buffer race, but a race that only manifests away from the seed can slip through (use `threaded_gradient=true` for the strict per-call check). On single-thread Julia (`-t 1`) it is silently serial (no probe); no-op for AD (`grad=`) fits. |

The default stays `false` because threading only pays off for expensive
FCNs at higher `n`; `:auto` is the opt-in "thread it safely without me
checking" switch.

```julia
m = Minuit(my_chi2, x0; error = errs, threaded_gradient = true)
migrad!(m)             # threading propagates through MINOS / contours too
```

The win scales with FCN cost and parameter count: an expensive FCN at higher
`n` benefits most, a sub-µs FCN not at all (threading overhead dominates), and a
**thread-unsafe FCN is refused outright** (see the contract below).

> **⚠ Thread-safety contract.** Your FCN must not share mutable state across
> threads (module-level scratch buffers, RNG, file I/O). The classic HEP
> anti-pattern is a `const T_BUF = zeros(ComplexF64, …)` mutated inside the FCN:
> parallel calls race on it and MIGRAD silently converges to the **wrong**
> minimum. JuMinuit ships a safety net — `threaded_gradient=true` auto-verifies
> the threaded gradient against the sequential one on the first call (raises
> `ThreadSafetyError` with a diagnostic), and `JuMinuit.is_thread_safe(cf, x0)`
> probes it standalone. JuMinuit's own buffers are all per-thread; the contract
> is on your FCN. See the manual for the full treatment and the worked failure
> case (`BenchmarkExamples/IAM_2Pformfactor/`).

**Fixing a thread-unsafe FCN — give each thread its own buffer.** Yes: replacing
a shared `const` scratch with one buffer per thread makes the FCN thread-safe.
Indexing the pool by `Threads.threadid()` is sound here because JuMinuit threads
the gradient with `Threads.@threads :static`, which pins each iteration to a
fixed thread — so `threadid()` is stable within a call (under the `:dynamic` /
`@spawn` schedules it would *not* be). Size the pool with `maxthreadid()`, not
`nthreads()` (Julia may hand out thread ids beyond `nthreads()`):

```julia
# Was:  const c_00_4 = zeros(ComplexF64, 3, 3)        # shared → racy
const C_POOL = [zeros(ComplexF64, 3, 3) for _ in 1:Threads.maxthreadid()]
function St4_00!(par)
    c = C_POOL[Threads.threadid()]                    # this thread's private buffer
    # ... fill and use c ...
end
```

Simpler and always correct (any schedule): **allocate the scratch per call**
inside the FCN — `c = zeros(ComplexF64, 3, 3)`, or
`Matrix{Complex{eltype(par)}}(undef, 3, 3)` to stay AD-generic (a hard-coded
`ComplexF64` buffer can't hold the `Complex{Dual}` values ForwardDiff produces).
For a millisecond-scale FCN that allocation is negligible.
Either way, confirm the fix with `JuMinuit.is_thread_safe(cf, x0)` (or just let
`threaded_gradient=true` auto-verify on the first call).

### When to choose which

```
                          per-FCN cost
                ┌────────────┬───────────────┬───────────────┐
                │ < ~500 ns  │ ~1-50 μs      │ ≥ ~50 μs      │
   ─────────────┼────────────┼───────────────┼───────────────┤
   n ≤ 5        │ numerical  │ AD            │ AD            │
   5 < n ≤ 30   │ numerical  │ AD            │ AD or 8T-num  │
   n > 30       │ numerical  │ 8T-num or AD  │ **8T-num**    │
   ─────────────┴────────────┴───────────────┴───────────────┘
   AD needs the FCN generic on element type.
   8T = threaded_gradient=true under julia -t 8 (any thread-safe FCN).
```

## Performance

Wall time for a full **MIGRAD** minimization vs C++ Minuit2 on five standard
test objectives — the FCN being minimized, *not* a Minuit operation (Apple M3 /
Julia 1.12 / OpenBLAS 0.3.29; `Strategy(0)`, single-threaded BLAS on both sides):

| Benchmark | Julia (μs) | C++ (μs) | Julia / C++ |
|---|---|---|---|
| `quad_4d` | 0.81 | 5.50 | **0.147×** |
| `rosenbrock_2d` | 9.50 | 37.62 | **0.253×** |
| `rosenbrock_10d` | 58.11 | 156.62 | **0.371×** |
| `gauss_ll_10_1000` | 32.26 | 44.96 | **0.718×** |
| `gauss_ll_2_100` | 20.16 | 22.71 | **0.888×** |

The objectives (the user FCN that MIGRAD minimizes — these are standard
optimization test problems, not Minuit operations) are:

- `quad_4d` — a 4-parameter quadratic (smooth and convex — the easy baseline).
- `rosenbrock_2d` / `rosenbrock_10d` — the Rosenbrock "banana" function (a
  curved, narrow valley; the classic hard test for a minimizer) in 2 and 10
  dimensions.
- `gauss_ll_<npar>_<ndata>` — a Gaussian negative-log-likelihood fit with
  `<npar>` free parameters over `<ndata>` data points (a realistic
  maximum-likelihood fit; e.g. `gauss_ll_10_1000` = 10 parameters, 1000 points).

**Why Julia wins**: a parametric `CostFunction{F}` devirtualizes the FCN call
site at compile time, whereas C++ Minuit2 pays for `shared_ptr` ref-counting and
`ABObj` expression-template dispatch. Reproduce: build the C++ side
(`cmake --build benchmark/cpp/build`), generate the Julia baseline
(`scripts/run_gate.sh --save-baseline`), then `julia benchmark/compare_cpp.jl`.

### Real-world physics fits

On actual HEP fits (vs `iminuit` via PyCall; `julia -t 8` except where noted):

- **X(3872) dip line shape** (3 params, J/ψρ + DD̄* coupled channels) — JuMinuit
  with AD gradients runs migrad+HESSE **1.6× faster than iminuit** (4.7 vs 7.4 ms)
  and MINOS **2.1×** faster (72.8 vs 154.7 ms); the numerical path is ~1.2×
  faster too. All schemes reach the published `fval = 0.0174`.
- **The IAM fit** ([fkguo/IAMfit](https://github.com/fkguo/IAMfit)) — an
  **ill-conditioned**, multi-basin Inverse-Amplitude-Method fit (7 free LECs,
  paper-faithful) and the worked **thread-safety** case study above (the
  shared-buffer race). A **robustness stress-test**, not a speed/fval showcase:
  JuMinuit and iminuit are numerically *identical* near a minimum (agree to
  ~10⁻⁹), but on a cold start which basin each reaches is path-sensitive on this
  chaotic surface. Here JuMinuit happens to converge to a *valid* minimum at its
  default (and runs MINOS/contours, which iminuit refuses on its invalid one)
  (details in [`BenchmarkExamples/RESULTS.md`](BenchmarkExamples/RESULTS.md)).
- **Large coupled-channel amplitude fit** — 57 free parameters, from an
  independent unpublished analysis (single-threaded; a heavy, multi-second-per-call
  FCN). The FCN is the **same Julia code** for both backends, so it cancels from
  the comparison — only the optimizer differs. By the metric that reflects that,
  JuMinuit lands on the **same minimum** (Δχ² ≈ 2×10⁻⁵; 55 of 57 free parameters
  agree to <1%, the rest weakly-constrained flat directions) in **nearly the same
  number of MIGRAD evaluations** (7562 vs 7446 — a 1.6% difference): its MIGRAD is
  about as call-efficient as C++ Minuit2's, which is what matters when each
  evaluation is expensive. Wall time is then just `nfcn × (shared FCN cost)` — the optimizer's
  own per-call overhead is negligible against a multi-second FCN, so here the FCN,
  not the optimizer, sets the clock. (The cheap-FCN benchmarks above are where
  that optimizer overhead — and JuMinuit's call-site advantage — actually shows.)

## Reliability

- **Full test suite passes** (3,400+ tests) — including Aqua project-quality
  checks (piracy, compat bounds, stale deps) and `@inferred` type-stability
  assertions on every public entry point. Run
  `julia --project=. -e 'using Pkg; Pkg.test()'`.
- **C++ JSON oracle parity** — reference cases generated by a C++ Minuit2
  harness are asserted in `test/test_cpp_oracle.jl`: unbounded Rosenbrock/Quad,
  bounded sin/upper/lower transforms, fixed parameters.
- **Line-by-line C++-fidelity audit** — every algorithm was diffed against
  upstream Minuit2 v6.24.0 and reviewed against the source; the audit trail and
  resolved findings live in [`docs/dev/`](docs/dev/).

## Documentation

- **[Manual](https://fkguo.github.io/JuMinuit.jl/dev)** — tutorials (quickstart,
  bounded parameters, MINOS & contours), error analysis, cost functions, and the
  full API reference.
- **[Error-analysis guide](docs/src/error_analysis.md)** — which uncertainty method
  to use, when, and why (HESSE / MINOS / MC-Δχ² / bootstrap / jackknife /
  multi-modal).
- **[`docs/dev/`](docs/dev/)** — design notes, the C++-fidelity audit, the
  roadmap, and the explicitly-deferred-features list.
- **[`docs/UPSTREAM.md`](docs/UPSTREAM.md)** — upstream provenance and LGPL
  attribution.

## Using JuMinuit with an AI coding agent (Claude Code skill)

This repository ships a [Claude Code](https://claude.com/claude-code) **skill**
that teaches an AI coding agent the JuMinuit API — the `Minuit` / `migrad!` /
`minos!` workflow, the Julia-native cost functions, bounds and fixed parameters,
AD & threaded gradients, and the error-analysis tools (`mncontour`,
`get_contours_samples`, `bootstrap` / `jackknife`, `find_deeper_minimum`, …).
With it installed, an agent writes **correct fits and error analysis** instead of
guessing the API or falling back to Python-`iminuit` / `IMinuit.jl` syntax. The
skill is a concise quick-reference; its authoritative source is the package's own
docstrings and [`docs/`](docs/), which it points to for depth.

It lives at
[`.claude/skills/juminuit-usage/SKILL.md`](.claude/skills/juminuit-usage/SKILL.md).

- **Working inside this repository:** nothing to do — Claude Code auto-discovers
  the project skill.
- **Using JuMinuit from your own projects:** install it once at the user level so
  every session can use it:

  ```bash
  mkdir -p ~/.claude/skills

  # Option A — copy it in:
  cp -r .claude/skills/juminuit-usage ~/.claude/skills/

  # Option B (instead of A) — symlink it, so a later `git pull` keeps the skill current:
  ln -s "$PWD/.claude/skills/juminuit-usage" ~/.claude/skills/juminuit-usage
  ```

The agent then picks it up automatically whenever a task involves fitting with
JuMinuit (writing a χ²/likelihood fit, running MIGRAD/MINOS, computing contours
or resampling errors, or porting iminuit / IMinuit.jl code).

## Citation

If you use JuMinuit.jl in a publication, please cite **both** JuMinuit.jl and the
upstream Minuit algorithms it ports. JuMinuit.jl ships a
[`CITATION.cff`](CITATION.cff) — use GitHub's **"Cite this repository"** button
for APA / BibTeX, or:

> F.-K. Guo, *JuMinuit.jl: a native-Julia port of Minuit2*,
> v0.3.0, https://github.com/fkguo/JuMinuit.jl (2026).
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
  scikit-hep) — the Python wrapper whose API JuMinuit mirrors.
- **AI coding agents** — portions of the port, its tests, and the documentation
  were drafted and adversarially cross-reviewed with the help of multiple AI
  coding agents. The algorithmic and C++-fidelity decisions, and final
  responsibility for the code, remain the authors'.
