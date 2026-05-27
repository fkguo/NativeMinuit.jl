# JuMinuit.jl

Native-Julia port of the C++ [Minuit2](https://root.cern.ch/doc/master/Minuit2Page.html)
function-minimization library — the workhorse of every HEP fit.
Designed as a drop-in replacement for the
[iminuit](https://github.com/scikit-hep/iminuit) +
[IMinuit.jl](https://github.com/fkguo/IMinuit.jl) stack with
**C++-comparable or better performance** (typically 0.13× to 0.89× C++
Minuit2 wall time on representative §3.3 benchmarks; Phase 0 §3.4 gate
verified).

License: **LGPL 2.1 or later** (mirrors upstream Minuit2).

For functions defined, click
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fkguo.github.io/JuMinuit.jl/dev)

<!-- Binder badge — activates once the repo is public on GitHub.
     mybinder.org needs to clone the repo (currently private → 503).
     Restore by un-commenting the line below when fkguo/JuMinuit.jl
     is made public; the `.binder/` config is already in place.

For interactive examples, click
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fkguo/JuMinuit.jl/main?urlpath=lab%2Ftree%2Fdocs%2Fexample.ipynb)
-->


## Quick start

```julia
using JuMinuit

# iminuit-style API
m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
            [0.0, 0.0];
            names = ["a", "b"],
            errors = [0.1, 0.1],
            limits = [(-5.0, 5.0), nothing])
migrad!(m)
minos!(m)

println(m.values)        # ≈ [1.0, 2.0]
println(m.errors)        # external 1σ errors
println(m.fval)          # ≈ 0.0
println(m.minos_errors)  # asymmetric ±σ per parameter
println(m)               # pretty table
```

## Features

### Core (Phase 0 + 1)

- **MIGRAD** — Variable-Metric (DFP) with central-difference numerical
  gradient. Faster than C++ Minuit2 on every benchmark in the test
  corpus.
- **HESSE** — full numerical Hessian + Bunch–Kaufman inversion +
  positive-definite enforcement.
- **MINOS** — asymmetric ±σ errors via parabolic root-find with inner
  re-minimization.
- **Contours** — 2D 1σ contour (Phase 1 first cut: ellipse approximation
  from MINOS + off-diagonal covariance; multi-parameter `MnFunctionCross`
  for the C++-exact contour is Phase 1.x).
- **Bounds and fixed parameters** — sin/sqrt parameter transformations
  matching C++ Minuit2 exactly; per-parameter `fixed` flags. The user
  FCN always sees external (physical) coordinates.
- **Named parameters** — `m.params`, `m["x"]`-style access from Phase 3.

### Performance (Phase 0 §3.4 Criterion 2)

| Benchmark | Julia (μs) | C++ (μs) | Julia / C++ |
|---|---|---|---|
| `quad_4d` | 1.27 | 9.71 | **0.131×** |
| `rosenbrock_2d` | 17.1 | 63.7 | **0.268×** |
| `rosenbrock_10d` | 98.1 | 270.2 | **0.363×** |
| `gauss_ll_10_1000` | 55.6 | 78.2 | **0.710×** |
| `gauss_ll_2_100` | 34.8 | 39.2 | **0.887×** |

Measured on Apple M3 / Julia 1.12 / OpenBLAS 0.3.29; Strategy(0),
single-threaded BLAS on both sides. Reproduce via
`benchmark/cpp/build/cpp_bench` + `benchmark/compare_cpp.jl`.

**Why Julia wins**: parametric `CostFunction{F}` devirtualizes the FCN
call site at compile time; C++ Minuit2 has overhead from `shared_ptr`
ref-counting and ABObj expression-template dispatch.

### Phase 2

- **AD-backed gradients** (2.1) via `ForwardDiff.jl` or any
  `gradient::Function` callback. Drop-in replacement for central diff —
  typically 5-10× fewer FCN evaluations on cheap FCNs.

  ```julia
  using ForwardDiff
  f = x -> sum(abs2, x .- [1.0, 2.0])
  cf = CostFunctionWithGradient(f, x -> ForwardDiff.gradient(f, x))
  m = migrad(cf, [0.0, 0.0], [0.1, 0.1])
  ```

- **Result serialization** (2.5) via `JuMinuit.to_dict(m)` for
  JSON / JLD2 storage of fit results.

## Beyond C++ Minuit2 — Julia-only features

JuMinuit ships two capabilities that **C++ Minuit2 cannot offer**, both
flowing directly from Julia's generic-function dispatch and lightweight
multithreading. Useful for HEP fits where the FCN is expensive or
contains complex-valued intermediates (amplitudes, propagators).

### Why these are Julia-only

C++ Minuit2's `MnFcn::operator()(const vector<double>&)` is a **virtual
function**. Virtual functions cannot be templated, so the input type is
locked to `double`. Generic AD (`ForwardDiff.Dual`, Enzyme, etc.) cannot
promote through this signature, and threading a single call across
parameters requires hand-written `vector<complex<double>>` overloads
per FCN. **Julia has no such barrier**: user FCNs are generic on
element type by default, so swapping `Vector{Float64}` for
`Vector{Dual{...}}` or running `n` evaluations in parallel requires
zero user-code changes.

### 1. AD gradients via `CostFunctionAD` (Phase F)

Load `ForwardDiff` (or any AD library that returns `Vector{Float64}`)
and the gradient routes through `ForwardDiff.gradient(f, x)` end-to-end
— MIGRAD, MINOS, contour boundaries all use the AD path.

```julia
using JuMinuit, ForwardDiff     # extension auto-activates

# Real parameters, complex amplitude intermediate — typical HEP fit
function chi2(par)
    mass, coupling, width = par
    χ² = 0.0
    for (sᵢ, yᵢ) in data
        # Complex Breit-Wigner — note `complex(...)` keeps type generic
        amp = coupling / (sᵢ - mass^2 - im * mass * width)
        model = abs2(amp)
        χ² += (model - yᵢ)^2
    end
    return χ²
end

cf = CostFunctionAD(chi2, 0.5)   # 0.5 = NLL convention
fmin = migrad(cf, x0, errs)
# or via high-level Minuit API:
m = Minuit(chi2, x0; error=errs, grad = x -> ForwardDiff.gradient(chi2, x))
migrad!(m)
```

**Common pitfall — your FCN must be generic on element type**:

- ❌ `function f(x::Vector{Float64}) ... end` blocks `Dual`
- ✓ `function f(x) ... end`
- ❌ `c::Complex{Float64} = ...` type-locks the intermediate
- ✓ `c = complex(...)` or `c = ... + im * something`
- ❌ Pre-allocated `Vector{Float64}` scratch *inside* `f`
- ✓ `scratch = similar(x, eltype(x))` or allocate fresh per call

If your FCN can't be made generic (mutates Float64 buffers, calls C
libraries, etc.), use option 2 below.

### 2. Threaded numerical gradient (Phase G)

Start Julia with multiple threads (`julia -t N`) and pass
`threaded_gradient=true`. The per-coordinate `for i in 1:n` loop inside
`numerical_gradient!` runs in parallel across `N` threads. Works on
**any** FCN — no type-genericity required, no AD library to wrestle
with type-locked buffers / C callbacks.

```julia
# Start: julia -t 8
using JuMinuit
m = Minuit(my_chi2, x0; error=errs, threaded_gradient=true)
migrad!(m)
mncontour(m, 1, 2)   # threading propagates through MINOS / contour too
```

Measured **~2× speedup** on `julia -t 8` for IAM-style FCN (9 params,
85 μs/call); see `benchmark/IAM_2Pformfactor` for the realistic
benchmark.

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

   AD requires user FCN generic on element type.
   8T = threaded_gradient=true under julia -t 8 (any FCN).
```

**Concrete decision tree** for a typical HEP fit:

1. Does your FCN evaluate in less than ~1 μs? → numerical (default).
   Threading overhead and AD bookkeeping would dominate.
2. Is your FCN written generically (no `::Vector{Float64}` restrictions,
   no `Complex{Float64}` literals)? → try **AD** (`CostFunctionAD(f)`).
3. Does your FCN have non-generic parts you don't want to rewrite (C
   library calls, `quadgk!` with fixed Float64 workspace, mutating
   internal state)? → use **threaded_gradient=true** with `julia -t N`.
4. n > 50 and FCN > 100 μs? → **threaded_gradient=true** wins even over
   AD (parallelism beats Dual-stack overhead at high n).

**Thread-safety contract for option 2**: the user FCN must not mutate
hidden global state (RNG, file I/O, shared caches). Pre-allocated
buffers inside the FCN's closure are fine if they're per-call (e.g.,
`s = 0.0` reduction variables); avoid module-level mutable scratch.

### Reliability

- **888/888 tests pass** (Aqua + JET clean).
- **C++ JSON oracle parity**: 7 reference cases generated by the C++
  Minuit2 harness, asserted in `test/test_cpp_oracle.jl` — covers
  unbounded Rosenbrock/Quad, bounded Sin/upper/lower, fixed parameters.
- **Four rounds of parallel multi-agent review** (codex gpt-5.5 xhigh
  + native Opus subagent) caught a real `sum_sym` blocking bug
  (covariance computation), a covariance asymmetric-read bug, an
  alpha-convention sign bug in contours, an internal/external
  coord-frame leak in bounded MINOS, and 8+ other surgical issues.
  All applied as commits with explicit source-cited diffs.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0: PoC | ✅ Done | MIGRAD beats C++ on every §3.3 benchmark |
| 1: Bounds + MINOS + Contours + HESSE | ✅ First cut | All algorithms in; Phase 1.x deeper parity below |
| 2.1: AD-backed gradients | ✅ Done | ForwardDiff via package ext (`CostFunctionAD`), threads MINOS+contour |
| 2.2: Threads-parallel gradient | ✅ Done | `threaded_gradient=true` opt-in, ~2× on `julia -t 8` for 50+ μs FCN |
| 2.3: Plot recipes | ⏳ Deferred | |
| 2.4: PrecompileTools | ⏳ Deferred | |
| 2.5: Result serialization | ✅ Done | `to_dict` + `from_dict` |
| 3: iminuit-style Minuit wrapper | ✅ First cut | `m.values`, `m.errors`, etc. |

See `ROADMAP.md` for the full per-phase plan and `docs/DEFERRED.md`
for explicitly-deferred features.

## Layout

```
src/
  JuMinuit.jl              # top-level
  precision.jl, strategy.jl
  state.jl                 # MinimumState, CovStatus, …
  fcn.jl, ad_gradient.jl   # CostFunction, CostFunctionWithGradient
  linalg.jl, gradient.jl   # symmetric BLAS + numerical gradient
  davidon.jl, edm.jl       # DFP update + EDM
  posdef.jl, negative_g2.jl
  linesearch.jl, seed.jl   # parabolic line search + MnSeedGenerator
  migrad.jl, result.jl     # MIGRAD loop + FunctionMinimum
  hesse.jl                 # full MnHesse
  transform.jl             # sin/sqrt bound transforms
  parameters.jl            # MinuitParameter + Parameters
  function_cross.jl, minos.jl, contours.jl
  covariance_squeeze.jl
  migrad_bounded.jl        # bound-aware MIGRAD wrapper
  minuit.jl                # iminuit-style Minuit struct
  serialize.jl             # to_dict / from_dict
test/                      # 888 tests; Aqua + JET clean
benchmark/                 # julia-perf + C++ wall-time comparison
tools/                     # C++ Minuit2 harness (cpp_trace_harness.cxx)
docs/                      # DESIGN.md, ROADMAP.md, etc.
reference/Minuit2_cpp/     # upstream C++ source (gitignored;
                           # pinned to GooFit/Minuit2 @ 57dc936 = v6.24.0)
```

## Reproducing the gate

```bash
# Phase 0 §3.4 Criterion 2 — Julia vs C++ wall time
scripts/run_gate.sh --save-baseline       # first run installs baseline
cd benchmark/cpp && mkdir -p build && cd build && cmake .. && make
cd ../../.. && julia benchmark/compare_cpp.jl
# Should print: Verdict: PASS, max ratio ≤ 0.89×
```

## Acknowledgements

- **C++ Minuit2** by M. Winkler, F. James, L. Moneta, A. Zsenei
  (CERN PH/SFT, 2003–) — the algorithmic basis. This Julia port is a
  derivative work.
- **[IMinuit.jl](https://github.com/fkguo/IMinuit.jl)** (Feng-Kun Guo)
  — the Julia wrapper this port replaces.
- **[iminuit](https://github.com/scikit-hep/iminuit)** (Hans Dembinski,
  scikit-hep) — the Python wrapper whose API Phase 3 mirrors.
