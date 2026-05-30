# JuMinuit.jl Roadmap

A native-Julia port of the C++ Minuit2 function minimization library
(GooFit/Minuit2; ROOT::Math::Minuit2), targeting drop-in replacement of
the iminuit/IMinuit.jl stack with C++-comparable performance.

> **Compass**: this document is read side-by-side with the C++ reference at
> `reference/Minuit2_cpp/` (pinned to GooFit/Minuit2 @ `57dc936`, v6.24.0 —
> see `docs/UPSTREAM.md`). Filenames cited without a path live under that
> tree. The module-mapping table (§7) is the authoritative porting compass.
>
> **v2** of this document, reconciled from two independent parallel
> reviews against v1.

---

## 1. Project goal

Reimplement, in idiomatic Julia, the Variable-Metric MIGRAD algorithm and
its companions (SIMPLEX, MINOS, HESSE, CONTOURS) that have defined HEP
fitting for forty years, while matching the C++ reference implementation
to ~1e-10 on shared benchmarks. The port eliminates the C++ runtime
dependency that IMinuit.jl currently carries through PyCall/iminuit, gives
Julia-native users automatic differentiation, threaded likelihoods, and
Plots/Makie recipes, and keeps the user-visible API close to iminuit's so
existing fits migrate.

The Variable-Metric MIGRAD loop is a thirty-year-old optimized inner cycle.
We will **port it, not redesign it**. Numerical reproducibility against
the C++ reference is the primary acceptance criterion; performance
follows.

---

## 2. Performance philosophy

Minuit2's hot paths are stereotyped. Knowing where the cycles go in the
C++ code tells us exactly where Julia must be careful.

### 2.1 Where the C++ spends its time

For a fit with `n` free parameters per MIGRAD iteration:

| Hot path                                       | C++ location                                       | C++ cost / iter                                          |
|------------------------------------------------|----------------------------------------------------|----------------------------------------------------------|
| User FCN evaluations                           | `MnUserFcn::operator()` → user lambda              | ~2n + line-search calls (5–15)                          |
| Numerical gradient (central diff, multi-cycle) | `Numerical2PGradientCalculator.cxx:63–230`         | Up to `2·n·Ncycle` FCN calls — **dominant for cheap FCNs** (each coord may break early on tol; see `Numerical2PGradientCalculator.cxx:132–203`) |
| Step = -V·g (DSPMV)                            | `LAVector.h:122–131`, `mndspmv.cxx`                | O(n²) FLOPs, packed sym                                  |
| Inner products `g·g`, `step·g`                 | `LaInnerProduct.cxx`, `mnddot.cxx`                 | O(n)                                                     |
| DFP Hessian update                             | `DavidonErrorUpdator.cxx:24–73`                    | O(n²) — outer products, scalar adds                     |
| EDM = ½ gᵀ V g                                 | `VariableMetricEDMEstimator.cxx`                   | O(n²) (similarity)                                       |
| Line search                                    | `MnLineSearch.cxx`                                 | 4–12 FCN calls, scalar math                              |
| Final HESSE (numerical 2nd derivs)             | `MnHesse.cxx:133–269`                              | Diagonal loop: up to 2 FCN/cycle × up-to-5 step retries per parameter; off-diagonal: 1 FCN per pair — one-shot |

Three regimes determine the optimization target:

- **Cheap FCN** (e.g. Rosenbrock, low-dim quadratic): gradient and
  linear-algebra loops dominate — Julia must match C++ on dense Float64
  BLAS and zero-allocate the inner loop.
- **Moderate FCN** (1k–10k events): FCN cost ≈ gradient overhead — type
  stability of the user closure is critical.
- **Expensive FCN** (large unbinned likelihoods): user code dominates —
  Julia shines (closures specialize), and Threads-parallel reductions
  over data may win outright vs. single-threaded C++.

### 2.2 Concrete Julia idioms (and what they map to in C++)

**Type stability & inference**
- `FCN` is held as a parametric type: `struct CostFunction{F,T}; f::F; up::T; end`
  — never as `Function` (the C++ `FCNBase &` is a vtable call; Julia closure
  call through a concrete type devirtualizes).
- All state structs (`MinimumState`, `FunctionGradient`, `MinimumError`,
  `MinimumParameters`) are concrete, parameterized only on element type
  when needed. No `Any` fields.

**MinimumState as an immutable wrapper**
- C++ uses `shared_ptr<BasicMinimumState>` (`MinimumState.h:66`). Julia
  uses an immutable `struct` whose `Vector`/`Matrix` fields are
  heap-allocated arrays *shared by reference* across iterations. Rebuilding
  the wrapper each iteration costs ~zero (just a pointer copy of the
  fields). **No `mutable struct` needed.** The workspace, by contrast,
  *is* mutable since it holds the scratch buffers.

**In-place linear algebra (zero allocation in the MIGRAD loop)**

The MIGRAD loop in `VariableMetricBuilder.cxx:237–341` allocates many
temporaries that Julia must preallocate to hit the zero-allocation gate.
The explicit scratch inventory (per `MigradWorkspace`) is:

| C++ temporary                         | Julia preallocation                       | Allocated at C++ line                    |
|---------------------------------------|--------------------------------------------|-------------------------------------------|
| `step`                                | `step::Vector{Float64}`                    | Outside loop, `VariableMetricBuilder.cxx:228` |
| `prevStep`                            | `prev_step::Vector{Float64}`               | Outside loop, `VariableMetricBuilder.cxx:235` |
| `g`, `g_prev`                         | `g`, `g_prev::Vector{Float64}`             | Per iter via gradient calculator         |
| `dx = x − x_prev`                     | `dx::Vector{Float64}`                      | `DavidonErrorUpdator.cxx:34`              |
| `dg = g − g_prev`                     | `dg::Vector{Float64}`                      | `DavidonErrorUpdator.cxx:35`              |
| `vg = V · dg`                         | `vg::Vector{Float64}`                      | `DavidonErrorUpdator.cxx:58`              |
| `vUpd` (rank-2 DFP base)              | `vUpd::Matrix{Float64}` (held as `Symmetric` view) | `DavidonErrorUpdator.cxx:60`        |
| `MinimumParameters`, `FunctionGradient`, `MinimumError`, `MinimumState` (per-iter immutable rebuilds) | rebuilt each iter; data arrays *not* copied | `VariableMetricBuilder.cxx:296–329` |
| External parameter vector for FCN call | `x_ext::Vector{Float64}` (reused; Phase 0 trivial since no bounds; Phase 1 swaps in transformation) | `MnUserFcn.cxx:23–29` (C++ allocates per call!) |
| Numerical gradient `vgrd`, `vgrd2`, `vgstep` (cycle scratch) | per-thread scratch in workspace | `Numerical2PGradientCalculator.cxx:121–130` |
| FunctionMinimum result-history vector | `history::Vector{MinimumState}` (configurable; see "Result history" below) | accumulated across iterations |

**Choice of `Matrix{Float64}` + `Symmetric` view vs packed storage** — for
the typical Minuit2 n ≤ 50, dense storage almost certainly wins on modern
CPUs: both packed (`n(n+1)/2`) and dense (`n²`) fit in L1 for n ≤ ~80,
and OpenBLAS DSYMV vectorizes much better than DSPMV. C++ chose packed
in 2003 when memory mattered more than vectorization. **Phase 0 defaults
to dense Symmetric**, with `linalg.jl` keeping a packed variant available
behind a feature flag for benchmark comparison. Decision frozen *after*
end-to-end MIGRAD benchmark, not on day 4–7.

**`Symmetric`/`syr!` caveat**: `BLAS.syr!` operates on `Matrix` not
`Symmetric`. To do an in-place symmetric rank-1 update, call
`BLAS.syr!('U', α, x, parent(S))` and remember which triangle (`'U'` or
`'L'`) the rest of the code reads from. Document this convention once,
in `src/linalg.jl`.

**Replacing C++ ABObj expressions**

The C++ uses `LASymMatrix vUpd = Outer_product(dx)/delgam - Outer_product(vg)/gvg`
which the ABObj expression-template engine fuses into a single in-place
BLAS call (`reference/Minuit2_cpp/inc/Minuit2/LASymMatrix.h:102–123`).
The naive Julia translation `vUpd = (dx*dx')/delgam - (vg*vg')/gvg`
allocates *three* intermediate `Matrix` temporaries. Replace with:

```julia
fill!(vUpd_buf, 0.0)
BLAS.syr!('U', 1/delgam, dx, vUpd_buf)
BLAS.syr!('U', -1/gvg,   vg, vUpd_buf)
```

**Result-history storage policy**

C++ `FunctionMinimum::Add(state)` accumulates every iteration when
`storage_level == 1` (default in MnApplication; `FunctionMinimum.h:148`).
For long fits this is a per-iteration allocation Julia must control:

- Phase 0: `storage_level=0` default (keep only seed + final state).
  Tests requiring full trace use `storage_level=1` explicitly.
- Phase 1: expose `storage_level::Int` keyword on `migrad`; document
  the per-iter allocation cost when `storage_level=1`.

**Small-dimension specialization**
- Profile-driven: if a meaningful fraction of usage is n ≤ ~16, we may
  add an `SVector`/`SMatrix` (StaticArrays.jl) inner path. **Defer until
  measured**; decision happens at Phase 0 day 26–28 (n=2,4,10,40 are
  already in the benchmark corpus). C++ does not specialize for small n;
  only StackAllocator helps it there.

**Memory management**
- `StackAllocator` (`StackAllocator.h`) is irrelevant in Julia — its
  purpose is to bypass `malloc`. Julia's GC pressure is solved by the
  preallocation strategy above, not by a custom allocator.
- All scratch buffers live in workspace structs passed through the call
  chain; no globals.

**LAPACK directly when beneficial**
- `mnvert.cxx` is a Gauss-Jordan symmetric inversion. For `n ≥ ~8`,
  switching to `LAPACK.sptrf!` / `LAPACK.sptri!` (Bunch-Kaufman on
  packed symmetric) is both faster and numerically better. Threshold
  measured at Phase 0 day 4–7.

**Eigenvalue routines (MnPosDef)**
- `MnPosDef.cxx:80` calls `eigenvalues(p)` where `p` is the *normalized
  correlation matrix* (the error matrix scaled by `1/sqrt(diag)` at
  lines 73–76), not the raw error matrix. Use
  `LAPACK.spev!('N','U', packed)` or `eigvals(Symmetric(M))` and apply
  to the same normalized form to match C++ bit-pattern.

**Parallel FCN evaluation**
- The C++ uses OpenMP in `Numerical2PGradientCalculator.cxx:112–127` to
  parallelize the gradient loop. Each parameter `i` writes to disjoint
  indices in `grd(i)`, `g2(i)`, `gstep(i)` — **no cross-thread reduction
  needed**; just per-thread independent writes to disjoint slots. Julia
  Phase 2.2: `Threads.@threads :static` over the parameter index.

**Type-stable error definitions**
- `up::Float64` lives in the FCN struct; we don't carry it through the
  call chain dynamically.

### 2.3 Pitfalls specific to this port

- **Closure specialization at the public boundary**: when the user calls
  `migrad(fcn::Function, x0, errors)`, the `fcn::Function`-typed argument
  must be immediately wrapped into a `CostFunction{typeof(fcn)}` so the
  *whole* internal call chain specializes. Verify with
  `JET.@report_call migrad(::Function, ::Vector{Float64}, ::Vector{Float64})`
  at the top-level entry, not just internal calls.
- **`@views` vs new alloc on slicing**: `g[k] - g_prev[k]` works
  element-wise but `g - g_prev` allocates. Use `@. dg = g - g_prev`.
- **BLAS thread interaction**: when the user wires multi-threaded
  likelihoods (Phase 2), `BLAS.set_num_threads(1)` is the safe default
  to avoid nested parallelism — same trick as iminuit + numpy. Also a
  **Phase 0 benchmark hygiene** concern: the gate script must call
  `BLAS.set_num_threads(1)` before `BenchmarkTools.@benchmark` to
  prevent OpenBLAS spinning up threads at small n (the spin-up cost
  exceeds the actual DSYMV work).
- **C++ ABObj expression templates** (`ABObj.h`, `ABSum.h`, `ABProd.h`):
  exist *only* to avoid intermediate `Matrix`/`Vector` allocation in
  compound arithmetic. Julia's broadcasting and `mul!`/`syr!`/`axpy!`
  into preallocated buffers solve this cleanly. **Do not port ABObj**.
  The real risk isn't missed BLAS fusion (Julia + LAPACK saturate BLAS
  already); the real risk is *intermediate allocation* when you write
  arithmetic-style expressions instead of in-place primitives.
- **`shared_ptr<BasicX>` indirection**: `MinimumState`, `FunctionGradient`,
  etc. use shared-pointer handles in C++ (e.g. `FunctionMinimum.h:99`).
  In Julia, plain immutable `struct`s whose heap-allocated array fields
  are shared by reference give the same sharing semantics for free. No
  `mutable struct` unless the type really mutates in place.

---

## 3. Phase 0 — Proof of concept

**Mandate**: prove the Julia port can reach C++-Minuit2 performance on
simple MIGRAD-only fits, within numerical equivalence, before any large
code surface is written.

**Scope (in)**:
- Unconstrained MIGRAD (gradient-based) with numerical gradient.
- Free parameters only — no bounds, no fixed parameters.
- **Strategy = 0 only** (no inner-loop `MnHesse` call). Strategy = 1, 2
  are Phase 1 — see Risk register entry "MnHesse inside MIGRAD".
- `MnUserFcn` call-counting + (trivial Phase-0) internal⇄external
  boundary wired in — even though no transform applies, the boundary
  must exist so Phase 1's bounds slot in without API churn.
- `NegativeG2LineSearch` — yes, in Phase 0. `MnSeedGenerator.cxx:80`
  calls `HasNegativeG2(...)` *unconditionally* on the seed gradient;
  skipping it risks NFcn mismatch and iteration-trajectory drift on the
  very Rosenbrock-10 benchmark in the gate.
- Reference-data generation harness in `tools/` (built **Day 0**, not as
  a side note) — without C++ JSON traces and per-iteration dumps, the
  Phase-0 acceptance tests have no oracle.
- Enough scaffolding to fit Rosenbrock-2, Rosenbrock-10, Quad4F, and
  Gaussian negative-log-likelihoods.

**Scope (out)**: parameter bounds (sin/sqrt transformations), fixed
parameters, MINOS errors, contours, SIMPLEX, Fumili, HESSE-after-MIGRAD,
Strategy ≥ 1, the IMinuit.jl named-parameter API surface, anything
plot-related.

**Exit gate**: ≤ 1.5× C++ wall time on the **entire benchmark corpus in
§3.3** (not just Rosenbrock-10), results identical to 1e-10 on parameter
values and function minimum, **with the julia-perf Level-2 evidence-gate
artifact contract satisfied** (see §3.4.1). If the gate fails, the
architectural assumptions in §2 are wrong and must be revisited before
Phase 1 starts.

### 3.1 Files to create under `src/`

```
src/
  JuMinuit.jl                # top-level module; reexports the public surface
  precision.jl               # MachinePrecision: mirrors MnMachinePrecision
  strategy.jl                # Strategy: mirrors MnStrategy (levels 0/1/2)
  fcn.jl                     # CostFunction{F,T} wrapper; ncalls counter;
                             # internal⇄external boundary (Phase 0: identity).
  parameters.jl              # MinuitParameter (single param); free-only first
  state.jl                   # MinimumParameters, MinimumError, FunctionGradient,
                             # MinimumState; concrete, immutable wrappers over
                             # shared-by-reference heap arrays
  workspace.jl               # MigradWorkspace: full scratch buffer inventory
                             # per §2.2 table
  linalg.jl                  # symmetric storage helpers (dense default + packed
                             # behind feature flag), mul!, syr!, axpy! wrappers,
                             # invert!, eigvals — thin layer for benchmarking
  gradient.jl                # initial_gradient!, numerical_gradient! (Numerical2P)
                             # Phase 0 ports no-limits code path only; HasLimits
                             # branches at C++ Numerical2PGradientCalculator.cxx:
                             # 136-139 and InitialGradientCalculator.cxx:47-58,
                             # 66-69 are tagged "# TODO Phase 1" inline.
  edm.jl                     # estimate_edm (mirrors VariableMetricEDMEstimator)
  posdef.jl                  # make_posdef! (mirrors MnPosDef)
  davidon.jl                 # davidon_update! — DFP rank-2 base, with
                             # rank-1 additive correction (added on top, NOT
                             # branched) when delgam > gvg. See §7 row + §8
                             # risk #1. Mirror FLOP order of
                             # DavidonErrorUpdator.cxx:58-69 exactly.
  negative_g2.jl             # NegativeG2LineSearch — called unconditionally in
                             # seed (MnSeedGenerator.cxx:80). ~80 LOC port.
  linesearch.jl              # parabolic line search (mirrors MnLineSearch op())
  migrad.jl                  # the iteration loop (mirrors VariableMetricBuilder);
                             # _migrad_outer! (handles maxfcn=80% trick at
                             # VariableMetricBuilder.cxx:54-203) wraps
                             # _migrad_inner! (the 205-375 inner loop).
  seed.jl                    # seed_state (mirrors MnSeedGenerator); invokes
                             # negative_g2 unconditionally per C++.
  result.jl                  # FunctionMinimum; storage_level=0 default in P0
  api.jl                     # migrad(fcn, x0, errors;
                             #        strategy=Strategy(0), tol=0.1,
                             #        maxfcn = 200 + 100*n + 5*n^2,
                             #        storage_level=0)
                             # The maxfcn default is from MnApplication.cxx:43.
```

**Why this many files**: each file maps 1-to-1 to a C++ translation unit
we'll diff against during the port. Smaller modules make line-by-line
review tractable and isolate any numerical regression to a unit test.

**What NOT to create yet**:
- `transform.jl` (sin/sqrt bound transforms) — Phase 1
- `migrad_api.jl` mimicking `MnMigrad` — Phase 1
- `minos.jl`, `hesse.jl`, `contours.jl`, `function_cross.jl`,
  `covariance_squeeze.jl` — Phase 1
- `simplex.jl` — Phase 1 (or Phase 2 if Phase 1 schedule slips; SIMPLEX is
  gradient-free and conceptually independent of MIGRAD)
- `precompile.jl`, plotting, AD glue — Phase 2

### 3.2 Tests under `test/`

TDD with the C++ reference as oracle. Test files mirror sources:

```
test/
  runtests.jl                # top-level driver
  test_linalg.jl             # spmv, syr, invert; cross-checked vs LAPACK
  test_precision.jl          # MachinePrecision constants match C++
  test_strategy.jl           # strategy level 0/1/2 values match MnStrategy.cxx:33-70
  test_initial_gradient.jl   # InitialGradientCalculator outputs on Quad4
  test_numerical_gradient.jl # Numerical2P outputs on Rosenbrock & Quad4
  test_davidon.jl            # synthetic 4D Hessian update reproduces C++ to 1e-12;
                             # MUST exercise the rank-1-additive branch
                             # (delgam > gvg case) explicitly
  test_negative_g2.jl        # synthetic 2D quadratic with deliberately-negative
                             # seed g2; check that HasNegativeG2 fires and the
                             # gradient is corrected
  test_linesearch.jl         # canned step/gradient configurations
  test_edm.jl                # gᵀ V g / 2 numerical equivalence
  test_seed.jl               # MnSeedGenerator output (including the
                             # NegativeG2LineSearch leg) matches C++
  test_migrad_quad4.jl       # full MIGRAD on Quad4F (MnTutorial/Quad4F.h);
                             # asserts min == 0, params == 0 to 1e-10, same NFcn
  test_migrad_rosenbrock.jl  # 2D and 10D Rosenbrock; results + iteration trace
                             # vs reference dump
  test_migrad_gauss_ll.jl    # 100-point Gaussian NLL fit (mirror of
                             # MnSim/GaussFcn.cxx); compare to fixed-seed reference
  test_zero_alloc.jl         # @allocated == 0 for one full inner iteration
                             # (FCN call + gradient + line search + DFP + EDM
                             # + state rebuild). Codex#3 + Opus#3.
  reference_data/            # JSON dumps from tools/regen_reference.sh — pinned
                             # to GooFit/Minuit2 @ 57dc936; see tools/README.
    quad4f_min.json
    rosenbrock2d_min.json
    rosenbrock10d_min.json
    gaussll_min.json
    davidon_trace_4d.json    # per-iteration inv_hessian for DFP audit
    seed_with_neg_g2.json    # NegativeG2 corner case oracle
```

**Reference-data generation** (Day 0 of Phase 0, in `tools/`):

```
tools/
  regen_reference.sh         # cmake + build reference/Minuit2_cpp/examples/
                             # + a tiny C++ harness that dumps Minuit2 internal
                             # state (FunctionMinimum + per-iteration trace if
                             # storage_level=1) as JSON to test/reference_data/
  regen_reference.md         # checklist: when to regen + why (mandatory diff
                             # vs previous), tested BLAS/compiler combo
  cpp_trace_harness.cxx      # built as a tiny CMake project under tools/;
                             # uses MnTraceObject to dump per-iter state
```

Check the JSON into `test/reference_data/` so the test suite stays
self-contained and doesn't require a working C++ toolchain on dev/CI.

### 3.3 Benchmark suite under `benchmark/`

```
benchmark/
  bench_migrad.jl            # BenchmarkTools.jl; Rosenbrock, Quad4, Gauss-LL
  bench_gradient.jl          # isolated numerical gradient on Rosenbrock-10
  bench_davidon.jl           # isolated DFP update at varying n
  bench_long_fit.jl          # cheap-FCN long fit; reports GC time separately
                             # (catches GC pauses uncorrelated with MIGRAD)
  perf-config.toml           # julia-perf evidence-gate config
                             # (mirrors .claude/skills/julia-perf/templates/)
  cpp/                       # tiny CMake project that builds the same fits
    CMakeLists.txt
    bench_rosenbrock.cxx
    bench_gauss_ll.cxx
  compare.jl                 # runs Julia BenchmarkTools median, parses C++
                             # stderr wall-time output, emits the artifact
                             # set via scripts/run_perf.jl
scripts/
  run_perf.jl                # the gate driver (consumes perf-config.toml;
                             # writes .julia-perf/runs/<ts>/{manifest,
                             # benchmarks,summary}.json + diagnostics.md).
                             # Exit codes: 0 pass / 1 fail / 2 soft-warn.
```

**Benchmark scenarios** (the exit-gate corpus):

| Name                       | n free | FCN cost   | Status     | Why it matters                                  |
|----------------------------|--------|------------|------------|-------------------------------------------------|
| Rosenbrock-2               | 2      | trivial    | blocking   | low-n, gradient-loop dominated; MVector candidate |
| Rosenbrock-10              | 10     | trivial    | blocking   | canonical Minuit stress test; non-quadratic curvature |
| Quad4F (analytic mode)     | 4      | trivial    | blocking   | sanity check; exact convergence                 |
| Gauss-LL-2 × 100           | 2      | 100 events | blocking   | typical "small fit", FCN ≈ overhead             |
| Gauss-LL-10 × 1000         | 10     | 1k events  | blocking   | typical HEP fit                                 |
| Gauss-LL-40 × 1000         | 40     | 1k events  | diagnostic | parallel-test analog (`MnSim/ParallelTest.cxx`); informs Phase 2.2 — not blocking for Phase 0 |
| Cheap-FCN long fit (1k iter on noisy 4D quadratic) | 4 | trivial | diagnostic | GC pressure check; not blocking |

### 3.4 Acceptance criteria (Phase 0 exit gate)

A merge to `main` enabling Phase 1 requires *all* of:

1. **Correctness**: every blocking benchmark in §3.3 reproduces the C++
   reference to:
   - `|Δ fval| / max(1, |fval|) ≤ 1e-10`
   - `|Δ params_i| / max(1, |params_i|) ≤ 1e-10`
   - `|Δ edm| / edm ≤ 1e-6` — note: C++ applies `edmval *= 0.002` at
     `VariableMetricBuilder.cxx:66`; the Julia port must replicate this
     tolerance-multiplier exactly so the convergence criterion matches
     and NFcn aligns.
   - `NFcn` within ±5 of C++ (widened from ±2 to absorb combined
     line-search + DFP rounding drift; see Risk #1).
2. **Performance**: median Julia wall time / median C++ wall time ≤ 1.5
   on every blocking benchmark in §3.3, on the designated reference
   machine (see Open Question 10), with `BLAS.set_num_threads(1)` on both
   sides, recorded in the gate manifest.
3. **Zero allocations in the inner loop**: `@allocated` for one full
   MIGRAD iteration on Rosenbrock-10 returns 0 — measured end-to-end
   (FCN transform + numerical gradient + line search + DFP update + EDM +
   state/history append), not just the linear algebra primitive.
4. **Clean run**: no warnings, no failed asserts, `Aqua.jl` clean, and
   `JET.@report_call` clean on the **public** API
   `migrad(::Function, ::Vector{Float64}, ::Vector{Float64})` from the
   top-level entry point (not just internal builder functions).
5. **Evidence-gate compliance** (§3.4.1).

#### 3.4.1 julia-perf Level-2 evidence-gate compliance

The performance claim in §3.4 criterion 2 must be reproducible per the
local `julia-perf` skill's Level-2 evidence-gate contract
(`.claude/skills/julia-perf/SKILL.md`). The gate run via
`scripts/run_perf.jl` must:

1. **Baseline existence**: on first run, `--save-baseline` creates the
   reference; subsequent runs compare to it.
2. **Diagnostic capture**: emit `JET.report_opt` and `Test.@inferred`
   results for the public `migrad` entry point into `diagnostics.md`.
   Capture allocation traces for one full MIGRAD iteration.
3. **Verification**: re-run the §3.3 blocking corpus with fixed gate
   parameters (BenchmarkTools `samples`, `evals`, `seconds`).
4. **Artifact emission**: write to `.julia-perf/runs/<ISO-timestamp>/`
   (standalone mode) or `artifacts/runs/<tag>/julia-perf/` (ecosystem
   mode):
   - `manifest.json` — full env: Julia version, OpenBLAS/MKL version,
     CPU model, governor state, `BLAS.get_num_threads()`, C++ Minuit2
     git SHA + compiler version, git commit of the Julia code under
     test.
   - `benchmarks.json` — per-benchmark median/min/max/allocs/memory,
     Julia-vs-C++ ratio.
   - `summary.json` — gate verdict (`pass | warn | fail`) with the rule
     each criterion was checked against.
   - `diagnostics.md` — human-readable JET + allocation + hotspot
     narrative.
5. **Exit code policy**: 0 if all criteria pass; 2 if any soft-warn
   (e.g., ratio is 1.5–1.6× on one benchmark); 1 if any hard-fail.

The §3.4 "Performance" criterion is **unverifiable** without these
artifacts. A `@time ratio = 1.4` log line does not constitute evidence.

### 3.5 Implementation order

Within Phase 0, the order minimizes integration risk. Note: cluster sizes
are widened vs. v1 to account for the depth of correctness work in DFP
and the four-pillar gate.

1. **Day 0**: stand up `tools/cpp_trace_harness.cxx` + `regen_reference.sh`;
   produce all `test/reference_data/*.json` from the pinned C++ build.
2. **Day 1–3**: scaffolding (`Project.toml`, `JuMinuit.jl`, `precision.jl`,
   `strategy.jl`, `state.jl`, `fcn.jl` with trivial Phase-0 boundary) +
   `test_precision.jl` / `test_strategy.jl` green.
3. **Day 4–7**: `linalg.jl` + `test_linalg.jl`. Both dense and packed
   variants behind a feature flag; defer the decision but lock the
   interface.
4. **Day 8–12**: `gradient.jl` (initial + numerical) + tests on Quad4
   and Rosenbrock matching C++ to ~1e-12. Includes the
   `NegativeG2LineSearch` precondition path.
5. **Day 13–18**: `davidon.jl` (correctness-critical — additive rank-1
   formula) — 5 days for the iteration-by-iteration trace audit vs C++;
   `edm.jl`, `posdef.jl`, `negative_g2.jl` finish out the week.
6. **Day 19–22**: `linesearch.jl` + tests, then `seed.jl` (invokes
   negative_g2 unconditionally per `MnSeedGenerator.cxx:79–86`).
7. **Day 23–28**: `migrad.jl` — the loop (outer + inner). End-to-end test
   on Quad4F first (closed form), then Rosenbrock-2, Rosenbrock-10,
   Gauss-LL.
8. **Day 29–35**: full benchmark sweep via `scripts/run_perf.jl`, profile,
   optimize. Run exit gate. Iterate if any criterion fails. Decide
   dense-vs-packed and Vector-vs-MVector based on measurement.

Any slip past day 35 triggers a design review of §2.2 idioms.

---

## 4. Phase 1 — Core port (goal + exit only)

**Goal**: feature-complete equivalence with `MnMigrad + MnHesse + MnMinos +
MnSimplex + MnContours` for the cases iminuit covers. Bounds, fixed
parameters, named parameters, MINOS asymmetric errors, contours via
profile re-minimization, HESSE-after-MIGRAD, strategy levels 0/1/2 all
working.

**Exit criteria**:
- All MnTutorial examples (`Quad1F`, `Quad4F`, `Quad8F`, `Quad12F`)
  reproduce.
- The `MnSim/GaussFcn`-class likelihood fits reproduce.
- Parameter bounds via `sin`/`sqrt` transforms produce identical internal
  parameter values to C++ at every iteration of a chosen reference fit
  (recorded with `MnTraceObject` equivalent — see Open Question 12).
- Transformed analytical-gradient and external-covariance tests pass for
  lower-, upper-, and double-bounded parameters. The C++ chain rule
  (`DInt2Ext` factor in `MnUserTransformation.cxx:143–166`) is sign-aware:
  `SqrtUpParameterTransformation.cxx:40–45` returns a *negative*
  derivative for one-sided upper limits, so off-diagonal covariance
  signs must be tested explicitly.
- User-supplied analytical gradients via `FCNGradAdapter`-equivalent
  work (Phase 2.1 ships the AD-driven variant; Phase 1 ships the
  user-supplied path).
- MINOS errors agree to 1e-8 with the C++ reference on Quad4F and a
  bounded Gauss fit.
- `MnContours` produces profile contours via re-minimization with two
  parameters fixed (the actual C++ algorithm at
  `MnContours.cxx:52–78,125–178` + `MnFunctionCross.cxx:117–178`), not
  a naive 2D grid scan or unprofiled level set.
- `MnCovarianceSqueeze` removes fixed parameters via invert→delete-
  row/col→re-invert with diagonal-fallback on inversion failure
  (`MnCovarianceSqueeze.cxx:19–62, 65–108`).
- `MnHesse`-inside-MIGRAD path works: when Strategy ≥ 1 and Dcovar >
  0.05, MIGRAD invokes Hesse internally
  (`VariableMetricBuilder.cxx:138–173`). Phase 1 must ship this — it is
  the iminuit default behavior (Strategy 1).
- Performance ratio ≤ 1.3× C++ on all Phase 0 benchmarks plus a bounded
  variant of Gauss-LL-10.
- No public name change between Phase 0 and Phase 1; Phase 0 users keep
  working.

Phase 1 task breakdown is **deliberately deferred** until Phase 0 exits.
What we learn from achieving 1.5× will reshape the Phase 1 plan.

---

## 5. Phase 2 — Julia-native extras

Each extra ships as an independent feature, gated behind its own
milestone. Order is by user demand (the IMinuit.jl user base) rather
than dependency.

- **2.1 AD-backed analytical gradients**
  - `FCNGradAdapter`-equivalent that takes a user FCN + an AD backend
    choice (`ForwardDiff` primary; `Enzyme` opt-in for hot inner loops).
  - Auto-fallback to numerical gradient if AD fails.
  - **Phase 0 prerequisite**: the public `migrad` entry point boundary
    must accept non-`Float64` element types at compile time (without
    forcing the internal MIGRAD arithmetic off Float64). Specifically:
    define objective-call boundaries via `eltype(x0)` rather than
    hard-coded `Float64`, so `ForwardDiff.Dual` users don't hit a Phase-1
    breaking change. Internal scratch buffers stay Float64 in Phase 0.
  - Verification: AD-gradient MIGRAD on Rosenbrock-10 should reduce
    NFcn by ~`2·n·NCycle` per iteration and beat numerical gradient by
    ≥ 2× wall time.

- **2.2 Threads-parallel numerical gradient**
  - `Threads.@threads :static` over the parameter index in
    `numerical_gradient!` (analog of the OpenMP block at
    `Numerical2PGradientCalculator.cxx:112–127`). Per-thread scratch in
    the workspace; writes are to disjoint `grd(i)/g2(i)/gstep(i)`
    slots, so no reduction step is needed.
  - Default off; opt-in via `migrad(fcn, ...; threaded_grad=true)`.
    Document BLAS thread interaction.

- **2.3 Plot recipes + ASCII fallback**
  - RecipesBase recipes for: profile likelihood (1D), contour, MINOS
    error bars; mirror `MnContours` output shape so plotting just
    consumes the result struct.
  - Provide an `mn_plot_text(::FunctionMinimum)` helper for terminal
    users (some IMinuit.jl users in non-graphical environments rely on
    `MnPlot`'s ASCII output).

- **2.4 Precompilation & startup**
  - `PrecompileTools.@compile_workload` covering
    `migrad(::CostFunction{F}, ::Vector{Float64}, ::Vector{Float64})`
    for a few common `F` patterns; measure TTFX vs. without.

- **2.5 Result serialization**
  - `FunctionMinimum` → `Dict` and `Dict` → `FunctionMinimum` for easy
    JSON/JLD2 persistence; useful for the JuMinuit-vs-Minuit regression
    CI.

- **2.x Closure-allocation diagnostic**
  - For users whose FCN allocates internally (e.g. `sum(log.(...))` in
    an unbinned NLL), provide `migrad`'s output with a `gc_time_pct`
    field and a hint if `@allocated migrad(fcn, x0, errors) > 0`. Phase
    0's zero-alloc gate is *internal* to JuMinuit; user closures can
    still allocate.

---

## 6. Phase 3 — API parity with IMinuit.jl / iminuit

The user-facing entry point should let a user copy-paste from iminuit
(Python) or IMinuit.jl with at most renaming. Parity targets:

- `Minuit(fcn, x0; name=names, error=errors, limit=limits, fix=fixed, ...)`
  constructor identical to iminuit's.
- `m.migrad()`, `m.hesse()`, `m.minos()`, `m.contour(i, j)` methods.
- Property access on the result: `m.values`, `m.errors`, `m.fmin`,
  `m.valid`, `m.covariance`, `m.params` **and** the function-style
  accessors `values(m)`, `errors(m)`, etc. (both ship; iminuit
  copy-paste users use the property style, Julia-idiomatic users use
  the accessor style).
- Pretty printing via `show(::IO, ::MIME"text/plain", ::Minuit)` matching
  iminuit's table-style output line by line where possible.
- IMinuit.jl compatibility shim: optional `IMinuitCompat` submodule
  re-exports IMinuit.jl's `Migrad`/`Minos`/etc. signatures so existing
  user code requires changing only the `using` line.
- Documentation: every iminuit tutorial reproduced as a Julia example
  in `docs/src/`.

**Exit criteria**: ten randomly chosen IMinuit.jl scripts in real-world
HEP fits run, with at most a `using JuMinuit` substitution, to numerical
equivalence with the original (≤ 1e-8 on parameter values).

---

## 7. Module mapping table

The porting compass. Read row-by-row against the C++ source.

| C++ class / file | Julia module / type | Phase | Notes |
|---|---|---|---|
| `Minuit2Minimizer.h/.cxx` | `JuMinuit.Minuit` (struct) + `migrad!/hesse!/minos!` methods | 3 | The ROOT `Math::Minimizer` facade — algorithm enum, status conventions, plugin surface. Much broader than iminuit-style `Minuit`. Phase 3 only. |
| `MnApplication.h/.cxx` | (internal facade) | 1 | Bundles FCN + state + strategy. **Stays internal** — `MnMigrad`/`MnHesse` derive from it in C++, but Julia users don't see it. Phase 1 ships `migrad`/`hesse`/`minos` free functions; Phase 3 ships the `Minuit` struct as the user entry. |
| `MnMigrad.h` | `migrad(fcn, x0, errors; ...)` free function (P0) and `Migrad(fcn, params; ...)` struct (P1) | 0 / 1 | Many overloads in C++; one keyword-driven Julia function. Default `maxfcn = 200 + 100n + 5n²` from `MnApplication.cxx:43`. |
| `MnSimplex.h` + `SimplexMinimizer.h` + `SimplexBuilder.cxx` + `SimplexParameters.cxx` + `SimplexSeedGenerator.cxx` | `JuMinuit.simplex(fcn, x0; ...)` + `SimplexBuilder` | 1 or 2 | Nelder-Mead, no derivatives. ~200 lines C++. Could ship as a Phase 2.x add-on if Phase 1 schedule slips — gradient-free, conceptually independent of MIGRAD. |
| `MnMinos.h/.cxx` + `MnCross.h` + `MinosError.h` | `JuMinuit.minos(fmin, fcn, ipar; ...)` + `MinosError` struct | 1 | Asymmetric errors. Uses `MnFunctionCross` under the hood. |
| `MnFunctionCross.cxx` | `JuMinuit._function_cross!(...)` (internal) | 1 | **New row** (was missing in v1). Underpins both MINOS asymmetric errors and `MnContours` re-minimization. `MnFunctionCross.cxx:117–178` is the core: fixes a parameter to a target value, re-minimizes the rest, returns the crossing. |
| `MnContours.h/.cxx` + `ContoursError.h` | `JuMinuit.contour(fmin, fcn, i, j; npoints=20)` | 1 | **Profile contour via re-minimization** with two parameters fixed (`MnContours.cxx:52–78,125–178`). First gets four MINOS axis points, then iterates `MnFunctionCross` at angles. Not a naive 2D grid scan and not a level set of the unprofiled likelihood. |
| `MnHesse.h/.cxx` | `JuMinuit.hesse!(state; ...)` + invoked internally from MIGRAD when Strategy ≥ 1 (`VariableMetricBuilder.cxx:138–173`) | 1 | Full numerical Hessian, lines 100–315. Diagonal loop has 2 FCN/cycle × up-to-5 step retries per parameter; off-diagonal 1 FCN per pair. |
| `MnScan.h/.cxx` + `ScanBuilder.cxx` + `MnParameterScan.cxx` | `JuMinuit.scan(fcn, ipar; ...)` | 2 | 1D function scan; mostly cosmetic. |
| `ModularFunctionMinimizer.h/.cxx` | (folded into `migrad`/`simplex`/...) | 0 | C++ class is an abstract dispatch over (SeedGenerator × Builder); Julia uses multiple dispatch on FCN type, no inheritance needed. |
| `VariableMetricMinimizer.h` | (delete; not needed in Julia layout) | 0 | Just a (SeedGenerator, VariableMetricBuilder) bundle. |
| `VariableMetricBuilder.h/.cxx` | `JuMinuit._migrad_outer!` + `JuMinuit._migrad_inner!` (internal) | 0 | C++ has **two** overloaded `Minimum(...)`: outer at lines 54–203 with the `maxfcn = floor(0.8 * maxfcn)` trick to leave budget for Hesse, inner at 205–375 with the 200-line MIGRAD iteration loop. Mirror the split. |
| `MnSeedGenerator.h/.cxx` | `JuMinuit._seed_state(fcn, x0, errors, strategy)` | 0 | `MnSeedGenerator.cxx:42–101`. Invokes `NegativeG2LineSearch.HasNegativeG2()` **unconditionally** at line 80. User-supplied prior covariance branch (`MnSeedGenerator.cxx:63–67`) is Phase 1+. |
| `MnLineSearch.h/.cxx` | `JuMinuit._line_search!(workspace, fcn, ...)` | 0 | Parabolic interpolation; `MnLineSearch.cxx:46–313`. Cubic and Brent variants are `#ifdef USE_OTHER_LS` — **do not port** (see Deferred §9). |
| `MnParabola.h/.cxx` + `MnParabolaFactory.h/.cxx` + `MnParabolaPoint.h` | inline helpers in `linesearch.jl` | 0 | Three tiny classes, fuse into local helpers. |
| `Numerical2PGradientCalculator.h/.cxx` | `JuMinuit._numerical_gradient!(workspace, fcn, p, prev_grad, strategy)` | 0 | Two-point central diff at `Numerical2PGradientCalculator.cxx:63–230`. Phase 0 ports **no-limits code path only**; the `HasLimits()` branches at lines 136–139 (and `InitialGradientCalculator.cxx:47–58,66–69`) get `# TODO Phase 1: HasLimits branch` markers, not silent deletion. OpenMP block (lines 116–127) → Phase 2.2. |
| `InitialGradientCalculator.h/.cxx` | `JuMinuit._initial_gradient!(workspace, p, fcn, trafo, strategy)` | 0 | Same HasLimits-elided-with-TODO-marker policy. |
| `HessianGradientCalculator.h/.cxx` | `JuMinuit._hessian_gradient!(...)` | 1 | Refined gradient inside `MnHesse`. |
| `AnalyticalGradientCalculator.h/.cxx` | `JuMinuit._analytical_gradient!(...)` (consumes user-provided `∇fcn`) | 1 / 2 | Phase 1 supports user-supplied gradient (with `DInt2Ext` chain rule); Phase 2.1 wires up AD. |
| `DavidonErrorUpdator.h/.cxx` | `JuMinuit._davidon_update!(workspace, V, p1, g1, s0)` | 0 | `DavidonErrorUpdator.cxx:24–73`. **Rank-2 DFP base, always**, plus a **rank-1 additive correction ADDED on top** (NOT branched) when `delgam > gvg` (lines 60–65: `vUpd += gvg * Outer_product(...)`). The C++ comment "use rank 1 formula" is misleading — it's a rank-3 BFGS-style hybrid. Match FLOP order line-for-line to keep numerical equivalence under 1e-12 over many iterations. See Risk #1. |
| `BFGSErrorUpdator.h/.cxx` | `JuMinuit._bfgs_update!(...)` | 2 | Alternative updator; `MnMigrad(BFGSType{})` path. Phase 2 opt-in. |
| `FumiliBuilder/FumiliMinimizer/Fumili*` (10 files) | (deferred) | — | See §9. |
| `MinimumBuilder.h/.cxx` | folded into builder functions | 0 | Print level + tracer + storage level become fields on the builder. |
| `MinimumSeed.h` + `BasicMinimumSeed.h` | `MinimumSeed` (immutable struct) | 0 | No shared-ptr indirection needed. |
| `MinimumState.h` + `BasicMinimumState.h` | `MinimumState` (immutable wrapper) | 0 | C++ uses `shared_ptr<BasicMinimumState>`. Julia uses an immutable `struct`; its `params::Vector{Float64}`, `inv_hessian::Symmetric{...,Matrix{Float64}}`, gradient fields are heap-allocated arrays shared by reference across iterations, so rebuilding the wrapper each iteration is free of bulk data copies. **No `mutable struct` needed**; the workspace is what mutates. |
| `MinimumParameters.h` + `BasicMinimumParameters.h` | `MinimumParameters` (struct: `x::Vector{Float64}`, `dirin::Vector{Float64}`, `fval::Float64`) | 0 | |
| `MinimumError.h` + `BasicMinimumError.h` | `MinimumError` (struct: `inv_hessian::Symmetric{...}`, `dcovar::Float64`, `status::CovStatus`) | 0 | Status enum is `@enum CovStatus { MnHesseFailed, MnMadePosDef, MnInvertFailed, MnNotPosDef, MnHesseValid }` (Decision Q7 locked). |
| `FunctionGradient.h` + `BasicFunctionGradient.h` | `FunctionGradient` (struct: `grad::Vector`, `g2::Vector`, `gstep::Vector`, `analytical::Bool`) | 0 | |
| `FunctionMinimum.h` + `BasicFunctionMinimum.h` | `FunctionMinimum` (struct: seed, states, up, status flags) | 0 / 1 | P0: `storage_level=0` default (seed + final only); P1: `storage_level=1` opt-in for full history. C++ default is `storage_level=1` in `MnApplication`. |
| `MinimumErrorUpdator.h` | (abstract interface; Julia uses dispatch on updator type) | 0 | |
| `FCNBase.h` + `FCNGradientBase.h` + `FCNAdapter.h` + `FCNGradAdapter.h` | `CostFunction{F, T}` and `CostFunctionWithGradient{F, G, T}` (parametric structs) | 0 / 1 | C++ inheritance hierarchy collapses to two concrete types; multiple dispatch on FCN type handles the grad-vs-no-grad branching. |
| `MnFcn.h/.cxx` | call-counting wrapper inside the builder (`workspace.nfcn += 1` after each FCN call) | 0 | C++ wraps FCN to count calls and repack `MnAlgebraicVector` → `std::vector` (`MnFcn.cxx:23–28`). Julia skips the repack entirely — small advantage. |
| `MnUserFcn.h/.cxx` | folded into `CostFunction.eval_at(x_internal)` which applies the internal→external transform | 0 (trivial) / 1 (with transform) | C++ allocates/copies an external vector on **every** call (`MnUserFcn.cxx:23–29`) and notes a ~10% Rosenbrock slowdown. Julia preallocates `x_ext` in workspace; same buffer reused. |
| `MnUserParameters.h/.cxx` | `Parameters` (thin wrapper around `Transformation`; only forwards accessors) | 1 | **Structural note**: in C++, `MnUserParameters` holds only a `MnUserTransformation` (line 112); the parameter vector + names actually live inside `MnUserTransformation` (`MnUserTransformation.cxx:241–249`). The two classes are tightly coupled — Julia ships a single `Parameters` struct that *is* `Transformation` + accessor methods, not two parallel structs. |
| `MnUserParameterState.h/.cxx` | `ParameterState` (struct) | 1 | |
| `MnUserCovariance.h` | `Covariance` (struct over `Symmetric` or packed `Vector{Float64}`) | 1 | |
| `MnUserTransformation.h/.cxx` | `Transformation` (struct holding parameter metadata + cached external values + name map) | 1 | Heart of the bounds path (`MnUserTransformation.cxx:99–141`). The `DInt2Ext` (`lines 143–166`) derivative is **sign-aware**: see `SqrtUpParameterTransformation.cxx:40–45` (negative for upper limit). Phase 1 must test transformed analytical gradient *and* transformed external covariance for lower-, upper-, double-bounded parameters. |
| `MinuitParameter.h` | `MinuitParameter` (struct) | 0 / 1 | Phase 0: just `value`, `error`, `fixed::Bool`. Phase 1: adds limits. |
| `MnGlobalCorrelationCoeff.h` | `GlobalCorrelation` (computed on demand) | 1 | |
| `SinParameterTransformation.cxx` + `SqrtUpParameterTransformation.cxx` + `SqrtLowParameterTransformation.cxx` | three `int2ext` / `ext2int` / `dint2ext` methods in `transform.jl` | 1 | Pure scalar math; 3 × ~50-line files. **Strict mirror** (Decision Q2 locked). |
| `MnPosDef.h/.cxx` | `_make_posdef!(error, prec)` | 0 | Adds-to-diagonal trick to enforce positive definiteness; `MnPosDef.cxx:30–104`. The matrix fed to `eigenvalues(p)` at line 80 is the **normalized correlation matrix** (`1/sqrt(diag)` scaling at lines 73–76), not the raw error matrix. Apply the same normalization in Julia. |
| `MnMachinePrecision.h/.cxx` | `MachinePrecision` (struct: `eps`, `eps2`); default from `eps(Float64)` | 0 | |
| `MnStrategy.h/.cxx` | `Strategy` (struct); `Strategy(0)`, `Strategy(1)`, `Strategy(2)` constructors | 0 | Values from `MnStrategy.cxx:33–70`. Phase 0 locks Strategy = 0 (no inner Hesse). |
| `VariableMetricEDMEstimator.h/.cxx` | `_estimate_edm(grad, error)` (free function) | 0 | `0.5 * similarity(grad, inv_hessian)`. |
| `NegativeG2LineSearch.h/.cxx` | `_negative_g2_line_search!(...)` in `src/negative_g2.jl` | 0 | **Phase 0 (not deferred).** Called **unconditionally** at `MnSeedGenerator.cxx:80`. ~80 LOC; depends only on `linesearch.jl` + `gradient.jl`. Skipping risks NFcn mismatch + trajectory drift on Rosenbrock-10. |
| `MPIProcess.h/.cxx` | (deferred) | — | See §9. |
| `LASymMatrix.h` + `LAVector.h` + `MnMatrix.h` | dense `Symmetric{Float64,Matrix{Float64}}` + `Vector{Float64}` (Phase 0 default, Decision Q3 locked); packed variant retained behind flag for benchmark comparison | 0 | C++ uses lower-triangular packed *also* because packed BLAS (DSPMV/DSPR) is tied into ABObj expression templates, not only for memory locality. Julia abandons both packed BLAS and ABObj in favor of dense + `LinearAlgebra.BLAS.*`. |
| `ABObj.h` + `ABSum.h` + `ABProd.h` + `ABTypes.h` + `LaSum.h` + `LaProd.h` + `LaOuterProduct.h` + `LaInverse.h` + `VectorOuterProduct.h` + `MatrixInverse.h` | (do not port; use `mul!`/`axpy!`/`syr!` into preallocated buffers) | — | Expression-template layer is a C++ workaround. Julia's pitfall is not "missed BLAS fusion" but intermediate-buffer allocation in arithmetic-style expressions (see §2.3 + §8 Risk #3). |
| `StackAllocator.h` | (do not port) | — | Workspace preallocation in `MigradWorkspace` is the Julia analog. |
| `MnRefCountedPointer.h` + `MnReferenceCounter.h` | (do not port) | — | Pre-`shared_ptr` reference counting; obsolete. |
| `mndaxpy.cxx` + `mndscal.cxx` + `mnddot.cxx` + `mndspmv.cxx` + `mndspr.cxx` + `mndasum.cxx` + `mnlsame.cxx` + `mnxerbla.cxx` | (do not port) | — | f2c-translated BLAS. Use `LinearAlgebra.BLAS.*` directly. |
| `mnvert.cxx` | `_sym_invert!(M)` calling `LAPACK.sptrf!`/`sptri!`, fallback to Gauss-Jordan for tiny n | 0 / 1 | Unit test comparing both paths on a 4×4 case. |
| `mnteigen.cxx` + `LaEigenValues.cxx` | `_sym_eigvals(M)` via `LAPACK.spev!('N', 'U', ...)` or `eigvals(Symmetric(M))` | 0 / 1 | Needed inside `MnPosDef`. |
| `mntplot.cxx` + `MnPlot.h/.cxx` + `mnbins.cxx` | RecipesBase recipes + `mn_plot_text` helper (Phase 2.3) | 2 | C++ text-plot users get a terminal fallback. |
| `MnPrint.h/.cxx` + `MnPrintImpl.cxx` | use `Logging` stdlib + `@debug`/`@info` | 0 | C++ rolls its own print system; Julia's logging is sufficient. |
| `MnTraceObject.h/.cxx` + `TMinuit2TraceObject.cxx` | optional callback in `migrad(... ; trace=callback)` | 1 / 2 | Per-iteration trace; mandatory for the Phase 1 line-by-line iteration-equivalence test (see Open Question 12). |
| `MnCovarianceSqueeze.h/.cxx` | dedicated `src/covariance_squeeze.jl` | 1 | **Not "inline helper".** Invert → delete row/col → invert back, with diagonal fallback on inversion failure (`MnCovarianceSqueeze.cxx:19–62, 65–108`). Numerically sensitive; deserves its own module + tests. |
| `MnEigen.h/.cxx` | `eigen(::Covariance)` method | 1 | Trivial. |
| `MinimizerOptions.cxx` (in `src/math/`) | `MinimizerOptions` keyword-args on `migrad` | 1 | |
| `ParametricFunction.h/.cxx` | (skip) | — | ROOT IFunction integration; standalone build doesn't need it. |
| `examples/simple/` | mirror as `examples/quad4.jl`, `examples/gauss_ll.jl` | 1 | Doc-driving examples. |
| `test/MnSim/*` + `test/MnTutorial/*` | mirror corresponding C++ tests as Julia tests; one Julia test = one C++ test | 0–1 | Phase-0 corpus is `Quad1F`, `Quad4F`. |
| `Math/Minimizer.h`, `Math/IFunction*.h`, `Fit/ParameterSettings.h` | (skip) | — | ROOT framework glue. |

**Counting**: ~26K LOC C++ → expected ~5–7K LOC Julia (much of the C++
is boilerplate around `shared_ptr` payloads, the expression-template
layer, and f2c BLAS translations). Phase 0 alone should be < 1.8K LOC
Julia + < 1.2K test LOC (the upward revision vs v1 reflects the explicit
inclusion of `negative_g2.jl`, the explicit scratch inventory, and the
reference-data harness).

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **DFP Hessian update numerical drift** vs C++. The C++ formula at `DavidonErrorUpdator.cxx:60–65` computes the rank-2 base **always** and **adds** a rank-1 correction when `delgam > gvg` (a rank-3 BFGS hybrid, not an either/or branch). Implementing it as an if/else picking rank-2 *or* rank-1 silently diverges from C++ after one update. Even bit-identical math through Julia BLAS vs reference BLAS will then diverge over many iterations due to FLOP-order differences. | High | High — breaks the 1e-10 acceptance criterion on long fits. | (a) Compare iteration-by-iteration `inv_hessian` to a captured C++ trace on Rosenbrock-10 for ≤ 50 iter; flag the first iteration where any element diverges past 1e-12 and root-cause. (b) Accept that final answers agree to 1e-10 even if iteration counts drift by ±5 (gate NFcn tolerance widened from ±2 in v1). (c) Implement the matrix updates in **exactly the same FLOP order** as `DavidonErrorUpdator.cxx:58–69`: compute base rank-2, test `delgam > gvg`, then `vUpd += gvg * Outer_product(dx/delgam − vg/gvg)` *additively*. |
| 2 | **BLAS thread interaction** when a user wires multi-threaded likelihoods (Phase 2). Nested OpenMP-in-OpenBLAS deadlocks or wastes cycles. Also a **Phase 0 benchmark hygiene** issue: OpenBLAS spinning up multiple threads at small n slows DSYMV via thread-spawn overhead. | Medium | Medium | Default `BLAS.set_num_threads(1)` when `Threads.nthreads() > 1`; document; provide a `with_blas_threads` helper. Gate script (`scripts/run_perf.jl`) sets it explicitly and records in `manifest.json`. |
| 3 | **Intermediate-buffer allocation in arithmetic expressions** (was "ABObj substitute under-performs" in v1; rephrased per Opus review). Writing `vUpd = (dx*dx')/delgam - (vg*vg')/gvg` in idiomatic Julia allocates **three** intermediate `Matrix` temporaries per iteration. The risk is not "missed BLAS fusion" — Julia + LAPACK saturate BLAS — but heap allocation that breaks the §3.4 zero-alloc gate. | Medium | High at Phase 0 gate | Replace arithmetic-style code with explicit `BLAS.syr!`/`BLAS.spr!`/`mul!`/`axpy!` into pre-allocated workspace buffers. Verified by `@allocated` per the §3.4 exit gate (criterion 3). |
| 4 | **Closure non-specialization** at the public API boundary. If `migrad(fcn::Function, x0, errors)` doesn't immediately wrap `fcn` into `CostFunction{typeof(fcn)}`, every FCN call goes through dynamic dispatch. iminuit users may pass closures from notebooks. | Medium | High | Wrap on entry. Use `JET.@report_call` on `migrad(::Function, ::Vector{Float64}, ::Vector{Float64})` at the *top-level* entry point in CI, not just internal builders (§3.4 criterion 4). |
| 5 | **Numerical instabilities with bounded parameters**. The sin transform (`SinParameterTransformation.cxx:38`) clamps internal values to `[-π/2, π/2)` minus a margin. Float64 vs C++ may handle the boundary case slightly differently, drifting MIGRAD iteration counts on bounded fits. The C++ chain rule sign for upper-only bounds is **negative** (`SqrtUpParameterTransformation.cxx:40–45`), so off-diagonal covariance signs must be tested explicitly. | Medium | Medium | Phase 1 scope; dedicated stress test fitting a Gaussian where the parameter starts at the bound. Use the same `prec.Eps2()` formula as C++ (`MnMachinePrecision.h:41`). |
| 6 | **Reference data generation cost**. Building the C++ benchmark/reference binaries is itself a multi-hour task (CMake + Minuit2 standalone). | Medium | Low | Commit JSON reference dumps under `test/reference_data/`; CI does not require the C++ build. Document the regen procedure under `tools/regen_reference.md`. Cap at 10 reference cases. **Pin to `57dc936` (v6.24.0)** — Decision Q8 locked. |
| 7 | **MnPosDef eigenvalue path divergence**. When the Hessian goes non-pos-def, `MnPosDef.cxx:80` calls `eigenvalues` on the *normalized correlation matrix* then adds to the diagonal of the original. Different eigenvalue routines (LAPACK `spev` vs Julia's default) may pick a different perturbation. | Low-Medium | Medium | Use `LAPACK.spev!` directly to match C++'s eigensolver choice. If still divergent, port the C++ Jacobi (`mnteigen.cxx`) verbatim for the pos-def branch only. |
| 8 | **API churn between Phases 0 → 1 → 3**. A Phase 0 user shouldn't see breaking changes. But the iminuit-style `Minuit(...)` constructor of Phase 3 is structurally different from the free-function `migrad(...)` of Phase 0. | Medium | Low-Medium | Phase 0 ships `JuMinuit.migrad` as the only public function. Phase 1 adds `Migrad`/`Hesse`/`Minos` types. Phase 3 adds `Minuit` as a new symbol. Nothing in Phase 0 is removed. Document this commitment in `CHANGELOG.md`. |
| 9 | **User FCN with mutation or global state** breaks threaded numerical gradients (Phase 2.2) even if workspace buffers are per-thread. | Medium | Medium | Document in Phase 2.2: `threaded_grad=true` requires a *pure* FCN (no captures over `Ref` / global / I/O). Provide a `JET`-style lint in CI to flag obvious cases. |
| 10 | **AD compatibility blocked by Phase-0 `Vector{Float64}` hard-coding**. If Phase 0 boundary signatures bake in `Vector{Float64}` everywhere, `ForwardDiff.Dual`/Enzyme paths in Phase 2.1 will require a disruptive rewrite. | Medium | Medium | Even in Phase 0, define the user-facing FCN call boundary via `eltype(x)` rather than literal `Float64`. Internal scratch and BLAS buffers stay Float64 in Phase 0 (numerical-gradient MIGRAD doesn't need AD); the *interface* is generic. |
| 11 | **GC latency in long cheap-FCN fits** dominates wall time even if `@allocated` for one isolated iteration is 0. The user's closure may allocate. | Medium | Medium | Phase 2.x ships a `gc_time_pct` diagnostic + advisory hint when `@allocated migrad(...)` > 0. The §3.4 gate measures the internal-only allocation; benchmark `bench_long_fit.jl` reports GC time separately. |
| 12 | **Benchmark drift** from compiler flags, libm, BLAS vendor/thread count, CPU governor invalidates the ≤ 1.5× claim across machines. | Medium | High | All of these go into `manifest.json` per §3.4.1. Pin to a designated reference machine (Open Question 10); cross-machine ratios are advisory not blocking. |
| 13 | **Result-history allocation regression**. C++ default `storage_level=1` appends a `MinimumState` per iteration. If Julia mirrors the default without policy, the zero-alloc gate quietly fails on real fits. | Medium | Medium | Phase 0 defaults `storage_level=0`. Phase 1 documents the per-iter alloc when `storage_level=1`. Test `test_zero_alloc.jl` runs with `storage_level=0` explicitly. |
| 14 | **BLAS implementation drift**. Julia 1.10 (OpenBLAS 0.3.21) vs 1.11+ (newer OpenBLAS) can shift DSYMV/DSYR result bits by 1 ULP, breaking 1e-12 reference traces. | Low-Medium | Low | Pin Julia version in `Project.toml`'s `[compat]`. Record tested OpenBLAS in `manifest.json`. Document the regen workflow when bumping. |
| 15 | **Cross-platform IEEE drift** in reference data: x86_64-built JSON dumped, Apple-Silicon CI loads, BLAS reduction order differs at the 1e-13 ULP, test fails on a "right answer". | Medium | Medium | Tolerance hierarchy: 1e-10 on final params (cross-platform), 1e-6 on `inv_hessian` element-wise (same-platform), 1e-3 on trace-element divergence after iteration 5 (any platform). Document the hierarchy in `tools/regen_reference.md`. |
| 16 | **MnHesse-inside-MIGRAD missing in Phase 0 with Strategy ≥ 1**. C++ `VariableMetricBuilder.cxx:138–173` calls `MnHesse` internally when `Strategy ≥ 1 && Dcovar > 0.05` (or Strategy == 2 unconditionally). iminuit default is Strategy 1. If a Phase 0 user runs with Strategy 1, they hit a missing code path. | High (if not locked) | High | **Lock Phase 0 to Strategy = 0** (now in §3 Scope). The full Strategy-0/1/2 trio is Phase 1 (§4 exit). |
| 17 | **`AbstractVector` type stability at the API boundary**. A user passing `x0::Vector{Real}` (heterogeneous) or `Vector{Float32}` derails workspace type stability. | Low | Medium | Enforce `Float64` at the API boundary: `x0 = convert(Vector{Float64}, x0)` at `migrad(...)` entry; document; defer arbitrary-precision to a Phase 2.x extension. |
| 18 | **`@inbounds` masking off-by-one** in workspace indexing — a test passes at n=4 but `@inbounds` hides a bug that fires at n=50 (or vice versa, n=2 special case). | Low | Medium | CI runs a `--check-bounds=yes` job in addition to the optimized job; do not gate the merge purely on the `@inbounds` job. |

---

## 9. Deferred

Listed explicitly so future maintainers know these *are* known, not
forgotten. Cross-referenced in `DEFERRED.md`.

- **Fumili minimizer** (`FumiliBuilder`, `FumiliMinimizer`,
  `FumiliErrorUpdator`, `FumiliGradientCalculator`, `FumiliChi2FCN`,
  `FumiliMaximumLikelihoodFCN`, `FumiliStandardChi2FCN`,
  `FumiliStandardMaximumLikelihoodFCN`, `FumiliFCNBase`,
  `FumiliFCNAdapter`, `MnFumiliMinimize.h/.cxx`). Specialized for χ² /
  max-likelihood with Jacobian-style updates; useful for some HEP fits
  (the ROOT-recommended minimizer for histogram fits). Out of scope
  until concrete user demand. ~3K LOC saved. Revisit in Phase 3 if
  IMinuit.jl users request it.
- **MPI support** (`MPIProcess.h/.cxx`, `MPI_SYNCH_PROC` guards in
  `Numerical2PGradientCalculator.cxx:102–214` and `MnHesse.cxx:240`).
  Replaced by `Distributed.jl` if/when needed in Phase 2+. Not for v1.0.
- **BFGS Hessian updator** (`BFGSErrorUpdator.h/.cxx`,
  `VariableMetricMinimizer(BFGSType)`). Use Davidon (DFP, the Minuit
  default). BFGS as a Phase 2 toggle.
- **CombinedMinimizer / ScanMinimizer** (`CombinedMinimizer.h/.cxx`,
  `CombinedMinimumBuilder.cxx`, `ScanMinimizer.h`, `ScanBuilder.cxx`).
  The "Combined" path is essentially MIGRAD + SIMPLEX fallback;
  provide as a Julia composition (`migrad(...) || simplex(...)`)
  rather than a new minimizer type. Phase 2.
- **`MnLineSearch::CubicSearch` and `MnLineSearch::BrentSearch`** at
  `MnLineSearch.cxx:321–820`. `#ifdef USE_OTHER_LS` paths, disabled by
  default in the C++ build. The default parabolic search is what every
  Minuit2 user runs.
- **ROOT serialization compatibility** (`G__DICTIONARY` in
  `FunctionMinimum.h:18–20`, `LinkDef.h`, the entire `inc/Math` and
  `inc/Fit` headers). Not relevant outside a ROOT process.
- **`MnTinyMain` / FORTRAN-era utility code** (`MnTiny.h/.cxx`,
  `TMinuit2TraceObject.cxx`). Compatibility shims for the old Fortran
  Minuit; obsolete.
- **OpenMP at the gradient level** (`#pragma omp parallel`/`for`
  blocks in `Numerical2PGradientCalculator.cxx`). Replaced by
  `Threads.@threads` in Phase 2.2.
- **`StackAllocator`** — moot in Julia (see §2.2).
- **`ABObj` and the expression-template layer** — replaced wholesale
  by Julia's broadcasting + in-place LAPACK calls.
- **Hand-rolled BLAS** (`mndaxpy`, `mndscal`, `mnddot`, `mndspmv`,
  `mndspr`, `mndasum`, `mnlsame`, `mnxerbla`) — use
  `LinearAlgebra.BLAS.*` directly.
- **`MnPlot` text plotting** (`MnPlot.cxx`, `mntplot.cxx`,
  `mnbins.cxx`) — Julia users get RecipesBase recipes in Phase 2.3
  plus an `mn_plot_text` helper for terminal use.
- **ParametricFunction integration** (`ParametricFunction.h/.cxx`) —
  ROOT function-object integration.

---

## 10. Open questions for the user

v1 had 8 open questions; this version locks 5 of them based on the
parallel review consensus and adds 8 new ones. Questions marked
**[DECIDED]** are locked; the rest still need user input before Phase 0
ends.

### Decisions locked in this revision

- **Q2 Bounds transform variant — [DECIDED: strict mirror sin/sqrt]**.
  C++ semantics for v1.0. Alternative transforms (tanh/logistic) add
  Phase 2 surface area but offer marginal benefit; the iminuit ecosystem
  expects sin/sqrt, and divergence would break copy-paste compatibility.
- **Q3 Default linalg storage — [DECIDED: dense `Symmetric{Float64,Matrix{Float64}}`]**.
  Day 4–7 benchmark confirms but the prior is strong: n ≤ 80 fits in L1
  either way; OpenBLAS DSYMV vectorizes better than DSPMV. Keep packed
  variant available behind a flag for benchmark comparison only.
- **Q5 Threads as default — [DECIDED: opt-in]**. `threaded_grad=false`
  default; `threaded_grad=true` opt-in. Predictable for benchmarking
  parity; matches iminuit/C++ default.
- **Q6 API parity vs idiomatic — [DECIDED: both]**. Property-style
  (`m.values`) via `Base.getproperty` overload **and** function-style
  (`values(m)`) accessors. Phase 3 ships both; one-line cost per
  accessor.
- **Q7 Status enums — [DECIDED: `@enum CovStatus`]**. Inferable,
  printable, matches Julia idiom.
- **Q8 Reference data freshness — [DECIDED: pin to `57dc936` (v6.24.0)]**.
  Document regen in `tools/regen_reference.md`. Address upstream bumps
  in v2.0.

### Still open

- **Q1 Internal vs external covariance**. Should the public Julia API
  return covariance / errors in *external* coordinates only (mirroring
  iminuit / IMinuit.jl), or also expose the internal representation
  (mirroring C++ `MnUserParameterState::IntCovariance()`) for advanced
  users? **Recommendation**: external as default; internal accessible
  via `JuMinuit.internal_covariance(m)` for debugging.
- **Q4 AD backend choice for Phase 2.1**. ForwardDiff.jl mature, composes
  with all Float64 code; Enzyme.jl faster on hot inner loops but rougher
  edges. **Recommendation**: ForwardDiff primary, Enzyme opt-in. Defer
  final lock-in until Phase 2.1 starts and AD ecosystem state is current.

### New questions (added by reconciliation)

- **Q9 BLAS vendor**. The 1.5× ratio depends on which BLAS the gate
  runs against. Options: OpenBLAS (Julia stdlib default), MKL.jl,
  AppleAccelerate. C++ side must match. Recommend OpenBLAS for
  cross-platform reproducibility unless the developer's reference
  machine has a strong reason otherwise.
- **Q10 Designated reference machine**. CPU vendor / model / Julia
  version / OpenBLAS version, recorded in `manifest.json` and on the
  CI badge. The ≤ 1.5× claim is reproducible only against this
  baseline. **Suggestion**: pick an x86_64 Linux machine the user owns
  + an Apple Silicon dev laptop as the two pinned environments.
- **Q11 License — [DECIDED 2026-05-25: LGPL 2.1+]**. Mirrors upstream
  C++ Minuit2. See DR-010 in `DESIGN.md`. `LICENSE` file added
  (full LGPL 2.1 text); every Julia source carries `# SPDX-License-
  Identifier: LGPL-2.1-or-later`.
- **Q12 Phase 1 trace mechanism**. The line-by-line iteration-
  equivalence test (Phase 1 exit) needs C++ traces. Options:
  (a) `MnTraceObject` (requires CMake rebuild + minor C++);
  (b) instrument the existing C++ MIGRAD with `printf` directly
  (faster, less precise).
- **Q13 Public `ErrorDef`/`up` mutation API**. iminuit exposes
  `m.errordef = X`; should JuMinuit allow post-construction mutation
  of `up`? Defaults: 1.0 (χ²) / 0.5 (NLL).
- **Q14 Minimum Julia version + dependency policy**. Pin in
  `[compat]`: `julia = "1.10"`? `StaticArrays`, `ForwardDiff`, `Enzyme`
  — runtime deps or weak deps via Requires.jl / package extensions?
- **Q15 Meaning of "v0.1 ready"**. Phase 0 exit = "MIGRAD + numerical
  gradient demo only" or "MIGRAD + HESSE + bounds + named parameters"
  (effectively Phase 1)?
- **Q16 GitHub repo public/private flip**. Currently private; when
  does it go public? Recommend: after Phase 0 gate passes (avoids
  attracting early users to an incomplete API).

---

## 11. Critical files for Phase 0 implementation

The four Julia files and two C++ files that together define the Phase 0
surface and the oracle against which everything is measured:

- `src/migrad.jl` — the MIGRAD loop (outer + inner)
- `src/davidon.jl` — the DFP update (correctness-critical; additive
  rank-1)
- `src/gradient.jl` — the numerical gradient (performance-critical)
- `src/linesearch.jl` — the parabolic line search
- `reference/Minuit2_cpp/src/VariableMetricBuilder.cxx` — line-by-line
  oracle for the Julia MIGRAD loop
- `reference/Minuit2_cpp/src/DavidonErrorUpdator.cxx` — oracle for the
  additive DFP update
