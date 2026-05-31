# Changelog

All notable changes to JuMinuit.jl. Follows [Keep a Changelog](https://keepachangelog.com/)
+ [Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-05-31

First public release. A complete error-analysis suite, a Julia-native
cost-function family, and an Optim.jl bridge land on top of the
now-C++-fidelity-audited core.

### Added

- **Julia-native cost functions** — `LeastSquares`, `UnbinnedNLL`, `BinnedNLL`,
  `ExtendedUnbinnedNLL`, `ExtendedBinnedNLL`, composable with `CostSum` (`+`),
  each carrying its `errordef`. Interoperate with the IMinuit.jl `chisq` / `Data`.
- **Error analysis beyond HESSE/MINOS** — Monte-Carlo Δχ² confidence regions
  (`get_contours_samples` / `contour_df_samples`, with `delta_chisq` / `chisq_cl`),
  data-resampling errors (`bootstrap`, `jackknife`, `correlation`), and
  **multi-modal solution detection** (`find_solution_modes`, via a Clustering.jl
  extension). `contour_parameter_sets` exposes the full parameter vector at every
  contour point.
- **Alternative-minimizer bridge** — `optim(m)` / `minimize_with(m, opt)` route a
  fit through any Optim.jl optimizer (the Julia analogue of iminuit's `scipy()`),
  via an Optim.jl extension.
- `hesse!` as a bang-named alias of `hesse` (consistent with `migrad!` / `minos!`).
- Threaded numerical gradient — a 3-way `threaded_gradient` switch
  (`false` / `true` / `:auto`) with a thread-safety pre-flight (`is_thread_safe`,
  `ThreadSafetyError`); `:auto` probes once (memoized), threading only when the
  FCN is safe and otherwise falling back to serial with a single warning (#32).
- **Plot recipes for the error-analysis suite** (RecipesBase, Plots/Makie-agnostic)
  — `plot(...)` on Monte-Carlo Δχ² samples (`get_contours_samples`),
  `BootstrapResult` / `JackknifeResult` distributions, and multi-modal
  `SolutionModes`, alongside the existing contour / MINOS / `FunctionMinimum`
  recipes (#35).

### Performance

- In-place `int_to_ext_vector!` removes the per-FCN-call allocation on the
  bounded MIGRAD path — toy fits now run at ~1 allocation/fit (#34).
- In-place `make_posdef!` reuse in HESSE, plus linalg / scratch micro-opts (#36).
- Extended PrecompileTools workload covering the cost-function classes, the
  error-analysis layer, and the package extensions — lower first-call latency
  (TTFX) (#33).

### Changed

- Defaults aligned with iminuit: `Strategy(1)` and `4·eps` machine precision.
- Documentation reorganised for a public release — a Documenter manual
  (tutorials, cost functions, error analysis, API reference); development and
  audit notes moved under `docs/dev/`.

### Fixed

- C++ Minuit2 v6.24.0 fidelity audit closed end to end: `MnMachinePrecision`
  (`4·ε`), Simplex / negative-g2 / positive-definite handling, the MnContours
  direction-switch retry, CheckGradient diagnostics, and covariance squeeze.

## [0.2.0-alpha] — 2026-05-25

Phase 1.x deep refinements + Phase 1 exit-gate + Phase 3 polish.
Builds on 0.1.0-alpha; 1012/1012 tests passing (Julia 1.12, 4 threads).

### Added

#### Phase 1.x — deep algorithmic refinements
- **D4** (`free_covariance`): n_free × n_free covariance sub-block accessor
  matching C++ `MnUserParameterState::Covariance()` shape. The default
  `ext_covariance` remains the n_total × n_total view with zero rows/cols
  for fixed parameters (convenient indexing); `free_covariance` is the
  C++-shape alternative on demand.
- **D5** (`int2ext_error`): C++ `MnUserTransformation::Int2extError`
  two-sided symmetric average for bounded parameter external errors:
  `0.5·(|du1| + |du2|)` with the double-bounded saturation clamp
  (`upper - lower` when err > 1). Captures the nonlinear remapping
  near bounds where the Jacobian-only formula under-reports.
- **A3/A4** (`function_cross`): full C++ 3-point parabolic search per
  `MnFunctionCross.cxx:117-507`. New helpers:
    - `_parabola_fit3` (Lagrange fit through 3 points)
    - `_parabola_solve_for_aim` (quadratic root + slope-sign root pick)
    - `_three_point_classify` (noless, ibest, iworst, ileft, iright,
      iout with default_ibest tie-break)
  Both `function_cross` and `function_cross_multi` now share a single
  `_cross_core` (probe-closure dispatch). L300/L460/L500 control flow
  ported as `@label`/`@goto`. Crossing convergence uses the hardcoded
  `tlr = 0.01` per C++ (user `tlr` controls only inner-MIGRAD).

#### Phase 1 exit gate — Strategy(1)/(2) inner-Hesse refinement
- New outer `do-while` wraps the existing DFP inner loop. After DFP
  convergence, if `Strategy == 2` (always) or `Strategy == 1 && Dcovar
  > 0.05`, call MnHesse on the converged state. If HESSE moves edm
  above tolerance and above machine accuracy, re-iterate the inner
  DFP loop. Per-pass budget bump `maxfcn → floor(maxfcn × 1.3)` on
  second pass (`VariableMetricBuilder.cxx:182-184`).
- Removed the Phase 0 `Strategy ≥ 1 throws` guard from `seed_state`.

#### Phase 3 — Polish + Documentation
- iminuit-style pretty-print boxes for `FunctionMinimum` + `MinosError`.
  Two-column 71-char Unicode tables with validity flags + parameter
  table (idx / name / value / Hesse err).
- Documenter.jl docs site (`docs/`): index, three tutorials (quickstart,
  bounded, MINOS+contours), API reference, internals (algorithm map,
  cross-search walk-through, multi-agent review history).
- GitHub Actions workflows:
    - `ci.yml`: test matrix on Julia 1.10 + 1, Ubuntu + macOS +
      macOS-aarch64; codecov upload.
    - `docs.yml`: Documenter build + gh-pages deploy on push to main.

### Fixed

#### Multi-agent review BLOCKING findings (round 5)

Independent Opus code-reviewer caught two BLOCKING parity gaps in
the A3/A4 first cut:

- **BLOCKING #1** — Crossing convergence used user's `tlr` (default 0.1)
  but C++ `MnFunctionCross.cxx:38-40` OVERRIDES to 0.01. Fixed by
  hardcoding `tlf = 0.01·up`, `tla_base = 0.01` in `_cross_core`.
- **BLOCKING #2** — Missing "new straight line thru first two points"
  fall-through (C++ lines 343-351). The cases `noless ∈ {0, 3} &&
  ibest == 3` (third probe is best, all 3 same-side of aim) silently
  fell to L500 with an unconverging parabola fit. Added the ELSE
  branch: `a[iworst] = a[3]; dfda = (f[2]-f[1])/(a[2]-a[1]);
  @goto l460_extrapolate`.

Folded IMPORTANT findings:
- L300-redo step counter is now a local `l300_step_count` reset to 0
  on each L300 entry (was using cumulative ipt → overshooting steps).
- `_three_point_classify` accepts `default_ibest` keyword (3 for the
  initial classifier, 1 for L500) to match C++ tie-break semantics.

### Stats

- Tests: 888 → 1012 (+ 124).
- Source files: src/ + tests + docs + CI workflows.
- 5 rounds of parallel multi-agent review (codex gpt-5.5 xhigh +
  native Opus subagent) caught 5+ BLOCKING bugs that would have
  shipped silently.

---

## [0.1.0-alpha] — 2026-05-25

First substantial alpha release. Phase 0 PoC + Phase 1 batch 1-3 +
Phase 2.1/2.4/2.5 + Phase 3 first cut shipped. 35 commits. 888/888
tests passing. Aqua + JET clean.

### Added

#### Phase 0 — Core MIGRAD
- `MachinePrecision`, `Strategy` (levels 0/1/2 with C++-exact constants).
- `MinimumState`, `CovStatus` enum, four state types (Parameters,
  Error, Gradient, full State).
- `CostFunction{F,T}` with parametric closure specialization +
  `Ref{Int}` call counter.
- Symmetric storage convention (`:U`) + BLAS-backed kernels in
  `linalg.jl`: `sym_mul!`, `sym_rank1_update!`, `sym_invert!`,
  `sym_eigvals`, `sum_sym`, `add_sym!`.
- Numerical gradient (`InitialGradientCalculator` cold-start +
  `Numerical2PGradientCalculator` two-point central-diff refinement).
- DFP Hessian update (rank-2 base + additive rank-1 correction when
  `delgam > gvg`).
- EDM (Expected Distance to Minimum) estimator.
- `MnPosDef` positive-definiteness enforcement via eigenvalue
  perturbation.
- `NegativeG2LineSearch` for the seed when initial g2 has negative
  entries.
- Parabolic 1D line search (`MnLineSearch` minus the deferred
  cubic/Brent variants).
- `MnSeedGenerator` for the initial MIGRAD state.
- Full MIGRAD loop with `FunctionMinimum` result type.

#### Phase 1 — Bounds + MINOS + Contours + HESSE
- `transform.jl`: sin / SqrtUp / SqrtLow / identity bound transforms
  matching C++ exactly (including the sign-aware SqrtUp derivative).
- `parameters.jl`: `MinuitParameter` + `Parameters` (collapsed
  `MnUserParameters` + `MnUserTransformation`).
- `hesse.jl`: full numerical Hessian with diagonal multiplier loop,
  off-diagonal pass, `MnPosDef` + invert, status flag handling.
- `covariance_squeeze.jl`: drop a row+col from a symmetric matrix via
  invert → squeeze → invert back, with diagonal fallback on failure.
- `function_cross.jl`: parabolic root-find with inner re-minimization;
  used by MINOS.
- `minos.jl`: asymmetric ±σ errors with `MinosError` result type.
- `contours.jl`: 2D 1σ contour via ellipse approximation from MINOS +
  off-diagonal covariance.
- `migrad_bounded.jl`: bound-aware MIGRAD via `Parameters` wrapper;
  internal MIGRAD operates in unbounded coords, user FCN sees
  external coords; full external covariance back-conversion via
  Jacobian chain rule.

#### Phase 2 — Polish
- `ad_gradient.jl` (2.1): `CostFunctionWithGradient{F,G,T}` for
  user-supplied or AD-produced gradients; ForwardDiff integration.
- `serialize.jl` (2.5): `to_dict` / `minimum_summary_from_dict` for
  JSON / JLD2 roundtrip of all result types.
- `precompile_workload.jl` (2.4): PrecompileTools workload reducing
  TTFX by ~50% on typical MIGRAD paths.

#### Phase 3 — User API
- `minuit.jl`: iminuit-style `Minuit` mutable struct with
  `migrad!`, `minos!`, `contour` methods and `m.values`, `m.errors`,
  `m.fval`, `m.edm`, `m.nfcn`, `m.valid`, `m.covariance` property
  access (via `Base.getproperty`).

#### Tooling
- `tools/cpp_trace_harness.cxx`: C++ Minuit2 reference-data generator
  producing JSON oracles for unbounded + bounded + fixed-parameter
  benchmark cases.
- `tools/regen_reference.sh`: build + run wrapper.
- `benchmark/cpp/cpp_bench.cxx`: wall-time benchmark of C++ Minuit2
  for §3.4 Criterion 2 cross-implementation comparison.
- `benchmark/compare_cpp.jl`: pulls Julia + C++ medians, prints
  ratio table, computes verdict.
- `benchmark/bench_migrad_suite.jl` + `benchmark/perf-config.toml`:
  julia-perf Level-2 evidence-gate suite.
- `scripts/run_gate.sh`: gate driver.

### Verified

- **Phase 0 §3.4 Criterion 1**: Quad-4D matches C++ Minuit2 reference
  JSON to fval ≤ 1e-15, params to 1e-10. Rosenbrock cases within
  Strategy(0) cross-impl variance.
- **Phase 0 §3.4 Criterion 2**: Julia ≤ 0.887× C++ wall time on
  every benchmark in the §3.3 corpus (max ratio 0.887×, mean 0.47×).
- **Phase 0 §3.4 Criterion 4**: Aqua + JET clean on the public API
  (`migrad(::Function, ::Vector{Float64}, ::Vector{Float64})`).
- **Phase 1 bounded oracle parity**: 4 bounded reference cases (Sin /
  upper-only / lower-only / fixed-parameter) match C++ Minuit2 output
  on fval, free-parameter values, and NFcn within documented Strategy(0)
  tolerance. External covariance verified symmetric.

### Audit trail

- Four rounds of independent parallel review:
  1. v1 → v2 ROADMAP reconciliation (caught a real `sum_sym`
     signed-vs-absolute blocking bug in linalg).
  2. Phase 0 MIGRAD integration (caught `reached_call_limit` AND-gate
     bug, plus 9 surgical issues).
  3. Phase 0 hot-path kernel retroactive review.
  4. Phase 1 mid-phase (caught the `initial_int_errors` Taylor vs
     two-sided perturbation divergence at bounds, plus 5 minor).
  5. Phase 1 batch 2+3 (caught D3 covariance asymmetric-read bug, C-2
     contour sign-blind selector, A7/B4 bounded coord-frame leak,
     A5 strategy no-op, B2 MinosError field semantics).

  All blocking and high-priority findings applied as commits with
  source-cited diffs.

### Deferred

#### Phase 1.x
- `function_cross` C++ 3-point parabolic algorithm parity (Julia
  uses a simplified 2-point fit + replace-worst).
- Multi-parameter `function_cross` for the C++-exact (non-ellipse)
  contour algorithm.
- Strategy(1+) `HessianGradientCalculator` refinement inside
  `MnHesse`.
- Bounded MINOS through the internal-coord-wrapped CostFunction
  (currently the unbounded MINOS path is wired; bounded MINOS via
  `Minuit.minos!(m, par)` uses the wrapped CF but doesn't refine
  bounds at the parabolic-cross step).
- `Int2extError` two-sided bounded errors (currently uses
  Jacobian-diagonal sqrt; matters near bounds).
- Variable-sized external covariance to match C++
  `MnUserParameterState` shape (currently full n_total × n_total
  with zero rows for fixed parameters).

#### Phase 2
- 2.2 Threads-parallel numerical gradient.
- 2.3 Plot recipes (RecipesBase).

#### Phase 3 polish
- Full iminuit pretty-print parity.
- `m.errors[name] = ...` setter API.
- Documentation site (Documenter.jl).

[0.3.0]: https://github.com/fkguo/JuMinuit.jl/releases/tag/v0.3.0
[0.2.0-alpha]: https://github.com/fkguo/JuMinuit.jl/releases/tag/v0.2.0-alpha
[0.1.0-alpha]: https://github.com/fkguo/JuMinuit.jl/releases/tag/v0.1.0-alpha
