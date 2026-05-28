# JuMinuit vs C++ Minuit2 — Gap Audit

Date: 2026-05-27 · Reference: GooFit/Minuit2 v6.24.0 @ 57dc936 ·
Comparison: `reference/Minuit2_cpp/{inc,src}/*` vs `src/*.jl`

This document supplements [ROADMAP §9 "Deferred"](../ROADMAP.md) — it
lists items that are **either intentionally deferred (cross-listed
with ROADMAP §9) or NOT yet documented as deferred** and may need
attention.

The two pre-existing follow-ups from
[`BenchmarkExamples/RESULTS.md`](../BenchmarkExamples/RESULTS.md)
(MINOS early termination on tight wells; IAM 9-LEC early-termination
divergence) are tracked separately and not re-listed here.

## Summary

| category | count | notes |
|---|---|---|
| ✓ ported | ~50 | full MIGRAD/HESSE/MINOS/CONTOURS/SIMPLEX/SCAN spine + bounds + Davidon + NegativeG2 + EDM + posdef + covariance-squeeze + linesearch + parameter transforms + Minuit struct + iminuit_compat |
| ◻ deferred (covered by ROADMAP §9) | ~22 | Fumili family, MPI, BFGS, ROOT shims, hand-rolled BLAS, ABObj layer, ... |
| ❌ MISSING (not in ROADMAP §9) | **6** | ← actionable, see below |
| ~ PARTIAL implementations | **5** | "first cut" items intended but not finished |

## ❌ MISSING — actionable

### M1 · `print_level` / iteration trace (BLOCKING for iminuit drop-in)

C++: `MnTraceObject.h/.cxx` + `MinimumBuilder::TraceIteration` (cf.
`VariableMetricBuilder.cxx:47`).

JuMinuit state: `Minuit.print_level` is stored as a field (set/get
via `setproperty!`) but **never consumed** by `migrad.jl` / `hesse.jl`
/ `minos.jl`. No `@debug` / `@info` emitted from the inner loops.

Impact: iminuit users routinely set `m.print_level = 2` to debug a
non-converging fit. JuMinuit silently swallows the setting — looks
like a bug to the user. ROADMAP §7 originally tagged this "Phase 1 /
2, mandatory for the line-by-line iteration-equivalence test"; the
test never landed.

Recommended fix: wire `print_level` through `_migrad_loop`,
`hesse`, `_cross_core`. Level 1 → per-iter `(iter, fval, edm, dcovar)`;
level 2 → full inner-loop trace; level 3 → also gradient + parameter
vectors. Either via stdlib `Logging` LogLevel filters or an optional
`trace::Function` kwarg.

### M2 · `mn_plot_text` ASCII plot helper (NICE-TO-HAVE)

C++: `MnPlot.cxx` + `mntplot.cxx` + `mnbins.cxx`.

ROADMAP §9 line 780 explicitly promised: *"Julia users get RecipesBase
recipes in Phase 2.3 plus an `mn_plot_text` helper for terminal use."*
The recipes shipped (`src/plot_recipes.jl`); the helper did not.

Impact: terminal / SSH / CI users with no GUI backend can't visually
sanity-check a MINOS / contour result.

Recommended fix: add `mn_plot_text(::ContoursError; width=60,
height=20)` + `mn_plot_text(::Vector{Tuple{Float64,Float64}})`
ASCII renderers per `mntplot.cxx` + `mnbins.cxx`. ~80 LOC.

### M3 · Per-parameter mutators on `Minuit` (BLOCKING for advanced drop-in)

C++: `MnUserParameters.h:75-95` + `MnApplication.cxx:117-180` —
`Fix(i)`, `Release(i)`, `SetValue(i,v)`, `SetError(i,v)`,
`SetLimits(i,lo,up)`, `RemoveLimits(i)`, `Add(name,val,err)`,
`SetName(i,name)` with name-overloads.

JuMinuit state: `Minuit` exposes **bulk** setters via `setproperty!`
(`m.values=[...]`, `m.fixed=[...]`, `m.limits=[...]` at
`src/minuit.jl:751-770`). No **per-parameter** mutators by index or
name. Bulk replacement drops `m.fmin` and `m.minos_errors`.

Impact: iminuit's idiomatic pattern is `m.fixed["alpha"] = True` for
fix-fit-release-fit profile-likelihood scans. JuMinuit forces full
vector replacement, breaking common interactive workflows.

Recommended fix: add `fix!(m, par)`, `release!(m, par)`,
`set_value!(m, par, v)`, `set_error!(m, par, e)`,
`set_limits!(m, par, lo, up)`, `remove_limits!(m, par)` (each
dispatching on `Int`/`String`). ~120 LOC; can route through existing
bulk helpers.

### M4 · `MinosError.UpperState()` / `LowerState()` (NICE-TO-HAVE)

C++: `MinosError.h:73-74` — full `MnUserParameterState` snapshot at
the ±σ MINOS crossing point.

JuMinuit state: `MinosError` struct (`src/minos.jl:48-58`) carries
errors, validity flags, and nfcn but **no per-parameter snapshot** at
the crossing endpoints. Boolean `*_par_limit` partially covers
"hit a bound" but loses parameter values.

Impact: HEP correlated-systematic studies and "at-bound" diagnostic
inspection need the crossing-point state. Currently inaccessible.

Recommended fix: add `upper_state::Union{Nothing,Vector{Float64}}`
and `lower_state::Union{Nothing,Vector{Float64}}` fields (full
parameter vector at the ±σ minimization endpoint). Populate inside
`_minos_external_via_function_cross`.

### M5 · `MnSeedGenerator` user-supplied covariance branch (NICE-TO-HAVE)

C++: `MnSeedGenerator.cxx:63-67` — when `state.HasCovariance()` the
seed uses the user-supplied `MnUserCovariance` and sets `dcovar=0`
instead of the default `dcovar=1.0`.

JuMinuit state: `src/seed.jl:11` documents *"No user-supplied
covariance prior (dcovar = 1.0 always)."* The `seed_state` /
`warm_restart_state` functions do not accept a prior covariance
argument.

Impact: warm restarts at slightly perturbed starting points pay
seed-iteration overhead they shouldn't. iminuit's `m.covariance = ...`
setter feeds back into the next MIGRAD seed via this branch.

Recommended fix: `seed_state(cf, x0, errs, strategy, prec; prior_cov::Union{Nothing,Symmetric{Float64,Matrix{Float64}}}=nothing)`.
~40 LOC. ROADMAP §7 marked this Phase 1+; it slipped past Phase 1.

### M6 · `FunctionMinimum::States()` iteration history (NICE-TO-HAVE)

C++: `BasicFunctionMinimum.h:109,165` — vector of per-iteration
`MinimumState`, gated by `storage_level`.

JuMinuit state: `FunctionMinimum` (`src/result.jl:33-42`) stores only
the final `state` + `seed`, no history. `result.jl:8-10` notes Phase 1
will add `storage_level=1`, but the kwarg/field doesn't exist
anywhere yet.

Impact: needed for the ROADMAP §4 exit-criterion "line-by-line
iteration-equivalence test" against C++. Also for publication-grade
convergence plots.

Recommended fix: add `states::Vector{MinimumState}` field +
`storage_level::Int` kwarg on `migrad` (default 0; opt-in 1 appends).
~50 LOC + carve-out from the zero-alloc gate.

## ~ PARTIAL implementations

### P1 · `HessianGradientCalculator` (Strategy ≥ 1 HESSE gradient refinement)

C++: `HessianGradientCalculator.h/.cxx` ~220 LOC; called from
`MnHesse.cxx:228-236` between diagonal and off-diagonal Hessian passes
when `Strategy ≥ 1`.

JuMinuit state: `src/hesse.jl:188-198` documents the no-op;
`strategy.hessian_grad_ncycles` is set but never consumed.

Impact: Strategy(1) HESSE produces slightly less accurate errors than
C++ on FCNs with significant g2 instability. Quadratics + smooth fits
are invisibly equivalent; bumpy χ² landscapes deviate. iminuit users
running `m.hesse()` on a tricky fit get marginally worse error bars
than they would with upstream Minuit2.

Recommended: port `HessianGradientCalculator` (~150 LOC Julia).

### P2 · HESSE with AD-gradient (couples with P1)

C++: `MnHesse.cxx:118-126` — when analytical gradient is supplied,
refreshes `gst`/`dirin`/`g2` triplet via `InitialGradientCalculator`.

JuMinuit state: when `m.cfwg !== nothing` (user supplied a `grad=...`)
the HESSE refresh of the numerical companions is skipped
(`src/hesse.jl:202-209`).

Impact: AD-gradient + `hesse()` users only. g2/gst carry their
last seed-time numerical estimates; for stiff Hessians this
biases EDM and the (also-missing) Strategy ≥ 1 refinement.

Recommended: couple fix with P1.

### P3 · `contour` default = ellipse vs `contour_exact` (potential drop-in surprise)

C++: `MnContours.cxx:52-78,125-178` — re-minimization contour is the
default.

JuMinuit state: `contour_exact` IS implemented C++-faithfully
(`src/contours.jl:89-241`). The default `contour(...)` is an
ellipse-from-MINOS-errors approximation. `mncontour` routes through
`contour_exact`, but the bare `contour` does not.

Impact: a user copying iminuit's `m.contour("x", "y")` (note: iminuit
also calls its ellipse `contour` and its re-minimization
`mncontour` — so behavior matches!) gets an ellipse. This is
**actually iminuit-compatible** — `m.contour` in iminuit is the same
ellipse approximation. Confirm by inspection that JuMinuit's naming
follows iminuit, NOT C++. If so, this is documented behavior, not a
gap; promote to "✓ Verified API parity with iminuit" instead.

Recommended: verify and document, no functional change.

### P4 · `MnUserCovariance` packed-storage layout (COSMETIC, no action)

The Julia API returns `Symmetric{Float64,Matrix{Float64}}` (per
DR Q3) rather than C++'s flat lower-triangular `fData` vector.
Matters only for binary interop with ROOT/iminuit which JuMinuit
explicitly does not target. No action.

### P5 · `minos(...; sigma=k)` for k ≠ 1 (NICE-TO-HAVE)

JuMinuit state: `src/minuit.jl:1003-1004` throws `ArgumentError` for
`sigma != 1` ("Phase 1.x deferred").

Impact: 5σ discovery / 95 % CL upper-limit studies need 2σ or higher.
iminuit and C++ Minuit2 both support it via `up · sigma²` scaling in
`MnFunctionCross`.

Recommended: thread `up_scale = sigma^2` through to
`function_cross_external` (the aim level becomes `f_min + up·sigma²`).
~10 LOC.

## Verified items (spot-checks)

These were explicitly verified during the audit and are CORRECT
relative to C++:

| item | status |
|---|---|
| `Strategy(2)` HESSE-inside-MIGRAD when `Dcovar > 0.05` | ✓ Implemented at `src/migrad.jl:749-756`, matches `VariableMetricBuilder.cxx:138` |
| `NegativeG2LineSearch` unconditional call in seed | ✓ `src/seed.jl:98-99` mirrors `MnSeedGenerator.cxx:80` |
| `MnPosDef` full port | ✓ `src/posdef.jl` (178 LOC vs C++ 108) |
| `MnGlobalCorrelationCoeff` exposed | ✓ `global_cc(m)` / `global_cc(cov)` |
| `MnEigen` exposed | ✓ `eigenvalues(m)` / `eigenvalues(cov)` |
| `MnParameterScan` + `MnScan` + `ScanBuilder` | ✓ all in `src/scan.jl` |
| `FCNGradAdapter` numerical-fallback + user-supplied | ✓ `Minuit(f, x0; grad=g)` dispatches via `CostFunctionWithGradient` |
| `CombinedMinimizer` as `migrad ∘ simplex` composition | ✓ `src/minuit.jl:382-409` (iminuit-style `_robust_low_level_fit` retry loop) |
| `ContoursError` struct | ✓ full struct with `par_x`/`par_y`/`points`/`minos_x`/`minos_y`/`nfcn`/`valid` |

## See also

- [ROADMAP.md §9 "Deferred"](../ROADMAP.md) — items intentionally not
  ported with rationale.
- [BenchmarkExamples/RESULTS.md](../BenchmarkExamples/RESULTS.md) —
  two algorithm bugs discovered via real-fit benchmarks, tracked
  separately.
