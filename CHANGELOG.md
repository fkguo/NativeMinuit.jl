# Changelog

All notable changes to JuMinuit.jl. Follows [Keep a Changelog](https://keepachangelog.com/)
+ [Semantic Versioning](https://semver.org/).

## [0.5.2] — 2026-06-13

### Fixed

- **Bounded-parameter Hesse errors were under-reported by √(2·up)** (errordef
  scale). The bounded int→ext error transform fed `int2ext_error` the raw
  `√V_int[i,i]` instead of the errordef-scaled internal 1σ error
  `√cov(i,i) = √(2·up·V_int[i,i])` that C++ Minuit2 uses
  (`MnUserParameterState.cxx:142` passes `√(2·up·InvHessian(i,i))`). For a χ²
  fit (`up = 1`) every **bounded** parameter's reported error (`m.errors`,
  `m.params.pars[i].error`) was a factor of √2 too small; the unbounded path was
  always correct. Now matches C++/iminuit (e.g. the issue-#38 example's bounded
  `a` goes from 0.7046 to ≈0.993). The C++ JSON oracle test now also asserts
  `ext_errors`, which would have caught this. MINOS/contour errors were
  unaffected (they already used the `2·up·inv_hessian` scaling).

- **`m.params` now reflects the fit** ([#38](https://github.com/fkguo/JuMinuit.jl/issues/38)).
  After `migrad!` (and `minos!`/`hesse!`), `m.params.pars[i]` exposed the
  *initial* value/step instead of the fitted ones, silently disagreeing with
  the already-correct `m.values[i]` / `m.errors[i]` views. The public
  `m.params` property now returns a fit-overlaid `Parameters` whenever a fit is
  cached, so `m.params.pars[i].value == m.values[i]` and
  `m.params.pars[i].error == m.errors[i]` (iminuit parity, where `m.params`
  reflects the converged state). Structure (names, bounds, fixed flags, index
  maps) is unchanged, and a fresh — or `reset(m)`'d — `Minuit` still reports the
  constructor-time configuration. Internal seed / retry / resume / cold-start /
  mutator code reads the raw constructor config via a new private
  `_init_params(m)`, so MIGRAD resume, the retry length scale, `reset(m)`, and
  resample / `contour_grid` clones keep using the user's original step sizes.

## [0.5.1] — 2026-06-11

### Added

- **Likelihood-ensemble MCMC + marginal quantile bands** (`src/mcmc.jl`) — the
  second leg of the error-analysis triangulation (profile extremization ↔
  ensemble quantiles ↔ MINOS), absorbing the hand-rolled Metropolis chains
  used in downstream analyses into a native feature. iminuit has no analogue
  (Python users bolt on emcee).
  - `mcmc_sample(m; nsteps=52_000, burn=2_000, thin=25, proposal=:hesse,
    scale=0.3, target_accept, seed/rng)` — random-walk Metropolis on the
    **exact FCN** (`exp(−Δfcn/(2·up))`, so χ² and `−log L` fits are handled
    uniformly), started at the best fit. Parameter `limits` are enforced by
    **rejection** (the chain samples the likelihood truncated to the allowed
    box — the one-sided mass pile-up at an active boundary is the correct
    marginal, not a bug); fixed parameters never move; non-finite FCN values
    are never accepted; `m.nfcn` is untouched. Proposal options: `:hesse`
    (fit covariance, correlation-aware; falls back to `:errors` with a warning
    when the covariance is unreliable), `:errors` (per-coordinate parabolic
    errors — the classic hand-rolled choice), an explicit per-coordinate σ
    vector, or an explicit covariance matrix. With `target_accept` the scale
    is adapted during burn-in only, then frozen (fixed kernel for the kept
    chain). Returns a `LikelihoodEnsemble` (samples in full external
    coordinates + FCN values + post-burn acceptance + metadata; iterable as a
    collection of parameter vectors).
  - `quantiles(ens, f; p=(0.16, 0.5, 0.84))` — marginal quantiles of a scalar
    derived quantity over the ensemble; `quantile_band(ens, f, xs;
    p=(0.16, 0.84), curve=false)` — pointwise quantile band of a curve
    (`curve=true` evaluates whole curves, one call per member, for expensive
    models).
  - `save_ensemble` / `load_ensemble` — plain-text persistence (`#` header +
    `fval p₁ p₂ …` rows) as reusable error sets; exact float round-trip;
    reads existing hand-rolled ensemble files (foreign headers ⇒ placeholder
    metadata, with `names`/`up` override keywords).
  - Docs: `error_analysis.md` gains the marginal-quantile-band vs
    profile-envelope-band comparison (the constructions legitimately separate
    at parameter limits — the band may exclude the best fit), the
    high-dimensional volume effect (`P(Δχ²₉ ≤ 1) ≈ 5.6e-4` — why a likelihood
    chain is *not* a `Δχ² ≤ 1` region sampler and vice versa), and the
    field-tested tuning recipe (step ≈ 0.25–0.35 × HESSE σ → acceptance
    0.2–0.3).
  - Tests (`test/test_mcmc.jl`): chain calibration on a known-covariance
    Gaussian target (mean/covariance/`Δχ² ~ χ²ₙ` PIT), analytic
    truncated-Gaussian quantiles on a boundary-pinned target (all samples
    in-box, one-sided pile-up, band excludes the best fit by construction),
    `up = 0.5` equivalence, burn-in scale adaptation both directions,
    fixed-parameter invariance, exact save/load round-trip + hand-rolled
    format compatibility, seeded reproducibility, and the
    unreliable-covariance fallback.
- **`extremize(m, f; cl, seeds, …)` — "MINOS for a function"** (no iminuit
  equivalent): the exact profile interval `[min f, max f]` of a **derived
  scalar** `f(θ)` over the region `FCN ≤ FCN_min + delta_chisq(cl, 1)·up`,
  all free parameters varied simultaneously, limits/fixed honoured. For
  `f(θ) = θ[i]` it reproduces MINOS; in the linear-Gaussian limit it is the
  projection theorem `f̂ ± √(Δχ²·cᵀCc)` (verified digit-level in the tests).
  Implementation: exterior-penalty MIGRAD with a stiffening λ-continuation
  ladder (`1 → 100 → lambda`, warm-restarted up to `rounds` times — a single
  stiff fit demonstrably stalls on the penalty shell), an acceptance gate
  `FCN ≤ bound + accept_tol·up`, and a strictly LOCAL pull-back of the
  endpoint onto the boundary (never across a χ² barrier). Returns an
  `ExtremizeResult` with the extremal parameter vectors and a **per-seed
  audit trail** (`diagnostics`: converged/accepted/`f`/winner per seed).
  Multiple seeds are load-bearing: a single seed stops at a local tangency
  and silently under-covers on a multi-corridor region — pass ensemble
  extremes via `seeds` and audit the diagnostics (regression-tested on a
  two-corridor toy where the default seed provably misses the far corridor).
- **`profile_band(m, f, xs; cl, seeds, warm, passes, …)`** — the pointwise
  profile-likelihood **error band** of a curve family `f(x, θ)` (`x` first,
  `θ` the full parameter vector — the package-wide `model(x, …)` convention
  and the same callback shape as `quantile_band`): per grid
  point the same Δχ²(`ndof = 1`) extremization, swept with warm starts
  (neighbour's extremal parameters seed the next point), alternating
  forward/reverse passes keeping the better envelope, and a
  contains-the-best-fit guarantee (`include_best`, on by default — the
  construction property posterior-quantile bands lack at parameter limits).
  Returns a `ProfileBand` with the envelope, per-point extremal vectors,
  the best-fit curve, a failure counter (`nfail`) and per-point diagnostics.
  Pointwise agreement with the analytic band is verified on a correlated
  3-parameter Gaussian.
- Both accept `cl` in the package-wide `delta_chisq` convention (`cl ≥ 1` →
  nσ, `0 < cl < 1` → probability; threshold `delta_chisq(cl, 1)`), an
  explicit `delta` override for joint statements (e.g.
  `delta = delta_chisq(0.68, 2)` for support-function tracing), `up`-scaled
  acceptance for −lnL fits (NLL/χ² parity is tested), and per-fit `maxfcn`
  budgets. New docs section in `docs/src/error_analysis.md`; API reference
  entries; `juminuit-usage` skill updated. Together with `mcmc_sample`/
  `quantile_band` above this completes the in-package error-analysis
  triangulation: profile extremization ↔ ensemble quantiles ↔ MINOS (a
  `LikelihoodEnsemble` is also a ready-made `seeds` pool for `extremize` —
  feed it the members extreme in `f`).

## [0.5.0] — 2026-06-10

Two independent overhauls. **`find_solution_modes`** gains cloud-scale whitening,
an `:auto` default, FCN-call policies, and a re-fit budget — driven by a field
stress test on a real 9-parameter two-solution coupled-channel fit (f1(1420),
2026-06) where the old fit-local metrics scored a genuine two-basin cloud as
"0 modes", used an untrustworthy covariance silently, and gave an expensive FCN
no way to bound the call count. The **2-D contour family** is reorganized for
iminuit fidelity and to resolve the `Plots.contour` name clash. The solver core
(MIGRAD/HESSE/MINOS) is unchanged. **Breaking**: the bare `contour` is no longer
exported (renamed `contour_ellipse`; iminuit's grid slice is the new
`contour_grid`), `mncontour` now traces joint confidence regions (iminuit `cl`
semantics), and `find_solution_modes`'s `whiten` default changes to `:auto`.

### Added

- **Cloud-scale whitening `whiten = :sample`** — per-coordinate robust scale
  `σ_k = 1.4826·MAD` of the sample column. Fit-independent: it measures the
  cloud with its own yardstick, which is what a multi-basin cloud (spread ≫
  local fit σ) needs — on such clouds the fit-local `:cov`/`:errors` metrics
  isolate every point and report 0 modes even with a perfect covariance.
  Degenerate coordinates (zero cloud spread) fall back to the fit σ, with a
  warning; never silently.
- **`whiten = :auto` — the new default.** Picks the metric from the cloud/fit
  width ratio: `:sample` when the cloud is wider than the fit's local scale in
  some coordinate (`max_k σ_cloud_k/σ_fit_k > 4`, the multi-basin regime),
  otherwise the previous `:cov` fallback chain — single-basin Δχ² clouds
  sampled at the fit scale keep the statistically tightest (Mahalanobis)
  metric and previous behavior exactly.
- **Zero-FCN-call / lazy χ² policies** for expensive cost functions:
  `fvals = :none` clusters with **zero** FCN evaluations (representatives =
  whitened-space medoids; `fval`/`delta_fval` = `NaN`; modes sorted by refined
  χ² or population), `fvals = :lazy` evaluates only the K cluster
  representatives. The previously implicit cost — N full FCN calls when
  `fvals` is omitted — is now documented prominently, and is skipped
  automatically when clustering finds no modes.
- **Re-fit budget + survivability** (`refine = true`): `refine_maxfcn` (FCN
  cap per MIGRAD attempt), `refine_strategy` and `refine_tol` (triage
  settings for slow FCNs), and `refine_callback` (fires once per finished
  mode with a self-contained NamedTuple — checkpointing for multi-hour runs;
  invocations serialized, exceptions caught). `SolutionMode` gains
  `refined_walltime` alongside `refined_nfcn`.
- **Actionable zero-modes diagnostics.** When clustering yields 0 modes (or
  bins > half the samples as noise), a warning reports the cloud's median /
  90%-quantile nearest-neighbour whitened distance against `threshold` and
  suggests the concrete fix (`whiten = :sample`, or the threshold that would
  reconnect the cloud).
- **Field-geometry regression tests**: the real two-basin 9-parameter
  geometry (multiplicative + error-scaled clouds, anchored-at-the-wrong-basin
  refine with `new_min` rescue), the `:auto` gate on fit-scale clouds,
  950/50 unbalanced clusters, degenerate coordinates, the untrusted-covariance
  fallback, FCN-call counting for `fvals = :none/:lazy`, and the refine
  budget/callback contract.
- **`contour_grid(m, par1, par2; size=50, bound=2, grid, subtract_min)`** —
  iminuit's `Minuit.contour` (the function IMinuit.jl exported as `contour`):
  the FCN evaluated on a 2-D grid with all **other parameters held fixed** at
  their current values — a slice, the 2-D analogue of `profile`. Returns a
  `ContourGrid` (new exported type) that destructures iminuit-style
  (`xs, ys, F = contour_grid(...)`, `F[i,j] = FCN(xs[i], ys[j])`) and has a
  filled-contour plot recipe with a marker at the minimum. The docstring spells
  out the statistics: a slice's Δχ² level curves are **conditional** regions —
  smaller than the true profile-likelihood region by ≈ `√(1−R²)` per axis when
  the pair correlates with the remaining free parameters — so it is a
  *landscape* tool; confidence regions come from `mncontour`. Numeric `bound`
  ranges are clipped to parameter limits; fixed parameters are rejected.

### Changed

- **`whiten` default: `:cov` → `:auto`.** Behavior is unchanged for clouds
  sampled at the fit's own scale (they resolve to `:cov` as before); clouds
  wider than the fit scale — where the old default returned 0 modes — now
  resolve to `:sample` and find the basins. Pass `whiten = :cov` to force the
  old metric unconditionally.
- **An untrustworthy covariance is no longer used silently** for
  `whiten = :cov`: if the fit is invalid or its Hessian was forced positive
  definite (`m.accurate == false`), the metric falls back to `:errors` with a
  warning (previously: silent use of the patched covariance → silent 0-mode
  results on the hard fits this tool targets).
- Degenerate fit-σ coordinates (zero/non-finite) are now **warned about**
  when they drop out of the `:errors` metric (previously a silent
  0-contribution that could collapse the metric without a trace).
- `find_deeper_minimum` (resampling strategy) now pins `whiten = :cov`
  explicitly for its internal clustering of converged candidates — the same
  metric it has always used, now independent of the `find_solution_modes`
  default (converged candidates are tight clumps at the fit scale, exactly
  the geometry the anchor metric is right for).
- With `fvals = :none`/`:lazy`, the cluster **representative is the
  whitened-space medoid** (most central member) instead of the min-χ² member
  (which would require evaluating every sample).
- **`mncontour` now traces joint confidence regions (iminuit ≥ 2.0 `cl`
  semantics).** Previously it always traced the C++ MnContours curve
  `FCN = fmin + up` (Δχ² = 1) and rejected `cl ≠ 1`. Now `cl` works like
  iminuit's: default → the joint 2-D 68 % region (`Δχ² = delta_chisq(0.68, 2)
  ≈ 2.28`); `0 < cl < 1` → that joint probability; `cl ≥ 1` → nσ (`cl = 2` →
  joint 95.45 %, Δχ² ≈ 6.18). The scaling factors match iminuit 2.31's
  `_cl_to_errordef` to 1e-10. Rationale (per F. James, *The Interpretation of
  Errors*, §1.3.3, and SMEP 2nd ed. Table 9.1/§9.3.3 — excerpted in the
  MINOS-contours tutorial): the Δχ²=1 curve's joint 2-D coverage is only
  39.3 %; for a *simultaneous* statement James prescribes the χ²(NPAR)
  quantile, which is exactly what `cl` now applies. **Migration:** the old
  Δχ²=1 curve is `mncontour(m, a, b; cl = chisq_cl(1, 2))` (≈ 0.3935) or the
  low-level `contour_exact` (`sigma = 1`, C++-identical); single-parameter
  errors remain `minos!` (Δχ² = up, unchanged). `contour_exact` gains a
  `sigma` kwarg (`fmin + up·sigma²`) so both conventions stay available.
- **`draw_mncontour` takes `cl`** (scalar or vector; overlays one contour per
  level, labelled by coverage) instead of the `nsigma=1`-only restriction;
  `draw_mnmatrix` likewise accepts `cl`.
- **`contour` → `contour_ellipse`** (both the `Minuit`-level and the low-level
  `(fmin, cf)` methods). The old name was doubly wrong: it was NOT iminuit's
  `contour` (that is a grid slice, now `contour_grid`), and exporting it made
  the bare name ambiguous against `Plots.contour` / `GR.contour`
  (`UndefVarError: contour not defined … ambiguity` under
  `using JuMinuit, Plots`). The bare `contour` is **no longer exported**;
  qualified `JuMinuit.contour(...)` keeps working as a deprecated alias of
  `contour_ellipse`.

### Fixed

- A `K = 0` clustering outcome no longer evaluates the FCN at all N samples
  for nothing (for a 0.5 s/call FCN and 800 samples that was ~7 minutes of
  wasted wall-time before an empty result).
- **`draw_mncontour` and `draw_mnmatrix` now draw the exact MINOS contour**
  (`mncontour` boundary search) as their names and docstrings always claimed.
  Previously both silently rendered the fast covariance-ellipse approximation.
  `draw_contour` now shows the iminuit-style `contour_grid` FCN landscape
  (filled contour, minimum subtracted) instead of the ellipse.

### Docs

- The MINOS-contours tutorial gains a "Δχ², coverage, and the two contour
  conventions" section with excerpts from F. James, *The Interpretation of
  Errors* (Minuit doc, 2004, §1.3.2–1.3.3) and *Statistical Methods in
  Experimental Physics* (2nd ed., Table 9.1, §9.3.3, p. 238): the Δχ²=1
  curve = single-parameter MINOS errors via its projections (joint coverage
  39.3 %); a joint region needs the χ²(NPAR) quantile (68 %: 1.00/2.30/3.53
  for 1/2/3 parameters), which is what `mncontour`'s `cl` applies.

## [0.4.1] — 2026-06-09

A display + diagnostics release. The solver core (MIGRAD/HESSE/MINOS) is unchanged
and the programmatic API is backward-compatible; this reworks how MINOS errors are
shown in the rich `Minuit` table so a non-converged cross-search is no longer
silently hidden behind the symmetric HESSE error.

### Added

- **MINOS non-convergence is surfaced, not silent.** When a MINOS cross-search
  fails to converge for a parameter, the rich table previously fell back to the
  symmetric HESSE error with no indication — indistinguishable from a genuine
  symmetric result or a never-run MINOS. A `⚠` line below the table now lists the
  affected parameters and which side(s) (`upper` / `lower` / `both`) did not
  converge.
- **Side-by-side Hesse vs MINOS columns after `minos!`.** Once MINOS has run, the
  rich `text/plain` and HTML tables widen from the compact merged `value ± error`
  column into separate `Value`, `Hesse` and `MINOS` columns (iminuit-style), so the
  asymmetric MINOS error sits next to its symmetric HESSE counterpart.
- **Per-side MINOS rendering.** Each MINOS side is shown independently: both
  converged → `+hi / −lo`; only one converged → that side's value with the other
  marked `invalid` (a one-sided result is no longer discarded); ran but both sides
  failed → `invalid`; MINOS not run for the parameter → `—`.

### Changed

- After `minos!`, the rich `Minuit` table uses the new multi-column comparison
  layout above (the pre-MINOS compact `value ± error` layout is unchanged).
  Programmatic access (`m.values`, `m.errors`, `m.merrors`, …) is unaffected.

### Docs

- BenchmarkExamples notebooks migrated to JuMinuit's native API (e.g.
  `contour_df_samples`); the IAM benchmark clarifies that its error band is the
  JOINT (simultaneous) 1σ region. Binder links updated (mybinder.org, after GESIS
  was decommissioned).

## [0.4.0] — 2026-06-05

`find_deeper_minimum` gains a data-resampling strategy and full parameter-constraint
support, and its API is unified onto `Minuit`. The solver core (MIGRAD/HESSE/MINOS)
is unchanged; the breaking change is contained to the `find_deeper_minimum` helper
that shipped in 0.3.1.

### Added

- **`find_deeper_minimum` data-resampling strategy** —
  `find_deeper_minimum(m::Minuit, refit, data; …)`: each round bootstrap-resamples
  the data and re-fits each resample (those drift toward whichever basin best
  explains that subset), clusters the candidates with
  `find_solution_modes(…; refine=true)` **re-evaluated on the original data**, and
  adopts the deepest valid new basin — far stronger than parameter perturbation on
  hard multi-basin data fits. Fresh-start `(cf/f, x0, errors, refit, data)`
  overloads and a `find_deeper_minimum(m::Minuit; …)` convenience overload included.
- **Parameter limits + fixed parameters are honoured.** Every fit in the search now
  routes through the high-level `Minuit` path, so a fit's `limits` and `fixed`
  flags survive the entire search — fixed parameters stay pinned, bounded ones stay
  in bounds (the perturbation jitters only free coordinates and clamps to bounds;
  the resampling refinement re-pins fixed parameters). The bare `(cf/f, x0, errors)`
  overloads accept `limits`/`fixed`/`names` keyword arguments.
- **`correlation(m::Minuit)`** — the HESSE parameter correlation matrix, for
  IMinuit.jl/iminuit parity (iminuit's `m.covariance.correlation()`). Equivalent
  to `matrix(m; correlation=true)`; the `correlation` function previously only
  accepted `BootstrapResult`/`JackknifeResult`.
- **`find_deeper_minimum` stops on CONVERGENCE, not a hard round count.** The
  search already broke when a round found no deeper basin; the default `max_rounds`
  is now a high safety backstop (`50`, was `6`) and a `@warn` fires if the cap is
  hit while the last round was *still improving* — so a low cap can no longer
  silently truncate a still-descending search.

### Changed (breaking)

- **`find_deeper_minimum` now always returns a `Minuit`** (MIGRAD + HESSE already
  run — check `.valid`). The 0.3.1 parameter-perturbation overloads returned a
  `FunctionMinimum`; both strategies now share one return type, ready for
  `minos!`/`hesse`. Migrate `is_valid(fm)` / `fm.state.parameters.x` / `values(fm)`
  to `m.valid` / `m.values`.
- **No longer "unbounded only".** The previous `find_deeper_minimum` ignored
  parameter limits and fixed flags (the docs told you to fold bounds into the FCN);
  it now honours them. Behaviour for unconstrained fits is unchanged.

### Fixed

- `find_solution_modes(…; refine=true)` now preserves the parent fit's
  `check_gradient` flag on each per-mode re-fit (it previously reverted to the
  constructor default `true`, emitting spurious `CheckGradient` warnings when the
  user had set it `false`).

## [0.3.1] — 2026-06-02

A tooling + documentation release. The solver core is unchanged (the v0.3.0
algorithms are byte-identical); this adds a global-search helper, regression
tests, real-physics error-analysis demonstrations, and a docs pass.

### Added

- **`find_deeper_minimum`** — a basin-hopping search for a *deeper* minimum on
  multi-basin objectives (a single MIGRAD only finds the basin its start point
  drains into). It perturbs the current best, re-fits, and adopts any deeper
  valid minimum, repeating until no round improves. A **heuristic** — it finds
  *a* deeper minimum, not a certified *global* one — complementing
  `find_solution_modes`; use it to escape a local basin before error analysis.
  Unbounded-only; returns a `FunctionMinimum` (check `is_valid`). (A deprecated
  `find_global_minimum` alias — the name in the first 0.3.1 build, which
  overclaimed globality — forwards here with a warning.)
- **JET optimisation-analysis regression guard** — `test/test_aqua_jet.jl`
  asserts (via `JET.@report_opt target_modules=(JuMinuit,)`) that the MIGRAD hot
  path stays free of runtime dispatch — locking in the `CostFunction{F}`
  devirtualisation the performance claim rests on. Verified on Julia 1.11 + 1.12.
- **Bootstrap-vs-MINOS cross-validation test** — the resampling suite now checks
  that a parametric bootstrap's ±1σ percentile interval reproduces each side of
  the analytic MINOS asymmetric error.
- **Error-analysis demonstrations** (`BenchmarkExamples/`, not unit tests):
  - `X3872_dip/error_crosscheck.jl` — HESSE / MINOS / parametric-bootstrap on
    the published single-basin dip fit (they agree).
  - `IAM_2Pformfactor/error_crosscheck.jl` — a **cautionary** multi-basin study:
    find the true minimum (multi-start + a `find_solution_modes(refine=true)`
    adopt-deeper-basin loop), then trust the **local** error methods
    (HESSE/MINOS/MC-Δχ²). Naive bootstrap/jackknife are shown to be unreliable
    there — their re-fits scatter across basins, inflating the spread by orders
    of magnitude.

### Changed

- Documentation aligned with the shipped code: the dev audit-trail docs
  (`ROADMAP`/`DESIGN`/`GAP_AUDIT`/`DEFERRED`/`CPP_FIDELITY_AUDIT`/
  `DAVIDON_CXX_AUDIT`) annotated as-built; the withdrawn `MnHesse` upstream-bug
  draft removed (its thesis was reverted in the code); the
  `_hesse_diagonal_failure` docstring/test corrected to the restored C++ clamp.
  The README clarifies that IMinuit.jl wraps the Python `iminuit`, adds a
  "multi-basin fits" workflow note, and acknowledges AI-agent assistance.

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
    - `ci.yml`: test matrix on Julia 1.11 + 1, Ubuntu + macOS +
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
