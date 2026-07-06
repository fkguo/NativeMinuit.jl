# Error analysis in NativeMinuit — which method, when

NativeMinuit offers eight ways to put an uncertainty on a fitted parameter — or on
a **derived quantity** (any scalar `f(θ)`, or a model curve's pointwise error
band). They are **not** interchangeable: they answer genuinely different
questions, and they diverge in exactly the situations where it matters. This
page is the map.

## The conceptual split

There are two families, and the difference is **what is held fixed**.

- **Likelihood-interrogating methods — HESSE, MINOS, extremize/profile_band,
  MC-Δχ², the MCMC ensemble** vary the **parameters** with the **data held
  fixed**. They ask:

  > *"Given **this** dataset, which parameter values are statistically
  > consistent with it?"*

  They read the shape of the cost surface `χ²(θ)` (or `−2 ln L(θ)`) around the
  minimum — the curvature (HESSE), the `ΔFCN = up` crossing along a profiled
  axis (MINOS — `up` is the *error definition*: `1` for a χ² (or `−2 ln L`) fit,
  `½` for a `−ln L` fit, defined just under the table below), a Monte-Carlo
  sample of the whole `ΔFCN ≤ up·delta_chisq(cl, ndof)` region
  (MC-Δχ²), or a likelihood-weighted Metropolis chain (`mcmc_sample`).
  The dataset never changes; only the trial parameters move.

- **Data-resampling methods — bootstrap, jackknife** resample the **data** and
  re-fit. They ask:

  > *"How much would the estimate θ̂ jump around if the experiment were
  > **repeated**?"*

  They perturb the dataset (draw points with replacement, or drop one) and let
  the fit respond, building up the empirical **sampling distribution of the
  estimator** directly.

For a correctly-specified, near-Gaussian fit the two families **agree** — that
agreement is itself a useful check. They **diverge** precisely when:

- **the error model is wrong** (the per-point `σ` are mis-scaled or
  mis-correlated). HESSE/MINOS/MC-Δχ² *trust* those `σ`; the bootstrap reads the
  spread that is actually in the data and ignores the stated `σ` (nonparametric)
  — so a bootstrap error much larger than the HESSE error is a red flag that the
  quoted `σ` are too optimistic; and
- **the estimator is biased** (nonlinear model, boundary, small sample). The
  jackknife *measures* that bias; the curvature-based methods cannot see it.

Use the likelihood methods as the primary error (they are cheap and standard);
reach for resampling when you doubt the error model or suspect estimator bias,
or when you want a model-light cross-check before quoting a result.

A **Bayesian** layer sits on top of the first family: `bayesian` /
`posterior_sample` multiply that same likelihood by an explicit **prior** and
sample the **posterior**, reporting **credible** intervals and one-sided
credible **limits**. Mechanically it is the likelihood-ensemble chain (data
fixed, parameters varied) — with `prior = :flat` the target is the pure
likelihood (a single chain reproduces the `mcmc_sample` path exactly); what
changes is the **interpretation** (a 68% credible interval is
a probability statement about θ *given the prior*, not a coverage statement) and
the **tooling** (priors, and a credible upper limit on a coupling pinned near a
boundary). It is the eighth method, detailed after the MCMC section below.

## The unified table

| Method | Varies | Needs | Best when | Caveats |
|---|---|---|---|---|
| **HESSE** | parameters (analytic 2nd-derivative covariance at the minimum) | a converged fit with a positive-definite covariance | a fast, symmetric error on a near-Gaussian, near-linear fit; the default | wrong when the cost surface is non-parabolic (nonlinear model); requires a valid, pos-def covariance — a `made_pos_def` status (the covariance had to be *forced* positive-definite) ⇒ treat with suspicion |
| **MINOS** | parameters (profiles each one, re-minimising the rest) | a converged, valid fit (`fmin` = the fit-minimum result; must be valid) | reporting **asymmetric** errors under mild–moderate nonlinearity | fails / misleads on an invalid `fmin`, strong nonlinearity, or a multimodal surface; one inner-minimisation per scan point (costlier than HESSE) |
| **extremize / profile_band** | parameters (constrained extremization of a derived `f(θ)` over the `Δχ²` region) | a converged, valid fit; seeds covering every low-χ² corridor | an interval on a **derived quantity** (a scalar `f(θ)` that is not a parameter), or the pointwise **error band** of a curve family — "MINOS for a function" | likelihood method (trusts the error model); `extremize` floors/ceils by the directional endpoints by default (fixes ill-conditioned single-seed under-coverage; `profile_band` does not — use its `mode=:directional` there), but a *disconnected multi-corridor* region still needs `seeds` — audit `.diagnostics` |
| **MC-Δχ² region** | parameters (fixed data), using the **true** `Δχ²` not a quadratic | the `χ²`/`−2lnL` plus a proposal (the fit covariance, or an explicit parameter range) | mapping a **non-Gaussian** confidence region, or a **joint** N-D region, where MINOS' 1-D profile is not enough | the proposal must **over-cover** when the covariance is unreliable, or the region is clipped; still *trusts the error model* (it is a likelihood method) |
| **Likelihood-ensemble MCMC** | parameters (fixed data): a Metropolis chain on the **true** likelihood `∝ exp(−fcn/(2·up))` | a fit to start from (HESSE helps the proposal); an FCN cheap enough for ~10⁴–10⁵ evaluations | **marginal quantiles & pointwise bands of derived quantities** (curves, ratios, …) under non-Gaussianity or active parameter limits; a reusable, likelihood-weighted error set | the quantile band is a *marginal* construction: at an active limit it can legitimately exclude the best fit (mode ≠ median — a property, not a failure); single chain — watch the acceptance and mixing; trusts the error model |
| **Bayesian posterior** | parameters (fixed data): the same Metropolis chain on `prior × exp(−fcn/(2·up))` | a fit to start from; an explicit **prior** (`:flat` ⇒ the likelihood path); FCN cheap enough for ~10⁴–10⁵ evals per chain | a **credible** interval or one-sided credible **limit** (e.g. an upper limit on a near-zero coupling), or any posterior summary under an informative prior | gives **credible** (prior-conditional), *not* confidence, statements; a flat prior is flat in **external** coords (parameterization-dependent, not "uninformative"); needs `nchains ≥ 2` for R̂; the posterior temperature follows `errordef` (keep `up` at 1 or ½) |
| **Bootstrap** | **data** (resample with replacement, then re-fit) | a resamplable dataset and many cheap re-fits | the error model is **uncertain or misspecified**; you want the estimator's empirical sampling distribution and robust, possibly asymmetric, CIs | expensive (`nresample` full re-fits); needs enough independent points; weak for binned / heavily-aggregated / strongly-correlated data |
| **Jackknife** | **data** (leave-one-out, then re-fit) | a dataset and `N` (delete-1) re-fits | a quick, almost assumption-free error **plus an explicit bias estimate** | coarser than the bootstrap; unreliable for highly nonlinear or non-smooth estimators; the delete-`d` block variant is coarser still |

`up` = `errordef` (1 for a `χ²` or `−2 ln L` fit, 0.5 for a `−ln L` fit — the
factor-of-2 between `−ln L` and `−2 ln L` is exactly why the two carry different
`up`). The `1σ` level is where the FCN rises by `up` above its minimum
(`ΔFCN = up`): for a `χ²` fit that is `Δχ² = 1`; for a bare `−ln L` fit it is
`ΔFCN = 0.5`. The χ²-equivalent displacement is always `ΔFCN / up`, so the `1σ`
χ²-equivalent level is `ΔFCN / up = 1` regardless of `up` — that normalization by
`up` is exactly what makes a `−ln L` fit and a `χ²` fit give the same errors.

## The methods in NativeMinuit

### HESSE — `hesse(m)`
Builds the covariance from the numerically-evaluated second-derivative (Hessian)
matrix at the minimum and inverts it: `cov = 2·up·H⁻¹`. Symmetric errors land in
`m.errors`, the full matrix in `m.covariance` / `matrix(m)`. This is the cheapest
error and the right default whenever the fit is near-Gaussian. Watch the
covariance status: a forced-positive-definite covariance (`m.accurate == false`)
means the quadratic approximation was poor and the error is unreliable.

### MINOS — `minos!(m[, par])`
Walks each parameter away from the minimum, re-minimising all the others, until
`χ²` rises by `up`. The result (`m.merrors`, `minos_lower`/`minos_upper`) is a
pair of generally **asymmetric** errors that follow real curvature of the cost
valley — the standard HEP choice when the parabolic HESSE error is too crude.
MINOS needs a genuinely valid minimum; on a broken or multimodal fit its
crossings are meaningless.

### Derived quantities — `extremize(m, f)` / `profile_band(m, f, xs)`

MINOS answers "which values of **parameter** `θᵢ` are consistent with the
data"; it has no notion of a **derived** scalar `f(θ)` — a peak position, a
ratio of amplitudes, the model curve at one energy, a Legendre moment.
[`extremize`](@ref NativeMinuit.extremize) is MINOS-for-a-function: the exact
profile interval

```
[min f(θ), max f(θ)]   over   { θ : FCN(θ) ≤ FCN_min + up·delta_chisq(cl, 1) }
```

with **all free parameters varied simultaneously** (limits and fixed
parameters honoured). The threshold uses `ndof = 1` no matter how many
parameters move: the quoted statement is ONE number — re-parametrize so `f`
is itself a coordinate and this is a single-parameter interval with the rest
profiled out (Wilks: one constraint ⇒ 1 dof; in the linear-Gaussian limit the
answer is exactly the projection theorem `f̂ ± √(Δχ²·cᵀCc)`, full parameter
correlations included). For `f(θ) = θ[i]` it reproduces the MINOS interval.

[`profile_band`](@ref NativeMinuit.profile_band) sweeps the same construction
along a grid for a curve family `f(x, θ)` (`x` first, `θ` the full parameter
vector — the same callback shape as `quantile_band`) — the standard **pointwise**
profile-likelihood error band for figures. Each `x` carries its own `cl`
statement, and the band **contains the best-fit curve by construction** —
the marginal `quantile_band` need not, when a parameter sits on a limit and
pushes the likelihood mass to one side (the two constructions are compared
side-by-side in the MCMC section below). Warm starts plus forward/reverse
passes keep the sweep cheap and the band edges smooth in `x`.

```julia
migrad!(m)

# 68.3 % profile interval of a derived scalar
r = extremize(m, θ -> θ[1] + θ[2] * 15.0)
r.lo, r.hi          # the interval (contains f at the best fit)
r.plo, r.phi        # the extremal parameter vectors realizing the endpoints
r.diagnostics       # per-seed audit: convergence, acceptance, who won

# pointwise 68.3 % band of a model curve on a grid, seeded from an
# mcmc_sample ensemble's f-extreme members (the ready-made seed bank)
S(θ) = model_moment(4420.0, θ) - model_moment(4465.0, θ)
ext = sort(collect(ens); by = S)
band = profile_band(m, (x, θ) -> model_moment(x, θ), 4360.0:2.0:4520.0;
                    seeds = [ext[1], ext[end]])
band.nfail == 0 || @warn "inspect band.diagnostics"
# plot: ribbon between band.lo and band.hi, central curve band.fbest
```

**Seed coverage is load-bearing, not a tunable parameter.** Each endpoint comes
from an exterior-penalty MIGRAD (a stiffening `λ`-continuation ladder with
warm restarts) run from every seed. A best-fit-only run can come out
**silently too narrow** — under-extremization — in two ways: (i) on a strongly
correlated / **ill-conditioned** region it stalls on the flat axis at a
feasible but non-extremal boundary point, and (ii) when the `Δχ²` region has
several disconnected low-χ² **corridors** (multi-basin fits; a parameter pinned
at a limit feeding a monotone map) the penalty cannot cross the barrier between
them. By default the result is **floored/ceiled by the directional
(HESSE-ellipse) endpoints** `θ̂ ± √δ·C∇f/σ_f` (feasible, exact in the
linear-Gaussian limit), so the interval is never narrower than the directional
one — removing case (i) automatically at the cost of one extra directional probe
(no extra penalty seeds; `directional_floor = false` opts out). Case (ii) still
needs you: pass everything that touches other corridors via `seeds`
(`mcmc_sample` ensemble members extreme in `f` — see the MCMC section below —
and `find_solution_modes` representatives). Audit `r.diagnostics`: per-seed
acceptance and `f` records, `directional_floor` (whether the floor supplied each
endpoint), plus
the winning seed per side (`winner_* == 0` with `naccepted_* > 0` means the
best fit is genuinely extremal; with `naccepted_* == 0` it means that side
FAILED).

Every reported endpoint is gate-certified: `FCN ≤ bound + accept_tol·up`
always, and `FCN ≤ bound` exactly whenever the local boundary pull-back
applies (the typical case) — the `fcn_*` diagnostics fields carry the
values, so feasibility can be checked rather than trusted.

For a genuinely **joint** statement (e.g. tracing a 2-D support function),
override the threshold explicitly: `extremize(m, f; delta = delta_chisq(cl, 2))`.

**Expensive FCN / `f` (seconds per evaluation).** The default `:full` algorithm
runs `2 sides × seeds × ladder-stages × rounds` MIGRADs and can be hours per
call — for the common **near-linear** case use `mode = :directional`, which
walks the single projection direction `C·∇f`, secant-roots the *true* FCN to
the `Δχ²` boundary on each side, and reports the *true* `f` there (≈ `n_free +
~15` paired evaluations, ~50× cheaper, exact in the linear-Gaussian limit). The
recommended procedure is **directional first, `:full` only if you suspect
non-linearity or the two disagree**. On the `:full` path, cut cost with
`rounds = 1`, `iterate = 1`, `strategy = 0`, and a modest `maxfcn`. For a long
run on a shared/kill-prone machine, attach `on_unit = …` (fired once per
penalty-MIGRAD unit) to checkpoint partial work externally. Finally, **`f` may
throw or return a non-finite value at infeasible θ — both are safe** (the probe
becomes a finite plateau the optimizer steers around, never a `NaN` into
MIGRAD); a genuinely non-finite-`f` region may legitimately *narrow* the
interval, which is safe — but do **not** return a sentinel like `0.0`, which
*centres* the endpoint (a silent bias).

When the band goes into a figure, write **pointwise** in the caption: each
`x` is its own 68 % statement, and the whole true curve lies inside the band
everywhere-at-once with lower probability.

### MC-Δχ² region — `get_contours_samples(m; ...)` / `contour_df_samples`
Samples trial parameter vectors (proposal = the fit covariance, or a user range),
keeps those inside the true `Δχ² ≤ up·delta_chisq(cl, ndof)` shell — **the exact
`χ²`/`−2lnL` re-evaluated at every sample** — and uses the surviving cloud to
describe the confidence region. Unlike MINOS it captures **non-Gaussian** and
**joint multi-parameter** regions directly. It is still a likelihood method (the
data are fixed and the per-point `σ` are trusted), so its honesty rests on the
error model just as HESSE's and MINOS' do. Returns a `NamedTuple` (kept
`samples`, per-parameter asymmetric `bounds`, `acceptance`, `under_coverage`, …);
`contour_df_samples` gives the same cloud as a `DataFrame` (it lives in a package
extension, so add `using DataFrames` to enable it). Validated against the
X(3872) published line-shape analysis (`BenchmarkExamples/X3872_dip`).

**The proposal is not the cut.** The Gaussian (or box) is only how trial points
are *proposed*; acceptance is the **true `Δχ²`**, never the Mahalanobis distance
`(x−μ)ᵀΣ⁻¹(x−μ)` — cutting on Mahalanobis would just reproduce the HESSE ellipse,
defeating the purpose. The optional Mahalanobis output is a diagnostic only.

**Proposal under-coverage (the pitfall).** A `MvNormal(best, Σ)` proposal (a
multivariate normal centred on the best fit `best` with covariance `Σ`)
**under-estimates** the region when `Σ` is unreliable (`made_pos_def` / invalid
`fmin`) or the posterior is strongly nonlinear (the true shell extends beyond the
local Gaussian). Mitigations, all built in: an `inflate` factor; **adaptive
widening** (auto-grow `inflate` until the region stops being clipped, with an
`under_coverage` flag if it can't); a covariance-free **`proposal = :uniform`**
box over explicit `ranges`; and a **warning** when the covariance looks unreliable
— it never silently under-estimates. When in doubt, use the range proposal: it
does not depend on `Σ` at all.

**Joint vs single-parameter level — `delta_chisq(cl, ndof)`.** The acceptance
threshold is `delta_chisq(cl, ndof)`, with two arguments:

- **`cl` — the confidence level**, read two ways depending on its magnitude (the
  same convention as iminuit, so a ported fit behaves identically):
  - `cl ≥ 1` → a number of **σ**: `1`→68.27 %, `2`→95.45 %, `3`→99.73 % (the
    Gaussian probability mass within ±`cl` σ);
  - `0 < cl < 1` → a **probability** directly: e.g. `0.95`→95 %.

  So `cl = 1` and `cl = 0.6827` request the *same* region — but mind the magnitude:
  **`cl = 2` (2σ ⇒ 95.45 %) is *not* `cl = 0.95` (95 %).**
- **`ndof` — the number of parameters defining the region** (how many you vary
  *jointly*), **not** the fit's total parameter count. A single-parameter 1σ
  interval is `Δχ² = 1`, but a **2-D joint** 68 % region is `Δχ² = 2.30`, and 3-D is
  `3.53` — *not* 1. The sampler defaults `ndof = n_free` (`n_free` = the number of
  free/floating parameters — the joint region over all sampled parameters), which is
  usually what you want; override it deliberately if not.

(`chisq_cl(Δχ², ndof)` is the inverse: given a `Δχ²` it returns the probability.)

The table below **is** `delta_chisq(cl, ndof)` on a grid — each row a `cl` (given as
nσ, with its probability), each column an `ndof`. Pick the column that matches the
**dimension of the region you report**, not the fit's parameter count:

| Confidence (`cl`) | Probability | `ndof` = 1 | `ndof` = 2 | `ndof` = 3 | `ndof` = 4 |
|:------------------|:-----------:|:----------:|:----------:|:----------:|:----------:|
| 1σ (`cl = 1`) | 68.27 % | **1.00** | **2.30** | **3.53** | **4.72** |
| 2σ (`cl = 2`) | 95.45 % | 4.00 | 6.18 | 8.02 | 9.72 |
| 3σ (`cl = 3`) | 99.73 % | 9.00 | 11.83 | 14.16 | 16.25 |

For example the **2σ / 3-param** cell is `delta_chisq(2, 3) ≈ 8.02` (`cl = 2` ⇒ 2σ ⇒
95.45 %, three jointly-varied parameters). To request the same region with a
probability instead, pass `delta_chisq(0.9545, 3)` — **not** `delta_chisq(0.95, 3)`,
which is the different (95 %) region. (`delta_chisq` is valid for any `ndof`; iminuit's
own helper covers only 1–2 parameters.)

```julia
# 1σ JOINT region for all 3 free parameters ⇒ Δχ² = 3.53 (NOT 1):
r = get_contours_samples(m; nsamples = 30_000, cl = 1)   # ndof defaults to n_free = 3
r.bounds          # asymmetric (min,max) per parameter
r.under_coverage  # false ⇒ the proposal covered the region
# Untrustworthy Σ → covariance-free box proposal:
r = get_contours_samples(m; proposal = :uniform, ranges = [(0,0.8),(-0.05,0.05),(-15,0)])
```

Related: `contour_parameter_sets(ce)` returns the full parameter vector at every
`contour_exact` / `mncontour` boundary point (the 2 contour coordinates + the
profiled rest) at no extra cost — the native analogue of IMinuit.jl's
`get_contours`.

### Likelihood-ensemble MCMC — `mcmc_sample` / `quantiles` / `quantile_band`

`mcmc_sample(m)` runs a random-walk Metropolis chain on the **exact FCN**
(acceptance `exp(−Δfcn/(2·up))` — `exp(−Δχ²/2)` for a χ² fit, `exp(−Δ(−log L))`
for `up = 0.5`) and returns a `LikelihoodEnsemble`: ~2000 parameter sets drawn
from the likelihood `L(θ) ∝ exp(−fcn(θ)/(2·up))`, each with its FCN value. Any
derived quantity then gets a **marginal quantile interval**
(`quantiles(ens, f)`) or a **pointwise quantile band** over a grid
(`quantile_band(ens, f, xs)`) by plain evaluation over the ensemble — no
propagation formula, no linearization, no re-fitting. iminuit has no native
analogue (Python users bolt on `emcee` for this).

```julia
m = Minuit(chi2, x0; names = names, limits = limits)
migrad!(m); hesse!(m)                       # HESSE shapes the proposal

ens = mcmc_sample(m; seed = 11)             # 52k steps, burn 2k, thin 25 → 2000 sets
ens.acceptance                              # healthy: ≈ 0.2–0.4

q16, q50, q84 = quantiles(ens, θ -> θ[2] - θ[1])          # scalar derived quantity
band = quantile_band(ens, (x, θ) -> model(x, θ), xgrid)    # nx × 2: 16% and 84% edges

save_ensemble("ensemble_B.dat", ens; comment = "error set B")   # reusable error set
ens = load_ensemble("ensemble_B.dat")       # …in a later session: no re-sampling

# A foreign / hand-rolled file (only `# …` comments + `fval p₁ p₂ …` rows, e.g.
# a `# cols: chi2 a b` header) loads too, but a non-NativeMinuit header is NOT
# parsed as metadata: names default to p1,p2,… and `up` to NaN unless supplied.
ens = load_ensemble("legacy.dat"; names = ["a", "b"], up = 1.0)
```

**Not the same animal as `get_contours_samples`.** The region sampler draws
proposals and *keeps only* those inside `Δχ² ≤ delta_chisq(cl, ndof)` — a hard
cut whose product is region **extents**. The MCMC ensemble has **no Δχ² cut at
all**: samples are kept in proportion to their likelihood, so they concentrate
at the *typical set* `Δχ² ≈ n_free`, not inside `Δχ² ≤ 1`. This is the
high-dimensional **volume effect**: for 9 free parameters
`chisq_cl(1, 9) = P(Δχ² ≤ 1) ≈ 5.6e-4`, so a 9-D likelihood chain essentially
never visits the `Δχ² ≤ 1` shell — and a region sampler asked for
`ndof = 1`-style thresholds in 9-D accepts almost nothing. Use
`get_contours_samples` when the deliverable is a **joint confidence region**;
use `mcmc_sample` when the deliverable is **likelihood-weighted quantiles of
derived quantities**. (The ensemble is also a ready-made *seed bank* for
[`extremize`](@ref NativeMinuit.extremize) / [`profile_band`](@ref NativeMinuit.profile_band)
— pass the members extreme in `f` via `seeds`; they tell the profile
optimizer where the low-χ² corridors are.)

**Marginal quantile band vs profile envelope band — quote which one you used.**
For a curve `f(x; θ)` there are two honest "1σ band" constructions, and at a
parameter limit they legitimately differ:

| | profile envelope band | likelihood-ensemble quantile band |
|---|---|---|
| construction | pointwise `[min, max]` of `f` over `{Δχ² ≤ delta_chisq(cl, 1)}` (constrained extremization — [`profile_band`](@ref NativeMinuit.profile_band); explicit `delta = delta_chisq(cl, k)` only for joint statements) | pointwise 16–84% quantiles of `f` over the likelihood ensemble (`quantile_band`) |
| nature | frequentist confidence band | likelihood/posterior-mass band |
| best fit | **contained by construction** | **need not be contained** (mode ≠ median) |
| needs | an optimizer reaching the region edge (`extremize`/`profile_band`; multi-start, seeds!) | likelihood-weighted samples (this section) |
| agree when | near-Gaussian interior — the two coincide | same |
| separate when | at an active parameter limit the envelope truncates at the boundary side | the truncation piles the mass on one side ⇒ the whole band shifts |

The boundary case is worth spelling out (it alarms people the first time).
Take a parameter with `limits = (0, ∞)` whose best fit lands **on** the
boundary (a coupling `g ≥ 0` fitted to data that prefer `g < 0`, say). Every
ensemble member has `g > 0`, so if `f` responds monotonically to `g`, the
ensemble's `f` values sit systematically on one side of the best-fit curve —
the 16–84% band can then **exclude the best fit entirely**. That is the
correct marginal statement about where the likelihood mass is, not a sampler
bug, and **more samples will not make it go away**; the profile envelope, by
construction, still contains the best fit. Report the band you used; when both
are quoted, their interior agreement (and explicable boundary separation) is a
strong cross-check — the triangulation *profile extremization ↔ ensemble
quantiles ↔ MINOS* in practice.

**Why not "uniform points in `Δχ² ≤ 1` + envelope"?** Three reasons: (i) the
volume effect above makes hitting the region hopeless in high dimension;
(ii) an envelope over finitely many points systematically **under-estimates**
the band (it misses the region's corners); (iii) quantiles of *uniform-in-region*
samples answer no calibrated statistical question. Sampling is the right tool
for **likelihood-weighted quantiles** (this section); region *edges* are an
**optimization** problem ([`extremize`](@ref NativeMinuit.extremize) / MINOS /
`contour_exact`).

**Tuning (field-tested recipe).** Proposal step ≈ `0.25–0.35 ×` the HESSE σ
gives acceptance ≈ 0.2–0.3 for ~10 free parameters; the defaults
(`nsteps = 52_000, burn = 2_000, thin = 25`, `proposal = :hesse`,
`scale = 0.3`) yield 2000 well-decorrelated sets. Set `target_accept = 0.25`
to have the scale tuned automatically during burn-in (then frozen, so the kept
chain has a fixed kernel). The proposal shape affects **mixing efficiency
only** — any symmetric proposal converges to the same distribution (contrast
the region sampler, where proposal under-coverage biases the result): when the
covariance is unreliable the sampler falls back to per-coordinate `m.errors`
steps with a warning, and `proposal = [σ₁, σ₂, …]` overrides everything (the
escape hatch when a parameter sits at a limit and both Σ and the parabolic
errors are meaningless there). Parameter `limits` are enforced by rejection —
the chain samples the likelihood truncated to the allowed box, which is
exactly what produces the one-sided boundary pile-up above. Convergence
sanity: `minimum(ens.fvals)` should come within `Δχ² ≈ O(1)` of `ens.fbest`
(in ~9-D it typically bottoms out around `Δχ² ~ 0.8` — the volume effect
again), quantiles should be stable against halving the ensemble, and
`minimum(ens.fvals) < ens.fbest` means the chain found a **deeper minimum** —
re-minimize (`find_deeper_minimum`) before quoting any errors.

### Bayesian posterior bridge — `bayesian` / `posterior_sample` / credible intervals

`posterior_sample(m)` and the one-step `bayesian(m)` reuse the **same** Metropolis
kernel as the likelihood ensemble, but multiply the likelihood by an explicit
**prior** and interpret the result as a **posterior** in full external
coordinates:

```math
\log p(θ \mid \text{data}) = -\,\frac{\text{fcn}(θ)}{2\,\text{up}} + \log \text{prior}(θ).
```

The result is a [`PosteriorSample`](@ref NativeMinuit.PosteriorSample) — a
`LikelihoodEnsemble` plus Bayesian provenance (prior, kept log-likelihood /
log-posterior, per-chain IDs, R̂ / ESS, boundary flags). Everything is
**non-mutating**: the fit, `m.values`, `m.errors`, and `m.nfcn` are untouched.

```julia
m = Minuit(chi2, x0; names = names, limits = limits)
migrad!(m); hesse!(m)                         # HESSE shapes the proposal

# One-step report: 68.3% equal-tailed credible intervals under a flat prior.
report = bayesian(m; level = 0.6827)          # m is left completely untouched

# Explicit prior + a reusable posterior sample (4 chains ⇒ split-R̂ / ESS).
pr   = normal_prior(m, :mass, 3.8717, 0.0002) # Gaussian prior on one parameter
post = posterior_sample(m; prior = pr, nchains = 4, seed = 11)
maximum(post.rhat) < 1.01 && minimum(post.ess) > 400   # converged & well-mixed?

ci   = credible_interval(post, :mass; level = 0.6827)            # (lo, hi)
gup  = upper_limit(post, :g; level = 0.90)                       # 90% credible upper limit
db   = derived_interval(post, θ -> θ[2] - θ[1]; level = 0.6827)  # any scalar f(θ)
posterior_summary(post; level = 0.6827)                         # per-parameter table
```

**Priors are small and explicit.** Each is a log-density over the *full*
external vector; Minuit limits are intersected in as physical support:

```julia
flat_prior(m)                    # default; flat in EXTERNAL coords (see caveat)
normal_prior(m, :x, μ, σ)        # Gaussian on one parameter, flat on the rest
uniform_prior(m, :g, 0.0, 0.8)   # proper box on one parameter
half_normal_prior(m, :g, σ)      # half-normal above the lower limit (or above 0)
combine_priors(p1, p2)           # add disjoint informative components (MVP: no overlap)
```

`prior = :flat` makes `posterior_sample` reproduce the single-chain likelihood
path **byte-for-byte** (`post.ensemble.samples == mcmc_sample(m; …).samples` at
the same seed) — the Bayesian layer adds new tools, never silently changes the
chain. Construction **fails loudly** if the best-fit point lies outside the
prior × limits support, rather than starting a dead chain.

!!! warning "Credible ≠ confidence, and three things to keep honest"
    - A credible interval/limit is a probability statement about θ **given the
      prior** — not a frequentist confidence interval, CLs, Feldman–Cousins, or
      MINOS interval. `upper_limit`/`lower_limit` return a
      [`CredibleLimit`](@ref NativeMinuit.CredibleLimit), not a `merror`.
    - `flat_prior` is flat in **external** coordinates — a parameterization
      choice, **not** an "uninformative"/Jeffreys prior. Re-parameterize and the
      flat prior changes.
    - The posterior **temperature follows `errordef`**: the likelihood enters as
      `exp(-fcn/(2·up))`, so keep `up = 1` (χ² / `-2 log L`) or `up = 0.5`
      (`-log L`). Inflating `up` to widen a MINOS interval tempers the posterior
      by the same `√up` — put extra information in the **prior**, not in
      `errordef`.

**Diagnostics & mixing.** `nchains` defaults to 4, each started **over-dispersed**
at `overdisperse` × the proposal/HESSE scale from the best fit (default `2`, i.e.
≈2σ wider than the posterior — the spread that makes split-R̂ a real convergence
test; a start landing on a non-finite posterior is retried, then warned).
`rhat(post, par)` is the **basic split-R̂** (needs `nchains ≥ 2`, want `< 1.01`;
not rank-normalized, so for a skewed or boundary-truncated marginal also check
ESS and the trace) and `effective_sample_size(post, par)` the autocorrelation-
adjusted ESS. A `boundary_active` warning means posterior mass piles against a
parameter limit —
the same one-sided boundary effect as the marginal quantile band above (mode ≠
median), here a genuine feature for upper-limit reporting. Proposal tuning
(`proposal`, `scale`, `target_accept`) is identical to `mcmc_sample`.

**When to use it.** Use the Bayesian posterior when the result you want is
explicitly Bayesian — an upper limit on a coupling consistent with zero, a
result under a physically-motivated prior, or a posterior probability — and the
likelihood ensemble / MINOS / profile band when the result you want is a frequentist
confidence statement. With a flat prior in a near-Gaussian interior the credible
and confidence intervals coincide; quote which one you computed.

### Bootstrap — `bootstrap(model, data, start; ...)`
Resamples the dataset and re-fits `nresample` times, returning a
[`BootstrapResult`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/resampling.jl):

- **Nonparametric** (`kind = :nonparametric`, default): draws `N` data points
  **with replacement** and re-fits. The spread of θ̂ over the resamples is the
  bootstrap error; its percentiles give **asymmetric** CIs. Crucially this reads
  the scatter that is *actually in the data*, so it does **not** depend on the
  quoted `σ` being right — the key reason to use it when the error model is in
  doubt.
- **Parametric** (`kind = :parametric`): regenerates `yᵢ* = model(xᵢ, θ̂) + σᵢ·zᵢ`
  from the best fit and the assumed Gaussian error model, then re-fits. This
  *does* trust `σ`, so it tracks the HESSE error closely — useful as a check that
  the resampling estimate and the curvature error agree.

Re-fits are warm-started from the full-data optimum by default, run serially
unless you pass `threaded = true` (the default is `threaded = false`), and are
**deterministic** given an explicit `seed` (the per-resample RNG seeds are drawn
serially, so a threaded run is bit-identical to a serial one). Percentile CIs default to a `0.68` (±1σ-equiv.)
coverage; pass `covariance = true` for the bootstrap covariance matrix.

`bootstrap` / `jackknife` accept three input shapes: **(i)** a cost object —
`bootstrap(cost, start)` for `LeastSquares` / `UnbinnedNLL` /
`ExtendedUnbinnedNLL` (the cost carries its own data + model, so nothing else is
passed; binned costs and `CostSum` are not point-resamplable and raise a clear
error); **(ii)** `model` + `Data` + start (shown above); and **(iii)** a generic
`refit(subdata) -> θ̂` callback over any indexable `data`. The cost-object form is
bit-identical to the `model` + `Data` form for the equivalent `LeastSquares` fit.

Note that for an `ExtendedUnbinnedNLL`, the nonparametric cost-object bootstrap
conditions on the sample size — every resample draws exactly `N` points — so the
fitted total-count / normalization parameter is pinned and its bootstrap `σ`
collapses to `≈ 0` rather than the Poisson `√N`; take that count error from
HESSE/MINOS, or from a parametric / Poisson-count bootstrap (see the
`bootstrap(cost, start)` docstring).

### Jackknife — `jackknife(model, data, start; ...)`
Deletes one point (delete-1, the default) — or one consecutive block
(`d > 1`) — re-fits, and aggregates the leave-one-out estimates θ̂₍ⱼ₎ into a
[`JackknifeResult`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/resampling.jl):

- **variance** `((g−1)/g)·Σⱼ(θ̂₍ⱼ₎ − θ̄)²` (with `g = N` groups for delete-1) —
  comparable to the HESSE error²;
- **bias** `(g−1)·(θ̄ − θ̂_full)` and the **bias-corrected** estimate
  `θ̂_full − bias`. For an unbiased (e.g. linear) estimator the bias is ≈ 0; a
  large value flags a nonlinear or small-sample bias the curvature methods miss.

The jackknife is cheaper and steadier than the bootstrap but coarser, and it is
unreliable for highly nonlinear or non-smooth estimators (where the
local-deletion linearisation breaks down). The delete-`d` block variant targets
**serially-correlated** data; for IID data it is a coarse, higher-variance
estimator (shuffle before blocking to restore exchangeability).

## Parameter correlations & nonlinear joint structure

Both resampling methods re-fit **all** parameters jointly on each resampled
dataset, so the joint distribution of θ̂ carries the parameter *correlations* —
but the per-parameter `std` / percentile CIs are **marginal** and drop that
joint information. To recover it:

- **Bootstrap.** `bootstrap(...; covariance=true)` stores the `npar × npar`
  covariance; `correlation(r)` returns the standardised correlation matrix from
  `r.samples` *regardless* of that flag. Because the bootstrap re-fits the full
  nonlinear model, the **raw `r.samples` cloud retains non-Gaussian joint
  structure** — a curved degeneracy / "banana" common in amplitude and
  phase-shift fits — that a covariance (a second moment) and the HESSE ellipse
  both flatten. Inspect it directly: scatter `r.samples[:,i]` vs `r.samples[:,j]`,
  or build a 2-D density / contour.
- **Jackknife.** `JackknifeResult` carries the full `covariance` matrix
  `((g−1)/g)·Σ(θ̂₍ⱼ₎−θ̄)(θ̂₍ⱼ₎−θ̄)ᵀ` (diagonal = `variance`), and `correlation(r)`
  standardises it. This captures correlation only to **first order** — the
  jackknife is a linearisation, so it cannot represent the nonlinear / asymmetric
  joint structure the bootstrap does. Use it for a quick correlation read, not as
  the primary tool when the model is strongly nonlinear.

So: for the *linear* correlation summary use `correlation(r)` (or `r.covariance`);
for the *full nonlinear* joint structure use the bootstrap's `r.samples`. The
likelihood-geometry counterparts are HESSE's covariance (Gaussian/elliptical) and
MC-Δχ² (the true, possibly non-elliptical, joint region with the data held fixed).

## Multi-modal solution detection

### The phenomenon: an acceptable χ² is not a unique solution

When you sample the parameter space and keep every point whose χ² is within Δχ²
of the best fit (the "statistically acceptable" set — see the **MC-Δχ² region**
method above),
you get a cloud of accepted parameter vectors. The implicit assumption when you
summarise that cloud with one mean ± one covariance is that the cloud is **one
connected region**.

**It often is not.** Widen the sampling range and the accepted set can break
into a *main* cluster plus one or more *separated* regions — sometimes only a
handful of points. Each separated region is a **distinct solution**: its χ² is
within Δχ² of the global best (so it is statistically acceptable), but its
parameters — and therefore its **physics** — are different.

```
  χ² surface (1-D cartoon)                       accepted set (Δχ² cut)
                                                 ┌────────┐      ┌──┐
   \                              /              │ main   │      │B │
    \         ___                /               │ cluster│      │  │   ← separated
     \   A   /   \   B   ___    /                └────────┘      └──┘     region =
      \_____/     \_____/   \__/                  ←—— same χ² band ——→     distinct
        ▲            ▲                                                     solution
     global       another
      min       acceptable min
```

Reporting a single error bar that spans both regions is **wrong**: it implies a
continuum of solutions between them that the χ² actually rejects. The two
regions must be reported and treated **independently**.

iminuit and C++ Minuit2 have **no auto-detection** of this. `find_solution_modes`
adds it.

### Physics example: distinct solutions in the X(3872) coupled-channel fit

The motivating case is the X(3872) line shape in the `J/ψρ + DD̄*` coupled-channel
fit (Baru, Guo, Hanhart, Nefediev, *Phys. Rev. D* **109**, L111501,
[arXiv:2404.12003](https://arxiv.org/abs/2404.12003)). The published data admit
**several physically distinct local minima** — different scattering-length
combinations that **all reproduce the characteristic near-threshold dip** yet
differ in the broader line shape, each fitting the data with comparable χ². A HESSE/MINOS
error bar around whichever minimum MIGRAD happened to land in hides the others
entirely. The distinct minima are different **physics conclusions**, not
different points of one error ellipse, and must be enumerated and fit separately.

### Step 1 — cluster the accepted samples into modes

```julia
using NativeMinuit
m = Minuit(chi2, x0; names = pnames);  migrad!(m)

r       = get_contours_samples(m; ...)        # NamedTuple; r.samples has rows = vectors
modes   = find_solution_modes(r.samples, m)   # cluster into distinct solutions

if length(modes) > 1
    @warn "multi-modal: $(length(modes)) statistically distinct solutions"
end
```

`modes` pretty-prints a report:

```
SolutionModes: 2 distinct solution(s) from 500 accepted sample(s)
  metric: whiten=:cov (Mahalanobis)  method=:components  threshold=1 σ  errordef(up)=1
  [1] main  :   312 pts ( 62.4%)  χ²=18.42                  rep=[0.997, 2.01, …]
  [2] mode 2:   188 pts ( 37.6%)  χ²=19.05      Δχ²=0.63     rep=[-1.4, 2.0, …]
  ⚠ separated modes have comparable χ² but DIFFERENT physics — treat
    them independently; do NOT merge into a single error bar.
```

Each [`SolutionMode`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/solution_modes.jl) carries: the minimum-χ²
**representative** sample of its cluster, that χ² and its **Δχ²** versus the
global best, the per-parameter **(min, max)** range over the cluster, the point
count and **fraction**, and the member row indices. Modes are sorted by χ²
(main = lowest first).

### The whitened (scale-normalized) distance is mandatory

> **This is the single most important correctness point.** Clustering in raw
> parameter coordinates is **wrong** and will silently merge distinct modes.

Fit parameters span wildly different scales — a low-energy constant at `~1e-3`
sits next to a coupling at `~1`. A naive Euclidean distance is **dominated by
the largest-scale parameter**, so two modes that differ only in a *small-scale*
parameter look identical: their separation in that parameter is swamped by the
ordinary within-mode spread of the large-scale parameter. The clusterer then
**merges them**.

The fix is to cluster in **whitened** coordinates, so that distance is measured
in units of σ and is dimensionless and scale-invariant. *Which* σ matters just
as much — the fit's local scale, or the sample cloud's own scale:

- **`whiten = :sample`** — per-parameter **robust cloud scale**,
  `z_k = x_k / (1.4826·MAD_k)` with `MAD_k` the median absolute deviation of
  the sample column (1.4826 makes it a consistent σ for normal data). This is
  the metric for **multi-basin clouds**: it measures the cloud with its own
  yardstick, independent of any fit. The fit-local metrics below fail exactly
  there — on a cloud spread over many fit-σ (a cross-basin scan, or scatter
  not generated from this fit's errors), *every* point is isolated in fit-σ
  units and the clustering reports **0 modes**. MAD (rather than the sample
  covariance or standard deviation) is essential: between-basin separation
  inflates a variance along the separation axis, compressing exactly the
  direction that must stay resolved, while the MAD of a two-cluster cloud
  stays at the half-separation scale.

- **`whiten = :cov`** — full **Mahalanobis** distance using the fit's
  free-parameter covariance Σ. With the Cholesky factor Σ = L·Lᵀ, the whitened
  coordinate is `z = L⁻¹·x`, and the pairwise distance becomes

  ```
  d(xᵢ, xⱼ)² = (xᵢ − xⱼ)ᵀ Σ⁻¹ (xᵢ − xⱼ)
  ```

  This both **rescales** each parameter to σ units **and decorrelates** them —
  the statistically tightest metric **when the cloud is a single-basin Δχ²
  region sampled at the fit's own scale**. It falls back to `:errors` (with a
  warning) when the covariance is unavailable, not positive-definite, or **not
  trustworthy** — an invalid fit or a forced-positive-definite Hessian
  (`m.accurate == false`); a patched-up covariance is not a metric, and using
  it silently is how a real two-solution cloud quietly becomes "0 modes".

- **`whiten = :errors`** — per-parameter fit scaling only, `z_k = x_k / σ_k`.
  Ignores correlations; cheaper and needs no matrix inversion. The robust
  fallback. Same local-metric caveat as `:cov` on wide clouds.

- **`whiten = :auto`** (default) — picks the metric from the cloud/fit width
  ratio: `:sample` when the cloud is wider than the fit's local scale in some
  coordinate (`max_k σ_cloud_k / σ_fit_k > 4` — the multi-basin regime),
  otherwise the `:cov` chain above. A Δχ² cloud sampled from the fit itself
  keeps the Mahalanobis metric; a cross-basin scan automatically gets the
  cloud metric instead of a useless all-noise result.

A naive unwhitened Euclidean metric is **deliberately not offered** as an option.
The contrast is real and reproducible — two modes separated by 5σ in a `1e-3`-scale
parameter but overlapping in a `~1`-scale parameter:

| metric | result |
|--------|--------|
| raw Euclidean (threshold 1) | **1 cluster** — modes merged (wrong) |
| `:errors` / `:cov` (threshold 1) | **2 clusters** — modes resolved (correct) |

And on a **multi-basin** cloud whose spread is many fit-σ wide (a real
9-parameter two-solution geometry from a coupled-channel fit):

| metric | result |
|--------|--------|
| `:cov` / `:errors` (fit-local) | **0 modes** — every point isolated (wrong) |
| `:sample` / `:auto` (cloud scale) | **2 clusters** — both solutions found (correct) |

(See `test/test_solution_modes.jl`, testsets *"WHITENED metric resolves
tiny-scale separation"* and *"two-bowl 9-par geometry"*.)

When clustering finds **no modes** (or bins most samples as noise),
`find_solution_modes` emits a diagnostic with the cloud's median
nearest-neighbour whitened distance versus `threshold`, and suggests the
concrete fix (`whiten = :sample`, or the threshold that would reconnect the
cloud) — a 0-mode result on a sane cloud almost always means the metric, not
the cloud.

### Step 2 (optional) — re-fit each mode: `refine = true`

Clustering finds *where* the modes are; it does not move to their exact minima.
`refine = true` runs a fresh MIGRAD from each cluster's representative — preserving
the parent fit's cost function, gradient, parameter limits, fixed-parameter
structure, `errordef`, strategy and tolerance — to recover that mode's **true
local minimum and its own errors**:

```julia
modes = find_solution_modes(samples, m; refine = true)
for s in modes
    println("mode $(s.index): χ²=$(s.refined_fval)  x=$(s.refined_values)")
end
```

The re-fit also flags a subtle, important case. If a separated cluster re-fits to
a minimum **deeper than the global best**, the main fit **missed the better basin**
— the cluster is the solution MIGRAD should have found. This is flagged
prominently:

```
  [1] main  :    40 pts ( 40.0%)  χ²=-0.628                 rep=[2.93, 2.91]
        ↳ re-fit: χ²=-0.629  (valid, 24 fcn)  ⚠ DEEPER than global best
  ...
  ⚠⚠ a refined mode reached a DEEPER minimum than the global best —
     the main fit likely missed the better basin (see `new_min`).
```

The flag is exposed as `mode.new_min`. This connects directly to the IAM
cold-start convergence gap (see [`IAM_CONVERGENCE_GAP.md`](https://github.com/fkguo/NativeMinuit.jl/blob/main/docs/dev/IAM_CONVERGENCE_GAP.md)):
a separated cluster can be exactly the basin a stiff cold-start fit failed to
reach. Per-mode re-fits are parallelized across threads when the fit opts into
threading (`m.threaded_gradient`, honoring the same FCN thread-safety requirement as
NativeMinuit's threaded gradient).

### Expensive cost functions: control every FCN call

For an FCN costing seconds per call, the defaults matter: `find_solution_modes`
evaluates the FCN at **every sample** (to pick min-χ² representatives and report
Δχ²), and an unbudgeted re-fit can spend thousands of calls per mode. Every one
of those calls is controllable:

```julia
# 0 FCN calls: cluster only; representatives = whitened-space medoids,
# modes sorted by population (fval/delta_fval are NaN).
modes = find_solution_modes(samples, m; fvals = :none)

# K FCN calls (one per cluster): full χ²/Δχ² report at medoid representatives.
modes = find_solution_modes(samples, m; fvals = :lazy)

# Or pass the per-sample χ² you already have (get_contours_samples keeps them).
modes = find_solution_modes(samples, m; fvals = my_chi2s)

# Budgeted, checkpointed refine: ≤ refine_maxfcn calls per MIGRAD attempt,
# triage strategy/tolerance, and a per-mode callback so a killed multi-hour
# run loses at most the mode in flight (invocations are serialized — writing
# to a file from the callback is safe).
modes = find_solution_modes(samples, m; fvals = :lazy, refine = true,
                            refine_maxfcn   = 500,
                            refine_strategy = 0,
                            refine_tol      = 1.0,
                            refine_callback = r -> serialize("mode_$(r.k).jls", r))
```

Each refined mode reports `refined_nfcn` and `refined_walltime`, so the cost of
a follow-up full-precision re-fit can be extrapolated before launching it.

### Escaping a local basin — `find_deeper_minimum`

`find_solution_modes` only *clusters what you already sampled*; its `new_min` flag
tells you a better basin exists but not how to reach it from scratch.
[`find_deeper_minimum`](@ref) is the **search** counterpart — it actively climbs
out of the basin a single MIGRAD lands in and returns the deeper minimum, so you
can do error analysis *there*. Every overload returns a [`Minuit`](@ref) (MIGRAD +
HESSE already run — check `.valid`), routes every fit through the high-level Minuit
path, and **honours your fit's parameter limits and fixed parameters** — the search
stays inside the same constrained parameter space as your fit, never mutating the
input. It ships in two flavours:

**Parameter-perturbation** (works for any objective — no data needed). Jitter the
current best by `perturb · scaleᵢ · randn` each restart (FREE coordinates only;
fixed parameters stay pinned, bounded ones are clamped to their bounds), MIGRAD,
adopt any deeper valid minimum, repeat:

```julia
m_deep = find_deeper_minimum(m)                   # from a converged Minuit (limits/fixed honoured)
m_deep = find_deeper_minimum(chi2, x0, errs;      # or from a cost function / callable…
                             limits = [(0,2), nothing], fixed = [false, true],
                             n_restarts = 40, perturb = 1.5, seed = 1)
m_deep.valid || error("search failed")
```

**Data-resampling** (for data fits — much stronger on hard multi-basin surfaces).
Each round bootstrap-resamples the data and re-fits each resample; those drift
toward whichever basin best explains that subset. The candidates are clustered
with `find_solution_modes(...; refine = true)`, **re-evaluated on the ORIGINAL
data** (so the χ² comparison is honest), and the deepest valid new basin is
adopted; repeat. Fixed parameters are re-pinned during refinement and bounds are
respected throughout:

```julia
# refit(subdata, start) -> parameter vector (NaNs ⇒ invalid, dropped)
refit = (sub, start) -> (fm = migrad(CostFunction(p -> chi2_on(sub, p)), start, errs);
                         is_valid(fm) ? collect(values(fm)) : fill(NaN, length(start)))
m_deep = find_deeper_minimum(m, refit, data)      # m may carry fixed/limits — honoured
m_deep.valid || error("search failed")
minos!(m_deep)                                    # now do error analysis HERE
```

If the first round finds no deeper basin, it warns and returns a fitted clone of
`m` (same minimum) — the surface may be single-basin, or bootstrap coverage was
insufficient (raise `n_discovery`); for parameter-space search try the
perturbation form instead.

!!! note "Worked example — IAM ππ"
    `BenchmarkExamples/IAM_2Pformfactor/find_deeper_minimum_demo.jl`: a cold
    Strategy-1 MIGRAD from the published LECs lands at **χ² ≈ 379** (a shallow
    basin, χ²/dof ≈ 4.9); `find_deeper_minimum`'s resampling dispatch drops it to
    **χ² ≈ 255** (χ²/dof ≈ 3.3) over four adopt-rounds — a **Δχ² ≈ 124** descent in
    one call, by the same mechanism `error_crosscheck.jl`'s hand-rolled PHASE 1
    loop uses (it reaches χ² ≈ 235 from a multi-start seed). Error analysis at 379
    would be meaningless; do it at the deep minimum.

!!! warning "A heuristic, not a global-optimum proof"
    Basin-hopping finds *a* deeper minimum when its restarts/resamples land in one;
    it cannot certify the result is global (hence the name — not
    `find_global_minimum`). Raise `n_restarts`/`perturb`/`n_discovery`/`max_rounds`
    and cross-check from independent seeds.

### Clustering backends

- **`method = :components`** (default, **zero dependencies**) — single-linkage
  connected components in whitened space: link any two samples whose whitened
  distance is ≤ `threshold`, then take connected components. `min_size` separates
  genuine sparse modes from stray noise points. Cost is O(N²·d) in the number of
  samples — fine for the hundreds-to-few-thousand a Δχ² scan produces.

- **`method = :dbscan`** (optional) — density-based clustering for arbitrary
  cluster shapes and explicit outlier handling, via the `Clustering.jl` package
  extension (`ext/NativeMinuitClusteringExt.jl`). Activates on `using Clustering`;
  without it, requesting `:dbscan` raises an actionable error pointing at
  `:components`. Uses a spatial tree, so ~O(N·log N) — prefer it for very large N.

### Tuning and caveats

- **`threshold`** (default `1.0`, in whitened σ units) sets the connection radius.
  *Smaller* is stricter — it will not merge distinct modes, but may split a
  sparsely-sampled one. *Larger* risks single-linkage **chaining** across a gap.
  Distinct physical modes are typically separated by many σ, so the default is a
  safe starting point; sanity-check by reading the per-mode Δχ² and representative
  separations in the report.
- **`min_size`** (default `1`) keeps every separated region — the input is already
  χ²-filtered, so even a few-point region is a candidate solution. Raise it to
  suppress scatter; the report states how many points were dropped as noise.
- **Not a global-optimization guarantee.** This detects multi-modality *in the
  sampled set*; it cannot prove that every basin was sampled. It is a diagnostic
  layer on top of MC-Δχ² sampling, not a global optimizer.

## Visualizing the results

Every error-analysis output above ships a [RecipesBase](https://github.com/JuliaPlots/RecipesBase.jl)
recipe, so `plot(...)` works from Plots.jl with no extra glue (NativeMinuit depends
only on `RecipesBase`, not on Plots). A parameter pair is chosen with `vars`
(indices or names; default the first two free parameters) and a single parameter
with `par`.

```julia
using NativeMinuit, Plots

# MC-Δχ² sample cloud — a 2D scatter of the accepted set, coloured by Δχ².
r = get_contours_samples(m; nsamples = 20_000, cl = 1, seed = 1)
plot(r)                          # first two free parameters
plot(r; vars = ("mass", "g"))    # pick the pair by name

# Bootstrap / jackknife — histogram of a parameter's resampled distribution
# (estimate and percentile-CI / mean drawn as reference lines). Its asymmetry
# about the estimate is exactly what a symmetric error bar cannot show.
plot(bootstrap(model, data, m; nresample = 2000, seed = 1))   # first free parameter
plot(jackknife(model, data, m); par = "k")

# Multi-modal solutions — colour one series per mode, mark each representative.
# Pass the same sample matrix used for clustering to scatter the point cloud:
S = r.samples
modes = find_solution_modes(S, m)
plot(modes, S)                   # cluster cloud, one colour per mode
plot(modes)                      # no samples → per-mode bounding boxes + reps
```

The recipes are backend-agnostic; pick a backend (`gr()`, `plotlyjs()`, …) as
usual. See [`src/plot_recipes.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/plot_recipes.jl).

## A short decision guide

1. **Default:** quote the **HESSE** error. If the fit is non-parabolic but valid,
   add **MINOS** for asymmetric errors.
2. **Non-Gaussian or joint region** where MINOS' 1-D profile is insufficient:
   map it with **MC-Δχ²**.
3. **Error on a derived quantity or a curve** (a ratio, a lineshape, a moment —
   not a single fit parameter): build a **likelihood ensemble** with
   `mcmc_sample` and read `quantiles` / `quantile_band` (marginal construction);
   for a band that must contain the best fit, use [`extremize`](@ref NativeMinuit.extremize) /
   [`profile_band`](@ref NativeMinuit.profile_band) over `Δχ² ≤ delta_chisq(cl, 1)`
   instead (profile construction; an explicit `delta = delta_chisq(cl, k)` is
   for genuinely joint statements only). At parameter
   limits the two differ legitimately — see the comparison table above.
4. **Want a Bayesian statement, an explicit prior, or an upper limit on a
   near-zero quantity** (a coupling consistent with 0, a rate, a positive
   parameter at its boundary): sample the posterior with `bayesian` /
   `posterior_sample` and read `credible_interval` / `upper_limit` — these are
   **credible** (prior-conditional) statements, not a confidence/CLs/Feldman–
   Cousins limit. A flat prior recovers the likelihood-ensemble quantiles. See
   the [Bayesian analysis guide](bayesian.md) for worked examples (upper limits,
   nuisance marginalization, derived quantities, EFT naturalness).
5. **Doubt the error model** (don't trust the quoted `σ`, suspect correlations or
   mis-scaling): cross-check with the **nonparametric bootstrap** — a bootstrap
   error far from the HESSE error tells you the `σ` are wrong.
6. **Suspect estimator bias** (nonlinear, boundary, small `N`): run the
   **jackknife** for an explicit bias estimate and a bias-corrected value.
7. **Check** that the resampling estimate agrees with the curvature error:
   the **parametric bootstrap** should reproduce the HESSE error.

## See also

- Resampling implementation: [`src/resampling.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/resampling.jl);
  tests [`test/test_resampling_errors.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/test/test_resampling_errors.jl)
- MC-Δχ² / `delta_chisq` implementation:
  [`src/error_sampling.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/error_sampling.jl); tests
  [`test/test_error_sampling.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/test/test_error_sampling.jl)
- Likelihood-ensemble MCMC / quantile bands:
  [`src/mcmc.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/mcmc.jl); tests
  [`test/test_mcmc.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/test/test_mcmc.jl)
- Bayesian posterior analysis (priors, posterior, credible intervals):
  [`src/posterior.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/posterior.jl),
  [`src/priors.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/priors.jl); tests
  [`test/test_bayesian_bridge.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/test/test_bayesian_bridge.jl)
- HESSE / MINOS / contours: [`src/hesse.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/hesse.jl),
  [`src/minos.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/minos.jl), [`src/contours.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/contours.jl)
