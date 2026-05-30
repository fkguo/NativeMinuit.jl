# Error analysis in JuMinuit — which method, when

JuMinuit offers five ways to put an uncertainty on a fitted parameter. They are
**not** interchangeable: they answer two genuinely different questions, and they
diverge in exactly the situations where it matters. This page is the map.

## The conceptual split

There are two families, and the difference is **what is held fixed**.

- **Likelihood-interrogating methods — HESSE, MINOS, MC-Δχ²** vary the
  **parameters** with the **data held fixed**. They ask:

  > *"Given **this** dataset, which parameter values are statistically
  > consistent with it?"*

  They read the shape of the cost surface `χ²(θ)` (or `−2 ln L(θ)`) around the
  minimum — the curvature (HESSE), the `Δχ² = up` crossing along a profiled
  axis (MINOS), or a Monte-Carlo sample of the whole `Δχ² ≤ threshold` region
  (MC-Δχ²). The dataset never changes; only the trial parameters move.

- **Data-resampling methods — bootstrap, jackknife** resample the **data** and
  re-fit. They ask:

  > *"How much would the estimate θ̂ jump around if the experiment were
  > **repeated**?"*

  They perturb the dataset (draw points with replacement, or drop one) and let
  the fit respond, building up the empirical **sampling distribution of the
  estimator** directly.

For a correctly-specified, near-Gaussian fit the two families **agree** — that
agreement is itself a useful sanity check. They **diverge** precisely when:

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

## The unified table

| Method | Varies | Needs | Best when | Caveats |
|---|---|---|---|---|
| **HESSE** | parameters (analytic 2nd-derivative covariance at the minimum) | a converged fit with a positive-definite covariance | a fast, symmetric error on a near-Gaussian, near-linear fit; the default | wrong when the cost surface is non-parabolic (nonlinear model); requires a valid, pos-def covariance — `made_pos_def` ⇒ treat with suspicion |
| **MINOS** | parameters (profiles each one, re-minimising the rest) | a converged, valid fit (`fmin` valid) | reporting **asymmetric** errors under mild–moderate nonlinearity | fails / misleads on an invalid `fmin`, strong nonlinearity, or a multimodal surface; one inner-minimisation per scan point (costlier than HESSE) |
| **MC-Δχ² region** | parameters (fixed data), using the **true** `Δχ²` not a quadratic | the `χ²`/`−2lnL` plus a proposal (the fit covariance, or an explicit parameter range) | mapping a **non-Gaussian** confidence region, or a **joint** N-D region, where MINOS' 1-D profile is not enough | the proposal must **over-cover** when the covariance is unreliable, or the region is clipped; still *trusts the error model* (it is a likelihood method) |
| **Bootstrap** | **data** (resample with replacement, then re-fit) | a resamplable dataset and many cheap re-fits | the error model is **uncertain or misspecified**; you want the estimator's empirical sampling distribution and robust, possibly asymmetric, CIs | expensive (`nresample` full re-fits); needs enough independent points; weak for binned / heavily-aggregated / strongly-correlated data |
| **Jackknife** | **data** (leave-one-out, then re-fit) | a dataset and `N` (delete-1) re-fits | a quick, almost assumption-free error **plus an explicit bias estimate** | coarser than the bootstrap; unreliable for highly nonlinear or non-smooth estimators; the delete-`d` block variant is coarser still |

`up` = `errordef` (1 for a `χ²` fit, 0.5 for a `−2 ln L` fit); the `1σ` level is
`Δχ² = up`.

## The methods in JuMinuit

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

### MC-Δχ² region — `get_contours_samples(m; ...)` / `contour_df_samples`
Samples trial parameter vectors (proposal = the fit covariance, or a user range),
keeps those inside the true `Δχ² ≤ up·delta_chisq(cl, ndof)` shell — **the exact
`χ²`/`−2lnL` re-evaluated at every sample** — and uses the surviving cloud to
describe the confidence region. Unlike MINOS it captures **non-Gaussian** and
**joint multi-parameter** regions directly. It is still a likelihood method (the
data are fixed and the per-point `σ` are trusted), so its honesty rests on the
error model just as HESSE's and MINOS' do. Returns a `NamedTuple` (kept
`samples`, per-parameter asymmetric `bounds`, `acceptance`, `under_coverage`, …);
`contour_df_samples` gives the same cloud as a `DataFrame`. Validated against the
X(3872) published line-shape analysis (`BenchmarkExamples/X3872_dip`).

**The proposal is not the cut.** The Gaussian (or box) is only how trial points
are *proposed*; acceptance is the **true `Δχ²`**, never the Mahalanobis distance
`(x−μ)ᵀΣ⁻¹(x−μ)` — cutting on Mahalanobis would just reproduce the HESSE ellipse,
defeating the purpose. The optional Mahalanobis output is a diagnostic only.

**Proposal under-coverage (the pitfall).** A `MvNormal(best, Σ)` proposal
**under-estimates** the region when `Σ` is unreliable (`made_pos_def` / invalid
`fmin`) or the posterior is strongly nonlinear (the true shell extends beyond the
local Gaussian). Mitigations, all built in: an `inflate` factor; **adaptive
widening** (auto-grow `inflate` until the region stops being clipped, with an
`under_coverage` flag if it can't); a covariance-free **`proposal = :uniform`**
box over explicit `ranges`; and a **warning** when the covariance looks unreliable
— it never silently under-estimates. When in doubt, use the range proposal: it
does not depend on `Σ` at all.

**Joint vs single-parameter level — `delta_chisq(cl, ndof)`.** The threshold is
`delta_chisq(cl, ndof)` (`chisq_cl` inverts it). `cl` follows iminuit (probability
if `< 1`, nσ if `≥ 1`). **`ndof` is the dimension of the region, not the parameter
count of the fit:** a single-parameter 1σ interval is `Δχ² = 1`, but a **2-D joint**
68 % region is `Δχ² = 2.30`, and 3-D is `3.53` — *not* 1. The sampler defaults
`ndof = n_free` (the joint region over all sampled parameters), which is usually
what you want; override it deliberately if not.

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

### Bootstrap — `bootstrap(model, data, start; ...)`
Resamples the dataset and re-fits `nresample` times, returning a
[`BootstrapResult`](../src/resampling.jl):

- **Nonparametric** (`kind = :nonparametric`, default): draws `N` data points
  **with replacement** and re-fits. The spread of θ̂ over the resamples is the
  bootstrap error; its percentiles give **asymmetric** CIs. Crucially this reads
  the scatter that is *actually in the data*, so it does **not** depend on the
  quoted `σ` being right — the key reason to use it when the error model is in
  doubt.
- **Parametric** (`kind = :parametric`): regenerates `yᵢ* = model(xᵢ, θ̂) + σᵢ·zᵢ`
  from the best fit and the assumed Gaussian error model, then re-fits. This
  *does* trust `σ`, so it tracks the HESSE error closely — useful as a check that
  the resampling machinery and the curvature error agree.

Re-fits are warm-started from the full-data optimum by default, run across
threads with `threaded = true`, and are **deterministic** given an explicit
`seed` (the per-resample RNG seeds are drawn serially, so a threaded run is
bit-identical to a serial one). Percentile CIs default to a `0.68` (±1σ-equiv.)
coverage; pass `covariance = true` for the bootstrap covariance matrix.

### Jackknife — `jackknife(model, data, start; ...)`
Deletes one point (delete-1, the default) — or one consecutive block
(`d > 1`) — re-fits, and aggregates the leave-one-out estimates θ̂₍ⱼ₎ into a
[`JackknifeResult`](../src/resampling.jl):

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

## A short decision guide

1. **Default:** quote the **HESSE** error. If the fit is non-parabolic but valid,
   add **MINOS** for asymmetric errors.
2. **Non-Gaussian or joint region** where MINOS' 1-D profile is insufficient:
   map it with **MC-Δχ²**.
3. **Doubt the error model** (don't trust the quoted `σ`, suspect correlations or
   mis-scaling): cross-check with the **nonparametric bootstrap** — a bootstrap
   error far from the HESSE error tells you the `σ` are wrong.
4. **Suspect estimator bias** (nonlinear, boundary, small `N`): run the
   **jackknife** for an explicit bias estimate and a bias-corrected value.
5. **Sanity check** that the resampling plumbing agrees with the curvature error:
   the **parametric bootstrap** should reproduce the HESSE error.

## See also

- Resampling implementation: [`src/resampling.jl`](../src/resampling.jl);
  tests [`test/test_resampling_errors.jl`](../test/test_resampling_errors.jl)
- MC-Δχ² / `delta_chisq` implementation:
  [`src/error_sampling.jl`](../src/error_sampling.jl); tests
  [`test/test_error_sampling.jl`](../test/test_error_sampling.jl)
- HESSE / MINOS / contours: [`src/hesse.jl`](../src/hesse.jl),
  [`src/minos.jl`](../src/minos.jl), [`src/contours.jl`](../src/contours.jl)
