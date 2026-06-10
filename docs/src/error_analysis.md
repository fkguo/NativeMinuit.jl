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
  axis (MINOS — `up` is the *error definition*: `1` for a χ² fit, `½` for `−2 ln L`,
  defined just under the table below), or a Monte-Carlo sample of the whole `Δχ² ≤ threshold` region
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
| **HESSE** | parameters (analytic 2nd-derivative covariance at the minimum) | a converged fit with a positive-definite covariance | a fast, symmetric error on a near-Gaussian, near-linear fit; the default | wrong when the cost surface is non-parabolic (nonlinear model); requires a valid, pos-def covariance — a `made_pos_def` status (the covariance had to be *forced* positive-definite) ⇒ treat with suspicion |
| **MINOS** | parameters (profiles each one, re-minimising the rest) | a converged, valid fit (`fmin` = the fit-minimum result; must be valid) | reporting **asymmetric** errors under mild–moderate nonlinearity | fails / misleads on an invalid `fmin`, strong nonlinearity, or a multimodal surface; one inner-minimisation per scan point (costlier than HESSE) |
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

### Bootstrap — `bootstrap(model, data, start; ...)`
Resamples the dataset and re-fits `nresample` times, returning a
[`BootstrapResult`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/resampling.jl):

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
[`JackknifeResult`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/resampling.jl):

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
using JuMinuit
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

Each [`SolutionMode`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/solution_modes.jl) carries: the minimum-χ²
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
cold-start convergence gap (see [`IAM_CONVERGENCE_GAP.md`](https://github.com/fkguo/JuMinuit.jl/blob/main/docs/dev/IAM_CONVERGENCE_GAP.md)):
a separated cluster can be exactly the basin a stiff cold-start fit failed to
reach. Per-mode re-fits are parallelized across threads when the fit opts into
threading (`m.threaded_gradient`, honoring the same FCN thread-safety contract as
JuMinuit's threaded gradient).

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
  extension (`ext/JuMinuitClusteringExt.jl`). Activates on `using Clustering`;
  without it, requesting `:dbscan` raises an actionable error pointing at
  `:components`. Uses a spatial tree, so ~O(N·log N) — prefer it for very large N.

### Tuning and caveats

- **`threshold`** (default `1.0`, in whitened σ units) sets the connection radius.
  *Smaller* is stricter — it will not bridge distinct modes, but may split a
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
recipe, so `plot(...)` works from Plots.jl with no extra glue (JuMinuit depends
only on `RecipesBase`, not on Plots). A parameter pair is chosen with `vars`
(indices or names; default the first two free parameters) and a single parameter
with `par`.

```julia
using JuMinuit, Plots

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
usual. See [`src/plot_recipes.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/plot_recipes.jl).

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

- Resampling implementation: [`src/resampling.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/resampling.jl);
  tests [`test/test_resampling_errors.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/test/test_resampling_errors.jl)
- MC-Δχ² / `delta_chisq` implementation:
  [`src/error_sampling.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/error_sampling.jl); tests
  [`test/test_error_sampling.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/test/test_error_sampling.jl)
- HESSE / MINOS / contours: [`src/hesse.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/hesse.jl),
  [`src/minos.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/minos.jl), [`src/contours.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/contours.jl)
