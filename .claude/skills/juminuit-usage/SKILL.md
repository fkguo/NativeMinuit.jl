---
name: juminuit-usage
description: >-
  Quick-reference for USING the JuMinuit.jl Julia package (a native-Julia
  Minuit2 port with an iminuit / IMinuit.jl-style API) to minimize a
  χ²/likelihood and analyze fit errors. Use when writing or editing Julia code
  that fits data with `Minuit`, `migrad!`, `minos!`, `hesse!`, the cost-function
  objects (`LeastSquares`/`UnbinnedNLL`/`BinnedNLL`/…), `mncontour`/`profile`,
  `extremize`/`profile_band` (derived-quantity intervals & error bands),
  `bayesian`/`posterior_sample` (Bayesian posterior, priors & credible
  intervals/limits), `bootstrap`/`jackknife`, `get_contours_samples`,
  `find_solution_modes`, or
  `find_deeper_minimum`; or when porting Python-iminuit or IMinuit.jl fitting
  code to JuMinuit. Covers the bang-method idiom (`migrad!(m)` not `m.migrad()`),
  the FCN/result conventions, AD & threaded gradients, bounds/fixed parameters,
  and the error-analysis decision guide — so the API is recalled, not guessed.
---

# JuMinuit.jl — usage quick-reference

JuMinuit.jl is a **pure-Julia port of C++ Minuit2** with an API that mirrors
Python **iminuit** / **IMinuit.jl**. This skill is for *using* JuMinuit as a
dependency in a fitting project. (If you are editing the JuMinuit repo itself,
its own `src/` docstrings and `docs/src/` are the source of truth.)

> **Golden rule — you are writing Julia, not Python.** The minimizers are
> **bang-mutating functions**, not object methods: write `migrad!(m)`, never
> `m.migrad()`. Result *accessors* stay as properties (`m.values`, `m.errors`).
> This single difference is the most common porting bug.

## iminuit / IMinuit.jl → JuMinuit map

| iminuit (Python) / IMinuit.jl | JuMinuit |
|---|---|
| `m.migrad()` | `migrad!(m)`  (or `migrad(m)`) |
| `m.hesse()` | `hesse!(m)` |
| `m.minos()` | `minos!(m)` |
| `m.values`, `m.errors`, `m.covariance` | **same** (property access) |
| `m.merrors` | **same** — `Dict{String,MinosError}` keyed by name; `m.minos_errors` keyed by Int |
| `m.mncontour("a","b")` | `mncontour(m, "a", "b")` |
| `m.profile("a")` / `m.mnprofile("a")` | `profile(m, "a")` / `mnprofile(m, "a")` |
| `m.scipy(method=...)` | `optim(m; method=...)`  (needs `using Optim`) |
| `Fit`, `ArrayFit` (IMinuit.jl) | exported **aliases of `Minuit`** |
| `chisq`, `Data`, `model_fit`, `args`, `matrix` (IMinuit.jl) | exported, **same signatures** |

`migrad`, `minos`, `hesse` (no `!`) also exist and forward to the bang forms, so
old IMinuit.jl functional code keeps working. The bang form is idiomatic and
chains: `m |> migrad! |> hesse! |> minos!`.

## Core workflow (copy-paste skeleton)

```julia
using JuMinuit

# FCN: any callable f(x::AbstractVector) -> Real. With bounds it ALWAYS sees
# external (physical) coordinates — you never touch the internal transform.
fcn(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2

m = Minuit(fcn, [0.0, 0.0];
           names  = ["a", "b"],        # default "x0","x1",…
           errors = [0.1, 0.1],        # initial step sizes
           limits = [(-5.0, 5.0), nothing])   # per-param: nothing | (lo,hi) | (lo,nothing) | (nothing,hi)

migrad!(m)                             # run MIGRAD, mutates m in place, returns m
minos!(m)                              # asymmetric errors (optional)

m.values        # ≈ [1.0, 2.0]   — indexable by Int OR name: m.values["b"], m.values[1]
m.errors        # symmetric 1σ HESSE errors
m.fval          # FCN value at the minimum
m.valid         # Bool: converged within tolerances?  (alias: m.is_valid)
m.accurate      # Bool: covariance trustworthy? false ⇒ no/invalid fit OR Hessian forced pos-def
m.covariance    # covariance matrix (nothing before migrad!)
m.merrors["a"]  # MinosError; .upper (≥0), .lower (≤0), is_valid(me), upper_valid/lower_valid
m               # rich table: HTML in Jupyter/Pluto, text in REPL
args(m)         # plain Vector{Float64} of values (≡ collect(m.values))
```

Gate on validity before trusting results:
```julia
m.valid || @warn "MIGRAD did not converge" m.fval
```
If invalid: loosen `tol`, raise `maxfcn`, bump `strategy`, or re-seed.

## The constructor

`Minuit(fcn, x0; kwargs...)` — keywords (singular iminuit names and plural
JuMinuit names both accepted):

| kwarg | default | meaning |
|---|---|---|
| `names` / `name` | `"x0","x1",…` | param names — `names=["a","b"]` (Strings only); `name=[:a,:b]` also takes Symbols |
| `errors` / `error` | `0.1` each | initial step sizes |
| `limits` | `nothing` | `Vector` of `nothing` / `(lo,hi)` / `(lo,nothing)` / `(nothing,hi)` |
| `fixed` | all free | `Vector{Bool}`, pin parameters |
| `up` / `errordef` | `1.0` | **1.0 for χ², 0.5 for −lnL** (cost objects set this for you) |
| `grad` | `nothing` | `x -> ∇f(x)::Vector{Float64}`, e.g. AD (see Gradients) |
| `check_gradient` | `true` | warn (not crash) if `grad` disagrees w/ numerical at seed |
| `strategy` | `Strategy(1)` | 0=fast, 1=default (iminuit parity), 2=thorough |
| `tol` | `0.1` | EDM convergence target |
| `threaded_gradient` | `false` | `true` / `:auto` to parallelize numerical gradient (needs `julia -t N`) |
| `print_level` | `0` | verbosity |

> **Likelihood fits need `up = 0.5`.** A bare `−lnL` FCN must be built with
> `Minuit(negloglike, x0; up = 0.5)` — the default `up = 1.0` is for χ², and a wrong
> `up` makes **every HESSE/MINOS error come out √2 too small, silently**. The cost
> objects (`UnbinnedNLL`, …) set the right `up` for you.

Per-parameter keywords also work: `error_a = 0.2`, `fix_b = true`,
`limit_mass = (0, nothing)`, etc. Override budgets per run on `migrad!`:
```julia
migrad!(m; strategy = 2, tol = 1e-3, maxfcn = 10_000)
m.strategy = 2   # or store on m so later fits reuse it
```

## Cost functions — don't hand-roll the χ²

For standard fits, use a cost object: it carries data + model + the right
`errordef`, so `Minuit(cost, x0)` reads `up` and the data count automatically
(enables `χ²/ndf` and p-value in the table). Each is callable `cost(p)` → a scalar
objective (`Float64` on ordinary numeric calls; a ForwardDiff `Dual` under AD).

| Cost | For | `errordef` | AD-generic? |
|---|---|---|---|
| `LeastSquares(x, y, σy, model; name=[…])` | χ² of `y±σ` vs `model(x,p)` | 1.0 | yes |
| `UnbinnedNLL(samples, pdf; name=[…])` | unbinned ML from a normalized pdf | 0.5 | yes |
| `ExtendedUnbinnedNLL(samples, density, integral)` | unbinned + yield param | 0.5 | yes |
| `BinnedNLL(counts, edges, cdf)` | histogram fit (cumulative model) | 0.5 | yes |
| `ExtendedBinnedNLL(counts, edges, scaled_cdf)` | histogram + yield | 0.5 | yes |
| `cA + cB` → `CostSum` | joint fit; params **unified by name** | rescaled | — |

```julia
model(x, p) = p[1]*x + p[2]
cost = LeastSquares(x, y, σy, model; name = [:a, :b])
m = Minuit(cost, [1.0, 0.0]); migrad!(m)       # up=1 & ndata read off the cost

# joint fit sharing slope `a`; `+` unifies params by name → (a, b, c)
joint = LeastSquares(xA,yA,σA,model; name=[:a,:b]) + LeastSquares(xB,yB,σB,model; name=[:a,:c])
m = Minuit(joint, [1.0,0.0,0.0]); migrad!(m)
```
`UnbinnedNLL(samples, f; log=true)` if `f` already returns `log(pdf)`. Pass
`mask=<BitVector>` to fit a subset without copying. IMinuit.jl `chisq`/`Data`/
`model_fit` remain available and give bit-identical results to `LeastSquares`.

## Bounds & fixed parameters

```julia
m = Minuit(fcn, x0; limits = [(0.0,1.0), (0.0,nothing)], fixed = [false, true])
# FCN sees PHYSICAL coords; bound respected at EVERY probe, not just the optimum.

fix!(m, "mass"); migrad!(m)             # fix–fit–release–fit scan
release!(m, "mass"); migrad!(m)
set_value!(m, "mass", 3.0); set_error!(m, "mass", 0.2)
set_limits!(m, "frac", 0.0, 1.0)        # ← two-sided, keeps BOTH sides
remove_limits!(m, "mass")
```
**Gotcha:** `set_lower_limit!` / `set_upper_limit!` set one side **and CLEAR the
other** (C++ Minuit2 semantics). For a two-sided bound use `set_limits!`.
Mutators take an index or a name, drop the cached fit, return `m`. A parameter
ending **at a bound** has unreliable HESSE/MINOS error (flagged in the table);
MINOS reports the *distance to the bound* and sets `upper_par_limit`/`lower_par_limit`.

## Errors: HESSE / MINOS / contours

```julia
hesse!(m)                 # symmetric covariance (migrad! already leaves one)
minos!(m)                 # asymmetric ±σ on every free param; minos!(m, "a") for one
minos!(m; sigma = 2)      # widen to the 2σ crossing
me = m.merrors["a"]; me.upper; me.lower; is_valid(me)   # gate on upper_valid/lower_valid

pts = mncontour(m, "a", "b"; numpoints = 40)   # EXACT joint 68% CL boundary → Vector{Tuple}
c   = contour_ellipse(m, "a", "b")              # fast HESSE-ellipse approx → ContoursError
c.points        # ← the boundary (Vector{Tuple});  NOT c.xs / c.ys (no such fields)
c.valid
xs, ys, F = contour_grid(m, "a", "b")           # iminuit m.contour: FCN GRID SLICE,
                                                #   others FIXED (landscape, NOT a CL region)
prof = profile(m, "a")                          # scan, no inner re-min (diagnostic)
mnp  = mnprofile(m, "a")                         # true profile likelihood (re-minimizes rest)

for (p, me) in m.merrors                         # report (value, +σ, −σ) per param (after minos!)
    println(p, " = ", m.values[p], "  +", me.upper, " / ", me.lower)
end
to_latex(m)                                      # LaTeX table of the result (Jupyter-friendly)
```
- **MINOS needs a covariance** — after `simplex`/`scan` (which leave none), run
  `hesse!(m)` first. A normal `migrad!` already leaves one.
- **0.5.0 renames**: bare `contour` is NO LONGER EXPORTED (it collided with
  `Plots.contour`). The old JuMinuit ellipse is `contour_ellipse`; iminuit's /
  IMinuit.jl's grid-scan `contour` is `contour_grid` (returns `ContourGrid`;
  destructures to `(xs, ys, F)`; `plot(g)` gives a filled contour). A grid
  slice's Δχ² level curves are CONDITIONAL (others pinned) — smaller than the
  true region by ≈√(1−R²) per axis on correlated fits → confidence regions
  come from `mncontour`, landscape from `contour_grid`.
- `draw_mncontour` / `draw_mnmatrix` render the **exact** `mncontour` (≥ 0.5.0;
  earlier versions silently drew the ellipse); `draw_contour` = `contour_grid`
  landscape (needs `using Plots`).
- Point-count kwargs **differ**: `mncontour` takes `numpoints` (default 100),
  `contour_ellipse` takes `npoints` (default 20), `contour_grid` takes `size`
  (default 50, per axis).
- `mncontour` **cl semantics (0.5.0+, = iminuit ≥ 2.0)**: default → JOINT 2-D
  68 % region (Δχ²≈2.28); `0<cl<1` → that joint prob; `cl≥1` → nσ (cl=2 →
  joint 95.45 %, Δχ²≈6.18). The old C++ Δχ²=1 curve (projections = MINOS ±1σ;
  joint coverage only 39.3 % — James, *Interpretation of Errors* §1.3.3) is
  `mncontour(m,a,b; cl = chisq_cl(1,2))` or low-level `contour_exact`
  (`sigma=1`). Pre-0.5.0 `mncontour` traced Δχ²=1 and ERRORED on `cl≠1`.
- MINOS/contours mislead on a flat, strongly non-Gaussian, or multimodal
  surface (`is_valid(me)==false`, `upper_new_min` fires, ragged contour) → use
  the sampling tools below.

## Derived quantities: `extremize` / `profile_band` (post-0.5.0; no iminuit equivalent)

"MINOS for a function": the profile interval of ANY scalar `f(θ)` over the
`Δχ² ≤ delta_chisq(cl,1)·up` region (all free params vary; limits/fixed
honoured; `f(θ)=θ[i]` ≡ MINOS; linear-Gaussian limit = `f̂ ± √(Δχ²·cᵀCc)`).
`ndof=1` regardless of n_params — the quoted statement is ONE number.

```julia
r = extremize(m, θ -> θ[1] + θ[2]*15.0)          # cl=1 (68.3%); cl=2, cl=0.95 ok
r.lo, r.hi          # interval; r.plo / r.phi = extremal param vectors
r.diagnostics       # PER-SEED audit: accepted? f? winner_min/max (0 = best-fit fallback)

ext = sort(collect(mcmc_sample(m; seed = 1)); by = θ -> model(4.42, θ))
band = profile_band(m, (x, θ) -> model(x, θ), 4360.0:2.0:4520.0;   # f(x, θ): x first
                    seeds = [ext[1], ext[end]])  # ensemble f-extremes = seed bank
band.lo, band.hi, band.fbest                     # ribbon + central curve
band.nfail == 0 || @warn "inspect band.diagnostics"
```
- **Seed coverage is load-bearing**: a single seed stops at a LOCAL tangency —
  on a multi-corridor region (multi-basin, param on a limit) the interval is
  silently too narrow. Pass `seeds =` `mcmc_sample` ensemble members extreme
  in `f` / `find_solution_modes` representatives (vector-of-vectors or matrix
  rows); the best fit is always seed 1. ALWAYS check `r.diagnostics` for who
  won (`winner_*==0` + `naccepted_*>0` = best fit genuinely extremal;
  `naccepted_*==0` = that side FAILED).
- Band is **pointwise** (each x its own 68%) and contains the best-fit curve
  by construction — say "pointwise" in the figure caption.
- Joint statement instead? `extremize(m, f; delta = delta_chisq(cl, 2))`.
- Cost ≈ `2 × seeds × ≤3 ladder stages × rounds` MIGRADs (band: × points ×
  passes); budget with `maxfcn`, `rounds`, fewer seeds.
- **Expensive FCN / `f` (≥ seconds/eval), near-linear ⇒ `mode = :directional`**
  (0.5.3, on both `extremize` and `profile_band`): walks `d = C·∇f` and
  secant/bisects the TRUE FCN to the boundary — `≈ n_free + ~15` paired evals
  (~50× cheaper than the default `:full`; a band is `×points`), **exact in the
  linear-Gaussian limit**. `r.mode`/`b.mode` flags it; pass `grad_f` (`θ->∇f`,
  or `(x,θ)->∇_θf` for the band) to skip the numeric gradient. It IGNORES
  `seeds`/limits and won't chase corridors — warns on bounded free params, and
  `profile_band` flags + best-fit-falls-back any point with a non-finite `f`
  or un-computable direction. Workflow: `:directional` first, then `:full`
  (default) if you suspect non-linearity / a binding limit / they disagree.
- On the `:full` path for an expensive FCN: `rounds=1, iterate=1, strategy=0`
  (+ `maxfcn`) is the cheapest; `on_unit = u -> …` fires per penalty-MIGRAD for
  live progress / external checkpointing.
- **`f`-failure contract** (0.5.3): `f` may THROW or return a non-finite value
  at infeasible θ — both are safe (treated as out-of-region; tallied
  `f_nonfinite`). Do NOT return a sentinel like `0.0` from a failing `f` — that
  silently biases the endpoint toward the centre.

## Gradients: AD & threading (beyond C++ Minuit2)

Default is serial central-difference numerical — right for cheap FCNs. Two extras:

**AD (ForwardDiff)** — one exact gradient call instead of `2n` finite diffs.
Best for expensive FCNs generic on element type.
```julia
using JuMinuit, ForwardDiff                       # extension auto-activates
m = Minuit(chi2, x0; error = errs, grad = x -> ForwardDiff.gradient(chi2, x))
migrad!(m)                                         # AD flows through MINOS/contours too
# or a cost object carrying its own AD gradient:
cf = CostFunctionAD(chi2, 1.0); fmin = migrad(cf, x0, errs)   # up=1 (χ²), 0.5 (NLL)
```
**AD requires the FCN generic on eltype** (it runs on `ForwardDiff.Dual`):
- write `f(x)` **not** `f(x::Vector{Float64})`;
- use `complex(...)` / `im`, **not** `Complex{Float64}` literals (HEP amplitudes "just work");
- scratch as `similar(x, eltype(x))` or `Matrix{Complex{eltype(par)}}(undef,…)`, never `zeros(Float64,…)` inside `f`.
A `check_gradient` warning at the seed almost always means the FCN is **not**
actually generic (AD silently fell back to something wrong).

**Threaded numerical gradient** — for expensive FCNs that can't be made generic.
Start `julia -t N`, set `threaded_gradient = true` (or `:auto`).
```julia
m = Minuit(my_chi2, x0; error = errs, threaded_gradient = true)   # auto-verifies safety
```
- `true` = force + verify — a **seed-point** probe: throws `ThreadSafetyError` if the
  race shows up at the seed (catches the common shared-buffer race, but is **not** a
  proof of safety away from it). `:auto` = thread if safe else `@warn` + serial (never
  throws). `false` = serial (default). Either way the FCN contract below still holds.
- **Contract: the FCN must not share mutable state across threads.** The classic
  HEP bug is a `const BUF = zeros(ComplexF64,…)` mutated inside the FCN — parallel
  calls race and MIGRAD **silently converges to the wrong minimum**. Fix: allocate
  scratch per call, or one buffer per thread indexed by `Threads.threadid()` and
  sized with `Threads.maxthreadid()` (JuMinuit threads with `@threads :static`).
  Probe standalone with `is_thread_safe(cf, x0)`.

## Hard surfaces: multi-basin / deeper minima

A single MIGRAD only reaches the basin its start drains into. On ill-conditioned
(e.g. coupled-channel / amplitude) fits, **find the true minimum first, then do
LOCAL error analysis there**. Naive bootstrap/jackknife are unreliable here (each
resample re-fits into a possibly-different basin).

```julia
# find_deeper_minimum: basin-hopping. Returns a Minuit (MIGRAD+HESSE run) — check .valid.
# Honors the fit's limits & fixed params. Two dispatches:

m_deep = find_deeper_minimum(m)                       # (1) parameter-perturbation: any objective
m_deep = find_deeper_minimum(m; n_restarts=40, perturb=1.5, seed=1)

m_deep = find_deeper_minimum(m, refit, data)          # (2) data-resampling: stronger on data fits
# refit(subdata, start) -> param vector (NaNs ⇒ invalid/dropped)
m_deep.valid || error("search failed")
minos!(m_deep)                                         # error analysis HERE, at the deep min

# cluster already-sampled points into statistically distinct solutions:
r     = get_contours_samples(m; nsamples = 30_000, cl = 1)
modes = find_solution_modes(r.samples, m; refine = true)   # refine ⇒ re-fit each mode
# mode.new_min == true ⇒ that cluster re-fit DEEPER than global best (main fit missed it)
# whiten=:auto (default) picks the metric: fit-scale cloud → :cov (Mahalanobis);
# cloud wider than the fit scale (multi-basin / cross-basin scan) → :sample
# (robust cloud MAD — the fit-local metrics report 0 modes there). Force with
# whiten=:sample/:cov/:errors. K=0 / mostly-noise emits a diagnostic (NN scale
# + suggested fix). Expensive FCN: fvals=:none (0 FCN calls, medoid reps,
# population-sorted) / :lazy (K calls) / fvals=<precomputed χ² vector>; budget
# re-fits with refine_maxfcn=500, refine_strategy=0, refine_tol=…, and
# checkpoint with refine_callback = r -> …  (serialized; r.k/r.K, r.refined_*,
# r.walltime; per-mode refined_nfcn + refined_walltime are also on the result).
```
`find_deeper_minimum` is a heuristic (finds *a* deeper min, not certified global);
raise `n_restarts`/`perturb`/`n_discovery`/`max_rounds` and cross-check seeds.

## Error-analysis menu — which method, when

Two families that **agree** on a clean near-Gaussian fit and **diverge** when it matters:

| Method | Call | Use when |
|---|---|---|
| **HESSE** | `hesse!(m)` → `m.errors`, `m.covariance` | default; fast symmetric error, near-Gaussian fit |
| **MINOS** | `minos!(m)` → `m.merrors` | asymmetric errors under mild–moderate nonlinearity |
| **extremize / profile band** | `extremize(m, f; seeds=…)`, `profile_band(m, f, xs; …)` | **profile interval/band of a derived quantity** ("MINOS for a function"); contains the best fit by construction |
| **MC-Δχ²** | `get_contours_samples(m; …)` | non-Gaussian or **joint** N-D confidence region |
| **MCMC ensemble** | `mcmc_sample(m; …)` → `quantiles` / `quantile_band` | **marginal quantile bands of derived quantities** (curves, ratios); active limits |
| **Bayesian posterior** | `bayesian(m; …)` / `posterior_sample(m; prior=…)` → `credible_interval` / `upper_limit` | a **credible** interval/limit under an explicit **prior** (e.g. upper limit on a near-zero coupling); `prior=:flat` ⇒ the MCMC path |
| **bootstrap** | `bootstrap(model, Data(x,y,σ), start)`, `bootstrap(cost, start)`, or `bootstrap(refit, data)` | you **doubt the error model** (quoted σ); want empirical sampling dist |
| **jackknife** | `jackknife(model, Data(x,y,σ), start)` (or a `cost` / `refit` form) | quick error **+ explicit bias** estimate |

`get_contours_samples` **gotchas**: (1) `ndof` is the **dimension of the region**, not
the fit's param count — it defaults to `n_free` (joint region). A 2-D joint 1σ is
`Δχ² = 2.30`, **not** 1.0. `cl` = confidence level (iminuit convention): `cl ≥ 1` ⇒ that
many σ (`cl=1`→68.27 %, `cl=2`→95.45 %), `0 < cl < 1` ⇒ a probability — so `cl=2` ≠ `cl=0.95`.
(2) it **samples all free parameters jointly**; `paras` only
filters the *reported* `bounds`/`names`, not the sampling — so `proposal = :uniform`'s
`ranges` needs **one `(lo,hi)` per free parameter**, in order. (3) the default
`proposal = :mvnormal` uses the fit covariance; when that is unreliable switch to the
covariance-free box `get_contours_samples(m; proposal = :uniform, ranges = […])`.
Returns a NamedTuple (`.samples`, `.bounds`, `.acceptance`, `.under_coverage`).
For a correlation matrix use `correlation(result)` on a **bootstrap/jackknife
result or a `Minuit`** — there is **no** `correlation` method for this sampler
NamedTuple; for the accepted cloud compute it directly (`using Statistics; cor(r.samples)`).

**Likelihood-ensemble MCMC** (`mcmc_sample`, 0.5+): Metropolis chain on the **exact
FCN** (`exp(−Δfcn/(2·up))`) — NOT a `Δχ²` region sampler; samples live at the typical
set `Δχ² ≈ n_free` (volume effect: `P(Δχ²₉ ≤ 1) ≈ 5.6e-4`), which is what makes it the
right tool for **derived-quantity bands** where `get_contours_samples(ndof=1)` accepts
almost nothing in high dimension.

```julia
ens = mcmc_sample(m; seed = 11)          # defaults 52k/burn 2k/thin 25 → 2000 sets; needs migrad! first
ens.acceptance                            # ~10D heuristic ≈ 0.2–0.3 (low-dim accepts higher, ~0.8 fine); target_accept=0.25 autotunes
q16, q50, q84 = quantiles(ens, θ -> θ[2]/θ[1])                  # scalar marginal quantiles
B = quantile_band(ens, (x, θ) -> model(x, θ), xs)               # nx×2 pointwise 16–84% band
B = quantile_band(ens, θ -> curve(xs, θ), xs; curve = true)     # 1 call/member (expensive models)
save_ensemble("ens.dat", ens; comment="…"); ens = load_ensemble("ens.dat")  # reusable error set
```
- `proposal=:hesse` (default; auto-falls back to `:errors` if Σ unreliable) /
  `:errors` / explicit `[σ₁,…]` per free param (the escape hatch when a parameter
  sits AT a limit and HESSE σ there is squeezed/meaningless) / explicit Σ matrix.
  Proposal shape affects **mixing only**, never the stationary distribution.
- `limits` enforced by **rejection** → the chain samples the TRUNCATED likelihood:
  one-sided pile-up at an active boundary is physics, and the 16–84% band may then
  legitimately **exclude the best fit** (mode ≠ median — property, not bug). A band
  that must contain the best fit is `profile_band` (the `extremize` profile-envelope
  construction) — quote which one you used. The ensemble doubles as the `seeds=`
  pool for `extremize`/`profile_band` (pass the members extreme in `f`).
- `minimum(ens.fvals) < ens.fbest` ⇒ chain found a deeper minimum → `find_deeper_minimum`.
- Fixed params don't move; `m.nfcn` untouched; `seed=`/`rng=` for reproducibility.

**Bayesian posterior bridge** (`bayesian` / `posterior_sample`, 0.5+): the SAME
Metropolis kernel, now sampling `prior × exp(−fcn/(2·up))` and reporting
**credible** (not confidence) summaries. Non-mutating: never writes `m.values` /
`m.errors` / `m.covariance` / `m.nfcn`.

```julia
report = bayesian(m; level = 0.6827)        # one-step report; flat prior; m untouched
pr   = normal_prior(m, :mass, 3.8717, 2e-4) # priors: flat_/normal_/uniform_/half_normal_prior, combine_priors
post = posterior_sample(m; prior = pr, nchains = 4, seed = 11)   # PosteriorSample (reusable)
maximum(post.rhat) < 1.01                    # split-R̂ (needs nchains≥2); effective_sample_size(post,:mass) too
ci  = credible_interval(post, :mass; level = 0.6827)            # (lo,hi) equal-tailed marginal
gup = upper_limit(post, :g; level = 0.90)                       # 90% credible upper limit (CredibleLimit)
db  = derived_interval(post, θ -> θ[2] - θ[1]; level = 0.6827)  # credible interval of any scalar f(θ)
posterior_mean/median/std(post, :mass); posterior_summary(post) # point summaries / table
```
- **Credible ≠ confidence**: `upper_limit`/`credible_interval` are prior-conditional;
  NOT MINOS/CLs/Feldman–Cousins. `prior=:flat` reproduces the single-chain
  `mcmc_sample` path **byte-for-byte** (same seed).
- `flat_prior` is flat in **external** coords (parameterization-dependent, not
  "uninformative"/Jeffreys). Support = Minuit `limits` ∩ prior support; construction
  **throws** if the best fit is outside it (re-minimize or fix the prior).
- **Posterior temperature follows `errordef`**: keep `up = 1` (χ²) or `0.5` (−log L);
  inflating `up` for a wider MINOS interval tempers the posterior by `√up` — put
  extra information in the **prior**, not in `errordef`.
- **Samplers** (`sampler=`): `:metropolis` (default, random walk; `proposal`/`scale`/
  `target_accept`/`overdisperse`, `nchains=4`, ≈2σ over-dispersed so split-R̂ is real);
  `:stretch` (affine-invariant Goodman–Weare ensemble — **gradient-free**, ANY FCN incl.
  non-AD complex-χ², beats RWM on strong correlation; knobs `nwalkers`/`stretch`);
  `:nuts` (gradient NUTS via AdvancedHMC **extension** — `using AdvancedHMC,
  LogDensityProblems, LogDensityProblemsAD, TransformVariables, ForwardDiff`; best for
  smooth high-dim; **needs an AD-able FCN** — errors→use `:stretch` — and a best fit off
  the limits). `rhat` is basic split-R̂ — for skewed/boundary marginals also check ESS +
  trace. `method=:central` only (HPD throws, not silently approximate).

## Gotchas cheat-sheet (the non-guessable list)

1. `migrad!(m)` / `hesse!(m)` / `minos!(m)` — **bang functions, mutate `m`**; not `m.migrad()`.
2. Algorithm helpers take `m` first: `mncontour(m,a,b)`, `profile(m,a)`. Params by Int **or** name (`String`).
3. FCN is `f(x::AbstractVector)->Real` and **always sees external (physical) coords**, even bounded.
4. Results are **properties**: `m.values`/`m.errors`/`m.fval`/`m.valid`/`m.covariance`/`m.merrors`. `args(m)` for a plain `Vector`.
5. `up`/`errordef` = **1.0 for χ², 0.5 for −lnL**; cost objects set it automatically.
6. `set_lower_limit!`/`set_upper_limit!` **clear the other side** → use `set_limits!` for two-sided.
7. MINOS needs a covariance → after `simplex`/`scan`, `hesse!(m)` first.
8. `mncontour` = exact CL boundary; `contour_ellipse` = fast ellipse (read `.points`/`.valid`, no `.xs`/`.ys`); `contour_grid` = iminuit grid slice (landscape, NOT a CL region). Bare `contour` is unexported since 0.5.0 (Plots clash).
9. AD: FCN **generic on eltype** + `using ForwardDiff` + `grad = x->ForwardDiff.gradient(f,x)`.
10. Threading: `julia -t N` + thread-safe FCN (no shared mutable buffers) + `threaded_gradient=true`.
11. `get_contours_samples` `ndof` = region dimension (joint by default; Δχ²=2.30 for 2-D 1σ).
12. On multi-basin surfaces: `find_deeper_minimum` first, **then** local errors; bootstrap/jackknife unreliable.
13. **Extensions** load on demand: `using Plots` (plotting / `draw_*`), `using Optim` (`optim`/`minimize_with`), `using ForwardDiff` (AD / `CostFunctionAD`), `using DataFrames` (`contour_df_samples`, `Data(::DataFrame)`, `DataFrame(bootstrap/jackknife result)`), `using Clustering` (`find_solution_modes(...; method=:dbscan)`).
14. `Fit` / `ArrayFit` are **aliases of `Minuit`** (no behavioral difference).
15. `mcmc_sample` quantile bands are **marginal** — at an active limit they may exclude the best fit (correct!); the always-contains-best-fit band is `profile_band` (the `extremize` `Δχ²≤delta_chisq(cl,1)` envelope). Don't "fix" one to match the other — quote which you used.

## Authoritative docs (read these for depth beyond this skill)

- **GitHub:** https://github.com/fkguo/JuMinuit.jl  · **Manual:** https://fkguo.github.io/JuMinuit.jl/dev
- **On this machine** (if present): `/Users/fkg/Coding/Agents/ResearchWork/JuMinuit` —
  `docs/src/` (quickstart, bounded, minos_contours, cost_functions, **error_analysis**,
  guides/gradients) and the docstrings in `src/`. Worked HEP examples in
  `BenchmarkExamples/` (e.g. `IAM_2Pformfactor/`, `X3872_dip/`).
- For the **full** error-analysis decision guide (HESSE/MINOS/MC-Δχ²/bootstrap/
  jackknife/multi-modal, with the Δχ² threshold table and pitfalls) see
  `docs/src/error_analysis.md` — it is the authoritative map; don't reproduce it from memory.
