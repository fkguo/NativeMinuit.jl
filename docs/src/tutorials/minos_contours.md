# MINOS errors & contours

After [`migrad!`](@ref) converges, the symmetric Hesse error matrix
assumes the cost surface is parabolic. When it isn't — non-Gaussian
likelihoods, nonlinear models — that under- or over-states the
uncertainty. MINOS computes **asymmetric ±σ errors** from the profile
likelihood (re-minimizing all other parameters while one is scanned);
contours extend the same idea to 2-D confidence regions. This tutorial
uses the [`Minuit`](@ref) object throughout.

We work with a deliberately non-parabolic example:

```julia
using JuMinuit

fcn(x) = (x[1] - 1.0)^2 + 0.5 * (x[1] - 1.0)^4 + (x[2] - 2.0)^2
m = Minuit(fcn, [0.0, 0.0]; names = ["a", "b"])
migrad!(m)
```

## Asymmetric MINOS errors

```julia
minos!(m)                  # MINOS on every free parameter
m.merrors                  # Dict{String,MinosError}, keyed by parameter name

# A single parameter (by name or 1-based index):
minos!(m, "a")
me = m.merrors["a"]
@show me.upper me.lower    # +σ (≥ 0) and −σ (≤ 0); generally |upper| ≠ |lower|
```

After `minos!`, displaying `m` widens the rich table into a side-by-side
comparison view: separate `Value`, `Hesse` and `MINOS` columns, so the
asymmetric MINOS error sits next to its symmetric Hesse counterpart. A MINOS
side that failed to converge is marked `invalid` (so a one-sided MINOS still
shows the side it got, and a fully-failed one shows `invalid`); `—` means
MINOS was not run for that parameter. Non-converged parameters are also
listed in a warning line below the table. Each
[`MinosError`](@ref) carries:

| Field             | Meaning                                                       |
|:------------------|:--------------------------------------------------------------|
| `par_idx`         | 1-based parameter index                                       |
| `min_par_value`   | parameter value at the minimum                                |
| `upper`           | positive 1σ error (≥ 0)                                       |
| `lower`           | negative 1σ error (≤ 0)                                       |
| `upper_valid`     | `true` if the upper crossing converged (or hit a bound)       |
| `lower_valid`     | `true` if the lower crossing converged (or hit a bound)       |
| `upper_new_min`   | `true` if a deeper minimum was found scanning upward          |
| `lower_new_min`   | `true` if a deeper minimum was found scanning downward        |
| `upper_par_limit` | `true` if the upward scan stopped at a parameter bound        |
| `lower_par_limit` | `true` if the downward scan stopped at a parameter bound      |
| `upper_fcn_limit` | `true` if the call budget ran out before the upper crossing   |
| `lower_fcn_limit` | `true` if the call budget ran out before the lower crossing   |
| `nfcn`            | total FCN calls across both directions                        |
| `upper_state`     | full parameter vector at the upper crossing (`nothing` if invalid) |
| `lower_state`     | full parameter vector at the lower crossing (`nothing` if invalid) |

`is_valid(me)` is `true` when both sides terminated cleanly (a bound counts
as clean — see [Bounded parameters](bounded.md)). Always gate on
`upper_valid` / `lower_valid` before trusting a side: on a failed crossing
the published value falls back to the symmetric Hesse placeholder.

If you only need one side, `minos_upper(m, "a")` /
`minos_lower(m, "a")` return just that value without mutating
`m`. The whole-fit alias `minos!(m; sigma = 2)` widens the scan to the
2σ crossing.

## 2-D confidence contours

`mncontour` traces the **exact** MINOS contour in a parameter plane — the
C++-faithful `MnContours` boundary search, re-minimizing the other
parameters at each point. It returns a vector of `(x, y)` points (matching
iminuit's `m.mncontour`):

```julia
pts = mncontour(m, "a", "b"; numpoints = 40)   # Vector{Tuple{Float64,Float64}}
xs = first.(pts);  ys = last.(pts)
```

If you'd rather have the structured result, the [`contour`](@ref)`(m, …)`
method returns a [`ContoursError`](@ref) — a fast ellipse approximation
from the Hesse covariance and the MINOS axes, good for a quick visual
check:

```julia
c = contour(m, "a", "b"; npoints = 32)
```

Read the boundary from `c.points` and the status from `c.valid` — **not**
`c.xs`/`c.ys` (there are no such fields):

```julia
c.points        # Vector{Tuple{Float64,Float64}} — the boundary in (x, y)
c.valid         # true if MINOS succeeded on both axes
```

The exact algorithm additionally records the **full parameter vector at
every boundary point** (the two contour coordinates plus the profiled
rest) at no extra cost. The lower-level [`contour_exact`](@ref) returns a
`ContoursError` whose [`contour_parameter_sets`](@ref) gives those vectors
— the native analogue of IMinuit.jl's `get_contours`. The ellipse
[`contour`](@ref) does no inner re-minimization, so it leaves that field
empty.

## 1-D profiles

Two scans trace the cost along a single parameter:

```julia
# profile: scan `a` WITHOUT re-minimizing the others — pure diagnostic,
# does not move m. Returns (a_value, fval) pairs.
prof = profile(m, "a"; bins = 100)

# mnprofile: at each grid point FIX `a` and RE-MINIMIZE the rest — the
# true profile likelihood. Returns (a_value, min_fval) pairs.
mnp = mnprofile(m, "a"; bins = 30)
```

The two default their range **differently** (both then clipped to any bound).
`mnprofile` always defaults to `m.values[par] ± 2·m.errors[par]`. `profile`
(which dispatches to [`scan`](@ref)) instead defaults to the parameter's **full
two-sided bounds** `(lower, upper)` when *both* are finite, and only falls back
to `m.values[par] ± 2·m.errors[par]` when the parameter is missing one or both
bounds. `mnprofile` is strictly more informative — its minimum-`fval` curve
crosses `fmin + up` exactly where MINOS reports the ±σ errors — but costs one
inner MIGRAD per point.

## Plotting

JuMinuit ships [RecipesBase](https://github.com/JuliaPlots/RecipesBase.jl)
recipes, so plotting works from Plots.jl with no glue:

```julia
using JuMinuit, Plots
plot(c)                                    # closed contour polygon
```

There are also IMinuit.jl-style draw helpers (`draw_mncontour`,
`draw_profile`, `draw_mnprofile`, `draw_mnmatrix`) that build the
corresponding plot for you when `using Plots` is loaded. Note that
`draw_mncontour` / `draw_mnmatrix` currently render the fast
covariance-**ellipse** [`contour`](@ref), not the exact `mncontour` /
[`contour_exact`](@ref) boundary, despite the `mn` in their names.

## When MINOS or MnContours fail

MINOS and the contour search both need a genuinely valid, locally
parabolic-enough minimum. They fail or mislead when the surface is **flat**
(an unconstrained direction), **strongly non-Gaussian**, or **multimodal**.
Symptoms: `is_valid(me)` is `false`, `upper_new_min` / `lower_new_min`
fires (MINOS fell into a deeper basin), `c.valid` is `false`, or a contour
comes back ragged.

When that happens, switch to the Monte-Carlo Δχ² and resampling tools in
the [Error analysis](../error_analysis.md) guide: `get_contours_samples`
maps a non-Gaussian or joint confidence region directly, `find_solution_modes`
detects distinct solutions hiding under one error bar, and
[`bootstrap`](@ref) / [`jackknife`](@ref) give a model-light cross-check
when you doubt the error model itself. That guide is the map for choosing
the right uncertainty method.
