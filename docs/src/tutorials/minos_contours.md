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

## 2-D contours: three tools, three jobs

| Function | What it computes | Cost | Use it for |
|---|---|---|---|
| [`mncontour`](@ref) | **exact** MINOS boundary (re-minimizes the others at every point) | high | **confidence regions** |
| [`contour_ellipse`](@ref) | error ellipse from the MINOS axes + covariance | low (2 MINOS runs) | quick near-quadratic preview |
| [`contour_grid`](@ref) | FCN values on a grid, **others held fixed** (iminuit's `contour`) | `size²` FCN calls | inspecting the **landscape** |

`mncontour` traces the **exact** MINOS contour in a parameter plane — the
`MnContours` boundary search, re-minimizing the other parameters at each
point. It returns a vector of `(x, y)` points and follows iminuit ≥ 2.0's
**joint-coverage `cl` semantics** (see the
[Δχ² conventions](@ref delta-chisq-conventions) section below):

```julia
pts = mncontour(m, "a", "b"; numpoints = 40)   # joint 2-D 68 % region (default)
pts95 = mncontour(m, "a", "b"; cl = 0.95)      # joint 95 % region
pts2σ = mncontour(m, "a", "b"; cl = 2)         # cl ≥ 1 ⇒ nσ → joint 95.45 %
xs = first.(pts);  ys = last.(pts)
```

If you'd rather have the structured result, [`contour_ellipse`](@ref)`(m, …)`
(named `contour` before 0.5.0) returns a [`ContoursError`](@ref) — a fast
ellipse approximation from the Hesse covariance and the MINOS axes, good
for a quick visual check:

```julia
c = contour_ellipse(m, "a", "b"; npoints = 32)
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
[`contour_ellipse`](@ref) does no inner re-minimization, so it leaves that
field empty.

### [Δχ², coverage, and the two contour conventions](@id delta-chisq-conventions)

A 2-D contour can answer two **different** statistical questions, and the
right `Δχ²` level differs between them. Both conventions come from
F. James himself; the distinction is spelled out in his Minuit document
*The Interpretation of Errors* (2004) and in Eadie/James et al.,
*Statistical Methods in Experimental Physics* (2nd ed., 2006; "SMEP"
below) — full citations at the end of this section.

**Question 1 — single-parameter errors.** The curve `FCN = fmin + up`
(`Δχ² = 1` for a χ² fit) is the curve whose *extreme points along each
axis are the MINOS ±1σ errors of that parameter*. This is what the raw
C++ `MnContours` traces, by design:

> "draw the contour line connecting all points where the function takes
> on the value `Fmin + UP` (MnContours will do this for you) … If MINOS
> is requested to find the errors in parameter one, it will find the
> extreme contour points A and B, whose x-coordinates … will be
> respectively the negative and positive MINOS errors of parameter one."
> — James, *The Interpretation of Errors*, §1.3.2

The single-parameter coverage of those projections is 68.3 % **whatever
the number of fit parameters** (SMEP p. 238: the MINOS interval for one
parameter uses `λ = Δ ln L = 1/2`, i.e. `Δχ² = 1`, in any dimension). But
read as a *2-D region*, this same curve covers far less:

> "The probability that parameter one *and* parameter two simultaneously
> take on values within the one-standard-deviation likelihood contour is
> **39.3 %**." — James, *The Interpretation of Errors*, §1.3.3

(SMEP Table 9.1 tabulates exactly this: the `K = 1` ellipse in two
variables has probability content 0.393; the worked example on p. 222
notes that two separate 68 % intervals cover both true values
simultaneously only 46 % of the time.)

**Question 2 — a joint confidence region.** For a *simultaneous*
statement about `NPAR` parameters, James prescribes scaling `up` by the
χ²(NPAR) quantile (his Table 1.3.3; SMEP §9.3.3 gives the same rule,
`ln L = ln L_max − ½χ²_β(k)`):

| coverage β | 1 par (`Δχ²`) | 2 par | 3 par |
|---|---|---|---|
| 68.3 % | 1.00 | 2.30 | 3.53 |
| 90 %   | 2.71 | 4.61 | 6.25 |
| 95 %   | 3.84 | 5.99 | 7.82 |
| 99 %   | 6.63 | 9.21 | 11.34 |

(For a negative-log-likelihood FCN all values are halved — that is what
`up = 0.5` already encodes. The general entry is
[`delta_chisq`](@ref)`(β, NPAR)`.)

**How JuMinuit maps the two conventions:**

- [`mncontour`](@ref)`(m, a, b; cl = …)` — **Question 2**, following
  iminuit ≥ 2.0: the default `cl` traces the joint 2-D 68 % region
  (`Δχ² = delta_chisq(0.68, 2) ≈ 2.28`); `cl ≥ 1` means nσ
  (`cl = 2` → joint 95.45 %, `Δχ² ≈ 6.18`).
- The **Question-1 curve** (projections = MINOS ±1σ; the C++/`MnContours`
  default and the convention of the 1994 MINUIT manual) is available as
  the low-level [`contour_exact`](@ref)`(fmin, cf, ix, iy)` (`sigma = 1`
  traces `fmin + up` exactly), or through `mncontour` with
  `cl = chisq_cl(1, 2) ≈ 0.3935`.
- Single-parameter errors themselves come from [`minos!`](@ref)
  (`Δχ² = up`, any dimension), not from a 2-D contour.

```julia
pts_joint = mncontour(m, "a", "b")                       # 68 % joint region
pts_cpp   = mncontour(m, "a", "b"; cl = chisq_cl(1, 2))  # Δχ²=1 curve (C++)
```

!!! warning "Label your contours"
    The two curves differ by √2.30 ≈ 1.5× in linear size. Calling the
    `Δχ² = 1` curve a "68 % confidence region" overstates its joint
    coverage (39.3 %); calling the joint-68 % contour's projections "the
    1σ parameter errors" overstates them by ~1.5×. State which convention
    a published contour uses.

**References.** F. James, *The Interpretation of Errors* (Minuit/Minuit2
documentation, CERN, 2004), §1.3 — distributed with Minuit2 and
[available from CERN](https://seal.web.cern.ch/documents/minuit/mnerror.pdf);
F. James, *MINUIT — Function Minimization and Error Analysis*, CERN
Program Library D506 (v94.1, 1994), §7; W. T. Eadie, D. Drijard,
F. E. James, M. Roos, B. Sadoulet, *Statistical Methods in Experimental
Physics*, 2nd ed. (World Scientific, 2006), §9.1.2–9.1.3 (Table 9.1),
§9.3.3, p. 238.

### FCN landscape: `contour_grid`

[`contour_grid`](@ref) is iminuit's `Minuit.contour` (what IMinuit.jl
exported as `contour`): the FCN evaluated on a 2-D grid with **all other
parameters pinned** at their best-fit values — the 2-D analogue of
[`profile`](@ref). No minimization happens; it is a cheap *map* of the
function near the minimum (valley orientation, hints of secondary minima):

```julia
xs, ys, F = contour_grid(m, "a", "b"; size = 50, bound = 2)  # iminuit-style
g = contour_grid(m, "a", "b"; subtract_min = true)
plot(g)                                       # filled-contour landscape
```

!!! warning "A slice is NOT a confidence region"
    Because the other parameters are *fixed* rather than re-minimized, the
    `Δχ²` level curves of a grid slice are **conditional** regions —
    systematically *smaller* than the true profile-likelihood region when
    `(a, b)` correlate with the remaining free parameters (per axis by
    ≈ `√(1−R²)`, `R` = multiple correlation with the rest; with only two
    free parameters slice ≡ profile). For confidence regions use
    [`mncontour`](@ref).

    Picking the level itself: `Δχ² = up` (i.e. `m.up`) is the curve whose
    per-axis **projections** are the single-parameter 68.27 % intervals
    (the C++ MnContours convention — its joint 2-D coverage is only
    39.3 %); the **joint** 2-D 68 % region needs
    `Δχ² = delta_chisq(0.68, 2) ≈ 2.28` (the [`mncontour`](@ref)
    default). See the
    [Δχ² conventions](@ref delta-chisq-conventions) section above.

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

There are also IMinuit.jl-style draw helpers (`draw_contour`,
`draw_mncontour`, `draw_profile`, `draw_mnprofile`, `draw_mnmatrix`) that
build the corresponding plot for you when `using Plots` is loaded.
`draw_mncontour` / `draw_mnmatrix` trace the exact [`mncontour`](@ref)
boundary (since 0.5.0 — earlier versions silently drew the ellipse
approximation); `draw_contour` shows the [`contour_grid`](@ref) FCN
landscape.

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
