# Bounded parameters

Real fits constrain parameters: a width must stay positive, a mixing angle
lives in `[0, π/2]`, a normalization is fixed by an external measurement.
NativeMinuit handles all of this through the [`Minuit`](@ref) object — bounds
and fixed flags are set at construction or mutated afterward, exactly like
iminuit.

## Adding bounds at construction

Pass `limits` — one entry per parameter. Each entry is `nothing`
(unbounded), `(lo, hi)` (two-sided), `(lo, nothing)` (lower only), or
`(nothing, hi)` (upper only):

```julia
using NativeMinuit

fcn(x) = (x[1] - 0.5)^2 + (x[2] - 3.0)^2

m = Minuit(fcn, [0.3, 1.0];
           names  = ["frac", "mass"],
           limits = [(0.0, 1.0), (0.0, nothing)])   # frac ∈ [0,1]; mass ≥ 0

migrad!(m)
@assert 0.0 ≤ m.values["frac"] ≤ 1.0                # bound respected exactly
```

The FCN **always sees external (physical) coordinates** — inside `fcn`,
`x[1]` is the real `frac`, already mapped into `[0, 1]`. You never deal
with the transformed coordinate yourself.

## What the transforms do

Minuit enforces bounds by optimizing an *unbounded* internal coordinate
and mapping it to the bounded external one (no projected-gradient or
active-set machinery). The map depends on which bounds are present:

| Bounds         | Transform | External value                                    |
|:---------------|:----------|:--------------------------------------------------|
| both `(lo,hi)` | sin       | `lo + (hi − lo)·(sin(int)+1)/2`                   |
| lower only     | √         | `lo − 1 + √(int² + 1)`                            |
| upper only     | √         | `hi + 1 − √(int² + 1)`                            |
| none           | identity  | `int`                                             |

This mirrors C++ Minuit2 exactly. The practical consequences:

- the optimizer can never propose an out-of-range value — the bound is
  respected at *every* probe, not just at the optimum;
- but near a bound the map flattens (its slope → 0 at the boundary), which
  distorts the error estimate — see *How bounds interact with HESSE and
  MINOS* below.

## Fixing parameters

A fixed parameter is held bit-exactly at its value and excluded from the
fit. Set `fixed` at construction:

```julia
m = Minuit(fcn, [0.3, 5.0];
           names = ["frac", "mass"],
           fixed = [false, true])     # mass pinned at 5.0

migrad!(m)
@assert m.values["mass"] == 5.0       # exact, no roundoff
@assert m.errors["mass"] == 0.0       # fixed → zero error
```

A common workflow is the *fix–fit–release–fit* scan, e.g. to study one
parameter's pull on the rest:

```julia
fix!(m, "mass"); migrad!(m)           # fit with mass held
release!(m, "mass"); migrad!(m)       # free it again and refit
```

## Editing bounds and fixes after construction

Every constructor option has a per-parameter mutator (each accepts an
integer index **or** a name, drops the cached fit, and returns `m` for
chaining). They mirror the C++ `MnUserParameters` methods:

```julia
fix!(m, "frac")                       # exclude from the fit
release!(m, 1)                        # re-include parameter 1
set_value!(m, "mass", 3.0)            # change the starting value
set_error!(m, "mass", 0.2)            # change the step size

set_limits!(m, "frac", 0.0, 1.0)     # two-sided bound, keeps both sides
set_lower_limit!(m, "mass", 0.0)      # lower bound only  → [0, ∞)
set_upper_limit!(m, "mass", 10.0)     # upper bound only  → (-∞, 10]
remove_limits!(m, "mass")             # drop both bounds
```

!!! warning "`set_lower_limit!` / `set_upper_limit!` clear the other side"
    Matching C++ Minuit2, `set_lower_limit!` sets the lower bound
    **and clears any upper bound**; `set_upper_limit!` does the
    reverse. To keep both sides, use the two-sided [`set_limits!`](@ref)`(m, par, lo, hi)`.

The same edits are available through the iminuit-style index-assignment
views, which write straight back into `m`:

```julia
m.fixed["frac"]  = true               # ≡ fix!(m, "frac")
m.limits["mass"] = (0.0, 10.0)        # ≡ set_limits!(m, "mass", 0.0, 10.0)
m.limits["mass"] = nothing            # ≡ remove_limits!(m, "mass")
m.values["mass"] = 3.0                # ≡ set_value!(m, "mass", 3.0)
```

## How bounds interact with HESSE and MINOS

Both error methods run in internal coordinates and map back, so they
respect the bounds — but two effects are worth knowing.

**The error at a bound is asymmetric and Jacobian-distorted.** Because the
transform's slope vanishes at the boundary, a naive `√(V[i,i])` would
*under-report* the error there. NativeMinuit instead uses the C++
two-sided `Int2extError` formula for `m.errors`, which probes the map at
`int ± err` and averages, capturing the curvature near the bound. So the
reported symmetric error stays sensible even close to a limit.

**A parameter pinned at a bound is flagged.** If MIGRAD ends with a value
sitting on its `lower` or `upper` limit, the rich result table prints a
warning, and the bound makes the local error unreliable:

```
⚠ Parameter `frac` is at its lower limit — Hesse/MINOS error is unreliable.
```

**MINOS reports the bound distance, not an overflow.** When a MINOS scan
runs into a bound before `χ²` rises by `up`, that side is a clean
termination: the published error is the *distance to the bound*
(`bound − value`), and the corresponding [`MinosError`](@ref) flag
(`upper_par_limit` / `lower_par_limit`) is raised — matching iminuit's
`m.merrors[name].is_valid` semantics (hitting a bound is legitimate, not a
failure). If a parameter sits at a bound, that is usually a sign your bound
is too tight or the data don't constrain the parameter on that side.

When a parameter rails against a bound and you suspect the error model is
the culprit, the model-light cross-checks in the
[Error analysis](../error_analysis.md) guide (bootstrap / jackknife) are
the next tool.

## Next

Continue to [MINOS errors & contours](minos_contours.md) for asymmetric
errors and 2-D confidence regions, including how bounds propagate into the
contour search.
