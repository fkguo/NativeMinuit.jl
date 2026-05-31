# MINOS errors & contours

After `migrad` converges, the symmetric Hesse error matrix may
under-report uncertainty for non-Gaussian likelihoods. MINOS computes
**asymmetric ±σ errors** by profile likelihood (re-minimizing all other
parameters while constraining one). Contours generalize this to 2-D
confidence regions.

## Asymmetric MINOS errors

```julia
using JuMinuit

cf = CostFunction(x -> (x[1] - 1.0)^2 + 0.5 * (x[1] - 1.0)^4 + (x[2] - 2.0)^2)
m = migrad(cf, [0.0, 0.0], [0.1, 0.1])

# Single-parameter MINOS
err1 = minos(m, cf, 1)
@show err1.min_par_value err1.upper err1.lower
@assert is_valid(err1)

# All parameters at once
errs = minos(m, cf)              # Vector{MinosError}
for e in errs
    println("par $(e.par_idx): +$(e.upper) −$(-e.lower)")
end
```

The `MinosError` fields:

| Field             | Meaning                                                  |
|:------------------|:---------------------------------------------------------|
| `par_idx`         | 1-based parameter index                                  |
| `min_par_value`   | Parameter value at the minimum                           |
| `upper`           | Positive 1σ error (≥ 0)                                  |
| `lower`           | Negative 1σ error (≤ 0)                                  |
| `upper_valid`     | `true` if the upper crossing converged                   |
| `lower_valid`     | `true` if the lower crossing converged                   |
| `upper_new_min`   | `true` if a new lower minimum was found scanning upward  |
| `lower_new_min`   | `true` if a new lower minimum was found scanning downward|
| `upper_fcn_limit` | `true` if maxcalls hit before upper crossing            |
| `lower_fcn_limit` | `true` if maxcalls hit before lower crossing            |

The pretty-print of `MinosError` shows a 3-row Unicode box matching
iminuit's text repr.

## 2-D confidence contours

```julia
# Ellipse approximation from the Hesse covariance (fast)
c = contour(m, cf, 1, 2; npoints = 32)
# c.points = vector of (x, y) tuples along the 1σ contour

# Exact profile contour via MnContours: more expensive but accurate
# even when the likelihood is non-Gaussian
c_exact = contour_exact(m, cf, 1, 2; npoints = 12)
```

Both `contour` and `contour_exact` return a `ContoursError` struct; the
contour points are in `c.points` (a vector of `(x, y)` tuples) and `c.valid`
flags success. The "exact" variant
re-minimizes the cost function with two parameters fixed at each
contour point — slow but matches C++ `MnContours` results to high
precision.

## Plotting

JuMinuit ships [RecipesBase](https://github.com/JuliaPlots/RecipesBase.jl)
recipes for results:

```julia
using Plots                 # or Makie + StatsMakie
plot(c)                      # closed polygon (1σ contour)
plot(errs)                   # bar-with-error-bars
plot(m)                      # parameter values + errors
```

The recipes attach reasonable defaults (markers for fixed parameters,
yerror from `errors(m)`) and pass through any `plot` kwargs.

## CONTOURS for the bounded path

When parameters have bounds, `contour_exact` uses the same internal-
coordinate MIGRAD-with-fixed machinery as `function_cross_multi`, then
maps the profile points back to external coordinates via the bound
transforms. The returned contour points (`c.points`) are always in external
(user) coords.
