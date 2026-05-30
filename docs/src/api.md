# API Reference

The public API is grouped by stage of a typical fit: declare the cost
function, run MIGRAD, refine with HESSE/MINOS, post-process. For
internal helpers and non-exported names see [Internals](internals.md).

## Cost function

```@docs
JuMinuit.CostFunction
JuMinuit.CostFunctionWithGradient
```

## MIGRAD (minimization)

```@docs
JuMinuit.migrad
JuMinuit.FunctionMinimum
JuMinuit.BoundedFunctionMinimum
```

## Parameters API (bounded / named / fixed)

```@docs
JuMinuit.MinuitParameter
JuMinuit.Parameters
```

## Bound transformations

```@docs
JuMinuit.BoundKind
JuMinuit.bound_kind
JuMinuit.int2ext
JuMinuit.ext2int
JuMinuit.dint2ext
JuMinuit.int2ext_error
```

## HESSE (covariance refinement)

```@docs
JuMinuit.hesse
```

## MINOS (asymmetric errors)

```@docs
JuMinuit.minos
JuMinuit.MinosError
JuMinuit.function_cross
JuMinuit.MnCross
```

## Contours

```@docs
JuMinuit.contour
JuMinuit.contour_exact
JuMinuit.ContoursError
```

## iminuit-style Minuit wrapper

```@docs
JuMinuit.Minuit
```

## Strategy & precision

```@docs
JuMinuit.Strategy
JuMinuit.MachinePrecision
```

## Cost functions

```@docs
JuMinuit.AbstractCost
JuMinuit.LeastSquares
JuMinuit.UnbinnedNLL
JuMinuit.ExtendedUnbinnedNLL
JuMinuit.BinnedNLL
JuMinuit.ExtendedBinnedNLL
JuMinuit.CostSum
JuMinuit.errordef
```

## Error analysis (sampling & confidence regions)

```@docs
JuMinuit.delta_chisq
JuMinuit.chisq_cl
JuMinuit.get_contours_samples
JuMinuit.contour_df_samples
JuMinuit.contour_parameter_sets
```

## Resampling (bootstrap & jackknife)

```@docs
JuMinuit.bootstrap
JuMinuit.jackknife
JuMinuit.BootstrapResult
JuMinuit.JackknifeResult
JuMinuit.correlation
```

## Multi-modal solution detection

```@docs
JuMinuit.find_solution_modes
JuMinuit.SolutionMode
JuMinuit.SolutionModes
```

## Alternative minimizers (Optim.jl bridge)

```@docs
JuMinuit.optim
JuMinuit.minimize_with
```

## Common accessors

These small accessor functions are exported but documented as part of
the parent struct (see `FunctionMinimum`, `MinosError`, etc.):

| Function          | Returns                                          |
|:------------------|:-------------------------------------------------|
| `fval(m)`         | function value at the minimum (Float64)          |
| `is_valid(m)`     | converged within tolerances (Bool)               |
| `nfcn(m)`         | total FCN call count (Int)                       |
| `errors(m)`       | 1σ Hesse errors per parameter (Vector{Float64})  |
| `covariance(m)`   | Symmetric covariance matrix                      |
| `gradient(m)`     | FunctionGradient at the minimum                  |
| `has_covariance(m)` | true if covariance is available                |
| `ext_covariance(m)` | full external covariance (bounded path)        |
| `free_covariance(m)` | n_free × n_free sub-block                     |
| `ext_errors(m)`   | external errors via Int2extError two-sided       |
| `has_limits(p)`   | both lower AND upper set (MinuitParameter)       |
| `is_fixed(p)`     | fixed flag (MinuitParameter)                     |

## Index

```@index
```
