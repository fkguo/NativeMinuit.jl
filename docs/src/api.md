# API Reference

The public API is grouped by stage of a typical fit: declare the cost
function, run MIGRAD, refine with HESSE/MINOS, post-process, and visualize.
For internal helpers and non-exported names see [Internals](internals.md).

## Cost function

```@docs
NativeMinuit.CostFunction
NativeMinuit.CostFunctionWithGradient
NativeMinuit.CostFunctionAD
```

## MIGRAD (minimization)

```@docs
NativeMinuit.migrad
NativeMinuit.FunctionMinimum
NativeMinuit.BoundedFunctionMinimum
```

## Parameters API (bounded / named / fixed)

```@docs
NativeMinuit.MinuitParameter
NativeMinuit.Parameters
```

## Bound transformations

```@docs
NativeMinuit.BoundKind
NativeMinuit.bound_kind
NativeMinuit.int2ext
NativeMinuit.ext2int
NativeMinuit.dint2ext
NativeMinuit.int2ext_error
```

## HESSE (covariance refinement)

```@docs
NativeMinuit.hesse
```

## Covariance & diagnostics

```@docs
NativeMinuit.covariance
NativeMinuit.eigenvalues
NativeMinuit.global_cc
NativeMinuit.CovStatus
```

## MINOS (asymmetric errors)

```@docs
NativeMinuit.minos
NativeMinuit.MinosError
NativeMinuit.minos_upper
NativeMinuit.minos_lower
NativeMinuit.function_cross
NativeMinuit.MnCross
```

## Contours & profiles

```@docs
NativeMinuit.mncontour
NativeMinuit.contour_grid
NativeMinuit.ContourGrid
NativeMinuit.contour_ellipse
NativeMinuit.contour_exact
NativeMinuit.ContoursError
NativeMinuit.profile
NativeMinuit.mnprofile
```

## iminuit-style Minuit wrapper

```@docs
NativeMinuit.Minuit
NativeMinuit.AbstractFit
NativeMinuit.Fit
NativeMinuit.ArrayFit
NativeMinuit.migrad!
NativeMinuit.minos!
NativeMinuit.hesse!
NativeMinuit.HesseResult
NativeMinuit.set_precision
```

## Gradients & threading

```@docs
NativeMinuit.is_thread_safe
NativeMinuit.ThreadSafetyError
```

## Other minimizers

```@docs
NativeMinuit.simplex
NativeMinuit.scan
```

## Per-parameter mutators

```@docs
NativeMinuit.fix!
NativeMinuit.release!
NativeMinuit.set_value!
NativeMinuit.set_error!
NativeMinuit.set_limits!
NativeMinuit.set_lower_limit!
NativeMinuit.set_upper_limit!
NativeMinuit.remove_limits!
```

## IMinuit.jl compatibility

```@docs
NativeMinuit.Data
NativeMinuit.chisq
NativeMinuit.model_fit
NativeMinuit.@model_fit
NativeMinuit.args
NativeMinuit.matrix
NativeMinuit.chi2
NativeMinuit.poisson_chi2
NativeMinuit.multinominal_chi2
NativeMinuit.func_argnames
```

## Strategy & precision

```@docs
NativeMinuit.Strategy
NativeMinuit.MachinePrecision
```

## Cost functions

```@docs
NativeMinuit.AbstractCost
NativeMinuit.LeastSquares
NativeMinuit.UnbinnedNLL
NativeMinuit.ExtendedUnbinnedNLL
NativeMinuit.BinnedNLL
NativeMinuit.ExtendedBinnedNLL
NativeMinuit.CostSum
NativeMinuit.errordef
```

## Error analysis (sampling & confidence regions)

```@docs
NativeMinuit.delta_chisq
NativeMinuit.chisq_cl
NativeMinuit.get_contours_samples
NativeMinuit.contour_df_samples
NativeMinuit.contour_parameter_sets
```

## Derived quantities (Δχ²-region intervals & profile bands)

```@docs
NativeMinuit.extremize
NativeMinuit.profile_band
NativeMinuit.ExtremizeResult
NativeMinuit.ProfileBand
```

## Likelihood-ensemble MCMC (marginal quantile bands)

```@docs
NativeMinuit.mcmc_sample
NativeMinuit.LikelihoodEnsemble
NativeMinuit.quantiles
NativeMinuit.quantile_band
NativeMinuit.save_ensemble
NativeMinuit.load_ensemble
```

## Bayesian posterior analysis (priors & credible intervals)

A Bayesian layer that never modifies the fit: `prior × likelihood` sampled in full external
coordinates, returning credible (not confidence) summaries. Three samplers —
`sampler = :stretch` (the default: the gradient-free, affine-invariant Goodman–Weare
ensemble), `sampler = :metropolis` (a HESSE-preconditioned random walk), and
`sampler = :nuts` (gradient-based NUTS, via the AdvancedHMC extension). With
`sampler = :metropolis, prior = :flat` a single chain (`nchains = 1`) reproduces
the likelihood path exactly at the same seed;
sampling never mutates the `Minuit` object or `m.nfcn`. See the
[Bayesian analysis guide](bayesian.md) for worked examples and how to enable NUTS.

```@docs
NativeMinuit.bayesian
NativeMinuit.BayesianReport
NativeMinuit.posterior_sample
NativeMinuit.PosteriorProblem
NativeMinuit.PosteriorSample
NativeMinuit.isconsistent
NativeMinuit.Prior
NativeMinuit.flat_prior
NativeMinuit.normal_prior
NativeMinuit.uniform_prior
NativeMinuit.half_normal_prior
NativeMinuit.combine_priors
NativeMinuit.credible_interval
NativeMinuit.derived_interval
NativeMinuit.upper_limit
NativeMinuit.lower_limit
NativeMinuit.CredibleLimit
NativeMinuit.posterior_summary
NativeMinuit.posterior_mean
NativeMinuit.posterior_median
NativeMinuit.posterior_std
NativeMinuit.effective_sample_size
NativeMinuit.rhat
```

## Resampling (bootstrap & jackknife)

```@docs
NativeMinuit.bootstrap
NativeMinuit.jackknife
NativeMinuit.BootstrapResult
NativeMinuit.JackknifeResult
NativeMinuit.correlation
```

## Multi-modal solution detection

```@docs
NativeMinuit.find_solution_modes
NativeMinuit.SolutionMode
NativeMinuit.SolutionModes
```

## Escaping a local basin

```@docs
NativeMinuit.find_deeper_minimum
```

## Alternative minimizers (Optim.jl integration)

```@docs
NativeMinuit.optim
NativeMinuit.minimize_with
```

## Plotting & rich output

`plot(...)` recipes for the result types are provided via `RecipesBase`
(rendered via Plots.jl); the `draw_*` helpers below are Plots.jl-specific and
load through the Plots extension (`using Plots`). See the
[Plotting & rich output](guides/plotting.md) guide.

```@docs
NativeMinuit.to_latex
NativeMinuit.mn_plot_text
NativeMinuit.draw_contour
NativeMinuit.draw_mncontour
NativeMinuit.draw_profile
NativeMinuit.draw_mnprofile
NativeMinuit.draw_mnmatrix
```

## Common accessors

These small accessor functions are exported but documented as part of
the parent struct (see [`FunctionMinimum`](@ref), [`MinosError`](@ref), etc.):

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
| `has_limits(p)`   | any finite lower or upper limit (MinuitParameter) |
| `is_fixed(p)`     | fixed flag (MinuitParameter)                     |

## Index

```@index
```
