# API Reference

The public API is grouped by stage of a typical fit: declare the cost
function, run MIGRAD, refine with HESSE/MINOS, post-process, and visualize.
For internal helpers and non-exported names see [Internals](internals.md).

## Cost function

```@docs
JuMinuit.CostFunction
JuMinuit.CostFunctionWithGradient
JuMinuit.CostFunctionAD
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

## Covariance & diagnostics

```@docs
JuMinuit.covariance
JuMinuit.eigenvalues
JuMinuit.global_cc
JuMinuit.CovStatus
```

## MINOS (asymmetric errors)

```@docs
JuMinuit.minos
JuMinuit.MinosError
JuMinuit.minos_upper
JuMinuit.minos_lower
JuMinuit.function_cross
JuMinuit.MnCross
```

## Contours & profiles

```@docs
JuMinuit.mncontour
JuMinuit.contour_grid
JuMinuit.ContourGrid
JuMinuit.contour_ellipse
JuMinuit.contour_exact
JuMinuit.ContoursError
JuMinuit.profile
JuMinuit.mnprofile
```

## iminuit-style Minuit wrapper

```@docs
JuMinuit.Minuit
JuMinuit.AbstractFit
JuMinuit.Fit
JuMinuit.ArrayFit
JuMinuit.migrad!
JuMinuit.minos!
JuMinuit.hesse!
JuMinuit.HesseResult
JuMinuit.set_precision
```

## Gradients & threading

```@docs
JuMinuit.is_thread_safe
JuMinuit.ThreadSafetyError
```

## Other minimizers

```@docs
JuMinuit.simplex
JuMinuit.scan
```

## Per-parameter mutators

```@docs
JuMinuit.fix!
JuMinuit.release!
JuMinuit.set_value!
JuMinuit.set_error!
JuMinuit.set_limits!
JuMinuit.set_lower_limit!
JuMinuit.set_upper_limit!
JuMinuit.remove_limits!
```

## IMinuit.jl compatibility

```@docs
JuMinuit.Data
JuMinuit.chisq
JuMinuit.model_fit
JuMinuit.@model_fit
JuMinuit.args
JuMinuit.matrix
JuMinuit.chi2
JuMinuit.poisson_chi2
JuMinuit.multinominal_chi2
JuMinuit.func_argnames
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

## Derived quantities (Δχ²-region intervals & profile bands)

```@docs
JuMinuit.extremize
JuMinuit.profile_band
JuMinuit.ExtremizeResult
JuMinuit.ProfileBand
```

## Likelihood-ensemble MCMC (marginal quantile bands)

```@docs
JuMinuit.mcmc_sample
JuMinuit.LikelihoodEnsemble
JuMinuit.quantiles
JuMinuit.quantile_band
JuMinuit.save_ensemble
JuMinuit.load_ensemble
```

## Bayesian posterior bridge (priors & credible intervals)

Non-mutating Bayesian layer: `prior × likelihood` sampled in full external
coordinates, returning credible (not confidence) summaries. Three samplers —
`sampler = :metropolis` (random walk), `sampler = :stretch` (the gradient-free,
affine-invariant Goodman–Weare ensemble), and `sampler = :nuts` (gradient-based
NUTS, via the AdvancedHMC extension). With `prior = :flat` a single Metropolis
chain (`nchains = 1`) reproduces the likelihood path exactly at the same seed;
sampling never mutates the `Minuit` object or `m.nfcn`. See the
[Bayesian analysis guide](bayesian.md) for worked examples and how to enable NUTS.

```@docs
JuMinuit.bayesian
JuMinuit.BayesianReport
JuMinuit.posterior_sample
JuMinuit.PosteriorProblem
JuMinuit.PosteriorSample
JuMinuit.isconsistent
JuMinuit.Prior
JuMinuit.flat_prior
JuMinuit.normal_prior
JuMinuit.uniform_prior
JuMinuit.half_normal_prior
JuMinuit.combine_priors
JuMinuit.credible_interval
JuMinuit.derived_interval
JuMinuit.upper_limit
JuMinuit.lower_limit
JuMinuit.CredibleLimit
JuMinuit.posterior_summary
JuMinuit.posterior_mean
JuMinuit.posterior_median
JuMinuit.posterior_std
JuMinuit.effective_sample_size
JuMinuit.rhat
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

## Escaping a local basin

```@docs
JuMinuit.find_deeper_minimum
```

## Alternative minimizers (Optim.jl bridge)

```@docs
JuMinuit.optim
JuMinuit.minimize_with
```

## Plotting & rich output

`plot(...)` recipes for the result types are provided via `RecipesBase`
(rendered via Plots.jl); the `draw_*` helpers below are Plots.jl-specific and
load through the Plots extension (`using Plots`). See the
[Plotting & rich output](guides/plotting.md) guide.

```@docs
JuMinuit.to_latex
JuMinuit.mn_plot_text
JuMinuit.draw_contour
JuMinuit.draw_mncontour
JuMinuit.draw_profile
JuMinuit.draw_mnprofile
JuMinuit.draw_mnmatrix
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
