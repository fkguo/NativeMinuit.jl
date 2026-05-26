# IAM fit — JuMinuit benchmark notes

The original `README.md` in this directory has the full physics
context (it predates JuMinuit). This file adds the benchmark
perspective.

## Why this fit is a good JuMinuit stress test

- **~10–20 free parameters**: more than the toy benchmarks (≤10) yet
  small enough to iterate quickly during JuMinuit optimization.
- **Moderately expensive FCN**: each call runs `quadgk` integration
  over the unitarity cut for multiple channels → tens of μs per
  evaluation. Highlights JuMinuit's per-iteration overhead instead
  of getting drowned out by FCN cost.
- **Multi-channel coupling**: parameter correlations span ππ, KK̄,
  πη amplitudes simultaneously. Tests MNCONTOUR on a covariance
  with strong cross-block correlations.
- **IMinuit.jl-style call sites**: written against `Minuit(fcn,
  start; name=..., error=..., grad=...)` and `mncontour` / `mnprofile`
  — drop-in target for JuMinuit's compatibility layer.

## Running with JuMinuit

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path = "../..")            # JuMinuit repo root
using JuMinuit
# include the model files
include("src/init_const.jl")
include("src/amplitudes.jl")
include("src/tmatrix.jl")
include("src/phaseshifts.jl")
# … the notebook drives the fit
```

The notebook's `using IMinuit` should be swapped for `using
JuMinuit`. All call sites (`Minuit`, `migrad`, `minos`, `args`,
`matrix`, `mncontour`, …) should work unchanged. If anything breaks,
that's a JuMinuit drop-in-compat regression worth reporting.

## Original publication

> Y.-J. Shi, C.-Y. Seng, F.-K. Guo, B. Kubis, U.-G. Meißner, W. Wang,
> *Two-Meson Form Factors in Unitarized Chiral Perturbation Theory*,
> [arXiv:2011.00921](https://arxiv.org/abs/2011.00921).

The numerical results in that paper come from this fit setup.
