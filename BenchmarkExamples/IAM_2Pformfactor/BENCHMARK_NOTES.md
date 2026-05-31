# IAM fit — JuMinuit benchmark notes

The original `README.md` in this directory has the full physics
context (it predates JuMinuit). This file adds the benchmark
perspective.

## Why this fit is a good JuMinuit stress test

- **9 free parameters (8 NLO LECs + 1)**: comparable to the larger toy
  benchmarks, but on a genuinely stiff physics landscape rather than a
  synthetic quadratic — exercises JuMinuit on a realistic, ill-conditioned
  Hessian.
- **Expensive FCN (~9 ms/call)**: each call runs `quadgk` integration over
  the unitarity cut for multiple channels. At ~9 ms the FCN dominates the
  fit wall-time (as in most real fits), so this stresses JuMinuit's
  convergence quality and iteration count on an ill-conditioned landscape
  rather than raw per-iteration speed.
- **Multi-wave coupling**: the fitted χ² combines three ππ partial waves
  (I=0 S, I=1 P, I=2 S) through a shared set of LECs. Tests MNCONTOUR on a
  covariance with strong cross-parameter correlations.
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
