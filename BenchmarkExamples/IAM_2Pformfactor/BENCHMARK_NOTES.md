# IAM fit — NativeMinuit benchmark notes

The original `README.md` in this directory has the full physics
context (it predates NativeMinuit). This file adds the benchmark
perspective.

## Why this fit is a good NativeMinuit stress test

- **7 free parameters (L1–L5, L7, L8; L6 fixed)**: comparable to the larger toy
  benchmarks, but on a genuinely stiff physics landscape rather than a
  synthetic quadratic — exercises NativeMinuit on a realistic, ill-conditioned
  Hessian.
- **Expensive FCN (~9 ms/call)**: each call runs `quadgk` integration over
  the unitarity cut for multiple channels. At ~9 ms the FCN dominates the
  fit wall-time (as in most real fits), so this stresses NativeMinuit's
  convergence quality and iteration count on an ill-conditioned landscape
  rather than raw per-iteration speed.
- **Multi-wave coupling**: the fitted χ² combines three ππ partial waves
  (I=0 S, I=1 P, I=2 S) through a shared set of LECs. Tests MNCONTOUR on a
  covariance with strong cross-parameter correlations.
- **IMinuit.jl-style call sites**: written against `Minuit(fcn,
  start; name=..., error=..., grad=...)` and `mncontour` / `mnprofile`
  — drop-in target for NativeMinuit's compatibility layer.

## Running with NativeMinuit

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path = "../..")            # NativeMinuit repo root
using NativeMinuit
# include the model files
include("src/init_const.jl")
include("src/amplitudes.jl")
include("src/tmatrix.jl")
include("src/phaseshifts.jl")
# … the notebook drives the fit
```

The notebook's `using IMinuit` should be swapped for `using
NativeMinuit`. All call sites (`Minuit`, `migrad`, `minos`, `args`,
`matrix`, `mncontour`, …) should work unchanged. If anything breaks,
that's a NativeMinuit drop-in-compat regression worth reporting.

## Original publication

> Y.-J. Shi, C.-Y. Seng, F.-K. Guo, B. Kubis, U.-G. Meißner, W. Wang,
> *Two-Meson Form Factors in Unitarized Chiral Perturbation Theory*,
> [arXiv:2011.00921](https://arxiv.org/abs/2011.00921).

The numerical results in that paper come from this fit setup.
