# X(3872) dip-structure fit — NativeMinuit benchmark example

Originally written for IMinuit.jl. Adapted here as a NativeMinuit
benchmark + stress test.

## Run online

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FX3872_dip%2FXdip_published.ipynb)

`Xdip_published.ipynb` runs interactively on [Binder](https://mybinder.org)
— free, no login. The first launch builds the image (several minutes); later
launches are cached. A transient **"GitHub API rate limit"** page is a
mybinder.org-wide limit (shared token, resets hourly) — just retry later.

## Files

| File | Purpose |
|---|---|
| `Xdip_published.ipynb` | Published fit notebook (outputs stripped) |
| `data.csv` | Cross-section measurements vs invariant mass |

## Physics context

Fit of the e⁺e⁻ → J/ψπ⁺π⁻ line-shape near the X(3872) mass with a
two-channel effective amplitude (J/ψρ + DD̄*; the DD̄* effective
scattering length `a22eff` is fixed from the published analysis). The
model exhibits a *dip* in the cross-section near the DD̄*⁰ threshold
for specific parameter regions, which serves as the discriminator.

3 free parameters in the shipped benchmark (`model1`); the published notebook explores additional variants:
- Effective coupling constants
- Mass parameter (loosely centered at threshold)
- Decay width parameter
- Normalization

## Fit characteristics relevant to NativeMinuit

| Property | Value |
|---|---|
| n_free | 3 (`model1`, the shipped benchmark) |
| Data points | 4 (`data.csv`) |
| FCN cost | ~38 µs/call (Gauss-convolved complex amplitude, 4 points) |
| Covariance | strongly correlated (degeneracy along coupling combos) |
| Posterior | non-Gaussian (banana-shape in some 2D projections) |

→ Useful for testing MNCONTOUR accuracy + MINOS asymmetry on a
real non-quadratic posterior. The "X3872 cusp + Flatté regime"
is exactly where naive cov-MC error bands fail (see paper text).

## How to reproduce

Original env was IMinuit.jl + Plots + DataFrames. To run against
NativeMinuit:

```julia
using Pkg
Pkg.activate(".")  # this directory
Pkg.add(["DataFrames", "CSV", "Plots", "Distributions",
         "ForwardDiff", "LinearAlgebra"])
Pkg.develop(path = "../..")   # NativeMinuit.jl repo root
using NativeMinuit, DataFrames, CSV, Distributions
# … see notebook for the rest
```

The notebook already uses `using NativeMinuit` (migrated from `using IMinuit`);
the rest of the API (`Minuit`, `migrad`, `minos`, `args`, `matrix`, `Data`,
`chisq`, `model_fit`, `@model_fit`, `mncontour`, …) is drop-in compatible
with IMinuit.jl. The online launch above uses the ready-made `.binder/`
environment, so no manual setup is needed there.

## Citing

If you reuse this fit setup or data in a publication, please cite the
original analysis and acknowledge NativeMinuit.jl:

> V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
> "How does the X(3872) show up in e⁺e⁻ collisions: Dip versus peak",
> *Phys. Rev. D* **109** (2024) 11, L111501,
> [arXiv:2404.12003](https://arxiv.org/abs/2404.12003),
> [INSPIRE 2778938](https://inspirehep.net/literature/2778938).
