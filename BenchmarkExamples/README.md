# BenchmarkExamples

Real-world fit examples for stress-testing JuMinuit.jl. **These are NOT part
of the package.** They live outside `src/`, `test/`, `ext/`, so `Pkg.test()`
never picks them up. They exist to:

1. Profile MIGRAD / HESSE / MINOS / MNCONTOUR on fits actually used in
   physics publications (where the FCN is non-trivial and the parameter
   covariance is non-diagonal).
2. Provide a public-data baseline so anyone can reproduce JuMinuit-vs-
   C++Minuit2-vs-iminuit comparisons on the same problems.
3. Serve as templates: copy-paste a fit, adapt the model, drop in your
   own data.

Each subdirectory has its own `README.md` with the physics context, the
data origin, and a runbook (typically `julia --project=. notebook.ipynb`
or `julia --project=. main.jl`).

---

## Cases

### `X3872_dip/` — X(3872) line-shape with a dip near threshold

▶ **Run online (no install):** [![Binder](https://img.shields.io/badge/launch-GESIS%20Binder-579ACA?logo=jupyter&logoColor=white)](https://notebooks.gesis.org/binder/v2/gh/fkguo/JuMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FX3872_dip%2FXdip_published.ipynb)

Fit of an effective coupled-channel amplitude to the e⁺e⁻ → J/ψπ⁺π⁻
data in the X(3872) mass region, demonstrating a dip structure near
the DD̄* threshold. The published version of the analysis underlying

> V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
> "How does the X(3872) show up in e⁺e⁻ collisions: Dip versus peak",
> *Phys. Rev. D* **109** (2024) 11, L111501,
> [arXiv:2404.12003](https://arxiv.org/abs/2404.12003),
> [INSPIRE 2778938](https://inspirehep.net/literature/2778938).

Files:
- `Xdip_published.ipynb` — clean version of the published notebook
  (outputs stripped; ~45 KB code+markdown only).
- `data.csv` — measured cross-sections + statistical errors.

The fit involves a non-trivial amplitude model evaluated at every FCN
call (multiple Riemann sheets, near-threshold expansion). Free
parameters: ~3–6 depending on which model variant. The covariance
matrix is highly correlated → ideal stress test for MNCONTOUR.

### `IAM_2Pformfactor/` — Inverse Amplitude Method on ππ / Kπ / πη / πK form factors

▶ **Run online (no install):** [![Binder](https://img.shields.io/badge/launch-GESIS%20Binder-579ACA?logo=jupyter&logoColor=white)](https://notebooks.gesis.org/binder/v2/gh/fkguo/JuMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FIAM_2Pformfactor%2Fiamfit.ipynb)

Multi-channel χPT-IAM fit of meson-meson scattering phase shifts +
form factors, used in

> F.-K. Guo et al., *Inverse-amplitude method study of two-pseudoscalar
> form factors*, 2020 (data from refs in the `datajl/` directory).

Files:
- `iamfit.ipynb` — fit driver (outputs stripped).
- `src/` — IAM amplitude code (`*.jl` modules).
- `datajl/` — data files for ππ, Kπ, πη, πK, KK̄ scattering and
  form-factor measurements (with source paper refs in filenames).
- `Project.toml` — original env (will need updating for current Julia).

Free parameters: 7 (paper-faithful: NLO LECs L1–L5, L7, L8; L6 fixed — see RESULTS.md).
FCN is moderately expensive (per-call quadgk integration over the
unitarity cut). Realistic stress test for MIGRAD wall-time on a
medium-dimensional non-trivial χ²-fit.

---

## Running an example

These examples were originally written against IMinuit.jl (PyCall
wrapper) and have been **migrated to JuMinuit** — the notebooks now use
`using JuMinuit` and run online via the Binder badges above. The migration
is a one-line swap because the API is drop-in compatible:

```julia
# Old
using IMinuit
fit = Minuit(my_chisq, [1.0, 2.0]; name = ["a", "b"], error = [0.1, 0.1])
migrad(fit)

# New (drop-in compatible)
using JuMinuit
fit = Minuit(my_chisq, [1.0, 2.0]; name = ["a", "b"], error = [0.1, 0.1])
migrad(fit)
```

The names / API match — see [`docs/src/guides/migration.md`](../docs/src/guides/migration.md) for
the full mapping. Functions / macros from IMinuit.jl that JuMinuit
implements natively:

- `Data`, `chisq`, `model_fit`, `@model_fit`, `func_argnames`
- `chi2`, `poisson_chi2`, `multinominal_chi2`
- `@plt_data`, `@plt_data!`, `@plt_best`, `@plt_best!`
- `args(m)`, `matrix(m)`, `reset(m)`, `set_precision(m, p)`
- `m.values`, `m.errors`, `m.fixed`, `m.limits`, `m.merrors`,
  `m.covariance`, `m.fval`, `m.is_valid`, `m.parameters`, …
- `simplex`, `scan`, `mncontour`, `profile`, `mnprofile`
- `eigenvalues(m)`, `global_cc(m)`

Error analysis — a from-scratch design, **implemented** (see the
[error-analysis guide](../docs/src/error_analysis.md)):
- **MC-Δχ² confidence regions** — `get_contours_samples` / `contour_df_samples`:
  sample the **true** `Δχ² ≤ delta_chisq(cl, ndof)` region (the exact χ²
  re-evaluated at every sample; over-coverage-aware via an inflation factor,
  adaptive widening, and a covariance-free box proposal). This supersedes the
  X3872 notebook's hand-rolled `MvNormal` + `Mahalanobis` sketch — Mahalanobis
  is a *diagnostic*, not the acceptance cut (cutting on it merely reproduces the
  HESSE ellipse).
- **Bootstrap / jackknife** — `bootstrap`, `jackknife`: data-resampling errors
  that don't trust the quoted `σ` (with full covariance + `correlation`).
- **Multi-modal solution detection** — `find_solution_modes`: cluster the
  accepted samples into statistically distinct solutions.
- **Full contour parameter sets** — `contour_parameter_sets`: the native
  analogue of IMinuit.jl's `get_contours`.

## What these examples drive

- `benchmark/compare_all.jl` benchmarks the toy §3.3 FCNs (rosenbrock,
  quad_4d, gauss_ll, …). Those are stylized stress tests.
- **These examples drive the REAL-FCN benchmarks** — slower, dirtier,
  but representative. Use them to validate that a JuMinuit optimization
  (e.g., MnContours warm-start improvement) actually helps on a fit
  someone publishes.
