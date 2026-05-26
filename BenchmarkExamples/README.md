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

Fit of an effective coupled-channel amplitude to the e⁺e⁻ → J/ψπ⁺π⁻
data in the X(3872) mass region, demonstrating a dip structure near
the DD̄* threshold. The published version of the analysis underlying

> F.-K. Guo & ... (2024), preprint in preparation.

Files:
- `Xdip_published.ipynb` — clean version of the published notebook
  (outputs stripped; ~45 KB code+markdown only).
- `data.csv` — measured cross-sections + statistical errors.

The fit involves a non-trivial amplitude model evaluated at every FCN
call (multiple Riemann sheets, near-threshold expansion). Free
parameters: ~3–6 depending on which model variant. The covariance
matrix is highly correlated → ideal stress test for MNCONTOUR.

### `IAM_2Pformfactor/` — Inverse Amplitude Method on ππ / Kπ / πη / πK form factors

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

Free parameters: ~10–20 (low-energy constants + subtraction constants).
FCN is moderately expensive (per-call quadgk integration over the
unitarity cut). Realistic stress test for MIGRAD wall-time on a
medium-dimensional non-trivial χ²-fit.

---

## Running an example

These examples were originally written against IMinuit.jl (PyCall
wrapper). To migrate to JuMinuit:

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

The names / API match — see `docs/migration_from_iminuit.md` (TBD) for
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

Not yet implemented (Monte-Carlo-based error analysis from contour.jl):
- `get_contours`, `get_contours_all`, `contour_df`,
  `get_contours_given_parameter`, `contour_df_given_parameter`,
  `get_contours_samples`, `contour_df_samples`.
- These rely on multiple constrained re-fits + DataFrame aggregation.
  See the X3872 notebook for a hand-rolled version using
  `MvNormal` sampling + `Mahalanobis-distance` filtering, which is
  arguably more robust on non-quadratic posteriors.

## What these examples drive

- `benchmark/compare_all.jl` benchmarks the toy §3.3 FCNs (rosenbrock,
  quad_4d, gauss_ll, …). Those are stylized stress tests.
- **These examples drive the REAL-FCN benchmarks** — slower, dirtier,
  but representative. Use them to validate that a JuMinuit optimization
  (e.g., MnContours warm-start improvement) actually helps on a fit
  someone publishes.
