# Migrating from iminuit / IMinuit.jl

NativeMinuit is a native-Julia port of Minuit2 and a **drop-in replacement for
[IMinuit.jl](https://github.com/fkguo/IMinuit.jl)**, with an API that stays
close to the Python [iminuit](https://github.com/scikit-hep/iminuit) where that
makes sense and leans on Julia strengths (generic FCNs, multiple dispatch,
package extensions) where that is better.

The most important difference in one sentence: iminuit's *callable methods*
`m.migrad()`, `m.hesse()`, `m.minos()` become **bang-mutating functions**
[`migrad!`](@ref)`(m)`, [`hesse!`](@ref)`(m)`, [`minos!`](@ref)`(m)`, while the
*data accessors* (`m.values`, `m.errors`, `m.covariance`, …) keep the same
property syntax. Existing IMinuit.jl scripts that already use functional forms
like `migrad(m)` / `chisq` / `Data` / `Fit` largely run unchanged.

## Mapping table

| iminuit / IMinuit.jl | NativeMinuit |
|:---|:---|
| `Minuit(fcn, x0; ...)` | same — `Minuit(fcn, x0; names, errors, limits, ...)` |
| `m.migrad()` / `migrad(m)` | [`migrad!`](@ref)`(m)` (or `migrad(m)`) |
| `m.hesse()` | [`hesse!`](@ref)`(m)` |
| `m.minos()` | [`minos!`](@ref)`(m)` |
| `m.values`, `m.errors`, `m.covariance` | same (property access) |
| `m.merrors` | same — `Dict` of MINOS errors keyed by name (also `m.minos_errors`, keyed by index) |
| `m.mncontour(a, b)` | [`mncontour`](@ref)`(m, a, b)` |
| `m.contour(a, b)` (IMinuit.jl `contour(m, a, b)`) | [`contour_grid`](@ref)`(m, a, b)` (renamed: avoids the `Plots.contour` clash) |
| `m.profile(a)` / `m.mnprofile(a)` | [`profile`](@ref)`(m, a)` / [`mnprofile`](@ref)`(m, a)` |
| IMinuit.jl `Fit`, `ArrayFit` | exported aliases of [`Minuit`](@ref) (see [`AbstractFit`](@ref)) |
| IMinuit.jl `chisq`, `Data` | exported, same signatures |
| IMinuit.jl `args(m)`, `matrix(m)` | same |
| IMinuit.jl `reset(m)`, `set_precision(m, p)` | same |
| `m.scipy(method=...)` | [`optim`](@ref)`(m; method=...)` (needs `using Optim`) |

## Constructor: unchanged

The `Minuit` constructor has the same shape as iminuit / IMinuit.jl. The
iminuit-style singular keywords (`name`, `error`) and the plural forms
(`names`, `errors`) are both accepted, and `up` / `errordef` are aliases:

```julia
using NativeMinuit

m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
           [0.0, 0.0];
           names  = ["a", "b"],
           errors = [0.1, 0.1],
           limits = [(-5.0, 5.0), nothing])
```

Per-parameter keywords work too (`error_a = 0.2`, `fix_b = true`, …), exactly
as in IMinuit.jl.

## Running the fit: `m.migrad()` → `migrad!(m)`

This is the one line you almost always have to change. In iminuit the
minimizers are methods on the object; in NativeMinuit they are mutating functions
(trailing `!`) that update `m` in place and return it, so they chain with `|>`:

```julia
# iminuit (Python) / IMinuit.jl method style
# m.migrad(); m.hesse(); m.minos()

# NativeMinuit
migrad!(m)
hesse!(m)
minos!(m)

# or, chained:
m |> migrad! |> hesse! |> minos!
```

A non-bang `migrad(m)` is also exported (it forwards to `migrad!`), so
IMinuit.jl code calling `migrad(m)` keeps working. `hesse!` is an alias of
`hesse`, and `minos(m, …)` forwards to `minos!`, so either spelling is fine —
the bang form is the idiomatic Julia choice because these methods mutate `m`.

## Reading results: property access stays the same

The result accessors are unchanged — they are properties on `m`:

```julia
m.values         # parameter values at the minimum
m.errors         # parabolic (HESSE) 1σ errors
m.covariance     # covariance matrix (nothing before MIGRAD)
m.fval           # FCN value at the minimum
m.merrors        # Dict{String,MinosError} of MINOS errors, keyed by name
m.valid          # did MIGRAD converge?
m              # rich table (HTML in Jupyter / Pluto, text in the REPL)
```

`m.merrors` mirrors iminuit's MINOS-errors dictionary (keyed by parameter
name). The index-keyed `m.minos_errors` (`Dict{Int,MinosError}`) is also
available. Other iminuit aliases are present too: `m.is_valid`, `m.ncalls`,
`m.parameters`, `m.accurate`, `m.npar`, …

IMinuit.jl's small functional accessors are exported unchanged:
[`args`](@ref)`(m)` (a `Vector{Float64}` of the current values, `≡ m.values`)
and [`matrix`](@ref)`(m; correlation=false)` (the covariance or correlation
matrix). `reset(m)` drops cached MIGRAD/MINOS results and
[`set_precision`](@ref)`(m, p)` overrides the machine precision.

## Contours and profiles: `m.mncontour(a,b)` → `mncontour(m, a, b)`

The contour / profile helpers become functions taking `m` as the first
argument:

```julia
# iminuit:  pts = m.mncontour("a", "b")
pts = mncontour(m, "a", "b")          # joint 68 % region (iminuit cl semantics)

# iminuit:  x, y, F = m.contour("a", "b")   (IMinuit.jl: contour(m, "a", "b"))
xs, ys, F = contour_grid(m, "a", "b") # FCN grid slice, others held fixed

prof  = profile(m, "a")               # 1D scan, no inner minimization
mprof = mnprofile(m, "a")             # MINOS profile (re-minimizes nuisances)
```

Parameters may be passed by 1-based integer index or by name (`String`).

!!! note "`contour` renames (0.5.0)"
    iminuit's / IMinuit.jl's grid-scan `contour` is [`contour_grid`](@ref)
    in NativeMinuit — the bare name `contour` would clash with `Plots.contour`
    under `using NativeMinuit, Plots`. NativeMinuit ≤ 0.4's own `contour` (a fast
    error-ellipse approximation, *not* iminuit's grid) is now
    [`contour_ellipse`](@ref); the unexported deprecated alias
    `NativeMinuit.contour` still forwards to it.

## `chisq` / `Data`: drop-in, same signatures

IMinuit.jl's least-squares helpers are exported with identical signatures and
no PyCall / matplotlib dependency:

```julia
using NativeMinuit

model(x, p) = p[1] * x + p[2]
data = Data(xs, ys, σs)               # holds x, y, err
m = model_fit(model, data, [1.0, 0.0])  # wraps chisq + Minuit
migrad!(m)

# equivalently, the raw cost:
χ² = chisq(model, data, m.values)     # or chisq(model, (xs, ys, σs), par)
```

Both [`model_fit`](@ref) and the `@model_fit` macro build a [`Minuit`](@ref)
from a model + `Data` + starting values, flowing any `Minuit` keywords through.
The `iminuit.cost` kernels `chi2`, `poisson_chi2`, and `multinominal_chi2` are
also provided as pure-Julia functions with matching signatures, as is
`func_argnames` for reflecting an FCN's argument names.

!!! note "`Data` vs. the cost-function objects"
    `chisq` / `Data` remain the quickest way to port IMinuit.jl code. For new
    work you can also use NativeMinuit's Julia-native cost objects
    ([`LeastSquares`](@ref), [`UnbinnedNLL`](@ref), …), which carry their own
    `errordef` and compose with `+` ([`CostSum`](@ref)). A `LeastSquares` fit
    and the matching `chisq` `model_fit` share one χ² kernel, so they give
    bit-identical results — see the [cost-functions guide](../cost_functions.md).

## `Fit` / `ArrayFit`: aliases of `Minuit`

IMinuit.jl exposes two concrete fit types, `Fit` (scalar-argument `fcn(a, b)`
construction) and `ArrayFit` (vector `fcn(par)` construction). That split was a
PyCall wrapping artifact. NativeMinuit always calls the FCN as `f(::AbstractVector)`
internally, so the two forms have no behavioural difference after construction;
[`Fit`](@ref) and [`ArrayFit`](@ref) are therefore exported as **aliases of
[`Minuit`](@ref)**. Code annotating `f::Fit` / `f::ArrayFit` / `f::AbstractFit`
or testing `f isa Fit` keeps working unchanged.

## `m.scipy(...)` → `optim(m; ...)`

iminuit's `m.scipy(method=...)` escapes to `scipy.optimize.minimize` when
MIGRAD struggles. The Julia-native analogue bridges to
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) instead of shelling out
to Python. It is a package extension, so load `using Optim` to enable it:

```julia
using NativeMinuit, Optim

m = Minuit(fcn, x0)
optim(m; method = :lbfgs)   # writes the optimum back into m
hesse!(m)                   # then get the covariance, à la iminuit's scipy-then-hesse
```

[`optim`](@ref) accepts iminuit-style method names (`:lbfgs` / `"L-BFGS-B"`,
`:bfgs`, `:neldermead`, `:newton`, …) and honours fixed parameters and box
limits. [`minimize_with`](@ref)`(m, LBFGS())` is a clearer-named alias that also
accepts an Optim optimizer object directly. Without `using Optim`, calling
either gives a clear "load Optim" message rather than a `MethodError`.

## What stays the same vs. what is idiomatically different

**Same:**

- the `Minuit(fcn, x0; names, errors, limits, …)` constructor shape;
- result accessors as properties (`m.values`, `m.errors`, `m.covariance`,
  `m.merrors`, `m.fval`, `m.valid`, …) and their iminuit aliases;
- `chisq`, `Data`, `model_fit`, `args`, `matrix`, `reset`, `set_precision` —
  exported with the same signatures;
- the MINOS / contour semantics (`mncontour` traces the exact MnContours
  boundary, not the HESSE ellipse, with iminuit ≥ 2.0's joint-coverage `cl`
  — default joint 2-D 68 %, `Δχ² ≈ 2.28`).

**Idiomatically different:**

- **bang-mutating minimizers** — `migrad!`, `hesse!`, `minos!` replace the
  `m.migrad()` / `m.hesse()` / `m.minos()` methods (and chain with `|>`);
- **algorithm helpers take `m` as an argument** — `mncontour(m, a, b)`,
  `profile(m, a)`, `mnprofile(m, a)` instead of `m.mncontour(a, b)` etc.;
- **generic Julia FCNs** — your `f(x::AbstractVector)` can be any callable;
  pass `grad = x -> ForwardDiff.gradient(f, x)` (or `using ForwardDiff`) for
  AD-backed gradients;
- **package extensions** — alternative minimizers (`using Optim`) and plotting
  (`using Plots`) load on demand rather than as hard dependencies.

For a fit from scratch, see the [Quickstart](../tutorials/quickstart.md); for the
full list of exported names, see the [API reference](../api.md).
