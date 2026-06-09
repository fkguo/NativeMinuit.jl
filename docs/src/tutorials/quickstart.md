# Quickstart

This tutorial gets you from an empty REPL to a fitted parameter with
asymmetric errors in five minutes. We use the iminuit / IMinuit.jl-style
[`Minuit`](@ref) object throughout: one mutable handle that bundles the
cost function, the parameters, and the fit result, with property access
(`m.values`, `m.errors`, `m.fval`, …) you can copy-paste from iminuit.

We assume Julia ≥ 1.11 and `JuMinuit` installed
(`Pkg.add("JuMinuit")` once registered, or `Pkg.add(url=...)` from the
repository).

## A first end-to-end fit

Define an FCN — any `f(x::AbstractVector) -> Real`. Here it is a simple
χ² with its minimum at `(1, 2, 3)`:

```julia
using JuMinuit

fcn(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2

m = Minuit(fcn, [0.0, 0.0, 0.0];
           names  = ["a", "b", "c"],   # parameter names (default "x0", "x1", …)
           errors = [0.1, 0.1, 0.1])   # initial step sizes

migrad!(m)                              # run MIGRAD; mutates m in place
```

Displaying `m` in a REPL or notebook prints the rich result table:

```
JuMinuit.Minuit  fval=0  edm=0  nfcn=36
[✓ Valid minimum] [✓ EDM below goal] [✓ Below call limit] [✓ Covariance accurate] [✓ No params at limit]
┌───┬──────┬─────────────┬─────────┬─────────┬───────┐
│ # │ Name │ Value       │ Limit − │ Limit + │ Fixed │
├───┼──────┼─────────────┼─────────┼─────────┼───────┤
│ 1 │ a    │ 1.00 ± 1.00 │ ─       │ ─       │       │
│ 2 │ b    │ 2.00 ± 1.00 │ ─       │ ─       │       │
│ 3 │ c    │ 3.00 ± 1.00 │ ─       │ ─       │       │
└───┴──────┴─────────────┴─────────┴─────────┴───────┘
```

## Reading the result

The fit result is exposed as iminuit-style properties on `m`:

```julia
m.values        # parameter values at the minimum (≈ [1, 2, 3])
m.errors        # symmetric 1σ Hesse errors, per parameter
m.fval          # FCN value at the minimum (≈ 0 here)
m.valid         # true if MIGRAD converged within tolerances
m.nfcn          # total FCN calls used
m.covariance    # external covariance matrix (nothing before migrad!)
```

`m.values` and `m.errors` read live from the fit and accept both an
integer index and a parameter name:

```julia
m.values[1]     # 1.0
m.values["b"]   # 2.0
m.errors["a"]   # the 1σ error on a
```

For a plain `Vector{Float64}` of the values use `args(m)` (≡
`collect(m.values)`).

## Asymmetric errors with MINOS

The symmetric Hesse error assumes a parabolic minimum. For a more honest,
generally **asymmetric** error, run [`minos!`](@ref) after `migrad!`:

```julia
minos!(m)               # MINOS on every free parameter
m.merrors               # Dict{String,MinosError}, keyed by parameter name

me = m.merrors["a"]
me.upper                # positive 1σ error (≥ 0)
me.lower                # negative 1σ error (≤ 0)
```

`minos!(m, "a")` (or `minos!(m, 1)`) does a single parameter. Once MINOS
has run, the rich table widens to show separate `Value`, `Hesse` and
`MINOS` columns side by side (a MINOS side that failed to converge is marked
`invalid`; `—` means MINOS was not run for that parameter). See
[MINOS errors & contours](minos_contours.md) for the full
[`MinosError`](@ref) field list and 2-D confidence contours.

!!! note "MINOS needs a covariance"
    MINOS derives its starting step from the inverse Hessian, so a fit
    that produced no covariance (e.g. [`simplex`](@ref) or [`scan`](@ref))
    must be refined with [`hesse!`](@ref)`(m)` first. A normal `migrad!`
    already leaves a covariance in place.

## A cost-function fit — `LeastSquares`

For a standard curve fit you don't have to spell out the χ² by hand. The
[`LeastSquares`](@ref) cost carries its own data, model, and error
definition, so `Minuit(cost, x0)` reads `up` and the data count off the
cost automatically:

```julia
using JuMinuit

model(x, p) = p[1] * x + p[2]            # a straight line y = a·x + b
xdata = [1.0, 2.0, 3.0, 4.0, 5.0]
ydata = [2.1, 3.9, 6.2, 7.8, 10.1]
σy    = fill(0.2, 5)

cost = LeastSquares(xdata, ydata, σy, model; name = [:a, :b])
m = Minuit(cost, [1.0, 0.0])             # up = 1 and ndata are taken from the cost
migrad!(m)

m.values        # ≈ [1.99, 0.05]  (slope, intercept)
m.fval          # χ² at the minimum
```

Because the cost knows it holds 5 data points, the rich table also shows
a `χ²/ndf` line and a fit p-value. The [Cost functions](../cost_functions.md)
guide covers the full family — `UnbinnedNLL`, `BinnedNLL`, their extended
variants, and joining several datasets into one fit with `CostSum` (`+`).

## Tolerances and budgets

`migrad!` mirrors iminuit's defaults (`Strategy(1)`, `tol = 0.1`); override
per call or store them on `m`:

```julia
migrad!(m;
    strategy = 2,        # 0 = fast, 1 = default, 2 = thorough
    tol      = 1e-3,     # EDM convergence target (× up × 0.002)
    maxfcn   = 10_000)   # call budget

m.strategy = 2           # or set once on the object; later fits reuse it
```

If `m.valid` is `false`, loosen `tol`, raise `maxfcn`, bump the strategy,
or re-seed from a better starting point. `m.accurate` separately reports
whether the covariance is trustworthy (it is `false` when MIGRAD had to
force the Hessian positive-definite).

## Where to go next

- [Bounded parameters](bounded.md) — add limits with `limits =` and fix
  parameters, via the constructor or per-parameter [`set_limits!`](@ref) /
  [`fix!`](@ref).
- [MINOS errors & contours](minos_contours.md) — asymmetric errors,
  `mncontour`, and profile likelihoods.
- [Cost functions](../cost_functions.md) — the Julia-native cost family.
- [Error analysis](../error_analysis.md) — which uncertainty method to
  use, and when.
