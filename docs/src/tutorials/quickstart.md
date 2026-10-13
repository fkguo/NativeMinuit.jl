# Quickstart

This tutorial gets you from an empty REPL to a fitted parameter with
asymmetric errors in five minutes. We use the iminuit / IMinuit.jl-style
[`Minuit`](@ref) object throughout: one mutable handle that bundles the
cost function, the parameters, and the fit result, with property access
(`m.values`, `m.errors`, `m.fval`, …) you can copy-paste from iminuit.

We assume Julia ≥ 1.11 and `NativeMinuit` installed with
`Pkg.add("NativeMinuit")`.

## A first end-to-end fit

Define a scalar objective `f(x::AbstractVector) -> Real`. The solver works in
`Float64`; a generic objective may also accept ForwardDiff dual numbers for AD. Here it is a simple
χ² with its minimum at `(1, 2, 3)`:

```jldoctest quickstart
julia> using NativeMinuit

julia> fcn(x) = sum(abs2, x .- [1.0, 2.0, 3.0]);

julia> m = Minuit(fcn, zeros(3); names=["a", "b", "c"], errors=fill(0.1, 3)); migrad!(m);

julia> m.valid && m.fval < 1e-10 && isapprox(collect(m.values), [1.0, 2.0, 3.0]; atol=1e-4, rtol=0)
true
```

Displaying `m` prints its values, covariance diagnostics and objective-call count:

```@example quickstart_table
using NativeMinuit
m = Minuit(x -> sum(abs2, x .- [1.0, 2.0, 3.0]), zeros(3);
           names=["a", "b", "c"], errors=fill(0.1, 3))
migrad!(m)
@assert m.valid && m.fval < 1e-10 # hide
@assert isapprox(collect(m.values), [1.0, 2.0, 3.0]; atol=1e-4, rtol=0) # hide
m
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

Use `args(m)` or `copy(m.values)` for a copy of the external values.
`args(m)` preserves a supported named container; `collect(m.values)` returns
a plain vector. The live `m.values` view belongs to the mutable fit.

## Asymmetric errors with MINOS

The symmetric Hesse error assumes a parabolic minimum. For a more honest,
generally **asymmetric** error, run [`minos!`](@ref) after `migrad!`:

```julia
minos!(m)               # MINOS on every free parameter
m.merrors               # Dict{String,MinosError}, keyed by parameter name

me = m.merrors["a"]
has_closed_interval(me)  # both valid crossings, with neither side at a limit
me.upper                # signed crossing offset, bound distance or invalid placeholder
me.lower                # inspect the flags before interpreting either value
```

`minos!(m, "a")` (or `minos!(m, 1)`) does a single parameter. Once MINOS
has run, the rich table widens to show separate `Value`, `Hesse` and
`MINOS` columns side by side (a MINOS side that failed to converge is marked
`invalid`; a bound-truncated side is labelled `at limit`; `—` means MINOS
was not run for that parameter). See
[MINOS errors & contours](minos_contours.md) for the full
[`MinosError`](@ref) field list and 2-D confidence contours.

> **Note — MINOS needs a covariance**
>
> MINOS derives its starting step from the inverse Hessian, so a fit
> that produced no covariance (e.g. [`simplex`](@ref) or [`scan`](@ref))
> must be refined with [`hesse!`](@ref)`(m)` first. A normal `migrad!`
> already leaves a covariance in place.

## A cost-function fit — `LeastSquares`

For a standard curve fit you don't have to spell out the χ² by hand. The
[`LeastSquares`](@ref) cost carries its own data, model, and error
definition, so `Minuit(cost, x0)` reads `up` and the data count off the
cost automatically:

```julia
using NativeMinuit

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

`tol` controls the internal EDM target:

```math
\mathrm{EDM}_{\mathrm{target}} = 0.002\max(\mathrm{tol}\,\mathrm{up},\mathrm{eps2}).
```

Here `eps2` is Minuit’s machine-precision floor. The final validity decision
also checks the call-limit, Hessian and finite-value status; a small EDM alone
is insufficient. The low-level result’s printed goal is a display convention;
use the requested tolerance and status flags when assessing termination.

`maxfcn` is a budget for an individual low-level minimization, not a hard cap
on all calls made by `migrad!`. Seed/gradient/HESSE probes can finish beyond
that threshold, and retries can run Simplex and MIGRAD with separate budgets.
`m.nfcn` counts calls across the attempted passes; `m.n_passes` counts MIGRAD
passes. MINOS and subsequent HESSE calls also add to the current fit’s count.

By default `migrad!` permits up to five MIGRAD passes. After an invalid pass
without a call-limit exit, it can run Simplex followed by `Strategy(2)` MIGRAD.
Use `iterate=1` for one pass, or `use_simplex=false` for retries retaining the
previous inverse Hessian. `multistart=true` adds perturbed Simplex starts.
Neither retries nor multistart establish a global minimum. The selected
finite result prefers valid over invalid candidates, then the lower objective
within the same validity class; a valid result may have a higher objective
than an invalid candidate. A repeated basin can terminate the loop early.

A later `migrad!(m)` resumes from the stored fit. `reset(m)` drops cached results
and returns to the stored parameter configuration. User parameter edits update
that configuration, so reset does not undo those edits.

```jldoctest
julia> using NativeMinuit

julia> m = Minuit(x -> sum(abs2, x .- [1.0, 2.0]), zeros(2)); migrad!(m; iterate=1);

julia> m.valid && m.n_passes == 1
true

julia> migrad!(m; iterate=1); reset(m);

julia> collect(m.values) == zeros(2) && m.fmin === nothing
true
```

For an invalid fit, inspect its termination reason and the objective before
choosing a better start, additional calls or a different strategy. Loosening
`tol` changes the stopping condition; it does not validate the uncertainty.
`m.accurate` requires a valid fit without a forced-positive-definite covariance.
It is a diagnostic flag, not a test of the statistical model or global optimality.
Symmetric errors use a local quadratic approximation; MINOS uses profile crossings.

## Where to go next

- [Bounded parameters](bounded.md) — add limits with `limits =` and fix
  parameters, via the constructor or per-parameter [`set_limits!`](@ref) /
  [`fix!`](@ref).
- [MINOS errors & contours](minos_contours.md) — asymmetric errors,
  `mncontour`, and profile likelihoods.
- [Cost functions](../cost_functions.md) — the Julia-native cost family.
- [Error analysis](../error_analysis.md) — which uncertainty method to
  use, and when.
