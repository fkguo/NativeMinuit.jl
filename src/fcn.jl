# SPDX-License-Identifier: LGPL-2.1-or-later

"""
    AbstractCostFunction

Supertype for every JuMinuit FCN wrapper. Two concrete subtypes ship:

- [`CostFunction`](@ref) — numerical-gradient path (central-difference
  via `numerical_gradient!`). The default.
- `CostFunctionWithGradient` — analytical-gradient path; the user (or
  a Package extension like ForwardDiff integration) provides
  `g(x) → Vector{Float64}`. The MIGRAD loop's `numerical_gradient!`
  dispatch is shadowed by `analytical_gradient!` for this subtype.

Phase F: the cross-search / MINOS / contour drivers all take
`cf::AbstractCostFunction` (not a concrete `CostFunction`) so the AD
gradient path propagates through the inner-MIGRAD chain. The
`_fix_one_param` / `_fix_multi_params` helpers have overloads for
both subtypes that splice either just `f` (CostFunction) or both
`f` and `g` (CostFunctionWithGradient).
"""
abstract type AbstractCostFunction end

"""
    CostFunction{F,T} <: AbstractCostFunction

Wraps a user FCN `f::F` with an error definition `up::T` and an internal
call counter. **Closure-specialized** via parametric `F` (ROADMAP §2.3
+ Risk #4) so the call site `cf(x)` devirtualizes through the concrete
type rather than hitting Julia's `::Function` vtable.

Mirrors `MnFcn` and `MnUserFcn` from C++ Minuit2, collapsed since Julia
doesn't need the C++ inheritance hierarchy (the call counter is in the
same struct as the user function; multiple dispatch handles
gradient-vs-no-gradient via a separate `CostFunctionWithGradient` type).

For an unbounded fit the call to `f(x)` passes the parameter vector
unchanged; for a bounded fit the internal→external sin/√ transform is
applied on the same call boundary (see `transform.jl`).

# Fields

- `f::F` — the user function. Must accept an `AbstractVector{Float64}`
  (or compatible) and return a `Float64`-convertible cost value.
- `up::T` — error definition (`1.0` for χ² fits, `0.5` for negative
  log-likelihood fits; default `1.0`). Parametric `T` keeps the door
  open for `ForwardDiff.Dual{...,Float64}` users in Phase 2.1 without
  re-shuffling the type.
- `nfcn::Base.RefValue{Int}` — call counter; mutates on each
  invocation via the call-operator overload. `Ref` is the idiomatic
  Julia way to hold mutable state inside an otherwise-immutable struct.
- `n_nonfinite::Base.RefValue{Int}` — count of calls whose return value
  was non-finite (`NaN`/`±Inf`). Mirrors iminuit's `FCN::check_value`
  NaN detection (iminuit `src/fcn.cpp`), except iminuit warns per
  occurrence (via MnPrint, suppressed at default print level) while
  JuMinuit aggregates here and the MIGRAD drivers warn ONCE at the end
  of a run that did not end valid (handoff F7/P6; valid fits stay
  silent). Read via [`nonfinite_calls`](@ref).

# Performance notes

- This struct is **not** `isbits` (because of the `Ref`). One heap
  allocation per `CostFunction` instance. That's fine — we construct
  one per `migrad` call.
- The `f(x)::Float64` annotation enforces the return type contract at
  the call boundary; if the user's FCN returns a non-`Float64`, you'll
  see a clear runtime error at first call, not silent type instability
  in the MIGRAD inner loop.

# Aliasing contract

The argument vector `x` passed to your FCN is a **borrowed reference**
to JuMinuit's internal workspace. The same `Vector{Float64}` instance
is reused across line-search and gradient-calculation calls within a
single MIGRAD iteration. Your FCN MUST NOT:

- **Retain** the vector (e.g. push into a captured array). It will be
  overwritten by the next call.
- **Mutate** the vector. Treat it as read-only.

If you need the values later, copy them: `xcopy = copy(x)`.

# Examples

```julia
julia> cf = CostFunction(x -> sum(abs2, x), 1.0);

julia> cf([1.0, 2.0, 3.0])
14.0

julia> ncalls(cf)
1

julia> reset_ncalls!(cf); ncalls(cf)
0
```
"""
struct CostFunction{F,T} <: AbstractCostFunction
    f::F
    up::T
    nfcn::Base.RefValue{Int}
    n_nonfinite::Base.RefValue{Int}
end

CostFunction(f, up = 1.0) = CostFunction(f, up, Ref(0), Ref(0))

"""
    (cf::CostFunction)(x::AbstractVector) -> Float64

Evaluate the user function at `x`, incrementing the call counter.
Returns `Float64`. Numeric returns (e.g. `Int`) are coerced via the
`Float64` constructor — isbits→isbits, zero allocation. Non-numeric
returns trigger a `MethodError`.

A non-finite return (`NaN`/`±Inf`) additionally increments the
`n_nonfinite` counter — the value itself is passed through UNCHANGED
(iminuit/Minuit2 parity: NaN propagates into the minimizer, whose
`<`-comparisons reject it against any finite incumbent; see
`_migrad_loop`'s non-finite handling).
"""
@inline function (cf::CostFunction)(x::AbstractVector)
    cf.nfcn[] += 1
    v = Float64(cf.f(x))::Float64
    isfinite(v) || (cf.n_nonfinite[] += 1)
    return v
end

"""
    ncalls(cf::CostFunction) -> Int

Number of times this `CostFunction` has been called.
"""
ncalls(cf::CostFunction) = cf.nfcn[]

"""
    reset_ncalls!(cf::CostFunction) -> CostFunction

Reset the call counter (and the non-finite-return counter) to zero.
Returns `cf`.
"""
function reset_ncalls!(cf::CostFunction)
    cf.nfcn[] = 0
    cf.n_nonfinite[] = 0
    return cf
end

"""
    nonfinite_calls(cf) -> Int

Number of FCN evaluations (so far) that returned a non-finite value
(`NaN`/`±Inf`). Defined for both `CostFunction` and
`CostFunctionWithGradient`. Reset together with the call counter by
[`reset_ncalls!`](@ref).
"""
nonfinite_calls(cf::CostFunction) = cf.n_nonfinite[]

"""
    errordef(cf::CostFunction)

Return the error definition (`up`) — `1.0` for χ², `0.5` for NLL.
"""
errordef(cf::CostFunction) = cf.up
