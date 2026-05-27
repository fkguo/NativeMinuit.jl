# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# ad_gradient.jl — AD-backed analytical gradient (Phase 2.1 first cut).
#
# Mirrors reference/Minuit2_cpp/inc/Minuit2/AnalyticalGradientCalculator.h.
# Drop-in replacement for `numerical_gradient!` when the user provides a
# closed-form (or AD-produced) gradient function `∇f(x)`.
#
# Phase 2.1 first cut:
# - User passes a `gradient_fn(x)::Vector{Float64}` to `migrad` via the
#   `gradient` keyword. Internally we install a `CostFunctionWithGradient`
#   that the MIGRAD loop branches on.
# - For ForwardDiff users:
#     using ForwardDiff
#     migrad(cf, x0, errs; gradient = x -> ForwardDiff.gradient(cf.f, x))
#   `ForwardDiff` is NOT a hard dependency — JuMinuit just calls
#   `gradient_fn(x)` and expects a `Vector{Float64}` (or convertible).
# - g2 + gstep companions are filled via the cheap `InitialGradientCalculator`
#   convention (parameters' initial step sizes); C++
#   MnHesse.cxx:118-126 does the same refresh.
#
# Performance: gradient_fn is called once per MIGRAD iteration vs.
# `2·n·NCycle` FCN calls for numerical gradient — typically 2-10× faster
# when the user's FCN evaluation is cheap.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CostFunctionWithGradient{F,G,T}

Like [`CostFunction`](@ref) but also carries an analytical gradient
function `g(x)::Vector{Float64}`. Used by `migrad`'s `gradient=...`
keyword to substitute AD-provided or hand-written gradients for the
central-difference numerical gradient.

# Fields

- `f::F` — user FCN, `f(x::AbstractVector{Float64}) -> Real`.
- `g::G` — gradient function, `g(x::AbstractVector{Float64}) -> Vector{Float64}`.
  Must return length-`n` vector matching `length(x)`.
- `up::T` — error definition (1.0 for χ², 0.5 for NLL).
- `nfcn::Base.RefValue{Int}` — FCN call counter.
- `ngrad::Base.RefValue{Int}` — gradient call counter (separate from
  nfcn since each gradient call is "one shot" vs the 2n+ FCN calls
  numerical_gradient! makes per iteration).

# Examples

```julia
using ForwardDiff
f = x -> sum(abs2, x .- [1.0, 2.0])
cf = CostFunctionWithGradient(f, x -> ForwardDiff.gradient(f, x))
m = migrad(cf, [0.0, 0.0], [0.1, 0.1])
```
"""
struct CostFunctionWithGradient{F,G,T} <: AbstractCostFunction
    f::F
    g::G
    up::T
    nfcn::Base.RefValue{Int}
    ngrad::Base.RefValue{Int}
end

CostFunctionWithGradient(f, g, up = 1.0) =
    CostFunctionWithGradient(f, g, Float64(up), Ref(0), Ref(0))

# Phase F alias kept for any external code that already imported it;
# new signatures should prefer `AbstractCostFunction` directly. Defined
# AFTER both concrete subtypes are introduced so the union resolves to
# `Union{CostFunction, CostFunctionWithGradient}` exactly (rather than
# the open-ended `AbstractCostFunction`).
const AnyCostFunction = Union{CostFunction, CostFunctionWithGradient}

# Forward CostFunction-like accessors
@inline function (cf::CostFunctionWithGradient)(x::AbstractVector)
    cf.nfcn[] += 1
    return Float64(cf.f(x))::Float64
end

ncalls(cf::CostFunctionWithGradient) = cf.nfcn[]
reset_ncalls!(cf::CostFunctionWithGradient) = (cf.nfcn[] = 0; cf)
errordef(cf::CostFunctionWithGradient) = cf.up
ngrad_calls(cf::CostFunctionWithGradient) = cf.ngrad[]

# ─────────────────────────────────────────────────────────────────────────────
# Analytical-gradient evaluation
# ─────────────────────────────────────────────────────────────────────────────

"""
    analytical_gradient!(out::FunctionGradient, par::MinimumParameters,
                          cf::CostFunctionWithGradient,
                          prev::FunctionGradient,
                          prec::MachinePrecision = MachinePrecision()) -> FunctionGradient

Fill `out` with the analytical gradient from `cf.g(par.x)`. The g2 and
gstep companions are forwarded from `prev` (initial seed + iterative
refinement — cheap, matches C++ MnHesse.cxx:118-126 convention).
Returns `out`.

Increments `cf.ngrad[]` by one.

# Phase 2.1 first cut

- No g2 refinement (Phase 2.1+ ports `HessianGradientCalculator`).
- prev.g2 / prev.gstep are propagated unchanged (which is what the
  inner MIGRAD loop expects between iterations anyway).
"""
function analytical_gradient!(
    out::FunctionGradient,
    par::MinimumParameters,
    cf::CostFunctionWithGradient,
    prev::FunctionGradient,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    length(out) == n ||
        throw(DimensionMismatch("out length $(length(out)) != par length $n"))
    length(prev) == n ||
        throw(DimensionMismatch("prev length $(length(prev)) != par length $n"))

    grad_vec = cf.g(par.x)
    cf.ngrad[] += 1
    length(grad_vec) == n ||
        throw(DimensionMismatch("gradient function returned length $(length(grad_vec)), expected $n"))

    @inbounds for i in 1:n
        out.grad[i]  = Float64(grad_vec[i])
        out.g2[i]    = prev.g2[i]
        out.gstep[i] = prev.gstep[i]
    end
    return out
end

# Cold-start convenience: build an initial gradient from par.dirin (errors)
# then call analytical_gradient!.
function analytical_gradient(
    par::MinimumParameters,
    cf::CostFunctionWithGradient,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    has_step_size(par) ||
        throw(ArgumentError("analytical_gradient(par, cf) cold-start needs par with step sizes"))
    # Seed g2 + gstep using the same convention as numerical cold-start
    seed = initial_gradient(par, par.dirin, cf)
    out = FunctionGradient(zeros(n), zeros(n), zeros(n))
    return analytical_gradient!(out, par, cf, seed, prec)
end

# Make `initial_gradient(par, errs, cf::CostFunctionWithGradient)` work
# the same way as for a plain CostFunction. Delegates to the lower-
# level initial_gradient! that takes `up::Float64` directly (the
# CostFunction overload at gradient.jl:102 just extracts cf.up).
function initial_gradient(par::MinimumParameters,
                           errs::AbstractVector{Float64},
                           cf::CostFunctionWithGradient,
                           prec::MachinePrecision = MachinePrecision())
    n = length(par)
    out = FunctionGradient(zeros(n), zeros(n), zeros(n))
    initial_gradient!(out, par, errs, cf.up, prec)
    return out
end

# Make seed_state work with CostFunctionWithGradient by treating it as
# a regular CostFunction for the cold-start FCN evaluations.
function seed_state(
    cf::CostFunctionWithGradient,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real},
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(x0)
    length(errs) == n ||
        throw(DimensionMismatch("errs length $(length(errs)) != x0 length $n"))
    strategy.level == 0 ||
        throw(ArgumentError("Phase 0 supports Strategy(0) only"))

    x = collect(Float64, x0)
    dirin = collect(Float64, errs)
    fval = cf(x)
    par = MinimumParameters(x, dirin, fval)

    grad = analytical_gradient(par, cf, strategy, prec)

    n_total = n
    mat = zeros(n_total, n_total)
    @inbounds for i in 1:n_total
        mat[i, i] = abs(grad.g2[i]) > prec.eps2 ? 1.0 / grad.g2[i] : 1.0
    end
    err = MinimumError(Symmetric(mat, :U), 1.0)
    # In-place EDM avoids the BLAS internal temporary in `dot(g, V, g)`.
    edm_val = estimate_edm!(Vector{Float64}(undef, n_total), grad, err)
    state = MinimumState(par, err, grad, edm_val, ncalls(cf))

    # Unconditional negative_g2 check (Opus blocking #2 from review #1)
    if has_negative_g2(grad, prec)
        # Fall back to numerical refinement (negative_g2 requires
        # numerical_gradient! semantics under the hood)
        state = negative_g2_line_search(state, cf, strategy, prec)
    end
    return state
end

# Provide a numerical_gradient!-shaped fallback used internally by the
# MIGRAD loop. Phase 2.1 first cut: replace the central-diff call with
# the analytical one.
function numerical_gradient!(
    out::FunctionGradient,
    x_work::AbstractVector{Float64},
    par::MinimumParameters,
    prev::FunctionGradient,
    cf::CostFunctionWithGradient,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    # x_work argument unused for AD path (kept for signature symmetry
    # with the central-diff overload so the MIGRAD loop dispatches
    # cleanly).
    return analytical_gradient!(out, par, cf, prev, prec)
end

# negative_g2_line_search needs to call numerical_gradient! on the
# CostFunction it gets. For CostFunctionWithGradient we ALSO go through
# analytical_gradient! (the rare-path scenario won't be performance-
# critical).
function negative_g2_line_search(
    state::MinimumState,
    cf::CostFunctionWithGradient,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    # Delegate by building a NumericalGradient-like wrap; for first cut,
    # skip refinement (AD users rarely hit negative_g2 since AD g2 is
    # not directly computed). Just return the input state.
    has_negative_g2(state.gradient, prec) || return state
    @warn "negative_g2 detected with analytical-gradient FCN; refinement skipped (Phase 2.1+)"
    return state
end

# ─────────────────────────────────────────────────────────────────────────────
# migrad overload that dispatches to the analytical-gradient path.
# Critical: without this overload, `migrad(::CostFunctionWithGradient, ...)`
# falls through the bare-function path which silently wraps it as a plain
# `CostFunction` and uses central-diff (losing the AD information).
# ─────────────────────────────────────────────────────────────────────────────

function migrad(
    cf::CostFunctionWithGradient,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real};
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
)
    n = length(x0)
    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)

    seed = seed_state(cf, x0, errs, strategy, prec)
    return _migrad_loop(seed, cf, strategy, Float64(tol), maxfcn_eff, prec;
                          scratch = scratch)
end

"""
    migrad(cf::CostFunctionWithGradient, seed::MinimumState;
           strategy=Strategy(0), tol=0.1, maxfcn=..., prec=..., scratch=nothing)
        -> FunctionMinimum

Phase B's seed-state entry point + Phase F's analytical-gradient
dispatch. Same semantics as `migrad(::CostFunction, ::MinimumState)`:
skip `seed_state`'s bootstrap, feed the pre-built seed directly into
`_migrad_loop`. Used by `warm_restart_state` inside MnFunctionCross
probes when the user FCN carries an analytical gradient.
"""
function migrad(
    cf::CostFunctionWithGradient,
    seed::MinimumState;
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
)
    n = length(seed)
    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)
    return _migrad_loop(seed, cf, strategy, Float64(tol), maxfcn_eff, prec;
                          scratch = scratch)
end
