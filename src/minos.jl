# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# minos.jl — MnMinos asymmetric errors (Phase 1 first cut).
#
# Mirrors reference/Minuit2_cpp/src/MnMinos.cxx.
#
# For each requested parameter, scans the function in both directions
# (+ and -) until f - fmin = up, giving asymmetric ±σ errors. Uses
# `function_cross` under the hood for each direction.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MinosError

The asymmetric error result for a single parameter. Mirrors C++
`MinosError`.

# Fields

- `par_idx::Int` — 1-based parameter index.
- `min_fval::Float64` — the function value at the minimum.
- `upper::Float64` — upper asymmetric error (`x_+σ - x_min`).
- `lower::Float64` — lower asymmetric error (`x_-σ - x_min`, ≤ 0).
- `upper_valid::Bool`, `lower_valid::Bool` — `true` if the crossing
  was found cleanly.
- `upper_new_min::Bool`, `lower_new_min::Bool` — `true` if a lower
  minimum was discovered during the scan (caller should restart
  MIGRAD from the better point).
- `upper_fcn_limit::Bool`, `lower_fcn_limit::Bool` — call budget hit.
- `nfcn::Int` — total FCN calls across both directions.

# Note on sign convention

`upper` is positive (one σ to the right), `lower` is negative (one σ
to the left). For a symmetric well-behaved parabolic minimum,
`upper ≈ -lower ≈ sqrt(2·up·V[i,i])`.
"""
struct MinosError
    par_idx::Int
    min_fval::Float64
    upper::Float64
    lower::Float64
    upper_valid::Bool
    lower_valid::Bool
    upper_new_min::Bool
    lower_new_min::Bool
    upper_fcn_limit::Bool
    lower_fcn_limit::Bool
    nfcn::Int
end

"""
    is_valid(e::MinosError) -> Bool

True if both upper and lower errors were found within tolerance.
"""
is_valid(e::MinosError) = e.upper_valid && e.lower_valid

# ─────────────────────────────────────────────────────────────────────────────

"""
    minos(fmin, cf, par_idx; tlr=0.1, maxcalls=1000,
          strategy=Strategy(0), prec=MachinePrecision()) -> MinosError

Compute asymmetric ±σ errors for parameter `par_idx`. Mirrors
`MnMinos::Minos(unsigned int, ...)` from
`reference/Minuit2_cpp/src/MnMinos.cxx`.

# Phase 1 first cut

- Unbounded parameters only (par_limit reserved for Phase 1+ bounds
  integration).
- Inner MIGRAD uses Strategy(0) by default. Strategy 1/2 affects the
  `tlr` propagation but not HESSE refinement.

# Returns

A [`MinosError`](@ref). Use `is_valid(e)` to check overall success.
"""
function minos(
    fmin::FunctionMinimum,
    cf::CostFunction,
    par_idx::Integer;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    state = fmin.state
    n = length(state.parameters)
    1 <= par_idx <= n ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n"))
    n > 1 ||
        throw(ArgumentError("MINOS requires n > 1 free parameters"))

    min_fval = state.parameters.fval

    # Upper direction (positive)
    up_cross = function_cross(fmin, cf, par_idx, +1.0;
                                tlr = tlr, maxcalls = maxcalls,
                                strategy = strategy, prec = prec)
    # 1-sigma external step (same as inside function_cross)
    sigma_i = sqrt(max(2.0 * cf.up * state.error.inv_hessian[par_idx, par_idx],
                        prec.eps2))
    upper = up_cross.valid ? up_cross.aopt * sigma_i : NaN
    nfcn_total = up_cross.nfcn

    # Lower direction (negative). aopt comes out positive; flip sign.
    lo_cross = function_cross(fmin, cf, par_idx, -1.0;
                                tlr = tlr,
                                maxcalls = maxcalls,
                                strategy = strategy, prec = prec)
    lower = lo_cross.valid ? -lo_cross.aopt * sigma_i : NaN
    nfcn_total += lo_cross.nfcn

    return MinosError(
        Int(par_idx),
        min_fval,
        upper,
        lower,
        up_cross.valid,
        lo_cross.valid,
        up_cross.new_min,
        lo_cross.new_min,
        up_cross.fcn_limit,
        lo_cross.fcn_limit,
        nfcn_total,
    )
end

"""
    minos(fmin, cf; tlr=0.1, maxcalls=1000, ...) -> Vector{MinosError}

Compute MINOS errors for ALL free parameters. Convenience wrapper
that calls the single-parameter overload in turn. Mirrors C++
`MnMinos::operator()` which iterates over `0..n-1`.
"""
function minos(fmin::FunctionMinimum, cf::CostFunction; kwargs...)
    n = length(fmin.state.parameters)
    return [minos(fmin, cf, i; kwargs...) for i in 1:n]
end
