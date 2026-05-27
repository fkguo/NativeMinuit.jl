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
- `min_par_value::Float64` — the **parameter value at the minimum**
  (NOT the function value). Mirrors C++ `MinosError::Min()`
  (`reference/Minuit2_cpp/inc/Minuit2/MinosError.h:85-86`,
  `fMinParValue` at line 30-31). Parallel-review #4 B2 — v1 of
  this field stored fval, which was a semantic mismatch.
- `upper::Float64` — upper asymmetric error (`x_+σ - x_min`).
- `lower::Float64` — lower asymmetric error (`x_-σ - x_min`, ≤ 0).
- `upper_valid::Bool`, `lower_valid::Bool` — `true` if the MINOS
  analysis completed cleanly on that side. **True also when the
  search saturated against a parameter bound** (the corresponding
  `upper_par_limit` / `lower_par_limit` flag is then raised and the
  published value is the physical bound_distance — `x_bound − x_min`).
  This matches iminuit's `m.merrors[name].is_valid` semantics: hitting
  a bound is a legitimate MINOS termination, not a failure.
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
    min_par_value::Float64
    upper::Float64
    lower::Float64
    upper_valid::Bool
    lower_valid::Bool
    upper_new_min::Bool
    lower_new_min::Bool
    upper_fcn_limit::Bool
    lower_fcn_limit::Bool
    # par_limit fields distinguish "ran out of FCN calls" (a budget
    # problem the user can fix by increasing maxcalls) from "search
    # hit a parameter bound" (a model problem the user can fix only
    # by relaxing the bound). C++ MnCross has the equivalent flag
    # via CrossParLimit(); Opus round-3 I-4. Set to false by the
    # backward-compatible constructor below.
    upper_par_limit::Bool
    lower_par_limit::Bool
    nfcn::Int
end

# Backward-compatible constructor (legacy callers that don't pass
# par_limit fields). Defaults the par_limit flags to false. Public
# API; bounded MINOS path passes them explicitly.
function MinosError(par_idx::Int, min_par_value::Float64,
                     upper::Float64, lower::Float64,
                     upper_valid::Bool, lower_valid::Bool,
                     upper_new_min::Bool, lower_new_min::Bool,
                     upper_fcn_limit::Bool, lower_fcn_limit::Bool,
                     nfcn::Int)
    MinosError(par_idx, min_par_value, upper, lower,
                upper_valid, lower_valid,
                upper_new_min, lower_new_min,
                upper_fcn_limit, lower_fcn_limit,
                false, false,   # par_limit defaults
                nfcn)
end

"""
    is_valid(e::MinosError) -> Bool

True if both upper and lower MINOS analyses completed cleanly.
Includes the at-bound case (`upper_par_limit` / `lower_par_limit`):
saturating against a parameter bound is treated as a clean termination
with a physically meaningful published value (the bound distance),
matching iminuit's `m.merrors[name].is_valid`.
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
    cf::AbstractCostFunction,
    par_idx::Integer;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
)
    state = fmin.state
    n = length(state.parameters)
    1 <= par_idx <= n ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n"))
    n > 1 ||
        throw(ArgumentError("MINOS requires n > 1 free parameters"))

    # The C++ Min() returns the parameter value at the minimum
    # (fMinParValue). Parallel-review #4 B2.
    min_par_value = state.parameters.x[par_idx]

    # Upper direction (positive). Threads the optional `scratch` —
    # both upper + lower cross searches use the same inner_dim (n-1),
    # so they can pool one MigradScratch across all ~6-10 inner-MIGRAD
    # probes per side.
    up_cross = function_cross(fmin, cf, par_idx, +1.0;
                                tlr = tlr, maxcalls = maxcalls,
                                strategy = strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient)
    # 1-sigma external step (same as inside function_cross)
    sigma_i = sqrt(max(2.0 * cf.up * state.error.inv_hessian[par_idx, par_idx],
                        prec.eps2))
    # Invalid-side encoding: 0.0 (NOT NaN), avoiding NaN propagation
    # in downstream code that gates on `e.upper > threshold`. The
    # bounded MINOS path (in src/minuit.jl) publishes the actual
    # bound_distance when `par_limit=true` and 0.0 only for non-bound
    # failure modes — unbounded fits have no `par_limit` so they
    # consistently fall into the 0.0 branch here. Round-3 I-2 (Opus);
    # bounded behavior refined in round-6.
    upper = up_cross.valid ? up_cross.aopt * sigma_i : 0.0
    nfcn_total = up_cross.nfcn

    # Lower direction (negative). aopt comes out positive; flip sign.
    lo_cross = function_cross(fmin, cf, par_idx, -1.0;
                                tlr = tlr,
                                maxcalls = maxcalls,
                                strategy = strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient)
    lower = lo_cross.valid ? -lo_cross.aopt * sigma_i : 0.0
    nfcn_total += lo_cross.nfcn

    return MinosError(
        Int(par_idx),
        min_par_value,
        upper,
        lower,
        up_cross.valid,
        lo_cross.valid,
        up_cross.new_min,
        lo_cross.new_min,
        up_cross.fcn_limit,
        lo_cross.fcn_limit,
        up_cross.par_limit,
        lo_cross.par_limit,
        nfcn_total,
    )
end

"""
    minos(fmin, cf; tlr=0.1, maxcalls=1000, ...) -> Vector{MinosError}

Compute MINOS errors for ALL free parameters. Convenience wrapper
that calls the single-parameter overload in turn. Mirrors C++
`MnMinos::operator()` which iterates over `0..n-1`.
"""
function minos(fmin::FunctionMinimum, cf::AbstractCostFunction; kwargs...)
    n = length(fmin.state.parameters)
    return [minos(fmin, cf, i; kwargs...) for i in 1:n]
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print (iminuit-style box) — Phase 3 parity polish
# ─────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", e::MinosError)
    # iminuit "Minos" box: 71-char width, 3-row status
    println(io, "┌", "─"^71, "┐")
    println(io, "│", _center("Minos — par x$(e.par_idx-1)", 71), "│")
    println(io, "├", "─"^35, "┬", "─"^35, "┤")
    val_str = " value = $(_fmt_num(e.min_par_value))"
    nfcn_str = _center("Nfcn = $(e.nfcn)", 35)
    println(io, "│", _ljust(val_str, 35), "│", nfcn_str, "│")
    println(io, "├", "─"^35, "┼", "─"^35, "┤")
    err_str = " error = +$(_fmt_num(e.upper))  −$(_fmt_num(-e.lower))"
    valid_str = is_valid(e) ? "Valid" : "INVALID"
    println(io, "│", _ljust(err_str, 35), "│", _center(valid_str, 35), "│")
    println(io, "├", "─"^35, "┼", "─"^35, "┤")
    up_status = if e.upper_new_min
        "Upper: NEW MIN found"
    elseif e.upper_par_limit
        "Upper: AT LIMIT"
    elseif e.upper_fcn_limit
        "Upper: call-limit hit"
    elseif e.upper_valid
        "Upper: OK"
    else
        "Upper: FAILED"
    end
    lo_status = if e.lower_new_min
        "Lower: NEW MIN found"
    elseif e.lower_par_limit
        "Lower: AT LIMIT"
    elseif e.lower_fcn_limit
        "Lower: call-limit hit"
    elseif e.lower_valid
        "Lower: OK"
    else
        "Lower: FAILED"
    end
    println(io, "│", _center(up_status, 35), "│", _center(lo_status, 35), "│")
    println(io, "└", "─"^35, "┴", "─"^35, "┘")
end

Base.show(io::IO, e::MinosError) =
    print(io, "MinosError(par=", e.par_idx, ", val=", e.min_par_value,
              ", +", e.upper, " −", -e.lower,
              ", valid=", is_valid(e), ")")

# Vector{MinosError} — one box per error
function Base.show(io::IO, mime::MIME"text/plain", es::AbstractVector{MinosError})
    if isempty(es)
        println(io, "Empty Vector{MinosError}")
        return
    end
    for (i, e) in enumerate(es)
        i > 1 && println(io)
        show(io, mime, e)
    end
end
