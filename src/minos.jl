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
- `upper_state::Union{Nothing,Vector{Float64}}`,
  `lower_state::Union{Nothing,Vector{Float64}}` — full parameter
  snapshot at the ±σ crossing endpoint. `nothing` when that side
  did not converge cleanly. Mirrors C++ `MinosError::UpperState()` /
  `LowerState()` (`MinosError.h:73-74`). Useful for HEP correlated-
  systematic studies and at-bound diagnostics — see GAP_AUDIT.md M4.

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
    # M4 (GAP_AUDIT): full parameter snapshot at the ±σ crossing
    # endpoints. C++ `MinosError::UpperState()` / `LowerState()`
    # (MinosError.h:73-74). `nothing` when that side did not converge
    # cleanly. The unbounded path assembles via
    # `_assemble_crossing_state` from the inner state + the scanned
    # parameter's crossing value; the bounded path reads
    # `MnCross.ext_state` (captured by `function_cross_external`'s
    # probe-Ref).
    upper_state::Union{Nothing,Vector{Float64}}
    lower_state::Union{Nothing,Vector{Float64}}
end

# Backward-compatible constructor (legacy callers that don't pass
# par_limit / state fields). Defaults par_limit flags to false and
# state snapshots to `nothing`. Public API; bounded MINOS path passes
# them explicitly.
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
                nfcn,
                nothing, nothing)   # state snapshot defaults
end

# Backward-compatible constructor: par_limit fields explicit, state
# snapshots defaulted to `nothing` (callers that ran BEFORE M4 landed).
function MinosError(par_idx::Int, min_par_value::Float64,
                     upper::Float64, lower::Float64,
                     upper_valid::Bool, lower_valid::Bool,
                     upper_new_min::Bool, lower_new_min::Bool,
                     upper_fcn_limit::Bool, lower_fcn_limit::Bool,
                     upper_par_limit::Bool, lower_par_limit::Bool,
                     nfcn::Int)
    MinosError(par_idx, min_par_value, upper, lower,
                upper_valid, lower_valid,
                upper_new_min, lower_new_min,
                upper_fcn_limit, lower_fcn_limit,
                upper_par_limit, lower_par_limit,
                nfcn,
                nothing, nothing)
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
    minos(fmin, cf, par_idx; tlr=0.1, maxcalls=1000, sigma=1,
          strategy=Strategy(0), prec=MachinePrecision()) -> MinosError

Compute asymmetric ±σ errors for parameter `par_idx`. Mirrors
`MnMinos::Minos(unsigned int, ...)` from
`reference/Minuit2_cpp/src/MnMinos.cxx`.

# Phase 1 first cut

- Unbounded parameters only (par_limit reserved for Phase 1+ bounds
  integration).
- Inner MIGRAD uses Strategy(0) by default. Strategy 1/2 affects the
  `tlr` propagation but not HESSE refinement.
- `sigma::Real=1` — confidence level in σ-units (P5). Threads
  `up · sigma²` into MnFunctionCross's `aim` (mirrors iminuit's
  `_TemporaryUp`). The returned `upper` / `lower` then correspond
  to the k-σ contour on the parameter.

# Returns

A [`MinosError`](@ref). Use `is_valid(e)` to check overall success.
`upper_state` / `lower_state` carry the full parameter vector at
the ±σ crossing endpoint (`nothing` when that side did not converge).
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
    sigma::Real = 1.0,
    print_level::Integer = 0,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    state = fmin.state
    n = length(state.parameters)
    1 <= par_idx <= n ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n"))
    n > 1 ||
        throw(ArgumentError("MINOS requires n > 1 free parameters"))

    # The C++ Min() returns the parameter value at the minimum
    # (fMinParValue). Parallel-review #4 B2.
    min_par_value = state.parameters.x[par_idx]
    par_idx_i = Int(par_idx)

    # gap M1: per-direction headers mirror C++ MnMinos.cxx:105
    # "Determination of upper/lower Minos error for parameter ...".
    # Outer-guarded — minos is called per parameter, so we avoid the
    # @sprintf String alloc per direction × per parameter at level 0.
    if print_level >= 1
        _trace_info(print_level, "MnMinos",
                    @sprintf("Determination of upper error for par=%d (value=%.10g)",
                              par_idx, min_par_value))
    end

    # Upper direction (positive). Threads the optional `scratch` —
    # both upper + lower cross searches use the same inner_dim (n-1),
    # so they can pool one MigradScratch across all ~6-10 inner-MIGRAD
    # probes per side.
    up_cross = function_cross(fmin, cf, par_idx, +1.0;
                                tlr = tlr, maxcalls = maxcalls,
                                strategy = strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient,
                                sigma = sigma,
                                print_level = print_level)
    # 1-sigma external step (same as inside function_cross). NOTE: for
    # sigma=k, aopt at convergence ≈ k so `aopt · sigma_i` is the k-σ
    # error (P5).
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
    # M4: full parameter snapshot at the upper ±σ crossing. The inner
    # state has dimension n-1 (par_idx removed); reinsert par_idx at
    # its slot with value `min_par_value + aopt · sigma_i`.
    upper_state = up_cross.valid ?
        _assemble_crossing_state(up_cross.state, par_idx_i,
                                  min_par_value + up_cross.aopt * sigma_i,
                                  n) : nothing
    nfcn_total = up_cross.nfcn

    if print_level >= 1
        _trace_info(print_level, "MnMinos",
                    @sprintf("Determination of lower error for par=%d (value=%.10g)",
                              par_idx, min_par_value))
    end

    # Lower direction (negative). aopt comes out positive; flip sign.
    lo_cross = function_cross(fmin, cf, par_idx, -1.0;
                                tlr = tlr,
                                maxcalls = maxcalls,
                                strategy = strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient,
                                sigma = sigma,
                                print_level = print_level)
    lower = lo_cross.valid ? -lo_cross.aopt * sigma_i : 0.0
    lower_state = lo_cross.valid ?
        _assemble_crossing_state(lo_cross.state, par_idx_i,
                                  min_par_value - lo_cross.aopt * sigma_i,
                                  n) : nothing
    nfcn_total += lo_cross.nfcn

    if print_level >= 1
        valid_str = up_cross.valid && lo_cross.valid ? "VALID" : "PARTIAL"
        _trace_info(print_level, "MnMinos",
                    @sprintf("done par=%d  +%.6g  -%.6g  ncalls=%d  %s",
                             par_idx, upper, -lower, nfcn_total, valid_str))
    end

    return MinosError(
        par_idx_i,
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
        upper_state,
        lower_state,
    )
end

# Internal: assemble the full n-dim parameter vector at a MnFunctionCross
# crossing endpoint. `inner_state` carries the (n-1)-dim inner-MIGRAD's
# converged free-coord values; we re-insert the scanned parameter
# (`par_idx`, value `par_val`) at its slot. Mirrors the C++
# `MinosError::UpperState() / LowerState()` snapshot (MinosError.h:73-74).
function _assemble_crossing_state(inner_state::MinimumState,
                                   par_idx::Int, par_val::Float64, n::Int)
    inner_x = inner_state.parameters.x
    # Defensive: shape mismatch implies the inner_state isn't the (n-1)-dim
    # cross-search state — fall through to `nothing` rather than scramble.
    length(inner_x) == n - 1 || return nothing
    out = Vector{Float64}(undef, n)
    @inbounds for k in 1:(par_idx - 1)
        out[k] = inner_x[k]
    end
    @inbounds out[par_idx] = par_val
    @inbounds for k in (par_idx + 1):n
        out[k] = inner_x[k - 1]
    end
    return out
end

"""
    minos(fmin, cf; tlr=0.1, maxcalls=1000, ...) -> Vector{MinosError}

Compute MINOS errors for ALL free parameters. Convenience wrapper
that calls the single-parameter overload in turn. Mirrors C++
`MnMinos::operator()` which iterates over `0..n-1`.
"""
function minos(fmin::FunctionMinimum, cf::AbstractCostFunction; kwargs...)
    n = length(fmin.state.parameters)
    # `print_level` (if present in kwargs) is plumbed through `kwargs...`
    # to the single-parameter overload above.
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
