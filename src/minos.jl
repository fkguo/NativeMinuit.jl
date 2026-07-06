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
  systematic studies and at-bound diagnostics — see docs/dev/GAP_AUDIT.md M4.

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

# Notes

- Bounded parameters are supported; hitting a bound is a clean termination
  (recorded in `upper_par_limit` / `lower_par_limit`).
- Inner MIGRAD uses Strategy(0) by default. Strategy 1/2 affects the
  `tlr` propagation but not HESSE refinement.
- `sigma::Real=1` — confidence level in σ-units (P5). Threads
  `up · sigma²` into MnFunctionCross's `aim` (mirrors iminuit's
  `_TemporaryUp`). The returned `upper` / `lower` then correspond
  to the k-σ contour on the parameter.

# C++ MnMinos algorithm reproduction (X(3872) follow-up)

This function reproduces C++ `MnMinos::FindCrossValue`
(`MnMinos.cxx:136-165`) including:

1. **Linear-correlation pre-shift** of OTHER free parameters along the
   inverse-Hessian direction before the first inner MIGRAD probe
   (avoids gradient-descent into side basins on non-convex profiles).
2. **Outer V → inner `prior_cov`**: the (n-1)×(n-1) minor of
   `state.error.inv_hessian` (par_idx row/col removed) is passed to
   the inner MIGRAD so the DFP starts with the OUTER's correlation
   structure (mirrors C++ MnMigrad's single-instance covariance reuse).
3. **Cold-fallback from `warm_state.x`** on subsequent probes when
   `warm_restart_state` fails (negative g2 / edm regression) — the
   inner DFP rebuilds g2 from scratch but keeps the converged position.
4. **±σ_HESSE placeholder** on invalid sides (matches C++
   `MinosError::Upper()` / `Lower()` at `MinosError.h:54`).

# Known limitations

- **Numerical-gradient inner MIGRAD on pathological profiles**: with
  finite-difference gradients (`grad=` not supplied to `Minuit`),
  some strongly-correlated non-convex fits (e.g. X(3872) `par[2]` /
  `par[3]` lower) can converge the inner MIGRAD to the wrong (side)
  basin even WITH pre-shift + `prior_cov`, returning the ±σ_HESSE
  placeholder. iminuit's numerical-gradient inner MIGRAD finds the
  same crossings via subtle line-search / step-size differences not
  yet replicated here. Supplying analytical gradient (`grad=` AD)
  closes this gap — see `BenchmarkExamples/X3872_dip/bench_full.jl`
  for a worked example where jm_ad matches iminuit and jm_num does
  not. Tracked as a follow-up: needs deeper investigation of
  Numerical2P step heuristics or Simplex retry within the inner
  cross-search.

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
    pars::Union{Nothing,Parameters} = nothing,
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

    # 1-sigma external step (same as inside function_cross). NOTE: for
    # sigma=k, aopt at convergence ≈ k so `aopt · sigma_i` is the k-σ
    # error (P5).
    sigma_i = sqrt(max(2.0 * cf.up * state.error.inv_hessian[par_idx, par_idx],
                        prec.eps2))

    # ── MnMinos linear-correlation pre-shift (C++ MnMinos.cxx:136-165) ──
    # When scanning par_idx by 1σ in direction `dir`, the linearized
    # χ² predicts that the OTHER free parameters shift by
    #     Δx_k = dir · σ_par_idx · V[par_idx, k] / V[par_idx, par_idx]
    # where V = state.error.inv_hessian. Verified algebraically:
    # `C++ xunit · m[ind,k] = sqrt(up/(2·V[ii,ii])) · 2·V[ik]
    #                       = sigma_i · V[ik]/V[ii]` since
    # `m = MinimumError::Matrix() = 2·V` (BasicMinimumError.h:104). The
    # 2·up and 2× factors cancel in the ratio, so the NativeMinuit formula
    # in V (internal inv_hessian) is identical to the C++ one in m.
    #
    # When `pars !== nothing` and some "other" param has external
    # bounds, we additionally do Int2ext → EXT clamp → Ext2int on the
    # pre-shifted internal value. This mirrors C++ MnMinos.cxx:152-160
    # (`Int2ext` + `min/max` against parameter limit + `SetValue`
    # which round-trips back through Ext2int). Without this clamp a
    # doubly-bounded "other" param with large cross-correlation can
    # be pre-shifted past ±π/2 in internal coords and the next FCN
    # evaluation aliases through `sin()`.
    x_min = state.parameters.x
    V = state.error.inv_hessian
    Vii = V[par_idx, par_idx]
    seed_upper = Vector{Float64}(undef, n - 1)
    seed_lower = Vector{Float64}(undef, n - 1)
    if isfinite(Vii) && Vii > 0
        # Sin-transform saturation limits for doubly-bounded params
        # (review v2 IMPORTANT B — aliasing pre-clamp). Mirrors
        # `sin_ext2int` (src/transform.jl:62-64): the valid internal
        # range for BothBounds is `[-π/2 + 8√eps2, π/2 - 8√eps2]`;
        # outside this range `sin()` aliases and an `int2ext +
        # EXT-clamp + ext2int` round-trip lands on the wrong asin
        # branch (silently, when the aliased EXT happens to stay
        # inside the user's bounds). Pre-clamping INT to this range
        # BEFORE the EXT round-trip eliminates the aliasing pathology
        # while preserving the saturation semantics of the
        # full-precision `sin_ext2int` path.
        piby2 = 2.0 * atan(1.0)
        distnn_int = 8.0 * sqrt(prec.eps2)
        vlimhi_int = piby2 - distnn_int
        vlimlo_int = -piby2 + distnn_int
        @inbounds for k in 1:n
            k == par_idx && continue
            shift = sigma_i * (V[par_idx, k] / Vii)
            j = k < par_idx ? k : k - 1
            su = x_min[k] + shift     # dir = +1, raw INT shift
            sl = x_min[k] - shift     # dir = -1, raw INT shift
            if pars !== nothing
                # Mixed case (caller from src/minuit.jl::minos!): the
                # internal index k corresponds to ext_of_int[k]. Map
                # to ext, clamp, map back to int.
                ext_idx = pars.ext_of_int[k]
                p_ext = pars.pars[ext_idx]
                if has_limits(p_ext)
                    kind = bound_kind(p_ext)
                    # For BothBounds (sin transform), saturate INT to
                    # the valid range BEFORE Int2ext to avoid sin()
                    # aliasing on large pre-shifts.
                    if kind == BothBounds
                        su = clamp(su, vlimlo_int, vlimhi_int)
                        sl = clamp(sl, vlimlo_int, vlimhi_int)
                    end
                    su_ext = int2ext(kind, su, p_ext.lower, p_ext.upper)
                    sl_ext = int2ext(kind, sl, p_ext.lower, p_ext.upper)
                    if has_upper_limit(p_ext)
                        su_ext = min(su_ext, p_ext.upper)
                        sl_ext = min(sl_ext, p_ext.upper)
                    end
                    if has_lower_limit(p_ext)
                        su_ext = max(su_ext, p_ext.lower)
                        sl_ext = max(sl_ext, p_ext.lower)
                    end
                    su = ext2int(kind, su_ext, p_ext.lower, p_ext.upper, prec)
                    sl = ext2int(kind, sl_ext, p_ext.lower, p_ext.upper, prec)
                end
            end
            seed_upper[j] = su
            seed_lower[j] = sl
        end
    else
        # Degenerate HESSE (Vii ≤ 0 or NaN) — fall back to unshifted
        # seed; function_cross will likely fail too but the failure
        # mode matches the pre-fix behavior.
        @inbounds for k in 1:n
            k == par_idx && continue
            j = k < par_idx ? k : k - 1
            seed_upper[j] = x_min[k]
            seed_lower[j] = x_min[k]
        end
    end

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
                                print_level = print_level,
                                other_param_seed = seed_upper)
    # Invalid-side encoding: ±σ_HESSE placeholder, matching C++
    # MinosError::Upper/Lower (MinosError.h:54) which return
    # `±State().Error(Parameter())` when the crossing search did not
    # converge. iminuit propagates this through `m.merrors[name].upper`
    # / `.lower`, so NativeMinuit's UX is now numerically interchangeable
    # with iminuit's published values regardless of `_valid` flags.
    # Consumers MUST gate on `e.upper_valid`/`e.lower_valid` to
    # distinguish a real crossing from the placeholder — sign and
    # magnitude alone don't.
    upper = up_cross.valid ? up_cross.aopt * sigma_i : sigma_i
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
                                print_level = print_level,
                                other_param_seed = seed_lower)
    lower = lo_cross.valid ? -lo_cross.aopt * sigma_i : -sigma_i
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
