# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# function_cross.jl — MnFunctionCross (Phase 1 first cut).
#
# Mirrors reference/Minuit2_cpp/src/MnFunctionCross.cxx:25-512.
#
# Given a converged minimum (state, fmin), a parameter index i, and a
# scan direction, find the value `α` such that:
#
#     min_{x_{-i}} f(x_i = x_min_i + α·step_i, x_{-i}) = fmin + up
#
# where `up` is the ErrorDef (1.0 for χ², 0.5 for NLL) and step_i is
# a step in parameter i (positive or negative). The minimization at
# each α is over all OTHER parameters with x_i FIXED.
#
# The algorithm is a parabolic root-find with up to 15 inner-MIGRAD
# iterations:
#
#   1. Initial MIGRAD with x_i = x_min_i + step_i (α = 1).
#   2. Quadratic estimate of α at aim: `√(up/(f - fmin)) - 1`.
#   3. Iterate: MIGRAD at new α, parabolic update, until either
#      (a) `|f - aim| < tlf AND |Δα| < tla` → converged,
#      (b) iteration cap or call cap hit,
#      (c) new lower minimum discovered,
#      (d) (Phase 1+) parameter bound hit.
#
# Phase 1 first cut: NO BOUNDS (the bounded path requires the
# Parameters-aware MIGRAD wiring in `migrad.jl` D3 follow-up). The
# `par_limit` flag is reserved but not raised here.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MnCross

Result of `function_cross`. Mirrors C++ `MnCross`
(`reference/Minuit2_cpp/inc/Minuit2/MnCross.h`).

# Fields

- `state::MinimumState` — the state at the crossing (or current best
  if invalid).
- `aopt::Float64` — the step multiplier at the crossing; `NaN` if
  invalid.
- `nfcn::Int` — cumulative FCN calls made by `function_cross`.
- `valid::Bool` — `true` if a crossing was found within tolerance.
- `new_min::Bool` — `true` if a lower minimum was discovered during the
  scan (Phase 1+ should restart MIGRAD here).
- `fcn_limit::Bool` — `true` if the call budget was exhausted.
- `par_limit::Bool` — `true` if a parameter bound was hit (Phase 1+
  only; always `false` in first cut).
"""
struct MnCross
    state::MinimumState
    aopt::Float64
    nfcn::Int
    valid::Bool
    new_min::Bool
    fcn_limit::Bool
    par_limit::Bool
end

MnCross(state::MinimumState, aopt::Real, nfcn::Integer; valid=true,
         new_min=false, fcn_limit=false, par_limit=false) =
    MnCross(state, Float64(aopt), Int(nfcn), valid, new_min,
            fcn_limit, par_limit)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: wrap a user FCN with one parameter fixed at a value.
# Returns a new (n-1)-dim CostFunction.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _fix_one_param(cf::CostFunction, i::Int, v::Float64, n::Int) -> CostFunction

Build an (n-1)-dim `CostFunction` from `cf` (an n-dim FCN) by fixing
the i-th argument to `v`. The returned CostFunction's call counter is
fresh; counts accrued in it must be added back to the outer counter.

Implementation: closure captures `cf.f`, `i`, `v`. Each call assembles
a temporary n-vector by splicing. Phase 1 first cut accepts the per-
call alloc; Phase 1.x can ship a workspace-passing variant.
"""
function _fix_one_param(cf::CostFunction, i::Integer, v::Float64, n::Integer)
    f = cf.f
    up = cf.up
    i_ = Int(i)
    n_ = Int(n)
    wrapped = function (y::AbstractVector{<:Real})
        full = Vector{Float64}(undef, n_)
        @inbounds for k in 1:(i_ - 1)
            full[k] = y[k]
        end
        full[i_] = v
        @inbounds for k in (i_ + 1):n_
            full[k] = y[k - 1]
        end
        return f(full)
    end
    return CostFunction(wrapped, up)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run inner MIGRAD with parameter i fixed at value v.
# Returns (inner_min, total_inner_nfcn).
# ─────────────────────────────────────────────────────────────────────────────

function _migrad_with_fixed(
    cf::CostFunction, state::MinimumState, i::Integer, v::Float64;
    tol::Float64, maxcalls::Integer, prec::MachinePrecision,
)
    n = length(state.parameters)
    # Build initial point + errors with parameter i removed
    x_min = state.parameters.x
    y0 = Vector{Float64}(undef, n - 1)
    @inbounds for k in 1:(i - 1)
        y0[k] = x_min[k]
    end
    @inbounds for k in (i + 1):n
        y0[k - 1] = x_min[k]
    end
    # Initial step sizes from the diagonal of inv_hessian (2·up·V[i,i] = σ²)
    errs = Vector{Float64}(undef, n - 1)
    V = state.error.inv_hessian
    @inbounds for k in 1:(i - 1)
        errs[k] = sqrt(max(abs(V[k, k]), prec.eps2))
    end
    @inbounds for k in (i + 1):n
        errs[k - 1] = sqrt(max(abs(V[k, k]), prec.eps2))
    end

    cf_fixed = _fix_one_param(cf, i, v, n)
    inner_min = migrad(cf_fixed, y0, errs;
                        tol = tol, maxfcn = Int(maxcalls), prec = prec)
    return inner_min, ncalls(cf_fixed)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main: function_cross — find the alpha such that min_{x_{-i}}(f) = fmin + up.
# ─────────────────────────────────────────────────────────────────────────────

"""
    function_cross(fmin, cf, par_idx, dir; tlr=0.1, maxcalls=1000,
                   strategy=Strategy(0), prec=MachinePrecision()) -> MnCross

Find the step multiplier α along parameter `par_idx` such that the
constrained-minimum (other params re-optimized) satisfies
`f - fmin = up`. Used by MINOS (asymmetric errors) and contours.

# Arguments

- `fmin::FunctionMinimum` — the converged MIGRAD result.
- `cf::CostFunction` — the user FCN (must match the one used for fmin).
- `par_idx::Integer` — 1-based parameter index to scan along.
- `dir::Real` — sign of the scan direction (+1.0 for upper error, -1.0
  for lower). Combined with the 1-sigma step from `state.error`.

# Keyword arguments

- `tlr::Real=0.1` — tolerance. Internal tolerances `tlf = tlr·up`
  and `tla = tlr` mirror C++ MnFunctionCross.cxx:42-44.
- `maxcalls::Integer=1000` — call budget across all inner MIGRADs.
- `strategy::Strategy=Strategy(0)` — passed to inner MIGRADs.
- `prec::MachinePrecision`.

# Returns

[`MnCross`](@ref). Check `.valid`, `.new_min`, `.fcn_limit` to interpret.

# Phase 1 first cut limitations

- No parameter bounds (par_limit always `false`).
- Inner MIGRAD does not propagate Strategy ≥ 1 HESSE refinement
  (cf. hesse.jl C8 deferral).
"""
function function_cross(
    fmin::FunctionMinimum,
    cf::CostFunction,
    par_idx::Integer,
    dir::Real;
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
        throw(ArgumentError("function_cross requires n > 1 (cannot fix the only parameter)"))

    x_min = state.parameters.x
    fmin_val = state.parameters.fval
    up = cf.up

    # 1-sigma external step along par_idx (Phase 1 first cut: assume
    # no bounds → internal == external; sigma = sqrt(2·up·V[i,i])).
    sigma_i = sqrt(max(2.0 * up * state.error.inv_hessian[par_idx, par_idx],
                        prec.eps2))
    step = Float64(dir) * sigma_i
    x_pivot = x_min[par_idx]

    # Tolerances (C++ MnFunctionCross.cxx:42-46)
    aim = fmin_val + up
    tlf = tlr * up
    tla = tlr
    maxitr = 15
    nfcn = 0

    # ── Probe 1: α = 1.0 ──────────────────────────────────────────
    v1 = x_pivot + 1.0 * step
    min1, nf1 = _migrad_with_fixed(cf, state, par_idx, v1;
                                    tol = 0.5 * tlr, maxcalls = maxcalls,
                                    prec = prec)
    nfcn += nf1

    if fval(min1) < fmin_val - tlf
        return MnCross(min1.state, NaN, nfcn; valid=false, new_min=true)
    end
    if min1.reached_call_limit
        return MnCross(min1.state, NaN, nfcn; valid=false, fcn_limit=true)
    end
    if !min1.is_valid
        return MnCross(state, NaN, nfcn; valid=false)
    end

    # Track 3 (α, f) points
    a = [0.0, 1.0, 0.0]
    f = [fmin_val, max(fval(min1), fmin_val + 0.1 * up), 0.0]
    ipt = 1   # number of inner-MIGRAD probes done so far (1)

    aopt = sqrt(up / (f[2] - fmin_val)) - 1.0
    if abs(f[2] - aim) < tlf
        return MnCross(min1.state, 1.0, nfcn; valid=true)
    end
    aopt = clamp(aopt, -0.5, 1.0)

    # ── Probe 2: α = aopt ─────────────────────────────────────────
    v2 = x_pivot + aopt * step
    min2, nf2 = _migrad_with_fixed(cf, state, par_idx, v2;
                                    tol = 0.5 * tlr, maxcalls = maxcalls - nfcn,
                                    prec = prec)
    nfcn += nf2

    if fval(min2) < fmin_val - tlf
        return MnCross(min2.state, NaN, nfcn; valid=false, new_min=true)
    end
    if min2.reached_call_limit
        return MnCross(min2.state, NaN, nfcn; valid=false, fcn_limit=true)
    end
    if !min2.is_valid
        return MnCross(state, NaN, nfcn; valid=false)
    end

    ipt = 2
    a[2] = aopt
    f[2] = fval(min2)
    dfda = (f[2] - f[1]) / (a[2] - a[1])

    # If slope wrong, extend α outward
    last_min = min2
    while dfda < 0 && ipt < maxitr
        a[1] = a[2]
        f[1] = f[2]
        aopt = a[1] + 0.2 * (ipt - 1)
        v = x_pivot + aopt * step
        m, nf = _migrad_with_fixed(cf, state, par_idx, v;
                                    tol = 0.5 * tlr, maxcalls = maxcalls - nfcn,
                                    prec = prec)
        nfcn += nf
        if fval(m) < fmin_val - tlf
            return MnCross(m.state, NaN, nfcn; valid=false, new_min=true)
        end
        if m.reached_call_limit
            return MnCross(m.state, NaN, nfcn; valid=false, fcn_limit=true)
        end
        if !m.is_valid
            return MnCross(state, NaN, nfcn; valid=false)
        end
        ipt += 1
        a[2] = aopt
        f[2] = fval(m)
        dfda = (f[2] - f[1]) / (a[2] - a[1])
        last_min = m
        if dfda > 0
            break
        end
    end  # end while dfda < 0

    if ipt >= maxitr && dfda <= 0
        return MnCross(state, NaN, nfcn; valid=false)
    end

    # ── Two-point linear extrapolation, then iterate up to maxitr ──
    while ipt < maxitr
        aopt = a[2] + (aim - f[2]) / dfda

        # Convergence check (C++ lines 252-258)
        fdist = min(abs(aim - f[1]), abs(aim - f[2]))
        adist = min(abs(aopt - a[1]), abs(aopt - a[2]))
        tla_loop = abs(aopt) > 1.0 ? tlr * abs(aopt) : tlr
        if adist < tla_loop && fdist < tlf
            return MnCross(last_min.state, aopt, nfcn; valid=true)
        end

        # Clamp aopt to extended bracket (C++ lines 261-266)
        bmin = min(a[1], a[2]) - 1.0
        bmax = max(a[1], a[2]) + 1.0
        aopt = clamp(aopt, bmin, bmax)

        v = x_pivot + aopt * step
        m, nf = _migrad_with_fixed(cf, state, par_idx, v;
                                    tol = 0.5 * tlr, maxcalls = maxcalls - nfcn,
                                    prec = prec)
        nfcn += nf
        if fval(m) < fmin_val - tlf
            return MnCross(m.state, NaN, nfcn; valid=false, new_min=true)
        end
        if m.reached_call_limit
            return MnCross(m.state, NaN, nfcn; valid=false, fcn_limit=true)
        end
        if !m.is_valid
            return MnCross(state, NaN, nfcn; valid=false)
        end

        ipt += 1
        # Replace the worst of (a[1], a[2]) by the new point so we
        # keep the bracket near `aim` (cf. C++ 3-point tracking).
        f3 = fval(m)
        _ = f3  # silence "unused" if optimizer warns
        if abs(f[1] - aim) > abs(f[2] - aim)
            a[1] = aopt
            f[1] = f3
        else
            a[2] = aopt
            f[2] = f3
        end
        # Maintain a[1] < a[2] ordering for dfda sign-correctness
        if a[1] > a[2]
            a[1], a[2] = a[2], a[1]
            f[1], f[2] = f[2], f[1]
        end
        dfda = (f[2] - f[1]) / (a[2] - a[1])
        if dfda <= 0
            return MnCross(m.state, NaN, nfcn; valid=false)
        end
        last_min = m
    end

    # Did not converge within maxitr
    return MnCross(last_min.state, aopt, nfcn; valid=false)
end
