# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# scan.jl — MnScan + MnParameterScan port.
#
# Mirrors reference/Minuit2_cpp/src/MnScan.cxx and
# reference/Minuit2_cpp/src/MnParameterScan.cxx.
#
# 1D parameter scan: evaluates the FCN at `maxsteps` equally-spaced
# points along one parameter, holding others fixed at their current
# values. Returns the (x, f(x)) sequence. If a lower fval is found
# anywhere, the central best-fit is updated.
#
# Common uses:
# - Diagnostic visualization (does the χ² have a single minimum?
#   how steep is the well?)
# - Robustness probe before MIGRAD (sanity-check the model)
# - 1D profiles (which iminuit/IMinuit.jl also expose as `profile`)
# ─────────────────────────────────────────────────────────────────────────────

"""
    scan(cf::CostFunction, x0, errs, par_idx;
         maxsteps=41, low=0.0, high=0.0) ->
        Vector{Tuple{Float64,Float64}}

1D parameter scan. Evaluates `cf(x)` at `maxsteps` equally-spaced
points along parameter `par_idx`, holding other parameters fixed at
the values in `x0`.

If `low == high == 0`, the scan range defaults to `x0[par_idx] ± 2·errs[par_idx]`.

Returns a vector of `(parameter_value, fcn_value)` pairs. The first
entry is always the central point `(x0[par_idx], cf(x0))` — useful
for assessing how far the scan extends from the minimum. The remaining
`maxsteps` entries are the equally-spaced scan points.

Mirrors `MnParameterScan::operator()` in
`reference/Minuit2_cpp/src/MnParameterScan.cxx`.
"""
function scan(
    cf::CostFunction,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real},
    par_idx::Integer;
    maxsteps::Integer = 41,
    low::Real = 0.0,
    high::Real = 0.0,
)
    n = length(x0)
    1 <= par_idx <= n ||
        throw(ArgumentError("scan: par_idx $par_idx out of bounds for n=$n"))
    length(errs) == n ||
        throw(DimensionMismatch("scan: errs / x0 length mismatch"))
    maxsteps >= 2 ||
        throw(ArgumentError("scan: maxsteps must be ≥ 2 (got $maxsteps)"))

    params = collect(Float64, x0)
    central = params[par_idx]
    amin = cf(params)

    result = Vector{Tuple{Float64,Float64}}()
    sizehint!(result, maxsteps + 1)
    push!(result, (central, amin))

    low_f, high_f = Float64(low), Float64(high)
    if low_f > high_f
        return result
    end

    # Default range: ±2σ around current value
    if low_f == 0.0 && high_f == 0.0
        low_f  = central - 2.0 * abs(Float64(errs[par_idx]))
        high_f = central + 2.0 * abs(Float64(errs[par_idx]))
    end

    stp = (high_f - low_f) / Float64(maxsteps - 1)
    @inbounds for i in 0:(maxsteps - 1)
        params[par_idx] = low_f + Float64(i) * stp
        fval_i = cf(params)
        if fval_i < amin
            amin = fval_i
        end
        push!(result, (params[par_idx], fval_i))
    end
    return result
end

# Bare-function convenience
function scan(
    f::F,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real},
    par_idx::Integer;
    up::Real = 1.0,
    kwargs...,
) where {F}
    cf = CostFunction(f, up)
    return scan(cf, x0, errs, par_idx; kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bound-aware scan — keeps user FCN on external coords, clips scan
# range against any limits.
# ─────────────────────────────────────────────────────────────────────────────

"""
    scan(cf::CostFunction, params::Parameters, par_idx;
         maxsteps=41, low=0.0, high=0.0) -> Vector{Tuple{Float64,Float64}}

Bound-aware 1D scan. Operates in EXTERNAL coordinates (user FCN sees
physical values). If the scan range straddles a parameter bound, the
range is clipped to the bound. If `par_idx` is fixed, throws an error.
"""
function scan(
    cf::CostFunction,
    params::Parameters,
    par_idx::Integer;
    maxsteps::Integer = 41,
    low::Real = 0.0,
    high::Real = 0.0,
)
    1 <= par_idx <= n_pars(params) ||
        throw(ArgumentError("scan: par_idx $par_idx out of bounds"))
    par = params.pars[par_idx]
    is_fixed(par) &&
        throw(ArgumentError("Cannot scan fixed parameter `$par_idx`"))

    x0 = [p.value for p in params.pars]
    errs = [p.error for p in params.pars]

    low_f, high_f = Float64(low), Float64(high)

    # Default range derivation matches C++ MnParameterScan.cxx:43-53.
    # If user passes (0, 0) and the parameter has limits, fall back to
    # the limit endpoints before applying the ±2σ default.
    if low_f == 0.0 && high_f == 0.0
        if has_lower_limit(par) && has_upper_limit(par)
            low_f, high_f = par.lower, par.upper
        else
            low_f  = par.value - 2.0 * abs(par.error)
            high_f = par.value + 2.0 * abs(par.error)
        end
    end

    # Clip against any bound.
    if has_lower_limit(par)
        low_f = max(low_f, par.lower)
    end
    if has_upper_limit(par)
        high_f = min(high_f, par.upper)
    end

    return scan(cf, x0, errs, par_idx;
                 maxsteps = maxsteps, low = low_f, high = high_f)
end

# ─────────────────────────────────────────────────────────────────────────────
# Best-value retention support — build a covariance-less FunctionMinimum at a
# given point so a Minuit-level scan can leave the fit at the best grid point
# (C++ MnParameterScan retains the best parameter set,
# reference/Minuit2_cpp/inc/Minuit2/MnParameterScan.h:42-43; iminuit
# `m.scan()` has the same coarse-pre-minimizer semantics). Used by
# `scan(m::Minuit, ...)` in iminuit_compat.jl.
# ─────────────────────────────────────────────────────────────────────────────

# Build a covariance-less `BoundedFunctionMinimum` at `params`' current point
# with function value `fval`. No minimization or Hessian pass is run — this is
# the scan analogue of the bound-aware `simplex` tail (simplex.jl:428-463): the
# error matrix is flagged `available = false`, so `m.matrix` / `eigenvalues` /
# `global_cc` correctly return `nothing` (a scan produces no inverse Hessian).
function _point_function_minimum(cf::CostFunction, params::Parameters,
                                  fval::Real)
    n = n_free(params)
    int_vals = initial_int_values(params)
    int_errs = initial_int_errors(params)
    cf_internal = _wrap_fcn_internal_to_external(cf, params)

    par_state = MinimumParameters(int_vals, int_errs, Float64(fval))
    err_state = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U),
                              1.0, MnHesseFailed, false)
    grad_state = FunctionGradient(zeros(n), zeros(n), zeros(n))
    state = MinimumState(par_state, err_state, grad_state, 0.0, ncalls(cf))
    # `is_valid = true`: a scan always "succeeds" (it just evaluates the
    # grid), matching iminuit `m.scan()` which leaves `m.valid == true`. The
    # `available = false` flag on `state.error` is the authoritative
    # "no covariance" indicator.
    fmin_int = FunctionMinimum(state, state, cf.up; is_valid = true)

    n_total = n_pars(params)
    ext_values     = Vector{Float64}(undef, n_total)
    ext_errors_vec = zeros(Float64, n_total)
    @inbounds for ext_idx in 1:n_total
        par = params.pars[ext_idx]
        int_idx = params.int_of_ext[ext_idx]
        if int_idx == 0
            ext_values[ext_idx] = par.value
        else
            ext_values[ext_idx] = int_to_ext_value(params, int_idx,
                                                    int_vals[int_idx])
            # Scan computes no curvature — surface the user's initial step as
            # the nominal error (same fallback a covariance-less state uses).
            ext_errors_vec[ext_idx] = par.error
        end
    end
    return BoundedFunctionMinimum(fmin_int, params, ext_values,
                                   ext_errors_vec, nothing, cf_internal)
end
