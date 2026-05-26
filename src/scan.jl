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
