# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# contours.jl — 1σ contours in 2-parameter projection.
#
# Mirrors reference/Minuit2_cpp/src/MnContours.cxx but in a SIMPLIFIED
# form. The C++ algorithm uses multi-parameter MnFunctionCross — fixing
# TWO parameters at a midpoint + ray direction, re-MIGRAD the rest, and
# parabolic-root-finding the boundary along the ray. Phase 1 first cut
# implements an ellipse approximation derived from MINOS asymmetric
# errors + the inverse-Hessian off-diagonal covariance.
#
# For symmetric quadratic FCNs near the minimum (well-conditioned
# fits), the ellipse is the exact 1σ contour. For non-quadratic FCNs
# (Rosenbrock-like) the ellipse is an approximation; the true contour
# requires the full MnContours port (Phase 1.x — see DEFERRED.md).
# ─────────────────────────────────────────────────────────────────────────────

"""
    ContoursError

Result of `contour`. Mirrors C++ `ContoursError`.

# Fields

- `par_x::Int`, `par_y::Int` — 1-based parameter indices.
- `points::Vector{Tuple{Float64,Float64}}` — the contour boundary in
  (x, y) external coordinates.
- `minos_x::MinosError`, `minos_y::MinosError` — the MINOS errors
  along each axis (used as the basis of the ellipse).
- `nfcn::Int` — total FCN calls.
- `valid::Bool` — true if MINOS succeeded on both axes.
"""
struct ContoursError
    par_x::Int
    par_y::Int
    points::Vector{Tuple{Float64,Float64}}
    minos_x::MinosError
    minos_y::MinosError
    nfcn::Int
    valid::Bool
end

"""
    contour(fmin, cf, par_x, par_y; npoints=20, kwargs...) -> ContoursError

Compute a 1σ contour in the (par_x, par_y) plane. **Phase 1 first cut:
ellipse approximation** from MINOS errors + off-diagonal covariance.

# Algorithm (Phase 1 first cut)

1. Run MINOS on par_x and par_y to get asymmetric errors
   (e_x_lo, e_x_up), (e_y_lo, e_y_up).
2. Get off-diagonal covariance `c_xy = 2·up·V[px,py]` from
   `state.error.inv_hessian`.
3. Parametrize the contour as a (possibly asymmetric) ellipse. For
   each angle θ ∈ [0, 2π):
     - `r_x = e_x_up if cos(θ) ≥ 0 else -e_x_lo`
     - `r_y = e_y_up if sin(θ) ≥ 0 else -e_y_lo`
     - The correlation factor adjusts the ellipse axis ratio:
       `ρ = c_xy / sqrt(σ_x · σ_y)`, scaled by the appropriate radii.

For a symmetric quadratic, this reduces to the exact ellipse from
the inverse Hessian. For non-quadratic FCNs it's a fast first-pass
approximation; the true contour curve requires the multi-parameter
MnFunctionCross root-find (deferred to Phase 1.x).

# Arguments

- `fmin::FunctionMinimum`, `cf::CostFunction` — converged MIGRAD result.
- `par_x::Integer`, `par_y::Integer` — 1-based parameter indices.

# Keyword arguments

- `npoints::Integer=20` — number of contour points.
- Other kwargs forwarded to `minos` (tlr, maxcalls, strategy, prec).

# Returns

[`ContoursError`](@ref). `.points` is a Vector{Tuple{Float64,Float64}}
in external coords, length `npoints`.
"""
function contour(
    fmin::FunctionMinimum,
    cf::CostFunction,
    par_x::Integer,
    par_y::Integer;
    npoints::Integer = 20,
    kwargs...,
)
    state = fmin.state
    n = length(state.parameters)
    1 <= par_x <= n ||
        throw(ArgumentError("par_x $par_x out of bounds for n=$n"))
    1 <= par_y <= n ||
        throw(ArgumentError("par_y $par_y out of bounds for n=$n"))
    par_x != par_y ||
        throw(ArgumentError("contour requires par_x ≠ par_y (got $par_x)"))
    npoints >= 4 ||
        throw(ArgumentError("contour requires npoints ≥ 4 (got $npoints)"))

    valx = state.parameters.x[par_x]
    valy = state.parameters.x[par_y]

    # MINOS on both axes
    mex = minos(fmin, cf, par_x; kwargs...)
    mey = minos(fmin, cf, par_y; kwargs...)
    nfcn = mex.nfcn + mey.nfcn

    if !is_valid(mex) || !is_valid(mey)
        return ContoursError(
            Int(par_x), Int(par_y),
            Tuple{Float64,Float64}[],
            mex, mey, nfcn, false,
        )
    end

    # Asymmetric semi-axes (lower returned as negative by MINOS)
    e_x_up = mex.upper      # > 0
    e_x_lo = -mex.lower     # > 0 (mex.lower < 0)
    e_y_up = mey.upper
    e_y_lo = -mey.lower

    # Correlation factor (Pearson ρ) from covariance scaled by σ_x·σ_y.
    # Use the geometric-mean σ for asymmetric ellipses.
    σ_x = 0.5 * (e_x_up + e_x_lo)
    σ_y = 0.5 * (e_y_up + e_y_lo)
    cov_xy = 2.0 * cf.up * state.error.inv_hessian[par_x, par_y]
    ρ = (σ_x > 0 && σ_y > 0) ? cov_xy / (σ_x * σ_y) : 0.0
    # Clamp ρ to (-1, 1) to keep the ellipse non-degenerate
    if ρ > 1.0
        ρ = 1.0 - eps(Float64)
    elseif ρ < -1.0
        ρ = -1.0 + eps(Float64)
    end

    sqrt_term = sqrt(max(0.0, 1.0 - ρ * ρ))

    points = Vector{Tuple{Float64,Float64}}(undef, npoints)
    @inbounds for k in 1:npoints
        θ = 2π * (k - 1) / npoints
        cos_θ = cos(θ)
        sin_θ = sin(θ)
        # Correlated bivariate ellipse displacement directions. For
        # ρ=0 this reduces to axis-aligned (sign(dy_unit) = sign(sin_θ));
        # for ρ ≠ 0 the y-displacement sign can flip relative to sin_θ,
        # so we MUST pick the asymmetric radius from the actual
        # displacement direction (parallel-review #4 C-2 blocking —
        # the v1 version selected `e_y` by `sign(sin_θ)` which gave
        # the wrong side when the correlation tilted the ellipse).
        dx_unit = cos_θ
        dy_unit = ρ * cos_θ + sqrt_term * sin_θ
        e_x = dx_unit >= 0 ? e_x_up : e_x_lo
        e_y = dy_unit >= 0 ? e_y_up : e_y_lo
        dx = e_x * dx_unit
        dy = e_y * dy_unit
        points[k] = (valx + dx, valy + dy)
    end

    return ContoursError(
        Int(par_x), Int(par_y),
        points,
        mex, mey,
        nfcn,
        true,
    )
end
