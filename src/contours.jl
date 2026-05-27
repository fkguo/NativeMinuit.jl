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
approximation; the true contour curve (multi-parameter MnFunctionCross
root-find) is available as [`contour_exact`](@ref).

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
# ─────────────────────────────────────────────────────────────────────────────
# contour_exact — Phase 1.x exact (C++-equivalent) contour algorithm.
# Uses function_cross_multi to find each boundary point.
# ─────────────────────────────────────────────────────────────────────────────

"""
    contour_exact(fmin, cf, par_x, par_y; npoints=20, kwargs...) -> ContoursError

C++ Minuit2-equivalent contour algorithm. Replaces the Phase 1
ellipse approximation with the proper boundary search:

1. Compute the 4 MINOS axis crossings (±σ_x at y=y_min, ±σ_y at x=x_min).
2. For each subsequent point: find the longest gap in the current
   boundary, compute its midpoint + perpendicular ray, call
   `function_cross_multi` to find the actual boundary along that ray.
3. Insert the new boundary point in the correct position.

Mirrors `reference/Minuit2_cpp/src/MnContours.cxx:34-204`. Much more
accurate than the Phase 1 ellipse approximation for non-quadratic FCNs;
slower (each boundary point requires a separate inner MIGRAD chain).
"""
function contour_exact(
    fmin::FunctionMinimum,
    cf::CostFunction,
    par_x::Integer,
    par_y::Integer;
    npoints::Integer = 20,
    tlr::Real = 0.1,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    state = fmin.state
    n = length(state.parameters)
    1 <= par_x <= n && 1 <= par_y <= n && par_x != par_y ||
        throw(ArgumentError("invalid par_x/par_y: got $par_x, $par_y for n=$n"))
    npoints >= 4 ||
        throw(ArgumentError("contour_exact needs npoints ≥ 4"))

    valx = state.parameters.x[par_x]
    valy = state.parameters.x[par_y]
    par_idxs = [Int(par_x), Int(par_y)]

    # Phase D — pool one MigradScratch per distinct inner_dim used by
    # this driver. MINOS and the 4 axis points all run inner MIGRAD at
    # n-1 free pars (one outer par fixed); ray points later use n-2
    # (two outer pars fixed simultaneously). Allocating once per dim
    # eliminates ~15 vector + 3 matrix allocs per probe × hundreds of
    # probes per contour.
    scratch_nm1 = n >= 2 ? MigradScratch(n - 1) : nothing
    scratch_nm2 = n >= 3 ? MigradScratch(n - 2) : nothing

    # Step 1: MINOS on both axes
    mex = minos(fmin, cf, par_x; tlr=tlr, strategy=strategy, prec=prec,
                  scratch=scratch_nm1)
    mey = minos(fmin, cf, par_y; tlr=tlr, strategy=strategy, prec=prec,
                  scratch=scratch_nm1)
    nfcn = mex.nfcn + mey.nfcn
    (is_valid(mex) && is_valid(mey)) ||
        return ContoursError(Int(par_x), Int(par_y),
                              Tuple{Float64,Float64}[], mex, mey, nfcn, false)

    # Inner MIGRAD with each axis MINOS minimum gives the "other" coord
    # at the four axis crossings. For each: fix par_x at val ± σ, MIGRAD
    # the rest; the y-coord at that minimum is the contour-crossing.
    function _axis_point(par_fix::Int, v_fix::Float64, par_other::Int)
        m_axis, nf_axis = _migrad_with_multi_fixed(
            cf, state, [par_fix], [v_fix];
            tol = 0.5 * tlr, maxcalls = 1000,
            prec = prec, strategy = strategy,
            scratch = scratch_nm1)
        if !m_axis.is_valid
            return nothing, nf_axis
        end
        return Base.values(m_axis)[par_other == par_fix ? 1 : (par_other > par_fix ? par_other - 1 : par_other)],
               nf_axis
    end

    # The 4 initial axis points (in external coords)
    y_at_xlo, nf_xlo = _axis_point(Int(par_x), valx + mex.lower, Int(par_y))
    y_at_xhi, nf_xhi = _axis_point(Int(par_x), valx + mex.upper, Int(par_y))
    x_at_ylo, nf_ylo = _axis_point(Int(par_y), valy + mey.lower, Int(par_x))
    x_at_yhi, nf_yhi = _axis_point(Int(par_y), valy + mey.upper, Int(par_x))
    nfcn += nf_xlo + nf_xhi + nf_ylo + nf_yhi

    if any(p === nothing for p in (y_at_xlo, y_at_xhi, x_at_ylo, x_at_yhi))
        return ContoursError(Int(par_x), Int(par_y),
                              Tuple{Float64,Float64}[], mex, mey, nfcn, false)
    end

    # Order: (xlo, _), (_, ylo), (xhi, _), (_, yhi) — counter-clockwise
    points = Tuple{Float64,Float64}[
        (valx + mex.lower, y_at_xlo),
        (x_at_ylo, valy + mey.lower),
        (valx + mex.upper, y_at_xhi),
        (x_at_yhi, valy + mey.upper),
    ]

    # Scaling factors for distance comparison (per C++ MnContours.cxx:112-113)
    scalx = 1.0 / (mex.upper - mex.lower)
    scaly = 1.0 / (mey.upper - mey.lower)

    # Step 2: for each new point, find longest gap, compute perpendicular ray.
    maxcalls = 100 * (npoints + 5) * (n + 1)
    for _ in 5:npoints
        # Find longest chord
        nn = length(points)
        bigdis = 0.0
        idist1 = 1
        idist2 = 2
        # Pairs (k, k+1) cyclically with wrap-around at the end
        for k in 1:nn
            kp1 = k == nn ? 1 : k + 1
            dx = (points[k][1] - points[kp1][1]) * scalx
            dy = (points[k][2] - points[kp1][2]) * scaly
            dist = dx * dx + dy * dy
            if dist > bigdis
                bigdis = dist
                idist1 = k
                idist2 = kp1
            end
        end

        # Midpoint + perpendicular direction
        a1 = 0.5; a2 = 0.5
        xmid = a1 * points[idist1][1] + a2 * points[idist2][1]
        ymid = a1 * points[idist1][2] + a2 * points[idist2][2]
        xdir = points[idist2][2] - points[idist1][2]
        ydir = points[idist1][1] - points[idist2][1]
        scalfac = max(abs(xdir * scalx), abs(ydir * scaly))
        scalfac == 0 && break  # degenerate
        xdircr = xdir / scalfac
        ydircr = ydir / scalfac

        # Find the boundary along (xmid, ymid) + α · (xdircr, ydircr).
        # Pass scratch_nm2 — ray-point inner MIGRAD fixes BOTH outer
        # pars simultaneously (inner_dim = n - 2).
        cross = function_cross_multi(
            fmin, cf, par_idxs, [xmid, ymid], [xdircr, ydircr];
            tlr = tlr, maxcalls = max(maxcalls - nfcn, 100),
            strategy = strategy, prec = prec,
            scratch = scratch_nm2)
        nfcn += cross.nfcn

        if !cross.valid || nfcn > maxcalls
            # Stop adding points; return what we have so far
            break
        end

        aopt = cross.aopt
        new_x = xmid + aopt * xdircr
        new_y = ymid + aopt * ydircr
        # Insert at idist2 position so points stay in order
        insert!(points, idist2 == 1 ? nn + 1 : idist2, (new_x, new_y))
    end

    return ContoursError(Int(par_x), Int(par_y), points, mex, mey, nfcn, true)
end

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
