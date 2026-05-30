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
- `full_points::Vector{Vector{Float64}}` — the **full n-dim parameter
  vector** at each contour boundary point: the 2 contour coordinates
  (`par_x`, `par_y`) plus the n−2 profiled (re-minimized) values from
  the inner cross-search, in the same coordinate frame as `points`.
  Populated by [`contour_exact`](@ref) at **zero extra cost** (the inner
  re-minimization state is already computed per boundary point — see
  [`contour_parameter_sets`](@ref)). Empty for the ellipse
  approximation [`contour`](@ref) (which does no inner re-minimization).
"""
struct ContoursError
    par_x::Int
    par_y::Int
    points::Vector{Tuple{Float64,Float64}}
    minos_x::MinosError
    minos_y::MinosError
    nfcn::Int
    valid::Bool
    # Phase 2.x: full n-dim parameter vector at each boundary point
    # (2 fixed at the contour coords + n−2 profiled). Filled from the
    # inner cross-search states `contour_exact` already computes; NO
    # extra fits (the IMinuit.jl `get_contours` re-fit was a PyCall
    # round-trip limitation that native Julia does not have). See
    # `contour_parameter_sets`.
    full_points::Vector{Vector{Float64}}
end

# Backward-compatible constructor (callers predating `full_points`).
# Defaults `full_points` to empty — mirrors the `MinosError` state-field
# compat constructors. The ellipse `contour` and all invalid-return
# paths use this 7-arg form; only the valid `contour_exact` return fills
# `full_points`.
ContoursError(par_x::Int, par_y::Int, points::Vector{Tuple{Float64,Float64}},
              minos_x::MinosError, minos_y::MinosError, nfcn::Int, valid::Bool) =
    ContoursError(par_x, par_y, points, minos_x, minos_y, nfcn, valid,
                  Vector{Float64}[])

"""
    contour_parameter_sets(ce::ContoursError) -> Vector{Vector{Float64}}

The **full parameter set at every contour boundary point** — one
n-dimensional vector per point in `ce.points`, each holding the 2 contour
coordinates (`par_x`, `par_y`) together with the n−2 other parameters at
their profiled (re-minimized) values along the contour.

This is the native-Julia analogue of IMinuit.jl's `get_contours`, but
with **no extra fits**: [`contour_exact`](@ref) already runs an inner
re-minimization at each boundary point, so the full state is captured for
free (IMinuit.jl re-fit each point only because PyCall could not return
the inner `MinimumState` across the Python boundary).

Each vector is in the **same coordinate frame as `ce.points`** (internal
coordinates when reached via `mncontour` on a bounded fit; identical to
external for unbounded fits). Empty when `ce` came from the ellipse
[`contour`](@ref) or when the contour was invalid.

# Example

```julia
ce = contour_exact(fmin, cf, 1, 2; npoints = 24)
psets = contour_parameter_sets(ce)        # 24 full parameter vectors
# Re-evaluating the FCN at any set returns ≈ fmin + up (the boundary):
cf(psets[1]) ≈ fval(fmin) + cf.up
```
"""
contour_parameter_sets(ce::ContoursError) = ce.full_points

# ── full-point assembly helpers (Phase 2.x; used by contour_exact) ───────────
# Re-insert the fixed contour coordinate(s) into an inner cross-search
# free vector to recover the full n-dim parameter vector at a boundary
# point. The inner free vector is ordered by ASCENDING original parameter
# index with the fixed parameter(s) removed — see `_fix_multi_params` /
# `_migrad_with_multi_fixed` (src/function_cross.jl). Returns `Float64[]`
# on a shape mismatch (defensive: never scramble a wrong-dim state).

# One parameter fixed (axis points; inner free vector has n−1 entries).
function _insert_one_fixed(freex::AbstractVector{<:Real}, fixed_i::Int,
                           vfix::Float64, n::Int)
    length(freex) == n - 1 || return Float64[]
    out = Vector{Float64}(undef, n)
    @inbounds for k in 1:(fixed_i - 1)
        out[k] = Float64(freex[k])
    end
    @inbounds out[fixed_i] = vfix
    @inbounds for k in (fixed_i + 1):n
        out[k] = Float64(freex[k - 1])
    end
    return out
end

# Two parameters fixed (ray points; inner free vector has n−2 entries).
function _contour_full_point(inner_x::AbstractVector{<:Real}, ix::Int, iy::Int,
                             xval::Float64, yval::Float64, n::Int)
    if n == 2
        # No profiled parameters: the full point is exactly the two contour
        # coordinates at their slots. Use the AUTHORITATIVE (xval, yval) from
        # the converged ray multiplier — NOT the inner state's last-probe
        # coords, which lag by one parabolic α step.
        out = Vector{Float64}(undef, 2)
        @inbounds (out[ix] = xval; out[iy] = yval)
        return out
    end
    length(inner_x) == n - 2 || return Float64[]
    out = Vector{Float64}(undef, n)
    lo, hi = minmax(ix, iy)
    vlo = ix < iy ? xval : yval
    vhi = ix < iy ? yval : xval
    f = 1
    @inbounds for k in 1:n
        if k == lo
            out[k] = vlo
        elseif k == hi
            out[k] = vhi
        else
            out[k] = Float64(inner_x[f]); f += 1
        end
    end
    return out
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
    cf::AbstractCostFunction,
    par_x::Integer,
    par_y::Integer;
    npoints::Integer = 20,
    tlr::Real = 0.1,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
    threaded_gradient::Bool = false,
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
                  scratch=scratch_nm1, threaded_gradient=threaded_gradient)
    mey = minos(fmin, cf, par_y; tlr=tlr, strategy=strategy, prec=prec,
                  scratch=scratch_nm1, threaded_gradient=threaded_gradient)
    nfcn = mex.nfcn + mey.nfcn
    (is_valid(mex) && is_valid(mey)) ||
        return ContoursError(Int(par_x), Int(par_y),
                              Tuple{Float64,Float64}[], mex, mey, nfcn, false)

    # Inner MIGRAD with each axis MINOS minimum gives the "other" coord
    # at the four axis crossings. For each: fix par_x at val ± σ, MIGRAD
    # the rest; the y-coord at that minimum is the contour-crossing.
    # Returns `(other_coord, full_vec, nfcn)`. `full_vec` is the full
    # n-dim parameter vector at the axis crossing (par_fix at v_fix +
    # the n−1 re-minimized free coords); captured for `full_points` at
    # NO extra cost (the inner re-minimization already ran).
    function _axis_point(par_fix::Int, v_fix::Float64, par_other::Int)
        m_axis, nf_axis = _migrad_with_multi_fixed(
            cf, state, [par_fix], [v_fix];
            tol = 0.5 * tlr, maxcalls = 1000,
            prec = prec, strategy = strategy,
            scratch = scratch_nm1,
            threaded_gradient = threaded_gradient)
        if !m_axis.is_valid
            return nothing, Float64[], nf_axis
        end
        freevals = Base.values(m_axis)
        other = freevals[par_other == par_fix ? 1 : (par_other > par_fix ? par_other - 1 : par_other)]
        full = _insert_one_fixed(freevals, par_fix, v_fix, n)
        return other, full, nf_axis
    end

    # The 4 initial axis points (in external coords)
    y_at_xlo, full_xlo, nf_xlo = _axis_point(Int(par_x), valx + mex.lower, Int(par_y))
    y_at_xhi, full_xhi, nf_xhi = _axis_point(Int(par_x), valx + mex.upper, Int(par_y))
    x_at_ylo, full_ylo, nf_ylo = _axis_point(Int(par_y), valy + mey.lower, Int(par_x))
    x_at_yhi, full_yhi, nf_yhi = _axis_point(Int(par_y), valy + mey.upper, Int(par_x))
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
    # full_points stays index-aligned with `points` through every insert!.
    full_points = Vector{Float64}[full_xlo, full_ylo, full_xhi, full_yhi]

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
        basefac = max(abs(xdir * scalx), abs(ydir * scaly))
        basefac == 0 && break  # degenerate geometry — no perpendicular ray

        # C++ MnContours.cxx:152-189 — `sca` direction-switch retry. The
        # cross search runs along the perpendicular ray scaled by `sca`
        # (init +1, the outward normal for the CCW point order). If the
        # search FAILS, C++ flips `sca → −1` (reversing the ray via
        # `scalfac = sca·max(...)`, MnContours.cxx:167/188) and retries the
        # SAME point once (`goto L300`) before bailing ("unable to find
        # point on Contour ... found only N points"). This recovers points
        # on irregular / non-convex contours where the crossing lies along
        # the −perpendicular direction. The `sca = +1` first attempt is
        # byte-identical to the pre-retry code (`1.0·basefac === basefac`),
        # so well-behaved contours are unchanged.
        found = false
        for sca in (1.0, -1.0)
            scalfac = sca * basefac
            xdircr = xdir / scalfac
            ydircr = ydir / scalfac

            # Find the boundary along (xmid, ymid) + α · (xdircr, ydircr).
            # Pass scratch_nm2 — ray-point inner MIGRAD fixes BOTH outer
            # pars simultaneously (inner_dim = n - 2).
            cross = function_cross_multi(
                fmin, cf, par_idxs, [xmid, ymid], [xdircr, ydircr];
                tlr = tlr, maxcalls = max(maxcalls - nfcn, 100),
                strategy = strategy, prec = prec,
                scratch = scratch_nm2,
                threaded_gradient = threaded_gradient)
            nfcn += cross.nfcn

            # Genuine call-limit exit (C++ re-checks nfcn>maxcalls at L300).
            nfcn > maxcalls && break
            if cross.valid
                aopt = cross.aopt
                new_x = xmid + aopt * xdircr
                new_y = ymid + aopt * ydircr
                # Full n-dim parameter vector at this ray crossing: the 2
                # contour coords + the n−2 profiled inner-state coords (no
                # extra fit). Built from whichever `sca` direction converged.
                full_pt = _contour_full_point(cross.state.parameters.x,
                                              Int(par_x), Int(par_y), new_x, new_y, n)
                # Insert at idist2 position so points stay in order; keep
                # full_points index-aligned with the same insertion.
                ins = idist2 == 1 ? nn + 1 : idist2
                insert!(points, ins, (new_x, new_y))
                insert!(full_points, ins, full_pt)
                found = true
                break
            end
            # Search failed but budget remains → flip `sca` (loop continues
            # to −1.0) and retry the same point along the reversed ray.
        end
        # Both ray directions failed, or the call limit was hit → stop
        # adding points and return what we have (C++ "found only N points").
        found || break
    end

    return ContoursError(Int(par_x), Int(par_y), points, mex, mey, nfcn, true,
                         full_points)
end

function contour(
    fmin::FunctionMinimum,
    cf::AbstractCostFunction,
    par_x::Integer,
    par_y::Integer;
    npoints::Integer = 20,
    threaded_gradient::Bool = false,
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
    mex = minos(fmin, cf, par_x; threaded_gradient = threaded_gradient, kwargs...)
    mey = minos(fmin, cf, par_y; threaded_gradient = threaded_gradient, kwargs...)
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
