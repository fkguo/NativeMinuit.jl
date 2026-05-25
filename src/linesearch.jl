# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# linesearch.jl — MnLineSearch parabolic line search.
#
# Mirrors reference/Minuit2_cpp/src/MnLineSearch.cxx:46-313.
#
# Algorithm: starting from x0 along a search direction `step`, find
# the minimum of f(x0 + slam·step) over slam. Uses two-point quadratic
# interpolation initially, then three-point parabolic fits to refine.
# Bounded by overal/undral envelope, capped at 12 iterations.
#
# Returns (slam_min, fval_min) — the relative step length and function
# value at the minimum.
#
# The `#ifdef USE_OTHER_LS` cubic + Brent variants in C++ are not
# ported (see ROADMAP §9 Deferred — disabled by default in C++ build).
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParabolaPoint(x, y)

A (slam, fval) tuple. Used as the return type of `line_search` and as
intermediate state during the parabolic interpolation.
"""
struct ParabolaPoint
    x::Float64
    y::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Parabola fitting helpers (replaces C++ MnParabola + MnParabolaFactory)
# ─────────────────────────────────────────────────────────────────────────────

# Fit parabola y = a·x² + b·x + c through three points.
# Returns (a, b, c) and the x of the minimum (-b/(2a)), or `nothing`
# for the minimum if a is negative (no minimum) or zero (degenerate).
function _parabola_3pt(p0::ParabolaPoint, p1::ParabolaPoint, p2::ParabolaPoint)
    x0, y0 = p0.x, p0.y
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y

    # Solve 3-point Lagrange-style for a, b, c
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    c = (x1 * x2 * (x1 - x2) * y0 +
         x2 * x0 * (x2 - x0) * y1 +
         x0 * x1 * (x0 - x1) * y2) / den
    return (a, b, c)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main line search
# ─────────────────────────────────────────────────────────────────────────────

"""
    line_search(cf, par, step, gdel, prec=MachinePrecision(), work_x=nothing)
        -> ParabolaPoint

Perform a line search from `par.x` along `step`, minimizing
`f(par.x + slam·step)` over slam. Returns the relative step length
and function value at the minimum found.

Mirrors `MnLineSearch::operator()` from
`reference/Minuit2_cpp/src/MnLineSearch.cxx:46-313`.

# Arguments

- `cf::CostFunction` — the user function (call counter increments
  ~12-18 times during a typical search).
- `par::MinimumParameters` — current point (provides `par.x` and
  `par.fval`).
- `step::AbstractVector{Float64}` — search direction. Length must equal
  `length(par.x)`.
- `gdel::Real` — `step · ∇f(par.x)` (directional derivative). Sign
  matters: line search expects `gdel < 0` (descent direction).
- `prec::MachinePrecision` — precision constants.
- `work_x::Union{Nothing,Vector{Float64}}` — preallocated workspace
  for FCN evaluation. Default `nothing` → allocates one.

# Constants (from C++ MnLineSearch.cxx:63-69)

| Name | Value | Purpose |
|---|---|---|
| `overal`  | 1000.0 | Upper bound on slam |
| `undral`  | -100.0 | Lower bound on slam |
| `toler`   | 0.05   | Tolerance for "close to last point" |
| `slambg`  | 5.0    | Max length of second step |
| `alpha`   | 2.0    | Search-bound expansion factor |
| `maxiter` | 12     | Cap on FCN calls inside the search |

# Performance

The inner loop reuses a single `work_x::Vector{Float64}` to assemble
`par.x + slam·step` for each FCN call. With a Float64-returning user
FCN, no heap allocation per iteration of the line search itself.
"""
function line_search(
    cf,                       # CostFunction OR CostFunctionWithGradient
    par::MinimumParameters,
    step::AbstractVector{Float64},
    gdel::Real,
    prec::MachinePrecision = MachinePrecision();
    work_x::Union{Nothing,Vector{Float64}} = nothing,
)
    n = length(par)
    length(step) == n ||
        throw(DimensionMismatch("step length $(length(step)) != par length $n"))

    # Workspace for x + slam·step
    if work_x === nothing
        work_x = similar(par.x)
    else
        length(work_x) == n ||
            throw(DimensionMismatch("work_x length $(length(work_x)) != par length $n"))
    end

    # Constants — match C++ MnLineSearch.cxx:63-69
    overal = 1000.0
    undral = -100.0
    toler = 0.05
    slambg = 5.0
    alpha = 2.0
    maxiter = 12
    niter = 1  # C++ counts from 1, increments per FCN call

    # ── slamin — minimum allowable step magnitude
    slamin = 0.0
    @inbounds for i in 1:n
        s = step[i]
        s == 0 && continue
        ratio = abs(par.x[i] / s)
        if slamin == 0 || ratio < slamin
            slamin = ratio
        end
    end
    if abs(slamin) < prec.eps
        slamin = prec.eps
    end
    slamin *= prec.eps2

    # ── Initial 2-point evaluation
    f0 = par.fval
    @inbounds @. work_x = par.x + step          # slam = 1
    f1 = cf(work_x)
    niter += 1

    fvmin = f0
    xvmin = 0.0
    if f1 < f0
        fvmin = f1
        xvmin = 1.0
    end

    toler8 = toler
    slamax = slambg
    flast = f1
    slam = 1.0

    p0 = ParabolaPoint(0.0, f0)
    p1 = ParabolaPoint(slam, flast)
    f2 = 0.0
    iterate_2pt = true

    # ── Quadratic 2-point iteration (C++ MnLineSearch.cxx:106-207)
    while iterate_2pt && niter < maxiter
        iterate_2pt = false

        # Quadratic fit through (0, f0), slope at 0 (= gdel), and (slam, flast)
        denom = 2.0 * (flast - f0 - gdel * slam) / (slam * slam)

        if denom != 0
            slam = -gdel / denom
        else
            denom = -0.1 * gdel
            slam = 1.0
        end

        if slam < 0
            slam = slamax
        end
        if slam > slamax
            slam = slamax
        end
        if slam < toler8
            slam = toler8
        end
        if slam < slamin
            return ParabolaPoint(xvmin, fvmin)
        end
        if abs(slam - 1.0) < toler8 && p1.y < p0.y
            return ParabolaPoint(xvmin, fvmin)
        end
        if abs(slam - 1.0) < toler8
            slam = 1.0 + toler8
        end

        @inbounds @. work_x = par.x + slam * step
        f2 = cf(work_x)
        niter += 1

        if f2 < fvmin
            fvmin = f2
            xvmin = slam
        end

        # If f0 ≈ fvmin (within Float64 precision), keep iterating
        if abs(p0.y - fvmin) < abs(fvmin) * prec.eps
            iterate_2pt = true
            flast = f2
            toler8 = toler * slam
            overal = slam - toler8
            slamax = overal
            p1 = ParabolaPoint(slam, flast)
        end
    end

    if niter >= maxiter
        return ParabolaPoint(xvmin, fvmin)
    end

    p2 = ParabolaPoint(slam, f2)

    # ── Three-point parabolic iteration (C++ MnLineSearch.cxx:219-309)
    while niter < maxiter
        slamax = max(slamax, alpha * abs(xvmin))
        a, b, c = _parabola_3pt(p0, p1, p2)

        if a < prec.eps2
            # Parabola coefficient too small — treat as locally linear
            slopem = 2.0 * a * xvmin + b
            slam = slopem < 0 ? xvmin + slamax : xvmin - slamax
        else
            # Parabola minimum at -b/(2a), clamped
            slam = -b / (2.0 * a)
            if slam > xvmin + slamax
                slam = xvmin + slamax
            end
            if slam < xvmin - slamax
                slam = xvmin - slamax
            end
        end

        if slam > 0
            if slam > overal
                slam = overal
            end
        else
            if slam < undral
                slam = undral
            end
        end

        # Inner cut-step loop — shrinks slam toward xvmin if f3 worse
        # than all three previous points.
        f3 = 0.0
        cut_step = true
        while cut_step && niter < maxiter
            cut_step = false
            toler9 = max(toler8, abs(toler8 * slam))
            if abs(p0.x - slam) < toler9 ||
               abs(p1.x - slam) < toler9 ||
               abs(p2.x - slam) < toler9
                return ParabolaPoint(xvmin, fvmin)
            end
            @inbounds @. work_x = par.x + slam * step
            f3 = cf(work_x)
            niter += 1

            # If f3 worse than all three, bisect toward xvmin
            if f3 > p0.y && f3 > p1.y && f3 > p2.y
                if slam > xvmin
                    overal = min(overal, slam - toler8)
                end
                if slam < xvmin
                    undral = max(undral, slam + toler8)
                end
                slam = 0.5 * (slam + xvmin)
                cut_step = true
            end
        end

        if niter >= maxiter
            return ParabolaPoint(xvmin, fvmin)
        end

        # Replace the worst of (p0, p1, p2) with the new point
        p3 = ParabolaPoint(slam, f3)
        if p0.y > p1.y && p0.y > p2.y
            p0 = p3
        elseif p1.y > p0.y && p1.y > p2.y
            p1 = p3
        else
            p2 = p3
        end

        if f3 < fvmin
            fvmin = f3
            xvmin = slam
        else
            if slam > xvmin
                overal = min(overal, slam - toler8)
            end
            if slam < xvmin
                undral = max(undral, slam + toler8)
            end
        end
    end

    return ParabolaPoint(xvmin, fvmin)
end
