# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# plot_text.jl — ASCII Cartesian-box renderer for 2D point sets.
#
# Mirrors reference/Minuit2_cpp/src/{MnPlot.cxx,mntplot.cxx,mnbins.cxx}.
# Provides the `mn_plot_text` helper promised in ROADMAP §9 ("Julia users
# get RecipesBase recipes in Phase 2.3 plus an `mn_plot_text` helper for
# terminal use"). Closes GAP_AUDIT.md item M2.
#
# Use when there is no GUI backend (headless CI, SSH session, terminal):
#
#     julia> println(mn_plot_text(contour(fmin, cf, 1, 2; npoints=24)))
# ─────────────────────────────────────────────────────────────────────────────

"""
    _mn_bins(a1, a2, naa) -> (bl, bh, nb, bwid)

Port of C++ `mnbins` (`reference/Minuit2_cpp/src/mnbins.cxx`). Given a
range `[a1, a2]` and a desired maximum bin count `naa`, returns
"reasonable" round bounds `(bl, bh)`, bin count `nb`, and bin width
`bwid` so that ticks fall on multiples of `bwid ∈ {2, 2.5, 5, 10} ·
10^k`.
"""
function _mn_bins(a1::Real, a2::Real, naa::Int)
    al = min(Float64(a1), Float64(a2))
    ah = max(Float64(a1), Float64(a2))
    al == ah && (ah = al + 1.0)

    na = max(naa - 1, 1)
    bwid = 0.0; nb = 0; bl = 0.0; bh = 0.0
    while true
        awid = (ah - al) / na
        # trunc-toward-zero matches C++ `int(...)`; mnbins then
        # conditionally decrements when awid ≤ 1 to keep
        # 1 ≤ sigfig = awid·10^(-log_) < 10.
        log_ = trunc(Int, log10(awid))
        awid <= 1.0 && (log_ -= 1)
        sigfig = awid * 10.0^(-log_)
        local sigrnd::Float64
        if sigfig > 5.0
            sigrnd = 1.0; log_ += 1
        elseif sigfig > 2.5
            sigrnd = 5.0
        elseif sigfig > 2.0
            sigrnd = 2.5
        else
            sigrnd = 2.0
        end
        bwid = sigrnd * 10.0^log_

        # Mirror C++ `int(alb); if (alb < 0) --lwid;` exactly. Julia's
        # `floor(Int, ...)` diverges on exact-negative-integer `alb`
        # (e.g. al=-1, bwid=0.25 → alb=-4: floor gives -4 but C++
        # produces -5, shifting the lower bound below `al` to stay
        # strictly-less even at exact multiples).
        alb_l = al / bwid
        lwid = trunc(Int, alb_l)
        alb_l < 0.0 && (lwid -= 1)
        bl = bwid * lwid

        alb_h = ah / bwid + 1.0
        kwid = trunc(Int, alb_h)
        alb_h < 0.0 && (kwid -= 1)
        bh = bwid * kwid
        nb = kwid - lwid

        if naa > 5
            (nb << 1) != naa && return bl, bh, nb, bwid
            na += 1
            continue
        end
        # naa ≤ 5 ("difficult case"): allow either the natural binning
        # or collapse to a single bin of doubled width.
        if naa > 1 || nb == 1
            return bl, bh, nb, bwid
        end
        bwid *= 2.0
        nb = 1
        return bl, bh, nb, bwid
    end
end

"""
    mn_plot_text(pts::AbstractVector{<:Tuple{Real,Real}};
                 width=60, height=20,
                 par_x="x", par_y="y",
                 x_center=nothing) -> String

Lower-level ASCII renderer for an arbitrary 2D point set (e.g. the
raw `Vector{Tuple{Float64,Float64}}` returned by `mncontour`). Returns
a multi-line `String` ready for `println` / `@info`.

The bounding box auto-scales to the data with a small margin and snaps
to round-number ticks via the Minuit2 `mnbins` heuristic. Cells are
single characters: `*` for a single point, `&` where two **differing**
characters collide (matches `mntplot.cxx` semantics — same-character
stamps coalesce silently). `x_center`, when supplied, marks the
minimum with `X`.

Width / height are hints — the actual grid size is the nearest "nice"
binning, so the rendered box may be a little larger than requested.

# Defensive behavior

- Empty `pts` → returns `"EMPTY plot — no points to render.\\n"`.
- Non-finite (NaN / ±Inf) points are silently skipped; if no finite
  points remain, behaves like the empty case.
"""
function mn_plot_text(pts::AbstractVector{<:Tuple{Real,Real}};
                       width::Integer = 60,
                       height::Integer = 20,
                       par_x::AbstractString = "x",
                       par_y::AbstractString = "y",
                       x_center::Union{Nothing,Tuple{Real,Real}} = nothing)
    io = IOBuffer()
    width = max(Int(width), 10)
    height = max(Int(height), 10)

    fpts = Tuple{Float64,Float64}[]
    sizehint!(fpts, length(pts))
    for p in pts
        x = Float64(p[1]); y = Float64(p[2])
        (isfinite(x) && isfinite(y)) && push!(fpts, (x, y))
    end
    if isempty(fpts)
        println(io, "EMPTY plot — no points to render.")
        return String(take!(io))
    end

    # Single-pass extrema (avoids two transient first./last. allocations).
    xmin = ymin = Inf; xmax = ymax = -Inf
    for (x, y) in fpts
        x < xmin && (xmin = x); x > xmax && (xmax = x)
        y < ymin && (ymin = y); y > ymax && (ymax = y)
    end
    # Mirror C++ mnplot: pad each axis by 0.1 % of the range.
    dx = (xmax - xmin) * 0.001
    dy = (ymax - ymin) * 0.001
    xmin -= dx; xmax += dx
    ymin -= dy; ymax += dy
    # Degenerate-axis fallback. Plain `+ 1.0` is too narrow at huge |x|
    # (1e300 + 1 == 1e300) and overly wide at tiny |x|, so scale by
    # max(1, |x|). Catches both "near-equal-after-padding" (e.g. single
    # point, or two points one ulp apart) and "tiny-but-not-zero" cases.
    if xmax <= xmin
        bump = max(1.0, abs(xmin)) * 1e-6
        xmax = xmin + bump
        xmax <= xmin && (xmax = nextfloat(xmin))
    end
    if ymax <= ymin
        bump = max(1.0, abs(ymin)) * 1e-6
        ymax = ymin + bump
        ymax <= ymin && (ymax = nextfloat(ymin))
    end

    xlo, xhi, nx, bwx = _mn_bins(xmin, xmax, width)
    ylo, yhi, ny, bwy = _mn_bins(ymin, ymax, height)

    grid = fill(' ', ny, nx)
    overprint = Ref(false)
    @inline function _stamp!(ix::Int, iy::Int, ch::Char)
        if 1 <= ix <= nx && 1 <= iy <= ny
            c = grid[iy, ix]
            if c == ' ' || c == ch
                grid[iy, ix] = ch
            else
                grid[iy, ix] = '&'
                overprint[] = true
            end
        end
    end

    # Grid indices: clamp to [1, nx] / [1, ny]. Without the clamp, a
    # point at exactly y = ylo maps to iy = ny + 1 (the (yhi - y)/bwy
    # division hits the right edge of the closed interval) and is
    # silently dropped — including the trivial 1-point degenerate case.
    @inline _gx(x) = clamp(floor(Int, (x - xlo) / bwx) + 1, 1, nx)
    @inline _gy(y) = clamp(floor(Int, (yhi - y) / bwy) + 1, 1, ny)

    for (x, y) in fpts
        _stamp!(_gx(x), _gy(y), '*')
    end
    if x_center !== nothing
        xc = Float64(x_center[1]); yc = Float64(x_center[2])
        if isfinite(xc) && isfinite(yc)
            _stamp!(_gx(xc), _gy(yc), 'X')
        end
    end

    # Header — parameter names, ranges, scale.
    println(io, " ", par_x, " (x) vs ", par_y, " (y)")
    @printf(io, " x ∈ [%.4g, %.4g]   Δx = %.3g / column\n", xlo, xhi, bwx)
    @printf(io, " y ∈ [%.4g, %.4g]   Δy = %.3g / row\n",    ylo, yhi, bwy)

    # Frame + body (Unicode box-drawing; matches src/minuit.jl pretty-print).
    println(io, "┌", "─"^nx, "┐")
    for i in 1:ny
        print(io, "│")
        for j in 1:nx
            print(io, grid[i, j])
        end
        println(io, "│")
    end
    println(io, "└", "─"^nx, "┘")

    # Legend.
    n = length(fpts)
    noun = n == 1 ? "point" : "points"
    if x_center === nothing
        @printf(io, " * = point   (%d %s)\n", n, noun)
        overprint[] && println(io, " & = overlap (two or more differing chars)")
    else
        legend = overprint[] ? "* = point, X = minimum, & = overlap" :
                                "* = point, X = minimum"
        @printf(io, " %s   (%d %s)\n", legend, n, noun)
    end
    return String(take!(io))
end

"""
    mn_plot_text(c::ContoursError; width=60, height=20) -> String

ASCII-render a 2-parameter MnContours result as a Cartesian box.
Returns a single multi-line `String` suitable for `println`. The
fitted minimum (from the embedded MINOS errors) is marked with `X`.

Invalid contours (`c.valid == false` or no points) render as an
empty-plot message rather than throwing.
"""
function mn_plot_text(c::ContoursError; width::Integer = 60,
                       height::Integer = 20)
    c.valid ||
        return "INVALID contour — fit did not converge (no points to plot).\n"
    par_x = "par[$(c.par_x)]"
    par_y = "par[$(c.par_y)]"
    center = (c.minos_x.min_par_value, c.minos_y.min_par_value)
    return mn_plot_text(c.points; width = width, height = height,
                         par_x = par_x, par_y = par_y, x_center = center)
end
