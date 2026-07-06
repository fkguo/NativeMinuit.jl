# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# NativeMinuitPlotsExt — Plots.jl-based visualization helpers.
#
# Activated automatically when the user has `using Plots` loaded
# alongside `using NativeMinuit`. Provides Julia equivalents of IMinuit.jl's
# `draw_contour`, `draw_mncontour`, `draw_profile`, `draw_mnprofile`,
# `draw_mnmatrix` (which depend on Python matplotlib).
#
# All these functions return Plots.Plot objects. The NativeMinuit core
# package declares the function names via `function draw_X end` stubs
# in `iminuit_compat.jl`; this extension adds the methods on Minuit.
# ─────────────────────────────────────────────────────────────────────────────

module NativeMinuitPlotsExt

using NativeMinuit
using Plots

"""
    draw_contour(m::Minuit, par1, par2; size=50, bound=2, bins=nothing, kws...) -> Plots.Plot

Filled-contour plot of the FCN **grid slice** from
`contour_grid(m, par1, par2; size, bound, subtract_min=true)` — iminuit's
`m.draw_contour`. A landscape view, NOT a confidence region (the grid
fixes the other parameters; see `contour_grid`). `bins` is accepted as a
legacy alias for `size`. The `kws...` flow through to `Plots.plot(...)`.
"""
function NativeMinuit.draw_contour(m::NativeMinuit.Minuit, par1, par2;
                                 size::Integer = 50,
                                 bound = 2,
                                 bins::Union{Integer,Nothing} = nothing,
                                 kws...)
    sz = bins === nothing ? size : Int(bins)
    g = NativeMinuit.contour_grid(m, par1, par2;
                               size = sz, bound = bound, subtract_min = true)
    return Plots.plot(g; kws...)
end

"""
    draw_mncontour(m::Minuit, par1, par2; numpoints=100, cl=nothing,
                   nsigma=nothing, kws...) -> Plots.Plot

Plot **exact** MINOS 2D confidence contours (`mncontour` — boundary
search with per-point re-minimization) at one or several confidence
levels. `cl` follows `mncontour`'s iminuit semantics (default → joint
2-D 68 % region; `0<cl<1` → that joint probability; `cl≥1` → nσ) and
may be a vector to overlay several contours (mirrors iminuit's
`draw_mncontour`). `nsigma` is a legacy alias for a scalar `cl`.
(≤ 0.4 this drew the fast `contour_ellipse` approximation at Δχ²=up
instead — fixed.)
"""
function NativeMinuit.draw_mncontour(m::NativeMinuit.Minuit, par1, par2;
                                  numpoints::Integer = 100,
                                  cl = nothing,
                                  nsigma::Union{Real,Nothing} = nothing,
                                  kws...)
    cls = cl === nothing ?
            (nsigma === nothing ? [0.68] : [Float64(nsigma)]) :
            (cl isa Real ? [Float64(cl)] : collect(Float64, cl))
    ix = par1 isa Integer ? Int(par1) : NativeMinuit.ext_index(m.params, String(par1))
    iy = par2 isa Integer ? Int(par2) : NativeMinuit.ext_index(m.params, String(par2))
    plt = Plots.plot(; xlabel = m.params.pars[ix].name,
                       ylabel = m.params.pars[iy].name, kws...)
    for c in cls
        pts = NativeMinuit.mncontour(m, par1, par2; numpoints = numpoints, cl = c)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        if !isempty(xs)         # close the boundary polygon
            push!(xs, xs[1])
            push!(ys, ys[1])
        end
        prob = c >= 1 ? NativeMinuit.chisq_cl(c^2, 1) : c    # nσ → probability
        Plots.plot!(plt, xs, ys;
                     label = string(round(100 * prob; digits = 1), "% CL"))
    end
    return plt
end

"""
    draw_profile(m::Minuit, par; bins=100, low=0, high=0, kws...) -> Plots.Plot

Plot the 1D scan from `profile(m, par)`. No inner minimization (use
`draw_mnprofile` for the MINOS-style profile).
"""
function NativeMinuit.draw_profile(m::NativeMinuit.Minuit, par;
                                bins::Integer = 100,
                                low::Real = 0, high::Real = 0,
                                kws...)
    pts = NativeMinuit.profile(m, par; bins = bins, low = low, high = high)
    # Drop the central probe (first element) — it's redundant with the
    # equally-spaced grid that follows.
    xs = [p[1] for p in pts[2:end]]
    ys = [p[2] for p in pts[2:end]]
    return Plots.plot(xs, ys;
                       xlabel = string("par ", par),
                       ylabel = "fval",
                       label  = "profile", kws...)
end

"""
    draw_mnprofile(m::Minuit, par; bins=30, low=0, high=0, kws...) -> Plots.Plot

Plot the 1D MINOS profile from `mnprofile(m, par)` (inner minimization
at each grid point).
"""
function NativeMinuit.draw_mnprofile(m::NativeMinuit.Minuit, par;
                                  bins::Integer = 30,
                                  low::Real = 0, high::Real = 0,
                                  kws...)
    pts = NativeMinuit.mnprofile(m, par; bins = bins, low = low, high = high)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return Plots.plot(xs, ys;
                       xlabel = string("par ", par),
                       ylabel = "min fval",
                       label  = "MINOS profile", kws...)
end

"""
    draw_mnmatrix(m::Minuit; numpoints=100, cl=nothing, kws...) -> Plots.Plot

Triangular matrix of all pairwise 2D contours. Each free parameter pair
gets one sub-plot; the diagonal shows the 1D MINOS profile. Mirrors
iminuit's `m.draw_mnmatrix()`. `cl` follows `mncontour`'s iminuit
semantics (default → joint 2-D 68 % region per pair).
"""
function NativeMinuit.draw_mnmatrix(m::NativeMinuit.Minuit;
                                 numpoints::Integer = 100,
                                 cl::Union{Real,Nothing} = nothing,
                                 kws...)
    n = NativeMinuit.n_pars(m.params)
    free_idx = [i for i in 1:n if !NativeMinuit.is_fixed(m.params.pars[i])]
    k = length(free_idx)
    k >= 2 ||
        throw(ArgumentError("draw_mnmatrix needs ≥ 2 free parameters (got $k)"))

    plots = Plots.Plot[]
    for ii in 1:k, jj in 1:k
        i, j = free_idx[ii], free_idx[jj]
        if i == j
            pts = NativeMinuit.mnprofile(m, i; bins = 30)
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            push!(plots, Plots.plot(xs, ys;
                                     title = m.params.pars[i].name,
                                     legend = false,
                                     ticks = nothing))
        elseif ii > jj
            # Exact MINOS boundary per pair (iminuit's draw_mnmatrix uses
            # mncontour; ≤ 0.4 this drew the ellipse approximation — fixed).
            pts = cl === nothing ?
                NativeMinuit.mncontour(m, i, j; numpoints = numpoints) :
                NativeMinuit.mncontour(m, i, j; numpoints = numpoints, cl = cl)
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            if !isempty(xs)
                push!(xs, xs[1])
                push!(ys, ys[1])
            end
            push!(plots, Plots.plot(xs, ys; legend = false, ticks = nothing))
        else
            push!(plots, Plots.plot(; legend = false, framestyle = :none))
        end
    end
    return Plots.plot(plots...; layout = (k, k), kws...)
end

end # module NativeMinuitPlotsExt
