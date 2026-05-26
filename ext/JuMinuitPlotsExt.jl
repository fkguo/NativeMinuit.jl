# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# JuMinuitPlotsExt — Plots.jl-based visualization helpers.
#
# Activated automatically when the user has `using Plots` loaded
# alongside `using JuMinuit`. Provides Julia equivalents of IMinuit.jl's
# `draw_contour`, `draw_mncontour`, `draw_profile`, `draw_mnprofile`,
# `draw_mnmatrix` (which depend on Python matplotlib).
#
# All these functions return Plots.Plot objects. The JuMinuit core
# package declares the function names via `function draw_X end` stubs
# in `iminuit_compat.jl`; this extension adds the methods on Minuit.
# ─────────────────────────────────────────────────────────────────────────────

module JuMinuitPlotsExt

using JuMinuit
using Plots

"""
    draw_contour(m::Minuit, par1, par2; bins=50, kws...) -> Plots.Plot

Plot the 2D contour from `contour(m, par1, par2; npoints=bins)`. The
`kws...` flow through to `Plots.plot(...)`.
"""
function JuMinuit.draw_contour(m::JuMinuit.Minuit, par1, par2;
                                 bins::Integer = 50, kws...)
    ce = JuMinuit.contour(m, par1, par2; npoints = bins)
    return Plots.plot(ce; kws...)
end

"""
    draw_mncontour(m::Minuit, par1, par2; numpoints=100, nsigma=1, kws...) -> Plots.Plot

Plot the MINOS 2D contour using `numpoints` boundary points.
"""
function JuMinuit.draw_mncontour(m::JuMinuit.Minuit, par1, par2;
                                  numpoints::Integer = 100,
                                  nsigma::Real = 1,
                                  kws...)
    isapprox(nsigma, 1.0) ||
        throw(ArgumentError("draw_mncontour nsigma ≠ 1 is Phase 1.x deferred"))
    ce = JuMinuit.contour(m, par1, par2; npoints = numpoints)
    return Plots.plot(ce; kws...)
end

"""
    draw_profile(m::Minuit, par; bins=100, low=0, high=0, kws...) -> Plots.Plot

Plot the 1D scan from `profile(m, par)`. No inner minimization (use
`draw_mnprofile` for the MINOS-style profile).
"""
function JuMinuit.draw_profile(m::JuMinuit.Minuit, par;
                                bins::Integer = 100,
                                low::Real = 0, high::Real = 0,
                                kws...)
    pts = JuMinuit.profile(m, par; bins = bins, low = low, high = high)
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
function JuMinuit.draw_mnprofile(m::JuMinuit.Minuit, par;
                                  bins::Integer = 30,
                                  low::Real = 0, high::Real = 0,
                                  kws...)
    pts = JuMinuit.mnprofile(m, par; bins = bins, low = low, high = high)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return Plots.plot(xs, ys;
                       xlabel = string("par ", par),
                       ylabel = "min fval",
                       label  = "MINOS profile", kws...)
end

"""
    draw_mnmatrix(m::Minuit; numpoints=100, kws...) -> Plots.Plot

Triangular matrix of all pairwise 2D contours. Each free parameter pair
gets one sub-plot; the diagonal shows the 1D MINOS profile. Mirrors
iminuit's `m.draw_mnmatrix()`.
"""
function JuMinuit.draw_mnmatrix(m::JuMinuit.Minuit;
                                 numpoints::Integer = 100,
                                 kws...)
    n = JuMinuit.n_pars(m.params)
    free_idx = [i for i in 1:n if !JuMinuit.is_fixed(m.params.pars[i])]
    k = length(free_idx)
    k >= 2 ||
        throw(ArgumentError("draw_mnmatrix needs ≥ 2 free parameters (got $k)"))

    plots = Plots.Plot[]
    for ii in 1:k, jj in 1:k
        i, j = free_idx[ii], free_idx[jj]
        if i == j
            pts = JuMinuit.mnprofile(m, i; bins = 30)
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            push!(plots, Plots.plot(xs, ys;
                                     title = m.params.pars[i].name,
                                     legend = false,
                                     ticks = nothing))
        elseif ii > jj
            ce = JuMinuit.contour(m, i, j; npoints = numpoints)
            push!(plots, Plots.plot(ce; legend = false, ticks = nothing))
        else
            push!(plots, Plots.plot(; legend = false, framestyle = :none))
        end
    end
    return Plots.plot(plots...; layout = (k, k), kws...)
end

end # module JuMinuitPlotsExt
