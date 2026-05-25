# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# plot_recipes.jl — Phase 2.3 RecipesBase recipes.
#
# Provides Plots.jl / Makie.jl-agnostic plot recipes for the visible
# result types:
#   - ContoursError → closed-polygon line plot in the (par_x, par_y) plane
#   - MinosError    → 1D error-bar marker at the central value with
#                     asymmetric whiskers
#   - FunctionMinimum / BoundedFunctionMinimum → parameter table
#                     visualization (axes = parameter index)
#
# Uses RecipesBase so users can `plot(c)` from either Plots.jl or
# Makie.jl without us depending on either.
# ─────────────────────────────────────────────────────────────────────────────

using RecipesBase

# Contour: closed polygon connecting boundary points + center marker.
RecipesBase.@recipe function f(c::ContoursError)
    seriestype := :path
    label --> "1σ contour"
    aspect_ratio --> :equal
    xguide --> "par[$(c.par_x)]"
    yguide --> "par[$(c.par_y)]"
    # Close the polygon
    pts = c.points
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    push!(xs, xs[1])
    push!(ys, ys[1])
    return xs, ys
end

# MinosError as an asymmetric error-bar at the central value.
# For a 1D plot: x = par_idx, y = min_par_value ± (upper / lower).
RecipesBase.@recipe function f(e::MinosError)
    seriestype := :scatter
    yerror --> ([abs(e.lower)], [e.upper])  # (lower_err, upper_err)
    markersize --> 5
    label --> "par[$(e.par_idx)] MINOS"
    xguide --> "parameter index"
    yguide --> "value"
    return [e.par_idx], [e.min_par_value]
end

# Vector of MinosError → multi-parameter error-bar plot.
RecipesBase.@recipe function f(errs::Vector{MinosError})
    seriestype := :scatter
    xs = [e.par_idx for e in errs]
    ys = [e.min_par_value for e in errs]
    lower_errs = [abs(e.lower) for e in errs]
    upper_errs = [e.upper for e in errs]
    yerror --> (lower_errs, upper_errs)
    markersize --> 5
    label --> "MINOS errors"
    xguide --> "parameter index"
    yguide --> "value"
    return xs, ys
end

# FunctionMinimum / BoundedFunctionMinimum: parameter table.
# Plots each parameter's value at its index with the symmetric Hesse
# error as a bar.
RecipesBase.@recipe function f(m::FunctionMinimum)
    n = length(m.state)
    xs = collect(1:n)
    ys = Base.values(m)
    cov = covariance(m)
    yerrs = if cov === nothing
        zeros(n)
    else
        [sqrt(max(cov[i, i], 0.0)) for i in 1:n]
    end
    seriestype := :scatter
    yerror --> yerrs
    markersize --> 5
    label --> "MIGRAD result"
    xguide --> "parameter index"
    yguide --> "value"
    return xs, ys
end

RecipesBase.@recipe function f(m::BoundedFunctionMinimum)
    n = n_pars(m.params)
    xs = collect(1:n)
    ys = m.ext_values
    yerrs = m.ext_errors
    seriestype := :scatter
    yerror --> yerrs
    markersize --> 5
    # Distinguish fixed parameters with a different marker
    fixed_mask = [is_fixed(p) for p in m.params.pars]
    marker_shapes = [fix ? :diamond : :circle for fix in fixed_mask]
    markershape --> marker_shapes
    xticks --> (1:n, [p.name for p in m.params.pars])
    label --> "MIGRAD result"
    xguide --> "parameter"
    yguide --> "value"
    return xs, ys
end

# Vector of (x, y) tuples returned by contour: helper for direct plotting.
RecipesBase.@recipe function f(::Type{Val{:juminuit_contour_points}},
                                pts::Vector{Tuple{Float64,Float64}})
    seriestype := :path
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    push!(xs, xs[1])
    push!(ys, ys[1])
    return xs, ys
end
