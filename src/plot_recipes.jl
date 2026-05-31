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
# …plus the "beyond iminuit" error-analysis families (see the second block
# below):
#   - get_contours_samples output → 2D scatter of the accepted Monte-Carlo
#                     Δχ² sample cloud, coloured by per-sample Δχ²
#   - BootstrapResult / JackknifeResult → histogram of a parameter's
#                     resampled distribution + estimate / CI reference lines
#   - SolutionModes / SolutionMode → colour-per-mode scatter of the clustered
#                     samples (or bounding boxes), each mode's representative
#                     marked — a visual showcase of multimodal detection
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

# ─────────────────────────────────────────────────────────────────────────────
# Error-analysis recipes — the "beyond iminuit" sampling / resampling /
# multimodal outputs. These are RecipesBase-only (no Plots/Makie dependency).
#
# Parameter / parameter-pair selection is a recipe-only option read straight
# from `plotattributes` with `pop!` (so it is consumed before reaching any
# backend and never triggers an "unsupported attribute" warning):
#   - `vars`  — the (x, y) parameter pair for the 2D scatters; a 2-tuple/vector
#               of 1-based indices or parameter-name strings (default `(1, 2)`).
#   - `par`   — the single parameter for the bootstrap / jackknife histograms;
#               an index or name (default: the first free/varying parameter).
# (`pop!` is used rather than a recipe keyword argument on purpose: the macro's
#  keyword-cleanup path calls `RecipesBase.is_key_supported`, which has no
#  method until a backend like Plots is loaded — `pop!` keeps the recipes, and
#  their tests, fully backend-free.)
# ─────────────────────────────────────────────────────────────────────────────

# Resolve a parameter selector (1-based index, or a name String/Symbol) to a
# column index against `names`. Integer selectors pass through unchanged.
function _par_index(sel, names)
    sel isa Integer && return Int(sel)
    s = String(sel)
    idx = findfirst(==(s), names)
    idx === nothing &&
        throw(ArgumentError("parameter \"$s\" not found in $(names)"))
    return idx
end

# Resolve a `vars` option (a 2-tuple/vector of indices or names, or a single
# selector applied to both axes) into an ordered (i, j) column pair.
function _resolve_pair(vars, names)
    v = vars isa Union{Tuple,AbstractVector} ? collect(vars) : [vars]
    isempty(v) && throw(ArgumentError("`vars` must select at least one parameter"))
    i = _par_index(v[1], names)
    j = _par_index(length(v) >= 2 ? v[2] : v[1], names)
    return i, j
end

# Axis label for column k: the parameter name if available, else `par[k]`.
_name_or_idx(names, k) =
    (1 <= k <= length(names) && !isempty(String(names[k]))) ? String(names[k]) : "par[$k]"

# First column of a resample matrix with finite, non-degenerate spread (the
# "first free parameter" default for the bootstrap / jackknife histograms);
# falls back to column 1 if nothing varies (e.g. an all-fixed degenerate fit).
function _first_varying(samples::AbstractMatrix)
    nrow, npar = size(samples)
    @inbounds for j in 1:npar
        lo = Inf
        hi = -Inf
        for i in 1:nrow
            v = samples[i, j]
            isfinite(v) || continue
            v < lo && (lo = v)
            v > hi && (hi = v)
        end
        (isfinite(lo) && hi > lo) && return j
    end
    return 1
end

# Finite entries of column j (drops the NaN rows left by non-converged re-fits).
_finite_col(samples::AbstractMatrix, j::Integer) =
    [v for v in view(samples, :, j) if isfinite(v)]

# Mean of column `col` over the given rows (a cluster-centroid coordinate, used
# to mark a mode when the sample matrix is free-width so the full-vector
# `representative` cannot be indexed directly).
_centroid(samples::AbstractMatrix, rows, col::Integer) =
    isempty(rows) ? NaN : mean(view(samples, rows, col))

# Legend label for mode rank k (1 = the main / lowest-χ² solution).
_mode_label(k::Integer) = k == 1 ? "main" : "mode $k"

# Closed-polygon (xs, ys) for the axis-aligned rectangle [xlo,xhi]×[ylo,yhi].
function _rect_xy(xr::Tuple, yr::Tuple)
    xlo, xhi = xr
    ylo, yhi = yr
    return [xlo, xhi, xhi, xlo, xlo], [ylo, ylo, yhi, yhi, ylo]
end

# Field-name signature of the `get_contours_samples` return NamedTuple. The
# recipe dispatches on exactly this shape (names only — robust to the field
# value-types). MUST stay in lock-step with the `get_contours_samples` return
# in `error_sampling.jl`; the recipe test exercises the real output, so any
# drift surfaces as a recipe MethodError there.
const _CONTOURS_SAMPLES_FIELDS =
    (:samples, :bounds, :best, :names, :free_names, :delta_chisq_values,
     :n_accepted, :n_total, :acceptance, :widen_rounds, :inflate_final,
     :delta, :up, :cl, :ndof, :proposal, :under_coverage, :mahalanobis)

# get_contours_samples output → 2D scatter of the accepted Monte-Carlo Δχ²
# sample cloud for a chosen free-parameter pair (default the first two free
# parameters), coloured by each sample's true Δχ². Small, low-α markers convey
# density. A single-free-parameter fit degrades to (value, Δχ²).
RecipesBase.@recipe function f(s::NamedTuple{_CONTOURS_SAMPLES_FIELDS})
    vars = pop!(plotattributes, :vars, (1, 2))
    smp = s.samples
    nfree = size(smp, 2)
    fnames = s.free_names
    dchi = s.delta_chisq_values

    seriestype := :scatter
    markersize --> 2
    markeralpha --> 0.4
    markerstrokewidth --> 0
    label --> "Δχ² samples (n=$(s.n_accepted))"

    if nfree == 1
        # One free parameter: there is no pair to scatter — show the accepted
        # values against their Δχ² instead (an accepted-region profile).
        xguide --> _name_or_idx(fnames, 1)
        yguide --> "Δχ²"
        view(smp, :, 1), dchi
    else
        i, j = _resolve_pair(vars, fnames)
        (1 <= i <= nfree && 1 <= j <= nfree) ||
            throw(ArgumentError("vars=$(vars) out of range for $nfree free parameters"))
        marker_z --> dchi
        colorbar_title --> "Δχ²"
        xguide --> _name_or_idx(fnames, i)
        yguide --> _name_or_idx(fnames, j)
        view(smp, :, i), view(smp, :, j)
    end
end

# BootstrapResult → histogram of one parameter's resampled θ̂ distribution
# (default: the first free parameter), with the point estimate and the
# percentile CI drawn as vertical reference lines. The asymmetry of the
# histogram about the estimate is the whole point versus a symmetric error bar.
RecipesBase.@recipe function f(r::BootstrapResult)
    par = pop!(plotattributes, :par, nothing)
    j = par === nothing ? _first_varying(r.samples) : _par_index(par, r.names)
    nm = _name_or_idx(r.names, j)
    col = _finite_col(r.samples, j)

    xguide --> nm
    yguide --> "count"
    legend --> true

    @series begin
        seriestype := :vline
        linecolor := :black
        linewidth --> 2
        label := "estimate"
        [r.estimate[j]]
    end
    @series begin
        seriestype := :vline
        linecolor := :firebrick
        linestyle := :dash
        label := "$(round(Int, 100 * r.ci_level))% CI"
        [r.ci_lower[j], r.ci_upper[j]]
    end

    seriestype := :histogram
    label --> "$(nm) ($(r.n_valid) valid)"
    bins --> :auto
    fillalpha --> 0.55
    col
end

# JackknifeResult → histogram of one parameter's leave-one-out estimates θ̂₍ⱼ₎
# (default: the first free parameter), with the full-data estimate and the
# jackknife mean θ̄ marked (their offset visualises the jackknife bias). The
# ±std is deliberately NOT drawn here: the (g-1)/g-inflated jackknife std lives
# on a different scale from the tightly-clustered leave-one-out values.
RecipesBase.@recipe function f(r::JackknifeResult)
    par = pop!(plotattributes, :par, nothing)
    j = par === nothing ? _first_varying(r.samples) : _par_index(par, r.names)
    nm = _name_or_idx(r.names, j)
    col = _finite_col(r.samples, j)

    xguide --> nm
    yguide --> "count"
    legend --> true

    @series begin
        seriestype := :vline
        linecolor := :black
        linewidth --> 2
        label := "estimate"
        [r.estimate[j]]
    end
    @series begin
        seriestype := :vline
        linecolor := :steelblue
        linestyle := :dash
        label := "mean (θ̄)"
        [r.mean[j]]
    end

    seriestype := :histogram
    label --> "$(nm) leave-one-out ($(r.n_valid) valid)"
    bins --> :auto
    fillalpha --> 0.55
    col
end

# SolutionModes + the sample matrix → the multimodal showcase: a colour-per-mode
# scatter of the clustered samples in a chosen 2-parameter projection, each
# mode's representative marked with a star. `samples` is the same matrix passed
# to `find_solution_modes`; `member_indices` index its rows. When `samples` is
# full external width the representative is indexed directly, otherwise the
# per-mode centroid is marked (free-width samples cannot be indexed by the
# full-vector representative).
RecipesBase.@recipe function f(modes::SolutionModes, samples::AbstractMatrix)
    vars = pop!(plotattributes, :vars, (1, 2))
    pnames = modes.param_names
    np = length(pnames)
    ncol = size(samples, 2)
    fullwidth = ncol == np
    legend --> true

    if ncol == 1
        # One free parameter: no pair to scatter — lay each mode's 1-D cluster
        # along x, separated vertically by mode index, with the representative
        # (full width) or centroid (free width) starred.
        xguide --> (fullwidth ? _name_or_idx(pnames, 1) : "parameter 1")
        yguide --> "mode"
        for (k, mode) in enumerate(modes)
            rows = mode.member_indices
            @series begin
                seriestype := :scatter
                seriescolor := k
                markersize --> 2
                markeralpha --> 0.35
                markerstrokewidth --> 0
                label := _mode_label(k)
                view(samples, rows, 1), fill(float(k), length(rows))
            end
            rx = fullwidth ? mode.representative[1] : _centroid(samples, rows, 1)
            @series begin
                seriestype := :scatter
                seriescolor := k
                markershape := :star5
                markersize --> 8
                markerstrokewidth --> 1
                label := ""
                primary := false
                [rx], [float(k)]
            end
        end
    else
        i, j = _resolve_pair(vars, fullwidth ? pnames : String[])
        (1 <= i <= ncol && 1 <= j <= ncol) ||
            throw(ArgumentError("vars=$(vars) selects columns ($i,$j) outside samples width $ncol"))
        xguide --> (fullwidth ? _name_or_idx(pnames, i) : "parameter $i")
        yguide --> (fullwidth ? _name_or_idx(pnames, j) : "parameter $j")
        for (k, mode) in enumerate(modes)
            rows = mode.member_indices
            @series begin
                seriestype := :scatter
                seriescolor := k
                markersize --> 2
                markeralpha --> 0.35
                markerstrokewidth --> 0
                label := _mode_label(k)
                view(samples, rows, i), view(samples, rows, j)
            end
            rx = fullwidth ? mode.representative[i] : _centroid(samples, rows, i)
            ry = fullwidth ? mode.representative[j] : _centroid(samples, rows, j)
            @series begin
                seriestype := :scatter
                seriescolor := k
                markershape := :star5
                markersize --> 8
                markerstrokewidth --> 1
                label := ""
                primary := false
                [rx], [ry]
            end
        end
    end
    nothing
end

# SolutionModes alone (no sample matrix) → graceful degradation: each mode's
# per-parameter bounding box (a low-α shape) plus its representative star, in a
# chosen 2-parameter projection. The clustered point cloud needs the sample
# matrix (use `plot(modes, samples)`); the boxes still show where the modes sit.
RecipesBase.@recipe function f(modes::SolutionModes)
    vars = pop!(plotattributes, :vars, (1, 2))
    pnames = modes.param_names
    np = length(pnames)
    legend --> true

    if np == 1
        # One parameter: draw each mode's 1-D range as a horizontal segment at
        # y = mode index, with its representative starred.
        xguide --> _name_or_idx(pnames, 1)
        yguide --> "mode"
        for (k, mode) in enumerate(modes)
            lo, hi = mode.param_ranges[1]
            @series begin
                seriestype := :path
                seriescolor := k
                linewidth --> 6
                linealpha --> 0.5
                label := _mode_label(k)
                [lo, hi], [float(k), float(k)]
            end
            @series begin
                seriestype := :scatter
                seriescolor := k
                markershape := :star5
                markersize --> 8
                label := ""
                primary := false
                [mode.representative[1]], [float(k)]
            end
        end
    else
        i, j = _resolve_pair(vars, pnames)
        (1 <= i <= np && 1 <= j <= np) ||
            throw(ArgumentError("vars=$(vars) out of range for $np parameters"))
        xguide --> _name_or_idx(pnames, i)
        yguide --> _name_or_idx(pnames, j)
        for (k, mode) in enumerate(modes)
            @series begin
                seriestype := :shape
                seriescolor := k
                fillalpha --> 0.15
                linealpha --> 0.7
                label := _mode_label(k)
                _rect_xy(mode.param_ranges[i], mode.param_ranges[j])
            end
            @series begin
                seriestype := :scatter
                seriescolor := k
                markershape := :star5
                markersize --> 8
                label := ""
                primary := false
                [mode.representative[i]], [mode.representative[j]]
            end
        end
    end
    nothing
end

# A single SolutionMode → its bounding box + representative star in a chosen
# 2-parameter projection (index-based labels; a lone mode carries no names).
RecipesBase.@recipe function f(mode::SolutionMode)
    vars = pop!(plotattributes, :vars, (1, 2))
    nd = length(mode.representative)

    if nd == 1
        # One parameter: the mode's 1-D range as a horizontal segment at y=0,
        # with its representative starred.
        xguide --> "par[1]"
        lo, hi = mode.param_ranges[1]
        @series begin
            seriestype := :path
            linewidth --> 6
            linealpha --> 0.5
            label := _mode_label(mode.index)
            [lo, hi], [0.0, 0.0]
        end
        seriestype := :scatter
        markershape --> :star5
        markersize --> 8
        label := ""
        primary := false
        [mode.representative[1]], [0.0]
    else
        i, j = _resolve_pair(vars, String[])
        (1 <= i <= nd && 1 <= j <= nd) ||
            throw(ArgumentError("vars=$(vars) out of range for $nd parameters"))
        xguide --> "par[$i]"
        yguide --> "par[$j]"
        @series begin
            seriestype := :shape
            fillalpha --> 0.15
            linealpha --> 0.7
            label := _mode_label(mode.index)
            _rect_xy(mode.param_ranges[i], mode.param_ranges[j])
        end
        seriestype := :scatter
        markershape --> :star5
        markersize --> 8
        label := ""
        primary := false
        [mode.representative[i]], [mode.representative[j]]
    end
end
