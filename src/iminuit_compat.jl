# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# iminuit_compat.jl — pure-Julia drop-in helpers mirroring IMinuit.jl.
#
# Provides Data, chisq, model_fit / @model_fit, func_argnames, and the
# @plt_data / @plt_data! / @plt_best / @plt_best! plotting macros. All
# pure Julia (no PyCall / matplotlib dependency). Plotting macros
# expand to `Plots.scatter(...)` calls — the caller must have
# `using Plots` in scope at macro-expansion time.
#
# Mirror file in IMinuit.jl: src/Data.jl + module entry in src/IMinuit.jl.
# ─────────────────────────────────────────────────────────────────────────────

"""
    Data(x, y, err)

Holds `x`, `y`, and symmetric `err`-on-y vectors for χ² fitting. Fields
`x, y, err, ndata`. Mirrors IMinuit.jl's `Data` struct.

Different `Data` sets can be concatenated with `vcat(d1, d2, ...)` and
sliced with `d[idx]`. Asymmetric errors are not supported (use a
custom FCN with two separate σ for that).
"""
struct Data
    x::Vector{Float64}
    y::Vector{Float64}
    err::Vector{Float64}
    ndata::Int
    function Data(x, y, err)
        _check_data(x, y, err)
        length(x) == length(y) == length(err) ||
            throw(ArgumentError("Data: x/y/err length mismatch"))
        return new(Float64.(x), Float64.(y), Float64.(err), length(x))
    end
end

function _check_data(xdata, ydata, errdata)
    if any(ismissing, xdata) || any(ismissing, ydata) || any(ismissing, errdata)
        throw(ArgumentError("Data contain `missing` values"))
    end
    if any(isinf, xdata) || any(isinf, ydata) || any(isinf, errdata) ||
       any(isnan, xdata) || any(isnan, ydata) || any(isnan, errdata)
        # Review IMPORTANT #5: IMinuit.jl missed `isinf(errdata)` here.
        # Without the err check, an `Inf` σ would silently give a
        # "perfect fit" mirage (residual/Inf → 0).
        throw(ArgumentError("Data contain `Inf` or `NaN` values"))
    end
    if any(iszero, errdata)
        throw(ArgumentError("Data contain 0 in errors"))
    end
end

Base.vcat(d1::Data, d2::Data) = Data(vcat(d1.x, d2.x), vcat(d1.y, d2.y),
                                      vcat(d1.err, d2.err))
Base.vcat(d1::Data, ds::Data...) = reduce(vcat, [d1, ds...])

Base.getindex(d::Data, idx) = Data(d.x[idx], d.y[idx], d.err[idx])
Base.length(d::Data) = d.ndata
Base.iterate(d::Data, state = 1) = state > d.ndata ? nothing :
    ((d.x[state], d.y[state], d.err[state]), state + 1)

function Base.show(io::IO, d::Data)
    print(io, "Data(ndata=", d.ndata, ")")
end
function Base.show(io::IO, ::MIME"text/plain", d::Data)
    println(io, "Data(", d.ndata, " points)")
    n = min(5, d.ndata)
    for i in 1:n
        @printf(io, "  x=%g  y=%g  err=%g\n", d.x[i], d.y[i], d.err[i])
    end
    d.ndata > n && println(io, "  ... (+", d.ndata - n, " more)")
end

# ─────────────────────────────────────────────────────────────────────────────
# chisq — χ² cost function for Data or tuple form
# ─────────────────────────────────────────────────────────────────────────────

@doc raw"""
    chisq(dist::Function, data::Data, par; fitrange=()) -> Float64
    chisq(dist::Function, data, par;       fitrange=()) -> Float64

χ² cost: `dist(x, par)` evaluates the model at scalar `x` with the
parameter container `par`. Returns
``\sum_i ((y_i - \mathrm{dist}(x_i, par)) / \sigma_i)^2``.

The second method accepts `data` as a tuple `(x, y, err)` or `(x, y)`
(unit errors). `fitrange` restricts the sum to a subset of data
indices (default: all points).

Use `chisq` as the FCN argument to `Minuit`/`migrad`, or wrap via
[`model_fit`](@ref) / `@model_fit`.
"""
function chisq(dist::Function, data::Data, par; fitrange = ())
    # Match IMinuit.jl semantics: fitrange collapses to `first:last`
    # (contiguous), even if the user passes a stepped range. Review
    # IMPORTANT #7 — preserving stride would silently diverge from
    # the drop-in source.
    rng = isempty(fitrange) ? (1:data.ndata) : (first(fitrange):last(fitrange))
    res = 0.0
    @inbounds @simd for i in rng
        res += ((data.y[i] - dist(data.x[i], par)) / data.err[i])^2
    end
    return res
end

function chisq(dist::Function, data, par; fitrange = ())
    _x = data[1]
    _y = data[2]
    _n = length(_x)
    _err = length(data) == 2 ? ones(_n) : data[3]
    rng = isempty(fitrange) ? (1:_n) : (first(fitrange):last(fitrange))
    res = 0.0
    @inbounds @simd for i in rng
        res += ((_y[i] - dist(_x[i], par)) / _err[i])^2
    end
    return res
end

# ─────────────────────────────────────────────────────────────────────────────
# iminuit.cost helpers — pure-Julia (no Python).
#
# Match the signatures of `iminuit.cost.chi2 / poisson_chi2 /
# multinominal_chi2` so IMinuit.jl code that uses them works unchanged.
# ─────────────────────────────────────────────────────────────────────────────

@doc raw"""
    chi2(y, yerror, ymodel) -> Float64

Pearson χ² for symmetric Gaussian errors. Mirrors
`iminuit.cost.chi2(y, yerror, ymodel)`. Computes
``\sum_i ((y_i - y_{model,i}) / \sigma_i)^2`` skipping bins with
`yerror[i] ≤ 0`.
"""
function chi2(y::AbstractVector{<:Real},
              yerror::AbstractVector{<:Real},
              ymodel::AbstractVector{<:Real})
    length(y) == length(yerror) == length(ymodel) ||
        throw(DimensionMismatch("chi2: y / yerror / ymodel length mismatch"))
    res = 0.0
    @inbounds @simd for i in eachindex(y)
        σ = yerror[i]
        σ > 0 && (res += ((y[i] - ymodel[i]) / σ)^2)
    end
    return res
end

@doc raw"""
    poisson_chi2(n, mu) -> Float64

Likelihood-ratio χ² for Poisson-distributed counts `n` with predicted
means `mu`. Mirrors `iminuit.cost.poisson_chi2(n, mu)`:
``2 \sum_i \left[ \mu_i - n_i + n_i \log(n_i / \mu_i) \right]``
with the `n_i log(n_i / μ_i)` term defined as 0 when `n_i = 0` (limit
of `x log x` at 0).
"""
function poisson_chi2(n::AbstractVector{<:Real}, mu::AbstractVector{<:Real})
    length(n) == length(mu) ||
        throw(DimensionMismatch("poisson_chi2: n / mu length mismatch"))
    res = 0.0
    @inbounds @simd for i in eachindex(n)
        μ_i = mu[i]
        μ_i > 0 || throw(DomainError(μ_i, "poisson_chi2: μ must be positive"))
        n_i = n[i]
        if n_i > 0
            res += 2.0 * (μ_i - n_i + n_i * log(n_i / μ_i))
        else
            res += 2.0 * μ_i
        end
    end
    return res
end

@doc raw"""
    multinominal_chi2(n, mu) -> Float64

Likelihood-ratio χ² for multinomial-distributed counts (only relative
proportions matter — constant offsets cancel). Mirrors
`iminuit.cost.multinominal_chi2(n, mu)`:
``2 \sum_i n_i \log(n_i / \mu_i)`` with the `0 log 0 = 0` convention.
"""
function multinominal_chi2(n::AbstractVector{<:Real},
                            mu::AbstractVector{<:Real})
    length(n) == length(mu) ||
        throw(DimensionMismatch("multinominal_chi2: n / mu length mismatch"))
    res = 0.0
    @inbounds @simd for i in eachindex(n)
        μ_i = mu[i]
        μ_i > 0 || throw(DomainError(μ_i, "multinominal_chi2: μ must be positive"))
        n_i = n[i]
        n_i > 0 && (res += 2.0 * n_i * log(n_i / μ_i))
    end
    return res
end

# ─────────────────────────────────────────────────────────────────────────────
# model_fit — convenience constructor wrapping chisq + Minuit.
# ─────────────────────────────────────────────────────────────────────────────

"""
    model_fit(model::Function, data::Data, start_values; kws...) -> Minuit
    model_fit(model::Function, data::Data, fit::Minuit;    kws...) -> Minuit

Build a [`Minuit`](@ref) fit of `model(x, par)` against `data` using
the χ² cost from [`chisq`](@ref). `start_values` may be an `AbstractVector`
of initial values or a previous `Minuit` fit (whose latest values are
reused as the new starting point).

`kws...` flow through to the `Minuit` constructor (`name`, `error`,
`limits`, `fixed`, `grad`, `strategy`, `tol`, etc.).
"""
function model_fit(model::Function, data::Data,
                    start_values::AbstractVector; kws...)
    _chisq(par) = chisq(model, data, par)
    return Minuit(_chisq, Float64.(start_values); kws...)
end

function model_fit(model::Function, data::Data, fit::Minuit; kws...)
    _chisq(par) = chisq(model, data, par)
    return Minuit(_chisq, fit; kws...)
end

"""
    @model_fit model data start_values kws...

Macro form of [`model_fit`](@ref). Expands to
`Minuit(par -> chisq(model, data, par), start_values; kws...)`.
"""
macro model_fit(model, data, start_values, kws...)
    expr = quote
        local _chisq = par -> JuMinuit.chisq($(esc(model)), $(esc(data)), par)
        $(isempty(kws) ?
            :( JuMinuit.Minuit(_chisq, $(esc(start_values))) ) :
            :( JuMinuit.Minuit(_chisq, $(esc(start_values));
                                $(esc.(kws)...)) )
        )
    end
    return expr
end

# ─────────────────────────────────────────────────────────────────────────────
# func_argnames — Julia reflection helper (mirrors IMinuit.jl).
# ─────────────────────────────────────────────────────────────────────────────

# Lifted from Base's methodshow.jl machinery — extracts the argument
# names of a `Method` as a `Vector{Symbol}`.
function _method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,),
                      m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

"""
    func_argnames(f::Function) -> Vector{Symbol}

Argument names of `f` (skipping the implicit `f::Function` slot).
Useful for auto-deriving parameter names from an FCN signature
(`func_argnames(par_a, par_b) -> [:par_a, :par_b]`).
"""
function func_argnames(f::Function)
    ms = collect(methods(f))
    return _method_argnames(last(ms))[2:end]
end

# ─────────────────────────────────────────────────────────────────────────────
# Plotting macros — IMinuit.jl-compatible.
#
# These expand to `Plots.scatter(...)` / `Plots.plot!(...)` calls; the
# caller must have `using Plots` in scope at expansion time. JuMinuit
# itself uses RecipesBase, so this is a soft Plots.jl dependency.
# ─────────────────────────────────────────────────────────────────────────────

"""
    @plt_data data kws...
    @plt_data! data kws...

Scatter plot of `data` with y-errorbars. The trailing `kws...` flow
through to `Plots.scatter(...)`. Requires `using Plots` in scope.
"""
macro plt_data(data, kws...)
    _plt = quote
        if isempty($kws)
            Plots.scatter($data.x, $data.y, yerror = $data.err,
                           xlab = "x", ylab = "y", label = "Data")
        else
            Plots.scatter($data.x, $data.y, yerror = $data.err; $(kws...))
        end
    end
    return esc(_plt)
end

macro plt_data!(data, kws...)
    _plt = quote
        if isempty($kws)
            Plots.scatter!($data.x, $data.y, yerror = $data.err,
                            xlab = "x", ylab = "y", label = "Data")
        else
            Plots.scatter!($data.x, $data.y, yerror = $data.err; $(kws...))
        end
    end
    return esc(_plt)
end

"""
    @plt_best dist fit data kws...
    @plt_best! dist fit data kws...

Plot model `dist(x, params)` evaluated at the best-fit `params` from
`fit::Minuit` over the data points. The three positional args can be
in any order — they're discriminated by type at runtime. Requires
`using Plots`.
"""
macro plt_best(dist, fit, data, kws...)
    _expr = quote
        _arr = ($dist, $fit, $data)
        _fit_  = _arr[findfirst(x -> x isa JuMinuit.Minuit, _arr)]
        _dist_ = _arr[findfirst(x -> x isa Function, _arr)]
        _data_ = _arr[findfirst(x -> x isa JuMinuit.Data, _arr)]
        _paras = JuMinuit.args(_fit_)
        _dis(x) = _dist_(x, _paras)
        _xrange = _data_.x
        _wv = LinRange(_xrange[1], _xrange[end], 100)
        Plots.scatter(_data_.x, _data_.y, yerror = _data_.err, label = "Data")
        if isempty($kws)
            Plots.plot!(_wv, _dis.(_wv), xlab = "x", ylab = "y",
                         label = "Best fit", lw = 1.5)
        else
            Plots.plot!(_wv, _dis.(_wv); $(kws...))
        end
    end
    return esc(_expr)
end

macro plt_best!(dist, fit, data, kws...)
    _expr = quote
        _arr = ($dist, $fit, $data)
        _fit_  = _arr[findfirst(x -> x isa JuMinuit.Minuit, _arr)]
        _dist_ = _arr[findfirst(x -> x isa Function, _arr)]
        _data_ = _arr[findfirst(x -> x isa JuMinuit.Data, _arr)]
        _paras = JuMinuit.args(_fit_)
        _dis(x) = _dist_(x, _paras)
        _xrange = _data_.x
        _wv = LinRange(_xrange[1], _xrange[end], 100)
        Plots.scatter!(_data_.x, _data_.y, yerror = _data_.err, label = "Data")
        if isempty($kws)
            Plots.plot!(_wv, _dis.(_wv), xlab = "x", ylab = "y",
                         label = "Best fit", lw = 1.5)
        else
            Plots.plot!(_wv, _dis.(_wv); $(kws...))
        end
    end
    return esc(_expr)
end

# ─────────────────────────────────────────────────────────────────────────────
# Minuit-method wrappers — iminuit `m.simplex(...)`, `m.scan(...)`,
# `m.mncontour(...)`, `m.profile(...)`, `m.mnprofile(...)`.
#
# Each operates on the stored `m.fcn` / `m.params`; results either
# replace `m.fmin` (algorithms that minimize) or are returned as
# arrays (profile / contour helpers).
# ─────────────────────────────────────────────────────────────────────────────

"""
    simplex(m::Minuit; maxfcn=nothing, ncall=maxfcn, minedm=nothing) -> Minuit

Run Nelder-Mead simplex on `m`. Updates `m.fmin` and returns `m`.
Useful as a robust fallback when MIGRAD fails (no gradient needed).

`ncall` is the iminuit-compatible alias for `maxfcn` (review IMPORTANT #3).
"""
function simplex(m::Minuit;
                  maxfcn::Union{Integer,Nothing} = nothing,
                  ncall::Union{Integer,Nothing} = nothing,
                  minedm::Union{Real,Nothing} = nothing)
    eff = ncall !== nothing ? ncall : maxfcn
    m.fmin = simplex(m.fcn, m.params;
                      maxfcn = eff, minedm = minedm,
                      prec = m.prec)
    return m
end

"""
    scan(m::Minuit, par; maxsteps=41, low=0, high=0) ->
        Vector{Tuple{Float64,Float64}}

1D scan along parameter `par` (Integer index or String name) of the
fitted Minuit. Returns `(x, fval)` pairs; the central point comes
first followed by `maxsteps` equally-spaced probes.

If `low == high == 0`, defaults to ±2σ around the current parameter
value (or the parameter limits if both are set). The scan does NOT
minimize over other parameters — for that use [`mnprofile`](@ref).

**Best-value retention (iminuit / C++ MnParameterScan semantics):** the scan
runs around the CURRENT values (after a fit, the held parameters sit at
`m.fmin`'s values, NOT the un-mutated constructor `m.params`), and as a side
effect `m` is left at the lowest-fval grid point found (the central point is
included in the comparison). So `m.values` / `m.fval` reflect the best grid
point — with the other parameters' current values preserved — and a follow-up
`migrad!`/`hesse` resumes from there. The covariance is NOT updated (scan
computes no Hessian; `m.matrix` → `nothing`). The returned point-list is
unchanged. To scan WITHOUT moving `m`, use [`profile`](@ref).
"""
function scan(m::Minuit, par::Integer;
                maxsteps::Integer = 41,
                low::Real = 0.0, high::Real = 0.0)
    base = _scan_base_params(m)
    points = scan(m.fcn, base, Int(par);
                  maxsteps = maxsteps, low = low, high = high)
    _scan_retain_best!(m, base, Int(par), points)
    return points
end
function scan(m::Minuit, par::AbstractString; kwargs...)
    return scan(m, ext_index(m.params, String(par)); kwargs...)
end

# The base Parameters a Minuit-level scan / profile operates on: the current
# best-fit values when a fit exists (`m.fmin`), else the constructor
# `m.params`. `m.params` is NOT mutated — this mirrors how migrad / simplex
# leave the user's initial params intact and report current values via
# `m.fmin`. Without this, a post-fit `scan` would scan around (and reset the
# held params to) the stale constructor values, discarding the fit.
_scan_base_params(m::Minuit) =
    m.fmin === nothing ? m.params : _build_resume_params(m, m.fmin)

# Leave `m` at the lowest-FINITE-fval grid point found by a scan, holding the
# other parameters at their `base` (current) values. The central point (index
# 1 of `points`) is included, so a flat / already-optimal scan stays at the
# central value. A covariance-less best-point fmin is installed (scan computes
# no Hessian) so `m.values` / `m.fval` / `m.valid` read correctly and a
# follow-up `migrad!` / `hesse` resumes from the best grid point. `m.params`
# is left untouched. Mirrors C++ MnParameterScan best-value retention
# (MnParameterScan.h:42-43) + iminuit `m.scan()`. If NO grid point has a
# finite fval the Minuit is left untouched — we never publish a NaN-valued
# "valid" state.
function _scan_retain_best!(m::Minuit, base::Parameters, par::Int,
                             points::Vector{Tuple{Float64,Float64}})
    best_x = 0.0
    best_f = Inf
    found = false
    @inbounds for (xk, fk) in points
        if isfinite(fk) && fk < best_f
            best_x = xk
            best_f = fk
            found = true
        end
    end
    found || return m
    new_pars = collect(base.pars)
    new_pars[par] = _build_value_par(new_pars[par], best_x)
    retained = Parameters(new_pars, m.prec)
    m.fmin = _point_function_minimum(m.fcn, retained, best_f)
    return m
end

"""
    mncontour(m::Minuit, par1, par2; numpoints=100, size=numpoints,
               sigma=1, cl=sigma, kws...) -> Vector{Tuple{Float64,Float64}}

MINOS 2D contour — the C++-faithful `MnContours` algorithm (boundary
search via multi-parameter `function_cross`, not the ellipse
approximation). Returns a vector of `(x, y)` points tracing the
`sigma`-σ contour in the (par1, par2) plane. Mirrors iminuit's
`m.mncontour(par1, par2)`.

`size` and `cl` are iminuit-compatible kwarg aliases for `numpoints`
and `sigma` respectively. The iminuit names win if both pairs are
passed.

Routes through [`contour_exact`](@ref) — the proper MNCONTOUR algorithm
from `reference/Minuit2_cpp/src/MnContours.cxx`. The faster but
approximate ellipse-based [`contour`](@ref) is kept for cases where a
quick visual check is enough. Spec cross-check (1994 §1.4.4.2 vs 2004
§4.4) confirms MnContours is the C++/iminuit `m.mncontour` default;
this routing matches that expectation.
"""
function mncontour(m::Minuit, par1, par2;
                    numpoints::Integer = 100,
                    size::Union{Integer,Nothing} = nothing,
                    sigma::Real = 1,
                    cl::Union{Real,Nothing} = nothing,
                    threaded_gradient::Bool = m.threaded_gradient,
                    kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `mncontour(m, ...)`"))
    npts = size === nothing ? numpoints : Int(size)
    σ    = cl === nothing ? sigma : Float64(cl)
    isapprox(σ, 1.0) ||
        throw(ArgumentError("mncontour sigma/cl ≠ 1 is Phase 1.x deferred; got $σ"))
    # Resolve indices then dispatch into `contour_exact` (real MnContours
    # algorithm) rather than `contour` (Phase-1 ellipse approximation).
    ix = par1 isa Integer ? Int(par1) : ext_index(m.params, String(par1))
    iy = par2 isa Integer ? Int(par2) : ext_index(m.params, String(par2))
    ix_int = m.params.int_of_ext[ix]
    iy_int = m.params.int_of_ext[iy]
    ce = contour_exact(m.fmin.internal, m.fmin.internal_cf,
                        ix_int, iy_int; npoints = npts,
                        threaded_gradient = threaded_gradient, kwargs...)
    return ce.points
end

"""
    profile(m::Minuit, par; bins=100, size=bins, low=0, high=0) ->
        Vector{Tuple{Float64,Float64}}

1D profile of the FCN along `par` — same as [`scan`](@ref) but with
iminuit's default `bins=100` (vs `maxsteps=41` for `scan`). NO inner
minimization. Returns `(par_value, fval)` pairs.

`size` is the iminuit-compatible alias for `bins` (review IMPORTANT #3).

Unlike [`scan`](@ref), `profile` is a **pure diagnostic** — it does NOT move
`m` to the best grid point (no best-value retention, no state change).
"""
function profile(m::Minuit, par;
                  bins::Integer = 100,
                  size::Union{Integer,Nothing} = nothing,
                  low::Real = 0.0, high::Real = 0.0)
    nb = size === nothing ? bins : Int(size)
    # Pure diagnostic: scan around the CURRENT values (like `scan`, via
    # `_scan_base_params`) but call the low-level scan directly so it does NOT
    # trigger `scan(m, par)`'s best-value retention — `m` is never mutated.
    idx = par isa Integer ? Int(par) : ext_index(m.params, String(par))
    return scan(m.fcn, _scan_base_params(m), idx;
                 maxsteps = nb, low = low, high = high)
end

"""
    mnprofile(m::Minuit, par; bins=30, low=0, high=0) ->
        Vector{Tuple{Float64,Float64}}

1D **MINOS profile**: at each grid point along `par`, FIX `par` at
that value and RE-MINIMIZE all other free parameters. Returns
`(par_value, min_fval)` pairs.

This is the "constrained-MIGRAD profile" iminuit exposes as
`m.mnprofile(par)`. It's strictly more informative than a bare scan
([`profile`](@ref)) because it shows the χ² well "as seen by" the
nuisance parameters — but each grid point costs an inner-MIGRAD.

Range defaults: if `low == high == 0`, uses
`m.values[par] ± 2 · m.errors[par]`.
"""
function mnprofile(m::Minuit, par::Integer;
                    bins::Integer = 30,
                    size::Union{Integer,Nothing} = nothing,
                    low::Real = 0.0, high::Real = 0.0)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `mnprofile(m, par)`"))
    1 <= par <= n_pars(m.params) ||
        throw(ArgumentError("mnprofile: par $par out of bounds"))
    is_fixed(m.params.pars[par]) &&
        throw(ArgumentError("Cannot mnprofile fixed parameter $par"))
    nb = size === nothing ? bins : Int(size)
    nb >= 2 || throw(ArgumentError("mnprofile: bins must be ≥ 2"))

    central = m.values[par]
    p_meta = m.params.pars[par]
    low_f, high_f = Float64(low), Float64(high)
    if low_f == 0.0 && high_f == 0.0
        low_f  = central - 2.0 * abs(m.errors[par])
        high_f = central + 2.0 * abs(m.errors[par])
    end
    # Clip against any limit
    if has_lower_limit(p_meta)
        low_f = max(low_f, p_meta.lower)
    end
    if has_upper_limit(p_meta)
        high_f = min(high_f, p_meta.upper)
    end

    stp = (high_f - low_f) / (nb - 1)
    result = Vector{Tuple{Float64,Float64}}()
    sizehint!(result, nb)

    # For each grid point: build a Minuit with `par` fixed at the
    # grid value, re-run migrad, record minimum fval. Bounds and
    # fixedness on the OTHER parameters are preserved. The user's
    # gradient (m.cfwg.g) is forwarded so each inner MIGRAD also
    # benefits from analytical differentiation — review NICE-TO-HAVE #8.
    nm  = [p.name for p in m.params.pars]
    er  = [p.error for p in m.params.pars]
    fx  = [is_fixed(p) for p in m.params.pars]
    lim = Vector{Any}(undef, n_pars(m.params))
    for (i, p) in enumerate(m.params.pars)
        lo = isnan(p.lower) ? nothing : p.lower
        hi = isnan(p.upper) ? nothing : p.upper
        lim[i] = (lo === nothing && hi === nothing) ? nothing : (lo, hi)
    end
    grad_fn = m.cfwg === nothing ? nothing : m.cfwg.g

    @inbounds for i in 0:(nb - 1)
        xval = low_f + i * stp
        x0_i = copy(m.values)
        x0_i[par] = xval
        fx_i = copy(fx)
        fx_i[par] = true
        m_i = Minuit(m.fcn.f, x0_i;
                      name = nm, error = er, fixed = fx_i, limits = lim,
                      up = m.fcn.up, prec = m.prec,
                      grad = grad_fn,
                      strategy = m.strategy, tol = m.tol)
        migrad!(m_i)
        push!(result, (xval, m_i.fval))
    end
    return result
end

function mnprofile(m::Minuit, par::AbstractString; kwargs...)
    return mnprofile(m, ext_index(m.params, String(par)); kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# draw_* — Plots-based visualization helpers (declared here, defined
# in ext/PlotsExt.jl when `using Plots` is loaded).
#
# Each `draw_X` returns a Plots.Plot object. Calling them without
# `using Plots` throws a clear error.
# ─────────────────────────────────────────────────────────────────────────────

"""
    draw_contour(m::Minuit, par1, par2; bins=50, kws...) -> Plots.Plot

Plot the 2D contour from `contour(m, par1, par2; npoints=bins)`. Requires
`using Plots`. Mirrors IMinuit.jl's `draw_contour`.
"""
function draw_contour end

"""
    draw_mncontour(m::Minuit, par1, par2; numpoints=100, nsigma=1, kws...) -> Plots.Plot

Plot the MINOS 2D contour. Equivalent to `draw_contour` for `sigma=1`.
Requires `using Plots`.
"""
function draw_mncontour end

"""
    draw_profile(m::Minuit, par; bins=100, kws...) -> Plots.Plot

Plot the 1D scan from `profile(m, par; bins=bins)`. Requires `using Plots`.
"""
function draw_profile end

"""
    draw_mnprofile(m::Minuit, par; bins=30, kws...) -> Plots.Plot

Plot the 1D MINOS profile from `mnprofile(m, par; bins=bins)`. Requires
`using Plots`.
"""
function draw_mnprofile end

"""
    draw_mnmatrix(m::Minuit; kws...) -> Plots.Plot

Plot all pairwise 2D contours arranged as a triangular matrix.
Requires `using Plots`.
"""
function draw_mnmatrix end

# scipy IMinuit alias — JuMinuit doesn't bundle scipy; throw a helpful
# error so existing IMinuit.jl call sites get a clear message.
"""
    scipy(m::Minuit; kws...) -> error

iminuit's `m.scipy(...)` delegates to Python `scipy.optimize`. JuMinuit
is a pure Julia port — no scipy. Use [`migrad`](@ref) (DFP variable-
metric) or [`simplex`](@ref) (Nelder-Mead) instead, or call a Julia
optimizer like `Optim.optimize(f, x0)` and feed the result back via
`Minuit(fcn, x0_opt)`.
"""
function scipy(m::Minuit; kwargs...)
    throw(ArgumentError(
        "scipy(m) is a Python-only iminuit feature. Use `migrad(m)` or " *
        "`simplex(m)` for pure-Julia minimization, or call Optim.jl " *
        "directly and seed Minuit with the result."))
end
