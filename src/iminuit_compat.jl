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

# THE shared least-squares kernel (Req 2: a single χ² implementation).
#
# Sums ``((yᵢ − model(xᵢ, par)) / errᵢ)²`` over `eachindex` of the three
# parallel arrays. `chisq`, `model_fit` (via `chisq`), and the
# `LeastSquares` cost type ALL route here — there is no second χ² loop
# anywhere in the package, so `LeastSquares(x,y,ye,model)(par)` is
# bit-identical to `chisq(model, Data(x,y,ye), par)` by construction.
#
# Pass full `Vector`s for the unmasked hot path (contiguous `@simd`
# reduction). For the restricted paths pass `@view`-sliced arrays — a
# contiguous `first:last` `fitrange` slice, or an integer-indexed mask
# slice (`@view x[active]`, `active::Vector{Int}`). Both view kinds keep
# the `@simd` reduction because they support fast linear / gather
# indexing (no data is copied — only the cheap `SubArray` wrapper).
@inline function _chisq_core(model::F, x, y, err, par) where {F}
    res = 0.0
    @inbounds @simd for i in eachindex(x, y, err)
        res += ((y[i] - model(x[i], par)) / err[i])^2
    end
    return res
end

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
    isempty(fitrange) &&
        return _chisq_core(dist, data.x, data.y, data.err, par)
    # Match IMinuit.jl semantics: fitrange collapses to `first:last`
    # (contiguous), even if the user passes a stepped range. Review
    # IMPORTANT #7 — preserving stride would silently diverge from
    # the drop-in source. The contiguous `@view` keeps the `@simd`
    # reduction and is value-for-value identical to iterating `rng`.
    rng = first(fitrange):last(fitrange)
    return @views _chisq_core(dist, data.x[rng], data.y[rng], data.err[rng], par)
end

function chisq(dist::Function, data, par; fitrange = ())
    _x = data[1]
    _y = data[2]
    _err = length(data) == 2 ? ones(length(_x)) : data[3]
    isempty(fitrange) && return _chisq_core(dist, _x, _y, _err, par)
    rng = first(fitrange):last(fitrange)
    return @views _chisq_core(dist, _x[rng], _y[rng], _err[rng], par)
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
    m = Minuit(_chisq, Float64.(start_values); kws...)
    # χ² cost over `data` → record the data-point count so the rich
    # display can show χ²/ndf + p-value (this is a χ² fit, errordef == 1).
    m.ndata = data.ndata
    return m
end

function model_fit(model::Function, data::Data, fit::Minuit; kws...)
    _chisq(par) = chisq(model, data, par)
    m = Minuit(_chisq, fit; kws...)
    m.ndata = data.ndata
    return m
end

"""
    @model_fit model data start_values kws...

Macro form of [`model_fit`](@ref). Expands to
`Minuit(par -> chisq(model, data, par), start_values; kws...)`.
"""
macro model_fit(model, data, start_values, kws...)
    expr = quote
        local _chisq = par -> NativeMinuit.chisq($(esc(model)), $(esc(data)), par)
        $(isempty(kws) ?
            :( NativeMinuit.Minuit(_chisq, $(esc(start_values))) ) :
            :( NativeMinuit.Minuit(_chisq, $(esc(start_values));
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
# caller must have `using Plots` in scope at expansion time. NativeMinuit
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
        _fit_  = _arr[findfirst(x -> x isa NativeMinuit.Minuit, _arr)]
        _dist_ = _arr[findfirst(x -> x isa Function, _arr)]
        _data_ = _arr[findfirst(x -> x isa NativeMinuit.Data, _arr)]
        _paras = NativeMinuit.args(_fit_)
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
        _fit_  = _arr[findfirst(x -> x isa NativeMinuit.Minuit, _arr)]
        _dist_ = _arr[findfirst(x -> x isa Function, _arr)]
        _data_ = _arr[findfirst(x -> x isa NativeMinuit.Data, _arr)]
        _paras = NativeMinuit.args(_fit_)
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
    # iminuit-style implicit resume (same idiom as `migrad!`): a repeat
    # `m.simplex()` starts from the CURRENT state — previous fit values
    # AND its updated per-parameter errors — not from the constructor
    # parameters. iminuit 2.31.3 empirical (warm bowl, errors=0.1): the
    # 2nd `simplex(ncall=6)` resumes with the fit-scale errors
    # [0.970…, 0.485…], burns 8 calls, and ends call-limit invalid.
    # `floor_errors = false`: iminuit carries the fit errors AS-IS, even
    # when they shrank below the constructor steps (error=1.0 bowl:
    # 2nd-run seed must use 0.25, not max(0.25, 1.0) — that's what lands
    # on iminuit's final [0.5, 0.5]).
    params_to_use = m.fmin === nothing ? _init_params(m) :
                    _build_resume_params(m; floor_errors = false)
    m.fmin = simplex(m.fcn, params_to_use;
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
`m.fmin`'s converged values — what `m.values`/`m.params` now report — not the
constructor initials), and as a side
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
# best-fit values when a fit exists (`m.fmin`), else the stored constructor
# config (`_init_params(m)` — internal consumers read the raw field, never
# the fit-overlaid `m.params` property). The stored config is NOT mutated —
# this mirrors how migrad / simplex leave the user's initial params intact
# and report current values via `m.fmin`. Without this, a post-fit `scan`
# would scan around (and reset the held params to) the stale constructor
# values, discarding the fit.
_scan_base_params(m::Minuit) =
    m.fmin === nothing ? _init_params(m) : _build_resume_params(m, m.fmin)

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
    mncontour(m::Minuit, par1, par2; cl=nothing, numpoints=100,
               size=numpoints, kws...) -> Vector{Tuple{Float64,Float64}}

MINOS 2D **confidence region** boundary — the `MnContours` profile
algorithm (each boundary point re-minimizes all other free parameters via
multi-parameter `function_cross`; not the ellipse approximation). Returns
a vector of `(x, y)` points in **physical (external) coordinates** — the
same frame as `m.values`, so the points plot directly (for bounded
parameters they are mapped back through the sin/√ transform). Mirrors
iminuit ≥ 2.0's `m.mncontour(par1, par2; cl=...)` **including its joint-
coverage `cl` semantics**:

- `cl = nothing` (default) → the **joint** 2-D 68 % confidence region,
  `Δχ² = delta_chisq(0.68, 2) ≈ 2.28` (× `m.up`).
- `0 < cl < 1` → the joint 2-D region with that probability content.
- `cl ≥ 1` → interpreted as nσ: the probability is `chisq_cl(cl², 1)`
  (e.g. `cl = 1` → 68.27 % → `Δχ² ≈ 2.30`; `cl = 2` → 95.45 % →
  `Δχ² ≈ 6.18`).

`size` is the iminuit-compatible alias for `numpoints` (it wins if both
are passed); `sigma` is a legacy alias for `cl`.

!!! note "Joint coverage vs the C++ `Δχ² = up` curve"
    Both conventions are F. James's own (*The Interpretation of Errors*,
    Minuit doc, §1.3.3): the raw C++ MnContours boundary `FCN = fmin + up`
    is the curve whose axis crossings are the single-parameter MINOS ±1σ
    errors — but as a 2-D region it covers only **39.3 %**; for a
    *simultaneous* statement about both parameters James prescribes
    scaling `up` by the χ²(2) quantile (his Table 1.3.3 — exactly what
    `cl` does here, following iminuit ≥ 2.0). For the unscaled C++ curve
    use the low-level [`contour_exact`](@ref) (`sigma = 1`), or
    `cl = chisq_cl(1, 2) ≈ 0.3935`. See the
    [MINOS errors & contours](@ref) tutorial for the full discussion with
    literature excerpts.

Routes through [`contour_exact`](@ref) with
`sigma = √(delta_chisq(cl, 2))` — the proper MNCONTOUR algorithm from
`reference/Minuit2_cpp/src/MnContours.cxx`, with the crossing aim scaled
to `fmin + up·sigma²` (mirrors iminuit's temporary-errordef scaling). The
faster but approximate ellipse-based [`contour_ellipse`](@ref) is kept
for cases where a quick visual check is enough.
"""
function mncontour(m::Minuit, par1, par2;
                    numpoints::Integer = 100,
                    size::Union{Integer,Nothing} = nothing,
                    sigma::Union{Real,Nothing} = nothing,
                    cl::Union{Real,Nothing} = nothing,
                    threaded_gradient::Union{Bool,Symbol} = m.threaded_gradient,
                    kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `mncontour(m, ...)`"))
    _tg = _use_threads(m, threaded_gradient)
    npts = size === nothing ? numpoints : Int(size)
    # cl wins over the legacy `sigma` alias; both default to iminuit's
    # literal 0.68 (NOT 0.6827 — verified against iminuit 2.31
    # `_cl_to_errordef(None, 2, 0.68) = 2.27886856637673`).
    _cl = cl === nothing ? (sigma === nothing ? 0.68 : Float64(sigma)) :
                            Float64(cl)
    _cl > 0 ||
        throw(ArgumentError("mncontour cl must be positive, got $_cl"))
    # Joint-2D scaling (James Table 1.3.3 / iminuit `_cl_to_errordef`):
    # Δχ² = delta_chisq(cl, ndof=2); the cross-search aim is
    # fmin + up·sigma² with sigma = √Δχ².
    σ_scale = sqrt(delta_chisq(_cl, 2))
    # Resolve indices then dispatch into `contour_exact` (real MnContours
    # algorithm) rather than the Phase-1 ellipse approximation.
    ix = par1 isa Integer ? Int(par1) : ext_index(m.params, String(par1))
    iy = par2 isa Integer ? Int(par2) : ext_index(m.params, String(par2))
    ix_int = m.params.int_of_ext[ix]
    iy_int = m.params.int_of_ext[iy]
    ce = contour_exact(m.fmin.internal, m.fmin.internal_cf,
                        ix_int, iy_int; npoints = npts,
                        threaded_gradient = _tg, sigma = σ_scale, kwargs...)
    # contour_exact works in internal (sin/√) coords; return physical
    # (external) coords matching `m.values` (no-op for unbounded params).
    return [(int_to_ext_value(m.params, ix_int, px),
             int_to_ext_value(m.params, iy_int, py)) for (px, py) in ce.points]
end

"""
    contour_grid(m::Minuit, par1, par2; size=50, bound=2, grid=nothing,
                 subtract_min=false) -> ContourGrid

iminuit's `Minuit.contour`: evaluate the FCN on a 2D grid in the
`(par1, par2)` plane with **all other parameters held fixed** at their
current (best-fit) values — a 2D **slice** of the FCN, the two-dimensional
analogue of [`profile`](@ref). No minimization is performed. (Named
`contour_grid` rather than iminuit's `contour` because the bare name
collides with `Plots.contour` under `using NativeMinuit, Plots`; this is also
what IMinuit.jl exported as `contour`.)

# Arguments

- `par1`, `par2` — Integer index, String, or Symbol name. Must be two
  distinct free parameters.
- `size=50` — number of grid points per axis.
- `bound=2` — scan range: a number `k` means `value ± k·σ` per axis
  (σ = current HESSE error), clipped to any parameter limits; or pass
  `((x_lo, x_hi), (y_lo, y_hi))` explicitly.
- `grid=(xs, ys)` — explicit grid axes (overrides `size`/`bound`).
- `subtract_min=false` — subtract the grid minimum from the values
  (`fval` becomes Δχ²-like).

Returns a [`ContourGrid`](@ref); destructures iminuit-style as
`xs, ys, F = contour_grid(m, "a", "b")` with `F[i, j] = FCN(xs[i], ys[j])`,
and plots directly (`plot(g)` → filled contour; or `draw_contour(m, ...)`).

!!! warning "A slice is NOT a confidence region"
    The grid fixes the other parameters instead of re-minimizing them, so
    its `Δχ²` level curves are **conditional** regions — systematically
    SMALLER than the true (profile) confidence region when `(par1, par2)`
    correlate with the remaining free parameters, by ≈ `√(1−R²)` per axis
    (R = multiple correlation with the others). With only 2 free
    parameters slice ≡ profile and the levels are exact. Use
    [`mncontour`](@ref) for confidence regions; use `contour_grid` to
    inspect the FCN landscape (valley orientation, secondary minima).
    For level lines: `Δχ² = m.up` projects to the single-parameter 68.27 %
    intervals; the joint-2D 68.27 % level is `delta_chisq(0.68, 2) ≈ 2.30`
    (× `m.up`).
"""
function contour_grid(m::Minuit, par1, par2;
                       size::Integer = 50,
                       bound = 2,
                       grid = nothing,
                       subtract_min::Bool = false)
    ix = par1 isa Integer ? Int(par1) : ext_index(m.params, String(par1))
    iy = par2 isa Integer ? Int(par2) : ext_index(m.params, String(par2))
    n = n_pars(m.params)
    (1 <= ix <= n && 1 <= iy <= n) ||
        throw(ArgumentError("contour_grid: parameter index out of bounds (got $ix, $iy for n=$n)"))
    ix != iy ||
        throw(ArgumentError("contour_grid requires two distinct parameters"))
    for idx in (ix, iy)
        is_fixed(m.params.pars[idx]) &&
            throw(ArgumentError("contour_grid: parameter `$(m.params.pars[idx].name)` is fixed"))
    end
    Int(size) >= 2 || throw(ArgumentError("contour_grid: size must be ≥ 2"))

    base = collect(Float64, m.values)   # post-fit values when fitted, else initial

    # Per-axis grid: explicit `grid` > explicit bound pairs > value ± k·σ.
    # Numeric bounds are clipped against the parameter's limits (iminuit
    # clips too — a grid point outside the limits would probe a region the
    # fit itself can never reach).
    function _axis(idx::Int, b)
        v = base[idx]
        lo, hi = if b isa Real
            s = abs(Float64(m.errors[idx]))
            (v - Float64(b) * s, v + Float64(b) * s)
        else
            (Float64(b[1]), Float64(b[2]))
        end
        p = m.params.pars[idx]
        has_lower_limit(p) && (lo = max(lo, p.lower))
        has_upper_limit(p) && (hi = min(hi, p.upper))
        lo < hi || throw(ArgumentError("contour_grid: empty scan range for `$(p.name)`"))
        return collect(range(lo, hi; length = Int(size)))
    end
    xs, ys = if grid !== nothing
        length(grid) == 2 ||
            throw(ArgumentError("contour_grid: grid must be a (xs, ys) pair"))
        (collect(Float64, grid[1]), collect(Float64, grid[2]))
    else
        bx, by = bound isa Real ? (bound, bound) : (bound[1], bound[2])
        (_axis(ix, bx), _axis(iy, by))
    end

    F = Matrix{Float64}(undef, length(xs), length(ys))
    x0 = copy(base)
    @inbounds for (j, yv) in enumerate(ys), (i, xv) in enumerate(xs)
        x0[ix] = xv
        x0[iy] = yv
        F[i, j] = m.fcn(x0)
    end
    subtract_min && (F .-= minimum(F))

    return ContourGrid(ix, iy,
                        m.params.pars[ix].name, m.params.pars[iy].name,
                        xs, ys, F, base[ix], base[iy],
                        Float64(m.fcn.up), subtract_min)
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
    # Raw config (original user steps), NOT the fit-overlaid `m.params`: each
    # grid point clones this config and re-runs MIGRAD, so the step sizes must
    # stay the constructor's, not the anchor fit's Hesse errors.
    cfg = _init_params(m)
    nm  = [p.name for p in cfg.pars]
    er  = [p.error for p in cfg.pars]
    fx  = [is_fixed(p) for p in cfg.pars]
    lim = Vector{Any}(undef, n_pars(cfg))
    for (i, p) in enumerate(cfg.pars)
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
    draw_contour(m::Minuit, par1, par2; size=50, bound=2, kws...) -> Plots.Plot

Filled-contour plot of the FCN **grid slice** from
[`contour_grid`](@ref)`(m, par1, par2; subtract_min=true)` — iminuit's
`m.draw_contour`. A landscape view, NOT a confidence region (see
[`contour_grid`](@ref)); for confidence contours use
[`draw_mncontour`](@ref). Requires `using Plots`. (≤ 0.4 this drew the
[`contour_ellipse`](@ref) approximation instead.)
"""
function draw_contour end

"""
    draw_mncontour(m::Minuit, par1, par2; numpoints=100, cl=nothing, kws...) -> Plots.Plot

Plot **exact** MINOS 2D confidence contours from [`mncontour`](@ref)
(boundary search with per-point re-minimization) at one or several
confidence levels — `cl` follows `mncontour`'s iminuit semantics
(default → joint 2-D 68 % region) and may be a vector to overlay several
contours. Requires `using Plots`. (≤ 0.4 this drew the fast
[`contour_ellipse`](@ref) approximation at Δχ² = `up` instead.)
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

# ─────────────────────────────────────────────────────────────────────────────
# optim / minimize_with — the alternative-minimizer bridge (Optim.jl extension).
#
# iminuit's `m.scipy(method=...)` minimises the FCN with `scipy.optimize.minimize`
# as the escape hatch when MIGRAD struggles (trust-region / derivative-free
# methods on stiff problems), then the user calls `hesse()` for the covariance.
# The Julia-optimal analog bridges to Optim.jl — the native, AD-friendly
# `scipy.optimize` equivalent — rather than shelling out to Python. The concrete
# implementation lives in `ext/NativeMinuitOptimExt.jl` (a package extension, like
# NativeMinuitForwardDiffExt / NativeMinuitPlotsExt), activated by `using Optim`.
#
# These thin entry points dispatch into the extension via `Base.get_extension`,
# so a missing `using Optim` yields a helpful message instead of a bare
# MethodError.
# ─────────────────────────────────────────────────────────────────────────────

# Helpful message when the Optim extension isn't loaded. A `const` so tests can
# assert on its content independent of the loaded state.
const _OPTIM_BRIDGE_NOT_LOADED =
    "optim(m) / minimize_with(m) is NativeMinuit's alternative-minimizer bridge, " *
    "the Julia analog of iminuit's scipy.optimize escape hatch — powered by " *
    "Optim.jl. Load it to enable: `using Optim`. (Or stay in pure NativeMinuit with " *
    "`migrad(m)` / `simplex(m)`.)"

# Fetch the loaded Optim extension module, or throw the helpful "load Optim"
# error. Centralised so `optim` and `minimize_with` share one dispatch point.
function _optim_bridge_ext()
    ext = Base.get_extension(@__MODULE__, :NativeMinuitOptimExt)
    ext === nothing && throw(ArgumentError(_OPTIM_BRIDGE_NOT_LOADED))
    return ext
end

"""
    optim(m::Minuit; method=:lbfgs, ncall=nothing, maxcall=nothing,
                     tol=nothing, options=nothing) -> Minuit

The Julia analog of iminuit's `m.scipy(...)` — the alternative-minimizer escape
hatch, powered by **Optim.jl** instead of Python's `scipy.optimize`. Load
`using Optim` to enable (it is an optional package extension).

Minimises the FCN with the chosen Optim optimizer starting from `m`'s current
parameter values, honours fixed parameters and box limits (via Optim's
`Fminbox`), writes the optimum back into `m` (so `m.values` / `m.fval` update),
and returns `m`. Run [`hesse`](@ref)`(m)` afterwards for the covariance —
matching iminuit's scipy-then-hesse flow. Use this when MIGRAD struggles
(stiff / ill-conditioned problems where a trust-region or derivative-free
method does better).

# `method` mapping (case / dash / underscore insensitive)

| `method`                                   | Optim optimizer        |
|:-------------------------------------------|:-----------------------|
| `:lbfgs`, `"L-BFGS-B"`                      | `LBFGS()`              |
| `:bfgs`                                     | `BFGS()`               |
| `:neldermead`, `:simplex`                   | `NelderMead()`         |
| `:newton`                                   | `Newton()`             |
| `:conjugategradient`, `:cg`                 | `ConjugateGradient()`  |
| `:gradientdescent`                          | `GradientDescent()`    |

Derivative-free (`:neldermead`) and second-order (`:newton`) methods cannot be
combined with box limits — Optim's `Fminbox` requires a first-order optimizer.
Use a first-order method (`:lbfgs` / `:bfgs` / `:conjugategradient` /
`:gradientdescent`) for bounded fits. When the constructor was given `grad=…`,
the analytical gradient is passed through to first-order optimizers; otherwise
Optim finite-differences it.

# Keyword arguments

- `method` — optimizer selector (see table). Default `:lbfgs`.
- `ncall` / `maxcall` — function-evaluation budget → Optim's `f_calls_limit`.
- `tol` — gradient-norm convergence tolerance → Optim's `g_tol`.
- `options` — a full `Optim.Options(...)` for fine control (overrides
  `ncall`/`maxcall`/`tol`).

!!! note "Bounded (Fminbox) fits"
    For bounded fits `ncall`/`maxcall`/`tol` configure Fminbox's *inner*
    optimizer (per outer iteration), not the global call budget or the outer
    stop criterion. For hard control of the outer Fminbox loop, pass a full
    `options=Optim.Options(outer_iterations=…, outer_g_abstol=…)`.

For full control over the optimizer object itself, use [`minimize_with`](@ref):
`minimize_with(m, Optim.LBFGS())`.

# Examples

```julia
using NativeMinuit, Optim
m = Minuit(fcn, x0)
optim(m; method=:lbfgs)   # or m |> optim
hesse(m)                  # covariance, à la iminuit
```
"""
optim(m::Minuit; kwargs...) = _optim_bridge_ext()._scipy_optim(m, nothing; kwargs...)

"""
    minimize_with(m::Minuit, optimizer=nothing; method=:lbfgs, ncall=nothing,
                  maxcall=nothing, tol=nothing, options=nothing) -> Minuit

Clearer-named alias of [`optim`](@ref): minimise `m`'s FCN with an alternative
optimizer from **Optim.jl**, write the optimum back, and return `m` (run
[`hesse`](@ref)`(m)` afterwards for the covariance). Load `using Optim` to
enable.

The first positional argument may be an Optim optimizer object, bypassing the
`method` name table for full control:

```julia
using NativeMinuit, Optim
minimize_with(m, LBFGS())                       # optimizer object
minimize_with(m, NelderMead())
minimize_with(m; method=:bfgs, tol=1e-10)       # by name, like optim(m; …)
```

See [`optim`](@ref) for the `method` mapping and keyword semantics.
"""
minimize_with(m::Minuit, optimizer = nothing; kwargs...) =
    _optim_bridge_ext()._scipy_optim(m, optimizer; kwargs...)
