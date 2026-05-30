# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# cost_functions.jl — a Julia-native cost type family.
#
# This is NOT a transliteration of iminuit's `iminuit.cost` Python class
# hierarchy (`Cost` base + `__call__` + `_ndata` + subclasses). The
# Julian expression of the same ideas:
#
#   • `abstract type AbstractCost end` (distinct from the internal FCN
#     wrapper `AbstractCostFunction` in fcn.jl) with concrete types each
#     PARAMETERIZED ON their model/pdf function (`LeastSquares{F}`, …)
#     for zero-cost call-site specialization — same trick as
#     `CostFunction{F}`.
#   • `errordef(::AbstractCost)` is a TRAIT dispatched on the cost type
#     (`errordef(::LeastSquares) = 1.0`, `errordef(::UnbinnedNLL) = 0.5`,
#     …), not a field copied from Python. It flows into the `Minuit`
#     `up` (ErrorDef) and from there into HESSE errors / covariance
#     (`2·up·H⁻¹`, result.jl) and MINOS scaling (minuit.jl).
#   • Composition is OPERATOR OVERLOADING: `a + b → CostSum`, with the
#     parameter set the union-by-name of the components and the FCN the
#     (errordef-rescaled) sum. Works for MIXED cost types (a `LeastSquares`
#     + an `UnbinnedNLL` simultaneous fit).
#   • Masking is a `BitVector` (`true` = keep) realised, with no data
#     copy, as an integer-indexed `@view` over the data — uniform across
#     every cost type.
#
# Dedup (Req 2): `LeastSquares` reuses the ONE χ² kernel `_chisq_core`
# (iminuit_compat.jl) that `chisq` / `model_fit` use — no second χ².
#
# Compatibility (Req 3): the IMinuit.jl function-style entry points
# (`chisq`, `model_fit`, `@model_fit`, `Data`) are untouched. The cost
# types are a thin second front-door onto the same core; `Minuit(cost,
# x0)` auto-extracts `errordef(cost)` and the data count.
# ─────────────────────────────────────────────────────────────────────────────

"""
    AbstractCost

Supertype of the Julia-native cost-function family — `LeastSquares`,
`UnbinnedNLL`, `BinnedNLL`, `ExtendedUnbinnedNLL`, `ExtendedBinnedNLL`,
and their `CostSum` composition.

A cost is a **callable** mapping a parameter vector to a `Float64`
objective: `(c::AbstractCost)(par)::Float64`. Two things make a cost
more than a bare closure:

- `errordef(c)` — a trait (dispatched on the concrete type) giving the
  Minuit ErrorDef (`1.0` for χ², `0.5` for a negative-log-likelihood).
  `Minuit(c, x0)` reads it automatically, so you never pass `up=`.
- `a + b` composes costs into a [`CostSum`](@ref) for simultaneous fits,
  taking the union of their (named) parameters.

Distinct from `AbstractCostFunction` (fcn.jl), which is the *internal*
FCN wrapper (`CostFunction` / `CostFunctionWithGradient`) that the
MIGRAD/HESSE/MINOS machinery consumes. A cost becomes an FCN when you
hand it to `Minuit`: it is wrapped in a `CostFunction` carrying
`up = errordef(c)`.
"""
abstract type AbstractCost end

# ─────────────────────────────────────────────────────────────────────────────
# Shared construction helpers — mask + parameter-name normalisation.
# ─────────────────────────────────────────────────────────────────────────────

# Normalise a user mask kwarg to a `BitVector` (or `nothing`). `true`
# keeps the point/bin; `false` drops it (iminuit convention).
_to_bitmask(::Nothing, ::Int) = nothing
function _to_bitmask(m::AbstractVector{Bool}, n::Int)
    length(m) == n || throw(ArgumentError(
        "mask has length $(length(m)) but the data has $n entries"))
    return BitVector(m)
end
_to_bitmask(m, ::Int) = throw(ArgumentError(
    "mask must be a Bool vector (true = keep), got $(typeof(m))"))

# The cached hot-path index list for a mask: `findall` once at
# construction, so the per-call kernel iterates an integer index vector
# (a `@simd`-friendly gather) rather than re-scanning the BitVector.
# No DATA is copied — only an `O(n_kept)` `Vector{Int}`.
_active_from_mask(::Nothing) = nothing
_active_from_mask(bm::BitVector) = findall(bm)

# Normalise parameter names (for composition by name) to `Vector{Symbol}`.
_to_pnames(::Nothing) = nothing
function _to_pnames(v::AbstractVector)
    s = Symbol.(v)
    allunique(s) ||
        throw(ArgumentError("parameter names must be unique within a cost: $s"))
    return s
end

"""
    parameter_names(c::AbstractCost) -> Union{Nothing,Vector{Symbol}}

Parameter names carried by the cost, or `nothing` if it was built
without a `name=` kwarg. Names are required to compose costs with `+`
(the union is taken by name) and, when present, become the default
parameter names of `Minuit(c, x0)`.
"""
parameter_names(c::AbstractCost) = c.pnames

# ─────────────────────────────────────────────────────────────────────────────
# LeastSquares — χ², errordef 1. Reuses `_chisq_core` (the ONE χ² kernel).
# ─────────────────────────────────────────────────────────────────────────────

"""
    LeastSquares(x, y, yerror, model; mask=nothing, name=nothing)
    LeastSquares(data::Data, model;   mask=nothing, name=nothing)

Least-squares cost ``\\sum_i ((y_i - \\mathrm{model}(x_i, par)) /
\\sigma_i)^2`` (errordef `1.0`). `model(x_scalar, par)` follows the
JuMinuit / IMinuit.jl convention (the same `model` you pass to
[`chisq`](@ref) / [`model_fit`](@ref)).

This is a thin object-style surface over the shared χ² kernel: an
unmasked `LeastSquares(x,y,ye,model)(par)` is **bit-identical** to
`chisq(model, Data(x,y,ye), par)`. Both are just `_chisq_core`.

`mask` is a `BitVector` (`true` keeps the point) applied with no data
copy. `name` (a vector of names) is only needed to compose this cost
with `+`; `Minuit(LeastSquares(...), x0)` otherwise names parameters
`x0, x1, …` like the bare-FCN constructor.

# Examples
```julia
julia> c = LeastSquares([0,1,2], [1,3,5], [0.1,0.1,0.1], (x,p)->p[1]*x+p[2]);

julia> c([2.0, 1.0])         # perfect fit ⇒ χ² = 0
0.0

julia> m = Minuit(c, [1.0, 0.0]); migrad!(m); m.values
2-element ...:
 2.0
 1.0
```
"""
struct LeastSquares{F} <: AbstractCost
    data::Data
    model::F
    mask::Union{Nothing,BitVector}
    active::Union{Nothing,Vector{Int}}
    pnames::Union{Nothing,Vector{Symbol}}
end

function LeastSquares(data::Data, model::F; mask = nothing, name = nothing) where {F}
    bm = _to_bitmask(mask, data.ndata)
    return LeastSquares{F}(data, model, bm, _active_from_mask(bm), _to_pnames(name))
end
LeastSquares(x, y, yerror, model; kw...) =
    LeastSquares(Data(x, y, yerror), model; kw...)

function (c::LeastSquares)(par)
    d = c.data
    if c.active === nothing
        return _chisq_core(c.model, d.x, d.y, d.err, par)
    end
    # Masked: integer-indexed @view (no data copy) — keeps @simd.
    return @views _chisq_core(c.model, d.x[c.active], d.y[c.active],
                              d.err[c.active], par)
end

# ─────────────────────────────────────────────────────────────────────────────
# Likelihood costs.
#
# All use the classic Minuit errordef-0.5 convention: the cost is the
# plain negative-log-likelihood −logL, so a 1σ excursion raises it by
# 0.5. (iminuit instead returns 2·(−logL) with errordef 1; both give
# IDENTICAL parameter values and uncertainties — the reported fval just
# differs by the factor 2. JuMinuit follows IMinuit.jl's 0.5 choice.)
# ─────────────────────────────────────────────────────────────────────────────

# Smallest normal Float64 — the floor inside every `log` so a degenerate
# evaluation (a `pdf`/`μ` that hits 0 at the edge of support, or a
# transient non-positive value while the optimizer probes a bad region)
# yields a large-but-finite term instead of `-Inf`/`NaN`-ing the gradient
# or throwing `DomainError`. Applied as `log(max(·, _NLL_TINY))`: a no-op
# for any valid positive value (`max` returns it unchanged), and it also
# tolerates a tiny negative undershoot (which iminuit's purely-additive
# `_TINY_FLOAT` would not). Same spirit as iminuit's `_TINY_FLOAT`
# (= np.finfo(float).tiny) safeguard.
const _NLL_TINY = floatmin(Float64)

# Σ log(pdf(xᵢ, par)) over `eachindex(x)` (x a Vector or a @view). When
# `islog`, `pdf` already returns the log-density (assumed finite — the
# caller owns that path). Shared by UnbinnedNLL and ExtendedUnbinnedNLL.
@inline function _nll_logsum(pdf::F, x, par, islog::Bool) where {F}
    s = 0.0
    if islog
        @inbounds @simd for i in eachindex(x)
            s += pdf(x[i], par)
        end
    else
        @inbounds @simd for i in eachindex(x)
            s += log(max(pdf(x[i], par), _NLL_TINY))
        end
    end
    return s
end

"""
    UnbinnedNLL(x, pdf; log=false, mask=nothing, name=nothing)

Unbinned negative-log-likelihood ``-\\sum_i \\log \\mathrm{pdf}(x_i, par)``
(errordef `0.5`). `pdf(x_scalar, par)` must be **normalised** over the
observable range. Pass `log=true` if your function returns the
log-density directly (numerically safer in the tails).

`mask` (`true` keeps the sample) and `name` behave as in
[`LeastSquares`](@ref).
"""
struct UnbinnedNLL{F} <: AbstractCost
    x::Vector{Float64}
    pdf::F
    log::Bool
    mask::Union{Nothing,BitVector}
    active::Union{Nothing,Vector{Int}}
    pnames::Union{Nothing,Vector{Symbol}}
end

function UnbinnedNLL(x, pdf::F; log::Bool = false, mask = nothing,
                     name = nothing) where {F}
    xx = collect(Float64, x)
    bm = _to_bitmask(mask, length(xx))
    return UnbinnedNLL{F}(xx, pdf, log, bm, _active_from_mask(bm), _to_pnames(name))
end

function (c::UnbinnedNLL)(par)
    if c.active === nothing
        return -_nll_logsum(c.pdf, c.x, par, c.log)
    end
    return -(@views _nll_logsum(c.pdf, c.x[c.active], par, c.log))
end

"""
    ExtendedUnbinnedNLL(x, density, integral; log=false, mask=nothing, name=nothing)

Extended unbinned negative-log-likelihood
``\\mu(par) - \\sum_i \\log \\rho(x_i, par)`` (errordef `0.5`), where
`density` is the differential intensity ``\\rho = \\mathrm{d}N/\\mathrm{d}x``
(`density(x_scalar, par)`) and `integral(par)` is its integral over the
observable range — the expected total event count ``\\mu``. The extended
likelihood fits the normalisation as well as the shape.

This is the scalar-convention split of iminuit's `scaled_pdf`, which
returns `(integral, density_array)` from one call. Pass `log=true` if
`density` returns its own logarithm.
"""
struct ExtendedUnbinnedNLL{F,G} <: AbstractCost
    x::Vector{Float64}
    density::F
    integral::G
    log::Bool
    mask::Union{Nothing,BitVector}
    active::Union{Nothing,Vector{Int}}
    pnames::Union{Nothing,Vector{Symbol}}
end

function ExtendedUnbinnedNLL(x, density::F, integral::G; log::Bool = false,
                             mask = nothing, name = nothing) where {F,G}
    xx = collect(Float64, x)
    bm = _to_bitmask(mask, length(xx))
    return ExtendedUnbinnedNLL{F,G}(xx, density, integral, log, bm,
                                    _active_from_mask(bm), _to_pnames(name))
end

function (c::ExtendedUnbinnedNLL)(par)
    if c.active === nothing
        return c.integral(par) - _nll_logsum(c.density, c.x, par, c.log)
    end
    return c.integral(par) -
           (@views _nll_logsum(c.density, c.x[c.active], par, c.log))
end

# Evaluate the cdf at every bin edge once (nb+1 calls) into a fresh
# buffer, so each bin's two edges are computed once (not once per adjacent
# bin) — important when `cdf` is expensive (the HEP norm). The buffer is
# fresh per call ⇒ thread-safe under `threaded_gradient` (a pooled/cached
# buffer would race); and it lets the multinomial path do its two passes
# (normalise to `ptot`, then accumulate) without re-evaluating the cdf.
# The cost is one small `O(nb)` array per FCN call — measured negligible
# (≈0.5 KB for 50 bins; a full 50-bin MIGRAD allocates ~0.02 MB total).
@inline function _edge_cdf(cdf::F, xe, par, nb::Int) where {F}
    v = Vector{Float64}(undef, nb + 1)
    @inbounds for k in 1:(nb + 1)
        v[k] = Float64(cdf(xe[k], par))
    end
    return v
end

# Multinomial (non-extended) binned NLL over bins `idx`, conditioning on
# the kept-bin total. Equals 0.5·multinominal_chi2(n, μ) up to the data
# constant, i.e. −Σ nᵢ log(μᵢ) shifted — so errordef 0.5.
function _binned_multinomial(n, cdfv, idx)
    ptot = 0.0
    ntot = 0.0
    @inbounds for i in idx
        ptot += cdfv[i + 1] - cdfv[i]
        ntot += n[i]
    end
    ptot = max(ptot, _NLL_TINY)               # guard a degenerate cdf
    res = 0.0
    @inbounds for i in idx
        p_i = cdfv[i + 1] - cdfv[i]
        mu_i = max(ntot * p_i / ptot, _NLL_TINY)
        n_i = n[i]
        n_i > 0 && (res += n_i * log(n_i / mu_i))
    end
    return res
end

# Poisson (extended) binned NLL over bins `idx`. Equals
# 0.5·poisson_chi2(n, μ) — the Baker–Cousins likelihood ratio scaled to
# errordef 0.5. `μᵢ` are expected COUNTS from the scaled cdf.
function _binned_poisson(n, cdfv, idx)
    res = 0.0
    @inbounds for i in idx
        mu_i = max(cdfv[i + 1] - cdfv[i], _NLL_TINY)   # guard log(n/μ), μ≥0
        n_i = n[i]
        if n_i > 0
            res += mu_i - n_i + n_i * log(n_i / mu_i)
        else
            res += mu_i
        end
    end
    return res
end

"""
    BinnedNLL(n, xe, cdf; mask=nothing, name=nothing)

Binned negative-log-likelihood for histogram counts `n` (length `nbins`)
with bin edges `xe` (length `nbins+1`). `cdf(x_scalar, par)` is the
**normalised** cumulative distribution; the per-bin probability is
``p_i = \\mathrm{cdf}(x_{i+1}) - \\mathrm{cdf}(x_i)``, and the fit uses the
multinomial likelihood (conditioning on the observed total). errordef
`0.5`. Use [`ExtendedBinnedNLL`](@ref) to also fit the normalisation.

The value equals `0.5 * multinominal_chi2(n, μ)` (the existing
likelihood-ratio kernel), with `μ` scaled to the kept-bin total.

`mask` (`true` keeps the bin) and `name` behave as in [`LeastSquares`](@ref).
"""
struct BinnedNLL{F} <: AbstractCost
    n::Vector{Float64}
    xe::Vector{Float64}
    cdf::F
    mask::Union{Nothing,BitVector}
    active::Union{Nothing,Vector{Int}}
    pnames::Union{Nothing,Vector{Symbol}}
end

function BinnedNLL(n, xe, cdf::F; mask = nothing, name = nothing) where {F}
    nn = collect(Float64, n)
    ee = collect(Float64, xe)
    length(ee) == length(nn) + 1 || throw(ArgumentError(
        "BinnedNLL: length(xe) must be length(n)+1 (got $(length(ee)) and $(length(nn)))"))
    bm = _to_bitmask(mask, length(nn))
    return BinnedNLL{F}(nn, ee, cdf, bm, _active_from_mask(bm), _to_pnames(name))
end

function (c::BinnedNLL)(par)
    nb = length(c.n)
    cdfv = _edge_cdf(c.cdf, c.xe, par, nb)
    return c.active === nothing ?
           _binned_multinomial(c.n, cdfv, Base.OneTo(nb)) :
           _binned_multinomial(c.n, cdfv, c.active)
end

"""
    ExtendedBinnedNLL(n, xe, scaled_cdf; mask=nothing, name=nothing)

Extended binned negative-log-likelihood: like [`BinnedNLL`](@ref) but
`scaled_cdf(x_scalar, par)` returns the **expected cumulative count**
(its integral is the expected total), so the per-bin expectation is
``\\mu_i = \\mathrm{scaled\\_cdf}(x_{i+1}) - \\mathrm{scaled\\_cdf}(x_i)`` and
the normalisation is fitted. Poisson likelihood; errordef `0.5`.

The value equals `0.5 * poisson_chi2(n, μ)`.
"""
struct ExtendedBinnedNLL{F} <: AbstractCost
    n::Vector{Float64}
    xe::Vector{Float64}
    cdf::F
    mask::Union{Nothing,BitVector}
    active::Union{Nothing,Vector{Int}}
    pnames::Union{Nothing,Vector{Symbol}}
end

function ExtendedBinnedNLL(n, xe, cdf::F; mask = nothing, name = nothing) where {F}
    nn = collect(Float64, n)
    ee = collect(Float64, xe)
    length(ee) == length(nn) + 1 || throw(ArgumentError(
        "ExtendedBinnedNLL: length(xe) must be length(n)+1 (got $(length(ee)) and $(length(nn)))"))
    bm = _to_bitmask(mask, length(nn))
    return ExtendedBinnedNLL{F}(nn, ee, cdf, bm, _active_from_mask(bm), _to_pnames(name))
end

function (c::ExtendedBinnedNLL)(par)
    nb = length(c.n)
    cdfv = _edge_cdf(c.cdf, c.xe, par, nb)
    return c.active === nothing ?
           _binned_poisson(c.n, cdfv, Base.OneTo(nb)) :
           _binned_poisson(c.n, cdfv, c.active)
end

# ─────────────────────────────────────────────────────────────────────────────
# CostSum — composition by operator overloading (simultaneous fits).
#
# The combined objective is the sum of the components, each rescaled to a
# COMMON χ² scale by 1/errordef so mixed-type sums stay statistically
# consistent: a 1σ move in any component raises the total by 1, hence
# errordef(CostSum) = 1. (A LeastSquares contributes χ² unchanged; an NLL
# contributes 2·(−logL) = its χ²-equivalent — exactly iminuit's
# all-errordef-1 combination, so a mixed simultaneous fit lands on the
# same minimum and the same uncertainties as iminuit.)
#
# The parameter set is the union of the components' names, ordered by
# first appearance. Each component is fed the `@view` of the global
# parameter vector at its own indices — parameters with a shared name are
# genuinely shared across components.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CostSum(costs...)
    c1 + c2

Sum of cost functions for a simultaneous fit. The parameter set is the
union (by name) of the components, so components sharing a parameter name
share that parameter. Works for mixed cost types
(`LeastSquares + UnbinnedNLL`).

Every component must carry parameter names (`name=` at construction);
the union becomes the default parameter names of `Minuit(c1+c2, x0)`.
errordef is `1.0`: each component is rescaled by `1/errordef` onto the
common χ² scale, so the combined fit's uncertainties match a single
joint likelihood (and iminuit).
"""
struct CostSum{C<:Tuple,M<:Tuple} <: AbstractCost
    costs::C            # flattened tuple of leaf (non-CostSum) costs
    maps::M             # NTuple of Vector{Int}: maps[k] = global indices of costs[k]
    pnames::Vector{Symbol}
end

# Flatten nested sums so `(a+b)+c` yields one flat CostSum.
_leaves(c::AbstractCost) = (c,)
_leaves(c::CostSum) = c.costs
_flatten(::Tuple{}) = ()
_flatten(t::Tuple) = (_leaves(t[1])..., _flatten(Base.tail(t))...)

function CostSum(costs::Vararg{AbstractCost})
    leaves = _flatten(costs)
    isempty(leaves) && throw(ArgumentError("CostSum needs at least one cost"))
    pnames = Symbol[]
    for c in leaves
        nm = parameter_names(c)
        nm === nothing && throw(ArgumentError(
            "cannot compose a cost with unnamed parameters; build it with " *
            "`name=[:a, :b, …]` (got a $(nameof(typeof(c))))"))
        for s in nm
            s in pnames || push!(pnames, s)
        end
    end
    index = Dict{Symbol,Int}(s => i for (i, s) in enumerate(pnames))
    maps = map(c -> Int[index[s] for s in parameter_names(c)], leaves)
    return CostSum(leaves, maps, pnames)
end

# Type-stable, compile-time-unrolled sum over the heterogeneous tuple of
# components (indexing a Tuple with a runtime counter would be dynamic).
@inline _summap(::Tuple{}, ::Tuple{}, par) = 0.0
@inline function _summap(costs::Tuple, maps::Tuple, par)
    c = costs[1]
    sub = @view par[maps[1]]
    return c(sub) / errordef(c) + _summap(Base.tail(costs), Base.tail(maps), par)
end
(cs::CostSum)(par) = _summap(cs.costs, cs.maps, par)

Base.:+(a::AbstractCost, b::AbstractCost) = CostSum(_leaves(a)..., _leaves(b)...)

# ─────────────────────────────────────────────────────────────────────────────
# errordef trait — dispatched on the cost type, NOT a stored field.
# (Defined here, after the types exist; the method signatures are
# evaluated at definition time.)
# ─────────────────────────────────────────────────────────────────────────────

"""
    errordef(c::AbstractCost) -> Float64

Minuit ErrorDef for the cost: `1.0` for χ² (`LeastSquares`, `CostSum`),
`0.5` for the likelihood costs. Pure trait dispatch on the type.

`Minuit(c, x0)` folds this into the underlying `CostFunction.up`, which
HESSE (`2·up·H⁻¹`) and MINOS then use to scale the parameter
uncertainties. (`errordef` is also defined on `CostFunction` in fcn.jl,
where it returns the stored `up`.)
"""
errordef(::LeastSquares) = 1.0
errordef(::UnbinnedNLL) = 0.5
errordef(::ExtendedUnbinnedNLL) = 0.5
errordef(::BinnedNLL) = 0.5
errordef(::ExtendedBinnedNLL) = 0.5
errordef(::CostSum) = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Data-point count for the rich-display χ²/ndf line (only shown when
# errordef == 1, so it is harmless metadata for the NLL costs).
# ─────────────────────────────────────────────────────────────────────────────
_cost_ndata(c::LeastSquares) =
    c.active === nothing ? c.data.ndata : length(c.active)
_cost_ndata(c::UnbinnedNLL) =
    c.active === nothing ? length(c.x) : length(c.active)
_cost_ndata(c::ExtendedUnbinnedNLL) =
    c.active === nothing ? length(c.x) : length(c.active)
_cost_ndata(c::BinnedNLL) =
    c.active === nothing ? length(c.n) : length(c.active)
_cost_ndata(c::ExtendedBinnedNLL) =
    c.active === nothing ? length(c.n) : length(c.active)
_cost_ndata(cs::CostSum) = sum(_cost_ndata, cs.costs)

# ─────────────────────────────────────────────────────────────────────────────
# Minuit(cost, x0) — the object-style front door onto the same core.
#
# Auto-extracts `up = errordef(cost)` (so no manual `up=`) and the data
# count (`m.ndata`), and uses the cost's parameter names (if any) as the
# defaults. Everything else forwards to the generic `Minuit(fcn, x0)`.
# ─────────────────────────────────────────────────────────────────────────────

"""
    Minuit(cost::AbstractCost, x0; kwargs...) -> Minuit

Build a `Minuit` fit from a cost object. The ErrorDef defaults to
`errordef(cost)` automatically (so you do not pass `up=`/`errordef=` —
though an explicit one still wins if you really want to override it),
the data-point count is recorded for the χ²/ndf display line, and the
cost's parameter names — when it has any (always for a [`CostSum`](@ref))
— become the default parameter names. All other keyword arguments
(`error`, `limits`, `fixed`, `grad`, `strategy`, `tol`, …) flow through
to the generic `Minuit(fcn, x0)` constructor.
"""
function Minuit(cost::AbstractCost, x0::AbstractVector{<:Real}; kwargs...)
    kw = values(kwargs)
    pn = parameter_names(cost)
    user_named = haskey(kw, :name) || haskey(kw, :names)
    if pn !== nothing && !user_named
        length(pn) == length(x0) || throw(ArgumentError(
            "Minuit(cost, x0): cost has $(length(pn)) named parameters " *
            "$(pn), but x0 has length $(length(x0))"))
    end
    # errordef(cost) is the DEFAULT up; an explicit user up/errordef wins.
    # Either way it must be passed positionally to `invoke`, so resolve it
    # here and strip both keys from the forwarded set.
    up_resolved = haskey(kw, :errordef) ? Float64(kw.errordef) :
                  haskey(kw, :up)       ? Float64(kw.up)       : errordef(cost)
    fwd = Base.structdiff(kw, NamedTuple{(:up, :errordef)})
    name_kw = (pn !== nothing && !user_named) ? (; name = String.(pn)) : (;)
    # `invoke` the generic (fcn, x0) method directly so the cost object IS
    # the FCN (`m.fcn.f === cost`) without re-dispatching to this method.
    m = invoke(Minuit, Tuple{Any,AbstractVector{<:Real}}, cost, x0;
               up = up_resolved, name_kw..., fwd...)
    m.ndata = _cost_ndata(cost)
    return m
end

"""
    Minuit(cost::AbstractCost, fit::Minuit; kwargs...) -> Minuit

Rebuild a fresh fit of `cost` reusing `fit`'s latest values, names,
errors, bounds and tuning as the starting point — the cost-object
counterpart of `Minuit(fcn, ::Minuit)`. Routes through `Minuit(cost, x0)`
so the ErrorDef and data count come from `errordef(cost)` / `cost`, not
from `fit` (which may have been built with a different cost type).
"""
function Minuit(cost::AbstractCost, fit::Minuit; kwargs...)
    # Recover the start config from `fit` (mirrors Minuit(fcn, ::Minuit)),
    # then route through Minuit(cost, x0) so up = errordef(cost) + ndata.
    x0 = fit.fmin === nothing ? [p.value for p in fit.params.pars] :
                                fit.fmin.ext_values
    nm = [p.name for p in fit.params.pars]
    er = fit.fmin === nothing ? [p.error for p in fit.params.pars] :
                                fit.fmin.ext_errors
    fx = [is_fixed(p) for p in fit.params.pars]
    lim = Vector{Any}(undef, n_pars(fit.params))
    for (i, p) in enumerate(fit.params.pars)
        lo = isnan(p.lower) ? nothing : p.lower
        hi = isnan(p.upper) ? nothing : p.upper
        lim[i] = (lo === nothing && hi === nothing) ? nothing : (lo, hi)
    end
    return Minuit(cost, x0; name = nm, error = er, fixed = fx, limits = lim,
                  prec = fit.prec, strategy = fit.strategy, tol = fit.tol,
                  print_level = fit.print_level,
                  threaded_gradient = fit.threaded_gradient,
                  verify_threading = fit.verify_threading, kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# show
# ─────────────────────────────────────────────────────────────────────────────
function _show_cost(io::IO, c::AbstractCost, n::Int)
    print(io, nameof(typeof(c)), "(ndata=", n, ", errordef=", errordef(c))
    pn = parameter_names(c)
    pn === nothing || print(io, ", params=", Tuple(pn))
    c isa CostSum || (getfield(c, :mask) === nothing || print(io, ", masked"))
    print(io, ")")
end
Base.show(io::IO, c::AbstractCost) = _show_cost(io, c, _cost_ndata(c))
function Base.show(io::IO, c::CostSum)
    print(io, "CostSum(", join((nameof(typeof(x)) for x in c.costs), " + "),
          "; params=", Tuple(c.pnames), ", ndata=", _cost_ndata(c), ")")
end
