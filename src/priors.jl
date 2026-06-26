# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# priors.jl — lightweight prior objects for the Bayesian bridge.
#
# Priors are log densities over the FULL external parameter vector. They are
# deliberately small: no new dependency, no model DSL, no attempt to certify
# posterior propriety. Minuit limits are intersected later by PosteriorProblem.
# ─────────────────────────────────────────────────────────────────────────────

"""
    Prior

Log-prior provenance for [`posterior_sample`](@ref) / [`bayesian`](@ref).
`logdensity(θ_full)` receives the full external parameter vector and returns
an unnormalized log density. `support_lo`/`support_hi` are full-length support
vectors (`±Inf` for unbounded coordinates). `informative` marks parameters
with explicit non-flat prior components and is used by [`combine_priors`](@ref)
to reject accidental overlap.

`all_components_proper` is provenance only; it is **not** a proof that the
posterior is proper.
"""
struct Prior
    logdensity::Function
    name::Symbol
    description::String
    support_lo::Vector{Float64}
    support_hi::Vector{Float64}
    names::Vector{String}
    all_components_proper::Bool
    informative::Vector{Bool}

    function Prior(logdensity::Function, name::Symbol, description::AbstractString,
                   support_lo::AbstractVector{<:Real},
                   support_hi::AbstractVector{<:Real},
                   names::AbstractVector{<:AbstractString},
                   all_components_proper::Bool,
                   informative::AbstractVector{Bool})
        n = length(names)
        length(support_lo) == n ||
            throw(DimensionMismatch("support_lo length $(length(support_lo)) != number of parameters $n"))
        length(support_hi) == n ||
            throw(DimensionMismatch("support_hi length $(length(support_hi)) != number of parameters $n"))
        length(informative) == n ||
            throw(DimensionMismatch("informative length $(length(informative)) != number of parameters $n"))
        lo = Float64.(collect(support_lo))
        hi = Float64.(collect(support_hi))
        @inbounds for i in 1:n
            (lo[i] == -Inf || isfinite(lo[i])) ||
                throw(ArgumentError("support_lo for $(names[i]) must be finite or -Inf, got $(lo[i])"))
            (hi[i] == Inf || isfinite(hi[i])) ||
                throw(ArgumentError("support_hi for $(names[i]) must be finite or +Inf, got $(hi[i])"))
            lo[i] <= hi[i] ||
                throw(ArgumentError("empty prior support for parameter $(names[i]): [$(lo[i]), $(hi[i])]"))
        end
        return new(logdensity, name, String(description), lo, hi,
                   String.(collect(names)), Bool(all_components_proper),
                   collect(informative))
    end
end

_prior_names(m::Minuit) = [p.name for p in _init_params(m).pars]

function _prior_index(names::Vector{String}, par)
    if par isa Integer
        idx = Int(par)
        1 <= idx <= length(names) ||
            throw(ArgumentError("parameter index $idx out of range 1:$(length(names))"))
        return idx
    else
        s = String(par)
        idx = findfirst(==(s), names)
        idx === nothing && throw(KeyError("parameter \"$s\" not found"))
        return idx
    end
end

"""
    flat_prior(m::Minuit) -> Prior

Flat prior in JuMinuit's full external parameter coordinates. This is not
"no prior"; it is a parameterization-dependent coordinate choice. On unbounded
coordinates posterior propriety relies on the likelihood.
"""
function flat_prior(m::Minuit)
    names = _prior_names(m)
    n = length(names)
    return Prior(_ -> 0.0, :flat,
                 "flat in external coordinates; parameterization-dependent",
                 fill(-Inf, n), fill(Inf, n), names, false, fill(false, n))
end

"""
    normal_prior(m::Minuit, par, μ, σ) -> Prior

Gaussian prior on parameter `par`, flat on the remaining coordinates.
The log density is unnormalized: `-0.5*((x-μ)/σ)^2`.
"""
function normal_prior(m::Minuit, par, μ::Real, σ::Real)
    σf = Float64(σ)
    (isfinite(σf) && σf > 0) ||
        throw(ArgumentError("normal_prior σ must be finite and > 0, got $σf"))
    isfinite(Float64(μ)) || throw(ArgumentError("normal_prior μ must be finite, got $μ"))
    names = _prior_names(m)
    idx = _prior_index(names, par)
    μf = Float64(μ)
    info = fill(false, length(names)); info[idx] = true
    # Generic over the element type of θ so the gradient samplers (NUTS) can push
    # ForwardDiff Duals through — no `Float64(θ[idx])` coercion on the live value.
    logp = θ -> begin
        x = θ[idx]
        return -0.5 * ((x - μf) / σf)^2
    end
    return Prior(logp, :normal,
                 "normal prior on $(names[idx]): μ=$μf, σ=$σf",
                 fill(-Inf, length(names)), fill(Inf, length(names)),
                 names, true, info)
end

"""
    uniform_prior(m::Minuit, par, lo, hi) -> Prior

Proper uniform prior support for one parameter, flat on the remaining
coordinates. Returns `-Inf` outside `[lo, hi]`.
"""
function uniform_prior(m::Minuit, par, lo::Real, hi::Real)
    lof, hif = Float64(lo), Float64(hi)
    (isfinite(lof) && isfinite(hif)) ||
        throw(ArgumentError("uniform_prior needs finite lo, hi for a proper prior, got [$lof, $hif]"))
    lof < hif || throw(ArgumentError("uniform_prior needs lo < hi, got [$lof, $hif]"))
    names = _prior_names(m)
    idx = _prior_index(names, par)
    slo = fill(-Inf, length(names)); shi = fill(Inf, length(names))
    slo[idx] = lof; shi[idx] = hif
    info = fill(false, length(names)); info[idx] = true
    logp = θ -> begin
        x = θ[idx]
        return (lof <= x <= hif) ? 0.0 : -Inf
    end
    return Prior(logp, :uniform,
                 "uniform prior on $(names[idx]) over [$lof, $hif]",
                 slo, shi, names, true, info)
end

"""
    half_normal_prior(m::Minuit, par, σ) -> Prior

Half-normal prior for a non-negative or lower-bounded parameter. If `par`
has a finite Minuit lower limit, the half-normal is centered at that lower
limit; otherwise it is centered at zero and supports `par >= 0`.
"""
function half_normal_prior(m::Minuit, par, σ::Real)
    σf = Float64(σ)
    (isfinite(σf) && σf > 0) ||
        throw(ArgumentError("half_normal_prior σ must be finite and > 0, got $σf"))
    names = _prior_names(m)
    idx = _prior_index(names, par)
    p = _init_params(m).pars[idx]
    center = has_lower_limit(p) ? p.lower : 0.0
    slo = fill(-Inf, length(names)); shi = fill(Inf, length(names))
    slo[idx] = center
    info = fill(false, length(names)); info[idx] = true
    logp = θ -> begin
        x = θ[idx]
        return x >= center ? -0.5 * ((x - center) / σf)^2 : -Inf
    end
    return Prior(logp, :half_normal,
                 "half-normal prior on $(names[idx]) above $center with σ=$σf",
                 slo, shi, names, true, info)
end

"""
    combine_priors(p1, p2, ...) -> Prior

Combine disjoint informative prior components by adding their log densities
and intersecting their supports. MVP behavior is deliberately strict:
two informative components on the same parameter raise an error.
"""
function combine_priors(p::Prior, ps::Prior...)
    priors = (p, ps...)
    names = p.names
    for q in ps
        q.names == names ||
            throw(ArgumentError("combine_priors requires priors over the same parameter names"))
    end
    info = fill(false, length(names))
    lo = fill(-Inf, length(names))
    hi = fill(Inf, length(names))
    desc = String[]
    proper = true
    for q in priors
        overlap = info .& q.informative
        any(overlap) && throw(ArgumentError(
            "combine_priors: overlapping informative components for " *
            join(names[findall(overlap)], ", ")))
        info .|= q.informative
        lo = max.(lo, q.support_lo)
        hi = min.(hi, q.support_hi)
        any(lo .> hi) &&
            throw(ArgumentError("combine_priors produced empty support"))
        push!(desc, q.description)
        proper &= q.all_components_proper
    end
    logp = θ -> begin
        s = zero(eltype(θ))                 # AD-generic accumulator (Float64 or Dual)
        for q in priors
            v = q.logdensity(θ)
            isfinite(v) || return oftype(s, -Inf)
            s += v
        end
        return s
    end
    return Prior(logp, :combined, join(desc, "; "), lo, hi, names, proper, info)
end
