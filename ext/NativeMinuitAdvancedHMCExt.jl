# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# NativeMinuitAdvancedHMCExt — gradient-based NUTS posterior sampling.
#
# Activated automatically when AdvancedHMC + LogDensityProblems +
# LogDensityProblemsAD + TransformVariables + ForwardDiff are all loaded
# alongside NativeMinuit. Implements `NativeMinuit._posterior_sample_nuts`, the
# `sampler = :nuts` path of `posterior_sample`.
#
# Design (matches the SOTA memo / consensus design):
#   • Bounded parameters are mapped to UNCONSTRAINED ℝ by TransformVariables
#     (log / logit), NOT by Minuit's internal sin/√ transform, and the
#     log-Jacobian of that map is added to the log-density so NUTS samples the
#     correct target. The samples are transformed back to full EXTERNAL
#     coordinates before they enter the `PosteriorSample`.
#   • The gradient comes from ForwardDiff. If the FCN cannot be differentiated,
#     construction FAILS LOUDLY (no finite-difference fallback — a noisy gradient
#     would silently wreck NUTS); the user is pointed at `sampler = :stretch`.
# ─────────────────────────────────────────────────────────────────────────────

module NativeMinuitAdvancedHMCExt

using NativeMinuit
using AdvancedHMC
using LogDensityProblems
using LogDensityProblemsAD
using TransformVariables
using ForwardDiff
using Random
using Statistics

import NativeMinuit: PosteriorProblem, PosteriorSample, LikelihoodEnsemble,
                 _posterior_rhat, _posterior_ess, _boundary_flags, _chain_seed

# Build the unconstraining transform from the per-free-coordinate limits
# (`NaN` ⇒ unbounded on that side). The effective support already folds in the
# prior support (`PosteriorProblem` intersected it), so the image of the
# transform is exactly where the posterior is supported.
function _build_transform(lo_free, hi_free)
    pieces = map(eachindex(lo_free)) do k
        lo = isnan(lo_free[k]) ? -TransformVariables.∞ : lo_free[k]
        hi = isnan(hi_free[k]) ?  TransformVariables.∞ : hi_free[k]
        as(Real, lo, hi)
    end
    return as(Tuple(pieces))
end

# LogDensityProblems target: the posterior in UNCONSTRAINED coordinates, i.e.
# `log p(θ(y)) + log|det J(y)|`, with `θ(y)` the constrained point. Generic in
# the number type so ForwardDiff can push Duals through.
struct PosteriorLogDensity{P,T}
    prob::P
    t::T
    nfree::Int
end

LogDensityProblems.dimension(ℓ::PosteriorLogDensity) = ℓ.nfree
LogDensityProblems.capabilities(::Type{<:PosteriorLogDensity}) =
    LogDensityProblems.LogDensityOrder{0}()

# Splice the free coordinates `θ` into a fresh full external vector of element
# type `R` (a fresh allocation is required anyway — ForwardDiff needs Duals).
@inline function _full(prob, θ, R, nfree)
    full = Vector{R}(undef, length(prob.best))
    @inbounds for i in eachindex(full)
        full[i] = prob.best[i]
    end
    @inbounds for j in 1:nfree
        full[prob.free_idx[j]] = θ[j]
    end
    return full
end

function LogDensityProblems.logdensity(ℓ::PosteriorLogDensity, y)
    prob = ℓ.prob
    θ, logj = TransformVariables.transform_and_logjac(ℓ.t, y)
    R = eltype(y)
    # SEPARATE buffers for the prior and the FCN, so a prior that mutates its
    # argument cannot shift the FCN point (mirrors `_eval_posterior!`).
    lp = prob.prior.logdensity(_full(prob, θ, R, ℓ.nfree))
    isfinite(lp) || return convert(R, -Inf)
    return -prob.fcn(_full(prob, θ, R, ℓ.nfree)) / (2 * prob.up) + lp + logj
end

# Constrained-space (fval, logprior) at a free-coordinate vector — for assembling
# the kept rows. The reported `logpost` is the CONSTRAINED posterior (no
# Jacobian); the Jacobian only enters the sampling target above. Separate buffers
# again, so neither a mutating FCN nor a mutating prior can corrupt the other.
function _eval_constrained(prob, θfree)
    nfree = length(prob.free_idx)
    c = Float64(prob.fcn(_full(prob, θfree, Float64, nfree)))
    lp = Float64(prob.prior.logdensity(_full(prob, θfree, Float64, nfree)))
    return c, lp
end

# Called from `NativeMinuit._posterior_sample_nuts` via `Base.get_extension`, so the
# stub and this implementation never share a method signature (no precompile
# method-overwrite).
function _nuts_impl(prob::PosteriorProblem; nsteps::Integer,
                    burn::Integer, thin::Integer, nchains::Integer,
                    target_accept::Real, diagnostics::Bool,
                    seed, rng, warn::Bool)
    nfree = length(prob.free_idx)
    ntot = length(prob.best)
    t = _build_transform(prob.lo_free, prob.hi_free)
    TransformVariables.dimension(t) == nfree ||
        throw(ArgumentError("internal error: transform dimension ≠ n_free"))
    pbest = prob.best[prob.free_idx]

    # Best fit in unconstrained coordinates — undefined if the best fit sits ON a
    # limit (the logit/log map sends a boundary to ±∞). NUTS cannot start there.
    y0 = TransformVariables.inverse(t, Tuple(pbest))
    all(isfinite, y0) || throw(ArgumentError(
        "sampler = :nuts: the best-fit point is on a parameter limit, where the " *
        "unconstrained (log/logit) transform is singular. Move off the boundary, " *
        "loosen the limit, or use sampler = :stretch (which handles boundaries)."))

    ℓ = PosteriorLogDensity(prob, t, nfree)
    ∂ℓ = LogDensityProblemsAD.ADgradient(:ForwardDiff, ℓ)
    # Probe the gradient once: a non-differentiable FCN throws here, and we
    # convert that into a clear, actionable error rather than a deep stack trace.
    try
        LogDensityProblems.logdensity_and_gradient(∂ℓ, collect(Float64, y0))
    catch err
        throw(ArgumentError(
            "sampler = :nuts: the FCN or the prior is not auto-differentiable with " *
            "ForwardDiff ($(sprint(showerror, err))). NUTS here builds the gradient with " *
            "ForwardDiff only — use the gradient-free sampler = :stretch instead."))
    end

    all_samples = Matrix{Float64}(undef, 0, ntot)
    all_fvals = Float64[]
    all_loglik = Float64[]
    all_logpost = Float64[]
    chain_ids = Int[]
    accs = Float64[]
    fbest = first(_eval_constrained(prob, pbest))

    for ch in 1:nchains
        rng_ch = seed !== nothing ? Random.MersenneTwister(_chain_seed(UInt64(seed), ch)) :
                 rng !== nothing ? rng : Random.default_rng()
        init = ch == 1 ? collect(Float64, y0) :
               collect(Float64, y0) .+ randn(rng_ch, nfree)   # over-dispersed start
        metric = DiagEuclideanMetric(nfree)
        ham = Hamiltonian(metric, ∂ℓ)
        # Pass the per-chain RNG: the no-RNG overload would draw the trial momenta
        # from the global default RNG, breaking same-seed reproducibility and
        # consuming global RNG state.
        ϵ = find_good_stepsize(rng_ch, ham, init)
        integrator = Leapfrog(ϵ)
        kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))  # NUTS
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
                                 StepSizeAdaptor(target_accept, integrator))
        ys, stats = AdvancedHMC.sample(rng_ch, ham, kernel, init, nsteps, adaptor, burn;
                                       drop_warmup = true, progress = false, verbose = false)
        acc = mean(s -> s.acceptance_rate, stats)
        thin > 1 && (ys = ys[thin:thin:end])      # honor `thin` (NUTS rarely needs it)
        nkept = length(ys)
        block = Matrix{Float64}(undef, nkept, ntot)
        for (r, y) in enumerate(ys)
            θ, _ = TransformVariables.transform_and_logjac(t, y)
            for i in 1:ntot
                block[r, i] = prob.best[i]
            end
            for (j, i) in enumerate(prob.free_idx)
                block[r, i] = θ[j]
            end
            c, lp = _eval_constrained(prob, ntuple(j -> θ[j], nfree))
            push!(all_fvals, c)
            push!(all_loglik, -c / (2 * prob.up))
            push!(all_logpost, -c / (2 * prob.up) + lp)
            push!(chain_ids, ch)
        end
        all_samples = vcat(all_samples, block)
        push!(accs, acc)
    end

    free = falses(length(prob.names)); free[prob.free_idx] .= true
    ens = LikelihoodEnsemble(all_samples, all_fvals, prob.names, collect(free),
                             prob.best, fbest, prob.up, mean(accs),
                             Int(nsteps), Int(burn), Int(thin), NaN,   # no proposal scale
                             :nuts, seed === nothing ? nothing : UInt64(seed))
    rh = diagnostics ? _posterior_rhat(ens.samples, chain_ids, prob.free_idx, Int(nchains)) :
                       fill(NaN, nfree)
    es = diagnostics ? _posterior_ess(ens.samples, chain_ids, prob.free_idx, Int(nchains)) :
                       fill(NaN, nfree)
    boundary = _boundary_flags(ens.samples, prob.free_idx, prob.lo_free, prob.hi_free)
    warns = String[]
    any(boundary) && push!(warns, "posterior mass is near at least one active parameter limit")
    return PosteriorSample(ens, prob.prior, :nuts, nchains, chain_ids,
                           all_loglik, all_logpost, rh, es, boundary, warns)
end

end # module
