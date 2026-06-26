# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# posterior.jl — non-mutating Bayesian bridge over the Minuit FCN.
# ─────────────────────────────────────────────────────────────────────────────

"""
    PosteriorProblem(m::Minuit; prior=:flat)

Snapshot of a Minuit fit plus an explicit prior, defining the posterior target

```julia
log p(θ | data) = -fcn(θ) / (2 * up) + logprior(θ)
```

in full external parameter coordinates. The object is non-mutating and stores a
snapshot of the fit state; use [`isconsistent`](@ref) to compare it with a later
`Minuit` object.

!!! warning "The posterior temperature follows `errordef`"
    The likelihood enters as `log L = -fcn/(2·up)`, the faithful Minuit relation
    `-2 log L = fcn/up`. With the **statistical** `errordef` — `up = 1` for a χ²
    (`-2 log L`) cost, `up = 0.5` for a `-log L` cost — the likelihood
    temperature is correct: under a **flat** prior on a near-Gaussian, interior
    fit the posterior width then matches the HESSE/MINOS errors (an informative
    prior or an active boundary will legitimately narrow or reshape it).
    But if you inflate `up` to make MINOS report an n-σ interval (e.g. `up = 4`
    for a 2σ χ² interval), the posterior is tempered by the **same** `√up`
    factor. Keep `errordef` at its statistical value (`1` or `0.5`) for Bayesian
    work; the prior, not `errordef`, is the place to encode extra information.

MVP semantics: Minuit limits are physical posterior support, intersected with
the prior support. If the current best point is outside that effective support,
construction fails loudly rather than starting a dead chain.
"""
struct PosteriorProblem{F}
    fcn::F                       # raw user FCN — may be a callable struct (a cost object)
    prior::Prior
    up::Float64
    free_idx::Vector{Int}
    names::Vector{String}
    best::Vector{Float64}
    fixed_values::Vector{Float64}
    lo_free::Vector{Float64}
    hi_free::Vector{Float64}
    errors_free::Vector{Float64}
    cov_free::Union{Nothing,Matrix{Float64}}
    cov_unreliable::Bool
    snapshot_signature::NamedTuple
    fval_free::Function
    loglik_free::Function
    logprior_free::Function
end

"""
    PosteriorSample

Posterior sample returned by [`posterior_sample`](@ref). The `ensemble` field is
a regular [`LikelihoodEnsemble`](@ref) in full external coordinates; its `fvals`
remain likelihood FCN values. Posterior-specific provenance lives in
`prior`, `loglik_kept`, `logpost_kept`, chain IDs, diagnostics, and warnings.
"""
struct PosteriorSample
    ensemble::LikelihoodEnsemble
    prior::Prior
    sampler::Symbol
    nchains::Int
    chain_ids::Vector{Int}
    loglik_kept::Vector{Float64}
    logpost_kept::Vector{Float64}
    rhat::Vector{Float64}
    ess::Vector{Float64}
    boundary_active::Vector{Bool}
    warnings::Vector{String}

    function PosteriorSample(ensemble::LikelihoodEnsemble, prior::Prior, sampler::Symbol,
                             nchains::Integer, chain_ids::AbstractVector{<:Integer},
                             loglik_kept::AbstractVector{<:Real},
                             logpost_kept::AbstractVector{<:Real},
                             rhat::AbstractVector{<:Real},
                             ess::AbstractVector{<:Real},
                             boundary_active::AbstractVector{Bool},
                             warnings::AbstractVector{<:AbstractString})
        n = length(ensemble)
        length(chain_ids) == n ||
            throw(DimensionMismatch("chain_ids length $(length(chain_ids)) != number of samples $n"))
        length(loglik_kept) == n ||
            throw(DimensionMismatch("loglik_kept length $(length(loglik_kept)) != number of samples $n"))
        length(logpost_kept) == n ||
            throw(DimensionMismatch("logpost_kept length $(length(logpost_kept)) != number of samples $n"))
        nfree = count(ensemble.free)
        length(rhat) == nfree ||
            throw(DimensionMismatch("rhat length $(length(rhat)) != number of free parameters $nfree"))
        length(ess) == nfree ||
            throw(DimensionMismatch("ess length $(length(ess)) != number of free parameters $nfree"))
        length(boundary_active) == nfree ||
            throw(DimensionMismatch("boundary_active length $(length(boundary_active)) != number of free parameters $nfree"))
        return new(ensemble, prior, sampler, Int(nchains), Int.(collect(chain_ids)),
                   Float64.(collect(loglik_kept)), Float64.(collect(logpost_kept)),
                   Float64.(collect(rhat)), Float64.(collect(ess)),
                   collect(boundary_active), String.(collect(warnings)))
    end
end

"""
    CredibleLimit

One-sided Bayesian credible limit returned by [`upper_limit`](@ref) or
[`lower_limit`](@ref). This is not a CLs, Feldman-Cousins, MINOS, or profile
confidence limit; it is conditional on the stated prior.
"""
struct CredibleLimit
    parameter::String
    limit::Float64
    level::Float64
    side::Symbol
    prior_name::Symbol
    boundary_active::Bool
end

"""
    BayesianReport

Non-mutating one-step Bayesian report returned by [`bayesian`](@ref). It holds
the reusable [`PosteriorSample`](@ref), the requested credibility `level`, the
interval method, and the parameter summary table.
"""
struct BayesianReport
    sample::PosteriorSample
    level::Float64
    interval::Symbol
    summary::Vector
end

Base.length(p::PosteriorSample) = length(p.ensemble)
Base.firstindex(p::PosteriorSample) = firstindex(p.ensemble)
Base.lastindex(p::PosteriorSample) = lastindex(p.ensemble)
Base.getindex(p::PosteriorSample, i::Integer) = p.ensemble[i]
Base.eltype(::Type{PosteriorSample}) = Vector{Float64}
Base.iterate(p::PosteriorSample, i::Int = 1) = iterate(p.ensemble, i)

function Base.show(io::IO, p::PosteriorSample)
    print(io, "PosteriorSample(", length(p), " samples, prior=:", p.prior.name,
          ", sampler=:", p.sampler, ")")
end

function Base.show(io::IO, ::MIME"text/plain", p::PosteriorSample)
    println(io, "PosteriorSample: ", length(p), " samples × ", count(p.ensemble.free),
            " free parameter", count(p.ensemble.free) == 1 ? "" : "s")
    println(io, "  prior :$(p.prior.name) — ", p.prior.description)
    unit = p.sampler === :stretch ? "walkers" : "chains"
    @printf(io, "  sampler :%s   %s %d   acceptance %.3f\n",
            p.sampler, unit, p.nchains, p.ensemble.acceptance)
    if p.nchains >= 2 && !isempty(p.rhat)
        @printf(io, "  max Rhat %.4g   min ESS %.4g\n", maximum(p.rhat), minimum(p.ess))
    else
        println(io, "  Rhat unavailable (single $unit)")
    end
    for w in p.warnings
        println(io, "  ⚠ ", w)
    end
end

function Base.show(io::IO, c::CredibleLimit)
    sgn = c.side === :upper ? "<" : ">"
    print(io, c.parameter, " ", sgn, " ", c.limit, " (",
          100 * c.level, "% Bayesian credible, prior=:", c.prior_name, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::BayesianReport)
    println(io, "BayesianReport: ", 100 * r.level, "% ", r.interval,
            " credible summaries")
    println(io, "  prior :$(r.sample.prior.name) — ", r.sample.prior.description)
    for row in r.summary
        println(io, "  ", row)
    end
end

function _resolve_prior(m::Minuit, prior)
    prior === :flat && return flat_prior(m)
    prior isa Prior && return prior
    throw(ArgumentError("MVP posterior prior must be `:flat` or a Prior object; got $(repr(prior))"))
end

function _snapshot_signature(m::Minuit, names, best, free, lo, hi, up)
    return (; names = copy(names), best = copy(best), free = copy(free),
            lower = copy(lo), upper = copy(hi), up = Float64(up))
end

function _current_signature(m::Minuit)
    # Read the static parameter config (names/limits/fixed) from `_init_params`,
    # not the `m.params` fit-overlay; the live best point comes from `m.values`.
    params = _init_params(m)
    names = [p.name for p in params.pars]
    best = collect(Float64, m.values)
    free = [!is_fixed(p) for p in params.pars]
    lo = [p.lower for p in params.pars]
    hi = [p.upper for p in params.pars]
    return _snapshot_signature(m, names, best, free, lo, hi, Float64(m.fcn.up))
end

# `isequal`, not `==`: open bounds are stored as `NaN`, and `NaN == NaN` is false
# (so `==` would report every unbounded fit as inconsistent), whereas
# `isequal(NaN, NaN)` is true.
"""
    isconsistent(prob::PosteriorProblem, m::Minuit) -> Bool

Return whether `m` still matches the fit snapshot captured in `prob`.
"""
isconsistent(prob::PosteriorProblem, m::Minuit) = isequal(prob.snapshot_signature, _current_signature(m))

function _limit_intersection(min_lo::Real, min_hi::Real, prior_lo::Real, prior_hi::Real)
    lo = isfinite(prior_lo) ? Float64(prior_lo) : NaN
    hi = isfinite(prior_hi) ? Float64(prior_hi) : NaN
    if !isnan(min_lo)
        lo = isnan(lo) ? Float64(min_lo) : max(lo, Float64(min_lo))
    end
    if !isnan(min_hi)
        hi = isnan(hi) ? Float64(min_hi) : min(hi, Float64(min_hi))
    end
    (!isnan(lo) && !isnan(hi) && lo > hi) &&
        throw(ArgumentError("empty effective posterior support"))
    return lo, hi
end

function PosteriorProblem(m::Minuit; prior = :flat)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `PosteriorProblem(m)` — the posterior starts at the best fit"))
    # `deepcopy` the prior into the snapshot (like the FCN below) so that a custom
    # prior closing over mutable state cannot retroactively change this posterior.
    pr = deepcopy(_resolve_prior(m, prior))
    # Static parameter config (names/limits/fixed) from `_init_params`, matching
    # priors.jl; the best-fit point itself is read from `m.values` below.
    params = _init_params(m)
    names = [p.name for p in params.pars]
    pr.names == names ||
        throw(ArgumentError("prior parameter names do not match this Minuit object"))

    ntot = n_pars(params)
    free_idx = [i for i in 1:ntot if !is_fixed(params.pars[i])]
    nfree = length(free_idx)
    nfree >= 1 || throw(ArgumentError("no free parameters to sample"))
    best = collect(Float64, m.values)
    up = Float64(m.fcn.up)
    (isfinite(up) && up > 0) || throw(ArgumentError("errordef (up) must be finite and > 0, got $up"))

    lo_all = [params.pars[i].lower for i in 1:ntot]
    hi_all = [params.pars[i].upper for i in 1:ntot]
    lo_free = Vector{Float64}(undef, nfree)
    hi_free = Vector{Float64}(undef, nfree)
    @inbounds for (j, i) in enumerate(free_idx)
        lo_free[j], hi_free[j] =
            _limit_intersection(lo_all[i], hi_all[i], pr.support_lo[i], pr.support_hi[i])
        if !isnan(lo_free[j]) && !isnan(hi_free[j]) && lo_free[j] == hi_free[j]
            throw(ArgumentError("zero-width effective posterior support for parameter $(names[i])"))
        end
    end
    # Fixed parameters never move, but a prior may still declare bounded support
    # on one. Verify the (fixed) best value lies inside the effective support, so
    # a malformed prior cannot pass construction and then report a fixed
    # coordinate sitting outside its own declared support.
    @inbounds for i in 1:ntot
        is_fixed(params.pars[i]) || continue
        flo, fhi = _limit_intersection(lo_all[i], hi_all[i], pr.support_lo[i], pr.support_hi[i])
        ((!isnan(flo) && best[i] < flo) || (!isnan(fhi) && best[i] > fhi)) &&
            throw(ArgumentError("fixed parameter $(names[i]) = $(best[i]) is outside its effective posterior support [$flo, $fhi]"))
    end

    # Freeze the FCN into the snapshot: `deepcopy` so that later mutation of the
    # user's cost object (e.g. `cost.data.y .= …`) or of a closure's captured data
    # cannot retroactively change this posterior. This is what makes
    # `PosteriorProblem` a genuine non-mutating snapshot rather than a live view.
    userf = deepcopy(m.fcn.f)
    fixed_values = copy(best)
    fval_free = let base = fixed_values, fi = free_idx, f = userf
        q -> begin
            full = copy(base)
            @inbounds for (j, i) in enumerate(fi)
                full[i] = q[j]
            end
            return Float64(f(full))
        end
    end
    logprior_free = let base = fixed_values, fi = free_idx, pr = pr
        q -> begin
            full = copy(base)
            @inbounds for (j, i) in enumerate(fi)
                full[i] = q[j]
            end
            return Float64(pr.logdensity(full))
        end
    end
    loglik_free = q -> -fval_free(q) / (2.0 * up)

    p0 = best[free_idx]
    for k in 1:nfree
        (!isnan(lo_free[k]) && p0[k] < lo_free[k]) &&
            throw(ArgumentError("best-fit point is outside effective posterior support; re-minimize with the prior or choose a compatible prior"))
        (!isnan(hi_free[k]) && p0[k] > hi_free[k]) &&
            throw(ArgumentError("best-fit point is outside effective posterior support; re-minimize with the prior or choose a compatible prior"))
    end
    lp0 = logprior_free(p0)
    isfinite(lp0) ||
        throw(ArgumentError("best-fit point has non-finite log-prior; re-minimize with the prior or choose a compatible prior"))

    cov = free_covariance(m.fmin)
    cov_free = cov === nothing ? nothing : Matrix{Float64}(cov)
    errors_free = collect(Float64, m.fmin.ext_errors)[free_idx]
    free = [!is_fixed(p) for p in params.pars]
    sig = _snapshot_signature(m, names, best, free, lo_all, hi_all, up)
    return PosteriorProblem(userf, pr, up, free_idx, names, best, fixed_values,
                            lo_free, hi_free, errors_free, cov_free,
                            _covariance_unreliable(m), sig,
                            fval_free, loglik_free, logprior_free)
end

function _posterior_proposal(prob::PosteriorProblem, proposal, warn::Bool)
    nfree = length(prob.free_idx)
    local steps_vec::Union{Nothing,Vector{Float64}} = nothing
    local Sfac::Union{Nothing,Matrix{Float64}} = nothing
    local prop_used::Symbol
    fallback_to_errors = false
    if proposal isa AbstractVector
        length(proposal) == nfree ||
            throw(DimensionMismatch("proposal σ vector has length $(length(proposal)), expected n_free = $nfree"))
        steps_vec = Float64.(collect(proposal))
        all(s -> isfinite(s) && s > 0, steps_vec) ||
            throw(ArgumentError("explicit proposal σ must all be finite and > 0"))
        prop_used = :steps
    elseif proposal isa AbstractMatrix
        size(proposal) == (nfree, nfree) ||
            throw(DimensionMismatch("proposal covariance is $(size(proposal)), expected ($nfree, $nfree)"))
        all(isfinite, proposal) ||
            throw(ArgumentError("explicit proposal covariance must be all-finite"))
        isposdef(Symmetric(Matrix{Float64}(proposal))) ||
            throw(ArgumentError("explicit proposal covariance must be positive-definite"))
        Sfac = _mvnormal_factor(proposal)
        prop_used = :matrix
    elseif proposal === :hesse
        if prob.cov_free === nothing
            fallback_to_errors = true
            warn && @warn "posterior_sample: no covariance available for proposal=:hesse; falling back to :errors"
        elseif prob.cov_unreliable
            fallback_to_errors = true
            warn && @warn "posterior_sample: fit covariance looks unreliable; falling back to :errors"
        else
            Sfac = _mvnormal_factor(prob.cov_free)
            prop_used = :hesse
        end
    elseif proposal === :errors
        fallback_to_errors = true
    else
        throw(ArgumentError("proposal must be :hesse, :errors, a σ vector, or a covariance matrix; got :$proposal"))
    end
    if fallback_to_errors
        all(e -> isfinite(e) && e > 0, prob.errors_free) ||
            throw(ArgumentError("the fit's parabolic errors are not all finite and positive — pass an explicit σ vector"))
        steps_vec = copy(prob.errors_free)
        prop_used = :errors
    end
    return steps_vec, Sfac, prop_used
end

function _chain_seed(seed::UInt64, chain_id::Integer)
    chain_id == 1 && return seed
    z = seed + UInt64(0x9e3779b97f4a7c15) * UInt64(chain_id - 1)
    z = (z ⊻ (z >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    z = (z ⊻ (z >> 27)) * UInt64(0x94d049bb133111eb)
    return z ⊻ (z >> 31)
end

function _draw_step!(out, rng, steps_vec, Sfac)
    n = length(out)
    if steps_vec !== nothing
        @inbounds for k in 1:n
            out[k] = steps_vec[k] * randn(rng)
        end
    else
        z = Vector{Float64}(undef, n)
        @inbounds for k in 1:n
            z[k] = randn(rng)
        end
        @inbounds for k in 1:n
            acc = 0.0
            for l in 1:n
                acc += Sfac[k, l] * z[l]
            end
            out[k] = acc
        end
    end
    return out
end

function _in_support(q, lo, hi)
    @inbounds for k in eachindex(q)
        (!isnan(lo[k]) && q[k] < lo[k]) && return false
        (!isnan(hi[k]) && q[k] > hi[k]) && return false
    end
    return true
end

# Draw an over-dispersed chain start `best + disp · step`, where `step` is one
# proposal-scale draw (∼ N(0, Σ) for :hesse/:matrix, σ-vector otherwise). `disp`
# is a multiple of the proposal/HESSE scale (default 2), so the start sits
# genuinely WIDER than the posterior — the condition Gelman/Vehtari R̂ needs to be
# a real convergence test. A few fresh draws are tried at each dispersion before
# it is halved, so a tight support narrows the start gracefully instead of
# collapsing it onto the MLE.
function _dispersed_start(best, rng, steps_vec, Sfac, lo, hi, disp, isvalid = _ -> true)
    q = copy(best)
    step = Vector{Float64}(undef, length(best))
    fac = Float64(disp)
    for attempt in 1:30
        _draw_step!(step, rng, steps_vec, Sfac)
        @inbounds for k in eachindex(best)
            q[k] = best[k] + fac * step[k]
        end
        _in_support(q, lo, hi) && isvalid(q) && return copy(q)
        attempt % 3 == 0 && (fac *= 0.5)
    end
    return nothing
end

"""
    posterior_sample(m::Minuit; prior=:flat, kwargs...) -> PosteriorSample
    posterior_sample(prob::PosteriorProblem; kwargs...) -> PosteriorSample

Sample the posterior `exp(-fcn/(2*up)) * prior` in external coordinates. The
returned object is independent of `m`; the fit state and `m.nfcn` are not mutated.
Two samplers are available:

- `sampler = :metropolis` (default) — a random-walk Metropolis chain, `nchains`
  of them (default 4). The proposal is set by `proposal` (`:hesse` / `:errors` /
  a σ vector / a covariance), tuned by `scale` and optional `target_accept`.
  Chains 2…n start **over-dispersed** at `overdisperse` × the proposal/HESSE
  scale from the best fit (default `2`, ≈2σ wider than the posterior) — the
  condition that makes the multi-chain split-R̂ a real convergence test.
- `sampler = :stretch` — the affine-invariant **ensemble** sampler (Goodman &
  Weare; the emcee kernel). `nwalkers` walkers (default `max(2·n_free+2, 8)`,
  rounded up to even) explore the posterior with stretch moves of scale
  `stretch` (`a`, default `2`). It is **gradient-free** (works for any FCN,
  including ones that cannot be auto-differentiated) and handles strongly
  correlated / skewed posteriors far better than a single random-walk chain. For
  `:stretch`, `nsteps` counts ensemble **iterations** (each updates every walker;
  defaults `nsteps=6000, burn=1000, thin=10`), and each walker is treated as a
  chain for the R̂ / ESS diagnostics. `proposal` / `scale` / `target_accept` /
  `overdisperse` are random-walk-only and ignored here.
- `sampler = :nuts` — gradient-based **NUTS** (No-U-Turn HMC), provided by the
  **AdvancedHMC extension** (load `AdvancedHMC, LogDensityProblems,
  LogDensityProblemsAD, TransformVariables, ForwardDiff` alongside JuMinuit).
  Bounded parameters are mapped to unconstrained ℝ with the proper log-Jacobian
  and sampled with a ForwardDiff gradient; it is the most efficient sampler for
  smooth, higher-dimensional posteriors but **requires an auto-differentiable
  FCN** (no finite-difference fallback — it errors and points to `:stretch`) and
  cannot start from a best fit on a parameter limit. The cost objects
  `LeastSquares` / `UnbinnedNLL` / `ExtendedUnbinnedNLL` / `CostSum` are
  ForwardDiff-differentiable and work with `:nuts`; `BinnedNLL` /
  `ExtendedBinnedNLL` are not (their CDF edge buffer is `Float64`), so use
  `:stretch` / `:metropolis` for those. Defaults `nsteps=2000` (per chain, incl.
  warmup), `burn=1000`, `target_accept=0.8`.

The reported `rhat` is the basic split-R̂ (not rank-normalized / folded); for
skewed or boundary-truncated marginals also check `effective_sample_size` and the
trace rather than trusting `R̂ < 1.01` alone. Bayesian credibility levels are
plain probabilities; no frequentist `nσ` overload is used.
"""
function posterior_sample(prob::PosteriorProblem;
                          sampler::Symbol = :metropolis,
                          nsteps::Union{Nothing,Integer} = nothing,
                          burn::Union{Nothing,Integer} = nothing,
                          thin::Union{Nothing,Integer} = nothing,
                          proposal::Union{Symbol,AbstractVector{<:Real},AbstractMatrix{<:Real}} = :hesse,
                          scale::Real = 0.3,
                          overdisperse::Real = 2.0,
                          target_accept::Union{Nothing,Real} = nothing,
                          adapt_every::Integer = 100,
                          nchains::Integer = 4,
                          nwalkers::Union{Nothing,Integer} = nothing,
                          stretch::Real = 2.0,
                          diagnostics::Bool = true,
                          seed::Union{Nothing,Integer} = nothing,
                          rng::Union{Nothing,Random.AbstractRNG} = nothing,
                          warn::Bool = true)
    sampler in (:metropolis, :stretch, :nuts) ||
        throw(ArgumentError("posterior_sample supports sampler ∈ (:metropolis, :stretch, :nuts), got :$sampler"))
    # Per-sampler default resolution: the samplers have very different step
    # economics (one chain × many steps; many walkers × fewer iterations;
    # gradient-based NUTS with a warmup), so nsteps/burn/thin carry
    # sampler-appropriate defaults when the caller leaves them unset.
    if sampler === :stretch
        nsteps = something(nsteps, 6_000)    # ensemble ITERATIONS (each updates every walker)
        burn = something(burn, 1_000)
        thin = something(thin, 10)
    elseif sampler === :nuts
        nsteps = something(nsteps, 2_000)    # total NUTS samples per chain (incl. warmup)
        burn = something(burn, 1_000)        # warmup / adaptation length
        thin = something(thin, 1)
    else
        nsteps = something(nsteps, 52_000)
        burn = something(burn, 2_000)
        thin = something(thin, 25)
    end
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1"))
    0 <= burn < nsteps || throw(ArgumentError("need 0 ≤ burn < nsteps, got burn=$burn, nsteps=$nsteps"))
    thin >= 1 || throw(ArgumentError("thin must be ≥ 1"))
    nsteps - burn >= thin ||
        throw(ArgumentError("no samples would be kept: need nsteps − burn ≥ thin"))
    nchains >= 1 || throw(ArgumentError("nchains must be ≥ 1"))
    seed !== nothing && rng !== nothing &&
        throw(ArgumentError("pass either `seed` or `rng`, not both"))
    seed !== nothing && !(0 <= seed <= typemax(UInt64)) &&
        throw(ArgumentError("seed must be in [0, typemax(UInt64)], got $seed"))

    sampler === :stretch && return _posterior_sample_stretch(
        prob; nwalkers = nwalkers, niter = Int(nsteps), burn = Int(burn),
        thin = Int(thin), stretch = stretch, diagnostics = diagnostics,
        seed = seed, rng = rng, warn = warn)

    if sampler === :nuts
        ta = target_accept === nothing ? 0.8 : Float64(target_accept)
        0 < ta < 1 || throw(ArgumentError("target_accept must be in (0, 1) for :nuts, got $ta"))
        return _posterior_sample_nuts(prob; nsteps = Int(nsteps), burn = Int(burn),
            thin = Int(thin), nchains = Int(nchains), target_accept = ta,
            diagnostics = diagnostics, seed = seed, rng = rng, warn = warn)
    end

    # ── :metropolis path. The random-walk-only knobs are validated here (they are
    #    ignored by :stretch / :nuts, so an unused value must not raise). ──
    (isfinite(scale) && scale > 0) || throw(ArgumentError("scale must be finite and > 0, got $scale"))
    (isfinite(overdisperse) && overdisperse > 0) ||
        throw(ArgumentError("overdisperse must be finite and > 0, got $overdisperse"))
    adapt_every >= 1 || throw(ArgumentError("adapt_every must be ≥ 1"))
    if target_accept !== nothing
        0 < target_accept < 1 ||
            throw(ArgumentError("target_accept must be in (0, 1), got $target_accept"))
        burn >= adapt_every ||
            throw(ArgumentError("target_accept adaptation runs during burn-in: need burn ≥ adapt_every"))
    end

    steps_vec, Sfac, prop_used = _posterior_proposal(prob, proposal, warn)
    pbest = prob.best[prob.free_idx]
    isfinite(prob.logprior_free(pbest)) ||
        throw(ArgumentError("best-fit point has non-finite log-prior; re-minimize with the prior or choose a compatible prior"))

    all_samples = Matrix{Float64}(undef, 0, length(prob.best))
    all_fvals = Float64[]
    all_loglik = Float64[]
    all_logpost = Float64[]
    chain_ids = Int[]
    accs = Float64[]
    scales = Float64[]
    fbest = NaN

    for ch in 1:Int(nchains)
        rng_ch = if seed !== nothing
            Random.MersenneTwister(_chain_seed(UInt64(seed), ch))
        elseif rng !== nothing
            rng
        else
            Random.default_rng()
        end
        p0 = if ch == 1
            pbest
        else
            valid_start = q -> begin
                lp = prob.logprior_free(q)
                isfinite(lp) || return false
                ll = prob.loglik_free(q)
                return isfinite(ll)
            end
            q0 = _dispersed_start(pbest, rng_ch, steps_vec, Sfac,
                                  prob.lo_free, prob.hi_free, overdisperse, valid_start)
            if q0 === nothing
                warn && @warn "posterior_sample: could not over-disperse chain $ch to a finite posterior point; starting at best"
                pbest
            else
                q0
            end
        end
        kept, fvals, loglik, logpost, acc, scale_final, fstart = _metropolis_chain(
            prob.fcn, prob.prior.logdensity,
            p0, prob.best, prob.free_idx, prob.lo_free, prob.hi_free,
            steps_vec, Sfac;
            up = prob.up, nsteps = nsteps, burn = burn, thin = thin,
            scale = scale, target_accept = target_accept,
            adapt_every = adapt_every, rng = rng_ch, warn = warn,
            context = "posterior_sample")
        ch == 1 && (fbest = fstart)
        all_samples = vcat(all_samples, kept)
        append!(all_fvals, fvals)
        append!(all_loglik, loglik)
        append!(all_logpost, logpost)
        append!(chain_ids, fill(ch, size(kept, 1)))
        push!(accs, acc)
        push!(scales, scale_final)
    end

    free = falses(length(prob.names)); free[prob.free_idx] .= true
    ens = LikelihoodEnsemble(all_samples, all_fvals, prob.names, collect(free),
                             prob.best, fbest, prob.up, mean(accs),
                             Int(nsteps), Int(burn), Int(thin), mean(scales),
                             prop_used, seed === nothing ? nothing : UInt64(seed))
    rh = diagnostics ? _posterior_rhat(ens.samples, chain_ids, prob.free_idx, Int(nchains)) :
                       fill(NaN, length(prob.free_idx))
    es = diagnostics ? _posterior_ess(ens.samples, chain_ids, prob.free_idx, Int(nchains)) :
                       fill(NaN, length(prob.free_idx))
    boundary = _boundary_flags(ens.samples, prob.free_idx, prob.lo_free, prob.hi_free)
    warns = String[]
    any(boundary) && push!(warns, "posterior mass is near at least one active parameter limit")
    return PosteriorSample(ens, prob.prior, :metropolis, nchains, chain_ids,
                           all_loglik, all_logpost, rh, es, boundary, warns)
end

posterior_sample(m::Minuit; prior = :flat, kwargs...) =
    posterior_sample(PosteriorProblem(m; prior = prior); kwargs...)

# Extension hook for `sampler = :nuts` (Hamiltonian Monte-Carlo / NUTS). The
# implementation lives in `JuMinuitAdvancedHMCExt`, loaded automatically when
# AdvancedHMC + LogDensityProblems + LogDensityProblemsAD + TransformVariables +
# ForwardDiff are all loaded alongside JuMinuit. Until then this stub explains how
# to enable it. NUTS needs an auto-differentiable FCN; the gradient-free
# `sampler = :stretch` is the fallback for FCNs that cannot be differentiated.
function _posterior_sample_nuts(prob::PosteriorProblem; kwargs...)
    ext = Base.get_extension(@__MODULE__, :JuMinuitAdvancedHMCExt)
    ext === nothing && throw(ArgumentError(
        "sampler = :nuts requires the AdvancedHMC extension. Enable it with:\n" *
        "    using AdvancedHMC, LogDensityProblems, LogDensityProblemsAD, TransformVariables, ForwardDiff\n" *
        "alongside `using JuMinuit`. NUTS needs an auto-differentiable FCN; for a " *
        "non-differentiable FCN use sampler = :stretch (gradient-free)."))
    return ext._nuts_impl(prob; kwargs...)
end

# Affine-invariant ensemble (`sampler = :stretch`) path of `posterior_sample`.
# A single population of `nwalkers` walkers explores the posterior with the
# Goodman–Weare stretch move (`_ensemble_chain`); each walker is treated as a
# chain for the split-R̂ / ESS diagnostics. Gradient-free, so it works for any
# FCN. The `proposal` / `scale` / `target_accept` / `overdisperse` knobs are
# random-walk-only and do not apply here.
function _posterior_sample_stretch(prob::PosteriorProblem; nwalkers, niter, burn,
                                   thin, stretch, diagnostics, seed, rng, warn)
    nfree = length(prob.free_idx)
    nw = nwalkers === nothing ? max(2 * nfree + 2, 8) : Int(nwalkers)
    nw >= 4 || throw(ArgumentError("ensemble sampler needs nwalkers ≥ 4, got $nw"))
    isodd(nw) && (nw += 1)                       # the split update needs an even count
    pbest = prob.best[prob.free_idx]
    isfinite(prob.logprior_free(pbest)) ||
        throw(ArgumentError("best-fit point has non-finite log-prior; re-minimize with the prior or choose a compatible prior"))
    # Walker-dispersal scale: the fit's parabolic errors when usable, else a
    # per-coordinate fallback so a single bad error does not collapse the ball.
    init_scale = [(isfinite(prob.errors_free[k]) && prob.errors_free[k] > 0) ?
                  prob.errors_free[k] : max(abs(pbest[k]) * 0.1, 1.0) for k in 1:nfree]
    rng_use = seed !== nothing ? Random.MersenneTwister(UInt64(seed)) :
              rng !== nothing ? rng : Random.default_rng()

    kept, fvals, loglik, logpost, chain_ids, acceptance, fbest = _ensemble_chain(
        prob.fcn, prob.prior.logdensity, pbest, prob.best, prob.free_idx,
        prob.lo_free, prob.hi_free, init_scale;
        up = prob.up, nwalkers = nw, niter = niter, burn = burn, thin = thin,
        stretch = stretch, rng = rng_use, warn = warn, context = "posterior_sample")

    free = falses(length(prob.names)); free[prob.free_idx] .= true
    ens = LikelihoodEnsemble(kept, fvals, prob.names, collect(free),
                             prob.best, fbest, prob.up, acceptance,
                             Int(niter), Int(burn), Int(thin), Float64(stretch),
                             :stretch, seed === nothing ? nothing : UInt64(seed))
    rh = diagnostics ? _posterior_rhat(ens.samples, chain_ids, prob.free_idx, nw) :
                       fill(NaN, length(prob.free_idx))
    es = diagnostics ? _posterior_ess(ens.samples, chain_ids, prob.free_idx, nw) :
                       fill(NaN, length(prob.free_idx))
    boundary = _boundary_flags(ens.samples, prob.free_idx, prob.lo_free, prob.hi_free)
    warns = String[]
    any(boundary) && push!(warns, "posterior mass is near at least one active parameter limit")
    return PosteriorSample(ens, prob.prior, :stretch, nw, chain_ids,
                           loglik, logpost, rh, es, boundary, warns)
end

# Split-R̂ (split Gelman–Rubin). Each chain is split into its first and second
# half, treated as 2·nchains sub-chains, so within-chain non-stationarity (a
# drifting chain, or one stuck in a different basin for its first half) inflates
# R̂ instead of hiding inside the within-chain variance — the modern standard
# over the plain between/within statistic. Chains too short to halve (< 4 kept
# draws) fall back to whole, unsplit chains, which keeps the degenerate-input
# behavior (constant chains → Inf/NaN) intact.
function _posterior_rhat(samples, chain_ids, free_idx, nchains)
    nfree = length(free_idx)
    nchains < 2 && return fill(NaN, nfree)
    out = fill(NaN, nfree)
    for (jj, col) in enumerate(free_idx)
        chains = [samples[chain_ids .== ch, col] for ch in 1:nchains]
        n = minimum(length, chains)
        n >= 2 || continue
        if n >= 4
            h = n ÷ 2
            subs = Vector{Vector{Float64}}(undef, 2 * nchains)
            @inbounds for (ci, c) in enumerate(chains)
                subs[2ci - 1] = c[1:h]
                subs[2ci]     = c[(h + 1):(2h)]
            end
            m = h
        else
            subs = [c[1:n] for c in chains]
            m = n
        end
        M = length(subs)
        means = [sum(s) / m for s in subs]
        grand = sum(means) / M
        B = m * sum((μ - grand)^2 for μ in means) / (M - 1)
        W = 0.0
        for (si, s) in enumerate(subs)
            W += sum((s[i] - means[si])^2 for i in 1:m) / (m - 1)
        end
        W /= M
        out[jj] = if W > 0
            sqrt(((m - 1) / m * W + B / m) / W)
        else
            B > 0 ? Inf : NaN
        end
    end
    return out
end

function _ess_one(v::Vector{Float64})
    n = length(v)
    n < 3 && return Float64(n)
    μ = sum(v) / n
    var0 = sum((x - μ)^2 for x in v) / n
    var0 <= 0 && return 0.0
    s = 0.0
    maxlag = min(n - 1, 1000)
    for lag in 1:maxlag
        ac = 0.0
        for i in 1:(n - lag)
            ac += (v[i] - μ) * (v[i + lag] - μ)
        end
        ρ = ac / ((n - lag) * var0)
        ρ <= 0 && break
        s += ρ
    end
    return n / (1 + 2s)
end

function _posterior_ess(samples, chain_ids, free_idx, nchains)
    out = Vector{Float64}(undef, length(free_idx))
    for (jj, col) in enumerate(free_idx)
        ess = 0.0
        for ch in 1:nchains
            ess += _ess_one(collect(samples[chain_ids .== ch, col]))
        end
        out[jj] = ess
    end
    return out
end

# Flag parameters whose posterior mass piles against an active limit. The "near
# the limit" band is scaled to the marginal posterior spread (≈ 0.2σ), not to a
# machine-eps absolute — so a continuous mode sitting AT a boundary (a coupling
# g ≥ 0 whose best fit is 0, say) is detected, not only a literal atom exactly on
# the bound. A flag means: report an upper/lower limit there, not a symmetric error.
function _boundary_flags(samples, free_idx, lo_free, hi_free)
    out = falses(length(free_idx))
    n = size(samples, 1)
    n == 0 && return out
    @inbounds for (j, col) in enumerate(free_idx)
        lo, hi = lo_free[j], hi_free[j]
        vals = @view samples[:, col]
        μ = sum(vals) / n
        sd = sqrt(max(sum(x -> (x - μ)^2, vals) / max(n - 1, 1), 0.0))
        # Band = 0.5σ, threshold 12%. Calibrated so a *pure* half-normal whose
        # mode sits exactly on the limit fires (it has ≈24% of its mass within
        # 0.5σ of the bound) while a mode ≳ 2σ inside does not (≲7%).
        band = sd > 0 ? 0.5 * sd : max(1e-8 * max(abs(μ), 1.0), 1e-12)
        if !isnan(lo)
            out[j] |= count(x -> x <= lo + band, vals) / n > 0.12
        end
        if !isnan(hi)
            out[j] |= count(x -> x >= hi - band, vals) / n > 0.12
        end
    end
    return out
end

function _posterior_col(post::PosteriorSample, par)
    if par isa Integer
        idx = Int(par)
        1 <= idx <= length(post.ensemble.names) ||
            throw(ArgumentError("parameter index $idx out of range"))
        return idx
    else
        s = String(par)
        idx = findfirst(==(s), post.ensemble.names)
        idx === nothing && throw(KeyError("parameter \"$s\" not found"))
        return idx
    end
end

_check_level(level) =
    (0 < level < 1) || throw(ArgumentError("level must be a probability in (0, 1), got $level"))

function _central_probs(level)
    _check_level(level)
    α = (1 - Float64(level)) / 2
    return (α, 1 - α)
end

"""
    credible_interval(post, par; level=0.6827, method=:central)

Equal-tailed marginal Bayesian credible interval for parameter `par`. MVP
supports `method=:central` only; HPD intervals are deferred because multimodal
marginals may have disjoint highest-density regions.
"""
function credible_interval(post::PosteriorSample, par; level::Real = 0.6827,
                           method::Symbol = :central)
    method === :central ||
        throw(ArgumentError("MVP credible_interval supports method=:central only"))
    idx = _posterior_col(post, par)
    lo, hi = quantile(collect(@view post.ensemble.samples[:, idx]), collect(_central_probs(level)))
    return (lo, hi)
end

"""
    upper_limit(post, par; level=0.90) -> CredibleLimit

One-sided Bayesian upper credible limit: the value `x` with
`P(par <= x | data, prior) = level`.
"""
function upper_limit(post::PosteriorSample, par; level::Real = 0.90)
    _check_level(level)
    idx = _posterior_col(post, par)
    lim = quantile(collect(@view post.ensemble.samples[:, idx]), Float64(level))
    free_pos = findfirst(==(idx), findall(post.ensemble.free))
    b = free_pos === nothing ? false : post.boundary_active[free_pos]
    return CredibleLimit(post.ensemble.names[idx], lim, Float64(level), :upper,
                         post.prior.name, b)
end

"""
    lower_limit(post, par; level=0.90) -> CredibleLimit

One-sided Bayesian lower credible limit: the value `x` with
`P(par >= x | data, prior) = level`.
"""
function lower_limit(post::PosteriorSample, par; level::Real = 0.90)
    _check_level(level)
    idx = _posterior_col(post, par)
    lim = quantile(collect(@view post.ensemble.samples[:, idx]), 1 - Float64(level))
    free_pos = findfirst(==(idx), findall(post.ensemble.free))
    b = free_pos === nothing ? false : post.boundary_active[free_pos]
    return CredibleLimit(post.ensemble.names[idx], lim, Float64(level), :lower,
                         post.prior.name, b)
end

"""
    derived_interval(post, f; level=0.6827, method=:central, warn=true)

Sample-wise credible interval for a scalar derived quantity `f(θ_full)`.
No delta method or linear propagation is used. `warn=false` silences the
"dropped N non-finite values" notice from the underlying quantile pass.
"""
function derived_interval(post::PosteriorSample, f; level::Real = 0.6827,
                          method::Symbol = :central, warn::Bool = true)
    method === :central ||
        throw(ArgumentError("MVP derived_interval supports method=:central only"))
    qs = quantiles(post.ensemble, f; p = _central_probs(level), warn = warn)
    return (qs[1], qs[2])
end

"""
    posterior_mean(post::PosteriorSample, par) -> Float64

Posterior mean of parameter `par` (name or index) over the kept samples.
"""
posterior_mean(post::PosteriorSample, par) =
    mean(@view post.ensemble.samples[:, _posterior_col(post, par)])

"""
    posterior_median(post::PosteriorSample, par) -> Float64

Posterior median of parameter `par` (name or index).
"""
posterior_median(post::PosteriorSample, par) =
    quantile(collect(@view post.ensemble.samples[:, _posterior_col(post, par)]), 0.5)

"""
    posterior_std(post::PosteriorSample, par) -> Float64

Posterior standard deviation of parameter `par` (sample std, `N − 1` divisor).
This is a marginal posterior spread, not a HESSE/MINOS error. Returns `NaN` for
≤ 1 kept samples, and `0.0` for a constant column (e.g. a fixed parameter).
"""
function posterior_std(post::PosteriorSample, par)
    v = @view post.ensemble.samples[:, _posterior_col(post, par)]
    length(v) > 1 || return NaN          # undefined for a single kept sample
    μ = mean(v)
    return sqrt(sum((x - μ)^2 for x in v) / (length(v) - 1))
end

"""
    effective_sample_size(post::PosteriorSample, par) -> Float64

Effective sample size for parameter `par`, summed over chains (autocorrelation-
adjusted). `NaN` for a fixed parameter. A small ESS relative to the kept-sample
count means the chain mixed slowly; raise `nsteps`/`thin` or retune the proposal.
"""
function effective_sample_size(post::PosteriorSample, par)
    idx = _posterior_col(post, par)
    free_pos = findfirst(==(idx), findall(post.ensemble.free))
    free_pos === nothing && return NaN
    return post.ess[free_pos]
end

"""
    rhat(post::PosteriorSample, par) -> Float64

Split-R̂ convergence diagnostic for parameter `par` (needs `nchains ≥ 2`).
Values near `1` (conventionally `< 1.01`) indicate the chains have mixed;
`NaN` for a fixed parameter or a single chain. This is the basic split-R̂
(not rank-normalized / folded), so for a skewed or boundary-truncated marginal
pair it with [`effective_sample_size`](@ref) and a trace check.
"""
function rhat(post::PosteriorSample, par)
    idx = _posterior_col(post, par)
    free_pos = findfirst(==(idx), findall(post.ensemble.free))
    free_pos === nothing && return NaN
    return post.rhat[free_pos]
end

"""
    posterior_summary(post; level=0.6827)

Return per-parameter summary rows `(parameter, mean, median, std, lower, upper,
level)` using central marginal credible intervals.
"""
function posterior_summary(post::PosteriorSample; level::Real = 0.6827)
    _check_level(level)
    rows = NamedTuple[]
    for (idx, name) in enumerate(post.ensemble.names)
        vals = collect(@view post.ensemble.samples[:, idx])
        μ = mean(vals)
        med = quantile(vals, 0.5)
        sd = length(vals) > 1 ? sqrt(sum((x - μ)^2 for x in vals) / (length(vals) - 1)) : NaN
        lo, hi = credible_interval(post, idx; level = level)
        push!(rows, (; parameter = name, mean = μ, median = med, std = sd,
                     lower = lo, upper = hi, level = Float64(level)))
    end
    return rows
end

"""
    bayesian(m::Minuit; prior=:flat, level=0.6827, interval=:central, kwargs...)
        -> BayesianReport

One-step, non-mutating Bayesian analysis convenience wrapper. It calls
[`posterior_sample`](@ref), summarizes the result, and returns a
[`BayesianReport`](@ref). It never writes Bayesian intervals into `m.errors`,
`m.covariance`, or MINOS state.
"""
function bayesian(m::Minuit; prior = :flat, level::Real = 0.6827,
                  interval::Symbol = :central, kwargs...)
    _check_level(level)
    interval === :central ||
        throw(ArgumentError("MVP bayesian supports interval=:central only"))
    post = posterior_sample(m; prior = prior, kwargs...)
    return BayesianReport(post, Float64(level), interval,
                          posterior_summary(post; level = level))
end
