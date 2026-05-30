# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# resampling.jl — data-resampling error analysis: bootstrap + jackknife.
#
# These estimate the SAMPLING distribution of the estimator θ̂ by resampling the
# DATA and re-fitting. They are complementary to HESSE / MINOS / MC-Δχ² (which
# vary the PARAMETERS with the data held FIXED — interrogating the likelihood):
#
#   • HESSE / MINOS / MC-Δχ²   — "what parameters are consistent with THIS data"
#   • bootstrap / jackknife    — "how would θ̂ vary across repeated experiments"
#
# They agree for a well-specified Gaussian model and diverge precisely when the
# error model is wrong or the estimator is biased — which is what makes them a
# useful cross-check. See docs/ERROR_ANALYSIS.md for the unified comparison.
#
# No C++ Minuit2 analogue (C++ Minuit2 has no resampling layer); this is a
# JuMinuit extension built on top of `Data` / `model_fit` / `migrad!`. Three
# entry shapes are supported (see each method):
#
#   (i)  cost objects (feat/iminuit-cost-classes) — PREFERRED, added as methods
#        once that branch lands (the cost object carries x/y/data, so the model
#        need not be passed separately).
#   (ii) `model::Function` + `data::Data` + start  — the present pre-cost-class
#        API; the model must be passed explicitly because a `model_fit` Minuit
#        bakes the data into an opaque closure (`par -> chisq(model, data, par)`)
#        from which the model can't be recovered.
#   (iii) a generic `refit(subdata) -> θ̂` callback on any indexable `data`.
#
# Determinism: bootstrap consumes randomness, but every resample's RNG seed is
# drawn SERIALLY from the master RNG up front, so the threaded and serial runs
# are bit-identical (the re-fits themselves are deterministic given fixed
# indices). Threading is opt-in and "Phase-H-aware": each resample builds its
# OWN Minuit (no shared mutable fit state) and the inner gradient threading is
# disabled to avoid oversubscription, so the only thread-safety requirement is
# that the user `model` / `refit` closure be pure (no hidden RNG/cache/IO).
# ─────────────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════════
# Result types
# ═════════════════════════════════════════════════════════════════════════════

"""
    BootstrapResult

Result of [`bootstrap`](@ref). Fields:

- `samples::Matrix{Float64}` — `nresample × npar` matrix of re-fitted parameter
  vectors θ̂ (one row per resample, columns in external-parameter order). Rows
  for non-converged re-fits are kept (as `NaN`) and flagged in `valid`.
- `valid::Vector{Bool}` — per-resample convergence mask (`m.valid`). Summary
  statistics below are computed over the `valid` rows only (see `filter_invalid`
  in [`bootstrap`](@ref)).
- `names::Vector{String}` — external parameter names.
- `estimate::Vector{Float64}` — the original full-data optimum θ̂_full (the
  anchor the re-fits are warm-started from).
- `mean::Vector{Float64}` — bootstrap mean of θ̂ over valid resamples.
- `std::Vector{Float64}` — bootstrap standard error (the spread of θ̂; comparable
  to the HESSE error for a well-specified Gaussian fit).
- `ci_lower`, `ci_upper::Vector{Float64}` — percentile confidence interval at
  `ci_level` (the `[(1-ci_level)/2, (1+ci_level)/2]` quantiles of θ̂). These are
  generally **asymmetric** about `estimate` — that asymmetry is the whole point
  versus the symmetric HESSE error.
- `ci_level::Float64` — the CI coverage (default `0.68`, i.e. a ±1σ-equivalent).
- `covariance::Union{Nothing,Matrix{Float64}}` — bootstrap covariance of θ̂
  (`npar × npar`), or `nothing` unless `covariance=true` was requested. The
  off-diagonal captures parameter correlations; [`correlation`](@ref) returns
  the standardised correlation matrix from `samples` regardless of this flag,
  and the raw `samples` cloud retains non-Gaussian joint structure a covariance
  cannot.
- `kind::Symbol` — `:nonparametric` (resample points with replacement) or
  `:parametric` (regenerate y from the best-fit model + error model).
- `nresample::Int`, `n_valid::Int` — total and converged resample counts.
- `seed` — the master seed used (for reproducibility), or `nothing`.
"""
struct BootstrapResult
    samples::Matrix{Float64}
    valid::Vector{Bool}
    names::Vector{String}
    estimate::Vector{Float64}
    mean::Vector{Float64}
    std::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    ci_level::Float64
    covariance::Union{Nothing,Matrix{Float64}}
    kind::Symbol
    nresample::Int
    n_valid::Int
    seed::Union{Nothing,UInt64}
end

"""
    JackknifeResult

Result of [`jackknife`](@ref). For the delete-1 jackknife there is one re-fit
per data point (`g = N`); for the optional delete-`d` block jackknife the data
are partitioned into `g = ceil(N/d)` consecutive blocks and one block is deleted
per re-fit. Fields:

- `samples::Matrix{Float64}` — `g × npar` matrix of leave-one-(group-)out
  estimates θ̂₍ⱼ₎.
- `valid::Vector{Bool}` — per-re-fit convergence mask.
- `names::Vector{String}` — external parameter names.
- `estimate::Vector{Float64}` — the full-data optimum θ̂_full.
- `mean::Vector{Float64}` — θ̄ = mean of the leave-one-out estimates.
- `bias::Vector{Float64}` — jackknife bias estimate `(g-1)·(θ̄ - θ̂_full)`.
- `bias_corrected::Vector{Float64}` — `θ̂_full - bias` (the debiased estimate).
- `variance::Vector{Float64}` — jackknife variance `((g-1)/g)·Σ(θ̂₍ⱼ₎ - θ̄)²`.
- `std::Vector{Float64}` — `sqrt.(variance)` (comparable to the HESSE error).
- `covariance::Matrix{Float64}` — the full jackknife covariance
  `((g-1)/g)·Σ(θ̂₍ⱼ₎ - θ̄)(θ̂₍ⱼ₎ - θ̄)ᵀ`, whose diagonal is `variance`. Its
  off-diagonal captures the **parameter correlations** (to first order — the
  jackknife is a linearisation; use [`bootstrap`](@ref) for non-Gaussian /
  strongly nonlinear joint structure). See [`correlation`](@ref).
- `n::Int` — number of data points `N`.
- `d::Int` — block size (`1` for the delete-1 jackknife).
- `g::Int` — number of deleted groups (`N` for delete-1).
- `n_valid::Int` — converged re-fit count.
"""
struct JackknifeResult
    samples::Matrix{Float64}
    valid::Vector{Bool}
    names::Vector{String}
    estimate::Vector{Float64}
    mean::Vector{Float64}
    bias::Vector{Float64}
    bias_corrected::Vector{Float64}
    variance::Vector{Float64}
    std::Vector{Float64}
    covariance::Matrix{Float64}
    n::Int
    d::Int
    g::Int
    n_valid::Int
end

# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

# Recover the `Minuit`-constructor keyword configuration from a (fitted) Minuit
# so each resample re-fit is built with identical names / errors / limits /
# fixed / errordef / strategy / tol / gradient / machine-precision. Carrying
# `prec` keeps a `set_precision`-tuned anchor's gradient step sizes from
# silently reverting to the 4·eps default in the resamples (iminuit-parity:
# no per-construction asymmetry). Threading flags are deliberately NOT carried —
# resample re-fits run single-threaded gradients (parallelism is at the
# resample level; nesting would oversubscribe).
function _fit_kwargs(m::Minuit)
    nm = [p.name for p in m.params.pars]
    er = [p.error for p in m.params.pars]
    fx = [is_fixed(p) for p in m.params.pars]
    lim = Vector{Any}(undef, n_pars(m.params))
    for (i, p) in enumerate(m.params.pars)
        lo = isnan(p.lower) ? nothing : p.lower
        hi = isnan(p.upper) ? nothing : p.upper
        lim[i] = (lo === nothing && hi === nothing) ? nothing : (lo, hi)
    end
    grad = m.cfwg === nothing ? nothing : m.cfwg.g
    return (; name = nm, error = er, fixed = fx, limits = lim,
            up = m.fcn.up, strategy = m.strategy, tol = m.tol, grad = grad,
            prec = m.prec)
end

# Resolve the master RNG: an explicit `seed` wins (and is recorded for
# reproducibility); otherwise use the caller-provided `rng` (default
# `Random.default_rng()`, so a `Random.seed!(...)` upstream is honoured).
# Returns `(master_rng, recorded_seed_or_nothing)`.
function _resolve_rng(seed, rng)
    if seed !== nothing
        seed >= 0 ||
            throw(ArgumentError("bootstrap: seed must be non-negative (got $seed)"))
        s = UInt64(seed)
        return Random.Xoshiro(s), s
    end
    return rng, nothing
end

# Rows that count toward the summary statistics. A non-finite row is ALWAYS
# dropped (keeping it would poison every statistic with NaN); on top of that,
# `filter_invalid` (the default) additionally requires the re-fit to have
# converged. So `false` includes non-converged-but-finite re-fits, `true`
# keeps only converged ones — neither ever lets a NaN row through.
function _stat_mask(samples::Matrix{Float64}, valid::AbstractVector{Bool},
                    filter_invalid::Bool)
    nrep = size(samples, 1)
    finite = Bool[all(isfinite, @view samples[k, :]) for k in 1:nrep]
    return filter_invalid ? (valid .& finite) : finite
end

# Per-resample seeds, drawn SERIALLY from the master RNG. This is the crux of
# threaded == serial determinism: the seeds (hence the resampled index sets and
# parametric noise) depend only on the master RNG state, never on thread
# scheduling. Each resample then runs an independent `Xoshiro(seeds[k])`.
_resample_seeds(master, nresample) = rand(master, UInt64, nresample)

# Drive `fit_one(k) -> (θ̂::Vector{Float64}, valid::Bool)` for k in 1:nrep,
# writing results by index into pre-allocated output (so order — and therefore
# the result — is independent of thread scheduling). Threading is opt-in. A
# re-fit that throws is caught and recorded as an invalid (`NaN`) row rather
# than aborting the whole sweep.
function _run_resamples(fit_one, nrep::Int, npar::Int, threaded::Bool)
    samples = Matrix{Float64}(undef, nrep, npar)
    valid = Vector{Bool}(undef, nrep)
    work = function (k)
        local θ, v
        try
            θ, v = fit_one(k)
        catch
            θ, v = fill(NaN, npar), false
        end
        @inbounds begin
            if length(θ) == npar
                samples[k, :] .= θ
            else
                samples[k, :] .= NaN
                v = false
            end
            valid[k] = v
        end
        return nothing
    end
    if threaded
        Threads.@threads for k in 1:nrep
            work(k)
        end
    else
        for k in 1:nrep
            work(k)
        end
    end
    return samples, valid
end

# Per-parameter bootstrap summary statistics over the VALID rows. `ci_level` is
# the central coverage → percentile bounds at (1∓ci_level)/2. Returns
# (mean, std, ci_lower, ci_upper, covariance_or_nothing, n_valid). Degenerate
# cases (0 or 1 valid resamples) yield NaN spreads rather than throwing.
function _bootstrap_stats(samples::Matrix{Float64}, valid::AbstractVector{Bool},
                          ci_level::Float64, want_cov::Bool)
    npar = size(samples, 2)
    rows = findall(valid)
    nv = length(rows)
    μ = fill(NaN, npar)
    σ = fill(NaN, npar)
    lo = fill(NaN, npar)
    hi = fill(NaN, npar)
    cov = nothing
    if nv >= 1
        S = @view samples[rows, :]
        α = (1 - ci_level) / 2
        for j in 1:npar
            col = @view S[:, j]
            μ[j] = mean(col)
            if nv >= 2
                σ[j] = std(col)                      # corrected (B-1) sample std
                lo[j] = quantile(col, α)
                hi[j] = quantile(col, 1 - α)
            else
                lo[j] = hi[j] = μ[j]
            end
        end
        if want_cov && nv >= 2
            cov = Matrix(Statistics.cov(Matrix(S)))  # npar×npar (cols = params)
        end
    end
    return μ, σ, lo, hi, cov, nv
end

# Consecutive index blocks to DELETE for the (delete-d) jackknife. d == 1 gives
# N singleton blocks (delete-1). For d > 1 the data are split into
# `ceil(N/d)` consecutive blocks — consecutive (not random) so any serial
# correlation in the data is preserved within a deleted block, which is the
# point of the block jackknife. The final block may be shorter when d ∤ N.
function _jackknife_blocks(N::Int, d::Int)
    d == 1 && return [[i] for i in 1:N]
    return [collect(b) for b in Iterators.partition(1:N, d)]
end

# Index vector that KEEPS everything except the given block (ascending order).
function _complement(N::Int, block::Vector{Int})
    keep = trues(N)
    @inbounds for i in block
        keep[i] = false
    end
    return (1:N)[keep]
end

# Shared jackknife reduction: given the g×npar leave-one-out `samples`, the
# `valid` mask and the full-data `estimate`, compute (mean, bias,
# bias_corrected, variance, std, n_valid) over the valid groups. `g_used` is the
# number of valid groups (the (g-1)/g scaling uses the count actually summed).
function _jackknife_stats(samples::Matrix{Float64}, valid::AbstractVector{Bool},
                          estimate::Vector{Float64})
    npar = size(samples, 2)
    # Use only converged AND finite groups (a NaN row would poison θ̄/variance).
    finite = Bool[all(isfinite, @view samples[r, :]) for r in 1:size(samples, 1)]
    rows = findall(valid .& finite)
    gv = length(rows)
    θ̄ = fill(NaN, npar)
    bias = fill(NaN, npar)
    bias_corr = fill(NaN, npar)
    variance = fill(NaN, npar)
    σ = fill(NaN, npar)
    cov = fill(NaN, npar, npar)
    if gv >= 1
        S = @view samples[rows, :]
        for j in 1:npar
            θ̄[j] = mean(@view S[:, j])
        end
        bias .= (gv - 1) .* (θ̄ .- estimate)
        bias_corr .= estimate .- bias
        if gv >= 2
            # Full jackknife covariance ((g-1)/g)·Σ(θ̂₍ⱼ₎-θ̄)(θ̂₍ⱼ₎-θ̄)ᵀ; the
            # diagonal is the per-parameter variance, the off-diagonal the
            # (first-order) parameter correlation structure.
            scale = (gv - 1) / gv
            for j in 1:npar, i in 1:npar
                acc = 0.0
                @inbounds for r in 1:gv
                    acc += (S[r, i] - θ̄[i]) * (S[r, j] - θ̄[j])
                end
                cov[i, j] = scale * acc
            end
            for j in 1:npar
                variance[j] = cov[j, j]
                σ[j] = sqrt(max(variance[j], 0.0))
            end
        end
    end
    return θ̄, bias, bias_corr, variance, σ, cov, gv
end

# ═════════════════════════════════════════════════════════════════════════════
# bootstrap
# ═════════════════════════════════════════════════════════════════════════════

"""
    bootstrap(model::Function, data::Data, start; kws...) -> BootstrapResult

Bootstrap error analysis for a χ² fit of `model(x, par)` to `data`. `start` is
either the initial-value `AbstractVector` (the full-data fit is run internally
to obtain the anchor optimum θ̂_full) or a `Minuit` template (its configuration —
names, limits, fixed flags, errordef, strategy, tol, gradient — is cloned, and
its current values seed the anchor; it is fitted first if not already).

Keyword arguments:

- `nresample::Int = 1000` — number of bootstrap resamples.
- `kind::Symbol = :nonparametric` — `:nonparametric` resamples the `N` data
  points **with replacement** and re-fits; `:parametric` regenerates
  `yᵢ* = model(xᵢ, θ̂_full) + σᵢ·zᵢ`, `zᵢ ~ 𝒩(0,1)`, holding `x` and `err`
  fixed (Gaussian error model from `data.err`).
- `seed::Union{Integer,Nothing} = nothing` — master RNG seed; pass an integer
  for reproducible (and threaded == serial bit-identical) results. When
  `nothing`, `rng` is used as-is.
- `rng::AbstractRNG = Random.default_rng()` — master RNG when `seed` is not set.
- `warm_start::Bool = true` — seed each re-fit from θ̂_full (fast, recommended).
  `false` cold-starts every re-fit from the original `start` values.
- `ci_level::Real = 0.68` — percentile-CI coverage (default ≈ ±1σ).
- `covariance::Bool = false` — also return the bootstrap covariance matrix.
- `filter_invalid::Bool = true` — compute summary stats over converged re-fits
  only. `false` includes every re-fit that produced a finite estimate (converged
  or not); a thrown / non-finite re-fit is dropped either way (it would poison
  the statistics). All rows, valid or not, are always kept in `samples`.
- `threaded::Bool = false` — run the re-fits across threads (requires
  `julia -t N`); the `model` closure must be thread-safe (pure). Deterministic:
  threaded and serial runs are bit-identical for the same `seed`.
- further `kws...` flow to `model_fit` / `Minuit` (e.g. `name`, `limits`) when
  `start` is a plain vector.

The returned [`BootstrapResult`](@ref) carries the θ̂ sample matrix plus
per-parameter mean / std / percentile CIs. The bootstrap `std` ≈ HESSE error
for a well-specified Gaussian fit and diverges when the error model is wrong.
"""
function bootstrap(model::Function, data::Data, start::Union{AbstractVector,Minuit};
                   nresample::Integer = 1000,
                   kind::Symbol = :nonparametric,
                   seed::Union{Integer,Nothing} = nothing,
                   rng::Random.AbstractRNG = Random.default_rng(),
                   warm_start::Bool = true,
                   ci_level::Real = 0.68,
                   covariance::Bool = false,
                   filter_invalid::Bool = true,
                   threaded::Bool = false,
                   kws...)
    nresample >= 1 || throw(ArgumentError("bootstrap: nresample must be ≥ 1"))
    0 < ci_level < 1 || throw(ArgumentError("bootstrap: ci_level must be in (0,1)"))
    kind === :nonparametric || kind === :parametric ||
        throw(ArgumentError("bootstrap: kind must be :nonparametric or :parametric (got :$kind)"))

    # Full-data anchor fit + cloned configuration.
    if start isa Minuit
        m_full = start.fmin === nothing ? migrad!(model_fit(model, data, start; kws...)) : start
        cold = [p.value for p in start.params.pars]
    else
        m_full = migrad!(model_fit(model, data, collect(Float64, start); kws...))
        cold = collect(Float64, start)
    end
    m_full.valid ||
        @warn "bootstrap: the full-data fit did not converge; θ̂_full anchor may be unreliable"
    θ_anchor = args(m_full)
    names = [p.name for p in m_full.params.pars]
    npar = length(θ_anchor)
    cfg = _fit_kwargs(m_full)
    start_vals = warm_start ? θ_anchor : cold

    # Parametric bootstrap regenerates from the model evaluated at θ̂_full; this
    # baseline is constant across resamples, so compute it once.
    N = data.ndata
    ymodel0 = kind === :parametric ?
        Float64[model(data.x[i], θ_anchor) for i in 1:N] : Float64[]

    master, recorded = _resolve_rng(seed, rng)
    seeds = _resample_seeds(master, Int(nresample))

    fit_one = function (k)
        rng_k = Random.Xoshiro(seeds[k])
        if kind === :nonparametric
            idx = rand(rng_k, 1:N, N)          # N draws WITH replacement
            d_k = data[idx]
        else
            y_star = ymodel0 .+ data.err .* randn(rng_k, N)
            d_k = Data(data.x, y_star, data.err)
        end
        m_k = model_fit(model, d_k, start_vals; cfg..., threaded_gradient = false)
        migrad!(m_k)
        return args(m_k), m_k.valid
    end

    samples, valid = _run_resamples(fit_one, Int(nresample), npar, threaded)
    valmask = _stat_mask(samples, valid, filter_invalid)
    μ, σ, lo, hi, cov, nv =
        _bootstrap_stats(samples, valmask, Float64(ci_level), covariance)

    nv == 0 && @warn "bootstrap: no resample re-fit converged; statistics are NaN"

    return BootstrapResult(samples, valid, names, θ_anchor, μ, σ, lo, hi,
                           Float64(ci_level), cov, kind, Int(nresample),
                           count(valid), recorded)
end

"""
    bootstrap(refit::Function, data; kws...) -> BootstrapResult

Generic (interface-iii) nonparametric bootstrap. `data` need only support
`length(data)` and `data[idxvec]` (e.g. `Data`, or any indexable collection),
and `refit(subdata) -> θ̂::AbstractVector` re-fits a resampled dataset and
returns the parameter vector. Warm-starting is the caller's responsibility (let
the closure start from the full-data optimum).

Keywords: `nresample`, `seed`, `rng`, `ci_level`, `covariance`, `threaded` as
for the model-based method, plus `names::Union{Vector{String},Nothing}=nothing`
to label the parameters (default `p1, p2, …`). `kind` is fixed to
`:nonparametric` here — a parametric bootstrap needs the model + error model,
so use the `model`/`Data` method for that.

The initial `refit(data)` probe (used to size the parameter vector) is **not**
guarded — if it throws, the whole call fails; only the per-resample re-fits are
caught and recorded as invalid.
"""
function bootstrap(refit::Function, data;
                   nresample::Integer = 1000,
                   seed::Union{Integer,Nothing} = nothing,
                   rng::Random.AbstractRNG = Random.default_rng(),
                   ci_level::Real = 0.68,
                   covariance::Bool = false,
                   filter_invalid::Bool = true,
                   threaded::Bool = false,
                   names::Union{Vector{<:AbstractString},Nothing} = nothing)
    nresample >= 1 || throw(ArgumentError("bootstrap: nresample must be ≥ 1"))
    0 < ci_level < 1 || throw(ArgumentError("bootstrap: ci_level must be in (0,1)"))
    N = length(data)
    # Probe on the full data → anchor estimate + parameter count.
    θ_full = collect(Float64, refit(data))
    npar = length(θ_full)
    nm = names === nothing ? ["p$i" for i in 1:npar] : String.(names)
    length(nm) == npar ||
        throw(ArgumentError("bootstrap: names length $(length(nm)) ≠ npar $npar"))

    master, recorded = _resolve_rng(seed, rng)
    seeds = _resample_seeds(master, Int(nresample))

    fit_one = function (k)
        rng_k = Random.Xoshiro(seeds[k])
        idx = rand(rng_k, 1:N, N)
        θ = collect(Float64, refit(data[idx]))
        return θ, all(isfinite, θ)   # generic: no convergence flag → use finiteness
    end

    samples, valid = _run_resamples(fit_one, Int(nresample), npar, threaded)
    valmask = _stat_mask(samples, valid, filter_invalid)
    μ, σ, lo, hi, cov, nv =
        _bootstrap_stats(samples, valmask, Float64(ci_level), covariance)

    nv == 0 && @warn "bootstrap: no resample re-fit produced a finite estimate; statistics are NaN"

    return BootstrapResult(samples, valid, nm, θ_full, μ, σ, lo, hi,
                           Float64(ci_level), cov, :nonparametric,
                           Int(nresample), count(valid), recorded)
end

# ═════════════════════════════════════════════════════════════════════════════
# jackknife
# ═════════════════════════════════════════════════════════════════════════════

"""
    jackknife(model::Function, data::Data, start; kws...) -> JackknifeResult

Delete-1 (or delete-`d` block) jackknife for a χ² fit of `model(x, par)` to
`data`. `start` is an initial-value vector or a `Minuit` template (configuration
cloned), exactly as for [`bootstrap`](@ref).

Keyword arguments:

- `d::Int = 1` — block size. `d = 1` deletes one point per re-fit (`g = N`
  re-fits); `d > 1` deletes **consecutive** blocks (`g = ceil(N/d)` re-fits),
  preserving any serial correlation within a deleted block. The block (delete-d)
  jackknife is intended for serially-correlated data and is a deliberately
  COARSE estimator: with only `g` groups it has ≈ `√(2/(g-1))` relative scatter,
  and for IID data consecutive blocks of sorted data are non-exchangeable (a
  mild downward variance bias) — shuffle the data before blocking if it is IID.
  The `(g-1)/g` variance scaling is exact for equal blocks (when `d ∣ N`) and
  approximate for the shorter trailing block otherwise. Prefer `d = 1` unless
  the data are correlated.
- `warm_start::Bool = true` — seed each re-fit from θ̂_full.
- `threaded::Bool = false` — run the re-fits across threads (deterministic; the
  jackknife uses no randomness, so threaded and serial results are identical).
- further `kws...` flow to `model_fit` / `Minuit` when `start` is a vector.

The returned [`JackknifeResult`](@ref) reports the bias estimate
`(g-1)·(θ̄ - θ̂_full)`, the bias-corrected estimate, the jackknife variance
`((g-1)/g)·Σ(θ̂₍ⱼ₎ - θ̄)²`, and the full jackknife `covariance` matrix (its
off-diagonal gives the first-order parameter correlations — see
[`correlation`](@ref)). For an unbiased (e.g. linear) estimator the bias is ≈ 0
and the variance ≈ the HESSE error².
"""
function jackknife(model::Function, data::Data, start::Union{AbstractVector,Minuit};
                   d::Integer = 1,
                   warm_start::Bool = true,
                   threaded::Bool = false,
                   kws...)
    d >= 1 || throw(ArgumentError("jackknife: d must be ≥ 1"))

    if start isa Minuit
        m_full = start.fmin === nothing ? migrad!(model_fit(model, data, start; kws...)) : start
        cold = [p.value for p in start.params.pars]
    else
        m_full = migrad!(model_fit(model, data, collect(Float64, start); kws...))
        cold = collect(Float64, start)
    end
    m_full.valid ||
        @warn "jackknife: the full-data fit did not converge; θ̂_full anchor may be unreliable"
    θ_full = args(m_full)
    names = [p.name for p in m_full.params.pars]
    npar = length(θ_full)
    cfg = _fit_kwargs(m_full)
    start_vals = warm_start ? θ_full : cold

    N = data.ndata
    Int(d) <= N || throw(ArgumentError("jackknife: d ($d) must be ≤ N ($N)"))
    blocks = _jackknife_blocks(N, Int(d))
    g = length(blocks)
    g >= 2 || throw(ArgumentError("jackknife: need ≥ 2 groups (got g=$g; reduce d)"))

    fit_one = function (j)
        keep = _complement(N, blocks[j])
        m_j = model_fit(model, data[keep], start_vals; cfg..., threaded_gradient = false)
        migrad!(m_j)
        return args(m_j), m_j.valid
    end

    samples, valid = _run_resamples(fit_one, g, npar, threaded)
    θ̄, bias, bias_corr, variance, σ, cov, gv = _jackknife_stats(samples, valid, θ_full)

    gv < g && @warn "jackknife: $(g - gv) of $g leave-one-out re-fits did not converge; statistics use the $gv valid groups"

    return JackknifeResult(samples, valid, names, θ_full, θ̄, bias, bias_corr,
                           variance, σ, cov, N, Int(d), g, gv)
end

"""
    jackknife(refit::Function, data; kws...) -> JackknifeResult

Generic (interface-iii) jackknife. `data` supports `length` and `data[idxvec]`;
`refit(subdata) -> θ̂::AbstractVector`. Keywords: `d`, `threaded`, and
`names` (parameter labels, default `p1, p2, …`).
"""
function jackknife(refit::Function, data;
                   d::Integer = 1,
                   threaded::Bool = false,
                   names::Union{Vector{<:AbstractString},Nothing} = nothing)
    d >= 1 || throw(ArgumentError("jackknife: d must be ≥ 1"))
    N = length(data)
    Int(d) <= N || throw(ArgumentError("jackknife: d ($d) must be ≤ N ($N)"))
    θ_full = collect(Float64, refit(data))
    npar = length(θ_full)
    nm = names === nothing ? ["p$i" for i in 1:npar] : String.(names)
    length(nm) == npar ||
        throw(ArgumentError("jackknife: names length $(length(nm)) ≠ npar $npar"))

    blocks = _jackknife_blocks(N, Int(d))
    g = length(blocks)
    g >= 2 || throw(ArgumentError("jackknife: need ≥ 2 groups (got g=$g; reduce d)"))

    fit_one = function (j)
        keep = _complement(N, blocks[j])
        θ = collect(Float64, refit(data[keep]))
        return θ, all(isfinite, θ)   # generic: no convergence flag → use finiteness
    end

    samples, valid = _run_resamples(fit_one, g, npar, threaded)
    θ̄, bias, bias_corr, variance, σ, cov, gv = _jackknife_stats(samples, valid, θ_full)

    return JackknifeResult(samples, valid, nm, θ_full, θ̄, bias, bias_corr,
                           variance, σ, cov, N, Int(d), g, gv)
end

# ═════════════════════════════════════════════════════════════════════════════
# correlation — parameter correlation matrix from a resampling result
# ═════════════════════════════════════════════════════════════════════════════

# Pearson correlation of the re-fitted θ̂ over the converged-and-finite resamples.
# For the jackknife this equals the standardised jackknife covariance (the
# (g-1)/g scale cancels), so the same code path serves both result types.
function _sample_correlation(samples::Matrix{Float64}, valid::AbstractVector{Bool},
                             npar::Int)
    rows = _stat_mask(samples, valid, true)
    count(rows) >= 2 || return fill(NaN, npar, npar)
    return Matrix(Statistics.cor(samples[rows, :]))
end

"""
    correlation(r::BootstrapResult) -> Matrix{Float64}
    correlation(r::JackknifeResult) -> Matrix{Float64}

The `npar × npar` Pearson **correlation matrix** of the parameter estimators,
computed from the re-fitted θ̂ over the converged-and-finite resamples (for the
jackknife this is the standardised `r.covariance`). The off-diagonal entries are
the estimated parameter correlations — the joint information the per-parameter
marginal CIs / errors drop.

Correlation is a **linear** (second-moment) summary. For a non-Gaussian or
nonlinearly-correlated joint distribution (e.g. a curved degeneracy / "banana"
in an amplitude fit) inspect the raw joint cloud `r.samples` directly — a
scatter of `r.samples[:, i]` vs `r.samples[:, j]`, a 2-D density, or a contour —
which retains the structure a correlation coefficient cannot. A fixed (zero-
variance) parameter yields `NaN` in its row/column (correlation undefined).
"""
correlation(r::BootstrapResult) = _sample_correlation(r.samples, r.valid, length(r.names))
correlation(r::JackknifeResult) = _sample_correlation(r.samples, r.valid, length(r.names))

# ═════════════════════════════════════════════════════════════════════════════
# Display — boxed tables in the house style (reuses _fmt_num/_ljust/_center).
# ═════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, ::MIME"text/plain", r::BootstrapResult)
    pct = round(Int, r.ci_level * 100)
    println(io, "Bootstrap ($(r.kind), $(r.nresample) resamples, $(r.n_valid) valid)")
    # idx(3) name(10) estimate(13) std(13) CI(27)
    ci_hdr = "$(pct)% CI"
    println(io, "┌", "─"^3, "┬", "─"^10, "┬", "─"^13, "┬", "─"^13, "┬", "─"^27, "┐")
    println(io, "│", _center("", 3), "│", _center("Name", 10), "│",
            _center("Estimate", 13), "│", _center("Boot. Std", 13), "│",
            _center(ci_hdr, 27), "│")
    println(io, "├", "─"^3, "┼", "─"^10, "┼", "─"^13, "┼", "─"^13, "┼", "─"^27, "┤")
    for i in eachindex(r.names)
        idx = _ljust(" $i", 3)
        nm = _ljust(" " * r.names[i], 10)
        est = _center(_fmt_num(r.estimate[i]), 13)
        sd = _center(_fmt_num(r.std[i]), 13)
        ci = _center("[$(_fmt_num(r.ci_lower[i])), $(_fmt_num(r.ci_upper[i]))]", 27)
        println(io, "│", idx, "│", nm, "│", est, "│", sd, "│", ci, "│")
    end
    println(io, "└", "─"^3, "┴", "─"^10, "┴", "─"^13, "┴", "─"^13, "┴", "─"^27, "┘")
    if r.covariance !== nothing
        println(io, "  (bootstrap covariance available in `.covariance`)")
    end
end

function Base.show(io::IO, r::BootstrapResult)
    print(io, "BootstrapResult(", r.kind, ", nresample=", r.nresample,
          ", n_valid=", r.n_valid, ", npar=", length(r.names), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::JackknifeResult)
    label = r.d == 1 ? "delete-1" : "delete-$(r.d) block"
    println(io, "Jackknife ($label, N=$(r.n), $(r.g) groups, $(r.n_valid) valid)")
    # idx(3) name(10) estimate(13) bias-corr(13) bias(13) std(13)
    println(io, "┌", "─"^3, "┬", "─"^10, "┬", "─"^13, "┬", "─"^13, "┬", "─"^13, "┬", "─"^13, "┐")
    println(io, "│", _center("", 3), "│", _center("Name", 10), "│",
            _center("Estimate", 13), "│", _center("Bias-corr.", 13), "│",
            _center("Bias", 13), "│", _center("Jack. Std", 13), "│")
    println(io, "├", "─"^3, "┼", "─"^10, "┼", "─"^13, "┼", "─"^13, "┼", "─"^13, "┼", "─"^13, "┤")
    for i in eachindex(r.names)
        idx = _ljust(" $i", 3)
        nm = _ljust(" " * r.names[i], 10)
        est = _center(_fmt_num(r.estimate[i]), 13)
        bc = _center(_fmt_num(r.bias_corrected[i]), 13)
        bi = _center(_fmt_num(r.bias[i]), 13)
        sd = _center(_fmt_num(r.std[i]), 13)
        println(io, "│", idx, "│", nm, "│", est, "│", bc, "│", bi, "│", sd, "│")
    end
    println(io, "└", "─"^3, "┴", "─"^10, "┴", "─"^13, "┴", "─"^13, "┴", "─"^13, "┴", "─"^13, "┘")
end

function Base.show(io::IO, r::JackknifeResult)
    print(io, "JackknifeResult(d=", r.d, ", N=", r.n, ", g=", r.g,
          ", n_valid=", r.n_valid, ", npar=", length(r.names), ")")
end
