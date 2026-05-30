# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# error_sampling.jl — sampling-based / contour error analysis.
#
# The principled error-analysis path for when MINOS cannot give valid
# errors: non-quadratic (non-parabolic) χ² posteriors, poorly-converged
# or invalid `FunctionMinimum`, or multi-modal solution spaces. Validated
# against the X(3872) published analysis notebook
# (`BenchmarkExamples/X3872_dip/Xdip_published.ipynb`), whose method is:
#
#     1. Propose parameter sets from `MvNormal(best, Σ)` (Σ from the fit).
#     2. KEEP the set iff the TRUE Δχ² = χ²(x) − χ²_min ≤ δχ²(cl, ndof).
#     3. (Mahalanobis distance is used ONLY for diagnostic plots there,
#        NEVER as the acceptance criterion.)
#
# This file provides two layers:
#
#   • `delta_chisq` / `chisq_cl` — the χ²-quantile ↔ confidence-level
#     conversions (Part c). Self-contained inverse-incomplete-gamma; NO
#     dependency on Distributions.jl or SpecialFunctions.jl.
#   • `get_contours_samples` — the Monte-Carlo true-Δχ² region sampler
#     (Part b), with explicit handling of proposal UNDER-COVERAGE (the
#     failure mode that makes naïve `MvNormal` sampling silently
#     under-estimate when Σ is unreliable or the posterior is nonlinear).
#
# See `docs/ERROR_ANALYSIS.md` for the full discussion, the Δχ² table,
# the joint-vs-single-parameter distinction, and a worked X(3872) example.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Part (c): χ² quantile ↔ confidence level.
#
# `delta_chisq(cl, ndof)` returns the Δχ² threshold for confidence level
# `cl` over `ndof` jointly-estimated parameters. It is the χ²(ndof)
# quantile at probability `p`, where `p` is derived from `cl`:
#
#   • `0 < cl < 1` → `p = cl` (interpret `cl` directly as a probability).
#   • `cl ≥ 1`     → interpret `cl` as nσ; `p` = Gaussian mass within ±nσ
#                     (1→0.6827, 2→0.9545, 3→0.9973). This matches
#                     iminuit's `cl` convention.
#
# The χ²(k) CDF is the regularized lower incomplete gamma
# `F(x;k) = P(k/2, x/2)`, so the quantile is `2·P⁻¹(p; k/2)`. We
# implement `P` (Numerical Recipes `gser`/`gcf`) and its inverse `P⁻¹`
# (NR `invgammp`) directly — no external special-function dependency.
# ─────────────────────────────────────────────────────────────────────────────

# Lanczos log-Γ (Numerical Recipes 3rd ed., `gammln`; ~2e-10 rel. accuracy
# for x > 0). Used only inside the incomplete-gamma routines below.
const _GAMMLN_COF = (
    57.1562356658629235, -59.5979603554754912, 14.1360979747417471,
    -0.491913816097620199, 0.339946499848118887e-4, 0.465236289270485756e-4,
    -0.983744753048795646e-4, 0.158088703224912494e-3, -0.210264441724104883e-3,
    0.217439618115212643e-3, -0.164318106536763890e-3, 0.844182239838527433e-4,
    -0.261908384015814087e-4, 0.368991826595316234e-5,
)

@inline function _gammln(x::Float64)
    x > 0.0 || throw(DomainError(x, "_gammln requires x > 0"))
    y = x
    tmp = x + 5.24218750000000000          # g = 671/128
    tmp = (x + 0.5) * log(tmp) - tmp
    ser = 0.999999999999997092
    @inbounds for j in 1:14
        y += 1.0
        ser += _GAMMLN_COF[j] / y
    end
    return tmp + log(2.5066282746310005 * ser / x)
end

# Regularized lower incomplete gamma P(a,x) by series (x < a+1).
function _gammp_series(a::Float64, x::Float64)
    gln = _gammln(a)
    ap = a
    del = 1.0 / a
    sum = del
    @inbounds for _ in 1:1000
        ap += 1.0
        del *= x / ap
        sum += del
        abs(del) < abs(sum) * eps(Float64) && break
    end
    return sum * exp(-x + a * log(x) - gln)
end

# Complement Q(a,x) = 1 − P(a,x) by Lentz's continued fraction (x ≥ a+1).
function _gammq_cf(a::Float64, x::Float64)
    FPMIN = 1.0e-300
    gln = _gammln(a)
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    @inbounds for i in 1:1000
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        abs(d) < FPMIN && (d = FPMIN)
        c = b + an / c
        abs(c) < FPMIN && (c = FPMIN)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) <= eps(Float64) && break
    end
    return exp(-x + a * log(x) - gln) * h
end

"""
    _gammp(a, x) -> Float64

Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, the χ²(2a)
CDF evaluated at `2x`. Internal; valid for `a > 0`, `x ≥ 0`. Series for
`x < a+1`, continued fraction for the complement otherwise (Numerical
Recipes). Accurate to ≈1e-12 for the `a = ndof/2 ≲ 100` we use.
"""
function _gammp(a::Float64, x::Float64)
    (x < 0.0 || a <= 0.0) && throw(DomainError((a, x), "_gammp requires a>0, x≥0"))
    x == 0.0 && return 0.0
    if x < a + 1.0
        return _gammp_series(a, x)
    else
        return 1.0 - _gammq_cf(a, x)
    end
end

"""
    _invgammp(p, a) -> Float64

Inverse of [`_gammp`](@ref): the `x` solving `P(a, x) = p`, for
`0 ≤ p ≤ 1`, `a > 0`. Halley iteration from a moment-matched initial
guess (Numerical Recipes 3rd ed., `Gamma::invgammp`).
"""
function _invgammp(p::Float64, a::Float64)
    a > 0.0 || throw(DomainError(a, "_invgammp requires a > 0"))
    (p < 0.0 || p > 1.0) && throw(DomainError(p, "_invgammp requires 0 ≤ p ≤ 1"))
    p <= 0.0 && return 0.0
    p >= 1.0 && return max(100.0, a + 100.0 * sqrt(a))
    EPS = 1.0e-8
    gln = _gammln(a)
    a1 = a - 1.0
    lna1 = a > 1.0 ? log(a1) : 0.0
    afac = a > 1.0 ? exp(a1 * (lna1 - 1.0) - gln) : 0.0
    # Initial guess.
    local x::Float64
    if a > 1.0
        pp = p < 0.5 ? p : 1.0 - p
        t = sqrt(-2.0 * log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        p < 0.5 && (x = -x)
        x = max(1.0e-3, a * (1.0 - 1.0 / (9.0 * a) - x / (3.0 * sqrt(a)))^3)
    else
        t = 1.0 - a * (0.253 + a * 0.12)
        if p < t
            x = (p / t)^(1.0 / a)
        else
            x = 1.0 - log(1.0 - (p - t) / (1.0 - t))
        end
    end
    # Halley refinement (≤ 12 iterations).
    @inbounds for _ in 1:12
        x <= 0.0 && return 0.0
        err = _gammp(a, x) - p
        t = a > 1.0 ? afac * exp(-(x - a1) + a1 * (log(x) - lna1)) :
                      exp(-x + a1 * log(x) - gln)
        u = err / t
        # Halley step (NR): `t` is reused as the increment.
        t = u / (1.0 - 0.5 * min(1.0, u * ((a - 1.0) / x - 1.0)))
        x -= t
        x <= 0.0 && (x = 0.5 * (x + t))   # half-step back into the domain
        abs(t) < EPS * x && break
    end
    return x
end

# Convert a `cl` argument (iminuit convention) to a probability in (0,1).
function _cl_to_prob(cl::Real)
    clf = Float64(cl)
    clf > 0.0 || throw(DomainError(cl, "confidence level `cl` must be > 0"))
    if clf < 1.0
        return clf                       # already a probability
    else
        # nσ → Gaussian probability mass within ±nσ = χ²(1) CDF at cl²
        #    = P(1/2, cl²/2)   (= erf(cl/√2)).
        return _gammp(0.5, 0.5 * clf * clf)
    end
end

"""
    delta_chisq(cl, ndof) -> Float64

The **Δχ² threshold** `χ²(ndof)`-quantile at confidence level `cl`, i.e.
the amount by which χ² rises from its minimum at the edge of the `cl`
confidence region for `ndof` jointly-estimated parameters.

# `cl` convention (matches iminuit)

- `0 < cl < 1` — interpret `cl` as a **probability** (e.g. `0.95`).
- `cl ≥ 1`     — interpret `cl` as **nσ** (Gaussian-equivalent): `1`→68.27 %,
  `2`→95.45 %, `3`→99.73 %.

# ⚠ Joint vs. single-parameter — READ THIS

`ndof` is the number of parameters **defining the region**, NOT the total
number of fit parameters. The two questions below give *different* Δχ²:

| Question                                              | `ndof` | 1σ Δχ² |
|:------------------------------------------------------|:------:|:------:|
| 1-D interval on **one** parameter (MINOS error)       |  `1`   | `1.00` |
| 2-D **joint** region for **two** parameters           |  `2`   | `2.30` |
| 3-D **joint** region for **three** parameters         |  `3`   | `3.53` |

A common mistake is to use Δχ²=1 for a 2-D contour: the 68 % **joint**
2-parameter region is Δχ²=**2.30**, not 1. The Monte-Carlo sampler
[`get_contours_samples`](@ref) samples all free parameters jointly, so
its default threshold uses `ndof = n_free`.

# Examples

```julia
delta_chisq(0.6827, 1)  # ≈ 1.00   (1σ, one parameter)
delta_chisq(0.6827, 2)  # ≈ 2.30   (1σ joint, two parameters)
delta_chisq(0.6827, 3)  # ≈ 3.53   (1σ joint, three parameters)
delta_chisq(0.95,   1)  # ≈ 3.84   (95 %, one parameter)
delta_chisq(1,      2)  # ≈ 2.30   (cl=1 ⇒ 1σ ⇒ 0.6827 ⇒ 2.30)
delta_chisq(2,      1)  # ≈ 4.00   (2σ, one parameter)
```

See also [`chisq_cl`](@ref) (the inverse), and `docs/ERROR_ANALYSIS.md`.
"""
function delta_chisq(cl::Real, ndof::Real)
    ndof > 0 || throw(DomainError(ndof, "ndof must be a positive integer"))
    p = _cl_to_prob(cl)
    return 2.0 * _invgammp(p, 0.5 * Float64(ndof))
end

"""
    chisq_cl(dchisq, ndof) -> Float64

Inverse of [`delta_chisq`](@ref): the **probability** (confidence level,
in `(0,1)`) that a `χ²(ndof)` variate is `≤ dchisq`. Equivalently the
χ²(`ndof`) CDF at `dchisq`.

```julia
chisq_cl(1.0,  1)  # ≈ 0.6827
chisq_cl(2.30, 2)  # ≈ 0.6827
chisq_cl(3.84, 1)  # ≈ 0.95
```

Round-trips with `delta_chisq` when `cl` is a probability:
`chisq_cl(delta_chisq(p, k), k) ≈ p` for `0 < p < 1`.
"""
function chisq_cl(dchisq::Real, ndof::Real)
    ndof > 0 || throw(DomainError(ndof, "ndof must be a positive integer"))
    d = Float64(dchisq)
    d >= 0 || throw(DomainError(dchisq, "dchisq must be ≥ 0"))
    return _gammp(0.5 * Float64(ndof), 0.5 * d)
end

# ─────────────────────────────────────────────────────────────────────────────
# Part (b): Monte-Carlo true-Δχ² region with proposal under-coverage handling.
# ─────────────────────────────────────────────────────────────────────────────

# Build a "square-root" factor S with S·Sᵀ = Σ so that `best + S·z`
# (z ~ N(0,I)) is distributed `MvNormal(best, Σ)`. Cholesky when Σ is
# positive-definite; eigen-decomposition fallback when it is not — this is
# the hand-rolled MvNormal the spec asks for, with NO Distributions.jl
# dependency.
#
# Non-PD note: negative eigenvalues (a `MnPosDef`-forced or numerically
# indefinite Σ) are clamped to 0, so the proposal puts ZERO spread along
# those degenerate directions. That makes the proposal mildly mis-specified
# there, but the true-Δχ² acceptance gate + adaptive widening / the
# covariance-free `:uniform` proposal are the intended remedies (an
# unreliable Σ already triggers the warn-and-steer path).
function _mvnormal_factor(Σ::AbstractMatrix{<:Real})
    S = Symmetric(Matrix{Float64}(Σ))
    try
        return Matrix(cholesky(S).L)
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
        E = eigen(S)
        return E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
    end
end

# Per-parameter (min,max) over the columns 1:m of an n×m sample buffer.
function _column_bounds(buf::Matrix{Float64}, n::Int, m::Int)
    bounds = Vector{Tuple{Float64,Float64}}(undef, n)
    if m == 0
        @inbounds for i in 1:n
            bounds[i] = (NaN, NaN)
        end
        return bounds
    end
    @inbounds for i in 1:n
        lo = buf[i, 1]
        hi = buf[i, 1]
        for j in 2:m
            v = buf[i, j]
            v < lo && (lo = v)
            v > hi && (hi = v)
        end
        bounds[i] = (lo, hi)
    end
    return bounds
end

# Relative change between two per-parameter bound sets (used as the
# "accepted region stopped growing" convergence signal across widening
# rounds). Returns the max over parameters of the relative shift of
# either edge, scaled by the current half-extent.
function _bounds_rel_change(cur::Vector{Tuple{Float64,Float64}},
                            prev::Vector{Tuple{Float64,Float64}})
    worst = 0.0
    @inbounds for i in eachindex(cur)
        clo, chi = cur[i]
        plo, phi = prev[i]
        (isfinite(clo) && isfinite(chi) && isfinite(plo) && isfinite(phi)) || return Inf
        scale = max(chi - clo, 1e-300)
        worst = max(worst, abs(clo - plo) / scale, abs(chi - phi) / scale)
    end
    return worst
end

"""
    _mc_chisq_region(best, χ², fmin, threshold; kwargs...) -> NamedTuple

Core Monte-Carlo true-Δχ² region sampler (coordinate-free; works on a
plain free-parameter vector and a χ²-like callable). `get_contours_samples`
wraps this with `Minuit`-level metadata. Exposed (un-exported) for direct
unit testing of the under-coverage logic.

- `best::Vector{Float64}` — best-fit free-parameter values (the proposal
  centre).
- `χ²` — callable `χ²(x::Vector{Float64}) -> Real`; the FCN value (NOT
  Δχ²) at a free-parameter vector. Must be thread-safe iff `threaded`.
- `fmin::Real` — the FCN value at `best` (subtracted to form Δ).
- `threshold::Real` — accept a sample iff `χ²(x) − fmin ≤ threshold`.
  (Caller passes `up · delta_chisq(cl, ndof)`; for a χ² fit `up = 1`.)

Keyword arguments: see [`get_contours_samples`](@ref).
"""
function _mc_chisq_region(best::Vector{Float64}, χ², fmin::Real, threshold::Real;
                          Σ::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                          ranges::Union{Nothing,AbstractVector} = nothing,
                          proposal::Symbol = :mvnormal,
                          nsamples::Integer = 10_000,
                          inflate::Real = 1.0,
                          adaptive::Bool = true,
                          max_widen_rounds::Integer = 5,
                          widen_factor::Real = 1.6,
                          clip_threshold::Real = 0.9,
                          extent_tol::Real = 0.02,
                          threaded::Bool = false,
                          rng::Random.AbstractRNG = Random.default_rng(),
                          mahalanobis::Bool = false)
    n = length(best)
    nsamples > 0 || throw(ArgumentError("nsamples must be > 0"))
    inflate > 0 || throw(ArgumentError("inflate must be > 0"))
    widen_factor > 1 || throw(ArgumentError("widen_factor must be > 1"))
    fmin_f = Float64(fmin)
    thr = Float64(threshold)

    # Proposal generators. Both draw INTO a preallocated n×nsamples buffer
    # SERIALLY (cheap; reproducible regardless of thread count). The
    # expensive χ² evaluation is what gets threaded.
    local draw!::Function
    if proposal === :mvnormal
        Σ === nothing &&
            throw(ArgumentError("proposal=:mvnormal needs a covariance Σ (got nothing); " *
                                "pass proposal=:uniform with `ranges` for a covariance-free run"))
        size(Σ, 1) == n == size(Σ, 2) ||
            throw(DimensionMismatch("Σ must be $n×$n to match best"))
        S = _mvnormal_factor(Σ)        # S·Sᵀ = Σ (Cholesky, or eigen if non-PD)
        z = Vector{Float64}(undef, n)
        draw! = function (buf, m, infl)
            @inbounds for j in 1:m
                for k in 1:n
                    z[k] = randn(rng)
                end
                for k in 1:n
                    acc = 0.0
                    for l in 1:n
                        acc += S[k, l] * z[l]
                    end
                    buf[k, j] = best[k] + infl * acc
                end
            end
            return buf
        end
    elseif proposal === :uniform
        ranges === nothing &&
            throw(ArgumentError("proposal=:uniform needs `ranges` (a vector of (lo,hi) per free parameter)"))
        length(ranges) == n ||
            throw(DimensionMismatch("ranges length $(length(ranges)) ≠ number of free parameters $n"))
        center = Vector{Float64}(undef, n)
        halfw = Vector{Float64}(undef, n)
        @inbounds for k in 1:n
            lo, hi = Float64(ranges[k][1]), Float64(ranges[k][2])
            hi > lo || throw(ArgumentError("ranges[$k] must have hi > lo, got ($lo, $hi)"))
            center[k] = 0.5 * (lo + hi)
            halfw[k] = 0.5 * (hi - lo)
        end
        draw! = function (buf, m, infl)
            @inbounds for j in 1:m, k in 1:n
                buf[k, j] = center[k] + infl * halfw[k] * (2.0 * rand(rng) - 1.0)
            end
            return buf
        end
    else
        throw(ArgumentError("proposal must be :mvnormal or :uniform, got :$proposal"))
    end

    propbuf = Matrix{Float64}(undef, n, nsamples)   # this round's proposals
    dfvals = Vector{Float64}(undef, nsamples)        # this round's Δ = χ² − fmin
    accepted = Matrix{Float64}(undef, n, 0)          # cumulative accepted (grown)
    acc_deltas = Float64[]                            # cumulative accepted Δ (FCN units)
    n_acc = 0
    n_total = 0
    widen_rounds = 0
    inflate_cur = Float64(inflate)
    under_coverage = false
    prev_bounds = nothing

    while true
        draw!(propbuf, nsamples, inflate_cur)
        # Evaluate Δ = χ²(x) − fmin for every proposal (threaded if safe).
        if threaded && Threads.nthreads() > 1
            Threads.@threads :static for j in 1:nsamples
                @inbounds dfvals[j] = Float64(χ²(@view propbuf[:, j])) - fmin_f
            end
        else
            @inbounds for j in 1:nsamples
                dfvals[j] = Float64(χ²(@view propbuf[:, j])) - fmin_f
            end
        end
        n_total += nsamples

        # This round's accepted columns + extents, and grow the cumulative
        # accepted buffer.
        round_acc = 0
        @inbounds for j in 1:nsamples
            (isfinite(dfvals[j]) && dfvals[j] <= thr) && (round_acc += 1)
        end
        if round_acc > 0
            newcols = Matrix{Float64}(undef, n, round_acc)
            c = 0
            @inbounds for j in 1:nsamples
                if isfinite(dfvals[j]) && dfvals[j] <= thr
                    c += 1
                    for k in 1:n
                        newcols[k, c] = propbuf[k, j]
                    end
                    push!(acc_deltas, dfvals[j])
                end
            end
            accepted = hcat(accepted, newcols)
            n_acc += round_acc
        end

        # Under-coverage detector: per-parameter, how much of the PROPOSED
        # extent the ACCEPTED samples fill (this round). A ratio ≈ 1 means
        # the proposal is clipping the true region (too-tight Σ, or a
        # nonlinear region extending beyond the local Gaussian).
        clip = 0.0
        if round_acc >= 2
            @inbounds for k in 1:n
                plo = phi = propbuf[k, 1]
                for j in 2:nsamples
                    v = propbuf[k, j]
                    v < plo && (plo = v); v > phi && (phi = v)
                end
                alo = ahi = NaN
                for j in 1:nsamples
                    if isfinite(dfvals[j]) && dfvals[j] <= thr
                        v = propbuf[k, j]
                        if isnan(alo)
                            alo = ahi = v
                        else
                            v < alo && (alo = v); v > ahi && (ahi = v)
                        end
                    end
                end
                pext = phi - plo
                pext > 0 && (clip = max(clip, (ahi - alo) / pext))
            end
        end

        cum_bounds = _column_bounds(accepted, n, n_acc)
        extent_converged = prev_bounds !== nothing &&
                           _bounds_rel_change(cum_bounds, prev_bounds) < extent_tol
        prev_bounds = cum_bounds

        # Decide whether to stop or widen.
        if clip < clip_threshold
            break                                    # proposal covers the region
        end
        if !adaptive
            under_coverage = true                    # clipped, but widening disabled
            break
        end
        if extent_converged && widen_rounds >= 1
            # The accepted region stopped growing across a widening round, so
            # we treat it as captured (e.g. a high-cl region that legitimately
            # fills the proposal). Be honest, though: if this round is STILL
            # clipping (clip ≥ threshold) we flag under_coverage so the caller
            # is not misled into reading a converged-but-clipped region as
            # fully covered.
            under_coverage = clip >= clip_threshold
            break
        end
        if widen_rounds >= max_widen_rounds
            under_coverage = true                    # hit the widening cap, still growing
            break
        end
        inflate_cur *= Float64(widen_factor)
        widen_rounds += 1
    end

    bounds = _column_bounds(accepted, n, n_acc)

    # Optional Mahalanobis diagnostic (NEVER the acceptance criterion —
    # included only to reproduce the X(3872) notebook's diagnostic plots).
    maha = nothing
    Σinv = nothing
    if mahalanobis && n_acc > 0 && Σ !== nothing
        # Diagnostic only; a singular/badly-conditioned Σ just yields no
        # Mahalanobis output rather than throwing.
        Σinv = try
            inv(Symmetric(Matrix{Float64}(Σ)))
        catch
            nothing
        end
    end
    if Σinv !== nothing
        maha = Vector{Float64}(undef, n_acc)
        d = Vector{Float64}(undef, n)
        @inbounds for j in 1:n_acc
            for k in 1:n
                d[k] = accepted[k, j] - best[k]
            end
            acc = 0.0
            for k in 1:n, l in 1:n
                acc += d[k] * Σinv[k, l] * d[l]
            end
            maha[j] = acc
        end
    end

    return (; samples = permutedims(accepted),   # n_accepted × n_free (row = sample)
              bounds,                             # Vector{(min,max)} per free param
              best = copy(best),
              deltas = acc_deltas,                # accepted Δ = χ² − fmin (FCN units)
              n_accepted = n_acc,
              n_total,
              acceptance = n_total > 0 ? n_acc / n_total : 0.0,
              widen_rounds,
              inflate_final = inflate_cur,
              threshold = thr,
              under_coverage,
              proposal,
              mahalanobis = maha)
end

"""
    get_contours_samples(m::Minuit; kwargs...) -> NamedTuple
    get_contours_samples(m::Minuit, χsq, paras=nothing, ranges=nothing; kwargs...)

Monte-Carlo **true-Δχ²** error region for a converged (or even an
ill-converged) [`Minuit`](@ref) fit — the principled alternative to MINOS
when the χ² posterior is non-parabolic or `m.fmin` is unreliable.

The method (mirroring the X(3872) published analysis):

1. Draw `nsamples` proposals for **all free parameters jointly** from
   `MvNormal(best, inflate²·Σ)` (`proposal=:mvnormal`, the default) or
   uniformly over a user box (`proposal=:uniform` with `ranges`).
2. **Keep** a proposal iff its TRUE Δχ² satisfies
   `χ²(x) − χ²_min ≤ up · delta_chisq(cl, ndof)` — the exact FCN is
   re-evaluated at every sample; the Gaussian is ONLY the proposal, never
   the acceptance test.
3. Report the kept sets and their per-parameter `(min, max)` extents.

# Proposal under-coverage (the critical pitfall)

A `MvNormal(best, Σ)` proposal **severely under-estimates** the region
when Σ is unreliable (poorly-converged / invalid fit, or `MnPosDef`-forced
covariance) or when the true region is highly nonlinear and extends beyond
the local Gaussian. This routine guards against that:

- `inflate::Real=1` — sample `MvNormal(best, inflate²·Σ)` (manual widening).
- `adaptive::Bool=true` — detect proposal-limited acceptance (the accepted
  extent filling the proposed extent) and geometrically grow `inflate`
  (`widen_factor`, up to `max_widen_rounds`), re-sampling until the region
  is covered. `widen_rounds` and `under_coverage` are reported.
- `proposal=:uniform` + `ranges` — a covariance-FREE box proposal that does
  not depend on Σ at all (use when Σ is meaningless).
- On `!is_valid` / `made_pos_def` / `hesse_failed`, a `MvNormal` run emits a
  warning recommending these mitigations rather than silently under-estimating.

# Keyword arguments

- `χsq` — optional explicit `χsq(x_full)::Real` over the **full external**
  parameter vector. Defaults to the fit's own FCN (`m.fcn`).
- `paras` — parameters (indices or names) to report `bounds` for; default
  all free parameters. (Sampling is always over all free parameters.)
- `nsamples::Integer=10_000` — proposals per round.
- `cl::Real=1` — confidence level (`<1` probability, `≥1` nσ; see
  [`delta_chisq`](@ref)).
- `ndof::Integer` — dof for the Δχ² threshold; default `n_free` (the joint
  region over all sampled parameters). **Keep `ndof = n_free`** unless you
  know you want a sub-dimensional threshold.
- `proposal::Symbol=:mvnormal` — `:mvnormal` or `:uniform`.
- `ranges` — `(lo,hi)` per free parameter; required for `:uniform`.
- `inflate`, `adaptive`, `max_widen_rounds`, `widen_factor` — see above.
- `clip_threshold::Real=0.9` — a round is "proposal-limited" when the
  accepted samples fill more than this fraction of the proposed extent in
  some parameter; triggers widening.
- `extent_tol::Real=0.02` — the accepted region is "captured" (stop
  widening) once its per-parameter `(min,max)` change between rounds drops
  below this relative tolerance.
- `threaded::Bool=false` — evaluate χ² over samples in parallel. **Phase-H
  aware**: the FCN thread-safety is verified first (on a throwaway counter
  so `m.nfcn` is untouched); a racey FCN falls back to serial with a warning
  (a racing FCN would corrupt acceptance). The χ² evaluations bypass the
  fit's call counter, so sampling does not change `m.nfcn`. Bounded
  parameters are sampled in external coordinates; proposals outside a
  parameter's limits are rejected before the FCN is called.
- `seed::Integer` — RNG seed for reproducible proposals (proposals are drawn
  serially, so results are reproducible regardless of `threaded`).
- `mahalanobis::Bool=false` — also return the per-sample Mahalanobis
  distance (DIAGNOSTIC only — never used for acceptance).

# Returns

A `NamedTuple` with fields: `samples` (`n_accepted × n_free` matrix, one row
per kept set), `bounds` (`Vector{(min,max)}` per reported parameter), `best`,
`names`, `free_names`, `delta_chisq_values` (per-sample true Δχ²),
`n_accepted`, `n_total`, `acceptance` (kept fraction — for a Gaussian
posterior this is ≈ `cl` **only when `ndof == n_free`**, a useful sanity
check), `widen_rounds`, `inflate_final`, `delta` (the Δχ² threshold in χ²
units), `up`, `cl`, `ndof`, `proposal`, `under_coverage` (true if widening
hit its cap, or stopped on a still-clipping round, while the region was
plausibly larger than sampled), and `mahalanobis` (or `nothing`).

See [`contour_df_samples`](@ref) for a `DataFrame` of `samples`, and
`docs/ERROR_ANALYSIS.md` for the full discussion + a worked example.
"""
function get_contours_samples(m::Minuit;
                              χsq = nothing,
                              paras = nothing,
                              nsamples::Integer = 10_000,
                              cl::Real = 1,
                              ndof::Union{Integer,Nothing} = nothing,
                              proposal::Symbol = :mvnormal,
                              ranges = nothing,
                              inflate::Real = 1.0,
                              adaptive::Bool = true,
                              max_widen_rounds::Integer = 5,
                              widen_factor::Real = 1.6,
                              clip_threshold::Real = 0.9,
                              extent_tol::Real = 0.02,
                              threaded::Bool = false,
                              seed::Union{Integer,Nothing} = nothing,
                              mahalanobis::Bool = false,
                              warn::Bool = true)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `get_contours_samples(m)`"))

    ntot = n_pars(m.params)
    free_idx = [i for i in 1:ntot if !is_fixed(m.params.pars[i])]
    nfree = length(free_idx)
    nfree >= 1 || throw(ArgumentError("no free parameters to sample"))

    best_full = collect(Float64, m.values)            # full external best fit
    best_free = best_full[free_idx]
    fmin = Float64(m.fval)
    isfinite(fmin) ||
        throw(ArgumentError("m.fval is not finite — run a successful `migrad(m)` first"))
    up = Float64(m.fcn.up)

    ndof_use = ndof === nothing ? nfree : Int(ndof)
    δ = delta_chisq(cl, ndof_use)
    threshold = up * δ

    # Per-free-parameter external bounds (NaN ⇒ unbounded on that side).
    # The proposals are drawn in EXTERNAL coordinates and ignore limits, so
    # a bounded parameter can be proposed outside its physical range; we
    # reject such draws (Δ = +Inf) BEFORE calling the user FCN rather than
    # evaluating it on an illegal argument. Rejection (not clamping) keeps
    # the true-Δχ² region unbiased.
    lo_free = [m.params.pars[i].lower for i in free_idx]
    hi_free = [m.params.pars[i].upper for i in free_idx]
    has_bounds = any(isfinite, lo_free) || any(isfinite, hi_free)

    # χ² closure over a FREE-parameter vector: expand to full external,
    # call the user FCN. Allocates a fresh `full` per call → thread-safe.
    userf = χsq === nothing ? m.fcn.f : χsq
    χ²free = let base = best_full, fi = free_idx, f = userf,
                 lo = lo_free, hi = hi_free, chk = has_bounds
        x -> begin
            if chk
                @inbounds for j in eachindex(x)
                    (isnan(lo[j]) || x[j] >= lo[j]) || return Inf
                    (isnan(hi[j]) || x[j] <= hi[j]) || return Inf
                end
            end
            full = copy(base)
            @inbounds for (j, i) in enumerate(fi)
                full[i] = x[j]
            end
            return Float64(f(full))
        end
    end

    # Covariance for the MvNormal proposal (free-parameter block).
    Σfree = nothing
    if proposal === :mvnormal
        cov = free_covariance(m.fmin)
        cov === nothing &&
            throw(ArgumentError("no covariance available for proposal=:mvnormal; " *
                                "run `hesse!(m)` or use proposal=:uniform with `ranges`"))
        Σfree = Matrix{Float64}(cov)
    end

    # Warn + steer on unreliable covariance (do NOT silently under-estimate).
    if warn && proposal === :mvnormal
        fm = m.fmin.internal
        cov_status = fm.state.error.status
        unreliable = !is_valid(m.fmin) || fm.made_pos_def ||
                     cov_status == MnHesseFailed || cov_status == MnMadePosDef ||
                     cov_status == MnNotPosDef
        if unreliable
            @warn """get_contours_samples: the fit covariance looks unreliable \
(is_valid=$(is_valid(m.fmin)), made_pos_def=$(fm.made_pos_def), cov_status=$cov_status). \
A MvNormal proposal centred on this Σ can SEVERELY UNDER-ESTIMATE the error region. \
Mitigations: rely on adaptive widening (adaptive=$adaptive), raise `inflate`, or use a \
covariance-free `proposal=:uniform` with explicit `ranges`. See docs/ERROR_ANALYSIS.md."""
        end
    end

    # Phase-H-aware threading: a racey FCN under parallel evaluation would
    # corrupt the acceptance test, so verify thread-safety first and fall
    # back to serial (with a warning) if it fails.
    threaded_eff = threaded
    if threaded && Threads.nthreads() > 1
        # Probe a COUNTER-FREE wrapper (fresh `nfcn` Ref) so the safety
        # check — which evaluates the FCN — does not pollute the user's
        # `m.nfcn`, and so the very `nfcn[] += 1` counter race we avoid in
        # the main loop is not exercised on the shared counter here either.
        safe = try
            is_thread_safe(CostFunction(m.fcn.f, m.fcn.up), best_full)
        catch
            false
        end
        if !safe
            threaded_eff = false
            warn && @warn "get_contours_samples: FCN is not thread-safe (Phase H check failed); " *
                          "falling back to serial χ² evaluation. Pass threaded=false to silence."
        end
    elseif threaded
        threaded_eff = false   # single Julia thread → nothing to parallelize
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    res = _mc_chisq_region(best_free, χ²free, fmin, threshold;
                           Σ = Σfree, ranges = ranges, proposal = proposal,
                           nsamples = nsamples, inflate = inflate,
                           adaptive = adaptive, max_widen_rounds = max_widen_rounds,
                           widen_factor = widen_factor, clip_threshold = clip_threshold,
                           extent_tol = extent_tol, threaded = threaded_eff,
                           rng = rng, mahalanobis = mahalanobis)

    free_names = [m.params.pars[i].name for i in free_idx]

    # Optionally restrict the reported `bounds`/`names` to `paras`.
    report_local = collect(1:nfree)
    if paras !== nothing
        sel = paras isa Union{Integer,AbstractString,Symbol} ? [paras] : collect(paras)
        report_local = Int[]
        for p in sel
            ext = p isa Integer ? Int(p) : ext_index(m.params, String(p))
            loc = findfirst(==(ext), free_idx)
            loc === nothing &&
                throw(ArgumentError("parameter $p is fixed or out of range — cannot report bounds"))
            push!(report_local, loc)
        end
    end

    # Ascribe the fields pulled from `res` to their concrete types so the
    # public NamedTuple return is type-stable (review IMPORTANT #1).
    bounds_all = res.bounds::Vector{Tuple{Float64,Float64}}
    best_all = res.best::Vector{Float64}
    return (; samples = res.samples::Matrix{Float64},
              bounds = bounds_all[report_local],
              best = best_all[report_local],
              names = free_names[report_local],
              free_names = free_names,
              delta_chisq_values = (res.deltas::Vector{Float64}) ./ up,  # per-sample Δχ²
              n_accepted = res.n_accepted::Int,
              n_total = res.n_total::Int,
              acceptance = res.acceptance::Float64,
              widen_rounds = res.widen_rounds::Int,
              inflate_final = res.inflate_final::Float64,
              delta = δ::Float64,
              up = up,
              cl = Float64(cl),
              ndof = ndof_use,
              proposal = res.proposal::Symbol,
              under_coverage = res.under_coverage::Bool,
              mahalanobis = res.mahalanobis::Union{Nothing,Vector{Float64}})
end

# IMinuit.jl-style positional form: `get_contours_samples(m, χsq, paras, ranges)`.
function get_contours_samples(m::Minuit, χsq, paras = nothing, ranges = nothing; kwargs...)
    return get_contours_samples(m; χsq = χsq, paras = paras, ranges = ranges, kwargs...)
end

"""
    contour_df_samples(m::Minuit; kwargs...) -> DataFrame

`DataFrame` of the accepted Monte-Carlo parameter sets from
[`get_contours_samples`](@ref): one row per kept set, one column per free
parameter (named after the parameters), plus a `:delta_chisq` column with
each set's true Δχ². Requires `using DataFrames` (provided by the
`JuMinuitDataFramesExt` package extension).

Accepts all [`get_contours_samples`](@ref) keyword arguments.
"""
function contour_df_samples end
