# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Δχ²-region extremization of DERIVED quantities, and the pointwise
# profile-likelihood band built from it.
#
# Minuit has MINOS for a *parameter*, but nothing for a derived scalar
# f(θ) — a peak position, a ratio of amplitudes, a model curve evaluated
# at one x, a Legendre moment… The exact (profile/Wilks) `cl` interval
# for such an f is the image of the Δχ² region under f:
#
#     I_f = [ min f(θ), max f(θ) ]   s.t.   FCN(θ) ≤ FCN_min + Δχ²·up ,
#
# with ALL free parameters varied simultaneously and Δχ² = delta_chisq(cl, 1)
# — `ndof = 1` because the quoted statement is ONE scalar, no matter how many
# parameters move (re-parametrize so f is itself a coordinate ⇒ a single-
# parameter interval; Wilks: the profile Δχ² of one constraint has 1 dof; in
# the linear-Gaussian limit the Lagrange condition gives the projection
# theorem  max f = f̂ + √(Δχ²·cᵀCc)  exactly). For f(θ) = θᵢ the construction
# reduces to MINOS, which is the same Δχ² = up crossing for cl = 1.
#
# Implementation (the production-validated recipe of the OPE-1c analysis):
# exterior penalty  obj(θ) = −sgn·f(θ) + λ·max(0, (FCN(θ) − bound)/up)²
# minimized by MIGRAD from MULTIPLE seeds, acceptance gate
# FCN ≤ bound + accept_tol·up, then a LOCAL pull-back onto the boundary so
# every reported endpoint is feasible (FCN ≤ bound exactly). Multiple seeds
# are a hard requirement, not an optimization: a single-seed run can stop at
# a *local* tangency and silently report a too-narrow interval when the
# region has several low-χ² corridors (an under-extremization incident that
# motivated this API — the fix was seeding from ensemble extremes). The
# per-seed diagnostics exist so that failure mode stays auditable.
#
# `profile_band` sweeps `extremize` over a grid x ↦ f(x, θ) with warm starts
# (the previous point's extremal parameters seed the next point), forward +
# reverse passes (keeping the better envelope), and a "band contains the
# best fit" guarantee that holds by construction (θ̂ is in the region).
#
# This file is JuMinuit-native functionality (iminuit has no equivalent —
# its `util.propagate` is first-order linear propagation only); the
# `cl`/`delta_chisq` conventions match the rest of the package.

# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExtremizeResult

Result of [`extremize`](@ref): the profile (Δχ²-region) interval of a derived
scalar `f(θ)`, the extremal parameter vectors realizing it, and the per-seed
audit trail.

# Fields
- `lo::Float64`, `hi::Float64` — the interval endpoints `[min f, max f]` over
  the region `FCN ≤ bound`. In the typical case each is attained at a
  FEASIBLE point (`FCN ≤ bound` exactly, after the boundary pull-back); a
  candidate whose pull-back is not locally possible is kept raw, within
  `FCN ≤ bound + accept_tol·up` (its record has `projected = false`).
- `plo`, `phi` — full external parameter vectors at which `lo`/`hi` are
  attained (fixed parameters pinned at their values).
- `fbest::Float64` — `f` at the best fit; `lo ≤ fbest ≤ hi` by construction.
- `bound::Float64` — the FCN acceptance bound `m.fval + delta·up`.
- `delta::Float64` — the Δχ² threshold actually used (`delta_chisq(cl, 1)`,
  or the explicit `delta` override).
- `cl::Float64` — the confidence-level argument as given (`NaN` when an
  explicit `delta` override was used).
- `up::Float64` — the fit's error definition (`m.up`).
- `diagnostics::NamedTuple` — `(min, max, winner_min, winner_max,
  naccepted_min, naccepted_max, fcn_min, fcn_max)`. `min`/`max` are per-seed
  record vectors, one row per penalty fit, with fields `seed` (index into
  the seed pool; seed 1 is always the best fit), `converged` (MIGRAD
  validity), `accepted` (passed the `bound + accept_tol·up` gate),
  `projected` (was pulled back onto the boundary), `fcn` (FCN at the raw
  penalty optimum), `f_raw`/`f` (`f` before/after the pull-back) and `nfcn`.
  `winner_min`/`winner_max` give the seed index whose candidate won each
  side — `0` means no penalty fit beat the best-fit value itself, which has
  two very different readings, disambiguated by `naccepted_*`: with
  `naccepted_* > 0` the best fit is genuinely extremal for that side (e.g.
  `f`'s unconstrained optimum sits at θ̂ — healthy); with `naccepted_* == 0`
  every penalty fit was rejected and that side FAILED (a warning fired).
  `fcn_min`/`fcn_max` are the FCN values at `plo`/`phi`: compare with
  `bound` to certify each endpoint's feasibility directly. Inspect these to
  audit seed coverage (the multi-corridor failure mode described in the
  docstring).
"""
struct ExtremizeResult{D<:NamedTuple}
    lo::Float64
    hi::Float64
    plo::Vector{Float64}
    phi::Vector{Float64}
    fbest::Float64
    bound::Float64
    delta::Float64
    cl::Float64
    up::Float64
    diagnostics::D
end

"""
    ProfileBand

Result of [`profile_band`](@ref): the pointwise profile-likelihood envelope
of a curve family `f(x, θ)` on a grid.

# Fields
- `x::Vector{Float64}` — the grid (a copy of `xs`).
- `lo`, `hi` — the band edges: per point, `[min f, max f]` over the region
  `FCN ≤ bound`. As in [`extremize`](@ref), an edge typically sits at a
  feasible point (`FCN ≤ bound` exactly, after the boundary pull-back) and
  always within `FCN ≤ bound + accept_tol·up` — check the per-point
  `fcn_lo`/`fcn_hi` diagnostics to certify. `NaN` only when
  `include_best = false` and every penalty fit at that point/side was
  rejected.
- `plo`, `phi` — per-point extremal parameter vectors (`nothing` exactly
  when the corresponding edge is `NaN`).
- `fbest::Vector{Float64}` — the best-fit curve `f(x, θ̂)`; with
  `include_best = true` (default) `lo .≤ fbest .≤ hi` by construction.
- `bound`, `delta`, `cl`, `up` — as in [`ExtremizeResult`](@ref).
- `nfail::Int` — number of (point, side, pass) extremization groups in
  which NO penalty fit passed the acceptance gate (the best-fit fallback
  still keeps the band finite when `include_best = true`); `0` on a
  healthy sweep.
- `diagnostics` — per-point NamedTuples `(x, failed_lo, failed_hi,
  accepted_lo, accepted_hi, nfits_lo, nfits_hi, fcn_lo, fcn_hi)`:
  cumulative accepted / attempted penalty-fit counts over all passes,
  `failed_*` flags for a side that NEVER had an accepted fit at that
  point, and the FCN values at the stored extremal points (= `m.fval` for
  a best-fit-fallback edge; `NaN` exactly when the edge is `NaN`).
"""
struct ProfileBand{D<:NamedTuple}
    x::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
    plo::Vector{Union{Nothing,Vector{Float64}}}
    phi::Vector{Union{Nothing,Vector{Float64}}}
    fbest::Vector{Float64}
    bound::Float64
    delta::Float64
    cl::Float64
    up::Float64
    nfail::Int
    diagnostics::Vector{D}
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Exception set treated as "this start point / probe point is outside the
# FCN's domain" rather than a bug — same guard as find_deeper_minimum's
# restart loop (a wild seed can push a constrained FCN into a throwing
# region: log of a negative, a singular matrix, …).
const _EXTREMIZE_CATCH =
    Union{DomainError,BoundsError,SingularException,ArgumentError,DivideError}

# FCN value at θ with the throw-guard: a probe outside the FCN's domain
# counts as infeasible (+Inf); it does not abort the extremization.
function _fcn_or_inf(fcnraw, θ::Vector{Float64})
    c = try
        fcnraw(θ)
    catch err
        err isa _EXTREMIZE_CATCH || rethrow()
        return Inf
    end
    return (c isa Real && isfinite(c)) ? Float64(c) : Inf
end

# Normalize the user `seeds` into the seed pool: full-length EXTERNAL
# parameter vectors, best fit FIRST (always seed #1), then the user seeds —
# a Vector of vectors, the rows of a Matrix (e.g. an MCMC ensemble's
# f-extreme members), or a single vector.
function _seed_pool(m::Minuit, seeds, fname::String)
    pool = [collect(Float64, m.values)]
    seeds === nothing && return pool
    rows = if seeds isa AbstractMatrix
        [collect(Float64, view(seeds, i, :)) for i in axes(seeds, 1)]
    elseif seeds isa AbstractVector{<:Real}
        [collect(Float64, seeds)]                  # a single seed vector
    else
        [collect(Float64, s) for s in seeds]
    end
    n = n_pars(m.params)
    for (k, s) in enumerate(rows)
        length(s) == n || throw(ArgumentError(
            "$fname: seed #$k has length $(length(s)) — expected $n " *
            "(a full EXTERNAL parameter vector, fixed parameters included)"))
        all(isfinite, s) || throw(ArgumentError(
            "$fname: seed #$k contains non-finite entries"))
        push!(pool, s)
    end
    return pool
end

# A usable start point built from a pool seed: fixed coordinates pinned at
# the fit's value (the region is defined WITH them fixed there), free ones
# clamped INTO their limits with a tiny relative inset — a start exactly ON
# a bound has dExt/dInt = 0 under the sin transform, leaving MIGRAD a dead
# direction at the seed.
function _usable_seed(m::Minuit, s::Vector{Float64})
    x = copy(s)
    @inbounds for i in eachindex(x)
        p = m.params.pars[i]
        if is_fixed(p)
            x[i] = p.value
            continue
        end
        lo, hi = p.lower, p.upper                  # NaN = absent
        (isnan(lo) && isnan(hi)) && continue
        inset = (!isnan(lo) && !isnan(hi)) ? 1e-8 * (hi - lo) :
                1e-8 * max(1.0, abs(isnan(lo) ? hi : lo))
        isnan(lo) || (x[i] = max(x[i], lo + inset))
        isnan(hi) || (x[i] = min(x[i], hi - inset))
    end
    return x
end

# The exterior-penalty objective for one direction (sgn = +1 maximizes f,
# sgn = −1 minimizes it). The constraint excess is normalized by `up` so
# `lambda` means the same thing for a χ² (up = 1) and a −lnL (up = 0.5) fit.
#
# The objective is TOTAL — it never throws and never returns ±Inf:
# - an FCN throw / non-finite value (a probe far outside the FCN's domain)
#   becomes a finite high plateau via the excess cap, not an Inf cliff. An
#   Inf reaching MIGRAD's numerical gradient turns the curvature matrix
#   into NaNs and aborts the whole fit from inside LinearAlgebra (observed
#   on a soft λ = 1 stage wandering up the Rosenbrock valley); a finite
#   plateau is simply never accepted by the line search.
# - a non-finite/throwing `f` maps to NaN, which MIGRAD rejects as a
#   candidate point (a non-finite fval can never look like a minimum).
function _penalty_obj(fcnraw, f, sgn::Int, bound::Float64, up::Float64,
                      lambda::Float64)
    return let fcnraw = fcnraw, f = f, s = Float64(sgn), bound = bound,
               up = up, lambda = lambda
        θ -> begin
            c = _fcn_or_inf(fcnraw, θ)              # throw / non-finite → Inf
            excess = min((c - bound) / up, 1e8)
            excess >= 1e8 && return lambda * 1e16   # flat plateau; skip f
            fv = try
                f(θ)
            catch err
                err isa _EXTREMIZE_CATCH || rethrow()
                return NaN
            end
            (fv isa Real && isfinite(fv)) || return NaN
            -s * Float64(fv) + (excess > 0.0 ? lambda * excess * excess : 0.0)
        end
    end
end

# Pull an accepted-but-slightly-exterior penalty optimum back ONTO the region
# boundary along the segment toward an interior anchor (the best fit), so the
# reported extremal point is feasible: FCN(θ) ≤ bound exactly. The walk is
# strictly LOCAL: a geometric ladder looks for the first feasible step at
# s ≤ 0.1 of the way to the anchor (a converged penalty optimum overshoots
# the boundary by O(1/lambda), so the crossing sits at tiny s), and the
# bisection then stays inside that one-decade bracket. If NO nearby step is
# feasible — the segment initially climbs AWAY from the region, e.g. from a
# stalled point on the anchor-facing edge of a far χ²-corridor — we return
# `nothing` rather than bisect across the barrier: that would drag the point
# back to the anchor's corridor and undo exactly the multi-corridor coverage
# the seed pool exists to provide (the caller then keeps the raw point,
# which is still within the acceptance tolerance).
function _project_to_bound(fcnraw, θstar::Vector{Float64},
                           anchor::Vector{Float64}, bound::Float64)
    d = anchor .- θstar
    at(s) = θstar .+ s .* d
    s_lo, s_hi = 0.0, NaN
    c_hi = NaN
    s = 1e-8
    while s <= 0.1
        c = _fcn_or_inf(fcnraw, at(s))
        if c <= bound
            s_hi = s
            c_hi = c
            break
        end
        s_lo = s
        s *= 10.0
    end
    isnan(s_hi) && return nothing      # no LOCAL feasible step — don't cross
    # Bisect [infeasible, feasible] keeping the feasible end; 40 halvings of
    # a ≤ one-decade bracket put the boundary mismatch far below any
    # statistically meaningful scale.
    for _ in 1:40
        sm = 0.5 * (s_lo + s_hi)
        c = _fcn_or_inf(fcnraw, at(sm))
        if c <= bound
            s_hi = sm
            c_hi = c
        else
            s_lo = sm
        end
    end
    return (θ = at(s_hi), fcn = c_hi)  # feasible by construction: fcn ≤ bound
end

# The penalty-continuation ladder up to the final stiffness `lambda`: a
# SINGLE MIGRAD at lambda = 1e4 systematically under-converges — the stiff
# penalty shell dominates the DFP curvature estimate, EDM goes tiny while
# the iterate has barely slid along the boundary toward the tangency point
# (observed: endpoints short by ~0.7σ_f on the analytic linear target). The
# classic fix is sequential penalty: solve a SOFT problem first (its optimum
# overshoots the boundary but lies on the true extremal ray), then re-solve
# with stiffer penalties warm-started from the previous stage. Two decades
# per stage, early stages at the O(1) balance scale (the excess is
# up-normalized, so O(1) is scale-free), final stage exactly `lambda`.
# Cold-seed accuracy on the analytic linear target: ~3e-8 relative, vs
# ~0.7 single-shot.
_penalty_ladder(lambda::Float64) =
    unique(Float64[min(lambda, 1.0), min(lambda, 100.0), lambda])

# One direction of the constrained extremization: penalty-continuation
# MIGRAD from each seed, gate on FCN ≤ bound + accept_tol·up, pull accepted
# exterior optima back onto the boundary (when the pull-back is locally
# possible), and keep the best candidate. The best-fit VALUE itself is a
# zero-cost feasible candidate when `include_best` (θ̂ is in the region), so
# the result can only be ≥ f̂ (max side) / ≤ f̂ (min side) — the "contains
# the best fit" construction property; `winner == 0` flags that fallback.
# Identical (post-pinning/clamping) starts are fitted only once, so record
# `seed` indices may skip duplicates.
function _extremize_dir(m::Minuit, f, sgn::Int, bound::Float64,
                        pool::Vector{Vector{Float64}}, fhat::Float64,
                        that::Vector{Float64};
                        lambda::Float64, accept_tol::Float64,
                        strategy, maxfcn, include_best::Bool, rounds::Int)
    fcnraw = m.fcn.f
    up = Float64(m.up)
    ladder = _penalty_ladder(lambda)

    best_v = include_best ? fhat : NaN
    best_p = include_best ? copy(that) : nothing
    best_c = include_best ? Float64(m.fval) : NaN   # FCN at the winning point
    winner = 0
    naccepted = 0

    records = NamedTuple{(:seed, :converged, :accepted, :projected,
                          :fcn, :f_raw, :f, :nfcn),
                         Tuple{Int,Bool,Bool,Bool,Float64,Float64,Float64,Int}}[]
    seen = Vector{Vector{Float64}}()
    for (k, raw) in enumerate(pool)
        x = _usable_seed(m, raw)
        any(s -> s == x, seen) && continue
        push!(seen, x)
        converged = false
        accepted = false
        projected = false
        c = NaN
        v_raw = NaN
        v = NaN
        nf = 0
        try
            # Each ladder stage gets its own Minuit clone of `m`: names,
            # limits, fixed flags, step sizes (post-fit errors), strategy and
            # tolerance are carried over, so the search runs in the same
            # constrained parameter space as the user's fit. `up = 1` because
            # no error analysis is ever done on the penalty objective. Stages
            # warm-start from the previous stage's optimum; step sizes are
            # re-seeded from `m` each stage (a fresh curvature estimate —
            # the stiffened problem invalidates the previous stage's).
            #
            # Subfit logs are suppressed: the penalty surface is C¹-kinked at
            # the boundary, so MIGRAD's internal warnings (e.g. the DFP
            # gvg ≤ 0 update skip) are expected noise there, and a band sweep
            # runs hundreds of these fits — real failures surface through the
            # acceptance gate and the diagnostics records instead. A stage may
            # still die on internal linear algebra under the extreme penalty
            # curvature; it is then skipped and the next (stiffer) stage
            # continues from the best point so far.
            #
            # The ladder is wrapped in up to `rounds` warm-started repeats:
            # inside the region the objective is just ∓f (zero penalty, and
            # for a near-linear f nearly zero curvature), so a single chain
            # can stall mid-way through a long interior traverse — e.g. along
            # a curved χ² valley — while reporting a feasible, innocuous-
            # looking endpoint. Re-running the ladder from the stall point
            # advances it; we stop as soon as a round no longer improves
            # sgn·f and keep the best endpoint seen.
            cur = x
            cur_best = x
            v_best = NaN
            local mm = nothing
            for _ in 1:rounds
                for lam in ladder
                    obj = _penalty_obj(fcnraw, f, sgn, bound, up, lam)
                    try
                        mmk = Minuit(obj, m; up = 1.0)
                        mmk.values = cur
                        Logging.with_logger(Logging.NullLogger()) do
                            migrad!(mmk; strategy = strategy, maxfcn = maxfcn)
                        end
                        cur = collect(Float64, mmk.values)
                        nf += mmk.nfcn
                        mm = mmk
                    catch err
                        err isa _EXTREMIZE_CATCH || rethrow()
                    end
                end
                # Guarded like the in-objective f: a throw here means the
                # round's endpoint sits on f's domain edge — stop iterating
                # and let the acceptance check judge the best point so far,
                # rather than unwinding (and losing) the whole seed.
                fv = try
                    f(cur)
                catch err
                    err isa _EXTREMIZE_CATCH || rethrow()
                    NaN
                end
                v_now = (fv isa Real && isfinite(fv)) ? Float64(fv) : NaN
                isnan(v_now) && break
                if isnan(v_best) || sgn * (v_now - v_best) > 0
                    improved = isnan(v_best) ||
                               sgn * (v_now - v_best) > 1e-9 + 1e-6 * abs(v_now)
                    cur_best = cur
                    v_best = v_now
                    improved || break
                else
                    break                  # the round went backwards — stop
                end
            end
            converged = mm === nothing ? false : mm.valid
            θ = cur_best
            c = _fcn_or_inf(fcnraw, θ)
            accepted = !isnan(v_best) && c <= bound + accept_tol * up
            if accepted
                v_raw = v_best             # = f(θ), computed in the loop
                v = v_raw
                θfin = θ
                cfin = c
                if c > bound                       # accepted but (slightly) exterior
                    proj = _project_to_bound(fcnraw, θ, that, bound)
                    if proj !== nothing            # local pull-back possible
                        # f at the pulled-back point, guarded like everywhere
                        # else: if it throws or is non-finite there, keep the
                        # raw point (still within the gate) instead of
                        # poisoning the candidate with a NaN value.
                        vb = try
                            f(proj.θ)
                        catch err
                            err isa _EXTREMIZE_CATCH || rethrow()
                            NaN
                        end
                        if vb isa Real && isfinite(vb)
                            projected = true
                            θfin = proj.θ
                            cfin = proj.fcn
                            v = Float64(vb)
                        end
                    end                            # else: keep the raw point —
                end                                # still within the gate
                naccepted += 1
                if best_p === nothing || sgn * v > sgn * best_v
                    best_v = v
                    best_p = θfin
                    best_c = cfin
                    winner = k
                end
            end
        catch err
            err isa _EXTREMIZE_CATCH || rethrow()
        end
        push!(records, (seed = k, converged = converged, accepted = accepted,
                        projected = projected, fcn = c, f_raw = v_raw, f = v,
                        nfcn = nf))
    end
    return (value = best_v, params = best_p, fcn = best_c, winner = winner,
            naccepted = naccepted, records = records)
end

# Shared validation + setup for extremize / profile_band: resolve the Δχ²
# threshold and FCN bound, and build the seed pool (best fit first).
function _extremize_setup(m::Minuit, cl, delta, lambda, accept_tol, seeds,
                          fname::String)
    m.fmin === nothing &&
        throw(ArgumentError("$fname: call `migrad!(m)` first"))
    m.npar >= 1 ||
        throw(ArgumentError("$fname: needs ≥ 1 free parameter " *
                            "(all parameters are fixed — nothing to vary)"))
    (isfinite(lambda) && lambda > 0) ||
        throw(ArgumentError("$fname: lambda must be finite and > 0"))
    (isfinite(accept_tol) && accept_tol >= 0) ||
        throw(ArgumentError("$fname: accept_tol must be finite and ≥ 0"))
    m.valid || @warn "$fname: the input fit is NOT valid — the Δχ² region is " *
                     "anchored at m.fval, which may not be the true minimum. " *
                     "Reach a valid minimum first (migrad!, find_deeper_minimum)."
    δ = delta === nothing ? delta_chisq(cl, 1) : Float64(delta)
    (isfinite(δ) && δ > 0) ||
        throw(ArgumentError("$fname: the Δχ² threshold must be finite and > 0"))
    up = Float64(m.up)
    (isfinite(up) && up > 0) ||
        throw(ArgumentError("$fname: m.up = $up — the error definition must " *
                            "be finite and > 0 (set m.errordef)"))
    bound = Float64(m.fval) + δ * up
    isfinite(bound) ||
        throw(ArgumentError("$fname: the FCN bound is not finite (m.fval = $(m.fval))"))
    pool = _seed_pool(m, seeds, fname)
    return δ, up, bound, pool
end

# ─────────────────────────────────────────────────────────────────────────────
# extremize — the Δχ²-region interval of a derived scalar
# ─────────────────────────────────────────────────────────────────────────────

"""
    extremize(m::Minuit, f; cl=1, seeds=nothing, kwargs...) -> ExtremizeResult

Profile (Δχ²-region) confidence interval of a **derived scalar** `f(θ)`:

    [min f(θ), max f(θ)]   over   { θ : FCN(θ) ≤ m.fval + delta_chisq(cl, 1)·m.up }

with **all free parameters varied simultaneously** (fixed parameters stay
pinned, limits are honoured). `f` receives the full EXTERNAL parameter
vector, in `m`'s parameter order. This is "MINOS for a function": for
`f(θ) = θ[i]` it reproduces the MINOS interval, and in the linear-Gaussian
limit it equals the projection-theorem result `f̂ ± √(delta·cᵀCc)` with the
full parameter correlations included. Returns an [`ExtremizeResult`](@ref);
`m` is not mutated.

# Why `ndof = 1` even though all parameters move

The threshold is `delta_chisq(cl, 1)` because the quoted statement is ONE
number: re-parametrize so `f` is itself a coordinate and this is a
single-parameter interval with the others profiled out (Wilks: 1 dof). Using
the joint `delta_chisq(cl, n_free)` here would over-cover (report errors up
to ~3× too wide for 9 parameters). For a genuinely **joint** statement, pass
an explicit `delta` — e.g. tracing a 2-D support function with
`delta = delta_chisq(cl, 2)`.

# Multiple seeds are load-bearing, not an optimization

Each direction is an exterior-penalty minimization
(`obj = ∓f + λ·max(0, (FCN−bound)/up)²`, solved as a short penalty-
continuation ladder of MIGRADs, `λ = 1 → 100 → lambda` warm-started) run
**from every seed**; a
single-seed run can stop at a *local* tangency and silently report a
too-narrow interval when the region has several low-χ² corridors (e.g. a
parameter against its limit feeding a monotone map). Default seeds = the
best fit only — pass everything you have that touches other corridors
(MCMC/ensemble members extreme in `f`, other `find_solution_modes`
representatives) via `seeds`, and **audit `r.diagnostics`**: per-seed
acceptance and `f` values, and which seed won each side (`winner_* == 0`
means no penalty fit beat the best-fit value itself).

# Keyword arguments
- `cl::Real = 1` — confidence level, [`delta_chisq`](@ref) convention:
  `cl ≥ 1` is **nσ** (1 → 68.27 %, 2 → 95.45 %), `0 < cl < 1` a
  **probability** (0.95 → 95 %). Threshold: `Δχ² = delta_chisq(cl, 1)`.
- `seeds = nothing` — extra start points: a vector of full external
  parameter vectors, the **rows** of a matrix, or a single vector. The best
  fit is always prepended as seed 1. Fixed coordinates are re-pinned and
  free ones clamped into limits before fitting.
- `lambda::Real = 1e4` — FINAL penalty stiffness; the continuation ladder
  is `unique([min(lambda,1), min(lambda,100), lambda])`. The raw optimum
  overshoots the boundary by `O(1/lambda)` and is then pulled back onto it.
- `accept_tol::Real = 0.05` — acceptance gate, in units of `up`: a penalty
  optimum with `FCN > bound + accept_tol·up` is discarded as not converged
  onto the region. The gate is applied to the **raw** penalty optimum,
  BEFORE the boundary pull-back — and that optimum always overshoots the
  boundary by `O(1/lambda)`, so `accept_tol` must stay above the overshoot
  (rule of thumb: `≳ 10/lambda` in `up` units). In particular
  `accept_tol = 0` is essentially never satisfiable: it rejects every
  candidate and silently collapses the result to the best-fit value —
  tighten `lambda` (stiffer penalty ⇒ smaller overshoot), not the gate.
  Exact feasibility of the reported endpoints is the pull-back's job, not
  the gate's; certify it via the `fcn_*` diagnostics.
- `delta::Union{Real,Nothing} = nothing` — explicit Δχ² threshold override
  (FCN units of `up`); when given, `cl` is ignored (and recorded as `NaN`).
- `rounds::Integer = 4` — maximum warm-started repeats of the penalty
  ladder per seed and direction. Inside the region the objective is just
  `∓f` (zero penalty, near-zero curvature for a near-linear `f`), so one
  chain can stall part-way through a long interior traverse — e.g. along a
  curved χ² valley — at a feasible but non-extremal point; repeats advance
  it. Iteration stops as soon as a round no longer improves `f` (a smooth
  problem therefore runs 2 rounds: one to converge, one to confirm).
- `strategy = m.strategy`, `maxfcn = nothing` — per-penalty-fit MIGRAD
  strategy and call budget.

# Cost & guarantees

2 directions × (number of distinct seeds) × (ladder stages, ≤ 3) × (rounds
actually run — ≥ 2 whenever `rounds ≥ 2`: one to converge plus one to
confirm; exactly 1 under `rounds = 1`, the [`profile_band`](@ref) default)
MIGRAD runs of the penalty objective (each evaluation calls the FCN and `f`
once), plus ≤ ~50 FCN calls per accepted endpoint for the boundary
pull-back.
`lo ≤ f(θ̂) ≤ hi` by construction; endpoints satisfy `FCN ≤ bound` exactly
when the pull-back applies (the typical case) and `FCN ≤ bound +
accept_tol·up` always. A side on which **no** penalty fit is accepted falls
back to the best-fit value with a warning — treat that side as failed and
investigate the diagnostics.

# Example

```julia
m = Minuit(chi2, x0; names = names, limits = limits)
migrad!(m)

# 68.3 % interval for a derived quantity (here: the model curve at x = 15)
r = extremize(m, θ -> θ[1] + θ[2] * 15.0)
r.lo, r.hi          # the interval
r.plo, r.phi        # parameter vectors realizing the endpoints
r.diagnostics       # per-seed audit: who converged, who was accepted, who won

# 95 % (2σ), seeding from ensemble members extreme in f
r2 = extremize(m, f; cl = 2, seeds = ens[sortperm(f.(eachrow(ens)))[[1, end]], :])
```

See also [`profile_band`](@ref) (pointwise band of a curve family),
[`minos!`](@ref) (the `f(θ) = θᵢ` special case), [`delta_chisq`](@ref),
and `docs/src/error_analysis.md` for the full decision guide.
"""
function extremize(m::Minuit, f; cl::Real = 1, seeds = nothing,
                   lambda::Real = 1e4, accept_tol::Real = 0.05,
                   delta::Union{Real,Nothing} = nothing,
                   rounds::Integer = 4,
                   strategy = m.strategy,
                   maxfcn::Union{Integer,Nothing} = nothing)
    rounds >= 1 || throw(ArgumentError("extremize: rounds must be ≥ 1"))
    δ, up, bound, pool = _extremize_setup(m, cl, delta, lambda, accept_tol,
                                          seeds, "extremize")
    that = pool[1]
    fhat = Float64(f(that))
    isfinite(fhat) ||
        throw(ArgumentError("extremize: f(best fit) is not finite"))

    λ = Float64(lambda)
    tolacc = Float64(accept_tol)
    lo = _extremize_dir(m, f, -1, bound, pool, fhat, that;
                        lambda = λ, accept_tol = tolacc, strategy = strategy,
                        maxfcn = maxfcn, include_best = true,
                        rounds = Int(rounds))
    hi = _extremize_dir(m, f, +1, bound, pool, fhat, that;
                        lambda = λ, accept_tol = tolacc, strategy = strategy,
                        maxfcn = maxfcn, include_best = true,
                        rounds = Int(rounds))
    for (side, r) in (("min", lo), ("max", hi))
        r.naccepted == 0 &&
            @warn "extremize: no penalty fit was accepted on the $side side — " *
                  "that endpoint is the best-fit value itself. Check the " *
                  "diagnostics records (convergence/acceptance per seed), " *
                  "and consider more seeds, a larger accept_tol, or maxfcn."
    end
    diag = (min = lo.records, max = hi.records,
            winner_min = lo.winner, winner_max = hi.winner,
            naccepted_min = lo.naccepted, naccepted_max = hi.naccepted,
            fcn_min = lo.fcn, fcn_max = hi.fcn)
    return ExtremizeResult(lo.value, hi.value, lo.params, hi.params, fhat,
                           bound, δ, delta === nothing ? Float64(cl) : NaN,
                           up, diag)
end

# ─────────────────────────────────────────────────────────────────────────────
# profile_band — pointwise profile envelope of a curve family
# ─────────────────────────────────────────────────────────────────────────────

"""
    profile_band(m::Minuit, f, xs; cl=1, seeds=nothing, warm=true, passes=2,
                 include_best=true, kwargs...) -> ProfileBand

Pointwise profile-likelihood **error band** of a curve family `f(x, θ)` on
the grid `xs` — `x` first, `θ` the full EXTERNAL parameter vector in `m`'s
parameter order: the package-wide `model(x, …)` convention, and the same
callback shape as [`quantile_band`](@ref). At each grid point,
[`extremize`](@ref)s `θ -> f(x, θ)` over the same fixed region
`{FCN ≤ m.fval + delta_chisq(cl, 1)·m.up}` (all free
parameters varied, limits/fixed honoured). Returns a [`ProfileBand`](@ref)
with the envelope (`lo`, `hi`), the per-point extremal parameter vectors
(`plo`, `phi`), the best-fit curve (`fbest`), the failure count and
per-point diagnostics. `m` is not mutated.

This is the standard pointwise (Δχ² ≤ 1 at `cl = 1`) construction for
figure bands: each x carries its own `cl` statement, and the band contains
the best-fit curve by construction — unlike posterior-quantile bands, which
can exclude it when a parameter sits on a limit.

!!! warning "Pointwise, not simultaneous"
    Each grid point is its own `ndof = 1` statement at confidence `cl`; the
    probability that the entire true curve lies inside the band everywhere
    at once is LOWER. That is the standard meaning of an error band — but
    say "pointwise" in the figure caption.

# Sweep strategy

The grid is swept `passes` times, alternating forward/reverse, keeping the
better (more extreme, still feasible) envelope per point. With
`warm = true` each side carries the previous point's extremal parameter
vector as an extra seed — extremal points move continuously along a smooth
curve family, so the warm seed is usually the best one. The full seed pool
(best fit + `seeds`) is **also** used at every point: corridor coverage
must not depend on the sweep having visited the right corridor earlier.
For an expensive FCN, the cost model is

    #MIGRAD ≈ length(xs) × 2 sides × passes × (pool + warm + incumbent) × stages × rounds

(`stages` ≤ 3, the penalty-continuation ladder; `rounds` defaults to 1
here) — trim `seeds`, lower `passes`, or set `maxfcn` (a per-MIGRAD budget)
to control it.

# Keyword arguments

`cl`, `seeds`, `lambda`, `accept_tol`, `delta`, `strategy`, `maxfcn` as in
[`extremize`](@ref), plus:
- `rounds::Integer = 1` — per-point penalty-ladder repeats (see
  [`extremize`](@ref)). The band default is 1 because the sweep itself
  iterates each point: the warm seed, the stored incumbent and the
  forward/reverse passes re-polish every point several times. Raise it if
  isolated points lag behind their neighbours.
- `warm::Bool = true` — warm-start each point from the neighbouring point's
  extremal parameters (per side, per pass direction).
- `passes::Integer = 2` — sweep count (1 = forward only; 2 adds the reverse
  sweep; more keep alternating). Two passes let a corridor discovered
  mid-grid propagate to BOTH sides.
- `include_best::Bool = true` — keep the best-fit value as a zero-cost
  feasible candidate per point, guaranteeing `lo ≤ fbest ≤ hi` and a finite
  band even where every penalty fit failed (such failures are still counted
  in `nfail`/diagnostics). With `false`, a fully-failed side is `NaN` —
  useful when benchmarking the optimizer itself.
- `verbose::Bool = false` — `@info` one line per pass.

# Example

```julia
m = Minuit(chi2, x0; names = names); migrad!(m)
mgrid = 4360.0:2.0:4520.0
band  = profile_band(m, (x, θ) -> moment_P2(x, θ), mgrid;
                     seeds = ens_extremes)         # ensemble extreme members
band.nfail == 0 || @warn "inspect band.diagnostics"
# plot: fill between band.lo and band.hi, line at band.fbest
```

See also [`extremize`](@ref), [`mnprofile`](@ref) (profile of a single
*parameter*), and `docs/src/error_analysis.md`.
"""
function profile_band(m::Minuit, f, xs::AbstractVector{<:Real}; cl::Real = 1,
                      seeds = nothing, warm::Bool = true, passes::Integer = 2,
                      include_best::Bool = true,
                      lambda::Real = 1e4, accept_tol::Real = 0.05,
                      delta::Union{Real,Nothing} = nothing,
                      rounds::Integer = 1,
                      strategy = m.strategy,
                      maxfcn::Union{Integer,Nothing} = nothing,
                      verbose::Bool = false)
    isempty(xs) && throw(ArgumentError("profile_band: xs is empty"))
    passes >= 1 || throw(ArgumentError("profile_band: passes must be ≥ 1"))
    rounds >= 1 || throw(ArgumentError("profile_band: rounds must be ≥ 1"))
    δ, up, bound, pool = _extremize_setup(m, cl, delta, lambda, accept_tol,
                                          seeds, "profile_band")
    that = pool[1]
    λ = Float64(lambda)
    tolacc = Float64(accept_tol)

    xv = collect(Float64, xs)
    n = length(xv)
    fbest = Vector{Float64}(undef, n)
    for i in 1:n
        fbest[i] = Float64(f(xv[i], that))
        isfinite(fbest[i]) || throw(ArgumentError(
            "profile_band: f(x = $(xv[i]), best fit) is not finite"))
    end

    lo = fill(NaN, n)
    hi = fill(NaN, n)
    plo = Vector{Union{Nothing,Vector{Float64}}}(nothing, n)
    phi = Vector{Union{Nothing,Vector{Float64}}}(nothing, n)
    fcn_lo = fill(NaN, n); fcn_hi = fill(NaN, n)       # FCN at stored endpoints
    acc_lo = zeros(Int, n); acc_hi = zeros(Int, n)     # accepted fits, cumulative
    fits_lo = zeros(Int, n); fits_hi = zeros(Int, n)   # attempted fits, cumulative
    nfail = 0

    for pass in 1:passes
        idxs = isodd(pass) ? (1:n) : reverse(1:n)
        wlo = nothing                  # warm carriers, per side per pass
        whi = nothing
        for i in idxs
            fi = let xi = xv[i]
                θ -> f(xi, θ)
            end
            # Per-point seed list: warm neighbour + current incumbent + the
            # full pool, duplicates skipped inside _extremize_dir. Seed
            # indices in the records refer to THIS list.
            slo = Vector{Vector{Float64}}()
            warm && wlo !== nothing && push!(slo, wlo)
            plo[i] === nothing || push!(slo, plo[i])
            append!(slo, pool)
            rlo = _extremize_dir(m, fi, -1, bound, slo, fbest[i], that;
                                 lambda = λ, accept_tol = tolacc,
                                 strategy = strategy, maxfcn = maxfcn,
                                 include_best = include_best,
                                 rounds = Int(rounds))
            shi = Vector{Vector{Float64}}()
            warm && whi !== nothing && push!(shi, whi)
            phi[i] === nothing || push!(shi, phi[i])
            append!(shi, pool)
            rhi = _extremize_dir(m, fi, +1, bound, shi, fbest[i], that;
                                 lambda = λ, accept_tol = tolacc,
                                 strategy = strategy, maxfcn = maxfcn,
                                 include_best = include_best,
                                 rounds = Int(rounds))

            fits_lo[i] += length(rlo.records); acc_lo[i] += rlo.naccepted
            fits_hi[i] += length(rhi.records); acc_hi[i] += rhi.naccepted
            rlo.naccepted == 0 && (nfail += 1)
            rhi.naccepted == 0 && (nfail += 1)

            if rlo.params !== nothing && (isnan(lo[i]) || rlo.value < lo[i])
                lo[i] = rlo.value
                plo[i] = rlo.params
                fcn_lo[i] = rlo.fcn
            end
            if rhi.params !== nothing && (isnan(hi[i]) || rhi.value > hi[i])
                hi[i] = rhi.value
                phi[i] = rhi.params
                fcn_hi[i] = rhi.fcn
            end
            if warm
                rlo.params === nothing || (wlo = rlo.params)
                rhi.params === nothing || (whi = rhi.params)
            end
        end
        verbose && @info "profile_band" pass nfail
    end

    diags = [(x = xv[i],
              failed_lo = acc_lo[i] == 0, failed_hi = acc_hi[i] == 0,
              accepted_lo = acc_lo[i], accepted_hi = acc_hi[i],
              nfits_lo = fits_lo[i], nfits_hi = fits_hi[i],
              fcn_lo = fcn_lo[i], fcn_hi = fcn_hi[i]) for i in 1:n]
    nfail == 0 ||
        @warn "profile_band: $nfail (point, side, pass) extremization group(s) " *
              "had no accepted penalty fit" *
              (include_best ? " (those edges fall back to the best-fit value)" :
                              " (those edges are NaN)") *
              " — inspect `.diagnostics`, and consider more seeds or a larger " *
              "accept_tol."
    return ProfileBand(xv, lo, hi, plo, phi, fbest, bound, δ,
                       delta === nothing ? Float64(cl) : NaN, up, nfail, diags)
end

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

_extremize_level(delta::Float64, cl::Float64) =
    isnan(cl) ? "Δχ² ≤ $(round(delta; sigdigits = 4)) (explicit)" :
                "cl = $(cl) → Δχ² ≤ $(round(delta; sigdigits = 4))"

function Base.show(io::IO, r::ExtremizeResult)
    print(io, "ExtremizeResult(f ∈ [", _fmt_num(r.lo), ", ", _fmt_num(r.hi), "])")
end

function Base.show(io::IO, ::MIME"text/plain", r::ExtremizeResult)
    d = r.diagnostics
    println(io, "extremize: f ∈ [", _fmt_num(r.lo), ", ", _fmt_num(r.hi),
            "]   (", _extremize_level(r.delta, r.cl), ", up = ", r.up, ")")
    println(io, "  best fit f̂ = ", _fmt_num(r.fbest), "   (inside by construction)")
    # winner 0 is benign when fits were accepted (the best fit IS the
    # extremum for that side) and a failure when none were.
    wname(w, nacc) = w != 0 ? "seed $w" :
                     nacc > 0 ? "best fit (genuinely extremal)" :
                                "best-fit fallback (side FAILED)"
    print(io, "  min: ", d.naccepted_min, "/", length(d.min),
          " accepted, winner ", wname(d.winner_min, d.naccepted_min),
          ";  max: ", d.naccepted_max, "/", length(d.max),
          " accepted, winner ", wname(d.winner_max, d.naccepted_max))
end

function Base.show(io::IO, b::ProfileBand)
    print(io, "ProfileBand(", length(b.x), " points)")
end

function Base.show(io::IO, ::MIME"text/plain", b::ProfileBand)
    nbad = count(d -> d.failed_lo || d.failed_hi, b.diagnostics)
    println(io, "profile_band: pointwise profile envelope, ", length(b.x),
            " points   (", _extremize_level(b.delta, b.cl), ", up = ", b.up, ")")
    print(io, "  group failures: ", b.nfail,
          "; points with a never-accepted side: ", nbad)
end
