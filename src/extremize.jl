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
- `mode::Symbol` — `:full` (the multi-seed penalty extremization) or
  `:directional` (the fast linear-direction crossing; see [`extremize`](@ref)).
  The `diagnostics` schema differs between the two modes.
- `diagnostics::NamedTuple` — **for `mode = :full`:** `(min, max, winner_min,
  winner_max,
  naccepted_min, naccepted_max, fcn_min, fcn_max, directional_floor)`.
  `min`/`max` are per-seed
  record vectors, one row per penalty fit, with fields `seed` (index into
  the seed pool; seed 1 is always the best fit), `converged` (MIGRAD
  validity), `accepted` (passed the `bound + accept_tol·up` gate),
  `projected` (was pulled back onto the boundary), `fcn` (FCN at the raw
  penalty optimum), `f_raw`/`f` (`f` before/after the pull-back), `nfcn`, and
  `f_nonfinite` (count of probes where `f` threw or returned non-finite and was
  steered around — see the `f`-failure contract under [`extremize`](@ref)).
  `winner_min`/`winner_max` give the seed index whose candidate won each
  side — `0` means no penalty fit beat the best-fit value itself, which has
  two very different readings, disambiguated by `naccepted_*`: with
  `naccepted_* > 0` the best fit is genuinely extremal for that side (e.g.
  `f`'s unconstrained optimum sits at θ̂ — healthy); with `naccepted_* == 0`
  every penalty fit was rejected and that side FAILED (a warning fired).
  `fcn_min`/`fcn_max` are the FCN values at `plo`/`phi`: compare with
  `bound` to certify each endpoint's feasibility directly. Inspect these to
  audit seed coverage (the multi-corridor failure mode described in the
  docstring). `directional_floor` is `(lo::Bool, hi::Bool)` — whether the
  directional floor/ceiling (not the penalty) supplied that endpoint.
  **For `mode = :directional`:** `(grad, dir, gCg, alpha_lin,
  alpha_lo, alpha_hi, fcn_lo, fcn_hi, nfcn, nf)` — the `f`-gradient and search
  direction `C·∇f` at θ̂, the linear step `√(delta/gᵀCg)`, the two true-FCN
  boundary crossings `alpha_lo`/`alpha_hi` (`±` along `dir`) with their FCN
  values (`≤ bound`), and the FCN/`f` call counts.
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
    mode::Symbol
    diagnostics::D
end

# Backward-compatible positional constructor: pre-0.5.3 `ExtremizeResult` had no
# `mode` field. Code that built one positionally with the old 10-arg arity keeps
# working and is tagged `:full`. (`ExtremizeResult` is exported; the field was
# inserted before `diagnostics`.)
ExtremizeResult(lo, hi, plo, phi, fbest, bound, delta, cl, up,
                diagnostics::NamedTuple) =
    ExtremizeResult(lo, hi, plo, phi, fbest, bound, delta, cl, up, :full, diagnostics)

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
- `mode::Symbol` — `:full` (per-point multi-seed penalty extremization) or
  `:directional` (per-point fast `C·∇f` crossing; see [`profile_band`](@ref)).
  The per-point `diagnostics` schema differs between the two modes.
- `nfail::Int` — number of failed (point, side) groups: for `:full`, (point,
  side, pass) extremization groups in which NO penalty fit passed the
  acceptance gate; for `:directional`, (point, side) crossings where `f` was
  non-finite or the direction was un-computable. The best-fit fallback still
  keeps the band finite when `include_best = true`; `0` on a healthy sweep.
- `diagnostics` — per-point NamedTuples. **`:full`:** `(x, failed_lo,
  failed_hi, accepted_lo, accepted_hi, nfits_lo, nfits_hi, fcn_lo, fcn_hi)` —
  cumulative accepted / attempted penalty-fit counts over all passes,
  `failed_*` flags for a side that NEVER had an accepted fit, and the FCN
  values at the stored extremal points. **`:directional`:** `(x, failed_lo,
  failed_hi, fcn_lo, fcn_hi, gCg, nfcn, nf)` — the per-point feasibility flags,
  boundary FCNs, `∇fᵀC∇f`, and FCN/`f` call counts.
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
    mode::Symbol
    nfail::Int
    diagnostics::Vector{D}
end

# Backward-compatible positional constructor (pre-0.5.3 ProfileBand had no
# `mode` field); old 11-arg positional construction keeps working as `:full`.
ProfileBand(x, lo, hi, plo, phi, fbest, bound, delta, cl, up, nfail::Integer,
            diagnostics::Vector) =
    ProfileBand(x, lo, hi, plo, phi, fbest, bound, delta, cl, up, :full,
                nfail, diagnostics)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Exception set treated as "this start point / probe point is outside the
# FCN's domain" rather than a bug — same guard as find_deeper_minimum's
# restart loop (a wild seed can push a constrained FCN into a throwing
# region: log of a negative, a singular matrix, …).
const _EXTREMIZE_CATCH =
    Union{DomainError,BoundsError,SingularException,ArgumentError,DivideError}

# Internal sentinel raised DELIBERATELY by the directional core when the search
# direction is genuinely un-computable at a point — `∇fᵀC∇f ≤ 0` (f flat along
# the covariance) or a non-finite `f` at a gradient probe. It is NOT a member of
# `_EXTREMIZE_CATCH`, so `profile_band(mode=:directional)` can catch ONLY this
# (falling back to the best fit for that point) while a genuinely buggy user
# `grad_f` — a `BoundsError`/`MethodError`/etc. — still propagates, exactly as
# it does in `extremize`. `extremize` re-surfaces it as a clear `ArgumentError`.
struct _DirectionUncomputable <: Exception
    msg::String
end

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

# True iff θ respects every parameter limit (within a tiny tolerance). Fixed
# parameters are pinned by the directional construction, so only free, bounded
# coordinates can stray — the directional ray ignores limits, so an endpoint may
# leave the box and is then NOT a member of the constrained Δχ² region.
function _within_limits(m::Minuit, θ::Vector{Float64})
    @inbounds for i in eachindex(θ)
        p = m.params.pars[i]
        is_fixed(p) && continue
        lo, hi = p.lower, p.upper                  # NaN = absent
        isnan(lo) || θ[i] >= lo - 1e-9 * max(1.0, abs(lo)) || return false
        isnan(hi) || θ[i] <= hi + 1e-9 * max(1.0, abs(hi)) || return false
    end
    return true
end

# The exterior-penalty objective for one direction (sgn = +1 maximizes f,
# sgn = −1 minimizes it). The constraint excess is normalized by `up` so
# `lambda` means the same thing for a χ² (up = 1) and a −lnL (up = 0.5) fit.
#
# The objective is TOTAL — it never throws and never returns ±Inf or NaN:
# - an FCN throw / non-finite value (a probe far outside the FCN's domain)
#   becomes a finite high plateau via the excess cap, not an Inf cliff. An
#   Inf reaching MIGRAD's numerical gradient turns the curvature matrix
#   into NaNs and aborts the whole fit from inside LinearAlgebra (observed
#   on a soft λ = 1 stage wandering up the Rosenbrock valley); a finite
#   plateau is simply never accepted by the line search.
# - a non-finite/throwing `f` ALSO maps to the SAME finite plateau (NOT NaN —
#   field report E4): a NaN returned to MIGRAD poisons the numerical-gradient
#   curvature exactly like the Inf-FCN case, and it tempts users with a
#   failure-prone `f` (e.g. an off-basin pole search) into returning a sentinel
#   like `0.0`, which SILENTLY BIASES the endpoint toward the centre. A finite
#   plateau makes the probe simply unattractive — MIGRAD steers around the
#   non-finite-`f` region — so the contract is: **`f` may throw OR return
#   non-finite at infeasible θ; both are safe (no NaN into MIGRAD, no `0.0`-
#   sentinel CENTERING bias)**. A genuinely non-finite-`f` region may still
#   legitimately NARROW the interval (the optimizer cannot reach past it) —
#   that is safe and correct, not a bias. Each such probe is tallied via
#   `nonfinite` (surfaced as `f_nonfinite` in the per-seed diagnostics records).
function _penalty_obj(fcnraw, f, sgn::Int, bound::Float64, up::Float64,
                      lambda::Float64, nonfinite::Base.RefValue{Int})
    return let fcnraw = fcnraw, f = f, s = Float64(sgn), bound = bound,
               up = up, lambda = lambda, nonfinite = nonfinite
        θ -> begin
            c = _fcn_or_inf(fcnraw, θ)              # throw / non-finite → Inf
            excess = min((c - bound) / up, 1e8)
            excess >= 1e8 && return lambda * 1e16   # flat plateau; skip f
            fv = try
                f(θ)
            catch err
                err isa _EXTREMIZE_CATCH || rethrow()
                nonfinite[] += 1
                return lambda * 1e16               # finite plateau, never NaN
            end
            if !(fv isa Real && isfinite(fv))
                nonfinite[] += 1
                return lambda * 1e16               # finite plateau, never NaN
            end
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
                        strategy, maxfcn, include_best::Bool, rounds::Int,
                        iterate::Int = 5, on_unit = nothing,
                        side::Symbol = :unknown)
    fcnraw = m.fcn.f
    up = Float64(m.up)
    ladder = _penalty_ladder(lambda)

    best_v = include_best ? fhat : NaN
    best_p = include_best ? copy(that) : nothing
    best_c = include_best ? Float64(m.fval) : NaN   # FCN at the winning point
    winner = 0
    naccepted = 0

    records = NamedTuple{(:seed, :converged, :accepted, :projected,
                          :fcn, :f_raw, :f, :nfcn, :f_nonfinite),
                         Tuple{Int,Bool,Bool,Bool,Float64,Float64,Float64,Int,Int}}[]
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
        # Tally of non-finite/throwing `f` probes the penalty objective steered
        # around (field report E4 / P4); shared across this seed's stages+rounds.
        nonfinite = Ref(0)
        # `on_unit` records are BUFFERED here and fired only AFTER the outer
        # try/catch below — the catch swallows `_EXTREMIZE_CATCH` (DomainError,
        # ArgumentError, …), so firing the user callback inside it would
        # silently eat a throwing checkpoint callback. Buffering keeps per-unit
        # granularity while letting callback exceptions propagate.
        pending = on_unit === nothing ? nothing : NamedTuple[]
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
            for ri in 1:rounds
                for (si, lam) in enumerate(ladder)
                    obj = _penalty_obj(fcnraw, f, sgn, bound, up, lam, nonfinite)
                    # The MIGRAD subfit and its domain-edge failures are caught
                    # here; the `on_unit` hook is fired OUTSIDE this try (below)
                    # so a throwing user callback is NEVER swallowed by the
                    # `_EXTREMIZE_CATCH` net (which includes ArgumentError /
                    # DomainError — a broken checkpoint must surface).
                    unit_mmk = nothing
                    try
                        mmk = Minuit(obj, m; up = 1.0)
                        mmk.values = cur
                        Logging.with_logger(Logging.NullLogger()) do
                            migrad!(mmk; strategy = strategy, maxfcn = maxfcn,
                                    iterate = iterate)
                        end
                        cur = collect(Float64, mmk.values)
                        nf += mmk.nfcn
                        mm = mmk
                        unit_mmk = mmk
                    catch err
                        err isa _EXTREMIZE_CATCH || rethrow()
                    end
                    # P5: buffer one record per completed penalty-MIGRAD unit
                    # (the resumable-journal granularity). The raw FCN at `cur`
                    # is only evaluated when a hook is attached, so the default
                    # path pays nothing. `_fcn_or_inf` never throws. The actual
                    # `on_unit` call happens after the outer try (see below).
                    if pending !== nothing && unit_mmk !== nothing
                        push!(pending, (side = side, seed = k, round = ri,
                                        stage = si, cur = copy(cur),
                                        nfcn = unit_mmk.nfcn,
                                        fcn = _fcn_or_inf(fcnraw, cur),
                                        valid = unit_mmk.valid))
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
        # Fire the progress hook OUTSIDE the catch above: a throwing checkpoint
        # callback must surface, not be swallowed by the domain-error net.
        if pending !== nothing
            for u in pending
                on_unit(u)
            end
        end
        push!(records, (seed = k, converged = converged, accepted = accepted,
                        projected = projected, fcn = c, f_raw = v_raw, f = v,
                        nfcn = nf, f_nonfinite = nonfinite[]))
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
    # Upper bound keeps the `lambda · 1e16` penalty plateau finite (the
    # f-failure / FCN-domain plateau in `_penalty_obj`) — `lambda ≤ 1e100`
    # ⇒ plateau ≤ 1e116 ≪ floatmax; far above any useful stiffness (default 1e4).
    (isfinite(lambda) && lambda > 0) ||
        throw(ArgumentError("$fname: lambda must be finite and > 0"))
    lambda <= 1e100 ||
        throw(ArgumentError("$fname: lambda must be ≤ 1e100 (the penalty plateau " *
                            "would overflow); the default 1e4 is ample"))
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
# Directional (fast linear-direction) mode — field report P2.
#
# The exact extremum of the Δχ² region for a derived scalar lies, in the
# linear-Gaussian limit, along d = C·∇f (the Lagrange/projection condition the
# file header quotes): max f = f̂ + √(Δχ²·∇fᵀC∇f). Instead of the multi-seed
# penalty machinery, walk that one ray and secant/bisect the TRUE FCN to its
# `bound` crossing on each side, then report the TRUE f there. Cost ≈ n_free
# (gradient) + ~2×(bracket+bisection) FCN calls + 2 f calls — ~50× cheaper than
# `:full` on the common near-linear case, and it uses the true FCN/f at the
# crossing so first-order direction error is the only approximation. It does
# NOT chase non-linear corridors or honour limits that bind before the
# crossing — use `:full` (optionally seeded) when the two disagree.
# ─────────────────────────────────────────────────────────────────────────────

# Forward-difference ∇f at θ̂ over the FREE parameters (fixed slots → 0). Only
# sets the search DIRECTION and linear step; the true-FCN root + true-f at the
# crossing correct any inaccuracy. n_free extra f-evals (tallied in `nf`).
function _grad_forward(m::Minuit, f, that::Vector{Float64}, fhat::Float64,
                       nf::Base.RefValue{Int})
    n = n_pars(m.params)
    g = zeros(Float64, n)
    @inbounds for i in 1:n
        is_fixed(m.params.pars[i]) && continue
        ei = m.errors[i]
        scale = (isfinite(ei) && ei > 0) ? ei : max(1.0, abs(that[i]))
        h = max(1e-10, 1e-6 * scale)
        xp = copy(that); xp[i] += h
        # An f-DOMAIN failure at the probe — a throw in `_EXTREMIZE_CATCH` OR a
        # non-finite return — is the documented "f may fail at infeasible θ"
        # case: map BOTH to the sentinel (→ ArgumentError in `extremize`,
        # per-point fallback in `profile_band`). A non-domain throw (a genuine
        # bug, e.g. MethodError) still propagates. Symmetric with the crossing
        # `evalf` guard and the `:full` path's f-contract.
        nf[] += 1                              # count the ATTEMPT (incl. a
        # throwing/non-finite probe), consistent with the crossing `evalf` and
        # the `nfcn` counter, which both tally attempts not just completions.
        fp = try
            f(xp)
        catch err
            err isa _EXTREMIZE_CATCH || rethrow()
            throw(_DirectionUncomputable(
                "f threw a domain error at a gradient probe of parameter $i — " *
                "supply `grad_f`, or use mode=:full."))
        end
        (fp isa Real && isfinite(fp)) || throw(_DirectionUncomputable(
            "f is non-finite at a gradient probe of parameter $i — supply " *
            "`grad_f`, or use mode=:full."))
        g[i] = (Float64(fp) - fhat) / h
    end
    return g
end

# Smallest α ≥ 0 with FCN(θ̂ + α·dir) crossing `bound`, returned on the FEASIBLE
# side (FCN ≤ bound) — the directional analogue of `_project_to_bound`. Linear
# guess `α_lin`, geometric bracket expansion, then a SECANT (regula-falsi) root
# with a bisection safeguard so it converges in a handful of FCN calls (the
# pure-bisection ladder cost ~40/side; secant ~5–8). Converges on the FCN
# residual `bound − FCN ≤ ftol·up` on the feasible side — α to machine ε is
# unnecessary (the endpoint `f` is evaluated there anyway). A throwing /
# non-finite FCN (outside the domain) counts as infeasible (c > bound).
function _root_on_ray(fcnraw, that::Vector{Float64}, dir::Vector{Float64},
                      bound::Float64, α_lin::Float64, up::Float64,
                      nf::Base.RefValue{Int};
                      ftol::Float64 = 1e-4, maxiter::Int = 40, expand_max::Int = 80)
    evalc(α) = (nf[] += 1; _fcn_or_inf(fcnraw, that .+ α .* dir))
    α_lo, c_lo = 0.0, evalc(0.0)                    # c(0) = m.fval < bound
    α_hi = α_lin > 0 ? α_lin : 1.0
    c_hi = evalc(α_hi)
    nexp = 0
    while c_hi <= bound && nexp < expand_max
        α_lo, c_lo = α_hi, c_hi
        α_hi *= 1.6
        c_hi = evalc(α_hi)
        nexp += 1
    end
    # Never left the region (flat/unbounded ray within budget): best feasible α.
    c_hi <= bound && return (alpha = α_lo, fcn = c_lo)
    for _ in 1:maxiter
        # Converged: feasible side within ftol·up below the boundary.
        (bound - c_lo) <= ftol * up && break
        h_lo = c_lo - bound
        h_hi = c_hi - bound
        # Secant/regula-falsi step from the bracket; bisect if it would leave
        # the bracket or hug an endpoint (the classic regula-falsi stall).
        αs = α_hi - h_hi * (α_hi - α_lo) / (h_hi - h_lo)
        if !(α_lo < αs < α_hi) || min(αs - α_lo, α_hi - αs) < 1e-3 * (α_hi - α_lo)
            αs = 0.5 * (α_lo + α_hi)
        end
        cs = evalc(αs)
        if cs <= bound
            α_lo, c_lo = αs, cs
        else
            α_hi, c_hi = αs, cs
        end
        (α_hi - α_lo) <= 1e-12 * max(1.0, α_hi) && break
    end
    return (alpha = α_lo, fcn = c_lo)               # feasible by construction
end

# One warning per call (not per grid point) when directional is used with a
# bounded fit — the ray ignores limits, so `plo`/`phi` may exit the limited set.
function _directional_limits_warn(m::Minuit, fname::String)
    if any(!is_fixed(p) && has_limits(p) for p in m.params.pars)
        @warn "$fname(mode=:directional): the fit has bounded free parameters; the " *
              "directional ray ignores limits, so a boundary crossing (and the " *
              "returned plo/phi) may lie outside them. Use mode=:full if a limit binds."
    end
end

# Core of `mode = :directional` for ONE scalar `f`, SHARED by `extremize` and
# `profile_band` so the algorithm is implemented (and reviewed) exactly once:
# g = ∇f(θ̂) (free slots; fixed → 0), d = C·g, α_lin = √(δ/gᵀCg), then
# secant/bisect the TRUE FCN to `bound` on each ±d ray and take the TRUE `f` at
# the crossings; the interval is min/max over {f̂, f₊, f₋} (contains the best
# fit by construction). Returns `(lo, hi, plo, phi, diag, f_failed_lo,
# f_failed_hi)`. THROWS `ArgumentError` (∈ `_EXTREMIZE_CATCH`) when the
# direction is un-computable — `grad_f` wrong length, a non-finite gradient
# probe, or ∇fᵀC∇f ≤ 0 — so `extremize` propagates it while `profile_band`
# catches it to flag that point. Does NOT warn; `C`/`nf_fcn`/`nf_f` are passed
# in so a band computes `C` once and tallies costs across points.
function _directional_interval(m::Minuit, f, grad_f, δ::Float64, up::Float64,
                               bound::Float64, that::Vector{Float64},
                               fhat::Float64, C, nf_fcn::Base.RefValue{Int},
                               nf_f::Base.RefValue{Int})
    n = n_pars(m.params)
    fcnraw = m.fcn.f
    g = if grad_f === nothing
        _grad_forward(m, f, that, fhat, nf_f)
    else
        gg = collect(Float64, grad_f(that))
        length(gg) == n || throw(ArgumentError(
            "extremize(mode=:directional): grad_f returned length $(length(gg)), " *
            "expected $n (full external gradient)"))
        @inbounds for i in 1:n
            is_fixed(m.params.pars[i]) && (gg[i] = 0.0)
        end
        # A non-finite SUPPLIED gradient (in a free slot) is a user grad_f BUG,
        # not an f-domain degeneracy: raise a plain ArgumentError so it
        # PROPAGATES in both `extremize` and `profile_band` (NOT the sentinel,
        # which `profile_band` would swallow as a per-point fallback). Fixed
        # slots were just zeroed, so a NaN there is harmless and ignored.
        all(isfinite, gg) || throw(ArgumentError(
            "extremize(mode=:directional): grad_f returned a non-finite gradient " *
            "$gg — fix grad_f (a non-finite gradient is a user error, not an " *
            "f-domain degeneracy)."))
        gg
    end
    Cg = C * g                                  # = C·∇f; fixed slots stay 0
    gCg = 0.0
    @inbounds for i in 1:n
        gCg += g[i] * Cg[i]
    end
    gCg > 0 || throw(_DirectionUncomputable(
        "∇fᵀC∇f = $gCg is not positive — f is flat along the covariance (or C is " *
        "degenerate); use mode=:full."))
    α_lin = sqrt(δ / gCg)
    rp = _root_on_ray(fcnraw, that, Cg, bound, α_lin, up, nf_fcn)   # +dir: f increases
    rm = _root_on_ray(fcnraw, that, -Cg, bound, α_lin, up, nf_fcn)  # −dir: f decreases
    θp = that .+ rp.alpha .* Cg
    θm = that .- rm.alpha .* Cg
    # `f` at each crossing — guarded exactly like the `:full` path's f-failure
    # contract (a throw here must NOT crash `:directional`). A throwing OR
    # non-finite `f` maps to NaN, is dropped from the candidate set, and the
    # per-side `f_failed_*` flag makes the resulting collapse to the best-fit
    # value detectable (it is otherwise silent).
    evalf(θ) = (nf_f[] += 1; v = try
                    f(θ)
                catch err
                    err isa _EXTREMIZE_CATCH || rethrow()
                    NaN
                end;
                (v isa Real && isfinite(v)) ? Float64(v) : NaN)
    fp = evalf(θp)
    fm = evalf(θm)
    # contains-best-fit by construction: min/max over {f̂, f₊, f₋} (f̂ finite).
    cands = Tuple{Float64,Vector{Float64}}[(fhat, copy(that))]
    isnan(fp) || push!(cands, (fp, θp))
    isnan(fm) || push!(cands, (fm, θm))
    hitem = argmax(c -> c[1], cands)
    loitem = argmin(c -> c[1], cands)
    diag = (grad = g, dir = Cg, gCg = gCg, alpha_lin = α_lin,
            alpha_lo = -rm.alpha, alpha_hi = rp.alpha,
            fcn_lo = rm.fcn, fcn_hi = rp.fcn,
            f_failed_lo = isnan(fm), f_failed_hi = isnan(fp),
            nfcn = nf_fcn[], nf = nf_f[])
    # Distinct `copy`s so `plo` and `phi` are never the SAME object even when
    # the best fit wins both sides (a degenerate `lo == hi == f̂`) — downstream
    # mutation of one endpoint must not corrupt the other.
    return (lo = loitem[1], hi = hitem[1],
            plo = copy(loitem[2]), phi = copy(hitem[2]),
            diag = diag, f_failed_lo = isnan(fm), f_failed_hi = isnan(fp))
end

function _extremize_directional(m::Minuit, f, grad_f, δ::Float64, up::Float64,
                                bound::Float64, that::Vector{Float64},
                                fhat::Float64, cl::Real, delta)
    C = m.covariance
    C === nothing && throw(ArgumentError(
        "extremize(mode=:directional): no covariance is available — run `migrad!(m)` " *
        "(and `hesse!(m)` for a reliable C) so `m.covariance` is set."))
    _directional_limits_warn(m, "extremize")
    nf_fcn = Ref(0)
    nf_f = Ref(0)
    # An un-computable direction surfaces to the user as a clear `ArgumentError`
    # (the sentinel is internal); a genuinely buggy `f`/`grad_f` propagates.
    r = try
        _directional_interval(m, f, grad_f, δ, up, bound, that, fhat, C, nf_fcn, nf_f)
    catch err
        err isa _DirectionUncomputable &&
            throw(ArgumentError("extremize(mode=:directional): " * err.msg))
        rethrow()
    end
    # A non-finite `f` at a crossing collapses that side onto the best fit —
    # warn (mirroring the `:full` `naccepted == 0` warning) so a degenerate
    # `lo == hi == fbest` is never mistaken for a genuinely tight interval.
    if r.f_failed_lo || r.f_failed_hi
        @warn "extremize(mode=:directional): f is non-finite at the " *
              (r.f_failed_lo && r.f_failed_hi ? "lower AND upper" :
               r.f_failed_lo ? "lower" : "upper") *
              " boundary crossing — that side falls back to the best-fit value " *
              "(the interval is degenerate there). Use mode=:full, or check f " *
              "near the Δχ² boundary."
    end
    return ExtremizeResult(r.lo, r.hi, r.plo, r.phi, fhat,
                           bound, δ, delta === nothing ? Float64(cl) : NaN, up,
                           :directional, r.diag)
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
**from every seed**; a single-seed run can stop short of the extremum and
report a too-narrow interval in two distinct ways: (i) on a strongly
correlated / **ill-conditioned** region the best-fit-anchored penalty stalls
on the flat axis at a feasible but NON-extremal boundary point, and (ii) when
the region splits into several disconnected low-χ² **corridors** the penalty
cannot cross the barrier between them.

To cure (i) at no user effort, the result is, by default, floored/ceiled by the
**directional (HESSE-ellipse) endpoints** `θ̂ ± √δ·C∇f/σ_f` (the `:directional`
construction below): they are feasible and exact in the linear-Gaussian limit,
so the reported interval is **never narrower than the directional one**, no
matter what the penalty did (including a degenerate `lambda` that accepts
nothing). It costs ONE extra directional probe (≈ `n_free` gradient + a dozen
FCN + 2 `f` evaluations), NOT extra penalty seeds. Disable with
`directional_floor = false`. If the direction is un-computable (`∇fᵀC∇f ≤ 0`,
degenerate `C`, non-finite `f` at the probe) the floor is silently skipped. The
floor does NOT cure case (ii): disconnected corridors are unreachable from a
straight ray, so **pass everything you have that touches other corridors**
(MCMC/ensemble members extreme in `f`, other `find_solution_modes`
representatives) via `seeds`. **Audit `r.diagnostics`**: per-seed acceptance and
`f` values, which seed won each side (`winner_* == 0` means no penalty fit beat
the best-fit value), and `directional_floor.lo/.hi` (whether the floor supplied
that endpoint).

# Cheap mode for expensive FCNs, and the `f`-failure contract

For an FCN/`f` that costs seconds, the default `:full` algorithm (multi-seed ×
ladder × rounds MIGRADs) can be hours per call — use `mode = :directional`
(below) for the common near-linear case (≈ `n_free + ~15` paired calls, ~50×
cheaper), and on the `:full` path set `rounds = 1`, a small `maxfcn`, and
`strategy = 0`. **`f` may throw OR return a non-finite value at infeasible θ —
both are safe**: such probes become a finite high plateau the optimizer steers
around (never `NaN` into MIGRAD), tallied as `f_nonfinite` in the diagnostics.
A genuinely non-finite-`f` region may legitimately NARROW the interval (the
optimizer cannot reach past it) — that is safe, not a bias. Do NOT instead
return a sentinel like `0.0` from a failing `f`: that *centers* the endpoint
(a silent bias). In `mode = :directional`, a non-finite `f` at a boundary
crossing collapses that side onto the best fit, with a warning and a
`f_failed_lo`/`f_failed_hi` diagnostic flag.

# Keyword arguments
- `cl::Real = 1` — confidence level, [`delta_chisq`](@ref) convention:
  `cl ≥ 1` is **nσ** (1 → 68.27 %, 2 → 95.45 %), `0 < cl < 1` a
  **probability** (0.95 → 95 %). Threshold: `Δχ² = delta_chisq(cl, 1)`.
- `mode::Symbol = :full` — `:full` is the multi-seed penalty extremization
  (handles non-linear / multi-corridor regions). `:directional` is the fast
  linear-direction crossing: it forms `d = C·∇f` at θ̂ (the Lagrange/projection
  direction this docstring's formula uses), secant/bisects the **true** FCN to
  its `bound` crossing on each side, and reports the **true** `f` there. Cost ≈
  `n_free` (gradient) + ~2× a dozen FCN calls + 2 `f` calls — exact in the
  linear-Gaussian limit, and `r.mode === :directional` flags it so it is not
  mistaken for the full profile. It ignores `seeds`, does not chase non-linear
  corridors, and does not honour parameter limits that bind before the crossing
  — `r.plo`/`r.phi` may then lie OUTSIDE the limits (it warns when free
  parameters are bounded). The recommended workflow is `:directional` first,
  then `:full` (optionally seeded) only if you suspect non-linearity, a binding
  limit, or the two disagree. Requires `m.covariance` (run `migrad!`/`hesse!`
  first).
- `grad_f = nothing` — optional `θ -> ∇f(θ)` (full external length) for
  `mode = :directional`; replaces the `n_free`-call forward-difference
  gradient. Ignored by `mode = :full`.
- `seeds = nothing` — extra start points: a vector of full external
  parameter vectors, the **rows** of a matrix, or a single vector. The best
  fit is always prepended as seed 1. Fixed coordinates are re-pinned and
  free ones clamped into limits before fitting.
- `directional_floor::Bool = true` (`mode = :full` only) — floor/ceil the
  result by the directional (HESSE-ellipse) endpoints so it is never narrower
  than the linear-Gaussian interval (see "Multiple seeds are load-bearing").
  Uses a numerical gradient (ignores `grad_f`, like the rest of `:full`); set
  `false` to skip the extra directional probe.
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
  strategy and call budget. For an expensive FCN, `strategy = 0` (no inner
  HESSE) and a modest `maxfcn` cut the cost sharply.
- `iterate::Integer = 5` — the per-penalty-MIGRAD retry budget forwarded to
  [`migrad!`](@ref) (its `_robust_low_level_fit` parity). The default helps a
  stiff-penalty stage that stalls invalid recover; for an expensive FCN set
  `iterate = 1` to forbid retries (the cheapest setting — combine with
  `rounds = 1`).
- `on_unit = nothing` — a callback `u -> …` fired once per completed penalty-
  MIGRAD unit with `u = (side, seed, round, stage, cur, nfcn, fcn, valid)` —
  the granularity of one MIGRAD (≈ one expensive step). Use it for live
  progress or to checkpoint partial work externally (a kill then only loses
  the in-flight unit). Attaching it adds one FCN evaluation per unit (to report
  `fcn`); the default `nothing` pays nothing. Ignored by `mode = :directional`.

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
                   mode::Symbol = :full, grad_f = nothing,
                   directional_floor::Bool = true,
                   lambda::Real = 1e4, accept_tol::Real = 0.05,
                   delta::Union{Real,Nothing} = nothing,
                   rounds::Integer = 4,
                   strategy = m.strategy,
                   maxfcn::Union{Integer,Nothing} = nothing,
                   iterate::Integer = 5, on_unit = nothing)
    mode === :full || mode === :directional ||
        throw(ArgumentError("extremize: mode must be :full or :directional, got :$mode"))
    rounds >= 1 || throw(ArgumentError("extremize: rounds must be ≥ 1"))
    iterate >= 1 || throw(ArgumentError("extremize: iterate must be ≥ 1"))
    δ, up, bound, pool = _extremize_setup(m, cl, delta, lambda, accept_tol,
                                          seeds, "extremize")
    that = pool[1]
    fhat = Float64(f(that))
    isfinite(fhat) ||
        throw(ArgumentError("extremize: f(best fit) is not finite"))

    if mode === :directional
        seeds === nothing ||
            @warn "extremize(mode=:directional): `seeds` are ignored (the " *
                  "directional mode walks the single C·∇f ray, not a seed pool)."
        return _extremize_directional(m, f, grad_f, δ, up, bound, that, fhat, cl, delta)
    end

    λ = Float64(lambda)
    tolacc = Float64(accept_tol)
    lo = _extremize_dir(m, f, -1, bound, pool, fhat, that;
                        lambda = λ, accept_tol = tolacc, strategy = strategy,
                        maxfcn = maxfcn, include_best = true,
                        rounds = Int(rounds), iterate = Int(iterate),
                        on_unit = on_unit, side = :lower)
    hi = _extremize_dir(m, f, +1, bound, pool, fhat, that;
                        lambda = λ, accept_tol = tolacc, strategy = strategy,
                        maxfcn = maxfcn, include_best = true,
                        rounds = Int(rounds), iterate = Int(iterate),
                        on_unit = on_unit, side = :upper)
    # Certified directional floor/ceiling (default; `directional_floor = false`
    # to disable). On a strongly correlated / ill-conditioned Δχ² region the
    # best-fit-anchored penalty can stall at a feasible but NON-extremal boundary
    # point — silently under-covering (observed 30–60 % of the true interval at
    # κ(C) ≳ 1e10, the endpoints sitting exactly on Δχ²=1 yet far from the
    # tangency) — or, with a degenerate `lambda`, accept nothing and fall back to
    # f̂. The directional endpoints θ̂ ± √δ·C∇f/σ_f are FEASIBLE (the ray is
    # root-found to the FCN bound) and exact in the linear-Gaussian limit, so we
    # fold them in as GUARANTEED candidates: the reported interval is then never
    # narrower than the directional one, whatever the penalty did. This is ONE
    # extra directional probe (≈ n_free gradient + ~a dozen FCN + 2 f calls),
    # NOT extra penalty seeds. Best-effort: an un-computable direction
    # (∇fᵀC∇f ≤ 0, degenerate C) or an f-domain probe failure simply skips the
    # floor, reproducing the prior penalty-only result (a swallowed error can
    # only COST the floor, never corrupt the answer). A directional endpoint that
    # leaves the parameter limits (the ray ignores limits) is not a region member
    # and is not folded.
    lo_v, lo_p, lo_c, lo_fl = lo.value, lo.params, lo.fcn, false
    hi_v, hi_p, hi_c, hi_fl = hi.value, hi.params, hi.fcn, false
    if directional_floor && m.covariance !== nothing
        fcnraw = m.fcn.f
        rdir = try
            _directional_interval(m, f, nothing, δ, up, bound, that, fhat,
                                  m.covariance, Ref(0), Ref(0))
        catch err
            (err isa _DirectionUncomputable || err isa _EXTREMIZE_CATCH) || rethrow()
            nothing
        end
        if rdir !== nothing
            # `rdir.plo`/`.phi` are the endpoints SELECTED BY f-VALUE (argmin/max
            # over the two rays plus the best fit); for a nonlinear f the lower
            # endpoint may be the PLUS ray (and vice versa). The per-ray
            # `f_failed_*`/`fcn_*` in `rdir.diag` are RAY-labeled, so they must
            # NOT be used for the selected endpoint — recompute the FCN at the
            # actual point and gate on feasibility there. (`rdir.lo == fhat` when
            # the directional collapsed to the best fit ⇒ never beats `lo_v`.)
            c_lo = _fcn_or_inf(fcnraw, rdir.plo)
            if isfinite(rdir.lo) && rdir.lo < lo_v &&
               c_lo <= bound + 1e-9 * up && _within_limits(m, rdir.plo)
                lo_v, lo_p, lo_c, lo_fl = rdir.lo, rdir.plo, c_lo, true
            end
            c_hi = _fcn_or_inf(fcnraw, rdir.phi)
            if isfinite(rdir.hi) && rdir.hi > hi_v &&
               c_hi <= bound + 1e-9 * up && _within_limits(m, rdir.phi)
                hi_v, hi_p, hi_c, hi_fl = rdir.hi, rdir.phi, c_hi, true
            end
        end
    end

    for (side, r, fl) in (("min", lo, lo_fl), ("max", hi, hi_fl))
        r.naccepted == 0 &&
            @warn "extremize: no penalty fit was accepted on the $side side — " *
                  (fl ? "using the directional (linear-Gaussian) floor for that " *
                        "endpoint. The penalty mechanism still found nothing; "
                      : "that endpoint is the best-fit value itself. ") *
                  "Check the diagnostics records, and consider more seeds, a " *
                  "larger accept_tol, or maxfcn."
    end
    diag = (min = lo.records, max = hi.records,
            winner_min = lo.winner, winner_max = hi.winner,
            naccepted_min = lo.naccepted, naccepted_max = hi.naccepted,
            fcn_min = lo_c, fcn_max = hi_c,
            directional_floor = (lo = lo_fl, hi = hi_fl))
    return ExtremizeResult(lo_v, hi_v, lo_p, hi_p, fhat,
                           bound, δ, delta === nothing ? Float64(cl) : NaN,
                           up, :full, diag)
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
- `mode::Symbol = :full` — `:full` runs the multi-seed penalty extremization at
  every grid point; `:directional` runs the fast `C·∇f` crossing at every
  point (see [`extremize`](@ref)'s `mode`). For an expensive FCN and a
  near-linear curve family, `:directional` makes the whole sweep
  orders-of-magnitude cheaper (≈ `npoints × (n_free + ~15)` evaluations) and is
  exact in the linear-Gaussian limit; it ignores `seeds`/`warm`/`passes`/
  `rounds`/`iterate`/`on_unit`, does not honour limits, and flags a point that
  fails (non-finite `f` or un-computable direction) in `nfail`/`diagnostics`
  with a best-fit fallback. `b.mode` records which was used. Requires
  `m.covariance`.
- `grad_f = nothing` — optional `(x, θ) -> ∇_θ f(x, θ)` (full external length)
  for `mode = :directional`; replaces the per-point forward-difference
  gradient. Ignored by `mode = :full`.
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
- `iterate::Integer = 5` — per-penalty-MIGRAD retry budget (see
  [`extremize`](@ref)); `iterate = 1` is the cheapest for an expensive FCN.
- `on_unit = nothing` — per-MIGRAD-unit progress callback (see
  [`extremize`](@ref)); the record additionally carries `(x, point, pass)` for
  the grid point. Enables external checkpointing of a long band sweep.
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
                      seeds = nothing, mode::Symbol = :full, grad_f = nothing,
                      warm::Bool = true, passes::Integer = 2,
                      include_best::Bool = true,
                      lambda::Real = 1e4, accept_tol::Real = 0.05,
                      delta::Union{Real,Nothing} = nothing,
                      rounds::Integer = 1,
                      strategy = m.strategy,
                      maxfcn::Union{Integer,Nothing} = nothing,
                      iterate::Integer = 5, on_unit = nothing,
                      verbose::Bool = false)
    mode === :full || mode === :directional ||
        throw(ArgumentError("profile_band: mode must be :full or :directional, got :$mode"))
    isempty(xs) && throw(ArgumentError("profile_band: xs is empty"))
    passes >= 1 || throw(ArgumentError("profile_band: passes must be ≥ 1"))
    rounds >= 1 || throw(ArgumentError("profile_band: rounds must be ≥ 1"))
    iterate >= 1 || throw(ArgumentError("profile_band: iterate must be ≥ 1"))
    δ, up, bound, pool = _extremize_setup(m, cl, delta, lambda, accept_tol,
                                          seeds, "profile_band")
    that = pool[1]
    λ = Float64(lambda)
    tolacc = Float64(accept_tol)

    xv = collect(Float64, xs)
    n = length(xv)
    cl_rec = delta === nothing ? Float64(cl) : NaN
    fbest = Vector{Float64}(undef, n)
    for i in 1:n
        fbest[i] = Float64(f(xv[i], that))
        isfinite(fbest[i]) || throw(ArgumentError(
            "profile_band: f(x = $(xv[i]), best fit) is not finite"))
    end

    if mode === :directional
        return _profile_band_directional(m, f, grad_f, xv, fbest, δ, up, bound,
                                         that, cl_rec, seeds)
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
            # Per-point progress hook: inject the grid point `x`/index/pass into
            # the unit record (P5); `nothing` when no hook is attached.
            ou = on_unit === nothing ? nothing : let xi = xv[i], ii = i, pp = pass
                u -> on_unit(merge(u, (x = xi, point = ii, pass = pp)))
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
                                 rounds = Int(rounds), iterate = Int(iterate),
                                 on_unit = ou, side = :lower)
            shi = Vector{Vector{Float64}}()
            warm && whi !== nothing && push!(shi, whi)
            phi[i] === nothing || push!(shi, phi[i])
            append!(shi, pool)
            rhi = _extremize_dir(m, fi, +1, bound, shi, fbest[i], that;
                                 lambda = λ, accept_tol = tolacc,
                                 strategy = strategy, maxfcn = maxfcn,
                                 include_best = include_best,
                                 rounds = Int(rounds), iterate = Int(iterate),
                                 on_unit = ou, side = :upper)

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
    return ProfileBand(xv, lo, hi, plo, phi, fbest, bound, δ, cl_rec, up,
                       :full, nfail, diags)
end

# Directional band: per grid point, the shared `_directional_interval` core on
# θ -> f(x_i, θ). C and the limits-warning are computed/emitted ONCE for the
# whole sweep; an un-computable direction or non-finite-`f` crossing flags that
# point (nfail) and falls back to the best-fit value rather than aborting. The
# gradient is recomputed per point (its direction varies with x); warm-starting
# it across adjacent points is a possible future micro-opt. `seeds`, `warm`,
# `passes`, `rounds`, `iterate`, `on_unit` do not apply to the directional path.
function _profile_band_directional(m::Minuit, f, grad_f, xv::Vector{Float64},
                                   fbest::Vector{Float64}, δ::Float64,
                                   up::Float64, bound::Float64,
                                   that::Vector{Float64}, cl_rec::Float64, seeds)
    seeds === nothing ||
        @warn "profile_band(mode=:directional): `seeds` are ignored (the directional " *
              "mode walks the single C·∇f ray per point, not a seed pool)."
    C = m.covariance
    C === nothing && throw(ArgumentError(
        "profile_band(mode=:directional): no covariance is available — run " *
        "`migrad!(m)` (and `hesse!(m)` for a reliable C) so `m.covariance` is set."))
    _directional_limits_warn(m, "profile_band")
    n = length(xv)
    lo = fill(NaN, n)
    hi = fill(NaN, n)
    plo = Vector{Union{Nothing,Vector{Float64}}}(nothing, n)
    phi = Vector{Union{Nothing,Vector{Float64}}}(nothing, n)
    fcn_lo = fill(NaN, n); fcn_hi = fill(NaN, n)
    gcg = fill(NaN, n)
    nfcn = zeros(Int, n); nfev = zeros(Int, n)
    failed_lo = falses(n); failed_hi = falses(n)
    nfail = 0
    for i in 1:n
        fi = let xi = xv[i]
            θ -> f(xi, θ)
        end
        gfi = grad_f === nothing ? nothing : let xi = xv[i]
            θ -> grad_f(xi, θ)
        end
        nf_fcn = Ref(0); nf_f = Ref(0)
        ok = true
        local r
        try
            r = _directional_interval(m, fi, gfi, δ, up, bound, that, fbest[i],
                                      C, nf_fcn, nf_f)
        catch err
            # ONLY the deliberate "direction un-computable here" sentinel is a
            # per-point fallback; a genuinely buggy f/grad_f (BoundsError,
            # MethodError, a non-domain ArgumentError, …) PROPAGATES — same as
            # `extremize(mode=:directional)` — so a real bug is never silently
            # collapsed into the band.
            err isa _DirectionUncomputable || rethrow()
            ok = false
        end
        nfcn[i] = nf_fcn[]; nfev[i] = nf_f[]
        if ok
            lo[i] = r.lo; hi[i] = r.hi
            plo[i] = r.plo; phi[i] = r.phi
            fcn_lo[i] = r.diag.fcn_lo; fcn_hi[i] = r.diag.fcn_hi
            gcg[i] = r.diag.gCg
            failed_lo[i] = r.f_failed_lo; failed_hi[i] = r.f_failed_hi
        else
            # degenerate point: fall back to the best fit on both sides
            # (distinct copies so plo[i] !== phi[i]).
            lo[i] = hi[i] = fbest[i]
            plo[i] = copy(that); phi[i] = copy(that)
            failed_lo[i] = failed_hi[i] = true
        end
        failed_lo[i] && (nfail += 1)
        failed_hi[i] && (nfail += 1)
    end
    diags = [(x = xv[i], failed_lo = failed_lo[i], failed_hi = failed_hi[i],
              fcn_lo = fcn_lo[i], fcn_hi = fcn_hi[i], gCg = gcg[i],
              nfcn = nfcn[i], nf = nfev[i]) for i in 1:n]
    nfail == 0 ||
        @warn "profile_band(mode=:directional): $nfail (point, side) crossing(s) had " *
              "a non-finite f or an un-computable direction (those edges fall back to " *
              "the best-fit value) — inspect `.diagnostics`, or use mode=:full there."
    return ProfileBand(xv, lo, hi, plo, phi, fbest, bound, δ, cl_rec, up,
                       :directional, nfail, diags)
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
    println(io, "extremize [", r.mode, "]: f ∈ [", _fmt_num(r.lo), ", ",
            _fmt_num(r.hi), "]   (", _extremize_level(r.delta, r.cl),
            ", up = ", r.up, ")")
    println(io, "  best fit f̂ = ", _fmt_num(r.fbest), "   (inside by construction)")
    if r.mode === :directional
        print(io, "  direction C·∇f, ∇fᵀC∇f = ", _fmt_num(d.gCg),
              ";  crossings α ∈ [", _fmt_num(d.alpha_lo), ", ",
              _fmt_num(d.alpha_hi), "]   (", d.nfcn, " FCN, ", d.nf, " f calls)")
        return
    end
    # :full — winner 0 is benign when fits were accepted (the best fit IS the
    # extremum for that side) and a failure when none were; the directional
    # floor (when it supplied the endpoint) overrides both labels.
    wname(w, nacc, fl) = fl ? "directional floor" :
                         w != 0 ? "seed $w" :
                         nacc > 0 ? "best fit (genuinely extremal)" :
                                    "best-fit fallback (side FAILED)"
    df = get(d, :directional_floor, (lo = false, hi = false))
    print(io, "  min: ", d.naccepted_min, "/", length(d.min),
          " accepted, winner ", wname(d.winner_min, d.naccepted_min, df.lo),
          ";  max: ", d.naccepted_max, "/", length(d.max),
          " accepted, winner ", wname(d.winner_max, d.naccepted_max, df.hi))
end

function Base.show(io::IO, b::ProfileBand)
    print(io, "ProfileBand(", length(b.x), " points)")
end

function Base.show(io::IO, ::MIME"text/plain", b::ProfileBand)
    nbad = count(d -> d.failed_lo || d.failed_hi, b.diagnostics)
    println(io, "profile_band [", b.mode, "]: pointwise profile envelope, ",
            length(b.x), " points   (", _extremize_level(b.delta, b.cl),
            ", up = ", b.up, ")")
    print(io, "  group failures: ", b.nfail,
          "; points with a failed side: ", nbad)
end
