# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Basin-hopping search for a DEEPER minimum on multi-basin objectives.
#
# MIGRAD (like every local optimiser) converges to whatever basin its start
# point drains into. On an ill-conditioned, multi-basin surface — IAM ππ being
# the worked example in `BenchmarkExamples/IAM_2Pformfactor` — that basin is
# often NOT the global one, and any error analysis done there is meaningless.
# `find_deeper_minimum` automates the "restart, find a deeper basin, adopt it,
# repeat" loop a user would otherwise run by hand. It is a HEURISTIC: it returns
# a deeper minimum than the start when its restarts find one, but cannot certify
# the result is global (hence the name — not `find_global_minimum`).
#
# Two complementary strategies share the name:
#   • parameter-perturbation  — jitter the current best and re-fit (any objective);
#   • data-resampling         — bootstrap-resample the data and re-fit (data fits;
#                               far stronger on hard multi-basin surfaces).
#
# EVERY fit goes through the high-level `Minuit` path, so a fit's PARAMETER LIMITS
# and FIXED parameters are honoured throughout — the search stays inside the same
# constrained parameter space as the user's fit. All overloads return a `Minuit`
# (MIGRAD + HESSE already run); check `.valid`.
#
# This is the SEARCH counterpart to `find_solution_modes` (which CLUSTERS an
# already-sampled set into distinct solutions).

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Clone a Minuit's FULL configuration — cost function, gradient + check_gradient,
# parameter NAMES, LIMITS, FIXED flags, errordef, strategy, tol, threading, prec,
# ndata — optionally overriding the start values / step sizes / strategy. This is
# the single place constraint-preservation is centralised: every (re-)fit in the
# search is built through here, so limits and fixed parameters always survive.
function _clone_minuit(m::Minuit; values = nothing, errors = nothing, strategy = m.strategy)
    grad = m.cfwg === nothing ? nothing : m.cfwg.g
    chkg = m.cfwg === nothing ? true    : m.cfwg.check_gradient
    # Minuit(fcn, m::Minuit) recovers names/limits/fixed/errors/up/tol/print_level/
    # threaded_gradient/verify_threading/prec from `m`; we override grad/check_gradient/strategy.
    mm = Minuit(m.fcn.f, m; grad = grad, check_gradient = chkg, strategy = strategy)
    mm.ndata = m.ndata                       # not recovered by the constructor
    values === nothing || (mm.values = collect(Float64, values))
    errors === nothing || (mm.errors = collect(Float64, errors))
    return mm
end

# A perturbed start point: jitter ONLY free parameters (fixed ones are left at
# their value), and clamp each jittered coordinate into its [lower, upper] bound.
function _perturb_point(best::Minuit, scale::Vector{Float64}, perturb::Real, rng)
    bx = collect(Float64, best.values)
    x  = copy(bx)
    @inbounds for i in eachindex(bx)
        is_fixed(best.params, i) && continue          # fixed: keep exactly
        xi = bx[i] + perturb * scale[i] * randn(rng)
        lo, hi = best.limits[i]                        # (NaN, NaN) when unbounded
        isnan(lo) || (xi = max(xi, lo))
        isnan(hi) || (xi = min(xi, hi))
        x[i] = xi
    end
    return x
end

# Project a discovery candidate onto `m`'s constraints: pin fixed parameters at the
# fit's value and clamp bounded ones into [lower, upper]. A user-supplied `refit`
# need not respect bounds; without this, find_solution_modes would score the
# ORIGINAL FCN at an out-of-bounds point and throw if the FCN is undefined there
# (the usual reason a parameter is bounded).
function _project_row(row::AbstractVector{<:Real}, m::Minuit)
    p = collect(Float64, row)
    @inbounds for i in eachindex(p)
        if is_fixed(m.params, i)
            p[i] = m.params.pars[i].value
        else
            lo, hi = m.limits[i]
            isnan(lo) || (p[i] = max(p[i], lo))
            isnan(hi) || (p[i] = min(p[i], hi))
        end
    end
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Perturbation strategy — core (m::Minuit) + convenience overloads
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_deeper_minimum(m::Minuit; kwargs...) -> Minuit

Parameter-perturbation basin-hopping search for a **deeper** minimum, starting
from the fit `m`. Each round draws `n_restarts` perturbed restarts around the
current best (each FREE coordinate jittered by `perturb · scaleᵢ · randn`, with
`scaleᵢ = max(|xᵢ|, |errorᵢ|, abs_floor)`), MIGRADs each, and **adopts any deeper
valid minimum**; it stops when a round finds no improvement or after `max_rounds`.

Every restart is a clone of `m` (via [`Minuit`](@ref)), so **`m`'s parameter
limits and fixed parameters are honoured** — fixed parameters are never jittered
and stay pinned at their value, and jittered coordinates are clamped into their
bounds. The search therefore stays inside the same constrained parameter space as
your fit. Returns the deepest `Minuit` found (MIGRAD + HESSE already run); check
its `.valid` property. `m` itself is not mutated.

# Keyword arguments
- `n_restarts::Integer = 24` — perturbed restarts per round (≥ 1).
- `perturb::Real = 1.0` — exploration radius, as a multiple of each parameter's
  scale. **The key knob on a hard surface.**
- `abs_floor::Real = 0.0` — absolute lower bound on each coordinate's jitter scale
  (raise it for a parameter sitting near 0 with a tiny step).
- `max_rounds::Integer = 50` — **safety backstop**, not the normal stop. The search
  stops when a round finds no deeper basin (convergence); `max_rounds` only bounds a
  pathological non-converging run. A `@warn` fires if it is hit while still improving.
- `strategy = m.strategy` — MIGRAD strategy for every (re-)fit.
- `maxfcn::Union{Integer,Nothing} = nothing` — per-fit MIGRAD call budget.
- `min_improvement::Real = 1e-3` — χ² drop required to adopt a restart.
- `seed::Union{Integer,Nothing} = nothing` — RNG seed for reproducible restarts.
- `verbose::Bool = false` — log the best χ² per round.

!!! note "Not a global-optimum guarantee (hence the name)"
    Basin-hopping finds *a* deeper basin when its restarts land in one; it cannot
    prove the result is global. Raise `n_restarts`/`perturb`/`max_rounds` and
    cross-check from independent seeds.

See also [`find_solution_modes`](@ref) and the data-resampling overload
[`find_deeper_minimum(m, refit, data)`](@ref).
"""
function find_deeper_minimum(m::Minuit;
        n_restarts::Integer = 24, perturb::Real = 1.0, abs_floor::Real = 0.0,
        max_rounds::Integer = 50, strategy = m.strategy,
        maxfcn::Union{Integer,Nothing} = nothing, min_improvement::Real = 1e-3,
        seed::Union{Integer,Nothing} = nothing, verbose::Bool = false)
    n_restarts >= 1 || throw(ArgumentError("find_deeper_minimum: n_restarts must be ≥ 1"))
    max_rounds >= 1 || throw(ArgumentError("find_deeper_minimum: max_rounds must be ≥ 1"))
    perturb > 0     || throw(ArgumentError("find_deeper_minimum: perturb must be > 0"))
    min_improvement >= 0 || throw(ArgumentError("find_deeper_minimum: min_improvement must be ≥ 0"))
    abs_floor >= 0  || throw(ArgumentError("find_deeper_minimum: abs_floor must be ≥ 0"))
    m.npar >= 1     || throw(ArgumentError("find_deeper_minimum: needs ≥ 1 free parameter " *
                                           "(all parameters are fixed — nothing to search)"))
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    n = length(m.values)

    # Work on a fitted clone — never mutate the caller's `m`.
    best = _clone_minuit(m; strategy = strategy)
    migrad!(best; maxfcn = maxfcn); hesse(best)

    converged = false
    for round in 1:max_rounds
        bx    = collect(Float64, best.values)
        berr  = collect(Float64, best.errors)
        scale = [is_fixed(best.params, i) ? 0.0 :
                 max(abs(bx[i]), abs(berr[i]), abs_floor, eps()) for i in 1:n]
        improved = false
        for _ in 1:n_restarts
            x = _perturb_point(best, scale, perturb, rng)
            # A wild jitter can push a constrained FCN into a throwing region
            # (log of a negative, a singular matrix, …); skip that restart.
            cand = _clone_minuit(best; values = x)
            ok = try
                migrad!(cand; maxfcn = maxfcn); true
            catch err
                err isa Union{DomainError,BoundsError,SingularException,ArgumentError,DivideError} || rethrow()
                false
            end
            ok || continue
            if cand.valid && isfinite(cand.fval) && cand.fval < best.fval - min_improvement
                hesse(cand)
                best = cand
                improved = true
            end
        end
        verbose && @info "find_deeper_minimum (perturbation)" round χ² = best.fval improved
        if !improved
            converged = true
            break
        end
    end
    converged || @warn "find_deeper_minimum (perturbation): reached max_rounds=$max_rounds while the " *
                       "last round was STILL improving — the search has not converged. The stopping " *
                       "criterion is convergence (a round with no deeper basin), not the round cap; " *
                       "raise `max_rounds` (and/or `perturb`/`n_restarts`) to let it finish."
    return best
end

"""
    find_deeper_minimum(cf, x0, errors; limits=nothing, fixed=nothing, names=nothing, kwargs...) -> Minuit

Parameter-perturbation search from a cost function / callable `cf` and start
`x0`/`errors`. Builds a [`Minuit`](@ref) with the given `limits`, `fixed` flags
and `names` (same meaning as the `Minuit` constructor) and delegates to
[`find_deeper_minimum(m::Minuit)`](@ref) — so bounds and fixed parameters are
honoured. `cf` may be an [`AbstractCostFunction`](@ref) (its gradient, if any, is
carried) or a bare callable (`up` sets the error definition).
"""
function find_deeper_minimum(cf::AbstractCostFunction, x0::AbstractVector, errors::AbstractVector;
        limits = nothing, fixed = nothing, names = nothing, name = nothing,
        strategy = Strategy(1), kwargs...)
    grad = cf isa CostFunctionWithGradient ? cf.g : nothing
    chkg = cf isa CostFunctionWithGradient ? cf.check_gradient : true
    m = Minuit(cf.f, collect(Float64, x0); error = collect(Float64, errors), up = cf.up,
               limits = limits, fixed = fixed, name = (names === nothing ? name : names),
               grad = grad, check_gradient = chkg, strategy = strategy)
    find_deeper_minimum(m; strategy = strategy, kwargs...)
end

find_deeper_minimum(f, x0::AbstractVector, errors::AbstractVector; up::Real = 1.0, kwargs...) =
    find_deeper_minimum(CostFunction(f, up), x0, errors; kwargs...)

# ─────────────────────────────────────────────────────────────────────────────
# Data-resampling strategy — core (m, refit, data) + overloads
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_deeper_minimum(m::Minuit, refit, data; kwargs...) -> Minuit

Data-resampling basin-hopping search for a **deeper** minimum on a multi-basin
data-fitting objective. Each round bootstrap-resamples `data` and re-fits each
resample with `refit` (those drift toward whichever basin best explains that
subset); the candidates are clustered with `find_solution_modes(...; refine=true)`
and **re-evaluated on the ORIGINAL objective** (so the χ² comparison is honest);
the deepest valid new basin is adopted, and the loop repeats.

**Constraints are honoured.** The re-fit/refinement runs through `Minuit(m.fcn.f, m)`
(which carries `m`'s limits + fixed flags and pins fixed parameters), and the
adoption rebuild clones the same configuration — so a fixed parameter stays fixed
and a bounded one stays in bounds throughout. Returns the deepest `Minuit` (MIGRAD
+ HESSE already run); check `.valid`. `m` is not mutated.

# Arguments
- `m::Minuit` — a converged fit. Supplies the cost function, current best as the
  warm start, the HESSE covariance for whitening, and the constraint structure.
- `refit` — `refit(subdata, start::Vector{Float64}) -> Vector{Float64}` (a `NaN`-
  filled vector of the same length ⇒ invalid, dropped). Accepts functors. To keep
  the discovery itself constrained, `refit` should apply the same fix/limits;
  any fixed parameter is in any case re-pinned during refinement.
- `data` — the full dataset, supporting `data[idx]` bootstrap indexing.

# Keyword arguments
- `n_discovery::Integer = 20` — bootstrap resamples per round (≥ 2; keep ≥ 10 for
  stable clustering).
- `max_rounds::Integer = 50` — **safety backstop**, not the normal stop. The search
  stops when a round adopts no deeper basin (convergence); a `@warn` fires if the cap
  is hit while still improving.
- `strategy = m.strategy` — MIGRAD strategy for the adoption re-fit.
- `min_improvement::Real = 1e-3` — χ² drop required to adopt a basin.
- `parallel::Union{Bool,Nothing} = nothing` — `true` threads the discovery loop
  (only if `refit` is thread-safe) AND is passed to `find_solution_modes`;
  `nothing`/`false` runs discovery serially.
- `seed::Union{Integer,Nothing} = nothing` — RNG seed (indices are pre-generated
  sequentially, so results are deterministic regardless of `parallel`).
- `verbose::Bool = false` — log χ² improvement per round.

# Suitability check
If the first round finds no deeper basin, a `@warn` is emitted and a fitted clone
of `m` (the same minimum, since `m` is never mutated) is returned — the surface
may be single-basin here, or bootstrap coverage was
insufficient (raise `n_discovery`); for parameter-space search try the
perturbation overload `find_deeper_minimum(m; perturb=…)`.
"""
function find_deeper_minimum(m::Minuit, refit, data;
        n_discovery::Integer = 20, max_rounds::Integer = 50, strategy = m.strategy,
        min_improvement::Real = 1e-3, maxfcn::Union{Integer,Nothing} = nothing,
        parallel::Union{Bool,Nothing} = nothing,
        seed::Union{Integer,Nothing} = nothing, verbose::Bool = false)
    n_discovery >= 2 || throw(ArgumentError("find_deeper_minimum: n_discovery must be ≥ 2"))
    max_rounds >= 1  || throw(ArgumentError("find_deeper_minimum: max_rounds must be ≥ 1"))
    min_improvement >= 0 || throw(ArgumentError("find_deeper_minimum: min_improvement must be ≥ 0"))
    n  = length(data)
    n >= 2 || throw(ArgumentError("find_deeper_minimum: data must have ≥ 2 elements"))
    m.npar >= 1 || throw(ArgumentError("find_deeper_minimum: needs ≥ 1 free parameter " *
                                       "(all parameters are fixed — nothing to search)"))
    np = length(m.values)
    rng    = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    do_par = parallel === true

    # Work on a fitted clone — never mutate the caller's `m`.
    cur = _clone_minuit(m; strategy = strategy)
    migrad!(cur; maxfcn = maxfcn); hesse(cur)

    converged = false
    for iter in 1:max_rounds
        p_star = collect(Float64, cur.values)

        # Pre-generate bootstrap indices sequentially → deterministic under parallel.
        all_idx = [rand(rng, 1:n, n) for _ in 1:n_discovery]

        rows = Vector{Union{Vector{Float64},Nothing}}(undef, n_discovery)
        if do_par
            Threads.@threads :static for k in 1:n_discovery
                r = refit(data[all_idx[k]], p_star)
                rows[k] = (length(r) == np && all(isfinite, r)) ? r : nothing
            end
        else
            for k in 1:n_discovery
                r = refit(data[all_idx[k]], p_star)
                rows[k] = (length(r) == np && all(isfinite, r)) ? r : nothing
            end
        end
        valid_rows = filter(!isnothing, rows)

        if length(valid_rows) < 2
            @warn "find_deeper_minimum (resampling): only $(length(valid_rows)) valid " *
                  "resample(s) in round $iter — FCN may throw or fail on most bootstrap " *
                  "subsets. Check your `refit` function."
            converged = true   # explicit stop (already warned); not a silent cap-hit
            break
        end

        # Project every candidate onto cur's constraints BEFORE clustering, so
        # find_solution_modes never evaluates the FCN out of bounds / off a fixed value.
        disc = Matrix{Float64}(undef, length(valid_rows), np)
        for (i, r) in enumerate(valid_rows)
            disc[i, :] .= _project_row(r, cur)
        end
        # find_solution_modes refines each mode through Minuit(m.fcn.f, m) — limits
        # + fixed flags preserved, fixed parameters pinned — on the ORIGINAL data.
        md = find_solution_modes(disc, cur; refine = true, parallel = parallel)

        if iter == 1 && !any(x.new_min for x in md)
            @warn "find_deeper_minimum (resampling): round 1 — " *
                  "$(length(valid_rows)) valid resample(s) formed $(length(md)) mode(s), " *
                  "none deeper than current best (χ² = $(round(cur.fval; digits=3))). " *
                  "No deeper basin found via data-resampling at this start. " *
                  "The surface may be single-basin here, or bootstrap coverage was insufficient " *
                  "(try a larger `n_discovery`). " *
                  "For parameter-space search try: `find_deeper_minimum(m; perturb=…, n_restarts=…)`."
            return cur
        end

        deeper = [x for x in md if x.new_min && x.refined_valid]
        if isempty(deeper)
            verbose && @info "find_deeper_minimum (resampling)" iter msg = "no deeper basin → stable"
            converged = true
            break
        end
        bn          = deeper[argmin(x.refined_fval for x in deeper)]
        fval_before = cur.fval
        fval_after  = bn.refined_fval
        if fval_before - fval_after < min_improvement
            converged = true       # deepest new basin is within tol of current → stable
            break
        end

        verbose && @info "find_deeper_minimum (resampling)" iter χ²_before = fval_before χ²_after = fval_after

        # Adopt: clone cur's constraints, set the new basin's values + step sizes.
        adopt_errs = isempty(bn.refined_errors) ? collect(Float64, cur.errors) :
                                                   collect(Float64, bn.refined_errors)
        cur = _clone_minuit(cur; values = bn.refined_values, errors = adopt_errs, strategy = strategy)
        migrad!(cur; maxfcn = maxfcn); hesse(cur)
    end
    converged || @warn "find_deeper_minimum (resampling): reached max_rounds=$max_rounds while the " *
                       "last round still adopted a deeper basin — the search has not converged. The " *
                       "stopping criterion is convergence (a round finding no deeper basin), not the " *
                       "round cap; raise `max_rounds` to let it finish."
    return cur
end

"""
    find_deeper_minimum(cf, x0, errors, refit, data; limits=nothing, fixed=nothing, names=nothing, kwargs...) -> Minuit

Data-resampling search from a cost function / callable. Builds a [`Minuit`](@ref)
with the given `limits`/`fixed`/`names`, fits it, and delegates to the
`(m::Minuit, refit, data)` overload (constraints honoured). When you already hold
a converged `Minuit`, prefer passing it directly.
"""
function find_deeper_minimum(cf::AbstractCostFunction, x0::AbstractVector,
                             errors::AbstractVector, refit, data;
                             limits = nothing, fixed = nothing, names = nothing, name = nothing,
                             strategy = Strategy(1), kwargs...)
    grad = cf isa CostFunctionWithGradient ? cf.g : nothing
    chkg = cf isa CostFunctionWithGradient ? cf.check_gradient : true
    m = Minuit(cf.f, collect(Float64, x0); error = collect(Float64, errors), up = cf.up,
               limits = limits, fixed = fixed, name = (names === nothing ? name : names),
               grad = grad, check_gradient = chkg, strategy = strategy)
    # The (m, refit, data) core fits a clone of `m` itself — no need to fit here.
    find_deeper_minimum(m, refit, data; strategy = strategy, kwargs...)
end

find_deeper_minimum(f, x0::AbstractVector, errors::AbstractVector, refit, data;
                    up::Real = 1.0, kwargs...) =
    find_deeper_minimum(CostFunction(f, up), x0, errors, refit, data; kwargs...)

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch disambiguators
# ─────────────────────────────────────────────────────────────────────────────
# A 3-arg call `(Minuit, AbstractVector, AbstractVector)` is otherwise ambiguous
# (perturbation plain-callable wrapper vs. the m::Minuit core), and a 5-arg
# `(Minuit, AbstractVector, AbstractVector, refit, data)` mixes the two API styles.
# These sentinels throw a helpful ArgumentError instead.
find_deeper_minimum(::Minuit, ::AbstractVector, ::AbstractVector; kwargs...) =
    throw(ArgumentError(
        "find_deeper_minimum: ambiguous 3-arg call (Minuit, AbstractVector, AbstractVector). " *
        "For parameter-perturbation from a Minuit use the 1-arg form `find_deeper_minimum(m; perturb=…)`. " *
        "For data-resampling pass a callable `refit` and a data collection: `find_deeper_minimum(m, refit, data)`."))

find_deeper_minimum(::Minuit, ::AbstractVector, ::AbstractVector, refit, data; kwargs...) =
    throw(ArgumentError(
        "find_deeper_minimum: invalid call (Minuit, AbstractVector, AbstractVector, refit, data) " *
        "mixes the two API styles. For data-resampling from a converged Minuit, drop x0/errors: " *
        "`find_deeper_minimum(m, refit, data)`. To start fresh, pass a cost function/callable " *
        "(not a Minuit): `find_deeper_minimum(cf, x0, errors, refit, data)`."))

# ─────────────────────────────────────────────────────────────────────────────
# Deprecated 0.3.1 name. Basin-hopping cannot certify a global minimum, so the
# honest name is `find_deeper_minimum`; this warning-emitting alias keeps any
# v0.3.1 code working.
function find_global_minimum(args...; kwargs...)
    Base.depwarn("`find_global_minimum` is deprecated; use `find_deeper_minimum` " *
                 "(basin-hopping cannot guarantee a *global* minimum).", :find_global_minimum)
    return find_deeper_minimum(args...; kwargs...)
end
