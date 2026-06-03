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
# This is the SEARCH counterpart to `find_solution_modes` (which CLUSTERS an
# already-sampled set into distinct solutions). Use it to escape a local basin
# before the usual error analysis (HESSE / MINOS / get_contours_samples) at the
# minimum it returns.

"""
    find_deeper_minimum(fcn, x0, errors; kwargs...) -> FunctionMinimum

Basin-hopping search for a **deeper** minimum on a multi-basin objective — it
escapes the local basin a single MIGRAD lands in. It does **not** certify the
result is global (see the note below). Starting from a MIGRAD
fit at `x0`, repeatedly draw `n_restarts` perturbed restarts around the current
best (each coordinate jittered by `perturb · scaleᵢ · randn`, with
`scaleᵢ = max(|xᵢ|, |errorᵢ|, abs_floor)`), MIGRAD each, and **adopt any deeper
valid minimum**. A round that finds no improvement means the search has
converged; otherwise it stops after `max_rounds`. Returns the deepest
[`FunctionMinimum`](@ref) found.

`fcn` may be a plain callable (wrapped in `CostFunction(fcn, up)`) or an
[`AbstractCostFunction`](@ref). `x0`/`errors` are the usual MIGRAD start point
and step sizes.

!!! warning "Unbounded only — and check validity"
    This routine fits through the **unbounded** MIGRAD path and **ignores
    parameter limits**: fold any bounds into `fcn` (a penalty, or a smooth
    reparameterisation) before calling. The returned `FunctionMinimum` can be
    invalid (e.g. if every restart failed) — always check `is_valid(result)`
    before using it.

# Keyword arguments

- `n_restarts::Integer = 24` — perturbed restarts per round (must be ≥ 1).
- `perturb::Real = 1.0` — exploration radius, as a multiple of each parameter's
  scale. Larger ⇒ jumps farther (more likely to escape a basin, at one full
  re-fit per restart). **The key knob to tune on a hard surface.**
- `abs_floor::Real = 0.0` — an absolute lower bound on each coordinate's jitter
  scale. Raise it if a parameter sits near 0 with a tiny step (otherwise
  `scaleᵢ → 0` and that coordinate is never explored).
- `max_rounds::Integer = 6` — stop after this many improvement rounds (≥ 1).
- `strategy = Strategy(1)` — MIGRAD strategy for every (re-)fit.
- `maxfcn::Union{Integer,Nothing} = nothing` — per-fit MIGRAD call budget
  (`nothing` ⇒ MIGRAD's default `200 + 100n + 5n²`). Raise it for expensive FCNs
  whose restarts need many calls to converge.
- `min_improvement::Real = 1e-3` — a restart must beat the current best χ² by
  more than this to be adopted (guards against same-basin numerical jitter).
  (Note: this is the *adoption margin*, NOT MIGRAD's EDM `tol`.)
- `up::Real = 1.0` — error definition, when `fcn` is a bare callable.
- `seed::Union{Integer,Nothing} = nothing` — RNG seed for reproducible restarts.
- `verbose::Bool = false` — log the best χ² per round.

!!! note "Not a global-optimum guarantee (hence the name)"
    Basin-hopping is a heuristic: it finds a *deeper* basin when its restarts
    land in one, but cannot prove the result is global — which is why this is
    `find_deeper_minimum`, not `find_global_minimum`. On the IAM ππ fit, for
    instance, it reaches χ²≈308 from a cold start but not the deeper ≈212 a
    data-resampling search finds: **a** deeper minimum, not **the** global one.
    Raise `n_restarts` / `perturb` / `max_rounds` for a more thorough search,
    and cross-check by re-running from independent seeds.

# Example

```julia
fm = find_deeper_minimum(chi2, x0, errs; n_restarts = 40, perturb = 1.5, seed = 1)
is_valid(fm) || error("search failed")
m = Minuit(chi2, values(fm); names = pnames)   # error analysis at the minimum
migrad!(m); hesse(m)
```

See also [`find_solution_modes`](@ref) (cluster sampled solutions into modes).
"""
function find_deeper_minimum(cf::AbstractCostFunction, x0::AbstractVector, errors::AbstractVector;
        n_restarts::Integer = 24, perturb::Real = 1.0, abs_floor::Real = 0.0,
        max_rounds::Integer = 6, strategy = Strategy(1),
        maxfcn::Union{Integer,Nothing} = nothing, min_improvement::Real = 1e-3,
        seed::Union{Integer,Nothing} = nothing, verbose::Bool = false)
    n_restarts >= 1 || throw(ArgumentError("find_deeper_minimum: n_restarts must be ≥ 1"))
    max_rounds >= 1 || throw(ArgumentError("find_deeper_minimum: max_rounds must be ≥ 1"))
    perturb > 0 || throw(ArgumentError("find_deeper_minimum: perturb must be > 0"))
    min_improvement >= 0 || throw(ArgumentError("find_deeper_minimum: min_improvement must be ≥ 0"))
    abs_floor >= 0 || throw(ArgumentError("find_deeper_minimum: abs_floor must be ≥ 0"))
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    errs = collect(Float64, errors)

    # Each fit gets a fresh call budget: `migrad` compares the cost function's
    # CUMULATIVE `nfcn` to `maxfcn`, so without resetting, later restarts on the
    # shared `cf` would start already over the limit and bail immediately.
    _fit(x) = (reset_ncalls!(cf); migrad(cf, x, errs; strategy = strategy, maxfcn = maxfcn))

    best = _fit(collect(Float64, x0))
    for round in 1:max_rounds
        bx = collect(Float64, best.state.parameters.x)
        scale = [max(abs(bx[i]), abs(errs[i]), abs_floor, eps()) for i in eachindex(bx)]
        improved = false
        for _ in 1:n_restarts
            x = bx .+ perturb .* scale .* randn(rng, length(bx))
            # A wild jitter can push a constrained FCN into a throwing region
            # (log of a negative, a singular matrix, …); skip that restart
            # rather than aborting the whole search.
            fm = try
                _fit(x)
            catch err
                err isa Union{DomainError,BoundsError,SingularException,ArgumentError,DivideError} || rethrow()
                continue
            end
            # NB: restarts in a round all jitter around this round's `bx` (the
            # current best is re-centred only at the next round boundary) — the
            # standard basin-hopping choice.
            if is_valid(fm) && isfinite(fval(fm)) && fval(fm) < fval(best) - min_improvement
                best = fm
                improved = true
            end
        end
        verbose && @info "find_deeper_minimum" round χ² = fval(best) improved
        improved || break
    end
    return best
end

find_deeper_minimum(f, x0::AbstractVector, errors::AbstractVector; up::Real = 1.0, kwargs...) =
    find_deeper_minimum(CostFunction(f, up), x0, errors; kwargs...)

# ─── Perturbation: Minuit convenience dispatch ────────────────────────────────

"""
    find_deeper_minimum(m::Minuit; kwargs...) -> FunctionMinimum

Convenience overload — equivalent to
`find_deeper_minimum(m.fcn, collect(m.values), collect(m.errors); kwargs...)`.
Avoids extracting `values` and `errors` by hand when you already hold a
converged [`Minuit`](@ref). All keyword arguments are forwarded to the
parameter-perturbation overload.
"""
find_deeper_minimum(m::Minuit; kwargs...) =
    # Route through m.cfwg (the CostFunctionWithGradient) when the user supplied an
    # analytical / AD gradient, so the perturbation restarts use the gradient path.
    # Fall back to m.fcn (numerical CostFunction) when no gradient was given.
    find_deeper_minimum(m.cfwg === nothing ? m.fcn : m.cfwg,
                        collect(Float64, m.values), collect(Float64, m.errors); kwargs...)

# ─── Resampling: pre-fitted Minuit (primary implementation) ──────────────────

"""
    find_deeper_minimum(m::Minuit, refit, data; kwargs...) -> Minuit

Data-resampling basin-hopping search for a **deeper** minimum on a multi-basin
data-fitting objective.

Unlike the parameter-perturbation overload, which randomly jitters parameters,
this overload exploits the *statistical structure of the data*: each bootstrap
resample of `data` is re-fit by `refit`, which naturally drifts toward the basin
that best explains *that* data subset — making it far more effective on surfaces
where basins are separated by large distances in parameter space relative to the
HESSE scale.

!!! note "Comparability: all χ² comparisons are on the original data"
    `refit(subdata, start)` runs on a bootstrap subsample and returns *parameter
    vectors* only, used as starting-point candidates. All function-value comparisons
    use [`find_solution_modes`](@ref) with `refine=true`, which re-evaluates each
    candidate on the *original* objective (via `m.fcn`). Comparability with
    `m.fval` is guaranteed by design.

!!! warning "Unbounded only — same contract as the perturbation overload"
    Parameter limits and fixed flags are **not** carried over to the adopted
    Minuit. Fold any bounds into the cost function before calling.
    Always check `is_valid` on the returned `Minuit`.

# Arguments
- `m::Minuit` — a converged fit (MIGRAD + HESSE already run). Provides the cost
  function, current best parameters as the warm-start for discovery, and the
  HESSE covariance for whitening inside [`find_solution_modes`](@ref).
- `refit` — any callable: `refit(subdata, start::Vector{Float64}) -> Vector{Float64}`.
  Must run MIGRAD on the resampled subset starting from `start`, and return the
  fitted parameter vector, or a `NaN`-filled vector of the same length if the
  fit is invalid. Accepts functors/callable structs as well as plain functions.
- `data` — the full dataset. Must support `data[idx]` indexing where `idx` is a
  `Vector{Int}` bootstrap index vector.

# Keyword arguments
- `n_discovery::Integer = 20` — bootstrap resamples per round. Keep ≥ 10 for
  [`find_solution_modes`](@ref) clustering to be stable; must be ≥ 2.
- `max_rounds::Integer = 6` — cap on adoption rounds. Each round runs
  `n_discovery` refits + one full-data re-minimisation if a deeper basin is found.
- `strategy = Strategy(1)` — MIGRAD strategy for the adoption re-fit (full data).
  Discovery-phase strategy is controlled inside `refit`.
- `min_improvement::Real = 1e-3` — minimum χ² drop required to adopt a basin.
- `parallel::Union{Bool,Nothing} = nothing` — controls threading in two places:
  (1) the discovery loop (`Threads.@threads`) is parallelised **only** when
  `parallel === true` — `refit` must be thread-safe (no shared mutable state);
  under `nothing` or `false` discovery is always serial.
  (2) the value is passed through to [`find_solution_modes`](@ref), which
  auto-threads the per-mode re-fits when `nothing` and `m.threaded_gradient` is set.
- `seed::Union{Integer,Nothing} = nothing` — RNG seed for reproducible bootstrap
  index generation. Indices are pre-generated sequentially before any parallel
  section, so results are deterministic regardless of `parallel`.
- `verbose::Bool = false` — log χ² improvement at each adoption round.

# Returns
The deepest `Minuit` found (MIGRAD + HESSE already run at the new basin), or the
*input `m` unchanged* if no deeper basin was found or the suitability check fired.

# Suitability check
After the **first** discovery round, if no resample finds a deeper basin the
surface appears single-basin for data-resampling at this start point — a `@warn`
is emitted and `m` is returned immediately. Try the parameter-perturbation
overload: `find_deeper_minimum(m; perturb=…, n_restarts=…)`.

# Example
```julia
m = Minuit(chi2, x0; names = pnames, errors = errs); migrad!(m); hesse(m)

refit = (subdata, start) -> begin
    cf = CostFunction(lec -> chi2_on(subdata, lec), 1.0)
    fm = migrad(cf, start, errs; strategy = Strategy(1))
    is_valid(fm) ? collect(values(fm)) : fill(NaN, length(start))
end

m_deep = find_deeper_minimum(m, refit, pts)
is_valid(m_deep) || error("search failed")
# m_deep already has HESSE; run minos!(m_deep) or get_contours_samples for errors
```

See also [`find_solution_modes`](@ref), the parameter-perturbation overload
[`find_deeper_minimum(fcn, x0, errors)`](@ref).
"""
function find_deeper_minimum(m::Minuit, refit, data;
        n_discovery     :: Integer                = 20,
        max_rounds      :: Integer                = 6,
        strategy                                  = Strategy(1),
        min_improvement :: Real                   = 1e-3,
        parallel        :: Union{Bool,Nothing}    = nothing,
        seed            :: Union{Integer,Nothing} = nothing,
        verbose         :: Bool                   = false)

    n_discovery >= 2 ||
        throw(ArgumentError("find_deeper_minimum: n_discovery must be ≥ 2"))
    max_rounds >= 1 ||
        throw(ArgumentError("find_deeper_minimum: max_rounds must be ≥ 1"))
    min_improvement >= 0 ||
        throw(ArgumentError("find_deeper_minimum: min_improvement must be ≥ 0"))
    n  = length(data)
    n >= 2 || throw(ArgumentError("find_deeper_minimum: data must have ≥ 2 elements"))
    np = length(m.values)

    rng    = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    # Parallelize discovery only when explicitly requested: refit must be thread-safe.
    do_par = parallel === true

    for iter in 1:max_rounds
        p_star = collect(Float64, m.values)

        # Pre-generate all bootstrap index vectors sequentially so the RNG state
        # is deterministic regardless of whether the discovery loop runs in parallel.
        all_idx = [rand(rng, 1:n, n) for _ in 1:n_discovery]

        # ── Discovery: refit each bootstrap resample ──────────────────────────
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

        # ── Too few valid resamples: bail ─────────────────────────────────────
        if length(valid_rows) < 2
            @warn "find_deeper_minimum (resampling): only $(length(valid_rows)) valid " *
                  "resample(s) in round $iter — FCN may throw or fail on most bootstrap " *
                  "subsets. Check your `refit` function."
            break
        end

        # ── Cluster and refine each mode on the ORIGINAL data (via m.fcn) ─────
        # Pre-allocate disc matrix (avoids repeated vcat allocations).
        disc = Matrix{Float64}(undef, length(valid_rows), np)
        for (i, r) in enumerate(valid_rows)
            disc[i, :] .= r
        end
        md = find_solution_modes(disc, m; refine = true, parallel = parallel)

        # ── Suitability check (round 1 only) ─────────────────────────────────
        # `new_min` is false when every resample converged to the same or a shallower
        # basin — the surface may be single-basin, or the bootstrap may just not
        # have explored deeply enough (try a larger `n_discovery`).
        if iter == 1 && !any(x.new_min for x in md)
            @warn "find_deeper_minimum (resampling): round 1 — " *
                  "$(length(valid_rows)) valid resample(s) formed $(length(md)) mode(s), " *
                  "none deeper than current best (χ² = $(round(m.fval; digits=3))). " *
                  "No deeper basin found via data-resampling at this start. " *
                  "The surface may be single-basin here, or bootstrap coverage was insufficient " *
                  "(try a larger `n_discovery`). " *
                  "For parameter-space search try: `find_deeper_minimum(m; perturb=…, n_restarts=…)`."
            return m
        end

        # ── Find the deepest valid new-minimum mode ───────────────────────────
        deeper = [x for x in md if x.new_min && x.refined_valid]
        if isempty(deeper)
            verbose && @info "find_deeper_minimum (resampling)" iter msg = "no deeper basin → stable"
            break
        end
        bn          = deeper[argmin(x.refined_fval for x in deeper)]
        fval_before = m.fval
        fval_after  = bn.refined_fval
        fval_before - fval_after >= min_improvement || break

        verbose && @info "find_deeper_minimum (resampling)" iter χ²_before = fval_before χ²_after = fval_after

        # ── Adopt: rebuild Minuit at new basin, re-fit + HESSE on full data ───
        # Preserve the analytical gradient function if the original fit used one.
        # Note: parameter limits and fixed flags are not carried over (unbounded
        # contract, matching the perturbation overload).
        grad = m.cfwg === nothing ? nothing : m.cfwg.g
        # Use the refined_errors from the new basin (populated by migrad! inside
        # _refine_mode) as step sizes — more appropriate than the old basin's
        # HESSE errors, which may differ in scale by orders of magnitude.
        # Fall back to m.errors only if refined_errors is unexpectedly empty.
        adopt_errs  = isempty(bn.refined_errors) ? collect(Float64, m.errors) :
                                                    collect(Float64, bn.refined_errors)
        prev_ndata  = m.ndata
        prev_prec   = m.prec
        prev_vt     = m.verify_threading
        m = Minuit(m.fcn.f, collect(Float64, bn.refined_values);
                   name              = collect(m.parameters),
                   error             = adopt_errs,
                   grad              = grad,
                   up                = m.fcn.up,
                   strategy          = strategy,
                   tol               = m.tol,
                   print_level       = m.print_level,
                   threaded_gradient = m.threaded_gradient,
                   verify_threading  = prev_vt,
                   prec              = prev_prec)
        m.ndata = prev_ndata   # not a constructor kwarg; set post-construction
        migrad!(m); hesse(m)
    end
    return m
end

# ─── Resampling: fresh-start (delegates to pre-fitted) ────────────────────────

"""
    find_deeper_minimum(cf::AbstractCostFunction, x0, errors, refit, data; kwargs...) -> Minuit

Fresh-start data-resampling overload. Runs an initial MIGRAD + HESSE from `x0`
to build the reference [`Minuit`](@ref), then delegates to the
`(m::Minuit, refit, data)` overload. See that overload for full keyword
documentation.

When you already have a converged `Minuit`, prefer passing it directly —
`find_deeper_minimum(m, refit, data)` skips the initial MIGRAD + HESSE.
"""
function find_deeper_minimum(cf::AbstractCostFunction, x0::AbstractVector,
                             errors::AbstractVector, refit, data;
                             strategy = Strategy(1), kwargs...)
    errs = collect(Float64, errors)
    m0   = Minuit(cf.f, collect(Float64, x0); error = errs, up = cf.up, strategy = strategy)
    migrad!(m0); hesse(m0)
    find_deeper_minimum(m0, refit, data; strategy = strategy, kwargs...)
end

# ─── Resampling: plain-callable wrapper ───────────────────────────────────────

find_deeper_minimum(f, x0::AbstractVector, errors::AbstractVector, refit, data;
                    up::Real = 1.0, kwargs...) =
    find_deeper_minimum(CostFunction(f, up), x0, errors, refit, data; kwargs...)

# ─── Dispatch disambiguator ───────────────────────────────────────────────────
# Without this method, a call shaped `(Minuit, AbstractVector, AbstractVector)`
# is ambiguous: the perturbation plain-callable wrapper is more specific on
# args 2–3, while the resampling pre-fitted overload is more specific on arg 1.
# Julia would throw MethodError. This sentinel is more specific on ALL three
# args and resolves the ambiguity with a helpful message.
find_deeper_minimum(::Minuit, ::AbstractVector, ::AbstractVector; kwargs...) =
    throw(ArgumentError(
        "find_deeper_minimum: ambiguous 3-arg call (Minuit, AbstractVector, AbstractVector). " *
        "For parameter-perturbation from a Minuit use the 1-arg form: " *
        "`find_deeper_minimum(m; perturb=…)`. " *
        "For data-resampling pass a callable `refit` and a data collection: " *
        "`find_deeper_minimum(m, refit, data)`."))


# Deprecated 0.3.1 name. Basin-hopping cannot certify a global minimum, so the
# honest name is `find_deeper_minimum`; this warning-emitting alias keeps any
# v0.3.1 code working.
function find_global_minimum(args...; kwargs...)
    Base.depwarn("`find_global_minimum` is deprecated; use `find_deeper_minimum` " *
                 "(basin-hopping cannot guarantee a *global* minimum).", :find_global_minimum)
    return find_deeper_minimum(args...; kwargs...)
end
