# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# minuit.jl — iminuit-style Minuit wrapper (Phase 3 first cut).
#
# Mirrors the iminuit Python API. A single mutable `Minuit` object
# bundles the CostFunction, Parameters, and lazily-computed
# FunctionMinimum + MINOS errors + contours. Methods are mutating
# (suffix `!`) to match Julia idiom while supporting iminuit-style
# property access (`m.values`, `m.errors`, etc.) via getproperty.
#
# Usage:
#
#   m = Minuit(my_fcn, [1.0, 2.0]; names = ["a", "b"],
#                                   errors = [0.1, 0.1],
#                                   limits = [(0, 10), nothing])
#   migrad!(m)
#   hesse!(m)
#   minos!(m)
#   println(m.values)        # ≈ [1.0, 2.0] for f = (x-1)² + (y-2)²
#   println(m.errors)        # external 1σ errors
#   println(m)               # pretty table with parameters + errors
# ─────────────────────────────────────────────────────────────────────────────

"""
    Minuit(fcn, x0; names, errors, limits, fixed, up=1.0, prec=...)

iminuit-style wrapper. Constructs the underlying `CostFunction` and
`Parameters` and exposes mutating MIGRAD / HESSE / MINOS / contour
methods plus iminuit-style property access.

# Arguments

- `fcn` — the user function `f(x::AbstractVector) -> Real`.
- `x0::AbstractVector{<:Real}` — initial parameter values (external).

# Keyword arguments

- `names::Vector{<:AbstractString}=["p1", ..., "pn"]` — parameter
  names.
- `errors::Vector{<:Real}=fill(0.1, n)` — initial step sizes.
- `limits::Vector` — per-parameter bounds. Each entry may be:
    - `nothing` for unbounded,
    - `(lo, up)` for both bounds,
    - `(nothing, up)` for upper-only,
    - `(lo, nothing)` for lower-only.
- `fixed::Vector{Bool}=fill(false, n)`.
- `up::Real=1.0` — ErrorDef. `1.0` for χ², `0.5` for NLL.
- `prec::MachinePrecision`.

# Methods

- `migrad!(m; strategy, tol, maxfcn)` — run MIGRAD.
- `hesse!(m; strategy)` — refine the Hessian.
- `minos!(m, par_idx_or_name; ...)` — single-parameter MINOS.
- `minos!(m; ...)` — MINOS on all free parameters.
- `contour(m, par_x, par_y; npoints)` — 2D contour.

# Properties (iminuit-style)

- `m.values` — external parameter values.
- `m.errors` — external 1σ errors.
- `m.fval`, `m.edm`, `m.nfcn`, `m.valid`.
- `m.covariance` — full external covariance matrix or `nothing`.
- `m.params` — the underlying `Parameters`.
- `m.fmin` — the underlying `BoundedFunctionMinimum` (`nothing`
  before `migrad!`).
"""
mutable struct Minuit
    # Phase F: was concretely `::CostFunction`. Now `::AbstractCostFunction`
    # so users can construct `Minuit(fcn, x0; grad=g, ...)` and have the
    # AD gradient survive the bounded MIGRAD → MINOS / contour chain.
    fcn::AbstractCostFunction
    params::Parameters
    fmin::Union{Nothing,BoundedFunctionMinimum}
    minos_errors::Dict{Int,MinosError}
    prec::MachinePrecision
    # When the user supplies `grad=...` to the constructor, the cached
    # CostFunctionWithGradient is kept here for migrad! / hesse to
    # dispatch into the AD-backed path. Shares the same `nfcn` Ref
    # as `fcn` so the call counter is consistent. `nothing` for plain
    # (no-gradient) fits.
    cfwg::Union{Nothing,CostFunctionWithGradient}
    # IMinuit.jl-compatible stored settings — read at migrad! /
    # minos! / hesse! time when not explicitly passed. Mirrors
    # iminuit's `m.strategy`, `m.tol`, `m.print_level` attributes.
    strategy::Strategy
    tol::Float64
    print_level::Int
    # Phase G: when `true`, the inner `numerical_gradient!` parallelizes
    # its `for i in 1:n` per-parameter loop via `Threads.@threads`. Requires
    # Julia started with `julia -t N`. The user FCN must be thread-safe
    # (no hidden RNG / cache / file I/O state); cf_fixed splice buffers
    # are already per-thread via Phase G.1. Default `false` (single-threaded
    # behavior identical to pre-Phase G).
    threaded_gradient::Bool
    # Phase H: when `true` AND `threaded_gradient=true`, JuMinuit runs
    # one extra sequential+threaded gradient comparison at the seed
    # point on the first migrad call. Throws `ThreadSafetyError` if the
    # FCN's threaded gradient disagrees with the sequential one (race
    # in user code → silently wrong minimum without this check). Default
    # `true` whenever `threaded_gradient=true` — costs ~2 extra gradient
    # evaluations (negligible for the typical case where threading is
    # only enabled for expensive FCNs). Set to `false` to bypass once
    # you've confirmed thread-safety another way.
    verify_threading::Bool
    # Diagnostic: number of MIGRAD passes the last `migrad!` executed
    # (1 = single-shot / no retry; >1 = the robust retry loop ran extra
    # passes). Lets callers and tests observe the fixed-point / multi-scale
    # early-stop behavior without instrumenting FCN call counts. `0` before
    # the first `migrad!`.
    n_passes::Int
end

function Minuit(
    fcn,
    x0::AbstractVector{<:Real};
    # IMinuit.jl / iminuit-compatible kwarg names (singular)
    name::Union{Vector{<:AbstractString},Vector{Symbol},Nothing} = nothing,
    error::Union{Vector{<:Real},Real,Nothing} = nothing,
    # JuMinuit-native plural forms (kept for backward compat with
    # existing tests; aliased to the singular ones above)
    names::Union{Vector{<:AbstractString},Nothing} = nothing,
    errors::Union{Vector{<:Real},Nothing} = nothing,
    limits::Union{Vector,Nothing} = nothing,
    fixed::Union{Vector{Bool},Nothing} = nothing,
    up::Real = 1.0,
    errordef::Union{Real,Nothing} = nothing,  # iminuit alias for `up`
    prec::MachinePrecision = MachinePrecision(),
    # IMinuit.jl-compatible: user-supplied gradient. Pass a callable
    # `g(x_ext) -> Vector{Float64}` (e.g. `x -> ForwardDiff.gradient(f, x)`)
    # to use analytical / AD-backed gradients instead of central-difference.
    # Typically 5-10× fewer FCN evaluations on cheap FCNs.
    grad::Union{Function,Nothing} = nothing,
    # IMinuit.jl-compatible stored settings. These become `m.strategy`,
    # `m.tol`, `m.print_level` and feed into subsequent migrad! calls
    # when not explicitly overridden.
    strategy::Union{Strategy,Integer} = Strategy(0),
    tol::Real = 0.1,
    print_level::Integer = 0,
    # Phase G: parallel inner numerical-gradient. Requires `julia -t N`.
    # Default `false` keeps the (single-threaded) reference behavior;
    # opt-in when (a) FCN > 500 ns/call, (b) n ≥ 4, (c) FCN is thread-
    # safe (no hidden mutable state).
    threaded_gradient::Bool = false,
    # Phase H: auto-verify FCN thread-safety on first gradient call when
    # threading is enabled. Default `true` whenever `threaded_gradient=true`
    # — costs ~2 extra gradient evaluations. Pass `false` to bypass once
    # you've confirmed safety via `JuMinuit.is_thread_safe(cf, x0)` or
    # otherwise.
    verify_threading::Bool = threaded_gradient,
    # Catch-all for per-parameter `error_<name>`, `fix_<name>`, `limit_<name>`
    # kwargs in the IMinuit.jl style.
    kwargs...,
)
    n = length(x0)

    # Resolve names: singular > plural > default
    nm = if name !== nothing
        String.(name)
    elseif names !== nothing
        String.(names)
    else
        ["x$(i-1)" for i in 1:n]   # iminuit-style x0, x1, ...
    end

    # Resolve errors: singular > plural > default 0.1
    er_vec = if error !== nothing
        error isa Real ? fill(Float64(error), n) : Float64.(error)
    elseif errors !== nothing
        Float64.(errors)
    else
        fill(0.1, n)
    end
    fx_vec = fixed === nothing ? fill(false, n) : collect(Bool, fixed)
    lim_vec = limits === nothing ? Vector{Any}(fill(nothing, n)) : Vector{Any}(limits)

    # Apply IMinuit.jl per-parameter kwargs: `error_<pname>`, `fix_<pname>`,
    # `limit_<pname>`. The parameter name lookup is by string match against
    # the resolved `nm`.
    name_to_idx = Dict{String,Int}(s => i for (i, s) in enumerate(nm))
    up_resolved = errordef !== nothing ? Float64(errordef) : Float64(up)

    for (k, v) in kwargs
        ks = String(k)
        us = findfirst('_', ks)
        if us === nothing
            throw(ArgumentError("Minuit: unrecognized keyword `$k`"))
        end
        typ = ks[1:us-1]
        pname = ks[us+1:end]
        idx = get(name_to_idx, pname, 0)
        idx == 0 &&
            throw(ArgumentError("Minuit: kwarg `$k` references unknown parameter `$pname`"))
        if typ == "error"
            er_vec[idx] = Float64(v)
        elseif typ == "fix"
            fx_vec[idx] = Bool(v)
        elseif typ == "limit"
            lim_vec[idx] = v
        else
            throw(ArgumentError("Minuit: unrecognized keyword prefix `$typ` in `$k`"))
        end
    end

    n == length(nm) == length(er_vec) == length(fx_vec) == length(lim_vec) ||
        throw(ArgumentError("Minuit: x0/names/errors/limits/fixed length mismatch"))

    # Translate the iminuit-style limits parametrization into the
    # (lower, upper) tuple format Parameters expects (NaN = absent).
    limit_tuples = Vector{Tuple{Float64,Float64}}(undef, n)
    for i in 1:n
        lo_i, up_i = NaN, NaN
        l = lim_vec[i]
        if l !== nothing
            # `l` can be a 2-tuple, a Vector{<:Real} of length 2, or
            # (nothing, x) / (x, nothing) for one-sided.
            lo_raw, up_raw = l
            if lo_raw !== nothing && !(lo_raw isa Real && isinf(lo_raw))
                lo_i = Float64(lo_raw)
            end
            if up_raw !== nothing && !(up_raw isa Real && isinf(up_raw))
                up_i = Float64(up_raw)
            end
        end
        limit_tuples[i] = (lo_i, up_i)
    end
    params = Parameters(nm, Float64.(x0), er_vec;
                         limits = limit_tuples, fixed = fx_vec,
                         prec = prec)
    cf = CostFunction(fcn, up_resolved)
    # Build cached CFwG when grad provided — share nfcn Ref so call
    # count is consistent across both views into the user FCN.
    cfwg = grad === nothing ? nothing :
        CostFunctionWithGradient(fcn, grad, up_resolved, cf.nfcn, Ref(0))
    strat = strategy isa Strategy ? strategy : Strategy(Int(strategy))
    return Minuit(cf, params, nothing, Dict{Int,MinosError}(), prec,
                  cfwg, strat, Float64(tol), Int(print_level),
                  Bool(threaded_gradient), Bool(verify_threading), 0)
end

# IMinuit.jl-style: named-parameter constructor where each parameter
# is given as a keyword argument by name.
#
#   m = Minuit(my_fcn; x = 1.0, y = 0.0,
#                       error_x = 0.1, error_y = 0.2,
#                       fix_x = false, limit_y = (0, 50))
#
# Parameter names are inferred from the kwargs (filtering out
# `error_*`, `fix_*`, `limit_*`, and known config kwargs). For
# Julia code with a `function f(par)` signature where `par` is a
# vector, this constructor would feed each scalar param to `f` as
# a separate argument — for that case use the `Minuit(fcn, x0)`
# vector-start constructor.
function Minuit(fcn;
                up::Real = 1.0,
                errordef::Union{Real,Nothing} = nothing,
                prec::MachinePrecision = MachinePrecision(),
                grad::Union{Function,Nothing} = nothing,
                strategy::Union{Strategy,Integer} = Strategy(0),
                tol::Real = 0.1,
                print_level::Integer = 0,
                threaded_gradient::Bool = false,
                verify_threading::Bool = threaded_gradient,
                kwargs...)
    # Separate `error_*`, `fix_*`, `limit_*`, and meta from the
    # parameter-name kwargs.
    par_kws = Pair{Symbol,Float64}[]
    other_kws = Pair{Symbol,Any}[]
    for (k, v) in kwargs
        ks = String(k)
        us = findfirst('_', ks)
        if us !== nothing
            prefix = ks[1:us-1]
            if prefix in ("error", "fix", "limit")
                push!(other_kws, k => v)
                continue
            end
        end
        if v isa Real
            push!(par_kws, k => Float64(v))
        else
            throw(ArgumentError("Minuit: parameter kwarg `$k` must be a Real (got $(typeof(v)))"))
        end
    end
    names = [String(k) for (k, _) in par_kws]
    x0 = [v for (_, v) in par_kws]
    # Wrap the user's `fcn(par::AbstractVector)` so it's called per
    # the JuMinuit convention. If the user's `fcn` takes positional
    # scalar args (e.g. `f(a, b, c)`), wrap with a splat.
    f_wrapped = if applicable(fcn, x0)
        fcn
    else
        x -> fcn(x...)
    end
    return Minuit(f_wrapped, x0; name = names, up = up,
                  errordef = errordef, prec = prec,
                  grad = grad, strategy = strategy, tol = tol,
                  print_level = print_level,
                  threaded_gradient = threaded_gradient,
                  verify_threading = verify_threading,
                  other_kws...)
end

# IMinuit.jl-style: copy-from-another-fit constructor.
function Minuit(fcn, m::Minuit; kwargs...)
    # Use the latest values (post-MIGRAD if available) as new starting
    # point, preserving param names and bound config unless overridden.
    x0 = m.fmin === nothing ? [p.value for p in m.params.pars] : m.fmin.ext_values
    nm = [p.name for p in m.params.pars]
    er = m.fmin === nothing ? [p.error for p in m.params.pars] : m.fmin.ext_errors
    fx = [is_fixed(p) for p in m.params.pars]
    lim = Vector{Any}(undef, n_pars(m.params))
    for (i, p) in enumerate(m.params.pars)
        lo = isnan(p.lower) ? nothing : p.lower
        hi = isnan(p.upper) ? nothing : p.upper
        lim[i] = (lo === nothing && hi === nothing) ? nothing : (lo, hi)
    end
    # Splat the recovered config into the main constructor; user kwargs
    # take precedence (we put theirs LAST in the call). Carry over the
    # source m's stored strategy/tol/print_level so a `Minuit(f, m)`
    # rebuild preserves the user's tuning.
    return Minuit(fcn, x0; name = nm, error = er, fixed = fx, limits = lim,
                            up = m.fcn.up, prec = m.prec,
                            strategy = m.strategy, tol = m.tol,
                            print_level = m.print_level,
                            threaded_gradient = m.threaded_gradient,
                            verify_threading = m.verify_threading, kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mutating methods
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad!(m::Minuit; strategy=m.strategy, tol=m.tol, maxfcn=nothing,
                       iterate=5, use_simplex=true,
                       threaded_gradient=m.threaded_gradient,
                       verify_threading=m.verify_threading,
                       print_level=m.print_level) -> Minuit

Run MIGRAD on `m` with an iminuit-shaped robust retry on top of the
M5 prior-cov mechanism from PR #4. If pass 1 fails to validate
(no-improvement / above-max-EDM exit, see C++
`VariableMetricBuilder.cxx:278`) and the call limit hasn't been
reached, retry up to `iterate-1` more passes. Each retry:

- Upgrades to `Strategy(2)` regardless of the user request (iminuit
  heuristic in `_robust_low_level_fit`). Exception: when the FCN was
  constructed with `grad=...`, the AD seed path requires Strategy(0),
  so the user's stored strategy is retained on retry — Simplex + the
  prior_cov carry remain in effect.
- Optionally runs a Nelder-Mead Simplex step from the failed point
  before the next MIGRAD (`use_simplex=true`, the default). With
  `use_simplex=false`, the failed fit's inverse Hessian is carried
  to the next MIGRAD seed as `prior_cov` (uses the M5 mechanism from
  PR #4, see `reference/Minuit2_cpp/src/MnSeedGenerator.cxx:63-67`
  HasCovariance branch). With `use_simplex=true`, the post-Simplex
  state has no usable inverse Hessian, so `prior_cov` is dropped and
  the next seed falls back to the diagonal-from-g2 estimate (this
  matches installed iminuit 2.32.0 `_robust_low_level_fit`). The
  Simplex seed step **grows geometrically** per retry (×1, ×2, ×4, …
  from the parameter error scale, capped at the parameter's physical
  range) so a neighbouring minimum reachable at some scale is found —
  a structured multistart, in the spirit of MINUIT `MnMinimize` and
  scipy `basinhopping`.

The loop is a structured multistart that stops early when it is
provably redundant:

1. **Fixed-point (cycle) detection.** A history of every converged
   `(x, fval)` is kept; if a pass re-converges to an already-visited
   minimum (within a parameter-error-relative tolerance) the retry map
   has cycled and the loop stops. This recovers the wasted retries on
   fits with a single reachable basin (e.g. IAM, where all retries
   re-converge to the same point).
2. **Multi-scale escape** for genuine multiple local minima (the common
   case in multi-parameter HEP amplitude / phase-shift / LEC fits): the
   growing Simplex perturbation reaches further out each pass until it
   escapes, cycles, or spans the physical range.

This does NOT prove the global minimum was found (global optimization is
undecidable). The defensible statement: searched perturbation scales up
to re-convergence on a known basin or the physical parameter range.

`iterate=1` disables the retry loop and reproduces single-shot
C++-faithful behavior. The **safety invariant** is guaranteed by
construction: `iterate=N` never yields a worse `fval` than `iterate=1`
(the lowest-fval pass is published, tie → the valid one). The number of
MIGRAD passes the call executed is recorded in `m.n_passes` (1 = no
retry).

Updates `m.fmin`. Returns `m` for chaining. If the constructor was
given `grad=...`, dispatches into the analytical-gradient path on
every pass. `strategy` / `tol` / `print_level` / threading flags
default to whatever the user stored on `m` (settable via
`m.strategy = ...`, `m.tol = ...` or the constructor kwargs).

If a prior `m.fmin` exists, pass 1 starts from the previous converged
point (iminuit-compatible implicit resume). Use [`reset`](@ref) (or
`migrad(m; resume=false)`) to drop the prior fit and restart from the
constructor's initial values. `m.params` itself is NEVER mutated —
the carry-forward builds a fresh `Parameters` only for the duration
of the inner MIGRAD call.
"""
function migrad!(m::Minuit;
                  strategy::Strategy = m.strategy,
                  tol::Real = m.tol,
                  maxfcn::Union{Integer,Nothing} = nothing,
                  iterate::Integer = 5,
                  use_simplex::Bool = true,
                  threaded_gradient::Bool = m.threaded_gradient,
                  verify_threading::Bool = m.verify_threading,
                  print_level::Integer = m.print_level)
    # iminuit's `_robust_low_level_fit` requires iterate ≥ 1. iterate=1
    # disables retry (only pass 1 runs) and reproduces single-shot
    # C++-faithful behavior; iterate ≤ 0 would silently skip MIGRAD
    # entirely, so reject it to surface the user mistake.
    iterate >= 1 ||
        throw(ArgumentError("iterate must be ≥ 1, got $iterate"))

    # Pass 1: user's stored Strategy, no prior_cov. Byte-identical to
    # the pre-retry-layer single-shot path — relies on `_migrad_into!`'s
    # default `prior_cov=nothing` matching the bounded `migrad`
    # overload's default. If a future change ever introduces a
    # non-`nothing` default anywhere in this kwarg chain, the
    # `iterate=1` test in `test_minuit_retry.jl` will catch it.
    #
    # iminuit-style implicit resume: if we already converged once, carry
    # the prior ext_values forward as the new starting point. m.params
    # untouched (review BLOCKING #2 from the original `migrad!` review).
    params_to_use = m.fmin === nothing ? m.params : _build_resume_params(m)
    bfm = _migrad_into!(m, params_to_use;
                         strategy = strategy, tol = tol, maxfcn = maxfcn,
                         threaded_gradient = threaded_gradient,
                         verify_threading = verify_threading,
                         print_level = print_level)

    # Robust retry loop — a structured multistart on top of the iminuit
    # `_robust_low_level_fit` shape (Simplex hop / prior_cov carry /
    # Strategy(2) bump). Two refinements over the PR #8 fixed-scale version:
    #
    #   (1) Fixed-point (cycle) detection. We keep a history of every
    #       converged (x, fval). If a later pass re-converges to an
    #       already-visited minimum (within a parameter-error-relative
    #       tolerance), the perturbed restart fell back into a basin we
    #       have already catalogued — the retry map has cycled — so we
    #       stop. This is the dedup-on-revisit rule scipy `basinhopping`
    #       (`niter_success`) and practical multistart use; it is a
    #       heuristic, NOT a proof that an even larger perturbation could
    #       not escape (that is undecidable). It recovers the wasted IAM
    #       retries, where every pass re-converges to the same fval.
    #
    #   (2) Geometrically-growing Simplex perturbation. When `use_simplex`,
    #       successive passes enlarge the simplex seed step ×2 each pass
    #       (`_retry_perturb_factor`), starting at the parameter error scale
    #       and capped at the parameter's physical range (its bounds, or a
    #       multiple of its error if unbounded). If a neighbouring minimum
    #       is reachable at *some* scale within the physical range, a
    #       growing search finds it (the X(3872) `J/ψρ + DD̄*` multi-dip
    #       case the fixed-scale hop could not escape). Once the perturbation
    #       spans the physical range of every free parameter, further growth
    #       is meaningless → natural termination, independent of `iterate`.
    #
    # We do NOT claim the global minimum is found — global optimization is
    # undecidable, and this deterministic Simplex+MIGRAD search is
    # basin/seed/call-limit dependent. The defensible statement is: it
    # samples increasing perturbation scales up to re-convergence on a known
    # basin or the parameter physical range, and MAY thereby find a deeper
    # minimum it would otherwise miss (cf. MINUIT `MnMinimize` =
    # Migrad+Simplex, scipy `basinhopping`, the multistart literature).
    #
    # Loop exits when the fit validates, the call limit is exhausted, the
    # perturbation saturates the physical range, or a cycle is detected.
    # The retry strategy is `Strategy(2)` for numerical-gradient FCNs (the
    # iminuit default). For analytical-gradient FCNs (`m.cfwg !== nothing`)
    # the AD `seed_state` rejects any strategy.level != 0 (see
    # `src/ad_gradient.jl:254-255`); rather than throw mid-retry we keep the
    # user's stored strategy on the AD path. Retry's value-add for AD users
    # then comes from the Simplex hop and the prior_cov carry, not the bump.
    #
    # Safety invariant (PR #8): `iterate=N` never yields a worse fval than
    # `iterate=1`. We guarantee it *by construction* — `best_bfm` tracks the
    # lowest-fval pass (exact tie → the valid one) and is what lands in
    # `m.fmin`. The loop-control `bfm` is the latest pass (drives the
    # valid/budget break and the next perturbation seed); the published
    # result is `best_bfm`.
    retry_strategy = m.cfwg === nothing ? Strategy(2) : strategy
    best_bfm = bfm
    npass = 1
    # The user's per-parameter step on m.params is the stable, pass-invariant
    # length scale for both the fixed-point tolerance and the growth ceiling.
    base_errs = [p.error for p in m.params.pars]
    visited = Tuple{Vector{Float64},Float64}[(copy(bfm.ext_values), fval(bfm))]
    for _pass in 2:Int(iterate)
        (is_valid(bfm.internal) || bfm.internal.reached_call_limit) && break

        params_next = _build_resume_params(m, bfm)
        prior_cov = nothing
        factor = _retry_perturb_factor(_pass)
        if use_simplex
            # Simplex from the failed point, with the per-pass growing seed
            # step. The result may have shifted into a different basin; we
            # carry its ext_values forward as the next MIGRAD seed. We do
            # NOT extract a prior_cov here — simplex never built an inverse
            # Hessian (its MinimumError is the I placeholder with
            # available=false), and `_retry_prior_cov(sx)` would return
            # nothing in any case. This matches installed iminuit 2.32.0
            # `_robust_low_level_fit`, which recreates `MnMigrad` from the
            # post-Simplex state with no warm-start covariance.
            params_pert = _retry_scaled_params(m, params_next, factor, base_errs)
            sx = simplex(m.fcn, params_pert;
                          maxfcn = maxfcn, prec = m.prec)
            params_next = _build_resume_params(m, sx)
        else
            prior_cov = _retry_prior_cov(bfm)
        end

        bfm = _migrad_into!(m, params_next;
                             strategy = retry_strategy, tol = tol,
                             maxfcn = maxfcn,
                             threaded_gradient = threaded_gradient,
                             verify_threading = verify_threading,
                             print_level = print_level,
                             prior_cov = prior_cov)
        npass += 1
        best_bfm = _retry_select_better(bfm, best_bfm)

        # Cycle detection: re-convergence to an already-catalogued basin.
        _retry_is_fixed_point(bfm, visited, base_errs) && break
        push!(visited, (copy(bfm.ext_values), fval(bfm)))

        # Natural termination: this pass already used `factor`; if that scale
        # has spanned every free parameter's physical range, no larger
        # meaningful hop remains, so stop (only relevant for `use_simplex`).
        use_simplex && _retry_perturb_saturated(m, factor, base_errs) && break
    end

    m.fmin = best_bfm
    m.n_passes = npass
    return m
end

# Dispatch the underlying `migrad(cf|cfwg, params; ...)` call. Used by
# `migrad!` for both pass 1 and every retry pass, sharing the cfwg-vs-cf
# branch in one place. Returns a fresh BoundedFunctionMinimum; the
# caller (migrad! retry loop) decides whether the result is final or
# just one intermediate pass. The bang reflects that the call mutates
# m's shared cf.nfcn counter via the inner MIGRAD's FCN evaluations.
function _migrad_into!(m::Minuit, params::Parameters;
                        strategy::Strategy, tol::Real,
                        maxfcn::Union{Integer,Nothing},
                        threaded_gradient::Bool,
                        verify_threading::Bool,
                        print_level::Integer,
                        prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing)
    if m.cfwg !== nothing
        return migrad(m.cfwg, params;
                       strategy = strategy, tol = tol, maxfcn = maxfcn,
                       prec = m.prec,
                       threaded_gradient = threaded_gradient,
                       verify_threading = verify_threading,
                       prior_cov = prior_cov,
                       print_level = print_level)
    else
        return migrad(m.fcn, params;
                       strategy = strategy, tol = tol, maxfcn = maxfcn,
                       prec = m.prec,
                       threaded_gradient = threaded_gradient,
                       verify_threading = verify_threading,
                       prior_cov = prior_cov,
                       print_level = print_level)
    end
end

# Extract the inv_hessian from a failed BoundedFunctionMinimum for use
# as `prior_cov` in the next MIGRAD pass — but ONLY when the prior fit
# actually produced a usable covariance. Simplex leaves the placeholder
# `MinimumError(I, ..., available=false)`; without this guard that
# identity placeholder would silently leak in as a fake prior, killing
# the retry's information transfer. The returned matrix is in INTERNAL
# coordinates (which is what the bounded `migrad(cf, params;
# prior_cov=...)` path expects — it feeds straight into
# `seed_state(cf_internal, int_vals, int_errs; prior_cov=...)`).
#
# Dimensional invariant: `n_free` is constant across all retry passes
# within one `migrad!` call (we never fix/release between passes — the
# loop runs to completion uninterrupted), so the (n_free × n_free)
# matrix shape automatically matches the next pass's `int_vals` length.
function _retry_prior_cov(bfm::BoundedFunctionMinimum)
    err = bfm.internal.state.error
    is_available(err) || return nothing
    return err.inv_hessian
end

# ─────────────────────────────────────────────────────────────────────────────
# Structured-multistart retry policy (fixed-point detection + multi-scale
# Simplex perturbation). These power the `migrad!` retry loop.
# ─────────────────────────────────────────────────────────────────────────────

# Position match tolerance for fixed-point detection, as a fraction of the
# per-coordinate length scale (max of |value|, |user step|). Two converged
# points count as "the same minimum" when every free coordinate agrees to
# this fraction AND their fvals agree (below). 1% of the value/step scale is
# tighter than the spacing of physically-distinct minima yet looser than a
# minimizer's position reproducibility. This is a HEURISTIC, not a proof:
# erring tight only misses a cycle (a few extra passes — harmless); the
# growing perturbation also means a revisit at one scale does not prove a
# larger scale could not escape (see the loop comment).
const _RETRY_XTOL_REL = 1.0e-2
# fval match tolerance for "same minimum". Deliberately MUCH tighter than the
# general retry tolerance: a genuine re-convergence reproduces fval to
# ~EDM-tolerance (≈1e-6 relative for the default tol), while physically
# distinct minima differ far more. Using a small epsilon (not the coarse
# retry `tol`) avoids treating a real improvement as "the same energy".
# Floor `max(_RETRY_FTOL_ABS, _RETRY_FTOL_REL·|fj|)` keeps it sane near fj≈0.
const _RETRY_FTOL_REL = 1.0e-6
const _RETRY_FTOL_ABS = 1.0e-12
# Unbounded "physical range" = this multiple of the parameter step. The
# perturbation growth is capped here; chosen large enough that for the
# typical `iterate` ≤ 5 the iterate cap (not this ceiling) governs unbounded
# fits, while bounded fits cap at their actual span.
const _RETRY_UNBOUNDED_RANGE_MULT = 1.0e3

# Geometric perturbation factor for retry pass `p` (p ≥ 2): 1, 2, 4, 8, …
# Pass 2 reproduces the PR #8 fixed-scale hop (factor 1); each later pass
# doubles the Simplex seed step so the search reaches further out.
_retry_perturb_factor(p::Integer) = 2.0^(p - 2)

# Per-parameter physical range used to cap perturbation growth. Two-sided
# bounds → the bound span; one-sided/unbounded → a multiple of the step
# (there is no finite physical range, so we cap on the natural scale).
function _retry_param_range(p::MinuitParameter, base_err::Float64)
    if has_lower_limit(p) && has_upper_limit(p)
        return p.upper - p.lower
    end
    return _RETRY_UNBOUNDED_RANGE_MULT * max(abs(base_err), eps())
end

# Build a `Parameters` clone of `params_next` with each free parameter's
# Simplex seed step grown by `factor`, capped so the simplex initial edge
# (10·step, see `simplex`) does not exceed the parameter's physical range.
# `factor ≤ 1` returns `params_next` unchanged so pass 2 is byte-identical
# to the PR #8 fixed-scale hop. Fixed parameters and all values pass
# through untouched. NOTE: the cap is reasoned in EXTERNAL coordinates; for
# two-sided-bounded parameters the bounded `simplex` actually perturbs in
# internal (arcsin-transformed) coordinates, so the cap is an external-
# coordinate proxy — it controls the (heuristic) growth schedule, while the
# int↔ext transform independently keeps every probe inside the bounds.
function _retry_scaled_params(m::Minuit, params_next::Parameters,
                               factor::Float64, base_errs::Vector{Float64})
    factor <= 1.0 && return params_next
    n = n_pars(params_next)
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        p = params_next.pars[i]
        if is_fixed(p)
            new_pars[i] = p
            continue
        end
        range_i = _retry_param_range(m.params.pars[i], base_errs[i])
        grown = p.error * factor
        # Bound GROWTH so the simplex edge (10·step) ≤ range, but never
        # shrink below the carried base step `p.error`. So `10·step ≤ range`
        # holds EXCEPT when the parameter's own step already exceeds
        # range/10 (an unusually large user step for a narrow range); there
        # the step is left at `p.error` and the bounded simplex's int↔ext
        # transform clamps any over-range probe back inside the bounds.
        step = max(min(grown, range_i / 10.0), p.error)
        new_pars[i] = MinuitParameter(p.name, p.value, step;
                                       lower = p.lower, upper = p.upper,
                                       fixed = p.fixed)
    end
    return Parameters(new_pars, m.prec)
end

# True once the (external-coordinate) perturbation has spanned the physical
# range of EVERY free parameter (10·factor·base_err ≥ range): further growth
# is meaningless, so the retry loop can stop independently of the `iterate`
# cap. Uses `base_errs` (the stable user step) for a pass-invariant
# schedule. For unbounded parameters the range is
# `_RETRY_UNBOUNDED_RANGE_MULT × step`, so this only binds for bounded fits
# or very large `iterate`. For two-sided-bounded parameters this is an
# external-coordinate proxy (the simplex perturbs in internal coords); it
# is a heuristic stop, never a correctness guarantee — the best-of-passes
# selector, not this predicate, protects the published fval.
function _retry_perturb_saturated(m::Minuit, factor::Float64,
                                    base_errs::Vector{Float64})
    @inbounds for i in 1:n_pars(m.params)
        p = m.params.pars[i]
        is_fixed(p) && continue
        range_i = _retry_param_range(p, base_errs[i])
        10.0 * factor * max(abs(base_errs[i]), eps()) < range_i && return false
    end
    return true
end

# Best-of-passes selector enforcing the safety invariant: a strictly-lower
# fval always wins; an exact tie goes to the valid result; a worse candidate
# never replaces the incumbent. A non-finite (NaN/Inf) candidate never wins,
# but a finite candidate DOES replace a non-finite incumbent — otherwise a
# NaN pass-1 fval could never be improved (codex BLOCKING). Guarantees the
# published fval is ≤ the pass-1 fval whenever any finite pass exists (so
# `iterate=N` ≤ `iterate=1`).
function _retry_select_better(cand::BoundedFunctionMinimum,
                               best::BoundedFunctionMinimum)
    fc = fval(cand)
    fb = fval(best)
    isfinite(fc) || return best   # NaN/Inf candidate never wins
    isfinite(fb) || return cand   # any finite candidate beats a non-finite incumbent
    fc < fb && return cand
    (fc == fb && is_valid(cand) && !is_valid(best)) && return cand
    return best
end

# Fixed-point (cycle) detection: true when `bfm`'s converged (x, fval)
# matches any previously-visited (x, fval) within the relative tolerances
# above — i.e. the retry map has returned to an already-explored minimum.
# fval is the coarse gate; position disambiguates fval-degenerate distinct
# minima. The per-coordinate length scale is max(|value|, |user step|).
# We deliberately do NOT include the converged `ext_errors`: on an invalid
# fit the int→ext Jacobian (near a bound or with a near-singular Hessian)
# can BLOW UP that uncertainty, which would widen the match window and risk
# merging two genuinely distinct minima — a false stop returning the worse
# fit. |value| and the (stable, pass-invariant) user step are sufficient
# scales. Erring tight only ever misses a cycle (≡ a few extra passes,
# harmless), never causes a false merge.
function _retry_is_fixed_point(bfm::BoundedFunctionMinimum,
                                visited::Vector{Tuple{Vector{Float64},Float64}},
                                base_errs::Vector{Float64})
    x = bfm.ext_values
    f = fval(bfm)
    isfinite(f) || return false   # NaN/Inf pass is not a usable fixed point
    @inbounds for (xj, fj) in visited
        isfinite(fj) || continue  # never match against a non-finite history entry
        abs(f - fj) <= max(_RETRY_FTOL_ABS, _RETRY_FTOL_REL * abs(fj)) || continue
        same = true
        for i in eachindex(x)
            scale = max(abs(x[i]), abs(xj[i]), abs(base_errs[i]), _RETRY_FTOL_ABS)
            if abs(x[i] - xj[i]) > _RETRY_XTOL_REL * scale
                same = false
                break
            end
        end
        same && return true
    end
    return false
end

# Build a fresh `Parameters` with values carried forward from a given
# `BoundedFunctionMinimum`. The user-original m.params is NOT mutated.
# Errors are taken as `max(bfm.ext_errors[i], p_old.error)` — the
# post-MIGRAD ext_error is usually a tighter estimate, but near a
# sin/sqrt bound the C++ Int2extError formula can collapse to a value
# far smaller than the natural scale, which would seed the next MIGRAD
# with steps below the numerical-gradient threshold. The `max` floor
# with the original step protects against this regression (review
# BLOCKING #1). The two-arg form is used by the migrad! retry loop to
# build the seed for each successive pass without first stowing the
# intermediate BFM in `m.fmin`. The one-arg form is the implicit-resume
# entrypoint used at the top of `migrad!`.
function _build_resume_params(m::Minuit, bfm::BoundedFunctionMinimum)
    new_pars = Vector{MinuitParameter}(undef, n_pars(m.params))
    @inbounds for i in 1:n_pars(m.params)
        p_old = m.params.pars[i]
        new_err = max(bfm.ext_errors[i], p_old.error)
        new_pars[i] = MinuitParameter(p_old.name,
                                       bfm.ext_values[i],
                                       new_err;
                                       lower = p_old.lower,
                                       upper = p_old.upper,
                                       fixed = p_old.fixed)
    end
    return Parameters(new_pars, m.prec)
end

_build_resume_params(m::Minuit) = _build_resume_params(m, m.fmin)

"""
    minos!(m::Minuit, par; kwargs...) -> Minuit

Run MINOS for parameter `par` (integer index or String name). Updates
`m.minos_errors`. Requires `m.fmin` to be available (call `migrad!`
first). Returns `m`.
"""
function minos!(m::Minuit, par::Integer;
                threaded_gradient::Bool = m.threaded_gradient,
                print_level::Integer = m.print_level,
                kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `minos!(m)`"))
    # MINOS derives sigma_i = sqrt(2·up·V[i,i]) from the inverse
    # Hessian, so it requires an actual covariance — not the
    # identity placeholder that simplex / scan leave behind. Force
    # the user to call hesse(m) first (review IMPORTANT #2 round-2).
    JuMinuit.is_available(m.fmin.internal.state.error) ||
        throw(ArgumentError(
            "MINOS requires a covariance matrix. The last fit produced " *
            "no inverse Hessian (likely simplex/scan, or HESSE didn't " *
            "run). Call `hesse(m)` first."))
    1 <= par <= n_pars(m.params) ||
        throw(ArgumentError("par index $par out of bounds"))
    is_fixed(m.params.pars[par]) &&
        return m  # skip fixed
    p = m.params.pars[Int(par)]
    has_any_bound = has_limits(p) || has_lower_limit(p) || has_upper_limit(p)
    if has_any_bound
        # Bound-aware EXT-coord MINOS (mirrors C++ MnMinos.cxx:119-131
        # architecture). Search runs in EXTERNAL coordinates with the
        # 1σ step truncated against the parameter bound BEFORE the
        # alpha-search starts. Inner MIGRAD at each probe uses the
        # bounded API, respecting bounds on the other free params.
        # Sign convention is automatic: no Jacobian-swap or sign-cross
        # detection needed; what comes out is directly the EXT error.
        #
        # Phase F: prefer `m.cfwg` (analytical gradient) when the user
        # supplied `grad=...` at construction — the inner MIGRAD chain
        # then runs through the AD path. Falls back to numerical `m.fcn`
        # when no gradient was supplied. (Codex review identified that
        # this path historically used `m.fcn` unconditionally, silently
        # dropping the AD gradient for bounded MINOS.)
        ext_cf = m.cfwg === nothing ? m.fcn : m.cfwg
        m.minos_errors[Int(par)] = _minos_external_via_function_cross(
            m.fmin, ext_cf, Int(par);
            threaded_gradient = threaded_gradient,
            print_level = print_level, kwargs...)
    else
        # Unbounded scanned parameter — search in the INTERNAL frame
        # (m.fmin.internal_cf takes internal coords; m.params.int_of_ext[par]
        # is the internal index of `par`).
        #
        # Mixed case (scanned param unbounded, some OTHER free params
        # bounded): the MnMinos linear-correlation pre-shift inside
        # `minos()` operates in internal coords. For bounded "other"
        # params it must additionally Int2ext + EXT clamp + Ext2int so
        # the pre-shifted internal value stays inside the valid
        # transform range (±π/2 for doubly-bounded). Pass `pars=m.params`
        # so `minos()` has the bound information to do that clamp.
        err = minos(m.fmin.internal, m.fmin.internal_cf,
                    m.params.int_of_ext[par];
                    pars = m.params,
                    threaded_gradient = threaded_gradient,
                    print_level = print_level, kwargs...)
        m.minos_errors[Int(par)] = err
    end
    return m
end

# Helper: run upper + lower function_cross_external and assemble a
# MinosError directly in EXT coords. The C++ analog is MnMinos.cxx's
# pair of MnFunctionCross calls (one per direction) wrapped in a
# MinosError constructor.
function _minos_external_via_function_cross(
    bfm,          # ::BoundedFunctionMinimum
    cf::AbstractCostFunction,
    par_idx::Int;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(1),
    prec::MachinePrecision = MachinePrecision(),
    threaded_gradient::Bool = false,
    sigma::Real = 1.0,
    print_level::Integer = 0,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    par = bfm.params.pars[par_idx]
    ext_min = bfm.ext_values[par_idx]
    ext_err = bfm.ext_errors[par_idx]
    # Compute the (truncated) ext step magnitudes for both directions.
    # For upper-search: step_up = min(par.upper, ext_min + ext_err) - ext_min
    #                          ≥ 0  (or 0 if saturated).
    # For lower-search: step_lo = max(par.lower, ext_min - ext_err) - ext_min
    #                          ≤ 0  (or 0 if saturated).
    # Use side-specific predicates only — has_limits would falsely
    # trigger on one-sided bounds (par.lower or par.upper = NaN → NaN
    # arithmetic). See function_cross_external for the same fix.
    step_up = ext_err
    if has_upper_limit(par)
        step_up = min(ext_err, par.upper - ext_min)
    end
    step_up = max(step_up, 0.0)
    step_lo = ext_err
    if has_lower_limit(par)
        step_lo = min(ext_err, ext_min - par.lower)
    end
    step_lo = max(step_lo, 0.0)

    # ── MnMinos linear-correlation pre-shift in EXT coords (C++
    # MnMinos.cxx:136-165). The bounded path operates in external
    # coords, so we compute the shift in INTERNAL (using the internal
    # inv_hessian), then Int2ext + EXT clamp per "other" free param,
    # producing a full length-n_total seed vector to hand to
    # `function_cross_external`. Mirrors C++ exactly:
    #     internal: xdev = xunit · m[ind,i]; xnew = xt[i] + dir · xdev
    #     ext:      unew = Int2ext(i, xnew); clamp; SetValue
    # Fixed parameters keep their converged ext value (the inner MIGRAD
    # never touches them).
    int_state = bfm.internal.state
    V_int = int_state.error.inv_hessian
    n_total = n_pars(bfm.params)
    n_free_int = n_free(bfm.params)
    ind_int = bfm.params.int_of_ext[par_idx]      # internal index of scanned
    seed_up_ext = nothing
    seed_lo_ext = nothing
    if 1 <= ind_int <= n_free_int && isfinite(V_int[ind_int, ind_int]) &&
            V_int[ind_int, ind_int] > 0
        sigma_int_ind = sqrt(max(2.0 * cf.up * V_int[ind_int, ind_int],
                                  prec.eps2))
        seed_up_ext = Vector{Float64}(undef, n_total)
        seed_lo_ext = Vector{Float64}(undef, n_total)
        # Sin-transform saturation limits for BothBounds pre-clamp
        # (review v2 IMPORTANT B). See src/minos.jl for the rationale —
        # same aliasing pathology applies to the bounded path.
        piby2 = 2.0 * atan(1.0)
        distnn_int = 8.0 * sqrt(prec.eps2)
        vlimhi_int = piby2 - distnn_int
        vlimlo_int = -piby2 + distnn_int
        @inbounds for ext_i in 1:n_total
            p_i = bfm.params.pars[ext_i]
            if is_fixed(p_i) || ext_i == par_idx
                seed_up_ext[ext_i] = bfm.ext_values[ext_i]
                seed_lo_ext[ext_i] = bfm.ext_values[ext_i]
                continue
            end
            i_int = bfm.params.int_of_ext[ext_i]
            shift_int = sigma_int_ind * (V_int[ind_int, i_int] /
                                          V_int[ind_int, ind_int])
            kind = bound_kind(p_i)
            xt_i_int = int_state.parameters.x[i_int]
            xnew_up_int = xt_i_int + shift_int        # dir = +1
            xnew_lo_int = xt_i_int - shift_int        # dir = -1
            # Pre-clamp INT for BothBounds before Int2ext to avoid
            # sin() aliasing on large linear pre-shifts.
            if kind == BothBounds
                xnew_up_int = clamp(xnew_up_int, vlimlo_int, vlimhi_int)
                xnew_lo_int = clamp(xnew_lo_int, vlimlo_int, vlimhi_int)
            end
            unew_up = int2ext(kind, xnew_up_int, p_i.lower, p_i.upper)
            unew_lo = int2ext(kind, xnew_lo_int, p_i.lower, p_i.upper)
            if has_upper_limit(p_i)
                unew_up = min(unew_up, p_i.upper)
                unew_lo = min(unew_lo, p_i.upper)
            end
            if has_lower_limit(p_i)
                unew_up = max(unew_up, p_i.lower)
                unew_lo = max(unew_lo, p_i.lower)
            end
            seed_up_ext[ext_i] = unew_up
            seed_lo_ext[ext_i] = unew_lo
        end
    end

    # gap M1: outer-guarded to avoid the @sprintf String alloc at level 0
    # (this helper is called per bounded-MINOS parameter request).
    if print_level >= 1
        _trace_info(print_level, "MnMinos",
                    @sprintf("Determination of upper error for par=%d (value=%.10g)",
                              par_idx, ext_min))
    end
    cr_up = function_cross_external(bfm, cf, par_idx, +1.0;
                                     tlr = tlr, maxcalls = maxcalls,
                                     strategy = strategy, prec = prec,
                                     threaded_gradient = threaded_gradient,
                                     sigma = sigma,
                                     print_level = print_level,
                                     other_param_seed_ext = seed_up_ext)
    if print_level >= 1
        _trace_info(print_level, "MnMinos",
                    @sprintf("Determination of lower error for par=%d (value=%.10g)",
                              par_idx, ext_min))
    end
    cr_lo = function_cross_external(bfm, cf, par_idx, -1.0;
                                     tlr = tlr, maxcalls = maxcalls,
                                     strategy = strategy, prec = prec,
                                     threaded_gradient = threaded_gradient,
                                     sigma = sigma,
                                     print_level = print_level,
                                     other_param_seed_ext = seed_lo_ext)

    # External errors. Cases per side:
    #   - search succeeded (valid)    → aopt · step (the asymmetric error)
    #   - search hit a bound (par_limit) → publish `bound − ext_min` (the
    #       physical distance from minimum to the constraining bound).
    #       Matches C++ MinosError::Upper() and iminuit's `m.merrors[].upper`
    #       semantics: "the parameter can move at most this much in this
    #       direction before hitting the bound."
    #   - other failure (fcn_limit, algorithmic invalid, etc.) → ±ext_err
    #       (the HESSE 1σ symmetric placeholder), mirroring C++
    #       MinosError::Upper/Lower (MinosError.h:54) which return
    #       `±State().Error(Parameter())` when invalid. Consistent with
    #       the unbounded MINOS path (src/minos.jl). Consumers MUST gate
    #       on `e.upper_valid`/`e.lower_valid` to distinguish real
    #       crossings from placeholders.
    # Sign convention: upper_err ≥ 0 by construction; lower_err ≤ 0.
    upper_err = if cr_up.valid
        cr_up.aopt * step_up
    elseif cr_up.par_limit
        par.upper - ext_min            # bound_distance (positive)
    else
        ext_err                        # ±σ_HESSE placeholder
    end
    lower_err = if cr_lo.valid
        -cr_lo.aopt * step_lo
    elseif cr_lo.par_limit
        par.lower - ext_min            # bound_distance (negative)
    else
        -ext_err                       # ±σ_HESSE placeholder
    end

    # M4: full ext-coord snapshot at the ±σ crossing. The bounded path
    # gets these from `MnCross.ext_state` (populated by
    # `function_cross_external`'s probe-Ref capture). `nothing` when no
    # valid inner BFM was ever reached on that side. The at-bound
    # `par_limit` case captures the snapshot from the last
    # truncated-but-valid probe — physically the "state at the bound".
    upper_state = cr_up.ext_state
    lower_state = cr_lo.ext_state

    # `upper_valid`/`lower_valid` lifted: clean crossing OR at-limit
    # both count as "MINOS analysis completed". Matches iminuit's
    # m.merrors[name].is_valid semantics (saturating against a bound is
    # a legitimate termination — the published bound_distance is a
    # physically meaningful value, not a failure indicator). The
    # MnCross-level `valid` stays C++-faithful (false at par_limit);
    # we lift the semantics only at the user-facing MinosError layer.
    # par_limit and fcn_limit remain PER-SIDE and DISTINGUISHABLE
    # (round-3 I-4): par_limit = "hit a bound", fcn_limit = "budget".
    return MinosError(par_idx, ext_min,
                       upper_err, lower_err,
                       cr_up.valid || cr_up.par_limit,
                       cr_lo.valid || cr_lo.par_limit,
                       cr_up.new_min, cr_lo.new_min,
                       cr_up.fcn_limit, cr_lo.fcn_limit,
                       cr_up.par_limit, cr_lo.par_limit,
                       cr_up.nfcn + cr_lo.nfcn,
                       upper_state, lower_state)
end
function minos!(m::Minuit, par_name::AbstractString; kwargs...)
    par_idx = ext_index(m.params, String(par_name))
    return minos!(m, par_idx; kwargs...)
end

"""
    minos!(m::Minuit; kwargs...) -> Minuit

Run MINOS on all free parameters.
"""
function minos!(m::Minuit; kwargs...)
    for ext_idx in 1:n_pars(m.params)
        is_fixed(m.params.pars[ext_idx]) && continue
        minos!(m, ext_idx; kwargs...)
    end
    return m
end

"""
    contour(m::Minuit, par_x, par_y; npoints=20, bins=nothing, kwargs...) -> ContoursError

Compute a 2D contour. `par_x` / `par_y` may be Integer or String.
The `bins=...` kwarg is an IMinuit.jl-compatible alias for `npoints`
(takes precedence when both are passed).
"""
function contour(m::Minuit, par_x::Integer, par_y::Integer;
                  npoints::Integer = 20,
                  bins::Union{Integer,Nothing} = nothing,
                  threaded_gradient::Bool = m.threaded_gradient,
                  kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `contour(m, ...)`"))
    npts = bins === nothing ? Int(npoints) : Int(bins)
    ix = m.params.int_of_ext[par_x]
    iy = m.params.int_of_ext[par_y]
    # Use the internal-coord-wrapped CostFunction (parallel-review #4
    # A7/B4 — see minos! for the rationale).
    return contour(m.fmin.internal, m.fmin.internal_cf, ix, iy;
                    npoints = npts,
                    threaded_gradient = threaded_gradient, kwargs...)
end

function contour(m::Minuit, px::AbstractString, py::AbstractString;
                  kwargs...)
    return contour(m, ext_index(m.params, String(px)),
                      ext_index(m.params, String(py)); kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Property-style access (iminuit copy-paste compatibility)
# ─────────────────────────────────────────────────────────────────────────────

function Base.getproperty(m::Minuit, name::Symbol)
    if name === :values
        return m.fmin === nothing ? [p.value for p in m.params.pars] :
                                     m.fmin.ext_values
    elseif name === :errors
        return m.fmin === nothing ? [p.error for p in m.params.pars] :
                                     m.fmin.ext_errors
    elseif name === :fval
        return m.fmin === nothing ? NaN : fval(m.fmin)
    elseif name === :edm
        return m.fmin === nothing ? NaN : edm(m.fmin)
    elseif name === :nfcn
        return m.fmin === nothing ? 0 : nfcn(m.fmin)
    # iminuit/IMinuit.jl alias: `ncalls`
    elseif name === :ncalls
        return m.fmin === nothing ? 0 : nfcn(m.fmin)
    elseif name === :valid
        return m.fmin === nothing ? false : is_valid(m.fmin)
    # iminuit/IMinuit.jl alias: `is_valid`
    elseif name === :is_valid
        return m.fmin === nothing ? false : is_valid(m.fmin)
    elseif name === :covariance
        return m.fmin === nothing ? nothing : ext_covariance(m.fmin)
    elseif name === :ndim
        return n_pars(m.params)
    elseif name === :npar
        return n_free(m.params)
    # ── IMinuit.jl property aliases ───────────────────────────────
    elseif name === :parameters
        # iminuit's `parameters` is a tuple of parameter names
        return Tuple(p.name for p in m.params.pars)
    elseif name === :fixed
        return [is_fixed(p) for p in m.params.pars]
    elseif name === :limits
        return [(p.lower, p.upper) for p in m.params.pars]
    elseif name === :errordef
        return m.fcn.up
    elseif name === :up
        return m.fcn.up
    elseif name === :merrors
        # iminuit's MINOS errors dict, keyed by parameter name
        out = Dict{String,MinosError}()
        for (i, e) in m.minos_errors
            out[m.params.pars[i].name] = e
        end
        return out
    elseif name === :accurate
        # iminuit's `m.accurate` ≡ "covariance is reliable"
        return m.fmin === nothing ? false :
               (is_valid(m.fmin) && !m.fmin.internal.made_pos_def)
    elseif name === :matrix
        # IMinuit.jl-compatible: `m.matrix` returns the external
        # covariance matrix as a `Matrix{Float64}` (free-parameter
        # block, matching IMinuit.jl's `f.matrix` getproperty hook).
        # `nothing` if MIGRAD hasn't run or no covariance available.
        return matrix(m)
    elseif name === :nfit
        # iminuit-compatible: total degrees of free parameters
        return n_free(m.params)
    elseif name === :ngrad
        # IMinuit.jl/iminuit-compatible: gradient call counter (only
        # nonzero when `grad=...` was supplied)
        return m.cfwg === nothing ? 0 : m.cfwg.ngrad[]
    else
        return getfield(m, name)
    end
end

# Settable iminuit-compatible properties. Lets users tune
# `m.strategy`, `m.tol`, `m.print_level`, `m.errordef`/`m.up`,
# `m.values`, `m.errors`, `m.limits`, `m.fixed` between fits without
# rebuilding the Minuit object.
function Base.setproperty!(m::Minuit, name::Symbol, val)
    if name === :strategy
        setfield!(m, :strategy, val isa Strategy ? val : Strategy(Int(val)))
    elseif name === :tol
        setfield!(m, :tol, Float64(val))
    elseif name === :print_level
        setfield!(m, :print_level, Int(val))
    elseif name === :errordef || name === :up
        # Mutates the underlying CostFunction.up — both `fcn` and (if
        # present) `cfwg` need to be re-wrapped because `up` is stored
        # in the struct rather than fetched on demand. Use the
        # original `f` (and `g`) so the user's closures survive.
        new_up = Float64(val)
        setfield!(m, :fcn, CostFunction(m.fcn.f, new_up, m.fcn.nfcn))
        if m.cfwg !== nothing
            setfield!(m, :cfwg,
                CostFunctionWithGradient(m.cfwg.f, m.cfwg.g, new_up,
                                          m.cfwg.nfcn, m.cfwg.ngrad))
        end
    elseif name === :values
        # iminuit-style `m.values = [...]`: replaces the per-parameter
        # initial values. Routed through `set_value!` so the cache-
        # invalidation semantics are guaranteed identical to the
        # per-parameter mutator (review IMPORTANT round-3).
        _bulk_set_values!(m, val)
    elseif name === :errors
        _bulk_set_errors!(m, val)
    elseif name === :limits
        _bulk_set_limits!(m, val)
    elseif name === :fixed
        _bulk_set_fixed!(m, val)
    else
        setfield!(m, name, val)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter mutators (gap M3) — mirror C++ `MnUserParameters` methods.
#
# C++ refs:
#   reference/Minuit2_cpp/inc/Minuit2/MnUserParameters.h:75-95
#   reference/Minuit2_cpp/src/MnApplication.cxx:117-180
#
# Each mutator accepts an `Integer` (1-based external index) OR an
# `AbstractString` (parameter name → `ext_index` lookup), rebuilds the
# single touched `MinuitParameter` in place, then invalidates the
# cached fit (`m.fmin = nothing`, `empty!(m.minos_errors)`) — matching
# the same staleness rule the bulk `setproperty!` paths use. The bulk
# `m.values=...` / `m.errors=...` / `m.fixed=...` / `m.limits=...`
# setters route through these mutators so behavior cannot drift.
#
# Returns `m` for chaining: `m |> migrad! |> (m -> fix!(m,"alpha")) |> migrad!`.
# ─────────────────────────────────────────────────────────────────────────────

# Validate `i` is a usable external index; throw `BoundsError` otherwise.
# Rejects `Bool` explicitly — `Bool <: Integer` in Julia, so without this
# guard `fix!(m, true)` would dispatch into the Integer method and the
# error would only surface from `Base.checkbounds` one line later with
# a confusing "invalid index: true of type Bool" message.
@inline function _check_par_index(m::Minuit, i::Integer)
    i isa Bool &&
        throw(ArgumentError("parameter index must be an Integer, got Bool"))
    1 <= i <= n_pars(m.params) ||
        throw(BoundsError(m.params.pars, i))
    return nothing
end

# Replace the entire `pars` vector, rebuild `Parameters` (so
# `ext_of_int` / `int_of_ext` / `name_to_ext` caches stay consistent),
# and drop any cached fit. Single atomic commit point — both
# per-parameter and bulk setters land here.
function _replace_all_params!(m::Minuit, new_pars::Vector{MinuitParameter})
    setfield!(m, :params, Parameters(new_pars, m.prec))
    setfield!(m, :fmin, nothing)
    empty!(m.minos_errors)
    return m
end

# Per-parameter pivot: clone the current vector, swap one entry, commit.
function _replace_one_param!(m::Minuit, i::Int, new_p::MinuitParameter)
    new_pars = collect(m.params.pars)
    new_pars[i] = new_p
    return _replace_all_params!(m, new_pars)
end

# Normalize a user-supplied bound spec to the `MinuitParameter` storage
# convention (NaN = absent). `nothing` and `±Inf` both mean "no bound",
# matching the existing bulk-limits setter behavior.
_normalize_bound(::Nothing) = NaN
function _normalize_bound(x::Real)
    xf = Float64(x)
    return (isnan(xf) || isinf(xf)) ? NaN : xf
end

# ── Builder helpers (validate + construct, no commit) ────────────────────────
#
# Each `_build_*_par(p, ...)` returns a NEW `MinuitParameter` derived from
# `p` with the requested field updated. Validation throws here (NaN for
# value, negative/NaN for error, lo >= up for limits via the
# `MinuitParameter` ctor). Used by both per-parameter mutators (one
# build, one commit) and bulk setters (build N first, commit once, so
# any single-element failure leaves `m` unchanged — exception atomicity).
function _build_value_par(p::MinuitParameter, v::Real)
    vf = Float64(v)
    isfinite(vf) ||
        throw(ArgumentError("set_value!: value must be finite, got $v"))
    return MinuitParameter(p.name, vf, p.error;
                             lower = p.lower, upper = p.upper, fixed = p.fixed)
end

function _build_error_par(p::MinuitParameter, e::Real)
    ef = Float64(e)
    (isfinite(ef) && ef >= 0) ||
        throw(ArgumentError("set_error!: step must be finite and ≥ 0, got $e"))
    return MinuitParameter(p.name, p.value, ef;
                             lower = p.lower, upper = p.upper, fixed = p.fixed)
end

function _build_fixed_par(p::MinuitParameter, fix::Bool)
    return MinuitParameter(p.name, p.value, p.error;
                             lower = p.lower, upper = p.upper, fixed = fix)
end

function _build_limits_par(p::MinuitParameter, lo, up)
    return MinuitParameter(p.name, p.value, p.error;
                             lower = _normalize_bound(lo),
                             upper = _normalize_bound(up),
                             fixed = p.fixed)
end

"""
    fix!(m::Minuit, par::Union{Integer,AbstractString}) -> Minuit

Mark parameter `par` as fixed (excluded from optimization). Mirrors
C++ `MnUserParameters::Fix(i)` / `Fix(name)`. Drops `m.fmin` and
clears `m.minos_errors`. Returns `m` for chaining.

`par` is either the 1-based external index or the parameter name.
Already-fixed parameters are still re-fixed (no-op on the fixed flag,
but cache is still invalidated for consistency).

```julia
fix!(m, 1)
fix!(m, "alpha")
```
"""
fix!(m::Minuit, par::AbstractString) =
    fix!(m, ext_index(m.params, String(par)))
function fix!(m::Minuit, i::Integer)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_fixed_par(m.params.pars[i], true))
end

"""
    release!(m::Minuit, par::Union{Integer,AbstractString}) -> Minuit

Clear parameter `par`'s fixed flag (re-include in optimization).
Mirrors C++ `MnUserParameters::Release(i)` / `Release(name)`. Drops
`m.fmin` and clears `m.minos_errors`. Returns `m` for chaining.

Pairs with [`fix!`](@ref) for fix-fit-release-fit profile-likelihood
scans:

```julia
fix!(m, "alpha"); migrad!(m); release!(m, "alpha"); migrad!(m)
```
"""
release!(m::Minuit, par::AbstractString) =
    release!(m, ext_index(m.params, String(par)))
function release!(m::Minuit, i::Integer)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_fixed_par(m.params.pars[i], false))
end

"""
    set_value!(m::Minuit, par::Union{Integer,AbstractString}, v::Real) -> Minuit

Set parameter `par`'s initial value to `v`. Mirrors C++
`MnUserParameters::SetValue(i, v)` / `SetValue(name, v)`. Drops
`m.fmin` and clears `m.minos_errors`. Returns `m` for chaining.

`v` must be finite (NaN / ±Inf throw `ArgumentError`) — matches the
iminuit Python wrapper's `setattr` guard. The int↔ext transform clamps
to bounds at minimization time, so no value-vs-limits check is done
here (matches C++).
"""
set_value!(m::Minuit, par::AbstractString, v::Real) =
    set_value!(m, ext_index(m.params, String(par)), v)
function set_value!(m::Minuit, i::Integer, v::Real)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_value_par(m.params.pars[i], v))
end

"""
    set_error!(m::Minuit, par::Union{Integer,AbstractString}, e::Real) -> Minuit

Set parameter `par`'s step size to `e`. Mirrors C++
`MnUserParameters::SetError(i, e)` / `SetError(name, e)`. Drops
`m.fmin` and clears `m.minos_errors`. Returns `m` for chaining.

`e` must be finite and non-negative (NaN / ±Inf / negative throw
`ArgumentError`). A non-positive step would degrade the numerical
gradient floor and could silently poison the seed.
"""
set_error!(m::Minuit, par::AbstractString, e::Real) =
    set_error!(m, ext_index(m.params, String(par)), e)
function set_error!(m::Minuit, i::Integer, e::Real)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_error_par(m.params.pars[i], e))
end

"""
    set_limits!(m::Minuit, par, lo, up) -> Minuit

Set parameter `par`'s bounds to `(lo, up)`. Mirrors C++
`MnUserParameters::SetLimits(i, lo, up)` / `SetLimits(name, lo, up)`.
Drops `m.fmin` and clears `m.minos_errors`. Returns `m` for chaining.

`lo` and `up` may each be a `Real`, `nothing`, or `±Inf` — the latter
two are stored as `NaN` (the "absent bound" sentinel), so passing
`(nothing, 10.0)` makes `par` upper-bounded only, and
`(nothing, nothing)` is equivalent to [`remove_limits!`](@ref). When
both are finite Reals, `lo < up` is required (else `ArgumentError`).
"""
set_limits!(m::Minuit, par::AbstractString, lo, up) =
    set_limits!(m, ext_index(m.params, String(par)), lo, up)
function set_limits!(m::Minuit, i::Integer, lo, up)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_limits_par(m.params.pars[i], lo, up))
end

"""
    remove_limits!(m::Minuit, par::Union{Integer,AbstractString}) -> Minuit

Clear both bounds on parameter `par`. Mirrors C++
`MnUserParameters::RemoveLimits(i)` / `RemoveLimits(name)`. Drops
`m.fmin` and clears `m.minos_errors`. Returns `m` for chaining.

After the call, `m.params.pars[par].lower` and `.upper` are both `NaN`.
"""
remove_limits!(m::Minuit, par::AbstractString) =
    remove_limits!(m, ext_index(m.params, String(par)))
function remove_limits!(m::Minuit, i::Integer)
    _check_par_index(m, i)
    return _replace_one_param!(m, Int(i),
        _build_limits_par(m.params.pars[i], nothing, nothing))
end

# ─────────────────────────────────────────────────────────────────────────────
# Bulk setters — share the per-parameter `_build_*_par` validation +
# construction helpers, then commit ALL changes in a single
# `_replace_all_params!` call. This preserves exception-atomicity (if
# any element's validation fails, `m` is untouched — matches the pre-M3
# semantics) AND guarantees identical validation rules to the
# per-parameter mutators above (a single point of truth via the build
# helpers).
# ─────────────────────────────────────────────────────────────────────────────

function _bulk_set_values!(m::Minuit, vals::AbstractVector)
    n = n_pars(m.params)
    length(vals) == n ||
        throw(DimensionMismatch("expected $n values, got $(length(vals))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        new_pars[i] = _build_value_par(m.params.pars[i], vals[i])
    end
    _replace_all_params!(m, new_pars)
    return nothing
end

function _bulk_set_errors!(m::Minuit, errs::AbstractVector)
    n = n_pars(m.params)
    length(errs) == n ||
        throw(DimensionMismatch("expected $n values, got $(length(errs))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        new_pars[i] = _build_error_par(m.params.pars[i], errs[i])
    end
    _replace_all_params!(m, new_pars)
    return nothing
end

function _bulk_set_fixed!(m::Minuit, fx::AbstractVector)
    n = n_pars(m.params)
    length(fx) == n ||
        throw(DimensionMismatch("expected $n fixed flags, got $(length(fx))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        new_pars[i] = _build_fixed_par(m.params.pars[i], Bool(fx[i]))
    end
    _replace_all_params!(m, new_pars)
    return nothing
end

function _bulk_set_limits!(m::Minuit, lim::AbstractVector)
    n = n_pars(m.params)
    length(lim) == n ||
        throw(DimensionMismatch("expected $n limit tuples, got $(length(lim))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        l = lim[i]
        new_pars[i] = if l === nothing
            _build_limits_par(m.params.pars[i], nothing, nothing)
        else
            lo_raw, up_raw = l
            _build_limits_par(m.params.pars[i], lo_raw, up_raw)
        end
    end
    _replace_all_params!(m, new_pars)
    return nothing
end

function Base.propertynames(m::Minuit, ::Bool = false)
    return (:fcn, :params, :fmin, :minos_errors, :prec, :cfwg,
            :strategy, :tol, :print_level, :n_passes,
            # JuMinuit-native
            :values, :errors, :fval, :edm, :nfcn, :valid,
            :covariance, :ndim, :npar,
            # IMinuit.jl-compatible aliases
            :ncalls, :is_valid, :parameters, :fixed, :limits,
            :errordef, :up, :merrors, :accurate,
            :matrix, :nfit, :ngrad)
end

# ─────────────────────────────────────────────────────────────────────────────
# IMinuit.jl-compatible no-bang method aliases.
#
# In IMinuit.jl (which wraps Python iminuit), the convention is that
# `migrad(f)` mutates `f` in place and returns it. JuMinuit's native
# style uses `migrad!(m)` (Julia idiom). The aliases below let
# existing IMinuit.jl code run unchanged.
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad(m::Minuit; ncall=nothing, resume=true, precision=nothing,
                       strategy=m.strategy, tol=m.tol,
                       iterate=5, use_simplex=true) -> Minuit

IMinuit.jl-compatible alias for [`migrad!`](@ref). Mutates `m.fmin`
and returns `m`. The `ncall` / `resume` / `precision` kwargs are
accepted for IMinuit.jl interface parity:

  - `ncall::Union{Integer,Nothing}` ≡ `maxfcn` cap (default uses
    JuMinuit's `200 + 100·n + 5·n²` formula).
  - `resume::Bool=true` — if `false`, reset `m.fmin` and `m.minos_errors`
    before running (matches iminuit's `resume` argument).
  - `precision::Union{Real,Nothing}` — override the `MachinePrecision`
    `eps` value (rarely used).

`strategy` and `tol` default to whatever the user stored on `m`
(settable via `m.strategy = ...`, `m.tol = ...`). Constructor default
is `Strategy(0)` — faster than iminuit's `Strategy(1)`; if you want
the iminuit-matching accuracy/cost trade pass `strategy=Strategy(1)`
or set `m.strategy = Strategy(1)` once before the first migrad.

`iterate` and `use_simplex` are threaded through to [`migrad!`](@ref)
unchanged — see its docstring for the iminuit
`_robust_low_level_fit`-shaped retry loop they control. Note that
retries silently upgrade to `Strategy(2)` (iminuit heuristic) on the
numerical-gradient path, even if you passed `strategy=Strategy(0)`
here. The AD-gradient path retains the user's strategy because its
seed rejects `strategy.level != 0`.
"""
function migrad(m::Minuit;
                 ncall::Union{Integer,Nothing} = nothing,
                 resume::Bool = true,
                 precision::Union{Real,Nothing} = nothing,
                 strategy::Strategy = m.strategy,
                 tol::Real = m.tol,
                 iterate::Integer = 5,
                 use_simplex::Bool = true)
    if !resume
        # Equivalent to IMinuit.jl `reset(m)`: drop any prior fmin/minos.
        m.fmin = nothing
        empty!(m.minos_errors)
    end
    if precision !== nothing
        m.prec = MachinePrecision(Float64(precision))
    end
    return migrad!(m; strategy = strategy, tol = tol, maxfcn = ncall,
                       iterate = iterate, use_simplex = use_simplex)
end

"""
    hesse(m::Minuit; strategy=Strategy(1), maxcall=0) -> Minuit

IMinuit.jl-compatible: re-run a full numerical HESSE at the current
converged minimum to refresh the covariance matrix.

Typical use: a fast Strategy(0) MIGRAD leaves the inverse-Hessian
as a DFP approximation, which is usually accurate but can drift in
ill-conditioned valleys. `hesse(m)` recomputes the full 2nd-derivative
Hessian numerically (mirrors `MnHesse` invoked standalone in C++)
and updates `m.fmin` in place with the refined `ext_covariance` and
`ext_errors`.

Strategy(1) is the iminuit default for HESSE; Strategy(2) is more
accurate but slower. The `maxcall` argument is accepted for IMinuit.jl
parity but currently unused (the HESSE implementation has its own
budget logic).

Internally:
  1. Take `m.fmin.internal.state` (the converged internal-coord state).
  2. Call `JuMinuit.hesse(cf_internal, state, strategy)` to refresh
     `state.error.inv_hessian` via numerical 2nd derivatives.
  3. Re-run the same int→ext Jacobian + `Int2extError` machinery used
     in `migrad(cf, params)` to rebuild `ext_covariance` + `ext_errors`.
  4. Wrap a fresh `BoundedFunctionMinimum` and overwrite `m.fmin`.

Returns `m` for chaining.
"""
function hesse(m::Minuit; strategy::Strategy = Strategy(1),
                           maxcall::Integer = 0,
                           print_level::Integer = m.print_level)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `hesse(m)`"))
    bfm = m.fmin

    # Refresh the internal-coord Hessian.
    new_state = JuMinuit.hesse(bfm.internal_cf, bfm.internal.state, strategy;
                                 prec = m.prec,
                                 print_level = print_level)

    # Wrap into a fresh FunctionMinimum reflecting the CURRENT covariance
    # state, not the union of historical states. iminuit's semantics is
    # "the cov is whatever HESSE just produced" — a successful HESSE
    # MUST be able to clear an earlier `made_pos_def` / `hesse_failed` /
    # `is_valid=false` flag, otherwise users can never recover state
    # after a transient setback without re-migrad'ing.
    #
    # `reached_call_limit` and `above_max_edm` ARE genuinely sticky
    # (they describe the MIGRAD convergence run that led to this state)
    # — keep them.
    hesse_now_failed = JuMinuit.hesse_failed(new_state.error) ||
                        JuMinuit.invert_failed(new_state.error)
    made_pos_def_now = JuMinuit.is_made_pos_def(new_state.error)
    # `is_valid` is recomputed from the new HESSE outcome (covariance
    # validity per `is_valid(error)`) AND the genuinely sticky MIGRAD
    # convergence flags (`reached_call_limit`, `above_max_edm` describe
    # how MIGRAD ran, not the current covariance — those don't change
    # under hesse).
    new_err_valid = JuMinuit.is_valid(new_state.error)
    new_is_valid  = new_err_valid &&
                    !bfm.internal.reached_call_limit &&
                    !bfm.internal.above_max_edm
    new_fmin_int = FunctionMinimum(new_state, bfm.internal.seed,
                                     bfm.internal.up;
                                     is_valid = new_is_valid,
                                     reached_call_limit = bfm.internal.reached_call_limit,
                                     above_max_edm = bfm.internal.above_max_edm,
                                     hesse_failed = hesse_now_failed,
                                     made_pos_def = made_pos_def_now)

    # Rebuild external view via the shared helper. Use bfm.internal.up
    # (the value attached to the internal-coord FM, set at migrad time)
    # rather than m.fcn.up — these are normally equal but m.fcn is
    # mutable in principle; bfm.internal.up is the value the internal
    # state actually corresponds to.
    ext_values, ext_errors_vec, ext_cov_mat =
        JuMinuit._internal_to_external_results(new_fmin_int, bfm.params,
                                                bfm.internal.up)

    m.fmin = BoundedFunctionMinimum(
        new_fmin_int, bfm.params, ext_values, ext_errors_vec, ext_cov_mat,
        bfm.internal_cf,
    )
    return m
end

"""
    minos(m::Minuit, var=nothing; sigma=1, maxcall=0, kwargs...) -> Minuit

IMinuit.jl-compatible alias for [`minos!`](@ref). When `var` is `nothing`,
runs MINOS on all free parameters. `var` may be an integer index, a
String/Symbol name, or a `Vector` of either.

The `sigma` kwarg (confidence level in σ-units) is threaded through
the MnFunctionCross `up · sigma²` scaling (P5 — see
[`function_cross`](@ref) for details). At sigma=1 the behavior is
C++-MnMinos-identical; at sigma=k the upper/lower errors correspond
to the k-σ contour. `maxcall` is accepted for IMinuit.jl parity but
currently unused.
"""
function minos(m::Minuit, var = nothing;
                sigma::Real = 1, maxcall::Integer = 0, kwargs...)
    sigma > 0 ||
        throw(ArgumentError("MINOS sigma must be positive, got $sigma"))
    if var === nothing
        return minos!(m; sigma = sigma, kwargs...)
    elseif var isa Integer
        return minos!(m, Int(var); sigma = sigma, kwargs...)
    elseif var isa AbstractString || var isa Symbol
        return minos!(m, String(var); sigma = sigma, kwargs...)
    elseif var isa AbstractVector
        for v in var
            minos(m, v; sigma = sigma, maxcall = maxcall, kwargs...)
        end
        return m
    else
        throw(ArgumentError("Unsupported `var` type for MINOS: $(typeof(var))"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# IMinuit.jl helpers: args(m), matrix(m), reset(m), set_precision(m, p).
# ─────────────────────────────────────────────────────────────────────────────

"""
    args(m::Minuit) -> Vector{Float64}

IMinuit.jl-compatible convenience: returns the current parameter
values as a `Vector{Float64}`. Equivalent to `m.values`.
"""
args(m::Minuit) = collect(Float64, m.values)

"""
    matrix(m::Minuit; correlation=false, skip_fixed=true) -> Matrix{Float64}

IMinuit.jl-compatible covariance matrix accessor.

- `correlation=false` (default): returns the external covariance.
- `correlation=true`: returns the correlation matrix `C[i,j] = V[i,j] /
  √(V[i,i]·V[j,j])`.
- `skip_fixed=true` (default): returns the n_free × n_free submatrix
  (the `free_covariance` shape, matching C++ `MnUserParameterState`).
- `skip_fixed=false`: returns the full n_total × n_total matrix with
  zero rows + cols for fixed parameters.

Returns `nothing` if MIGRAD hasn't been called or the covariance is
unavailable.
"""
function matrix(m::Minuit; correlation::Bool = false, skip_fixed::Bool = true)
    m.fmin === nothing && return nothing
    V = skip_fixed ? free_covariance(m.fmin) : ext_covariance(m.fmin)
    V === nothing && return nothing
    Vmat = collect(V)   # ensure Matrix{Float64}, not Symmetric{...}
    if correlation
        n = size(Vmat, 1)
        C = similar(Vmat)
        for j in 1:n, i in 1:n
            denom = sqrt(Vmat[i, i] * Vmat[j, j])
            C[i, j] = denom > 0 ? Vmat[i, j] / denom : 0.0
        end
        return C
    end
    return Vmat
end

"""
    reset(m::Minuit) -> Minuit

IMinuit.jl-compatible: drop any cached MIGRAD/MINOS results so the
next `migrad(m)` starts fresh from `m.params`'s initial values.
Extends `Base.reset` (which has unrelated methods for IO streams),
so dispatch picks the right one by argument type.
"""
function Base.reset(m::Minuit)
    m.fmin = nothing
    empty!(m.minos_errors)
    return m
end

"""
    set_precision(m::Minuit, p::Real) -> Minuit

IMinuit.jl-compatible: override the floating-point precision used by
MIGRAD/HESSE/MINOS. The default `MachinePrecision()` is `eps(Float64)`;
override only when fitting with synthetic-precision FCN models.
"""
function set_precision(m::Minuit, p::Real)
    m.prec = MachinePrecision(Float64(p))
    return m
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

# ── Helpers for pretty-print (Phase 3 C1 polish) ─────────────────────────────

"""
    _at_limit_indices(m::Minuit; n_sigma=1.0) -> Vector{Int}

Return external indices of parameters whose converged value sits
within `n_sigma · Hesse_err` of one of their explicit limits. That's
the iminuit-style "the limit is within 1σ of the fit value" test —
when it's true the Hesse/MINOS error is suspect because the sin/sqrt
transform's Jacobian collapses near the boundary, and the 1σ
contour gets cut off by the limit.

If the Hesse error is zero or NaN (e.g., before HESSE has converged),
falls back to `0.01 × |range|` so the detector still flags clearly
saturated parameters.
"""
function _at_limit_indices(m::Minuit; n_sigma::Real = 1.0)
    out = Int[]
    m.fmin === nothing && return out
    @inbounds for (i, p) in enumerate(m.params.pars)
        is_fixed(p) && continue
        v = m.values[i]
        e = m.errors[i]
        # Use 1σ if available, else fall back to 1% of the bound range.
        δ = if isfinite(e) && e > 0
            n_sigma * e
        elseif has_limits(p)
            0.01 * (p.upper - p.lower)
        else
            0.01 * max(1.0, abs(v))
        end
        hit_lower = (has_limits(p) || has_lower_limit(p)) &&
                    (v - p.lower) < δ
        hit_upper = (has_limits(p) || has_upper_limit(p)) &&
                    (p.upper - v) < δ
        (hit_lower || hit_upper) && push!(out, i)
    end
    return out
end

# Format a Float64 for the pretty-print table. Uses 4 significant
# digits by default; "─" placeholder for non-applicable cells (fixed
# params' error, missing MINOS, etc.).
_fmt_cell(::Nothing) = "─"
_fmt_cell(x::Float64) = isnan(x) ? "─" : (@sprintf "%.4g" x)
_fmt_cell(x::Real) = _fmt_cell(Float64(x))

# Per-parameter row tuple for the table. Shared by text/plain and HTML
# renderers (Phase 3 C1 (b) + (c)).
function _param_row_data(m::Minuit, i::Int)
    p = m.params.pars[i]
    fixed = is_fixed(p)
    value = m.values[i]
    hesse_err = fixed ? nothing : m.errors[i]
    minos_lo = nothing
    minos_hi = nothing
    if haskey(m.minos_errors, i)
        me = m.minos_errors[i]
        # New semantics: `lower_valid`/`upper_valid` are TRUE also at
        # par_limit (the bound_distance is physically meaningful), so
        # this single test covers both clean-crossing and at-limit.
        minos_lo = me.lower_valid ? me.lower : nothing
        minos_hi = me.upper_valid ? me.upper : nothing
    end
    limit_lo = has_lower_limit(p) ? p.lower : nothing
    limit_hi = has_upper_limit(p) ? p.upper : nothing
    return (idx = i, name = p.name, value = value,
            hesse = hesse_err, minos_lo = minos_lo, minos_hi = minos_hi,
            limit_lo = limit_lo, limit_hi = limit_hi, fixed = fixed)
end

# Status line: "Valid ✓" or "INVALID ✗", with key diagnostic bits
# only when relevant (we don't show "Below call limit ✓" because
# that's the default; same for hesse-ok).
function _status_summary(m::Minuit)
    m.fmin === nothing && return "not yet minimized"
    bits = String[]
    push!(bits, m.is_valid ? "Valid ✓" : "INVALID ✗")
    bfm = m.fmin
    bfm.internal.reached_call_limit && push!(bits, "call-limit ✗")
    bfm.internal.above_max_edm        && push!(bits, "EDM-above-max ✗")
    bfm.internal.hesse_failed         && push!(bits, "Hesse failed ✗")
    bfm.internal.made_pos_def         && push!(bits, "force-PosDef")
    return join(bits, "  ")
end

# ─────────────────────────────────────────────────────────────────────────────
# text/plain — Unicode box-drawn table (Phase 3 C1 (b))
# ─────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", m::Minuit)
    if m.fmin === nothing
        println(io, "JuMinuit.Minuit  ── not yet minimized; call `migrad(m)` ──")
        println(io, "  parameters (initial):")
        for (i, p) in enumerate(m.params.pars)
            fixed_tag = is_fixed(p) ? "  [FIXED]" : ""
            bounds = if has_limits(p)
                "  [$(p.lower), $(p.upper)]"
            elseif has_upper_limit(p)
                "  (-∞, $(p.upper)]"
            elseif has_lower_limit(p)
                "  [$(p.lower), ∞)"
            else
                ""
            end
            println(io, "    [", i, "] ", p.name, " = ", p.value,
                    " ± ", p.error, fixed_tag, bounds)
        end
        return
    end

    # Header line
    @printf(io, "JuMinuit.Minuit  fval=%.6g  edm=%.3g  nfcn=%d  %s\n",
            m.fval, m.edm, m.nfcn, _status_summary(m))

    # Build rows + compute column widths
    headers = ["#", "Name", "Value", "Hesse ±", "Minos −", "Minos +",
               "Limit −", "Limit +", "Fixed"]
    rows = [_param_row_data(m, i) for i in 1:n_pars(m.params)]
    cells = [[
        string(r.idx),
        r.name,
        _fmt_cell(r.value),
        _fmt_cell(r.hesse),
        _fmt_cell(r.minos_lo),
        _fmt_cell(r.minos_hi),
        _fmt_cell(r.limit_lo),
        _fmt_cell(r.limit_hi),
        r.fixed ? "yes" : "",
    ] for r in rows]
    widths = [maximum(length(c) for c in [headers[k]; [row[k] for row in cells]])
              for k in 1:length(headers)]
    pad = w -> w + 2

    # Top / mid / bottom border builders
    border(left, mid, right) = string(left,
        join(["─" ^ pad(w) for w in widths], mid), right)
    top    = border("┌", "┬", "┐")
    middle = border("├", "┼", "┤")
    bottom = border("└", "┴", "┘")

    row_str(cs) = string("│",
        join([" " * rpad(c, widths[k]) * " " for (k, c) in enumerate(cs)], "│"),
        "│")

    println(io, top)
    println(io, row_str(headers))
    println(io, middle)
    for row in cells
        println(io, row_str(row))
    end
    println(io, bottom)

    # At-limit warnings (Phase 3 C1 (a))
    al = _at_limit_indices(m)
    if !isempty(al)
        for i in al
            p = m.params.pars[i]
            print(io, "⚠ Parameter `", p.name, "` is at its ")
            v = m.values[i]
            # has_limits(p) is true for one-sided too (it's
            # `has_lower_limit || has_upper_limit`), so the
            # two-sided case must be tested with the explicit AND.
            side = if has_lower_limit(p) && has_upper_limit(p)
                (v - p.lower) < (p.upper - v) ? "lower" : "upper"
            elseif has_upper_limit(p)
                "upper"
            else
                "lower"
            end
            println(io, side, " limit — Hesse/MINOS error is unreliable.")
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# text/html — IJulia / Pluto notebook display (Phase 3 C1 (c))
# ─────────────────────────────────────────────────────────────────────────────

# Minimal HTML escaping. We don't pull a dep for this; the entity
# set below is what protects against parameter names containing
# `<`, `>`, `&`, `"`, or `'` from breaking the table markup (or
# enabling code injection in IJulia/Pluto notebook output).
function _html_escape(s::AbstractString)
    out = IOBuffer()
    @inbounds for c in s
        if     c == '&'  print(out, "&amp;")
        elseif c == '<'  print(out, "&lt;")
        elseif c == '>'  print(out, "&gt;")
        elseif c == '"'  print(out, "&quot;")
        elseif c == '\'' print(out, "&#39;")
        else             print(out, c)
        end
    end
    return String(take!(out))
end

function Base.show(io::IO, ::MIME"text/html", m::Minuit)
    if m.fmin === nothing
        print(io, "<div><strong>JuMinuit.Minuit</strong> ",
              "(not yet minimized; call <code>migrad(m)</code>)</div>")
        return
    end

    # Header line with status badge
    status = _status_summary(m)
    badge_color = m.is_valid ? "#1a7f37" : "#cf222e"   # GitHub green / red
    @printf(io, """<div style="font-family:monospace;font-size:0.95em">""")
    @printf(io,
        """<strong>JuMinuit.Minuit</strong>  fval=%.6g  edm=%.3g  nfcn=%d  """,
        m.fval, m.edm, m.nfcn)
    @printf(io, """<span style="color:%s;font-weight:bold">%s</span><br>""",
            badge_color, _html_escape(status))

    # Table
    headers = ["#", "Name", "Value", "Hesse ±", "Minos −", "Minos +",
               "Limit −", "Limit +", "Fixed"]
    print(io, """<table style="border-collapse:collapse;margin-top:0.5em">""")
    print(io, "<thead><tr>")
    for h in headers
        print(io, """<th style="border:1px solid #d0d7de;padding:2px 8px;background:#f6f8fa">""",
              h, "</th>")
    end
    print(io, "</tr></thead><tbody>")
    for i in 1:n_pars(m.params)
        r = _param_row_data(m, i)
        # Parameter `r.name` is user-controlled; escape it before
        # interpolating into the HTML cell. Other cells (numbers,
        # "yes"/"") are safe.
        cells = [string(r.idx), _html_escape(r.name), _fmt_cell(r.value),
                 _fmt_cell(r.hesse), _fmt_cell(r.minos_lo), _fmt_cell(r.minos_hi),
                 _fmt_cell(r.limit_lo), _fmt_cell(r.limit_hi),
                 r.fixed ? "yes" : ""]
        print(io, "<tr>")
        for c in cells
            print(io, """<td style="border:1px solid #d0d7de;padding:2px 8px">""",
                  c, "</td>")
        end
        print(io, "</tr>")
    end
    print(io, "</tbody></table>")

    # At-limit warnings
    al = _at_limit_indices(m)
    if !isempty(al)
        print(io, """<div style="color:#bf8700;margin-top:0.5em">""")
        for i in al
            p = m.params.pars[i]
            v = m.values[i]
            # has_limits(p) is true for one-sided too (it's
            # `has_lower_limit || has_upper_limit`), so the
            # two-sided case must be tested with the explicit AND.
            side = if has_lower_limit(p) && has_upper_limit(p)
                (v - p.lower) < (p.upper - v) ? "lower" : "upper"
            elseif has_upper_limit(p)
                "upper"
            else
                "lower"
            end
            print(io, "⚠ Parameter <code>", _html_escape(p.name),
                  "</code> is at its ", side,
                  " limit — Hesse/MINOS error is unreliable.<br>")
        end
        print(io, "</div>")
    end
    print(io, "</div>")
end

# Short one-line repr for `print` / inline display in Vector etc.
Base.show(io::IO, m::Minuit) =
    print(io, "Minuit(", n_pars(m.params), " params, ",
              m.fmin === nothing ? "not minimized" : "fval=$(m.fval)", ")")
