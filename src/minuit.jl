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
    fcn::CostFunction
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
                  cfwg, strat, Float64(tol), Int(print_level))
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
                            print_level = m.print_level, kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mutating methods
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad!(m::Minuit; strategy=m.strategy, tol=m.tol, maxfcn=nothing) -> Minuit

Run MIGRAD on `m`. Updates `m.fmin`. Returns `m` for chaining.
If the constructor was given `grad=...`, dispatches into the
analytical-gradient path. `strategy` and `tol` default to whatever
the user stored on `m` (settable via `m.strategy = ...`, `m.tol = ...`
or the constructor's `strategy`/`tol` kwargs).

If a prior `m.fmin` exists, the next MIGRAD starts from the previous
converged point (iminuit-compatible implicit resume). Use
[`reset`](@ref) (or `migrad(m; resume=false)`) to drop the prior fit
and restart from the constructor's initial values. `m.params` itself
is NEVER mutated — the carry-forward builds a fresh `Parameters` only
for the duration of the inner MIGRAD call.
"""
function migrad!(m::Minuit;
                  strategy::Strategy = m.strategy,
                  tol::Real = m.tol,
                  maxfcn::Union{Integer,Nothing} = nothing)
    # iminuit-style implicit resume: if we already converged once,
    # build a temporary Parameters carrying those values forward. The
    # user's m.params stays untouched so that `reset(m)` + migrad
    # returns to the constructor's initial values (review BLOCKING #2).
    params_to_use = m.fmin === nothing ? m.params : _build_resume_params(m)
    if m.cfwg !== nothing
        m.fmin = migrad(m.cfwg, params_to_use;
                         strategy = strategy, tol = tol, maxfcn = maxfcn,
                         prec = m.prec)
    else
        m.fmin = migrad(m.fcn, params_to_use;
                         strategy = strategy, tol = tol, maxfcn = maxfcn,
                         prec = m.prec)
    end
    return m
end

# Build a fresh `Parameters` with values carried forward from the last
# converged fit. The user-original m.params is NOT mutated. Errors are
# taken as `max(bfm.ext_errors[i], p_old.error)` — the post-MIGRAD
# ext_error is usually a tighter estimate, but near a sin/sqrt bound
# the C++ Int2extError formula can collapse to a value far smaller
# than the natural scale, which would seed the next MIGRAD with steps
# below the numerical-gradient threshold. The `max` floor with the
# original step protects against this regression (review BLOCKING #1).
function _build_resume_params(m::Minuit)
    bfm = m.fmin
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

"""
    minos!(m::Minuit, par; kwargs...) -> Minuit

Run MINOS for parameter `par` (integer index or String name). Updates
`m.minos_errors`. Requires `m.fmin` to be available (call `migrad!`
first). Returns `m`.
"""
function minos!(m::Minuit, par::Integer; kwargs...)
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
        m.minos_errors[Int(par)] = _minos_external_via_function_cross(
            m.fmin, m.fcn, Int(par); kwargs...)
    else
        # Unbounded — search in the user FCN's frame directly.
        # m.fmin.internal_cf == m.fcn here; m.params.int_of_ext[par]
        # == par when no params are bounded. Use the wrapped path so
        # that mixed (some bounded, some not) configurations route
        # consistently.
        err = minos(m.fmin.internal, m.fmin.internal_cf,
                    m.params.int_of_ext[par]; kwargs...)
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
    cf::CostFunction,
    par_idx::Int;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(1),
    prec::MachinePrecision = MachinePrecision(),
)
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

    cr_up = function_cross_external(bfm, cf, par_idx, +1.0;
                                     tlr = tlr, maxcalls = maxcalls,
                                     strategy = strategy, prec = prec)
    cr_lo = function_cross_external(bfm, cf, par_idx, -1.0;
                                     tlr = tlr, maxcalls = maxcalls,
                                     strategy = strategy, prec = prec)

    # External errors = aopt · (truncated ext step). Three cases per
    # side:
    #   - search succeeded (valid)    → aopt · step (the asymmetric error)
    #   - search hit a bound (par_limit) → publish `bound − ext_min` (the
    #       physical distance from minimum to the constraining bound).
    #       Matches C++ MinosError::Upper() and iminuit's `m.merrors[].upper`
    #       semantics: "the parameter can move at most this much in this
    #       direction before hitting the bound."
    #   - other failure (fcn_limit, etc.) → 0.0 (no information).
    # Sign convention: upper_err ≥ 0 by construction; lower_err ≤ 0.
    upper_err = if cr_up.valid
        cr_up.aopt * step_up
    elseif cr_up.par_limit
        par.upper - ext_min            # bound_distance (positive)
    else
        0.0
    end
    lower_err = if cr_lo.valid
        -cr_lo.aopt * step_lo
    elseif cr_lo.par_limit
        par.lower - ext_min            # bound_distance (negative)
    else
        0.0
    end

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
                       cr_up.nfcn + cr_lo.nfcn)
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
                  kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `contour(m, ...)`"))
    npts = bins === nothing ? Int(npoints) : Int(bins)
    ix = m.params.int_of_ext[par_x]
    iy = m.params.int_of_ext[par_y]
    # Use the internal-coord-wrapped CostFunction (parallel-review #4
    # A7/B4 — see minos! for the rationale).
    return contour(m.fmin.internal, m.fmin.internal_cf, ix, iy;
                    npoints = npts, kwargs...)
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
        # initial values. Any prior `m.fmin` becomes invalid (its
        # covariance is aligned with the OLD values), so we drop it.
        # Review IMPORTANT round-3.
        _set_param_field!(m, val, :value)
        setfield!(m, :fmin, nothing)
        empty!(m.minos_errors)
    elseif name === :errors
        _set_param_field!(m, val, :error)
        setfield!(m, :fmin, nothing)
        empty!(m.minos_errors)
    elseif name === :limits
        _set_param_limits!(m, val)
        setfield!(m, :fmin, nothing)
        empty!(m.minos_errors)
    elseif name === :fixed
        _set_param_fixed!(m, val)
        setfield!(m, :fmin, nothing)
        empty!(m.minos_errors)
    else
        setfield!(m, name, val)
    end
end

# Helper: update one field of every parameter from a vector of new
# values. Used by `m.values = [...]` and `m.errors = [...]`.
function _set_param_field!(m::Minuit, vals::AbstractVector, field::Symbol)
    n = n_pars(m.params)
    length(vals) == n ||
        throw(DimensionMismatch("expected $n values, got $(length(vals))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        p = m.params.pars[i]
        new_val   = field === :value ? Float64(vals[i]) : p.value
        new_err   = field === :error ? Float64(vals[i]) : p.error
        new_pars[i] = MinuitParameter(p.name, new_val, new_err;
                                       lower = p.lower, upper = p.upper,
                                       fixed = p.fixed)
    end
    setfield!(m, :params, Parameters(new_pars, m.prec))
    return nothing
end

# Helper: update bounds. `lim` is a vector of `(lo, hi)` tuples (or
# `nothing` for unbounded; or `(nothing, x)` / `(x, nothing)` for one-
# sided). Matches the iminuit `m.limits` setter shape.
function _set_param_limits!(m::Minuit, lim::AbstractVector)
    n = n_pars(m.params)
    length(lim) == n ||
        throw(DimensionMismatch("expected $n limit tuples, got $(length(lim))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        p = m.params.pars[i]
        lo, hi = NaN, NaN
        l = lim[i]
        if l !== nothing
            lo_raw, up_raw = l
            if lo_raw !== nothing && !(lo_raw isa Real && isinf(lo_raw))
                lo = Float64(lo_raw)
            end
            if up_raw !== nothing && !(up_raw isa Real && isinf(up_raw))
                hi = Float64(up_raw)
            end
        end
        new_pars[i] = MinuitParameter(p.name, p.value, p.error;
                                       lower = lo, upper = hi,
                                       fixed = p.fixed)
    end
    setfield!(m, :params, Parameters(new_pars, m.prec))
    return nothing
end

# Helper: update fixed flags from a `Vector{Bool}`.
function _set_param_fixed!(m::Minuit, fx::AbstractVector)
    n = n_pars(m.params)
    length(fx) == n ||
        throw(DimensionMismatch("expected $n fixed flags, got $(length(fx))"))
    new_pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        p = m.params.pars[i]
        new_pars[i] = MinuitParameter(p.name, p.value, p.error;
                                       lower = p.lower, upper = p.upper,
                                       fixed = Bool(fx[i]))
    end
    setfield!(m, :params, Parameters(new_pars, m.prec))
    return nothing
end

function Base.propertynames(m::Minuit, ::Bool = false)
    return (:fcn, :params, :fmin, :minos_errors, :prec, :cfwg,
            :strategy, :tol, :print_level,
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
                       strategy=m.strategy, tol=m.tol) -> Minuit

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
"""
function migrad(m::Minuit;
                 ncall::Union{Integer,Nothing} = nothing,
                 resume::Bool = true,
                 precision::Union{Real,Nothing} = nothing,
                 strategy::Strategy = m.strategy,
                 tol::Real = m.tol)
    if !resume
        # Equivalent to IMinuit.jl `reset(m)`: drop any prior fmin/minos.
        m.fmin = nothing
        empty!(m.minos_errors)
    end
    if precision !== nothing
        m.prec = MachinePrecision(Float64(precision))
    end
    return migrad!(m; strategy = strategy, tol = tol, maxfcn = ncall)
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
                           maxcall::Integer = 0)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `hesse(m)`"))
    bfm = m.fmin

    # Refresh the internal-coord Hessian.
    new_state = JuMinuit.hesse(bfm.internal_cf, bfm.internal.state, strategy;
                                 prec = m.prec)

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

The `sigma` kwarg (confidence level in σ-units) and `maxcall` are
accepted for parity but currently `sigma > 1` would require a
configurable `up·sigma²` scaling on the MnFunctionCross aim, which is
Phase 1.x deferred. `sigma == 1` (the default) is fully supported.
"""
function minos(m::Minuit, var = nothing;
                sigma::Real = 1, maxcall::Integer = 0, kwargs...)
    isapprox(sigma, 1.0) ||
        throw(ArgumentError("MINOS sigma ≠ 1 is Phase 1.x deferred; got $sigma"))
    if var === nothing
        return minos!(m; kwargs...)
    elseif var isa Integer
        return minos!(m, Int(var); kwargs...)
    elseif var isa AbstractString || var isa Symbol
        return minos!(m, String(var); kwargs...)
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
