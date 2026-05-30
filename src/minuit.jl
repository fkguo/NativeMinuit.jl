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
    AbstractFit

Supertype of every JuMinuit fit-frontend object. Currently the only
concrete subtype is [`Minuit`](@ref); the abstraction exists so that
generic user/ecosystem code can dispatch on `f::AbstractFit` (the same
way IMinuit.jl / iminuit code does — `migrad(f::AbstractFit)`,
`minos(f::AbstractFit, …)`), and so that any future alternative fit
frontend can slot in without breaking such code.

# IMinuit.jl drop-in note

IMinuit.jl exposes two concrete subtypes — `Fit` (keyword/scalar-arg
`fcn(a, b)` construction) and `ArrayFit` (vector `fcn(par)`
construction). That split is a PyCall wrapping artifact: the two need
different `PyObject` construction. JuMinuit is native Julia and always
calls the FCN as `f(::AbstractVector)` internally (the keyword
constructor wraps `fcn(a,b)` into `x -> fcn(x...)`), so the two forms
have **no behavioural difference** after construction. We therefore
provide [`Fit`](@ref) and [`ArrayFit`](@ref) as *aliases* of `Minuit`
rather than distinct types — a type that doesn't differ in behaviour
shouldn't be a distinct type in idiomatic Julia. Code annotating
`f::Fit` / `f::ArrayFit` / `f::AbstractFit` or testing `f isa Fit`
keeps working; only code that dispatches `Fit` and `ArrayFit` to
*different* methods (rare — the split was never semantic) would need a
touch-up.
"""
abstract type AbstractFit end

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
- `threaded_gradient::Union{Bool,Symbol}=false` — parallelize the
  per-coordinate numerical gradient (needs `julia -t N`). `false` (default) =
  serial; `true` = force threaded, raising `ThreadSafetyError` if the FCN is
  not thread-safe; `:auto` = thread when `nthreads()>1` and the FCN probes
  thread-safe, else warn once and fall back to serial (never throws). The
  `:auto` probe runs at most once and is memoized on the fit. No-op for AD
  (`grad=`) fits.
- `verify_threading::Bool` — for `threaded_gradient=true`, verify thread safety
  on the first gradient call (default `true` when forcing threading).

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
mutable struct Minuit <: AbstractFit
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
    # behavior identical to pre-Phase G). `:auto` adds a memoized one-shot
    # thread-safety probe (warn + serial on failure); resolved by `_use_threads`.
    threaded_gradient::Union{Bool,Symbol}
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
    # Number of data points behind the cost function, when known. Drives
    # the χ²/ndf + p-value line in the rich display, and is only physically
    # meaningful for a χ² fit (`errordef == 1`). `nothing` when the FCN is a
    # bare closure with no associated dataset; auto-populated by `model_fit`
    # from `Data.ndata`, and settable directly via `m.ndata = N`.
    ndata::Union{Int,Nothing}
    # Memoized `:auto` thread-safety probe result (`nothing` = not yet probed).
    # Computed once on first use by `_use_threads(m)` and reused by every later
    # gradient / MINOS / contour evaluation, so the probe never re-runs.
    _auto_threads::Base.RefValue{Union{Nothing,Bool}}
    # Inner constructor: defaults `_auto_threads` to an unprobed Ref so the
    # 13-positional-arg construction used by the keyword constructors keeps
    # working unchanged (defining any inner ctor suppresses the auto-generated
    # all-field one).
    function Minuit(fcn, params, fmin, minos_errors, prec, cfwg, strategy,
                    tol, print_level, threaded_gradient, verify_threading,
                    n_passes, ndata)
        return new(fcn, params, fmin, minos_errors, prec, cfwg, strategy, tol,
                   print_level, threaded_gradient, verify_threading, n_passes,
                   ndata, Ref{Union{Nothing,Bool}}(nothing))
    end
end

# IMinuit.jl drop-in aliases. In IMinuit.jl `Fit` (keyword/scalar-arg
# construction) and `ArrayFit` (vector construction) are distinct PyCall
# wrapper subtypes; in native JuMinuit both reduce to the same `Minuit`
# (the keyword constructor wraps `fcn(a,b)` into `x -> fcn(x...)`), so
# they are aliases here, not separate types. See `AbstractFit` docs.
const Fit = Minuit
const ArrayFit = Minuit

# ── threaded_gradient policy ────────────────────────────────────────────────
# `threaded_gradient` is a 3-way switch resolved in ONE place by these helpers:
#   false (default) → serial;  true → force threaded (errors if unsafe);
#   :auto → thread iff the FCN probes thread-safe, else warn once + serial.
# `_check_threaded_gradient` validates it at construction; `_use_threads`
# resolves it to the concrete Bool handed to the numerical-gradient path.
_check_threaded_gradient(tg::Bool) = tg
function _check_threaded_gradient(tg::Symbol)
    tg === :auto || throw(ArgumentError(
        "threaded_gradient must be `true`, `false`, or `:auto`; got `:$tg`"))
    return tg
end
_check_threaded_gradient(tg) = throw(ArgumentError(
    "threaded_gradient must be `true`, `false`, or `:auto`; got `$(repr(tg))`"))

# Resolve `m`'s threaded_gradient policy (or an explicit `mode` override) into
# the concrete Bool passed downstream as `threaded_gradient`:
#   false → false (serial).
#   true  → true  (force threaded; the leaf numerical_gradient! still gates on
#           nthreads()>1 and the strict ThreadSafetyError path is preserved).
#   :auto → true iff nthreads()>1, the fit is numerical (not AD), and the FCN
#           probes thread-safe; else false. The is_thread_safe probe runs at
#           most once (memoized on m._auto_threads); the unsafe->serial
#           fallback emits a single @warn. Never throws.
# REVISIT: a future cost-aware :auto could thread by default based on per-FCN
# cost x n; that needs benchmarks and is intentionally not implemented now (the
# default stays false).
function _use_threads(m::Minuit, mode::Union{Bool,Symbol} = m.threaded_gradient)
    mode === false && return false
    mode === true && return true
    mode === :auto || _check_threaded_gradient(mode)   # throws on a bad Symbol
    # :auto below. AD fits compute the gradient in one call (no per-coordinate
    # loop), so threading is a no-op — skip the probe and the (misleading) warn.
    m.cfwg === nothing || return false
    # Single-threaded Julia: threading is impossible — no probe, no warning.
    Threads.nthreads() == 1 && return false
    cached = m._auto_threads[]
    if cached === nothing
        x0 = [p.value for p in m.params.pars]
        # Probe a fresh CostFunction view (own nfcn Ref) so the safety check's
        # FCN evaluations don't pollute the user's call counter.
        cached = is_thread_safe(CostFunction(m.fcn.f, m.fcn.up), x0)
        m._auto_threads[] = cached
        cached || @warn(
            "threaded_gradient=:auto: FCN is not thread-safe (its threaded " *
            "gradient disagrees with the serial one at the seed) — falling " *
            "back to the serial gradient. Fix the FCN's shared mutable state " *
            "(see the README thread-safety contract), or pass " *
            "threaded_gradient=true to get a ThreadSafetyError instead.")
    end
    return cached
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
    # When a `grad=` is supplied, validate it against a numerical 2-point
    # estimate at the seed and warn on disagreement (the C++ Minuit2
    # `CheckGradient` discrepancy check; MnSeedGenerator.cxx:124-144).
    # Default `true` matches C++ `FCNGradientBase::CheckGradient()`; set
    # `false` to skip the (one-time, seed) check. No-op without `grad=`.
    check_gradient::Bool = true,
    # IMinuit.jl-compatible stored settings. These become `m.strategy`,
    # `m.tol`, `m.print_level` and feed into subsequent migrad! calls
    # when not explicitly overridden.
    #
    # Default `Strategy(1)` matches the iminuit `Minuit` class default
    # (`self._strategy = MnStrategy(1)`) and C++ Minuit2's `MnStrategy()`
    # default (`SetMediumStrategy`), so a bare `migrad!(m)` is drop-in-
    # equivalent to iminuit's `m.migrad()` — for BOTH numerical and
    # analytical/AD (`grad=`) FCNs (iminuit applies strategy 1 regardless of
    # whether a gradient is supplied; the AD `seed_state` in ad_gradient.jl
    # supports all strategy levels). Strategy 1 enables the dcovar-triggered
    # inner-HESSE refinement inside `_migrad_loop`, which re-seeds the DFP
    # curvature mid-run and reaches deeper minima on stiff fits than the
    # coarse 2-cycle Strategy(0) gradient (see docs/dev/IAM_CONVERGENCE_GAP.md).
    #
    # The *low-level* `migrad(cf, …)` / `seed` / `function_cross` / `minos`
    # / `contours` entry points keep their own `Strategy(0)` defaults
    # (pinned to the C++ oracle reference data — see test_cpp_oracle.jl).
    strategy::Union{Strategy,Integer} = Strategy(1),
    tol::Real = 0.1,
    print_level::Integer = 0,
    # Phase G/I: parallel inner numerical-gradient (requires `julia -t N`).
    # `false` (default) = serial; `true` = force threaded (raises
    # `ThreadSafetyError` if the FCN is not thread-safe); `:auto` = thread iff
    # safe, else warn + serial (never throws). Default stays `false`: threading
    # only pays off for expensive FCNs at higher n, so threading by default
    # would slow the common cheap-FCN case and add probe overhead to every
    # `julia -t N` fit. `:auto` is the opt-in "thread it safely without me
    # checking" switch. Opt into `true`/`:auto` when (a) FCN > 500 ns/call,
    # (b) n ≥ 4, (c) the FCN is thread-safe (no hidden mutable state).
    threaded_gradient::Union{Bool,Symbol} = false,
    # Phase H: for `threaded_gradient=true`, auto-verify FCN thread-safety on
    # the first gradient call (default `true` when forcing threading; ~2 extra
    # gradient evaluations). Pass `false` to bypass once you've confirmed
    # safety via `JuMinuit.is_thread_safe(cf, x0)`. No-op for `:auto` (its own
    # one-shot memoized probe) and for `false`.
    verify_threading::Bool = (threaded_gradient === true),
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
        CostFunctionWithGradient(fcn, grad, up_resolved, cf.nfcn, Ref(0);
                                 check_gradient = check_gradient)
    strat = strategy isa Strategy ? strategy : Strategy(Int(strategy))
    return Minuit(cf, params, nothing, Dict{Int,MinosError}(), prec,
                  cfwg, strat, Float64(tol), Int(print_level),
                  _check_threaded_gradient(threaded_gradient),
                  Bool(verify_threading), 0, nothing)
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
                # iminuit Minuit-class default (level 1), for numerical AND
                # AD FCNs. See the Minuit(fcn, x0) constructor for the
                # full rationale.
                strategy::Union{Strategy,Integer} = Strategy(1),
                tol::Real = 0.1,
                print_level::Integer = 0,
                threaded_gradient::Union{Bool,Symbol} = false,
                verify_threading::Bool = (threaded_gradient === true),
                # Seed-time gradient-check toggle — see the Minuit(fcn, x0)
                # constructor. Declared explicitly (not left to `kwargs...`)
                # so a `check_gradient=false` is NOT mis-parsed as a
                # parameter-value kwarg (since `false isa Real`).
                check_gradient::Bool = true,
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
                  check_gradient = check_gradient,
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
                       iterate=5, use_simplex=false,
                       threaded_gradient=m.threaded_gradient,
                       verify_threading=m.verify_threading,
                       print_level=m.print_level) -> Minuit

Run MIGRAD on `m`. By default this is **drop-in-equivalent to iminuit's
`m.migrad()`**: a single MIGRAD followed by iminuit's `_robust_low_level_fit`
retry — if a pass fails to validate (no-improvement / above-max-EDM exit, see
C++ `VariableMetricBuilder.cxx:278`) and the call limit hasn't been reached,
re-run MIGRAD from the last converged point **at the same strategy** (a fresh
re-seed, discarding the possibly-degraded DFP inverse-Hessian), up to
`iterate-1` more times. C++ Minuit2 itself has no retry — this loop is iminuit's
addition, reproduced faithfully here. The re-seed lets a stalled fit escape
(IAM cold start: S=0 613 → ~383, S=1 330 → ~326 — via iminuit's retry
*mechanism*; the exact basin reached differs from iminuit's on this
ill-conditioned problem, see docs/dev/IAM_CONVERGENCE_GAP.md § Fidelity).

**`use_simplex=true` (opt-in, NOT the default) enables a structured Simplex
multistart that is NOT part of C++ Minuit2 or iminuit** — a JuMinuit extension
for genuinely multi-minimum landscapes (e.g. the X(3872) `J/ψρ + DD̄*`
multi-dip fit). Each retry pass takes a Nelder-Mead Simplex hop with a
geometrically-growing seed step (×1, ×2, ×4, … from the parameter error scale,
capped at the physical range) before re-MIGRAD at `Strategy(2)` for numerical
FCNs (the AD path keeps the user strategy). It is this opt-in path's
`Strategy(2)` escalation that walks the IAM x_jm WARM start to χ²=322 (PR #10);
at the faithful default, x_jm converges to iminuit's 325.8 and 322 is reached
the C++/iminuit way — by passing `strategy=2`.

Both paths keep a fixed-point (cycle) detector — re-convergence to an
already-visited `(x, fval)` basin stops the loop (recovering wasted passes on
single-basin fits) — and the safety invariant: `iterate=N` never yields a worse
fval than `iterate=1` (`best_bfm`, the lowest-fval pass, is published).

(The opt-in `use_simplex=true` multistart's multi-scale escape — for genuine
multiple local minima in multi-parameter HEP amplitude / phase-shift / LEC
fits — does NOT prove the global minimum was found; global optimization is
undecidable. The defensible statement: it searches perturbation scales up to
re-convergence on a known basin or the physical parameter range.)

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
                  use_simplex::Bool = false,
                  threaded_gradient::Union{Bool,Symbol} = m.threaded_gradient,
                  verify_threading::Bool = m.verify_threading,
                  print_level::Integer = m.print_level)
    # iminuit's `_robust_low_level_fit` requires iterate ≥ 1. iterate=1
    # disables retry (only pass 1 runs) and reproduces single-shot
    # C++-faithful behavior; iterate ≤ 0 would silently skip MIGRAD
    # entirely, so reject it to surface the user mistake.
    iterate >= 1 ||
        throw(ArgumentError("iterate must be ≥ 1, got $iterate"))

    # Resolve the 3-way `threaded_gradient` policy ONCE for this fit (and all
    # retry passes): `:auto` probes thread-safety here (memoized on `m`) and
    # falls back to serial + warn if unsafe; `true`/`false` pass through. For
    # `:auto` the strict `verify_threading` re-check is disabled (the probe is
    # the single safety gate), so MIGRAD never re-verifies per pass.
    _tg = _use_threads(m, threaded_gradient)
    _vt = threaded_gradient === :auto ? false : verify_threading

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
                         threaded_gradient = _tg,
                         verify_threading = _vt,
                         print_level = print_level)

    # ── Robust retry loop ───────────────────────────────────────────────
    # Default (`use_simplex=false`): a faithful reproduction of iminuit's
    # `_robust_low_level_fit` — re-run MIGRAD from the last converged point
    # at the SAME strategy (a fresh re-seed; the degraded DFP inverse-Hessian
    # is discarded), up to `iterate` passes, stopping when the fit validates.
    # C++ Minuit2 has no retry; iminuit adds exactly this loop, so `migrad!(m)`
    # is drop-in-equivalent to iminuit's `m.migrad()`. The re-seed is what lets
    # a stalled fit escape (IAM cold start: S=0 613 → ~383, S=1 330 → ~326 — via
    # iminuit's retry *mechanism*; the exact basin differs from iminuit's on the
    # ill-conditioned IAM, see docs/dev/IAM_CONVERGENCE_GAP.md § Fidelity).
    #
    # Opt-in (`use_simplex=true`): a structured Simplex multistart that is NOT
    # part of C++ Minuit2 or iminuit — a JuMinuit EXTENSION for genuinely
    # multi-minimum landscapes (e.g. the X(3872) `J/ψρ + DD̄*` multi-dip fit).
    # Each pass takes a Nelder-Mead Simplex hop with a geometrically-growing
    # seed step (`_retry_perturb_factor`, ×2 per pass, capped at the physical
    # range) before re-MIGRAD at `retry_strategy` (Strategy(2) for numerical
    # FCNs — the heavier level; the AD path keeps the user strategy since its
    # seed supports all levels). It is this opt-in path's Strategy(2)
    # escalation that walks the IAM x_jm WARM start to χ²=322 (PR #10); at the
    # faithful default x_jm converges to iminuit's 325.8, and 322 is reached
    # the C++/iminuit way — by passing `strategy=2`.
    #
    # Both paths share: (a) fixed-point (cycle) detection — if a pass
    # re-converges to an already-visited (x, fval) basin (parameter-error-
    # relative tolerance) the loop stops, recovering the wasted passes on
    # single-basin fits like IAM; (b) the safety invariant (PR #8) —
    # `iterate=N` never yields a worse fval than `iterate=1`, guaranteed by
    # construction: `best_bfm` tracks the lowest-fval pass (tie → the valid
    # one) and is what lands in `m.fmin`. The opt-in path additionally stops
    # when the perturbation saturates every free parameter's physical range.
    #
    # We do NOT claim the global minimum is found (global optimization is
    # undecidable). Defensible statements: the default matches iminuit; the
    # opt-in samples increasing perturbation scales up to re-convergence or
    # the physical range.
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
        factor = 1.0
        pass_strategy = strategy          # faithful default: same strategy, no bump
        if use_simplex
            # Opt-in JuMinuit multistart EXTENSION (NOT in C++ Minuit2 or
            # iminuit): a Nelder-Mead Simplex hop with a geometrically-growing
            # seed step, then re-MIGRAD at `retry_strategy`. The post-Simplex
            # state has no usable inverse Hessian (MinimumError is the I
            # placeholder), so we carry only its ext_values forward and let the
            # next MIGRAD cold-seed (prior_cov stays nothing).
            pass_strategy = retry_strategy
            factor = _retry_perturb_factor(_pass)
            params_pert = _retry_scaled_params(m, params_next, factor, base_errs)
            sx = simplex(m.fcn, params_pert; maxfcn = maxfcn, prec = m.prec)
            params_next = _build_resume_params(m, sx)
        end

        bfm = _migrad_into!(m, params_next;
                             strategy = pass_strategy, tol = tol,
                             maxfcn = maxfcn,
                             threaded_gradient = _tg,
                             verify_threading = _vt,
                             print_level = print_level,
                             prior_cov = prior_cov)
        npass += 1
        best_bfm = _retry_select_better(bfm, best_bfm)

        # Cycle detection: re-convergence to an already-catalogued basin.
        _retry_is_fixed_point(bfm, visited, base_errs) && break
        push!(visited, (copy(bfm.ext_values), fval(bfm)))

        # Opt-in path: stop once the perturbation spans every free parameter's
        # physical range (no larger meaningful hop remains).
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
                threaded_gradient::Union{Bool,Symbol} = m.threaded_gradient,
                print_level::Integer = m.print_level,
                kwargs...)
    1 <= par <= n_pars(m.params) ||
        throw(ArgumentError("par index $par out of bounds"))
    # Fixed parameters carry no MINOS error — skip silently and return `m`
    # so `minos!(m)` (all-params) and method chaining keep working.
    is_fixed(m.params.pars[Int(par)]) && return m
    m.minos_errors[Int(par)] = _minos_error(m, Int(par);
                                              threaded_gradient = threaded_gradient,
                                              print_level = print_level, kwargs...)
    return m
end

# C++/iminuit default per-cross-search MINOS FCN budget (MnMinos.cxx
# :111-114): when the user passes maxcalls == 0,
#     maxcalls = 2·(nvar+1)·(200 + 100·nvar + 5·nvar²)
# where `nvar` is the number of variable (free) parameters
# (`MnUserParameterState::VariableParameters()`, the JuMinuit `n_free`).
# Replaces the legacy hardcoded 1000, which could trip `fcn_limit` on
# larger fits where C++/iminuit would keep iterating. Each ± cross-search
# gets the full budget (C++ recomputes it per `FindCrossValue` call).
_minos_default_maxcalls(nvar::Integer) =
    2 * (nvar + 1) * (200 + 100 * nvar + 5 * nvar * nvar)

# Internal: compute the asymmetric MinosError for ONE parameter WITHOUT
# storing it on `m`. Shared choke point for `minos!` (which stores the
# result), `minos_lower`, and `minos_upper`. Validates, translates the
# iminuit / C++ MnMinos control-name kwargs to the internal
# `function_cross` names, and dispatches to the bounded / unbounded path.
function _minos_error(m::Minuit, par::Int;
                       threaded_gradient::Union{Bool,Symbol} = m.threaded_gradient,
                       print_level::Integer = m.print_level,
                       maxcall::Integer = 0,
                       tol::Union{Real,Nothing} = nothing,
                       toler::Union{Real,Nothing} = nothing,
                       kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before MINOS"))
    # Resolve the 3-way threaded_gradient policy (`:auto` → memoized probe).
    _tg = _use_threads(m, threaded_gradient)
    # MINOS derives sigma_i = sqrt(2·up·V[i,i]) from the inverse Hessian, so
    # it requires an actual covariance — not the identity placeholder that
    # simplex / scan leave behind. Force the user to call hesse(m) first
    # (review IMPORTANT #2 round-2).
    JuMinuit.is_available(m.fmin.internal.state.error) ||
        throw(ArgumentError(
            "MINOS requires a covariance matrix. The last fit produced " *
            "no inverse Hessian (likely simplex/scan, or HESSE didn't " *
            "run). Call `hesse(m)` first."))
    1 <= par <= n_pars(m.params) ||
        throw(ArgumentError("par index $par out of bounds"))
    is_fixed(m.params.pars[par]) &&
        throw(ArgumentError("MINOS is undefined for fixed parameter $par"))

    # iminuit / C++ MnMinos control-name translation. `maxcall` (iminuit,
    # singular) → internal `maxcalls`; `toler` (C++ MnMinos positional) and
    # `tol` (iminuit kwarg) both → internal `tlr`. When the user passes no
    # explicit `maxcall` (the `maxcall == 0` sentinel) we forward the C++/
    # iminuit n-scaled default budget (MnMinos.cxx:111-114) instead of
    # letting the downstream fall back to its legacy hardcoded 1000.
    # Explicit internal-name kwargs (`maxcalls` / `tlr`) passed by power
    # users still win over the translation via the final merge.
    fwd = NamedTuple()
    if maxcall > 0
        fwd = merge(fwd, (; maxcalls = Int(maxcall)))
    else
        fwd = merge(fwd, (; maxcalls = _minos_default_maxcalls(n_free(m.params))))
    end
    if toler !== nothing
        fwd = merge(fwd, (; tlr = Float64(toler)))
    elseif tol !== nothing
        fwd = merge(fwd, (; tlr = Float64(tol)))
    end
    fwd = merge(fwd, (; kwargs...))

    p = m.params.pars[par]
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
        return _minos_external_via_function_cross(
            m.fmin, ext_cf, par;
            threaded_gradient = _tg,
            print_level = print_level, fwd...)
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
        return minos(m.fmin.internal, m.fmin.internal_cf,
                     m.params.int_of_ext[par];
                     pars = m.params,
                     threaded_gradient = _tg,
                     print_level = print_level, fwd...)
    end
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
    minos_upper(m::Minuit, par; kwargs...) -> Float64
    minos_lower(m::Minuit, par; kwargs...) -> Float64

Return ONLY the upper (`minos_upper`, ≥ 0) or lower (`minos_lower`, ≤ 0)
asymmetric MINOS error for parameter `par` (Integer index or String name).
Mirror of C++ `MnMinos::Upper` / `MnMinos::Lower`
(`reference/Minuit2_cpp/inc/Minuit2/MnMinos.h:50-58`).

The sign convention matches [`MinosError`](@ref)`.upper` / `.lower`:
`minos_upper` is one σ to the right (positive), `minos_lower` one σ to the
left (negative). The returned value is identical to the corresponding side
of a full [`minos!`](@ref)`(m, par)` (same `function_cross` machinery).

Unlike `minos!`, these are **pure queries**: they do NOT mutate `m`
(no `m.minos_errors` update), matching the C++ const accessors. The full
asymmetric error is computed internally and the requested side returned —
if you need both sides, prefer a single `minos!(m, par)`.

Accepts the same control kwargs as `minos!`: `maxcall`, `tol` / `toler`,
`sigma`, `strategy`, `print_level`.
"""
function minos_upper(m::Minuit, par::Integer; kwargs...)
    return _minos_error(m, Int(par); kwargs...).upper
end
minos_upper(m::Minuit, par::AbstractString; kwargs...) =
    minos_upper(m, ext_index(m.params, String(par)); kwargs...)
minos_upper(m::Minuit, par::Symbol; kwargs...) =
    minos_upper(m, String(par); kwargs...)

function minos_lower(m::Minuit, par::Integer; kwargs...)
    return _minos_error(m, Int(par); kwargs...).lower
end
minos_lower(m::Minuit, par::AbstractString; kwargs...) =
    minos_lower(m, ext_index(m.params, String(par)); kwargs...)
minos_lower(m::Minuit, par::Symbol; kwargs...) =
    minos_lower(m, String(par); kwargs...)

"""
    contour(m::Minuit, par_x, par_y; npoints=20, bins=nothing, kwargs...) -> ContoursError

Compute a 2D contour. `par_x` / `par_y` may be Integer or String.
The `bins=...` kwarg is an IMinuit.jl-compatible alias for `npoints`
(takes precedence when both are passed).
"""
function contour(m::Minuit, par_x::Integer, par_y::Integer;
                  npoints::Integer = 20,
                  bins::Union{Integer,Nothing} = nothing,
                  threaded_gradient::Union{Bool,Symbol} = m.threaded_gradient,
                  kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `contour(m, ...)`"))
    _tg = _use_threads(m, threaded_gradient)
    npts = bins === nothing ? Int(npoints) : Int(bins)
    ix = m.params.int_of_ext[par_x]
    iy = m.params.int_of_ext[par_y]
    # Use the internal-coord-wrapped CostFunction (parallel-review #4
    # A7/B4 — see minos! for the rationale).
    return contour(m.fmin.internal, m.fmin.internal_cf, ix, iy;
                    npoints = npts,
                    threaded_gradient = _tg, kwargs...)
end

function contour(m::Minuit, px::AbstractString, py::AbstractString;
                  kwargs...)
    return contour(m, ext_index(m.params, String(px)),
                      ext_index(m.params, String(py)); kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Property-style access (iminuit copy-paste compatibility)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Write-back parameter views (iminuit `ValueView` / `LimitView` parity).
#
# `m.values`, `m.errors`, `m.fixed`, and `m.limits` each return a
# lightweight `ParameterView` that READS live from `m` and WRITES back
# through the per-parameter mutators. This makes iminuit's canonical
# indexed-assignment idiom mutate `m` in place instead of a throwaway
# copy (the silent-no-op bug this type fixes):
#
#     m.fixed["alpha"] = true            # → fix!(m, "alpha")
#     m.values["x"]    = 1.5             # → set_value!(m, "x", 1.5)
#     m.errors[1]      = 0.3             # → set_error!(m, 1, 0.3)
#     m.limits["x"]    = (0.0, nothing)  # → set_limits!(m, "x", 0.0, nothing)
#     m.limits["x"]    = nothing         # → remove_limits!(m, "x")
#
# Each view subtypes `AbstractVector{T}` so *reading* is byte-for-byte
# what these properties used to return as freshly built `Vector`s: `==`
# against a plain `Vector`, `collect`, `copy`, broadcasting
# (`m.values .+ 1`), iteration, and scalar `getindex` all work
# unchanged (the AbstractArray fallbacks derive them from `size` +
# `getindex`, and `similar`/`copy` produce a plain `Array`). The `kind`
# type parameter (`:values`/`:errors`/`:fixed`/`:limits`) selects the
# read source and write target; `T` is the element type.
#
# Indexing accepts an `Int` (1-based external index) OR an
# `AbstractString` (parameter name → `ext_index`). Writes route through
# the same mutators as the explicit API, so cache-invalidation
# (`m.fmin=nothing`, `empty!(m.minos_errors)`) and validation are
# guaranteed identical. Whole-vector assignment (`m.values = [...]`) is
# handled by `setproperty!` and is unaffected.
# ─────────────────────────────────────────────────────────────────────────────

struct ParameterView{kind,T} <: AbstractVector{T}
    m::Minuit
end

# `:limits` reads/writes `(lower, upper)` tuples (NaN sentinel = absent
# bound), matching the pre-view `[(p.lower, p.upper) for p in ...]`.
_view_eltype(::Val{:values}) = Float64
_view_eltype(::Val{:errors}) = Float64
_view_eltype(::Val{:fixed})  = Bool
_view_eltype(::Val{:limits}) = Tuple{Float64,Float64}

ParameterView(m::Minuit, kind::Symbol) =
    ParameterView{kind,_view_eltype(Val(kind))}(m)

# All views span every (free AND fixed) parameter, like iminuit.
Base.size(v::ParameterView) = (n_pars(v.m.params),)
Base.IndexStyle(::Type{<:ParameterView}) = IndexLinear()

# ── reads (live; mirror the old getproperty branches exactly) ────────────────
# `:values`/`:errors` prefer the post-fit external vector when a fit is
# cached, else the stored initial value/step.
_view_get(::Val{:values}, m::Minuit, i::Int) =
    m.fmin === nothing ? m.params.pars[i].value : m.fmin.ext_values[i]
_view_get(::Val{:errors}, m::Minuit, i::Int) =
    m.fmin === nothing ? m.params.pars[i].error : m.fmin.ext_errors[i]
_view_get(::Val{:fixed}, m::Minuit, i::Int) = is_fixed(m.params.pars[i])
_view_get(::Val{:limits}, m::Minuit, i::Int) =
    (m.params.pars[i].lower, m.params.pars[i].upper)

@inline function Base.getindex(v::ParameterView{kind}, i::Int) where {kind}
    @boundscheck checkbounds(v, i)
    return _view_get(Val(kind), v.m, i)
end

# Name indexing: `m.values["x"]`. `ext_index` throws `KeyError` for an
# unknown name (same as the explicit mutators).
Base.getindex(v::ParameterView, name::AbstractString) =
    v[ext_index(v.m.params, String(name))]

# ── writes (route through the per-parameter mutators) ────────────────────────
_view_set!(::Val{:values}, m::Minuit, i::Int, val) = set_value!(m, i, val)
_view_set!(::Val{:errors}, m::Minuit, i::Int, val) = set_error!(m, i, val)
_view_set!(::Val{:fixed}, m::Minuit, i::Int, val) =
    Bool(val) ? fix!(m, i) : release!(m, i)
# `m.limits[i] = (lo, hi)` (either side may be `nothing`/`±Inf` → absent)
# or `m.limits[i] = nothing` to clear both. Mirrors `_bulk_set_limits!`.
function _view_set!(::Val{:limits}, m::Minuit, i::Int, val)
    if val === nothing
        remove_limits!(m, i)
    elseif val isa Union{Tuple,AbstractVector} && length(val) == 2
        set_limits!(m, i, val[1], val[2])
    else
        # A scalar (e.g. `m.limits[i] = 5.0`) would otherwise throw a
        # cryptic destructuring `BoundsError`; give the user-facing API a
        # clear message instead.
        throw(ArgumentError(
            "m.limits[par] expects a 2-tuple `(lo, hi)` (either side may be " *
            "`nothing`/`±Inf`) or `nothing` to clear; got $(typeof(val))"))
    end
end

@inline function Base.setindex!(v::ParameterView{kind}, val, i::Int) where {kind}
    @boundscheck checkbounds(v, i)
    _view_set!(Val(kind), v.m, i, val)
    return v
end

Base.setindex!(v::ParameterView, val, name::AbstractString) =
    setindex!(v, val, ext_index(v.m.params, String(name)))

function Base.getproperty(m::Minuit, name::Symbol)
    if name === :values
        return ParameterView(m, :values)
    elseif name === :errors
        return ParameterView(m, :errors)
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
        return ParameterView(m, :fixed)
    elseif name === :limits
        return ParameterView(m, :limits)
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
    elseif name === :ndata
        # Number of data points for the χ²/ndf + p-value display line.
        # Accept an integer count or `nothing` to clear it; coerce Reals
        # to Int so `m.ndata = length(data)` works regardless of eltype.
        setfield!(m, :ndata, val === nothing ? nothing : Int(val))
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
                                          m.cfwg.nfcn, m.cfwg.ngrad;
                                          check_gradient = m.cfwg.check_gradient))
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

"""
    set_upper_limit!(m::Minuit, par::Union{Integer,AbstractString}, hi::Real) -> Minuit

Constrain `par` from ABOVE ONLY: set its upper bound to `hi` and clear
any lower bound, leaving it half-open `(-∞, hi]`. Mirrors C++
`MnUserParameters::SetUpperLimit` → `MinuitParameter::SetUpperLimit`
(`reference/Minuit2_cpp/inc/Minuit2/MinuitParameter.h:123-129`), which
sets `fLoLimValid=false`, `fUpLimValid=true`. Drops `m.fmin` and clears
`m.minos_errors`. Returns `m` for chaining.

NB: clearing the lower bound is the C++ behavior — to instead KEEP an
existing lower bound and add an upper one, use the two-sided
[`set_limits!`](@ref) (or `m.limits[par] = (lo, hi)`). `hi` must be
finite (NaN / ±Inf throw `ArgumentError`); use [`remove_limits!`](@ref)
to drop a bound.
"""
set_upper_limit!(m::Minuit, par::AbstractString, hi::Real) =
    set_upper_limit!(m, ext_index(m.params, String(par)), hi)
function set_upper_limit!(m::Minuit, i::Integer, hi::Real)
    _check_par_index(m, i)
    isfinite(Float64(hi)) ||
        throw(ArgumentError("set_upper_limit!: upper bound must be finite, got $hi"))
    # C++ SetUpperLimit clears the lower bound (fLoLimValid=false).
    return _replace_one_param!(m, Int(i),
        _build_limits_par(m.params.pars[i], nothing, hi))
end

"""
    set_lower_limit!(m::Minuit, par::Union{Integer,AbstractString}, lo::Real) -> Minuit

Constrain `par` from BELOW ONLY: set its lower bound to `lo` and clear
any upper bound, leaving it half-open `[lo, +∞)`. Mirrors C++
`MnUserParameters::SetLowerLimit` → `MinuitParameter::SetLowerLimit`
(`reference/Minuit2_cpp/inc/Minuit2/MinuitParameter.h:131-137`), which
sets `fLoLimValid=true`, `fUpLimValid=false`. Drops `m.fmin` and clears
`m.minos_errors`. Returns `m` for chaining.

NB: clearing the upper bound is the C++ behavior — to instead KEEP an
existing upper bound and add a lower one, use the two-sided
[`set_limits!`](@ref) (or `m.limits[par] = (lo, hi)`). `lo` must be
finite (NaN / ±Inf throw `ArgumentError`); use [`remove_limits!`](@ref)
to drop a bound.
"""
set_lower_limit!(m::Minuit, par::AbstractString, lo::Real) =
    set_lower_limit!(m, ext_index(m.params, String(par)), lo)
function set_lower_limit!(m::Minuit, i::Integer, lo::Real)
    _check_par_index(m, i)
    isfinite(Float64(lo)) ||
        throw(ArgumentError("set_lower_limit!: lower bound must be finite, got $lo"))
    # C++ SetLowerLimit clears the upper bound (fUpLimValid=false).
    return _replace_one_param!(m, Int(i),
        _build_limits_par(m.params.pars[i], lo, nothing))
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
            :strategy, :tol, :print_level, :n_passes, :ndata,
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
(settable via `m.strategy = ...`, `m.tol = ...`). The constructor
default is `Strategy(1)` — matching iminuit's `Minuit`-class default and
C++ Minuit2's `MnStrategy()` — for both numerical and analytical/AD
(`grad=`) FCNs. Override per call with `strategy=...`, or set
`m.strategy = ...` once before the first migrad.

`iterate` and `use_simplex` are threaded through to [`migrad!`](@ref)
unchanged. By default (`use_simplex=false`) the retry is iminuit's
`_robust_low_level_fit` — re-run at the user's strategy, no Simplex, no
strategy bump — so this is drop-in-equivalent to iminuit's `m.migrad()`.
The opt-in `use_simplex=true` enables JuMinuit's Simplex multistart (which
*does* bump numerical FCNs to `Strategy(2)`); that is a documented extension
beyond C++ Minuit2 / iminuit — see the [`migrad!`](@ref) docstring.
"""
function migrad(m::Minuit;
                 ncall::Union{Integer,Nothing} = nothing,
                 resume::Bool = true,
                 precision::Union{Real,Nothing} = nothing,
                 strategy::Strategy = m.strategy,
                 tol::Real = m.tol,
                 iterate::Integer = 5,
                 use_simplex::Bool = false)
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

    # Refresh the internal-coord Hessian. Thread the per-parameter bound
    # flags so the diagonal step clamp (C++ MnHesse.cxx:160-167, 194-195)
    # fires for bounded parameters — `bfm.internal.state` is in INTERNAL
    # (transformed) coordinates, exactly the frame the C++ clamp targets.
    new_state = JuMinuit.hesse(bfm.internal_cf, bfm.internal.state, strategy;
                                 prec = m.prec,
                                 has_limits = JuMinuit._has_limits_internal(bfm.params),
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
to the k-σ contour. `maxcall` (iminuit, singular) caps the FCN calls
of each cross-search and `tol` / `toler` set the cross-search tolerance;
both are forwarded to `function_cross` via `minos!`.
"""
function minos(m::Minuit, var = nothing;
                sigma::Real = 1, kwargs...)
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
            minos(m, v; sigma = sigma, kwargs...)
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
MIGRAD/HESSE/MINOS. The default `MachinePrecision()` uses `4·eps(Float64)`
(matching C++ Minuit2's `fEpsMac = 4·ε`, audit §14); override only when
fitting with synthetic-precision FCN models.
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
    @printf(io, "JuMinuit.Minuit  fval=%.6g  edm=%.3g  nfcn=%d\n",
            m.fval, m.edm, m.nfcn)
    chi = _chi2_summary(m)
    if chi !== nothing
        @printf(io, "χ²/ndf = %.4g/%d = %.3g  (p = %.3g)\n",
                chi.chi2, chi.ndf, chi.ratio, chi.p)
    end
    # Validity checklist (replaces the old single status string).
    println(io, _checklist_text(m))

    # Build rows + compute column widths. The Value column merges the
    # central value with its uncertainty — asymmetric MINOS when present,
    # else the symmetric Hesse error — rounded to the uncertainty.
    headers = ["#", "Name", "Value", "Limit −", "Limit +", "Fixed"]
    rows = [_param_row_data(m, i) for i in 1:n_pars(m.params)]
    cells = [[
        string(r.idx),
        r.name,
        _value_cell(r; mode = :text),
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

    # Strong-correlation (near-degeneracy) warnings.
    _render_corr_warning_text(io, m)
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

    @printf(io, """<div style="font-family:monospace;font-size:0.95em">""")
    @printf(io,
        """<strong>JuMinuit.Minuit</strong>  fval=%.6g  edm=%.3g  nfcn=%d<br>""",
        m.fval, m.edm, m.nfcn)
    chi = _chi2_summary(m)
    if chi !== nothing
        @printf(io, """χ²/ndf = %.4g/%d = %.3g&nbsp;&nbsp;(p = %.3g)<br>""",
                chi.chi2, chi.ndf, chi.ratio, chi.p)
    end
    # Validity checklist chips (replaces the old single status badge).
    _render_checklist_html(io, m)

    # Parameter table. The Value column merges the central value with its
    # uncertainty — asymmetric MINOS when present, else symmetric Hesse.
    headers = ["#", "Name", "Value", "Limit −", "Limit +", "Fixed"]
    print(io, """<table style="border-collapse:collapse;margin-top:0.3em">""")
    print(io, "<thead><tr>")
    for h in headers
        print(io, """<th style="border:1px solid #d0d7de;padding:2px 8px;background:#f6f8fa">""",
              h, "</th>")
    end
    print(io, "</tr></thead><tbody>")
    for i in 1:n_pars(m.params)
        r = _param_row_data(m, i)
        # Parameter `r.name` is user-controlled → escape it. The Value
        # cell is a plain number, a "v ± e" string, or the asymmetric
        # `<sup>/<sub>` markup from `_format_value_minos` — that markup is
        # intentional and built only from formatted numbers (no
        # user-controlled text), so it must NOT be escaped. Limit / Fixed
        # cells are numeric or fixed literals and are safe.
        cells = [string(r.idx), _html_escape(r.name), _value_cell(r; mode = :html),
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

    # Correlation-matrix heatmap + strong-correlation warnings.
    _render_heatmap_html(io, m)
    _render_corr_warning_html(io, m)
    print(io, "</div>")
end

# Short one-line repr for `print` / inline display in Vector etc.
Base.show(io::IO, m::Minuit) =
    print(io, "Minuit(", n_pars(m.params), " params, ",
              m.fmin === nothing ? "not minimized" : "fval=$(m.fval)", ")")
