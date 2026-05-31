# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# JuMinuitOptimExt — the `optim(m)` / `minimize_with(m)` alternative-minimizer
# bridge, powered by Optim.jl.
#
# iminuit's `m.scipy(method=...)` minimises the FCN with `scipy.optimize.minimize`
# from the current parameter values — the escape hatch for when MIGRAD struggles
# (e.g. trust-region / derivative-free methods on stiff problems). It writes the
# result back into the Minuit and the user then calls `hesse()` for the
# covariance.
#
# The Julia-optimal analog bridges to Optim.jl — the native, AD-friendly
# `scipy.optimize` equivalent and the community standard — instead of shelling
# out to Python. This is a package extension (mirrors JuMinuitForwardDiffExt /
# JuMinuitPlotsExt / JuMinuitDataFramesExt): Optim pulls in a sizeable transitive
# stack (NLSolversBase, LineSearches, PositiveFactorizations, …), so making it a
# hard dependency would inflate every JuMinuit install. `using Optim` activates
# the bridge; the thin `optim` / `minimize_with` entry points in
# `src/iminuit_compat.jl` dispatch here via `Base.get_extension`, and emit a
# helpful "load Optim" message when it is absent (rather than a bare MethodError).
#
# Design: Optim optimises in EXTERNAL parameter coordinates with native box
# constraints (Fminbox) — cleaner than re-using JuMinuit's internal sin/sqrt
# bound transform, and exactly how a Julia user would reach for Optim directly.
# Fixed parameters are held out of the optimisation vector. The converged point
# is written back by constructing a `BoundedFunctionMinimum` the SAME way
# `migrad(cf, params)` does — but seeded AT the optimum (no DFP iterations) — so
# `m.values` / `m.fval` are correct and a subsequent `hesse(m)` refines the
# covariance, matching iminuit's scipy-then-hesse flow.
# ─────────────────────────────────────────────────────────────────────────────

module JuMinuitOptimExt

using JuMinuit
using Optim

# Core types/helpers reused from JuMinuit's own bounded-MIGRAD build path. The
# exported names (Minuit, Parameters, …) come in via `using JuMinuit`; the
# underscore-internal helpers are referenced fully-qualified below.

# ─────────────────────────────────────────────────────────────────────────────
# method spec → Optim optimizer
# ─────────────────────────────────────────────────────────────────────────────

# Maps an iminuit-style `method=` name (Symbol or String, case/dash/underscore
# insensitive) to a freshly-constructed Optim optimizer. Documented in the
# `optim` docstring (src/iminuit_compat.jl). Power users can bypass this table
# entirely by passing an Optim optimizer object to `minimize_with(m, opt)`.
const _METHOD_TABLE = Dict{Symbol,Function}(
    :lbfgs             => () -> LBFGS(),
    :l_bfgs_b          => () -> LBFGS(),     # scipy's "L-BFGS-B"
    :bfgs              => () -> BFGS(),
    :neldermead        => () -> NelderMead(),
    :nelder_mead       => () -> NelderMead(),
    :simplex           => () -> NelderMead(),
    :newton            => () -> Newton(),
    :conjugategradient => () -> ConjugateGradient(),
    :cg                => () -> ConjugateGradient(),
    :gradientdescent   => () -> GradientDescent(),
)

# Normalise a user method spec ("L-BFGS-B", :Nelder_Mead, "lbfgs", …) to a
# lookup symbol: lowercase, dashes/spaces → underscores.
_normalize_method(method) =
    Symbol(replace(lowercase(String(method)), '-' => '_', ' ' => '_'))

function _resolve_optimizer(method)
    key = _normalize_method(method)
    haskey(_METHOD_TABLE, key) || throw(ArgumentError(
        "optim(m): unknown method $(repr(method)). Supported names: " *
        join(sort!(string.(collect(keys(_METHOD_TABLE)))), ", ") *
        ". Or pass an Optim optimizer object directly, e.g. " *
        "`minimize_with(m, LBFGS())`."))
    return _METHOD_TABLE[key]()
end

# Build Optim.Options from the iminuit-style knobs. `ncall`/`maxcall` map to
# Optim's function-evaluation budget (`f_calls_limit`, the closest analog of
# iminuit's `ncall`); `tol` maps to the gradient-norm convergence tolerance
# (`g_tol`). Anything finer: pass a full `options=Optim.Options(...)`.
function _build_options(nmax::Union{Integer,Nothing}, tol::Union{Real,Nothing})
    kw = Pair{Symbol,Any}[]
    nmax === nothing || push!(kw, :f_calls_limit => Int(nmax))
    tol  === nothing || push!(kw, :g_tol => Float64(tol))
    return Optim.Options(; kw...)
end

# Strictly-interior clamp of the start point for Fminbox: its log-barrier
# requires `lower < x0 < upper` on every two-sided coordinate (an x0 sitting
# exactly on, or outside, a bound otherwise errors). One-sided bounds are
# nudged off the finite side only.
function _clamp_interior(x::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    out = copy(x)
    @inbounds for i in eachindex(out)
        l, h = lo[i], hi[i]
        if isfinite(l) && isfinite(h)
            pad = 1.0e-6 * (h - l)
            out[i] = clamp(out[i], l + pad, h - pad)
        elseif isfinite(l)
            out[i] = max(out[i], l + 1.0e-6 * max(abs(l), 1.0))
        elseif isfinite(h)
            out[i] = min(out[i], h - 1.0e-6 * max(abs(h), 1.0))
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# The bridge
# ─────────────────────────────────────────────────────────────────────────────

# Called by `JuMinuit.optim` / `JuMinuit.minimize_with` (which dispatch here
# through `Base.get_extension`). `optimizer`, when non-`nothing`, is an Optim
# optimizer object that overrides `method`.
function _scipy_optim(m::Minuit, optimizer = nothing;
                       method = :lbfgs,
                       ncall::Union{Integer,Nothing} = nothing,
                       maxcall::Union{Integer,Nothing} = nothing,
                       tol::Union{Real,Nothing} = nothing,
                       options::Union{Optim.Options,Nothing} = nothing)
    params = m.params
    ntot   = n_pars(params)
    nfree  = n_free(params)
    nfree > 0 || throw(ArgumentError(
        "optim(m): all parameters are fixed — nothing to optimise."))

    # External indices of the free parameters, in internal order. `Parameters`
    # already maintains exactly this as its canonical int→ext map, so reuse it
    # rather than recomputing the comprehension — this guarantees the bridge's
    # free↔full mapping is identical to the one `seed_state` /
    # `initial_int_values` / `_internal_to_external_results` use on write-back.
    free_idx = params.ext_of_int

    # Start from the CURRENT external values (post-fit if a prior fit exists,
    # else the constructor's initial values) — iminuit's scipy starts from
    # wherever the Minuit currently sits.
    x0_full = Float64[m.fmin === nothing ? params.pars[i].value : m.fmin.ext_values[i]
                      for i in 1:ntot]
    x0_free = x0_full[free_idx]

    # Objective over the FREE sub-vector; fixed params held at x0_full. Kept
    # generic over `eltype(xfree)` for safety, though in practice it is only ever
    # called with `Float64` — with no `grad=` Optim uses its finite-difference
    # backend (Float64), and with `grad=` the analytical `g!` below is supplied
    # explicitly. One small allocation per call — acceptable for an opt-in escape
    # hatch, not a hot path.
    f = m.fcn.f
    objective = let f = f, x0_full = x0_full, free_idx = free_idx, ntot = ntot
        function (xfree)
            T = eltype(xfree)
            xfull = Vector{T}(undef, ntot)
            @inbounds for i in 1:ntot
                xfull[i] = T(x0_full[i])
            end
            @inbounds for (k, i) in enumerate(free_idx)
                xfull[i] = xfree[k]
            end
            return f(xfull)
        end
    end

    opt  = optimizer === nothing ? _resolve_optimizer(method) : optimizer
    opts = options === nothing ? _build_options(maxcall === nothing ? ncall : maxcall, tol) :
                                 options

    # Analytical-gradient passthrough when the user supplied `grad=` to the
    # constructor and the method is FIRST-order. Optim works in external coords,
    # so the user's external gradient — restricted to the free block — is exactly
    # what a first-order method needs (no int↔ext chain rule here). We restrict to
    # `FirstOrderOptimizer`: derivative-free methods don't take a gradient, and
    # second-order (Newton) needs a Hessian — not a bare `g!` — so for those we
    # let Optim build derivatives from the objective itself.
    use_grad = m.cfwg !== nothing && opt isa Optim.FirstOrderOptimizer
    g! = if use_grad
        guser = m.cfwg.g
        let guser = guser, x0_full = x0_full, free_idx = free_idx, ntot = ntot
            function (G, xfree)
                xfull = Vector{Float64}(undef, ntot)
                @inbounds for i in 1:ntot
                    xfull[i] = x0_full[i]
                end
                @inbounds for (k, i) in enumerate(free_idx)
                    xfull[i] = xfree[k]
                end
                gext = guser(xfull)
                @inbounds for (k, i) in enumerate(free_idx)
                    G[k] = gext[i]
                end
                return G
            end
        end
    else
        nothing
    end

    # Bounds (external). Fminbox needs full lower/upper vectors; ±Inf = open.
    any_bound = any(i -> has_lower_limit(params.pars[i]) ||
                          has_upper_limit(params.pars[i]), free_idx)

    local result
    if any_bound
        # Optim's Fminbox requires a FIRST-order inner optimizer: it rejects both
        # derivative-free (:neldermead) AND second-order/Newton methods. Gate on
        # `FirstOrderOptimizer` so the bridge emits its own guided message instead
        # of Optim's bare "X is not supported as the Fminbox optimizer".
        opt isa Optim.FirstOrderOptimizer || throw(ArgumentError(
            "optim(m): method $(repr(method)) does not support box constraints — " *
            "Optim's Fminbox needs a first-order optimizer (derivative-free and " *
            "Newton/second-order methods are rejected). Use a first-order gradient " *
            "method (:lbfgs, :bfgs, :conjugategradient, :gradientdescent) for " *
            "bounded fits, or remove the limits."))
        lower = Float64[has_lower_limit(params.pars[i]) ? params.pars[i].lower : -Inf
                        for i in free_idx]
        upper = Float64[has_upper_limit(params.pars[i]) ? params.pars[i].upper :  Inf
                        for i in free_idx]
        x0c = _clamp_interior(x0_free, lower, upper)
        result = use_grad ?
            Optim.optimize(objective, g!, lower, upper, x0c, Fminbox(opt), opts) :
            Optim.optimize(objective, lower, upper, x0c, Fminbox(opt), opts)
    else
        result = use_grad ?
            Optim.optimize(objective, g!, x0_free, opt, opts) :
            Optim.optimize(objective, x0_free, opt, opts)
    end

    x_opt_free = Optim.minimizer(result)
    converged  = Optim.converged(result)
    # `f_calls` is stable across Optim 1.9–2.1; only swallow a missing-method
    # break (don't mask an unrelated error as "0 calls").
    nfev       = try Int(Optim.f_calls(result)) catch err
        err isa MethodError ? 0 : rethrow()
    end

    _writeback!(m, params, free_idx, x0_full, x_opt_free, converged, nfev)
    return m
end

# Write the Optim optimum back into `m`: build a `BoundedFunctionMinimum` at the
# converged point using JuMinuit's own seed machinery (identical to what
# `migrad(cf, params)` constructs, minus the DFP loop). After this `m.values` /
# `m.fval` are correct and `hesse(m)` refines `m.covariance` / `m.errors`.
function _writeback!(m::Minuit, params, free_idx::Vector{Int},
                      x0_full::Vector{Float64}, x_opt_free, converged::Bool,
                      nfev::Int)
    ntot = n_pars(params)

    # Parameters at the optimum: free ← x_opt, fixed kept; bounds/fixed/names
    # preserved, user step `error` carried (it sets the seed gradient scale).
    val_full = copy(x0_full)
    @inbounds for (k, i) in enumerate(free_idx)
        val_full[i] = Float64(x_opt_free[k])
    end
    new_pars = Vector{MinuitParameter}(undef, ntot)
    @inbounds for i in 1:ntot
        p = params.pars[i]
        new_pars[i] = MinuitParameter(p.name, val_full[i], p.error;
                                       lower = p.lower, upper = p.upper,
                                       fixed = p.fixed)
    end
    params_opt = Parameters(new_pars, m.prec)

    # Reuse the canonical bounded-MIGRAD construction, seeded at the optimum.
    # `m.cfwg` carries the analytical gradient when `grad=` was supplied; the
    # plain `CostFunction` otherwise. `seed_state` builds the internal-coord
    # MinimumState (gradient + diagonal inverse-Hessian + EDM) exactly as the
    # first MIGRAD step does — for a clean quadratic this is already the exact
    # diagonal covariance; `hesse(m)` recovers the full (off-diagonal) matrix.
    cf = m.cfwg === nothing ? m.fcn : m.cfwg
    cf_internal = JuMinuit._wrap_fcn_internal_to_external(cf, params_opt)
    int_vals = JuMinuit.initial_int_values(params_opt)
    int_errs = JuMinuit.initial_int_errors(params_opt)
    seed = JuMinuit.seed_state(cf_internal, int_vals, int_errs, m.strategy, m.prec)

    # Surface Optim's function-evaluation count as the fmin's nfcn (the seed
    # itself only spent ~1 + gradient calls); the converged-status flag drives
    # `m.valid`.
    state = MinimumState(seed.parameters, seed.error, seed.gradient, seed.edm,
                          max(seed.nfcn, nfev))
    fmin_int = FunctionMinimum(state, seed, cf.up; is_valid = converged)

    ext_values, ext_errors_vec, ext_cov =
        JuMinuit._internal_to_external_results(fmin_int, params_opt, cf.up)
    m.fmin = BoundedFunctionMinimum(fmin_int, params_opt, ext_values,
                                     ext_errors_vec, ext_cov, cf_internal)
    empty!(m.minos_errors)   # any cached MINOS errors are stale at the new point
    return m
end

# NOTE: both `JuMinuit.optim` and `JuMinuit.minimize_with` are defined in
# src/iminuit_compat.jl and dispatch into `_scipy_optim` here via
# `Base.get_extension` — so this module deliberately does NOT add methods to
# them (which would overwrite the src dispatch + helpful-error fallback). The
# only public surface from here is what those entry points call: `_scipy_optim`.

# ─────────────────────────────────────────────────────────────────────────────
# Precompile the `optim(m)` bridge so the first `optim(m; method=:lbfgs)` after
# `using Optim` doesn't cold-compile the whole Optim path (objective closure,
# LBFGS, the seed-at-optimum write-back). Calls `_scipy_optim` DIRECTLY rather
# than the `JuMinuit.optim` entry point: during this extension's own
# precompilation `Base.get_extension(JuMinuit, :JuMinuitOptimExt)` is not yet
# resolvable, so `optim(m)` would take the "load Optim" fallback and compile
# nothing. try/catch-wrapped so a workload hiccup never breaks precompilation.
# ─────────────────────────────────────────────────────────────────────────────
using PrecompileTools

PrecompileTools.@setup_workload begin
    _wl_f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    PrecompileTools.@compile_workload begin
        try
            _m = Minuit(_wl_f, [0.0, 0.0])
            _scipy_optim(_m, nothing; method = :lbfgs)
        catch
            # Don't fail precompile on transient issues
        end
    end
end

end # module JuMinuitOptimExt
