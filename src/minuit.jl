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
    return Minuit(cf, params, nothing, Dict{Int,MinosError}(), prec)
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
                  errordef = errordef, prec = prec, other_kws...)
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
    # take precedence (we put theirs LAST in the call).
    return Minuit(fcn, x0; name = nm, error = er, fixed = fx, limits = lim,
                            up = m.fcn.up, prec = m.prec, kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mutating methods
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad!(m::Minuit; strategy=Strategy(0), tol=0.1, maxfcn=nothing) -> Minuit

Run MIGRAD on `m`. Updates `m.fmin`. Returns `m` for chaining.
"""
function migrad!(m::Minuit;
                  strategy::Strategy = Strategy(0),
                  tol::Real = 0.1,
                  maxfcn::Union{Integer,Nothing} = nothing)
    m.fmin = migrad(m.fcn, m.params;
                     strategy = strategy, tol = tol, maxfcn = maxfcn,
                     prec = m.prec)
    return m
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
    1 <= par <= n_pars(m.params) ||
        throw(ArgumentError("par index $par out of bounds"))
    is_fixed(m.params.pars[par]) &&
        return m  # skip fixed
    # Use the internal-coord-wrapped CostFunction stored on m.fmin.
    # Passing m.fcn (which takes EXTERNAL coords) to minos(internal_fmin)
    # would feed internal coords into the user FCN — coordinate frame
    # leak (parallel-review #4 A7/B4 blocking).
    err = minos(m.fmin.internal, m.fmin.internal_cf,
                m.params.int_of_ext[par]; kwargs...)
    # Convert internal-coord errors to external for bounded parameters,
    # matching iminuit / C++ MnMinos semantics (which report external
    # asymmetric ± offsets from the external minimum value). For
    # unbounded params the conversion is a no-op (internal == external).
    m.minos_errors[Int(par)] = _minos_int_to_ext(err, m.params.pars[Int(par)])
    return m
end

"""
    _minos_int_to_ext(err, par) -> MinosError

Convert a MinosError computed in INTERNAL parameter coordinates (what
JuMinuit's `minos` returns when called on the bounded fit's internal
state) into EXTERNAL coordinates — the form users expect, matching
C++ MnMinos and iminuit.

For unbounded parameters this is a no-op. For bounded:
  - `min_par_value` ← `int2ext(int_min)`
  - `upper` ← `int2ext(int_min + upper_int) - ext_min` (the EXT shift
     at the upper crossing point)
  - `lower` ← `int2ext(int_min + lower_int) - ext_min` (similarly;
     note `lower_int` is negative)

The shift is exact in external coordinates — there is no Jacobian
approximation. This is what C++ `MnMinos::Minos` returns at the end
(`MnMinos.cxx:120-126`).
"""
function _minos_int_to_ext(err::MinosError, par::MinuitParameter)
    has_limits(par) || has_lower_limit(par) || has_upper_limit(par) ||
        return err  # unbounded: int == ext
    kind = bound_kind(par.lower, par.upper)
    int_min = err.min_par_value
    ext_min = int2ext(kind, int_min, par.lower, par.upper)
    upper_ext = err.upper_valid ?
        int2ext(kind, int_min + err.upper, par.lower, par.upper) - ext_min :
        err.upper
    lower_ext = err.lower_valid ?
        int2ext(kind, int_min + err.lower, par.lower, par.upper) - ext_min :
        err.lower
    return MinosError(err.par_idx, ext_min,
                       upper_ext, lower_ext,
                       err.upper_valid, err.lower_valid,
                       err.upper_new_min, err.lower_new_min,
                       err.upper_fcn_limit, err.lower_fcn_limit,
                       err.nfcn)
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
    contour(m::Minuit, par_x, par_y; npoints=20, kwargs...) -> ContoursError

Compute a 2D contour. `par_x` / `par_y` may be Integer or String.
"""
function contour(m::Minuit, par_x::Integer, par_y::Integer;
                  npoints::Integer = 20, kwargs...)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad!(m)` before `contour(m, ...)`"))
    ix = m.params.int_of_ext[par_x]
    iy = m.params.int_of_ext[par_y]
    # Use the internal-coord-wrapped CostFunction (parallel-review #4
    # A7/B4 — see minos! for the rationale).
    return contour(m.fmin.internal, m.fmin.internal_cf, ix, iy;
                    npoints = npoints, kwargs...)
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
    else
        return getfield(m, name)
    end
end

function Base.propertynames(m::Minuit, ::Bool = false)
    return (:fcn, :params, :fmin, :minos_errors, :prec,
            # JuMinuit-native
            :values, :errors, :fval, :edm, :nfcn, :valid,
            :covariance, :ndim, :npar,
            # IMinuit.jl-compatible aliases
            :ncalls, :is_valid, :parameters, :fixed, :limits,
            :errordef, :up, :merrors, :accurate)
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
                       strategy=Strategy(1), tol=0.1) -> Minuit

IMinuit.jl-compatible alias for [`migrad!`](@ref). Mutates `m.fmin`
and returns `m`. The `ncall` / `resume` / `precision` kwargs are
accepted for IMinuit.jl interface parity:

  - `ncall::Union{Integer,Nothing}` ≡ `maxfcn` cap (default uses
    JuMinuit's `200 + 100·n + 5·n²` formula).
  - `resume::Bool=true` — if `false`, reset `m.fmin` and `m.minos_errors`
    before running (matches iminuit's `resume` argument).
  - `precision::Union{Real,Nothing}` — override the `MachinePrecision`
    `eps` value (rarely used).

The default `strategy=Strategy(1)` matches iminuit's default; JuMinuit's
native `migrad!` defaults to `Strategy(0)` (faster).
"""
function migrad(m::Minuit;
                 ncall::Union{Integer,Nothing} = nothing,
                 resume::Bool = true,
                 precision::Union{Real,Nothing} = nothing,
                 strategy::Strategy = Strategy(1),
                 tol::Real = 0.1)
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
    hesse(m::Minuit; maxcall=0) -> Minuit

IMinuit.jl-compatible alias. Currently a no-op when `m.fmin` is
already populated (the bounded-aware MIGRAD path runs HESSE inside
for `Strategy ≥ 1`). For `Strategy(0)` runs followed by an
explicit `hesse(m)` call, this would re-run the full numerical
Hessian — Phase 1.x deferred (rarely used directly when migrad
already populates the covariance).

Currently returns `m` unchanged. Workaround: re-run `migrad(m;
strategy=Strategy(2))` to force the full HESSE pass.
"""
function hesse(m::Minuit; maxcall::Integer = 0)
    m.fmin === nothing &&
        throw(ArgumentError("Call `migrad(m)` before `hesse(m)`"))
    # The bounded MIGRAD path already runs HESSE for Strategy ≥ 1.
    # Re-running standalone HESSE here would require re-doing the
    # int↔ext transform plumbing — deferred. Users wanting a guaranteed
    # full HESSE should pass `strategy=Strategy(2)` to `migrad(m)`.
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

function Base.show(io::IO, ::MIME"text/plain", m::Minuit)
    println(io, "JuMinuit.Minuit (iminuit-style wrapper)")
    if m.fmin === nothing
        println(io, "  ── not yet minimized; call `migrad!(m)` ──")
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
    else
        println(io, "  valid:    ", m.valid)
        println(io, "  fval:     ", m.fval)
        println(io, "  edm:      ", m.edm)
        println(io, "  nfcn:     ", m.nfcn)
        println(io, "  parameters (external):")
        for (i, p) in enumerate(m.params.pars)
            val = m.values[i]
            err = m.errors[i]
            fixed_tag = is_fixed(p) ? "  [FIXED]" : ""
            mn_tag = ""
            if haskey(m.minos_errors, i)
                me = m.minos_errors[i]
                mn_tag = "  MINOS: +$(me.upper) -$(-me.lower)"
            end
            println(io, "    [", i, "] ", p.name, " = ", val,
                    " ± ", err, fixed_tag, mn_tag)
        end
    end
end

Base.show(io::IO, m::Minuit) =
    print(io, "Minuit(", n_pars(m.params), " params, ",
              m.fmin === nothing ? "not minimized" : "fval=$(m.fval)", ")")
