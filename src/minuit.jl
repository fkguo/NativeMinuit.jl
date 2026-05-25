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
    names::Union{Vector{<:AbstractString},Nothing} = nothing,
    errors::Union{Vector{<:Real},Nothing} = nothing,
    limits::Union{Vector,Nothing} = nothing,
    fixed::Union{Vector{Bool},Nothing} = nothing,
    up::Real = 1.0,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(x0)
    nm = names === nothing ? ["p$i" for i in 1:n] : names
    er = errors === nothing ? fill(0.1, n) : errors
    fx = fixed === nothing ? fill(false, n) : fixed
    lim = limits === nothing ? fill(nothing, n) : limits

    n == length(nm) == length(er) == length(fx) == length(lim) ||
        throw(ArgumentError("Minuit: x0/names/errors/limits/fixed length mismatch"))

    # Translate the iminuit-style limits parametrization into the
    # (lower, upper) tuple format Parameters expects (NaN = absent).
    limit_tuples = Vector{Tuple{Float64,Float64}}(undef, n)
    for i in 1:n
        lo_i, up_i = NaN, NaN
        l = lim[i]
        if l !== nothing
            lo_raw, up_raw = l
            if lo_raw !== nothing
                lo_i = Float64(lo_raw)
            end
            if up_raw !== nothing
                up_i = Float64(up_raw)
            end
        end
        limit_tuples[i] = (lo_i, up_i)
    end
    params = Parameters(String.(nm), Float64.(x0), Float64.(er);
                         limits = limit_tuples, fixed = collect(Bool, fx),
                         prec = prec)
    cf = CostFunction(fcn, Float64(up))
    return Minuit(cf, params, nothing, Dict{Int,MinosError}(), prec)
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
    m.minos_errors[Int(par)] = err
    return m
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
    elseif name === :valid
        return m.fmin === nothing ? false : is_valid(m.fmin)
    elseif name === :covariance
        return m.fmin === nothing ? nothing : ext_covariance(m.fmin)
    elseif name === :ndim
        return n_pars(m.params)
    elseif name === :npar
        return n_free(m.params)
    else
        return getfield(m, name)
    end
end

function Base.propertynames(m::Minuit, ::Bool = false)
    return (:fcn, :params, :fmin, :minos_errors, :prec,
            :values, :errors, :fval, :edm, :nfcn, :valid,
            :covariance, :ndim, :npar)
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
