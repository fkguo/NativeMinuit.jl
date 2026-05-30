# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# parameters.jl — MinuitParameter + Parameters (collapsed MnUserParameters
# + MnUserTransformation per parallel-review #1 D5 finding).
#
# Mirrors:
#   reference/Minuit2_cpp/inc/Minuit2/MinuitParameter.h
#   reference/Minuit2_cpp/inc/Minuit2/MnUserTransformation.h (the bulk)
#   reference/Minuit2_cpp/src/MnUserTransformation.cxx
#
# In C++, `MnUserParameters` is a thin wrapper that owns a
# `MnUserTransformation` (which itself holds `vector<MinuitParameter> +
# fExtOfInt + Sin/SqrtUp/SqrtLow transformation objects + cache`). The
# two were tightly coupled. Julia collapses them into a single
# `Parameters` struct since shared_ptr indirection isn't needed.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MinuitParameter

A single fit parameter — name, value, step size, optional bounds,
optional fixed flag. Mirrors C++ `MinuitParameter` from
`reference/Minuit2_cpp/inc/Minuit2/MinuitParameter.h`.

# Fields

- `name::String` — display name.
- `value::Float64` — current external value.
- `error::Float64` — initial step size (also called "error" in
  iminuit/Minuit2 nomenclature).
- `lower::Float64` — lower bound (`NaN` if unbounded below).
- `upper::Float64` — upper bound (`NaN` if unbounded above).
- `fixed::Bool` — `true` if parameter is currently fixed (excluded
  from optimization).

`NaN` is the explicit "absent bound" sentinel, matching the
`bound_kind` classifier in `transform.jl`.
"""
struct MinuitParameter
    name::String
    value::Float64
    error::Float64
    lower::Float64
    upper::Float64
    fixed::Bool
end

function MinuitParameter(name::AbstractString, value::Real, error::Real;
                          lower::Real = NaN, upper::Real = NaN,
                          fixed::Bool = false)
    lo = Float64(lower)
    up = Float64(upper)
    # Validate via the same logic transform.jl uses.
    if !isnan(lo) && !isnan(up) && !(lo < up)
        throw(ArgumentError("MinuitParameter '$name': lower ($lo) must be < upper ($up)"))
    end
    return MinuitParameter(String(name), Float64(value), Float64(error), lo, up, fixed)
end

has_lower_limit(p::MinuitParameter) = !isnan(p.lower)
has_upper_limit(p::MinuitParameter) = !isnan(p.upper)
has_limits(p::MinuitParameter) = has_lower_limit(p) || has_upper_limit(p)
bound_kind(p::MinuitParameter) = bound_kind(p.lower, p.upper)
is_fixed(p::MinuitParameter) = p.fixed

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — vector of MinuitParameter + int↔ext index maps + name lookup
# ─────────────────────────────────────────────────────────────────────────────

"""
    Parameters

Collection of `MinuitParameter`s plus internal/external index mappings
+ precision context. Replaces the tightly-coupled C++ pair
`MnUserParameters` + `MnUserTransformation`.

# Fields

- `pars::Vector{MinuitParameter}` — full parameter list (variable + fixed).
- `ext_of_int::Vector{Int}` — `ext_of_int[i_internal] = external_index`.
  Length = number of free parameters.
- `int_of_ext::Vector{Int}` — `int_of_ext[i_external] = internal_index`, or
  `0` if the external parameter is fixed. Length = total parameters.
- `name_to_ext::Dict{String,Int}` — name → external index (1-based).
- `prec::MachinePrecision` — used for `ext2int` clamping in Sin transform.

The mappings are computed once at construction; they don't change as
parameters are fixed/released (would require rebuilding).
"""
struct Parameters
    pars::Vector{MinuitParameter}
    ext_of_int::Vector{Int}
    int_of_ext::Vector{Int}
    name_to_ext::Dict{String,Int}
    prec::MachinePrecision
end

function Parameters(pars::Vector{MinuitParameter},
                     prec::MachinePrecision = MachinePrecision())
    n = length(pars)
    ext_of_int = Int[]
    int_of_ext = Vector{Int}(undef, n)
    name_to_ext = Dict{String,Int}()
    int_idx = 0
    for (ext_idx, p) in enumerate(pars)
        if haskey(name_to_ext, p.name)
            throw(ArgumentError("duplicate parameter name: \"$(p.name)\""))
        end
        name_to_ext[p.name] = ext_idx
        if p.fixed
            int_of_ext[ext_idx] = 0
        else
            int_idx += 1
            push!(ext_of_int, ext_idx)
            int_of_ext[ext_idx] = int_idx
        end
    end
    return Parameters(pars, ext_of_int, int_of_ext, name_to_ext, prec)
end

"""
    Parameters(names, values, errors;
               limits=nothing, fixed=nothing, prec=MachinePrecision())

Vector-style convenience constructor. `names`, `values`, `errors` are
each length-n. `limits` (if provided) is a length-n vector of
`(lower, upper)` tuples (use `(NaN, NaN)` for unbounded). `fixed` (if
provided) is a length-n vector of Bool.
"""
function Parameters(
    names::AbstractVector,
    values::AbstractVector{<:Real},
    errors::AbstractVector{<:Real};
    limits::Union{Nothing,AbstractVector} = nothing,
    fixed::Union{Nothing,AbstractVector{Bool}} = nothing,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(names)
    length(values) == n && length(errors) == n ||
        throw(DimensionMismatch("names/values/errors must be the same length"))
    limits === nothing || length(limits) == n ||
        throw(DimensionMismatch("limits must be length $n"))
    fixed === nothing || length(fixed) == n ||
        throw(DimensionMismatch("fixed must be length $n"))

    pars = Vector{MinuitParameter}(undef, n)
    @inbounds for i in 1:n
        lo, up = limits === nothing ? (NaN, NaN) : Tuple(limits[i])
        fx = fixed === nothing ? false : fixed[i]
        pars[i] = MinuitParameter(String(names[i]), Float64(values[i]),
                                   Float64(errors[i]);
                                   lower = lo, upper = up, fixed = fx)
    end
    return Parameters(pars, prec)
end

# ─────────────────────────────────────────────────────────────────────────────
# Accessors
# ─────────────────────────────────────────────────────────────────────────────

"`n_pars(p)` — total external parameter count (variable + fixed)."
n_pars(p::Parameters) = length(p.pars)

"`n_free(p)` — variable (non-fixed) parameter count."
n_free(p::Parameters) = length(p.ext_of_int)

Base.length(p::Parameters) = n_pars(p)
is_fixed(p::Parameters, ext_idx::Integer) = p.pars[ext_idx].fixed

"`ext_index(p, name)` — external index for parameter named `name` (1-based)."
ext_index(p::Parameters, name::AbstractString) =
    get(p.name_to_ext, String(name)) do
        throw(KeyError("parameter \"$name\" not in Parameters"))
    end

# ─────────────────────────────────────────────────────────────────────────────
# Internal ↔ external conversion
# ─────────────────────────────────────────────────────────────────────────────

"""
    int_to_ext_value(p, int_idx, int_val) -> Float64

Convert one internal-parameter value to external. Mirrors
`MnUserTransformation::Int2ext` at
`reference/Minuit2_cpp/src/MnUserTransformation.cxx:99-118`.
"""
function int_to_ext_value(p::Parameters, int_idx::Integer, int_val::Real)
    ext_idx = p.ext_of_int[int_idx]
    par = p.pars[ext_idx]
    return int2ext(bound_kind(par), Float64(int_val), par.lower, par.upper)
end

"""
    ext_to_int_value(p, ext_idx, ext_val) -> Float64

Convert one external value to internal. The `ext_idx` is the
*external* (full-list) parameter index. Mirrors
`MnUserTransformation::Ext2int` at
`reference/Minuit2_cpp/src/MnUserTransformation.cxx:122-140`.
"""
function ext_to_int_value(p::Parameters, ext_idx::Integer, ext_val::Real)
    par = p.pars[ext_idx]
    return ext2int(bound_kind(par), Float64(ext_val), par.lower, par.upper, p.prec)
end

"""
    dint2ext_value(p, int_idx, int_val) -> Float64

`d(ext)/d(int)` for one internal parameter — used by the gradient
chain rule and covariance transformation.
"""
function dint2ext_value(p::Parameters, int_idx::Integer, int_val::Real)
    ext_idx = p.ext_of_int[int_idx]
    par = p.pars[ext_idx]
    return dint2ext(bound_kind(par), Float64(int_val), par.lower, par.upper)
end

"""
    int_to_ext_vector!(ext, p, int_vec) -> ext

In-place form of [`int_to_ext_vector`](@ref): write the full external
vector into the caller-supplied `ext` (length `n_pars(p)`) and return
it. Lets hot-path callers — the MIGRAD internal→external FCN wrappers,
invoked once per FCN evaluation — reuse a buffer instead of allocating
a fresh `Vector` every call. The result is bit-identical to the
allocating method.
"""
function int_to_ext_vector!(ext::AbstractVector{<:Real}, p::Parameters,
                            int_vec::AbstractVector{<:Real})
    length(int_vec) == n_free(p) ||
        throw(DimensionMismatch("int_vec length $(length(int_vec)) != n_free $(n_free(p))"))
    length(ext) == n_pars(p) ||
        throw(DimensionMismatch("ext length $(length(ext)) != n_pars $(n_pars(p))"))
    @inbounds for ext_idx in 1:n_pars(p)
        par = p.pars[ext_idx]
        if par.fixed
            ext[ext_idx] = par.value
        else
            int_idx = p.int_of_ext[ext_idx]
            ext[ext_idx] = int_to_ext_value(p, int_idx, int_vec[int_idx])
        end
    end
    return ext
end

"""
    int_to_ext_vector(p, int_vec) -> Vector{Float64}

Map a free-parameter internal vector (length `n_free(p)`) to the full
external vector (length `n_pars(p)`). Fixed-parameter entries take
their static `pars[ext_idx].value`. Mirrors the C++
`MnUserTransformation::operator()(const MnAlgebraicVector&)`.

Allocates a fresh result; see [`int_to_ext_vector!`](@ref) for the
in-place, buffer-reusing variant used on the per-FCN-call hot path.
"""
int_to_ext_vector(p::Parameters, int_vec::AbstractVector{<:Real}) =
    int_to_ext_vector!(Vector{Float64}(undef, n_pars(p)), p, int_vec)

"""
    ext_to_int_vector(p, ext_vec) -> Vector{Float64}

Map the full external vector to the free-parameter internal vector.
Fixed parameters are dropped. Length: `n_free(p)`.
"""
function ext_to_int_vector(p::Parameters, ext_vec::AbstractVector{<:Real})
    length(ext_vec) == n_pars(p) ||
        throw(DimensionMismatch("ext_vec length $(length(ext_vec)) != n_pars $(n_pars(p))"))
    int = Vector{Float64}(undef, n_free(p))
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = p.ext_of_int[int_idx]
        int[int_idx] = ext_to_int_value(p, ext_idx, ext_vec[ext_idx])
    end
    return int
end

# ─────────────────────────────────────────────────────────────────────────────
# Initial-state helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    initial_int_values(p) -> Vector{Float64}

Compute the initial internal (unbounded) values of all FREE parameters,
applying ext2int on the user-supplied initial values.
"""
function initial_int_values(p::Parameters)
    int = Vector{Float64}(undef, n_free(p))
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = p.ext_of_int[int_idx]
        par = p.pars[ext_idx]
        int[int_idx] = ext2int(bound_kind(par), par.value, par.lower, par.upper, p.prec)
    end
    return int
end

"""
    initial_int_errors(p) -> Vector{Float64}

Initial step sizes for FREE parameters in internal coordinates.

For unbounded parameters this is the identity (`int_err = ext_err`).
For bounded parameters we mirror the C++ two-sided perturbation at
`reference/Minuit2_cpp/src/InitialGradientCalculator.cxx:43-63`:

1. Compute the external position `sav = int2ext(int_val)`.
2. Forward step: `sav_plus = min(sav + werr, upper)` clamped to the bound;
   map back: `var_plus = ext2int(sav_plus)`; `vplu = var_plus - int_val`.
3. Backward step: `sav_minus = max(sav - werr, lower)` clamped;
   `vmin = ext2int(sav_minus) - int_val`.
4. Floor at machine-precision step: `gsmin = 8·eps2·(|int_val| + eps2)`.
5. `int_err = max(0.5·(|vplu| + |vmin|), gsmin)`.

The v1 of this function used `int_err = ext_err / |dext/dint|` (a
first-order Taylor expansion). Near a bound that diverged from C++
because `d(ext)/d(int) → 0` makes the Taylor estimate blow up where
the C++ two-sided formula clamps gracefully (parallel-review #2 B4).
"""
function initial_int_errors(p::Parameters)
    int_vals = initial_int_values(p)
    errs = Vector{Float64}(undef, n_free(p))
    eps2 = p.prec.eps2
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = p.ext_of_int[int_idx]
        par = p.pars[ext_idx]
        if !has_limits(par)
            errs[int_idx] = par.error
        else
            kind = bound_kind(par.lower, par.upper)
            var = int_vals[int_idx]
            sav = int2ext(kind, var, par.lower, par.upper)
            werr = par.error

            # Forward perturbation, clamped at the upper bound if present
            sav_plus = sav + werr
            if kind == BothBounds || kind == UpperOnly
                if sav_plus > par.upper
                    sav_plus = par.upper
                end
            end
            var_plus = ext2int(kind, sav_plus, par.lower, par.upper, p.prec)
            vplu = var_plus - var

            # Backward perturbation, clamped at the lower bound if present
            sav_minus = sav - werr
            if kind == BothBounds || kind == LowerOnly
                if sav_minus < par.lower
                    sav_minus = par.lower
                end
            end
            var_minus = ext2int(kind, sav_minus, par.lower, par.upper, p.prec)
            vmin = var_minus - var

            gsmin = 8.0 * eps2 * (abs(var) + eps2)
            errs[int_idx] = max(0.5 * (abs(vplu) + abs(vmin)), gsmin)
        end
    end
    return errs
end
