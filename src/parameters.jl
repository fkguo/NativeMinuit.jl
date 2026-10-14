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
# ParameterMetadata — per-parameter configuration without the numbers
# (issue #45). Internal type: not exported, not public API until a 1.0
# surface review.
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParameterMetadata

Per-parameter configuration that is NOT numerical coordinate state:
display name, bounds, fixed flag. Carries no value/error fields by
design (issue #45): the canonical numbers live exactly once, in
`Parameters.values` / `Parameters.errors`. Note `fixed` makes this
configuration, not pure presentation data — the index maps and the
`IntIsExt` type parameter of `Parameters` are derived from it at
construction. Internal type (not exported).
"""
struct ParameterMetadata
    name::String
    lower::Float64   # NaN = unbounded below (same sentinel as MinuitParameter)
    upper::Float64   # NaN = unbounded above
    fixed::Bool
end

has_lower_limit(md::ParameterMetadata) = !isnan(md.lower)
has_upper_limit(md::ParameterMetadata) = !isnan(md.upper)
has_limits(md::ParameterMetadata) = has_lower_limit(md) || has_upper_limit(md)
bound_kind(md::ParameterMetadata) = bound_kind(md.lower, md.upper)
is_fixed(md::ParameterMetadata) = md.fixed

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — metadata records + int↔ext index maps + canonical value/error
# vectors + name lookup
# ─────────────────────────────────────────────────────────────────────────────

"""
    Parameters

Per-parameter metadata plus internal/external index mappings, precision
context, and the canonical external-coordinate value and step-size vectors.
Replaces the tightly-coupled C++ pair `MnUserParameters` +
`MnUserTransformation`.

The numbers live exactly **once** (issue #45): `values` and `errors` are the
canonical, single storage of every parameter's external value and initial
step size. No record mirror exists that could desynchronize from them.
`p.pars` remains available as a **derived compatibility view** that lazily
materializes `MinuitParameter` records from `metadata`/`values`/`errors` on
access (see `ParameterRecords`); it is read-only.

Coordinate arrays must use one-based indexing; structured one-based vectors
and ordinary views are supported.

All fields are **read-only** by contract: mutate through the `Minuit` API
(`set_value!`/`set_error!`/`set_limits!`/`fix!`/`release!`, or
`m.values`/`m.errors`/`m.limits`/`m.fixed`), which rebuilds the `Parameters`
consistently. Writing to the fields (or to containers reachable from them)
directly is unsupported.

# Fields

- `metadata::Vector{ParameterMetadata}` — per-parameter name, bounds, fixed
  flag (variable + fixed parameters; no numbers). Internal.
- `ext_of_int::Vector{Int}` — `ext_of_int[i_internal] = external_index`.
  Length = number of free parameters.
- `int_of_ext::Vector{Int}` — `int_of_ext[i_external] = internal_index`, or
  `0` if the external parameter is fixed. Length = total parameters.
- `name_to_ext::Dict{String,Int}` — name → external index (1-based).
- `prec::MachinePrecision` — used for `ext2int` clamping in Sin transform.
- `values::AbstractVector{Float64}` — canonical external values. This is
  also the allocation source for coordinate workspaces, so structured vector
  axes survive through ordinary `similar(values)` operations. The `values=`
  keyword of the record constructor donates only the **container/axes**; the
  numbers always come from the records.
- `errors::AbstractVector{Float64}` — canonical initial step sizes ("errors"
  in iminuit/Minuit2 nomenclature). Allocated as `similar(values, Float64)`,
  so it carries the same structured axes as `values`.

The mappings are computed once at construction; they don't change as
parameters are fixed/released (would require rebuilding).

# The `IntIsExt` type parameter

The second type parameter records whether the minimizer's INTERNAL coordinate
space coincides with the user's EXTERNAL one — true only when every parameter
is free (same dimension) AND unbounded (identity transform). It gates
[`_internal_vector`](@ref): when the two spaces coincide, an internal workspace
is allocated from `values` and a structured container keeps its axes, because
each slot still denotes the parameter its label names. When they differ — a
fixed parameter shortens the vector, a bound makes the stored number an
arcsin/sqrt-transformed coordinate — the internal workspace is a plain
`Vector{Float64}`. Carrying user axis labels on transformed coordinates would
attach a physically meaningful name to a number that is not that quantity.
Structure is still preserved on every external buffer that reaches the user's
objective, gradient, and results, which is what callers actually observe.
"""
struct Parameters{P<:AbstractVector{Float64},IntIsExt}
    metadata::Vector{ParameterMetadata}
    ext_of_int::Vector{Int}
    int_of_ext::Vector{Int}
    name_to_ext::Dict{String,Int}
    prec::MachinePrecision
    values::P
    errors::P
end

function Parameters(pars::Vector{MinuitParameter},
                     prec::MachinePrecision = MachinePrecision();
                     values::AbstractVector = [p.value for p in pars])
    Base.require_one_based_indexing(values)
    n = length(pars)
    length(values) == n ||
        throw(DimensionMismatch(
            "values length $(length(values)) != parameter length $n"))
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
    # Decompose the records: metadata (name/bounds/fixed) goes into its own
    # vector; the numbers for BOTH canonical vectors come from the records —
    # `values[i] := pars[i].value`, `errors[i] := pars[i].error`. The `values=`
    # keyword donates only the container/axes. This is load-bearing for
    # `_build_resume_params`: resume-step records must win over any donor's
    # stored errors.
    metadata = Vector{ParameterMetadata}(undef, n)
    external = similar(values, Float64)
    errors_vec = similar(external, Float64)
    @inbounds for i in 1:n
        p = pars[i]
        metadata[i] = ParameterMetadata(p.name, p.lower, p.upper, p.fixed)
        external[i] = p.value
        errors_vec[i] = p.error
    end
    # Internal ≡ external only when nothing reshapes or reparameterizes the
    # coordinates: no fixed parameter (same length) and no bound (identity
    # transform). See the `IntIsExt` section of the docstring.
    int_is_ext = int_idx == n && !any(has_limits, pars)
    # Type parameters bound from the ACTUAL allocation: `similar(x, Float64)
    # isa typeof(x)` is not part of the AbstractVector contract.
    return Parameters{typeof(external),int_is_ext}(
        metadata, ext_of_int, int_of_ext, name_to_ext, prec, external,
        errors_vec)
end

Parameters(pars::Vector{MinuitParameter}, source::Parameters) =
    Parameters(pars, source.prec; values = source.values)

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
    Base.require_one_based_indexing(names, values, errors)
    limits === nothing || Base.require_one_based_indexing(limits)
    fixed === nothing || Base.require_one_based_indexing(fixed)
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
    return Parameters(pars, prec; values = values)
end

# ─────────────────────────────────────────────────────────────────────────────
# Accessors
# ─────────────────────────────────────────────────────────────────────────────

"`n_pars(p)` — total external parameter count (variable + fixed)."
n_pars(p::Parameters) = length(getfield(p, :metadata))

"`n_free(p)` — variable (non-fixed) parameter count."
n_free(p::Parameters) = length(getfield(p, :ext_of_int))

Base.length(p::Parameters) = n_pars(p)
is_fixed(p::Parameters, ext_idx::Integer) =
    getfield(p, :metadata)[ext_idx].fixed

# Function-barrier accessor for callers holding an abstractly-typed
# `Parameters` (e.g. the `Minuit.params` field): specializes once on the
# concrete container type and copies the canonical step-size store into a
# plain `Vector{Float64}`. Explicit element loop — `collect(Float64, x)`
# would preserve a structured container, which is not wanted here.
function _config_step_vector(p::Parameters)
    errs = getfield(p, :errors)
    out = Vector{Float64}(undef, length(errs))
    @inbounds for i in eachindex(out)
        out[i] = errs[i]
    end
    return out
end

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
    ext_idx = getfield(p, :ext_of_int)[int_idx]
    md = getfield(p, :metadata)[ext_idx]
    return int2ext(bound_kind(md), Float64(int_val), md.lower, md.upper)
end

"""
    ext_to_int_value(p, ext_idx, ext_val) -> Float64

Convert one external value to internal. The `ext_idx` is the
*external* (full-list) parameter index. Mirrors
`MnUserTransformation::Ext2int` at
`reference/Minuit2_cpp/src/MnUserTransformation.cxx:122-140`.
"""
function ext_to_int_value(p::Parameters, ext_idx::Integer, ext_val::Real)
    md = getfield(p, :metadata)[ext_idx]
    return ext2int(bound_kind(md), Float64(ext_val), md.lower, md.upper,
                   getfield(p, :prec))
end

"""
    dint2ext_value(p, int_idx, int_val) -> Float64

`d(ext)/d(int)` for one internal parameter — used by the gradient
chain rule and covariance transformation.
"""
function dint2ext_value(p::Parameters, int_idx::Integer, int_val::Real)
    ext_idx = getfield(p, :ext_of_int)[int_idx]
    md = getfield(p, :metadata)[ext_idx]
    return dint2ext(bound_kind(md), Float64(int_val), md.lower, md.upper)
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
    Base.require_one_based_indexing(ext, int_vec)
    length(int_vec) == n_free(p) ||
        throw(DimensionMismatch("int_vec length $(length(int_vec)) != n_free $(n_free(p))"))
    length(ext) == n_pars(p) ||
        throw(DimensionMismatch("ext length $(length(ext)) != n_pars $(n_pars(p))"))
    metadata = getfield(p, :metadata)
    values = getfield(p, :values)
    int_of_ext = getfield(p, :int_of_ext)
    @inbounds for ext_idx in 1:n_pars(p)
        md = metadata[ext_idx]
        if md.fixed
            ext[ext_idx] = values[ext_idx]
        else
            int_idx = int_of_ext[ext_idx]
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
    int_to_ext_vector!(similar(getfield(p, :values), Float64), p, int_vec)

"""
    ext_to_int_vector(p, ext_vec) -> Vector{Float64}

Map the full external vector to the free-parameter internal vector.
Fixed parameters are dropped. Length: `n_free(p)`.
"""
function ext_to_int_vector(p::Parameters, ext_vec::AbstractVector{<:Real})
    Base.require_one_based_indexing(ext_vec)
    length(ext_vec) == n_pars(p) ||
        throw(DimensionMismatch("ext_vec length $(length(ext_vec)) != n_pars $(n_pars(p))"))
    int = Vector{Float64}(undef, n_free(p))
    ext_of_int = getfield(p, :ext_of_int)
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = ext_of_int[int_idx]
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
    int = _internal_vector(p)
    metadata = getfield(p, :metadata)
    values = getfield(p, :values)
    ext_of_int = getfield(p, :ext_of_int)
    prec = getfield(p, :prec)
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = ext_of_int[int_idx]
        md = metadata[ext_idx]
        int[int_idx] = ext2int(bound_kind(md), values[ext_idx],
                              md.lower, md.upper, prec)
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
    errs = _internal_vector(p)
    metadata = getfield(p, :metadata)
    ext_errors = getfield(p, :errors)
    ext_of_int = getfield(p, :ext_of_int)
    prec = getfield(p, :prec)
    eps2 = prec.eps2
    @inbounds for int_idx in 1:n_free(p)
        ext_idx = ext_of_int[int_idx]
        md = metadata[ext_idx]
        if !has_limits(md)
            errs[int_idx] = ext_errors[ext_idx]
        else
            kind = bound_kind(md.lower, md.upper)
            var = int_vals[int_idx]
            sav = int2ext(kind, var, md.lower, md.upper)
            werr = ext_errors[ext_idx]

            # Forward perturbation, clamped at the upper bound if present
            sav_plus = sav + werr
            if kind == BothBounds || kind == UpperOnly
                if sav_plus > md.upper
                    sav_plus = md.upper
                end
            end
            var_plus = ext2int(kind, sav_plus, md.lower, md.upper, prec)
            vplu = var_plus - var

            # Backward perturbation, clamped at the lower bound if present
            sav_minus = sav - werr
            if kind == BothBounds || kind == LowerOnly
                if sav_minus < md.lower
                    sav_minus = md.lower
                end
            end
            var_minus = ext2int(kind, sav_minus, md.lower, md.upper, prec)
            vmin = var_minus - var

            gsmin = 8.0 * eps2 * (abs(var) + eps2)
            errs[int_idx] = max(0.5 * (abs(vplu) + abs(vmin)), gsmin)
        end
    end
    return errs
end

# Container choice is encoded in `Parameters`' second type parameter
# (`IntIsExt`). The all-free, unbounded path infers a structured active vector
# whose labels still name the parameters they hold; a fixed-parameter
# projection or a bounded reparameterization gets an honestly different
# dense-vector type rather than mislabeled transformed coordinates.
_internal_vector(p::Parameters{P,true}) where {P} =
    similar(getfield(p, :values), Float64)
_internal_vector(p::Parameters{P,false}) where {P} =
    Vector{Float64}(undef, n_free(p))

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility view: `p.pars` — lazily materialized MinuitParameter records
# (issue #45). The canonical numbers live in `values`/`errors`; the view
# derives each record on access, so it can never desynchronize. Internal
# type: not exported.
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParameterRecords

Read-only `AbstractVector{MinuitParameter}` view over a `Parameters`,
returned by `p.pars`. Each `getindex` materializes a `MinuitParameter`
from the canonical `metadata`/`values`/`errors` stores, so
`p.pars[i] isa MinuitParameter` holds and the record is always in sync
with the stored numbers (it is derived, not stored). `setindex!` throws:
mutate through the `Minuit` API instead. Internal type (not exported).
"""
struct ParameterRecords{P<:Parameters} <: AbstractVector{MinuitParameter}
    p::P
end

Base.size(v::ParameterRecords) = (length(getfield(getfield(v, :p), :metadata)),)
Base.IndexStyle(::Type{<:ParameterRecords}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(v::ParameterRecords, i::Int)
    @boundscheck checkbounds(v, i)
    p = getfield(v, :p)
    md = getfield(p, :metadata)[i]
    # Direct 6-positional-arg inner constructor: the fields are already
    # validated Float64s; the validating keyword constructor is not needed.
    return MinuitParameter(md.name, getfield(p, :values)[i],
                           getfield(p, :errors)[i],
                           md.lower, md.upper, md.fixed)
end

# Concrete signature (not a varargs catch-all): stays out of Aqua's
# method-ambiguity checks and gives every write path — including generic
# code like `sort!` that routes through `setindex!` — the guiding error.
function Base.setindex!(v::ParameterRecords, val, i::Int)
    throw(ArgumentError(
        "p.pars is a derived read-only view; mutate through set_value!/" *
        "set_error!/set_limits!/fix!/release! (or m.values/m.errors/" *
        "m.limits/m.fixed)"))
end

# `@inline` is load-bearing: the `s === :pars` branch must constant-fold
# away for literal-symbol field reads on the per-FCN hot path, and the
# heuristic inliner must not be trusted with that (an allocation gate does
# not detect added call overhead).
Base.@inline function Base.getproperty(p::Parameters, s::Symbol)
    s === :pars ? ParameterRecords(p) : getfield(p, s)
end

# Advertised property surface = the supported read surface; internal fields
# remain reachable but are only listed under `private = true`
# (completeness/debugging).
Base.propertynames(p::Parameters, private::Bool = false) =
    private ? (:pars, fieldnames(Parameters)...) :
              (:pars, :values, :errors, :prec)
