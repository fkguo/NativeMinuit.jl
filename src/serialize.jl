# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# serialize.jl — Result <-> Dict round-trip (Phase 2.5).
#
# Converts FunctionMinimum / BoundedFunctionMinimum / MinosError /
# ContoursError into plain `Dict{String,Any}` representations suitable
# for JSON/JLD2 storage. The inverse `from_dict` reconstructs the
# result type without re-running MIGRAD (useful for CI regression
# baselines and reproducible publication artifacts).
#
# Phase 2.5 first cut: round-trip the data fields, NOT the internal
# CostFunction (which carries closures). Useful when the user wants
# to ship a fit result as JSON without rerunning the optimizer.
# ─────────────────────────────────────────────────────────────────────────────

"""
    to_dict(m::BoundedFunctionMinimum) -> Dict{String,Any}

Serialize a `BoundedFunctionMinimum` to a `Dict{String,Any}` for JSON
or JLD2 storage. Captures all numeric state needed to reconstruct the
external-coordinate result; does NOT capture the user FCN.
"""
function to_dict(m::BoundedFunctionMinimum)
    cov = ext_covariance(m)
    return Dict{String,Any}(
        "type" => "BoundedFunctionMinimum",
        "fval" => fval(m),
        "edm"  => edm(m),
        "nfcn" => nfcn(m),
        "valid" => is_valid(m),
        "reached_call_limit" => m.internal.reached_call_limit,
        "above_max_edm" => m.internal.above_max_edm,
        "nonfinite_fval" => m.internal.nonfinite_fval,
        "n_nonfinite_calls" => m.internal.n_nonfinite_calls,
        "ext_values" => collect(m.ext_values),
        "ext_errors" => collect(m.ext_errors),
        "ext_covariance" => cov === nothing ? nothing : collect(cov),
        "param_names"  => [p.name  for p in m.params.pars],
        "param_lower"  => [isnan(p.lower) ? nothing : p.lower for p in m.params.pars],
        "param_upper"  => [isnan(p.upper) ? nothing : p.upper for p in m.params.pars],
        "param_fixed"  => [p.fixed for p in m.params.pars],
        "errordef" => m.internal.up,
    )
end

"""
    to_dict(m::FunctionMinimum) -> Dict{String,Any}

Serialize an unbounded [`FunctionMinimum`](@ref). Simpler than the
bounded variant — no parameter metadata.
"""
function to_dict(m::FunctionMinimum)
    cov = covariance(m)
    return Dict{String,Any}(
        "type" => "FunctionMinimum",
        "fval" => fval(m),
        "edm"  => edm(m),
        "nfcn" => nfcn(m),
        "valid" => is_valid(m),
        "reached_call_limit" => m.reached_call_limit,
        "above_max_edm" => m.above_max_edm,
        "nonfinite_fval" => m.nonfinite_fval,
        "n_nonfinite_calls" => m.n_nonfinite_calls,
        "values" => collect(Base.values(m)),
        "covariance" => cov === nothing ? nothing : collect(cov),
        "errordef" => m.up,
    )
end

"""
    to_dict(e::MinosError) -> Dict{String,Any}

Serialize a [`MinosError`](@ref).
"""
function to_dict(e::MinosError)
    return Dict{String,Any}(
        "type" => "MinosError",
        "par_idx" => e.par_idx,
        "min_par_value" => e.min_par_value,
        "upper" => e.upper,
        "lower" => e.lower,
        "upper_valid" => e.upper_valid,
        "lower_valid" => e.lower_valid,
        "upper_new_min" => e.upper_new_min,
        "lower_new_min" => e.lower_new_min,
        "upper_fcn_limit" => e.upper_fcn_limit,
        "lower_fcn_limit" => e.lower_fcn_limit,
        "nfcn" => e.nfcn,
    )
end

"""
    to_dict(c::ContoursError) -> Dict{String,Any}

Serialize a [`ContoursError`](@ref).
"""
function to_dict(c::ContoursError)
    return Dict{String,Any}(
        "type" => "ContoursError",
        "par_x" => c.par_x,
        "par_y" => c.par_y,
        "points" => [collect(p) for p in c.points],  # vector of [x, y]
        "minos_x" => to_dict(c.minos_x),
        "minos_y" => to_dict(c.minos_y),
        "nfcn" => c.nfcn,
        "valid" => c.valid,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Deserialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    minos_error_from_dict(d::AbstractDict) -> MinosError

Reconstruct a `MinosError` from a `to_dict`-produced dictionary.
"""
function minos_error_from_dict(d::AbstractDict)
    return MinosError(
        Int(d["par_idx"]),
        Float64(d["min_par_value"]),
        Float64(d["upper"]),
        Float64(d["lower"]),
        Bool(d["upper_valid"]),
        Bool(d["lower_valid"]),
        Bool(d["upper_new_min"]),
        Bool(d["lower_new_min"]),
        Bool(d["upper_fcn_limit"]),
        Bool(d["lower_fcn_limit"]),
        Int(d["nfcn"]),
    )
end

"""
    minimum_summary_from_dict(d::AbstractDict) -> NamedTuple

Reconstruct a lightweight summary from a serialized
`BoundedFunctionMinimum` or `FunctionMinimum` dict. Returns a NamedTuple
with `(; fval, edm, nfcn, valid, values, errors, covariance,
parameter_names)`. We do NOT rebuild the full Julia struct (which
contains closure-captured references); the NamedTuple is the
contract for JSON-roundtripped use.
"""
function minimum_summary_from_dict(d::AbstractDict)
    typ = get(d, "type", "")
    is_bounded = typ == "BoundedFunctionMinimum"
    values_field = is_bounded ? "ext_values" : "values"
    errors_field = is_bounded ? "ext_errors" : nothing
    cov_field = is_bounded ? "ext_covariance" : "covariance"

    values_arr = haskey(d, values_field) ? Float64.(d[values_field]) : Float64[]
    errors_arr = if errors_field !== nothing && haskey(d, errors_field)
        Float64.(d[errors_field])
    else
        # For unbounded, derive errors from sqrt(diag(cov))
        cov = get(d, cov_field, nothing)
        if cov === nothing || cov isa Nothing
            Float64[]
        else
            cov_mat = reduce(hcat, [Float64.(r) for r in cov])
            [sqrt(max(cov_mat[i, i], 0.0)) for i in 1:size(cov_mat, 1)]
        end
    end

    cov_obj = get(d, cov_field, nothing)
    cov_mat = if cov_obj === nothing || cov_obj isa Nothing
        nothing
    else
        # Convert list-of-list back to Matrix
        rows = [Float64.(r) for r in cov_obj]
        n = length(rows)
        m = Matrix{Float64}(undef, n, length(rows[1]))
        for (i, r) in enumerate(rows)
            m[i, :] = r
        end
        m
    end

    names = haskey(d, "param_names") ? String.(d["param_names"]) :
             ["p$i" for i in 1:length(values_arr)]

    return (
        fval = Float64(d["fval"]),
        edm  = Float64(d["edm"]),
        nfcn = Int(d["nfcn"]),
        valid = Bool(d["valid"]),
        values = values_arr,
        errors = errors_arr,
        covariance = cov_mat,
        parameter_names = names,
        type = typ,
    )
end
