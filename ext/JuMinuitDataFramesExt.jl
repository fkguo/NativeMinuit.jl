# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# JuMinuitDataFramesExt — `Data(::DataFrame)` IMinuit.jl drop-in compat.
#
# IMinuit.jl exposes `Data(df::DataFrame) = Data(df[:,1], df[:,2], df[:,3])`
# (assumes 3 columns interpreted as x, y, err). This was missing from the
# JuMinuit core API, breaking notebooks that pass DataFrames directly
# (e.g., BenchmarkExamples/IAM_2Pformfactor/iamfit.ipynb).
#
# Added as a Package Extension rather than a hard dependency: DataFrames
# pulls in PrettyTables + Compat + a stack of CSV-ecosystem deps, so
# making it mandatory would inflate JuMinuit's install footprint. Users
# who already have DataFrames loaded get the convenience method for free.
# ─────────────────────────────────────────────────────────────────────────────

module JuMinuitDataFramesExt

using JuMinuit
using DataFrames

"""
    JuMinuit.Data(df::DataFrame)

Construct a `Data` from a 3-column DataFrame; columns interpreted as
`(x, y, err)` by position (not by name). Mirrors IMinuit.jl's
`Data(::DataFrame)` for drop-in notebook compatibility.

# Examples

```julia
using DataFrames, CSV, JuMinuit
df = DataFrame(CSV.File("data.csv"; header=[:w, :y, :err]))
d = Data(df)        # → Data(df[:,1], df[:,2], df[:,3])
```

If you need explicit column selection by name, use the 3-arg form:
`Data(df.w, df.y, df.err)`.
"""
JuMinuit.Data(df::DataFrame) = JuMinuit.Data(df[:, 1], df[:, 2], df[:, 3])

# ─────────────────────────────────────────────────────────────────────────────
# Resampling-result → DataFrame (resampling.jl). Per-parameter summary tables;
# the raw θ̂ sample matrix stays in `r.samples` (one row per resample).
# ─────────────────────────────────────────────────────────────────────────────

"""
    DataFrame(r::BootstrapResult)

Per-parameter bootstrap summary: `parameter`, `estimate` (θ̂_full), `mean`,
`std` (bootstrap SE), `ci_lower`, `ci_upper` (percentile CI at `r.ci_level`).
The raw `nresample × npar` θ̂ matrix is in `r.samples`.
"""
function DataFrames.DataFrame(r::JuMinuit.BootstrapResult)
    return DataFrame(parameter = r.names,
                     estimate = r.estimate,
                     mean = r.mean,
                     std = r.std,
                     ci_lower = r.ci_lower,
                     ci_upper = r.ci_upper)
end

"""
    DataFrame(r::JackknifeResult)

Per-parameter jackknife summary: `parameter`, `estimate` (θ̂_full), `mean` (θ̄),
`bias`, `bias_corrected`, `variance`, `std` (jackknife SE). The raw
`g × npar` leave-one-out matrix is in `r.samples`.
"""
function DataFrames.DataFrame(r::JuMinuit.JackknifeResult)
    return DataFrame(parameter = r.names,
                     estimate = r.estimate,
                     mean = r.mean,
                     bias = r.bias,
                     bias_corrected = r.bias_corrected,
                     variance = r.variance,
                     std = r.std)
end

"""
    JuMinuit.contour_df_samples(m::Minuit; kwargs...) -> DataFrame

`DataFrame` of the accepted Monte-Carlo parameter sets from
[`JuMinuit.get_contours_samples`](@ref): one row per kept set, one column
per free parameter (named after the parameters), plus a `:delta_chisq`
column carrying each set's true Δχ². All
[`JuMinuit.get_contours_samples`](@ref) keyword arguments are forwarded.

```julia
using DataFrames, JuMinuit
df = contour_df_samples(m; nsamples = 30_000, cl = 1)
df.delta_chisq          # each kept set's χ²(x) − χ²_min (in χ² units)
```
"""
function JuMinuit.contour_df_samples(m::JuMinuit.Minuit; kwargs...)
    res = JuMinuit.get_contours_samples(m; kwargs...)
    # Each matrix column is one free parameter; rows are accepted samples.
    df = DataFrame(res.samples, Symbol.(res.free_names); makeunique = true)
    # Per-sample true Δχ² (χ²-equivalent FCN delta). `makeunique` above
    # guards parameter-name collisions; guard the diagnostic column too.
    dcol = :delta_chisq in propertynames(df) ? :delta_chisq_ : :delta_chisq
    df[!, dcol] = res.delta_chisq_values
    return df
end

end # module JuMinuitDataFramesExt
