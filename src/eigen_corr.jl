# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# eigen_corr.jl — MnEigen + MnGlobalCorrelationCoeff ports.
#
# Mirrors reference/Minuit2_cpp/src/MnEigen.cxx and
# reference/Minuit2_cpp/src/MnGlobalCorrelationCoeff.cxx.
#
# Two small diagnostic accessors on the covariance matrix:
# - eigenvalues(cov) — eigenvalues of the covariance, sorted ascending.
#   Small/negative eigenvalues flag ill-conditioning.
# - global_cc(cov)  — Global Correlation Coefficient per parameter,
#   `ρ_i = sqrt(1 - 1/(C_ii · C⁻¹_ii))`. Measures how much the i-th
#   parameter is statistically determined by the OTHERS — `ρ_i → 1`
#   means parameter `i` is essentially a function of the rest (full
#   degeneracy), `ρ_i → 0` means it's independent.
# ─────────────────────────────────────────────────────────────────────────────

"""
    eigenvalues(cov::AbstractMatrix{<:Real}) -> Vector{Float64}

Eigenvalues of a covariance matrix, sorted ascending. Mirrors
`MnEigen::operator()` in `reference/Minuit2_cpp/src/MnEigen.cxx`.

`cov` may be `Symmetric{Float64}`, `Matrix{Float64}`, or any
`AbstractMatrix{<:Real}`. The matrix MUST be symmetric (use
`Symmetric(cov)` to mark it explicitly if needed).

A small or negative eigenvalue (≪ trace / n) is a red flag: the
covariance is ill-conditioned, often because two parameters are nearly
degenerate. Use `global_cc(cov)` to identify WHICH parameters are
degenerate.
"""
function eigenvalues(cov::AbstractMatrix{<:Real})
    n = size(cov, 1)
    n == size(cov, 2) ||
        throw(ArgumentError("eigenvalues: cov must be square (got $(size(cov)))"))
    sym = cov isa Symmetric ? cov : Symmetric(Matrix(cov))
    return sort(LinearAlgebra.eigvals(sym))
end

"""
    eigenvalues(m::Minuit) -> Union{Nothing,Vector{Float64}}
    eigenvalues(bfm::BoundedFunctionMinimum) -> Union{Nothing,Vector{Float64}}

Convenience overloads. Returns `nothing` if no covariance is available
(MIGRAD hasn't run, or it failed to produce one). Otherwise calls
[`eigenvalues`](@ref eigenvalues(::AbstractMatrix)) on the external
covariance matrix.
"""
function eigenvalues(m::Minuit)
    m.fmin === nothing && return nothing
    return eigenvalues(m.fmin)
end

function eigenvalues(bfm::BoundedFunctionMinimum)
    cov = free_covariance(bfm)
    cov === nothing && return nothing
    return eigenvalues(cov)
end

# ─────────────────────────────────────────────────────────────────────────────
# Global Correlation Coefficient
# ─────────────────────────────────────────────────────────────────────────────

"""
    global_cc(cov::AbstractMatrix{<:Real}) ->
        (cc::Vector{Float64}, valid::Bool)

Global Correlation Coefficient per parameter, mirrors
`MnGlobalCorrelationCoeff` in
`reference/Minuit2_cpp/src/MnGlobalCorrelationCoeff.cxx`.

Returns a pair `(cc, valid)`. `cc[i] = sqrt(1 - 1/(C_ii · C⁻¹_ii))`
when the denominator is well-conditioned; falls back to `0.0` if the
intermediate `C_ii · C⁻¹_ii < 1` (numerical near-degeneracy that would
otherwise give a NaN). `valid = false` if the covariance can't be
inverted (HESSE didn't converge — rare).

Interpretation:
- `cc[i] ≈ 1` → parameter `i` is strongly determined by the OTHER
  parameters (statistical degeneracy; consider fixing or re-parametrizing).
- `cc[i] ≈ 0` → parameter `i` is statistically independent of the rest.

Mirrors `MnGlobalCorrelationCoeff::MnGlobalCorrelationCoeff`.
"""
function global_cc(cov::AbstractMatrix{<:Real})
    n = size(cov, 1)
    n == size(cov, 2) ||
        throw(ArgumentError("global_cc: cov must be square (got $(size(cov)))"))
    Vmat = Matrix{Float64}(cov)
    inv_ok = true
    Vinv = try
        LinearAlgebra.inv(Symmetric(Vmat))
    catch err
        inv_ok = false
        zeros(n, n)
    end
    cc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        if !inv_ok
            cc[i] = 0.0
        else
            denom = Vinv[i, i] * Vmat[i, i]
            # C++ MnGlobalCorrelationCoeff.cxx:34-39 only checks
            # `0 < denom < 1` and clamps to 0; for `denom ≤ 0` (rare
            # MnInvert failure mode → non-PD inverse) the formula
            # gives NaN/Inf, and `denom = NaN` (V_ii ≤ 0 with Inf
            # diagonal in Vinv) breaks all comparisons. The single
            # `denom > 1.0` test below clamps EVERY pathological
            # branch (≤ 0, in (0,1), exactly 0, NaN) to a safe 0.0
            # via the `NaN > 1.0 == false` shortcircuit. Review I5.
            cc[i] = denom > 1.0 ? sqrt(1.0 - 1.0 / denom) : 0.0
        end
    end
    return cc, inv_ok
end

"""
    global_cc(m::Minuit) -> Union{Nothing,Tuple{Vector{Float64},Bool}}
    global_cc(bfm::BoundedFunctionMinimum) -> ...

Convenience overloads. Returns `nothing` if no covariance is available.
"""
function global_cc(m::Minuit)
    m.fmin === nothing && return nothing
    return global_cc(m.fmin)
end

function global_cc(bfm::BoundedFunctionMinimum)
    cov = free_covariance(bfm)
    cov === nothing && return nothing
    return global_cc(cov)
end
