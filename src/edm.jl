# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# edm.jl — Expected Distance to Minimum estimator.
#
# Mirrors VariableMetricEDMEstimator from
# reference/Minuit2_cpp/inc/Minuit2/VariableMetricEDMEstimator.h
# and reference/Minuit2_cpp/src/VariableMetricEDMEstimator.cxx.
#
# EDM is the standard MIGRAD convergence indicator: 0.5·g'·V·g where g
# is the gradient and V is the inverse Hessian. It estimates the
# function-value gap to the minimum under the local quadratic
# approximation.
# ─────────────────────────────────────────────────────────────────────────────

"""
    estimate_edm(grad, error) -> Float64

Compute the Expected Distance to Minimum: `0.5 · g' · V · g`.

Allocating — for the hot path use [`estimate_edm!`](@ref) with a
preallocated workspace.

Mirrors `VariableMetricEDMEstimator::Estimate`.
"""
function estimate_edm(grad::FunctionGradient, error::MinimumError)
    # dot(x, A, y) for Symmetric{Float64,Matrix{Float64}} dispatches to
    # BLAS via LinearAlgebra; may use an internal temporary. Use
    # estimate_edm! to avoid this.
    g = grad.grad
    V = error.inv_hessian
    return 0.5 * dot(g, V, g)
end

"""
    estimate_edm!(work, grad, error) -> Float64

Zero-allocation EDM computation using a preallocated workspace `work`
of length `n`. The workspace is overwritten with `V·g`.

The standard hot-path call site inside MIGRAD: after each iteration's
gradient/Hessian update, recompute EDM for the convergence check.
"""
function estimate_edm!(
    work::AbstractVector{Float64},
    grad::FunctionGradient,
    error::MinimumError,
)
    g = grad.grad
    V = error.inv_hessian
    length(work) == length(g) ||
        throw(DimensionMismatch("work length $(length(work)) != grad length $(length(g))"))
    sym_mul!(work, V, g)
    return 0.5 * dot(g, work)
end
