# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# negative_g2.jl — NegativeG2LineSearch.
#
# Mirrors reference/Minuit2_cpp/src/NegativeG2LineSearch.cxx:26-121.
#
# When the numerical gradient's diagonal-second-derivative estimate
# `g2[i]` comes out non-positive (which makes the trivial diagonal
# inverse-Hessian guess `1/g2[i]` invalid), perform a one-dimensional
# line search along that parameter direction, recompute the gradient
# at the new point, and check again. Iterate up to 2n times.
#
# **Phase 0 must include this** (Opus parallel-review blocking #2):
# C++ `MnSeedGenerator.cxx:80` calls `HasNegativeG2(...)` *unconditionally*
# on the seed gradient before constructing the initial state. Skipping
# this in Phase 0 risks NFcn mismatch and iteration-trajectory drift
# on the Phase-0 benchmark corpus (Rosenbrock-10 in particular can
# trigger marginally-negative g2 from central-diff refinement).
#
# Phase 0 uses Numerical2P for the gradient recomputation; Phase 1+
# could plug in any GradientCalculator via multiple dispatch.
# ─────────────────────────────────────────────────────────────────────────────

"""
    has_negative_g2(grad::FunctionGradient, prec=MachinePrecision()) -> Bool

Returns `true` if any entry of `grad.g2` is non-positive (i.e. `≤ 0`).
Mirrors `NegativeG2LineSearch::HasNegativeG2` at
`reference/Minuit2_cpp/src/NegativeG2LineSearch.cxx:109-121`.

The precision argument is accepted for C++ API parity (the C++ takes
it but doesn't use it) — Julia signature follows the same convention
for forward compatibility.
"""
function has_negative_g2(grad::FunctionGradient,
                          ::MachinePrecision = MachinePrecision())
    @inbounds for i in eachindex(grad.g2)
        grad.g2[i] <= 0 && return true
    end
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# Main: negative-g2 line search loop
# ─────────────────────────────────────────────────────────────────────────────

"""
    negative_g2_line_search(state, cf, strategy, prec=MachinePrecision())
        -> MinimumState

If `state.gradient` has any non-positive `g2[i]`, do a 1-D line search
along the offending parameter, recompute the gradient at the new point,
and iterate up to `2n` times until either all `g2[i] > 0` or the iter
cap is reached. Return a new `MinimumState` with updated parameters /
gradient / error / EDM / nfcn.

Mirrors `NegativeG2LineSearch::operator()` at
`reference/Minuit2_cpp/src/NegativeG2LineSearch.cxx:26-107`.

# Behavior

- If `!has_negative_g2(state.gradient, prec)` on entry, returns
  `state` unchanged (no work done).
- For each iteration, scan parameters in order; on the first `g2[i] ≤ 0`:
  - **Skip** if both `|grad[i]| < eps` and `|g2[i]| < eps`
    (algorithm convention: parameter has zero gradient AND zero
    curvature, so neither direction is meaningfully downhill).
  - Otherwise build a step `s = ±gstep[i] · e_i` with sign opposite the
    gradient (downhill convention), and call `line_search`.
  - Apply `slam · s` to update the parameter vector, then recompute
    the full gradient via `numerical_gradient!`.
  - Break the inner loop and repeat the outer iteration.

# Output state

- Parameters: refined; `dirin` is dropped (matches C++ which uses the
  2-arg `MinimumParameters(vec, fval)` constructor at line 82).
- Error: built as a diagonal `1/g2[i]` (or `1` when `|g2[i]| ≤ eps2`).
  Status is `MnHesseValid` normally; `MnNotPosDef` if the resulting
  EDM is negative.
- Gradient: the last refreshed gradient.
- EDM: `0.5 · g' · diag(1/g2) · g`.
- NFcn: `ncalls(cf)` at exit.

# Allocation

This is a **rare-path** function (only fires when the initial gradient
estimate has bad curvature). It allocates internally per call —
acceptable since it doesn't run on the MIGRAD inner loop. Future
optimization could add a workspace-passing variant.
"""
function negative_g2_line_search(
    state::MinimumState,
    cf::CostFunction,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    if !has_negative_g2(state.gradient, prec)
        return state
    end

    n = length(state.parameters)
    # Working buffers
    x       = copy(state.parameters.x)
    fval    = state.parameters.fval
    work_g  = FunctionGradient(copy(state.gradient.grad),
                                copy(state.gradient.g2),
                                copy(state.gradient.gstep);
                                analytical = false)
    step    = zeros(n)
    x_work  = similar(x)
    ls_work = similar(x)
    eps  = prec.eps
    eps2 = prec.eps2

    iter = 0
    iterate_flag = true
    while iter < 2 * n && iterate_flag
        iter += 1
        iterate_flag = false
        @inbounds for i in 1:n
            if work_g.g2[i] <= 0
                # Skip if gradient AND g2 are both negligible — neither
                # direction is meaningfully downhill at this param.
                if abs(work_g.grad[i]) < eps && abs(work_g.g2[i]) < eps
                    continue
                end
                # Build step along i; sign downhill (away from positive grad).
                fill!(step, 0.0)
                step[i] = work_g.grad[i] < 0 ? work_g.gstep[i] : -work_g.gstep[i]
                gdel = step[i] * work_g.grad[i]
                # Line search at current point
                pa_tmp = MinimumParameters(x, fval)
                pp = line_search(cf, pa_tmp, step, gdel, prec; work_x = ls_work)
                # Apply scaled step
                step[i] *= pp.x
                x[i] += step[i]
                fval = pp.y
                # Recompute the full gradient at the new point. Reuse
                # `work_g` as both the previous-gradient input and the output
                # (numerical_gradient! uses copyto! for the prev-fields,
                # which is a safe no-op for `prev === out` aliasing — verified
                # at src/gradient.jl:186-188). Parallel-review #2 C4 — saves
                # one fresh FunctionGradient allocation per inner fix.
                new_par = MinimumParameters(x, fval)
                numerical_gradient!(work_g, x_work, new_par, work_g,
                                     cf, strategy, prec)
                iterate_flag = true
                break
            end
        end
    end

    # Build diagonal inverse-Hessian: 1/g2[i] when meaningful, else 1.
    M = zeros(n, n)
    @inbounds for i in 1:n
        M[i, i] = abs(work_g.g2[i]) > eps2 ? 1.0 / work_g.g2[i] : 1.0
    end
    err = MinimumError(Symmetric(M, :U), 1.0)
    edm_val = estimate_edm(work_g, err)

    if edm_val < 0
        # Re-mark with MnNotPosDef per C++ lines 102-104
        err = MinimumError(Symmetric(M, :U), MnNotPosDef)
    end

    new_par = MinimumParameters(x, fval)
    return MinimumState(new_par, err, work_g, edm_val, ncalls(cf))
end
