# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# seed.jl — MnSeedGenerator equivalent (Phase 0 numerical-gradient path).
#
# Mirrors reference/Minuit2_cpp/src/MnSeedGenerator.cxx:41-101 (the
# numerical-gradient overload). Analytical-gradient overload at C++
# lines 103-174 is Phase 1+ (when FCNGradAdapter lands).
#
# Phase 0:
# - No user-supplied covariance prior (dcovar = 1.0 always).
# - NegativeG2LineSearch invoked UNCONDITIONALLY whenever
#   has_negative_g2(grad) holds (Opus parallel-review blocking #2).
# - Strategy(0) only — the `if Strategy()==2 && !HasCovariance` branch
#   that runs MnHesse at C++ line 88-98 is Phase 1.
# ─────────────────────────────────────────────────────────────────────────────

"""
    seed_state(cf, x0, errs, strategy=Strategy(0), prec=MachinePrecision())
        -> MinimumState

Build the initial `MinimumState` for a MIGRAD fit. Mirrors
`MnSeedGenerator::operator()` (numerical-gradient overload) from
`reference/Minuit2_cpp/src/MnSeedGenerator.cxx:41-101`.

# Steps (matching C++)

1. Wrap `x0` in `MinimumParameters` along with the initial step sizes
   `errs` and the first FCN evaluation `fval = cf(x0)`.
2. Compute the initial gradient via `numerical_gradient(par, cf, ...)`
   — cold-start variant that first computes the rough estimate via
   step sizes, then refines via two-point central diff.
3. Build the initial diagonal inverse-Hessian: `diag(1/g2[i])` when
   `|g2[i]| > eps2`, else `1.0` (matches C++ lines 69-70).
4. Estimate EDM via `0.5·g'·V·g`.
5. Construct `MinimumState`.
6. **Unconditional** `has_negative_g2` check; if any `g2[i] ≤ 0`,
   refine via `negative_g2_line_search` (C++ lines 79-86).
7. Phase 0: skip the MnHesse branch (Strategy(0) only).

# Arguments

- `cf::CostFunction` — the user FCN (call counter starts fresh; the
  seed accounts for ~1 + (2·n·grad_ncycles) calls).
- `x0::AbstractVector{<:Real}` — initial parameter values.
- `errs::AbstractVector{<:Real}` — initial step sizes (per-parameter
  "error" estimates). Should be non-negative; the gradient algorithm
  uses `|werr|`.
- `strategy::Strategy` — Phase 0 must be `Strategy(0)`.
- `prec::MachinePrecision`.
"""
function seed_state(
    cf::CostFunction,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real},
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(x0)
    length(errs) == n ||
        throw(DimensionMismatch("errs length $(length(errs)) != x0 length $n"))
    # Strategy ≥ 1 is now supported via the inner-Hesse refinement in the
    # MIGRAD outer loop (`_migrad_loop` post-DFP block). The seed-stage
    # MnHesse bootstrap (C++ MnSeedGenerator.cxx — Strategy==2 && !HasCov
    # branch) is intentionally deferred; the inner-Hesse path at the
    # MIGRAD level catches up after the first DFP pass, so the final
    # state matches C++ within the documented tolerance.

    # MinimumParameters with explicit step sizes (so the cold-start
    # numerical_gradient can use them via has_step_size).
    x = collect(Float64, x0)
    dirin = collect(Float64, errs)
    fval = cf(x)
    par = MinimumParameters(x, dirin, fval)

    # Cold-start gradient computation (initial rough + Numerical2P refine).
    grad = numerical_gradient(par, cf, strategy, prec)

    # Diagonal inverse-Hessian (C++ MnSeedGenerator.cxx:69-70).
    mat = zeros(n, n)
    @inbounds for i in 1:n
        mat[i, i] = abs(grad.g2[i]) > prec.eps2 ? 1.0 / grad.g2[i] : 1.0
    end
    err = MinimumError(Symmetric(mat, :U), 1.0)

    edm_val = estimate_edm(grad, err)
    state = MinimumState(par, err, grad, edm_val, ncalls(cf))

    # Unconditional NegativeG2 check (Opus blocking #2).
    if has_negative_g2(grad, prec)
        state = negative_g2_line_search(state, cf, strategy, prec)
    end

    # Phase 0: no MnHesse seed-refinement (DR-008).
    return state
end
