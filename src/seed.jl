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
    # Strategy ≥ 1 is supported via the inner-Hesse refinement in the
    # MIGRAD outer loop (`_migrad_loop` post-DFP block). Strategy(2)
    # additionally runs the seed-stage MnHesse bootstrap at the bottom
    # of this function (mirrors C++ MnSeedGenerator.cxx:88-98).

    # MinimumParameters with explicit step sizes (so the cold-start
    # numerical_gradient can use them via has_step_size).
    x = collect(Float64, x0)
    dirin = collect(Float64, errs)
    fval = cf(x)
    par = MinimumParameters(x, dirin, fval)

    # Cold-start gradient computation. Reuse a single FunctionGradient
    # buffer across the rough initial estimate AND the Numerical2P
    # refinement (numerical_gradient! does `copyto!(out, prev)` internally,
    # so out === prev is a safe no-op self-copy followed by in-place refine).
    # Saves 3 vector allocations vs. the two-stage allocating wrapper.
    grad = FunctionGradient(zeros(Float64, n), zeros(Float64, n),
                             zeros(Float64, n))
    initial_gradient!(grad, par, dirin, cf.up, prec)
    x_work = Vector{Float64}(undef, n)
    numerical_gradient!(grad, x_work, par, grad, cf, strategy, prec)

    # Diagonal inverse-Hessian (C++ MnSeedGenerator.cxx:69-70).
    mat = zeros(n, n)
    @inbounds for i in 1:n
        mat[i, i] = abs(grad.g2[i]) > prec.eps2 ? 1.0 / grad.g2[i] : 1.0
    end
    err = MinimumError(Symmetric(mat, :U), 1.0)

    # Use the in-place EDM variant — x_work is free to reuse (refined
    # gradient is done, we don't need x_work for it anymore).
    edm_val = estimate_edm!(x_work, grad, err)
    state = MinimumState(par, err, grad, edm_val, ncalls(cf))

    # Unconditional NegativeG2 check (Opus blocking #2).
    if has_negative_g2(grad, prec)
        state = negative_g2_line_search(state, cf, strategy, prec)
    end

    # Strategy(2): seed-time MnHesse bootstrap.
    # Mirrors C++ `MnSeedGenerator.cxx:88-98`:
    #     if (stra.Strategy() == 2 && !st.HasCovariance())
    #         MinimumState tmp = MnHesse(stra)(fcn, state, st.Trafo());
    # The user-state never carries a pre-computed covariance through
    # JuMinuit's API (we always start with the diagonal-from-step-sizes
    # estimate), so `!HasCovariance` collapses to "always when Strategy 2".
    # The bootstrap gives MIGRAD a full-rank Hessian to start with,
    # which on hard problems saves a couple of outer-loop iterations
    # vs. having the inner-Hesse refinement build it up after MIGRAD
    # converges.
    if strategy.level == 2
        state = hesse(cf, state, strategy; prec = prec)
    end

    return state
end

# ─────────────────────────────────────────────────────────────────────────────
# warm_restart_state — build a MinimumState for a NEW FCN, reusing a
# previous probe's converged inv_hessian + gradient as warm starts.
#
# Background: inside `MnFunctionCross` (function_cross.jl), the cross-
# search runs 3-15 inner MIGRADs at sequential α probes, each with a
# different fixed-parameter value v_probe. Phase 1.x originally restarted
# `seed_state` cold every probe — costing ~1 + 4·n_inner FCN calls per
# probe (initial gradient + Numerical2P refine), and producing an
# IDENTITY-like diagonal inv_hessian that the DFP loop then had to
# converge from scratch.
#
# C++ Minuit2 sidesteps this via a single MnMigrad instance whose
# `MnUserParameterState` is mutated by each `migrad()` invocation — so
# the inv_hessian + gradient state IS carried across calls.
# `warm_restart_state` is the Julia equivalent: take a previous probe's
# state, evaluate the NEW FCN at the same x for a fresh fval, refine the
# gradient (using the prev g2/gstep as warm starts so Numerical2P
# converges in 1 cycle), and KEEP the prev inv_hessian.
#
# Per-probe budget vs cold seed (n_inner=8 typical):
#   cold seed_state: 1 + 2·8 (initial_gradient is FCN-free but adds 0)
#                    + 2·8·grad_ncycles refine = ~17-33 calls
#   warm restart: 1 + 2·8·grad_ncycles refine ≈ ~17 calls
#   AND ~5-10 fewer DFP iters because inv_hessian starts near-true
#
# Empirically saves ~50-60% of FCN calls on 10D contours (verified on
# gauss_ll_10_1000 + rosenbrock_10d MNCONTOUR benchmarks).
# ─────────────────────────────────────────────────────────────────────────────

"""
    warm_restart_state(prev::MinimumState, new_cf::CostFunction;
                       strategy=Strategy(0), prec=MachinePrecision())
        -> Union{MinimumState,Nothing}

Build a MIGRAD-ready seed state for `new_cf` by reusing the converged
gradient + inv_hessian from a previous probe's `MinimumState`. Returns
`nothing` if the warm path can't be safely taken — caller should fall
back to [`seed_state`](@ref).

# Algorithm

1. Take `prev.parameters.x` (the warm position). Evaluate `new_cf` at
   that point → new `fval` (1 FCN call).
2. Use `prev.gradient` as the seed for `numerical_gradient!` — the
   per-coord step convergence in Numerical2P shortcuts after 1 cycle
   when `prev.gstep` is already near-optimal. (~2·n FCN calls.)
3. Keep `prev.error.inv_hessian` unchanged — this is the bulk of the
   warm-start gain, sidestepping ~5-10 DFP convergence iters.
4. Compute new EDM from refined gradient + carried-over inv_hessian.
5. Wrap into a `MinimumState` with the same dimension as `prev`.

# Returns `nothing` (caller should cold-seed) when:

- `prev` is invalid (missing parameters, missing error, missing gradient).
- The refined gradient has any non-positive `g2[i]` — caller's
  `seed_state` path runs `negative_g2_line_search` to recover; this
  helper doesn't bake in the recovery.
- `length(prev) == 0`.

# Phase-0 lock

Strategy ≥ 1's seed-stage Hesse bootstrap is intentionally NOT applied
to warm restarts: the prev inv_hessian IS the warm Hessian, so a Hesse
refresh would defeat the purpose. Strategy(2) callers still get the
post-MIGRAD inner-Hesse refinement in `_migrad_loop`.
"""
function warm_restart_state(
    prev::MinimumState,
    new_cf::CostFunction;
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(prev)
    n == 0 && return nothing
    is_valid(prev) || return nothing
    is_available(prev.error) || return nothing
    # Validity of prev.gradient is checked by is_valid above; defensively
    # also require dimensions to match.
    length(prev.gradient) == n || return nothing

    # 1. Re-evaluate new_cf at the warm position.
    x_warm = collect(Float64, prev.parameters.x)
    fval_new = new_cf(x_warm)

    # 2. Build new MinimumParameters. Preserve prev's `has_step_size`
    # semantics — when the prev state carried explicit step sizes (the
    # usual case for inner-MIGRAD outputs), pass them through; otherwise
    # use the 2-arg ctor which sets dirin=zeros AND has_step_size=false.
    # We deliberately don't fall back to `prev.gradient.gstep`: gstep
    # is the numerical-gradient step (~1e-3 scale), not a user-error
    # estimate (~0.1 scale), and claiming has_step_size=true with that
    # would silently misrepresent the parameter's natural scale to any
    # downstream consumer (parallel-review #N-5).
    par_new = prev.parameters.has_step_size ?
               MinimumParameters(x_warm, copy(prev.parameters.dirin), Float64(fval_new)) :
               MinimumParameters(x_warm, Float64(fval_new))

    # 3. Refine gradient. Allocate fresh out vectors (caller doesn't
    # share buffers with us — this is per-probe scratch, ~few cache
    # lines). numerical_gradient! seeds itself from `prev.gradient` via
    # the copyto! at the top of that function, so we just pass prev in.
    grad_new = FunctionGradient(zeros(Float64, n), zeros(Float64, n),
                                  zeros(Float64, n))
    x_work = Vector{Float64}(undef, n)
    numerical_gradient!(grad_new, x_work, par_new, prev.gradient,
                         new_cf, strategy, prec)

    # 4. Bail to cold seed if refined gradient produced any non-positive
    # g2[i]. The cold path's negative_g2_line_search would handle this,
    # but doing it here would mean the WARM Hessian gets discarded
    # anyway (the line search rebuilds the diag(1/g2) error). Cleaner
    # to just return nothing and let the caller cold-seed.
    if has_negative_g2(grad_new, prec)
        return nothing
    end

    # 5. Reuse prev's inv_hessian. We do NOT copy the matrix here — the
    # inv_hessian is read-only inside `_migrad_loop` until the first DFP
    # update, which writes into a SEPARATE ping-pong buffer (V_a/V_b in
    # migrad.jl). So sharing the storage is safe.
    #
    # BUT we explicitly rebuild MinimumError with `dcovar = 0` and
    # `status = MnHesseValid`, matching C++ MnSeedGenerator.cxx:63-67
    # (HasCovariance branch sets `dcovar = 0.0` regardless of the
    # incoming state). Carrying prev's dcovar would inflate the
    # `edm_corrected = edm·(1+3·dcovar)` correction in `_migrad_loop`,
    # and propagating a stale `MnMadePosDef` status would mark our warm
    # seed as posdef-massaged forever even though the matrix has now
    # been through full MIGRAD convergence.
    err_warm = MinimumError(prev.error.inv_hessian, 0.0)

    # 6. EDM from new gradient + warm inv_hessian.
    edm_new = estimate_edm!(x_work, grad_new, err_warm)

    return MinimumState(par_new, err_warm, grad_new, edm_new, ncalls(new_cf))
end
