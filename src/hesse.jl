# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# hesse.jl — full numerical Hessian.
#
# Mirrors reference/Minuit2_cpp/src/MnHesse.cxx:93-330 (the
# operator()(MnFcn, MinimumState, MnUserTransformation, maxcalls)
# overload — the "real" MnHesse).
#
# Algorithm:
#   1. Evaluate FCN at the current point (amin).
#   2. Diagonal pass: for each parameter i, multi-cycle central-difference
#      refinement to determine g2[i] = ∂²f/∂x_i². Each cycle:
#        - Find a step d such that sag = (f(x+d) + f(x-d) - 2·f) ≠ 0
#          (multiplier loop up to 5× growth, bounded at 0.5 if param
#          has limits).
#        - Update d using `d = sqrt(2·aimsag / |g2|)`; clamp to
#          [dmin, 10·dlast] / [0.1·dlast, 0.5 if has-limits].
#        - Break if d-step or g2 has converged below the strategy
#          tolerances.
#      vhmat[i, i] = g2[i].
#   3. (Strategy > 0): refine gradient via `hessian_gradient!`
#      (port of C++ `HessianGradientCalculator`). Per-coordinate
#      central-difference iteration on the FCN, up to
#      `strategy.hessian_grad_ncycles` cycles. Updates `grd` and
#      `gst`; leaves `g2` (from step 2) and `dirin` alone.
#   4. Off-diagonal pass: for each pair (i, j) with i < j, compute
#      `(f(x + d_i + d_j) + f(x) - f(x + d_i) - f(x + d_j)) / (d_i d_j)`.
#      Uses cached single-direction values `yy[i]` from the diagonal
#      pass.
#   5. MnPosDef enforcement.
#   6. Sym invert. If fails → MnInvertFailed diagonal matrix.
#   7. New EDM via the standard estimator.
#   8. Return new MinimumState with the updated MinimumError.
#
# This is the standalone MnHesse. Calling it from inside MIGRAD when
# Strategy ≥ 1 + Dcovar > 0.05 (the inner-HESSE refinement) is the
# Phase 1 integration step — see migrad.jl follow-up.
# ─────────────────────────────────────────────────────────────────────────────

"""
    hesse(cf, state, strategy=Strategy(1); prec=MachinePrecision(), maxcalls=0)
        -> MinimumState

Compute the full numerical Hessian at `state.parameters.x` and return
a new `MinimumState` with the refined error matrix, recomputed EDM,
and updated FCN call count.

Mirrors C++ `MnHesse::operator()(MnFcn, MinimumState,
MnUserTransformation, maxcalls)` —
`reference/Minuit2_cpp/src/MnHesse.cxx:93-330`.

# Arguments

- `cf::CostFunction` — the user FCN (operates on the parameter
  vector that `state.parameters.x` reports — Phase 1 first cut: no
  bounds, so internal == external; bounded HESSE is a follow-up).
- `state::MinimumState` — current state. The gradient field provides
  initial step sizes (`gst[i] = state.gradient.gstep[i]`) and the
  algorithm refines `g2[i]`.
- `strategy::Strategy` — controls `hessian_ncycles` (cycles per
  parameter), `hessian_step_tolerance`, `hessian_g2_tolerance`,
  and (Strategy ≥ 1) gradient refinement.
- `prec::MachinePrecision` — floor for step sizes and pos-def gate.
- `maxcalls::Integer` — FCN call cap; `0` means use the default
  `200 + 100n + 5n²` per `MnApplication.cxx:43`.

# Return statuses

The returned `MinimumState`'s `error.status` may be:
- `MnHesseValid` — full success.
- `MnMadePosDef` — pos-def perturbation was applied.
- `MnHesseFailed` — sag stayed zero or maxcalls hit; matrix is
  diagonal `1/g2[i]` (or 1 where g2 is too small).
- `MnInvertFailed` — inversion failed; matrix is the same diagonal.
"""
function hesse(
    cf::AbstractCostFunction,
    state::MinimumState,
    strategy::Strategy = Strategy(1);
    prec::MachinePrecision = MachinePrecision(),
    maxcalls::Integer = 0,
    print_level::Integer = 0,
)
    # HESSE computes 2nd derivatives by central differences on `cf(x)`
    # — the user's analytical gradient (when `cf isa
    # CostFunctionWithGradient`) doesn't speed this up and the algorithm
    # ignores it. Accept any `AbstractCostFunction` so that
    # `Minuit(...; grad=...)` users can still call `hesse(m)`.
    n = length(state)
    if maxcalls == 0
        maxcalls = 200 + 100 * n + 5 * n * n
    end

    x = copy(state.parameters.x)
    amin = cf(x)
    aimsag = sqrt(prec.eps2) * (abs(amin) + cf.up)

    # gap M1: header for level 1. Mirrors C++ `print.Info(...)` calls
    # at MnHesse.cxx top of operator().
    if print_level >= 1
        _trace_info(print_level, "MnHesse",
                    @sprintf("start: n=%d  fval=%.10g  ncycles=%d  maxcalls=%d",
                             n, amin, strategy.hessian_ncycles, maxcalls))
    end

    # Scratch — independent vectors so we don't mutate state.gradient
    g2 = copy(state.gradient.g2)
    gst = copy(state.gradient.gstep)
    grd = copy(state.gradient.grad)
    dirin = copy(gst)
    yy = zeros(Float64, n)

    # ── AD-gradient numerical-companion refresh ───────────────────
    # Mirrors C++ `MnHesse.cxx:118-126`. When the input gradient is
    # analytical (the user supplied `grad=...` and we're running
    # through `CostFunctionWithGradient`), `state.gradient.g2` and
    # `state.gradient.gstep` carry their last seed-time numerical
    # estimates — never refreshed during MIGRAD because the analytical
    # gradient short-circuits the central-difference path that would
    # have updated them. C++ recomputes the (gst, dirin, g2) triplet
    # via `Numerical2PGradientCalculator` before the diagonal Hessian
    # pass begins; we do the same here.
    #
    # We don't refresh `grd` — the analytical value is kept (matches
    # C++ where the refresh only writes back `gst`, `dirin`, `g2`).
    # The diagonal pass below overwrites `grd[i]` via central diffs
    # anyway, so the input `grd` only matters if the diagonal pass
    # fails early — in that case the AD-derived value is the better
    # answer (cf. `_hesse_diagonal_failure`).
    #
    # NOTE — seed-semantics divergence from C++ Minuit2:
    # C++ `MnHesse.cxx:121` calls the cold-start
    # `Numerical2PGradientCalculator::operator()(par)` overload, which
    # internally constructs a fresh `InitialGradientCalculator` that
    # reads per-parameter user errors from `MnUserTransformation`
    # (`InitialGradientCalculator.cxx:25-75`). JuMinuit's `MinimumState`
    # does not retain those user errors past MIGRAD (`migrad.jl:672`
    # builds `MinimumParameters(nx_buf, pp.y)` with the no-step
    # constructor that zeros `dirin`), so we instead seed the
    # `numerical_gradient!` central-difference iteration with the
    # current (stale) `state.gradient`. For smooth FCNs the iterative
    # `step ← max(optstp, 0.1·gstep)` refinement converges in 1-2
    # cycles regardless of seed, so the observable g2/gstep output
    # matches C++ to within FP precision. For pathological FCNs where
    # convergence depends on the initial step, the two paths can
    # diverge — see [docs/GAP_AUDIT.md] P2 follow-up note for the fix
    # (propagate seed-time errors through `MinimumState`).
    #
    # NOTE — gate fidelity vs C++:
    # C++ `MnHesse.cxx:120` gates on `st.Gradient().IsAnalytical()` —
    # the `analytical` flag on the input `FunctionGradient`. JuMinuit's
    # gate is `cf isa CostFunctionWithGradient` because today neither
    # `analytical_gradient!` (`src/ad_gradient.jl:184-208`) nor
    # `migrad.jl:677` sets the `FunctionGradient.analytical` flag to
    # `true` for AD-MIGRAD states (the flag was added in `state.jl:123`
    # but never threaded through the AD path). Effect: a *repeated*
    # call to `hesse(cf::CostFunctionWithGradient, ...)` triggers the
    # refresh even when the prior HESSE has already left
    # `state.gradient.g2` numerical-grade. The refresh is observably
    # idempotent (it just re-converges on the same g2/gstep), so the
    # cost is the extra `numerical_gradient!` FCN calls only, not a
    # correctness bug. Documented follow-up: thread the `analytical`
    # flag through MIGRAD + AD path and gate this branch on
    # `is_analytical(state.gradient)` to match C++ exactly.
    if cf isa CostFunctionWithGradient
        # CostFunction wrapper SHARES `cf.f`, `cf.up`, and `cf.nfcn` —
        # FCN calls against `cf_numeric` increment the same counter so
        # the `maxcalls` budget below remains correct.
        cf_numeric = CostFunction(cf.f, cf.up, cf.nfcn)
        refresh_out = FunctionGradient(zeros(Float64, n),
                                        zeros(Float64, n),
                                        zeros(Float64, n))
        x_refresh = similar(x)
        numerical_gradient!(refresh_out, x_refresh, state.parameters,
                             state.gradient, cf_numeric, strategy, prec)
        copyto!(g2, refresh_out.g2)
        copyto!(gst, refresh_out.gstep)
        copyto!(dirin, refresh_out.gstep)
    end

    vhmat = zeros(Float64, n, n)

    # No `has_limits` info per parameter in Phase 1 first-cut (bounds
    # integration is the follow-up). Treat all params as unbounded for
    # the d-clamp branches; this matches the migrad.jl Phase 0 caller.
    has_limits = false

    # ── Diagonal pass ─────────────────────────────────────────────
    for i in 1:n
        xtf = x[i]
        dmin = 8.0 * prec.eps2 * (abs(xtf) + prec.eps2)
        d = abs(gst[i])
        d < dmin && (d = dmin)

        fs1 = 0.0
        fs2 = 0.0
        sag = 0.0
        converged = false

        for icyc in 1:strategy.hessian_ncycles
            # Multiplier loop — grow d until sag ≠ 0
            sag = 0.0
            mlp_failed = false
            for multpy in 1:5
                x[i] = xtf + d
                fs1 = cf(x)
                x[i] = xtf - d
                fs2 = cf(x)
                x[i] = xtf
                sag = 0.5 * (fs1 + fs2 - 2.0 * amin)
                sag != 0 && break
                if has_limits
                    d > 0.5 && (mlp_failed = true; break)
                    d *= 10
                    d > 0.5 && (d = 0.51)
                else
                    d *= 10.0
                end
                multpy == 5 && (mlp_failed = true)
            end

            if sag == 0 || mlp_failed
                # Sag stayed zero → return failure with diagonal matrix
                return _hesse_diagonal_failure(state, g2, prec, ncalls(cf), MnHesseFailed)
            end

            g2bfor = g2[i]
            g2[i] = 2.0 * sag / (d * d)
            grd[i] = (fs1 - fs2) / (2.0 * d)
            gst[i] = d
            dirin[i] = d
            yy[i] = fs1
            dlast = d

            d = sqrt(2.0 * aimsag / abs(g2[i]))
            has_limits && (d = min(0.5, d))
            d < dmin && (d = dmin)

            # Convergence checks
            if abs((d - dlast) / d) < strategy.hessian_step_tolerance
                converged = true
                break
            end
            # The `g2[i] != 0` guard is a defensive Julia addition. C++
            # `MnHesse.cxx:203-206` has no such guard: when `g2(i) == 0`
            # it evaluates `(g2-g2bfor)/0 = ±Inf`, then `fabs(±Inf) < tol`
            # is `false` and the break doesn't fire — same observable
            # behavior. Julia's guard prevents a `0/0 = NaN` that would
            # also fail the `< tol` test (parallel-review #2 C3).
            if g2[i] != 0 && abs((g2[i] - g2bfor) / g2[i]) < strategy.hessian_g2_tolerance
                converged = true
                break
            end
            d = min(d, 10.0 * dlast)
            d = max(d, 0.1 * dlast)
        end

        vhmat[i, i] = g2[i]

        # gap M1: outer guard — without it the @sprintf would fire per
        # parameter at level 0 (O(n) wasted Strings per HESSE).
        if print_level >= 2
            _trace_info(print_level, "MnHesse",
                        @sprintf("diag i=%d  g2=%.6g  d=%.6g  converged=%s",
                                  i, g2[i], dirin[i], converged ? "yes" : "no"))
        end

        if ncalls(cf) > maxcalls
            if print_level >= 1
                _trace_warn(print_level, "MnHesse",
                            @sprintf("maxcalls exceeded during diagonal pass at i=%d", i))
            end
            return _hesse_diagonal_failure(state, g2, prec, ncalls(cf), MnHesseFailed)
        end
    end

    # ── Strategy ≥ 1: gradient refinement via HessianGradientCalculator ──
    # Mirrors C++ `MnHesse.cxx:228-235`. Between the diagonal and
    # off-diagonal passes, refine `grd` and `gst` per parameter via
    # per-coordinate central-difference iteration (up to
    # `strategy.hessian_grad_ncycles` cycles). `g2` and `dirin` are
    # **not** touched — `g2` came from the diagonal pass just above,
    # and `dirin` carries the converged d-value that the off-diagonal
    # pass below will use as the step size.
    #
    # The C++ gate is on `fStrategy.Strategy() > 0` (line 228 there).
    # Strategy(0) skips HGC; Strategy(1)/(2) run it with
    # `hessian_grad_ncycles = 2 / 6` respectively.
    if strategy.level > 0
        # `dgrd_scratch` is the per-parameter gradient uncertainty
        # that C++ `HessianGradientCalculator::DeltaGradient` returns
        # alongside the refined gradient (a `std::pair<FunctionGradient,
        # MnAlgebraicVector>`). MnHesse never consumes it — only
        # callers like `Numerical2PGradientCalculator::operator()` use
        # the analog. We discard. Pre-zeroed (not `undef`) so a future
        # refactor that conditionally skips the inner writeback can't
        # leak uninitialized values.
        dgrd_scratch = zeros(Float64, n)
        x_hgc = similar(x)
        hessian_gradient!(grd, gst, dgrd_scratch, x_hgc,
                           state.parameters, cf, g2, strategy, prec)
    end

    # ── Off-diagonal pass ─────────────────────────────────────────
    # All pairs (i, j) with i < j.
    if n > 1
        # gap M1: O(n²) inner loop — outer guard is essential. Without
        # it the @sprintf would alloc one String per (i,j) pair at
        # every level (including 0) — 190 wasted allocs for n=20.
        if print_level >= 2
            _trace_info(print_level, "MnHesse", "starting off-diagonal pass")
        end
        for i in 1:n
            x[i] += dirin[i]
            for j in (i + 1):n
                x[j] += dirin[j]
                fs1 = cf(x)
                vhmat[i, j] = (fs1 + amin - yy[i] - yy[j]) / (dirin[i] * dirin[j])
                if print_level >= 2
                    _trace_info(print_level, "MnHesse",
                                @sprintf("off-diag i=%d j=%d  H=%.6g",
                                          i, j, vhmat[i, j]))
                end
                x[j] -= dirin[j]
            end
            x[i] -= dirin[i]
        end
    end

    # ── Pos-def enforcement on the H matrix ───────────────────────
    # NOTE: We pass the **Hessian** (vhmat: second derivatives) into
    # `make_posdef` even though MnPosDef's documented purpose is to
    # ensure positive-definiteness of the **inverse Hessian** (V) — this
    # mirrors C++ Minuit2 exactly (reference/Minuit2_cpp/src/MnHesse.cxx:278:
    # `MinimumError tmpErr = MnPosDef()(MinimumError(vhmat, 1.), prec);`).
    # The algorithm is matrix-agnostic (normalizes by diagonal, eigen-
    # adjusts), so it works on either H or V mathematically; the C++
    # design choice perturbs H's diagonal which bounds V's eigenvalues
    # by `1/(λ_H + δ)` rather than perturbing V by `δ` directly — the
    # conservative choice for covariance reporting. Parallel-review #2 C6.
    err_tmp = make_posdef(MinimumError(Symmetric(vhmat, :U), 1.0), prec)
    vhmat_pd = copy(parent(err_tmp.inv_hessian))

    # ── Symmetric invert ──────────────────────────────────────────
    inv_ok = true
    try
        sym_invert!(Symmetric(vhmat_pd, :U))
    catch _
        inv_ok = false
    end

    if !inv_ok
        return _hesse_diagonal_failure(state, g2, prec, ncalls(cf), MnInvertFailed)
    end

    # ── New EDM with the refined error ────────────────────────────
    refined_grad = FunctionGradient(grd, g2, gst)
    new_err_status = is_made_pos_def(err_tmp) ? MnMadePosDef : MnHesseValid
    new_dcov = is_made_pos_def(err_tmp) ? 1.0 : 0.0
    new_err = MinimumError(Symmetric(vhmat_pd, :U), new_dcov, new_err_status, true)
    # In-place EDM — reuse `yy` (already length n, contents no longer needed
    # after the off-diagonal pass completed above).
    new_edm = estimate_edm!(yy, refined_grad, new_err)

    if print_level >= 1
        status_str = new_err_status == MnHesseValid ? "valid" :
                     new_err_status == MnMadePosDef ? "made-pos-def" : "invalid"
        _trace_info(print_level, "MnHesse",
                    @sprintf("done: status=%s  edm=%.6g  dcovar=%.4g  ncalls=%d",
                             status_str, new_edm, new_dcov, ncalls(cf)))
    end

    return MinimumState(state.parameters, new_err, refined_grad,
                        new_edm, ncalls(cf))
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _hesse_diagonal_failure(state, g2, prec, nfcn, status) -> MinimumState

Build the failure-mode `MinimumState` with a diagonal inverse-Hessian
of `1/g2[i]` (clamped to 1 when g2 is too small). Mirrors C++
`MnHesse.cxx:177-184` (and the analogous block at lines 216-223 for
maxcalls overrun).
"""
function _hesse_diagonal_failure(state::MinimumState, g2::Vector{Float64},
                                  prec::MachinePrecision, nfcn::Integer,
                                  status::CovStatus)
    n = length(g2)
    M = zeros(Float64, n, n)
    @inbounds for j in 1:n
        tmp = g2[j] < prec.eps2 ? 1.0 : 1.0 / g2[j]
        M[j, j] = tmp < prec.eps2 ? 1.0 : tmp
    end
    err = MinimumError(Symmetric(M, :U), status)
    return MinimumState(state.parameters, err, state.gradient,
                        state.edm, nfcn)
end
