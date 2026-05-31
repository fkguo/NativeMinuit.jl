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

- `cf::CostFunction` — the user FCN. Operates on the coordinate
  frame `state.parameters.x` reports: for a bounded fit this is the
  INTERNAL (sin/sqrt-transformed) frame supplied by the
  `migrad(cf, params)` / `hesse(m::Minuit)` wrappers, which is exactly
  the frame the C++ step clamp (`has_limits`) targets.
- `state::MinimumState` — current state. The gradient field provides
  initial step sizes (`gst[i] = state.gradient.gstep[i]`) and the
  algorithm refines `g2[i]`.
- `strategy::Strategy` — controls `hessian_ncycles` (cycles per
  parameter), `hessian_step_tolerance`, `hessian_g2_tolerance`,
  and (Strategy ≥ 1) gradient refinement.
- `prec::MachinePrecision` — floor for step sizes and pos-def gate.
- `maxcalls::Integer` — FCN call cap; `0` means use the default
  `200 + 100n + 5n²` per `MnApplication.cxx:43`.
- `has_limits::Union{Nothing,AbstractVector{Bool}}` — per-parameter
  (internal index) bound flags. `nothing` (default) ⇒ all unbounded
  ⇒ no step clamp ⇒ byte-identical to an unbounded HESSE. When a
  parameter is flagged, its diagonal probe step `d` is clamped at 0.5
  in internal coordinates (C++ `MnHesse.cxx:160-167, 194-195`).

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
    has_limits::Union{Nothing,AbstractVector{Bool}} = nothing,
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
    # diverge — see [docs/dev/GAP_AUDIT.md] P2 follow-up note for the fix
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

    # Per-parameter bound flags for the C++ MnHesse step clamp
    # (MnHesse.cxx:160-167, 194-195): when a parameter HasLimits(), the
    # diagonal probe step `d` is bounded at 0.5 in INTERNAL (sin/sqrt-
    # transformed) coordinates. Near a bound the transform is steep, so
    # an unclamped `d` maps to a wild external excursion and a wrong 2nd
    # derivative. `has_limits === nothing` (the unbounded default — and
    # every unbounded caller, including the standalone `hesse(f,x0,err)`)
    # makes the per-parameter `lim_i` below always `false`, so the
    # diagonal pass is byte-identical to the no-clamp path. The bounded
    # callers (`migrad(cf, params)`, `hesse(m::Minuit)`) pass a length-n
    # vector indexed by INTERNAL/free-parameter position
    # (`_has_limits_internal`), the JuMinuit analogue of C++
    # `trafo.Parameter(i).HasLimits()`.
    has_limits === nothing || length(has_limits) == n ||
        throw(DimensionMismatch(
            "hesse: has_limits length $(length(has_limits)) != n=$n"))

    # ── Diagonal pass ─────────────────────────────────────────────
    for i in 1:n
        # C++ `trafo.Parameter(i).HasLimits()` (MnHesse.cxx:160, 194).
        lim_i = has_limits === nothing ? false : has_limits[i]
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
                if lim_i
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
            lim_i && (d = min(0.5, d))
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
    # In-place pos-def enforcement: `vhmat` is a transient (its values are
    # not read after this point except through the matrix `new_err` keeps),
    # so `make_posdef!` perturbs it in place and reports whether the
    # MnMadePosDef tag would apply — bit-identical to the allocating
    # `make_posdef(MinimumError(Symmetric(vhmat,:U),1.))` it replaces, but
    # without that call's input copy, p/s scratch, and result wrapper.
    made_pd = make_posdef!(Symmetric(vhmat, :U), prec)

    # ── Symmetric invert (in place) ───────────────────────────────
    # Invert `vhmat` in its own storage → it now holds V = H⁻¹. The
    # pre-invert pos-def Hessian is not needed afterwards, which removes
    # the previous `vhmat_pd = copy(...)` round-trip. On factorization
    # failure the partial Bunch–Kaufman state in `vhmat` is discarded by
    # the `_hesse_diagonal_failure` path (which rebuilds V from `g2`).
    inv_ok = true
    try
        sym_invert!(Symmetric(vhmat, :U))
    catch _
        inv_ok = false
    end

    if !inv_ok
        return _hesse_diagonal_failure(state, g2, prec, ncalls(cf), MnInvertFailed)
    end

    # ── New EDM with the refined error ────────────────────────────
    refined_grad = FunctionGradient(grd, g2, gst)
    new_err_status = made_pd ? MnMadePosDef : MnHesseValid
    new_dcov = made_pd ? 1.0 : 0.0
    new_err = MinimumError(Symmetric(vhmat, :U), new_dcov, new_err_status, true)
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
of `1/g2[j]` (clamped to 1 when g2[j] is degenerate). Mirrors C++
`MnHesse.cxx:177-184` (and the analogous block at lines 216-223 for
maxcalls overrun) — with a documented divergence on the very-large-g2
case described below.

# Intentional divergence from C++

C++ uses a **double** eps2 check:

```cpp
double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
vhmat(j, j) = tmp < prec.Eps2() ? 1. : tmp;
```

The first check guards against `1/0` when `g2 < eps2`. The second
check fires when `1/g2 < eps2`, i.e. `g2 > 1/eps2 ≈ 6.7e7`, and
replaces the meaningful `1/g2` with `1.0`. The intuition "1/g2 = 1e-10
means the parameter is well determined" suggests the second clamp is
counterproductive — and PR #6 (commit a1fa015) acted on that
intuition, removing it. But the empirical IAM x_jm warm-start audit
(docs/dev/DAVIDON_CXX_AUDIT.md) revealed the opposite: that clamp is
**load-bearing** for cross-basin convergence. The story:

- At the IAM x_jm warm start, all 8 active LECs have `g2 ≈ 1e10` and
  the 9th coord has `g2 = 0` (zero-curvature). The clamp produces
  `V ≈ I` (1 on every diagonal). Newton step `−V·g ≈ −g` with
  `|g| ≈ 500` is large enough to cross the local-minimum boundary;
  iminuit walks to χ²=322.59 over 8 inner-DFP iters.
- PR #6's `V = diag(1/g2)` produces step `−V·g ≈ 1e-10·g ≈ 1e-8`
  per coord — sub-precision, no walk, MIGRAD bails at χ²=325.80.
- The OTHER regime PR #6 cared about (paras0 cold seed with
  `|g| ≈ 1e6`): C++ `V ≈ I` does cause line-search blowup, MIGRAD
  bails. iminuit hits the same trap. The "fix" was JuMinuit-only.

We restored the C++ clamp here because the IAM x_jm correctness gain
outweighs the paras0 cold-seed regression (which exists in iminuit
too — see iminuit S=2 from paras0 = 1268.65 stuck). The standard
HEP workflow of "S=0/1 from cold seeds, polish at S=2" already
avoids the paras0+S=2 trap.

# Edge cases

- `g2[j] = Inf` → `1/g2 = 0`. Both clamps fall back to `V[j,j] = 1`
  (the first check on `g2 < eps2` is false, then the second check
  on `tmp < eps2` is true → 1).
- `g2[j] = NaN` → first check uses `abs(NaN) < eps2 == false` →
  goes to `1/NaN = NaN`; second check uses `abs(NaN) < eps2 == false`
  → propagates NaN to `V[j,j]`. Pathological FCN, downstream
  MnPosDef has to recover.
- `g2[j] = -1e10` (negative — defensively): both checks operate on
  `abs(...)`, so first check sees `1e10 ≥ eps2` → uses `1/(-1e10) =
  -1e-10`; second check on `abs(-1e-10) < eps2` → fires → `V[j,j]
  = 1`. Matches C++ behavior (which uses raw `g2(j)` without abs in
  the first check; for `g2 < 0` the first check `g2 < eps2` is true
  → C++ falls back to 1 immediately).

See `docs/dev/DAVIDON_CXX_AUDIT.md` for the audit trail.
"""
function _hesse_diagonal_failure(state::MinimumState, g2::Vector{Float64},
                                  prec::MachinePrecision, nfcn::Integer,
                                  status::CovStatus)
    n = length(g2)
    M = zeros(Float64, n, n)
    # Mirrors C++ MnHesse.cxx:177-180 fallback EXACTLY — the two-stage
    # clamp:
    #     tmp = (g2[j] < eps2) ? 1 : 1/g2[j]
    #     vhmat[j,j] = (tmp < eps2) ? 1 : tmp
    # The SECOND clamp catches the case `1/g2 < eps2` (i.e., g2 too
    # large, > 1/eps2 ≈ 3e7) and falls back to V[j,j] = 1. PR #6
    # (commit a1fa015) had removed this clamp on the theory it was a
    # C++ bug; the empirical IAM x_jm + iminuit cross-check audit
    # (docs/dev/DAVIDON_CXX_AUDIT.md) showed the clamp is what produces
    # the iminuit-style V ≈ I that walks the warm start across basins
    # to χ²=322.59 in 8 DFP iters. Restored here for C++ parity.
    @inbounds for j in 1:n
        tmp = abs(g2[j]) < prec.eps2 ? 1.0 : 1.0 / g2[j]
        M[j, j] = abs(tmp) < prec.eps2 ? 1.0 : tmp
    end
    err = MinimumError(Symmetric(M, :U), status)
    return MinimumState(state.parameters, err, state.gradient,
                        state.edm, nfcn)
end

# ─────────────────────────────────────────────────────────────────────────────
# Standalone HESSE from vectors — the FCN+params+errors C++ overload.
#
# Mirrors C++ `MnHesse::operator()(const FCNBase&, const std::vector<double>&
# par, const std::vector<double>& err, unsigned int maxcalls)`
# (reference/Minuit2_cpp/inc/Minuit2/MnHesse.h:57-74): compute a
# Hessian/covariance at a *user-supplied point* WITHOUT a prior MIGRAD.
# ─────────────────────────────────────────────────────────────────────────────

"""
    HesseResult

Result of a standalone [`hesse`](@ref)`(f, x0, errors)` call. Exposes the
covariance and errors computed at the supplied point.

# Fields

- `x::Vector{Float64}` — the point the Hessian was evaluated at.
- `covariance::Matrix{Float64}` — the parameter covariance `2·up·V` where
  `V = inv(H)` is the inverse Hessian (matches
  [`covariance`](@ref)`(::FunctionMinimum)`; for χ² fits with `up=1` it is
  `2·V`).
- `errors::Vector{Float64}` — 1σ errors `sqrt(diag(covariance))`.
- `edm::Float64` — expected distance to minimum at the point.
- `nfcn::Int` — FCN calls consumed (seed gradient + Hessian passes).
- `status::CovStatus` — covariance status (see [`CovStatus`](@ref)).
- `valid::Bool` — `true` if the covariance is usable (`MnHesseValid` /
  `MnMadePosDef`).
- `state::MinimumState` — the full internal state, for advanced consumers
  (raw `inv_hessian`, gradient, …).
"""
struct HesseResult
    x::Vector{Float64}
    covariance::Matrix{Float64}
    errors::Vector{Float64}
    edm::Float64
    nfcn::Int
    status::CovStatus
    valid::Bool
    state::MinimumState
end

function HesseResult(state::MinimumState, up::Real)
    V = state.error.inv_hessian          # Symmetric{:U} inverse Hessian
    n = size(V, 1)
    factor = 2.0 * Float64(up)
    cov = Matrix{Float64}(undef, n, n)
    # Read symmetrically through the Symmetric view (not `parent`) so the
    # lower triangle is mirrored, then scale by 2·up — same convention as
    # `covariance(::FunctionMinimum)` (result.jl) and the int→ext path
    # (`_internal_to_external_results`, migrad_bounded.jl:222).
    @inbounds for j in 1:n, i in 1:n
        cov[i, j] = factor * V[i, j]
    end
    errs = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        errs[i] = sqrt(max(cov[i, i], 0.0))
    end
    return HesseResult(copy(state.parameters.x), cov, errs, state.edm,
                       state.nfcn, state.error.status, is_valid(state.error),
                       state)
end

function Base.show(io::IO, r::HesseResult)
    status_str = r.status == MnHesseValid ? "valid" :
                 r.status == MnMadePosDef ? "made-pos-def" :
                 r.status == MnHesseFailed ? "hesse-failed" :
                 r.status == MnInvertFailed ? "invert-failed" : "not-pos-def"
    print(io, "HesseResult(n=", length(r.x), ", errors=", r.errors,
              ", status=", status_str, ", nfcn=", r.nfcn, ")")
end

"""
    hesse(f, x0, errors; up=1.0, strategy=Strategy(1),
          prec=MachinePrecision(), maxcalls=0, print_level=0) -> HesseResult

Compute a full numerical Hessian (and the derived covariance + errors) for
the FCN `f` at the point `x0`, using `errors` as the initial per-parameter
step sizes. **No MIGRAD is run** — this is the standalone `MnHesse`, the
C++ `MnHesse::operator()(FCNBase, par, err, maxcalls)` overload
(`reference/Minuit2_cpp/inc/Minuit2/MnHesse.h:57-74`).

The errors are computed at the **given point**, which need not be a
minimum — that is the caller's responsibility. At a non-stationary point
the Hessian is still well defined; `MnPosDef` enforcement may flip the
status to `MnMadePosDef` if the curvature is not positive-definite there
(e.g. a saddle).

# Arguments

- `f` — the user FCN; takes an `AbstractVector{Float64}`, returns a real.
- `x0::AbstractVector{<:Real}` — the point to evaluate at.
- `errors::AbstractVector{<:Real}` — initial step sizes (one per parameter),
  the natural scale of each parameter; used to seed the central-difference
  Hessian and refined internally.

# Keyword arguments

- `up::Real=1.0` — error definition (`1.0` for χ², `0.5` for NLL).
- `strategy::Strategy=Strategy(1)` — HESSE refinement level.
- `prec::MachinePrecision`, `maxcalls::Integer`, `print_level::Integer` —
  forwarded to the internal Hessian pass (`maxcalls=0` ⇒ default budget).

# Returns

A [`HesseResult`](@ref) exposing `.covariance`, `.errors`, `.edm`,
`.status`, and `.valid`.

# Example

```julia
julia> r = hesse(x -> (x[1]-1)^2 + 4(x[2]-2)^2, [0.0, 0.0], [1.0, 1.0]);

julia> r.covariance        # ≈ [1.0 0.0; 0.0 0.25]  (2·up·inv(H), H=diag(2,8))
```
"""
function hesse(
    f::F,
    x0::AbstractVector{<:Real},
    errors::AbstractVector{<:Real};
    up::Real = 1.0,
    strategy::Strategy = Strategy(1),
    prec::MachinePrecision = MachinePrecision(),
    maxcalls::Integer = 0,
    print_level::Integer = 0,
) where {F}
    n = length(x0)
    length(errors) == n ||
        throw(DimensionMismatch("hesse: errors length $(length(errors)) != x0 length $n"))
    cf = CostFunction(f, up)

    # Build a seed MinimumState AT x0. Unlike `seed_state`, we deliberately
    # do NOT run `negative_g2_line_search` (which would move the point) or
    # the Strategy(2) seed-time Hesse bootstrap — the contract is "errors at
    # the given point". Mirrors C++ MnHesse's user-state overload, which
    # only computes the gradient before handing off to the core Hessian pass.
    x = collect(Float64, x0)
    dirin = collect(Float64, errors)
    fval = cf(x)
    par = MinimumParameters(x, dirin, fval)
    grad = FunctionGradient(zeros(Float64, n), zeros(Float64, n),
                             zeros(Float64, n))
    initial_gradient!(grad, par, dirin, cf.up, prec)
    x_work = Vector{Float64}(undef, n)
    numerical_gradient!(grad, x_work, par, grad, cf, strategy, prec)

    # Diagonal seed inverse-Hessian (C++ MnSeedGenerator.cxx:69-70). The
    # internal hesse re-derives g2 from scratch via central differences, so
    # this only seeds the starting step; a non-positive g2 here is harmless.
    mat = zeros(Float64, n, n)
    @inbounds for i in 1:n
        mat[i, i] = abs(grad.g2[i]) > prec.eps2 ? 1.0 / grad.g2[i] : 1.0
    end
    err0 = MinimumError(Symmetric(mat, :U), 1.0)
    edm0 = estimate_edm!(x_work, grad, err0)
    seed = MinimumState(par, err0, grad, edm0, ncalls(cf))

    final = hesse(cf, seed, strategy; prec = prec, maxcalls = maxcalls,
                  print_level = print_level)
    return HesseResult(final, cf.up)
end
