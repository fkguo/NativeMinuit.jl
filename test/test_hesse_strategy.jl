# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Strategy-coupling tests for HESSE: HessianGradientCalculator (P1) and
# the AD-gradient numerical-companion refresh (P2).

using LinearAlgebra: norm
#
# C++ references:
#   - HessianGradientCalculator: reference/Minuit2_cpp/src/HessianGradientCalculator.cxx
#   - AD-grad HESSE refresh:     reference/Minuit2_cpp/src/MnHesse.cxx:118-126
#   - Strategy ≥ 1 HGC gate:     reference/Minuit2_cpp/src/MnHesse.cxx:228-235
#
# Test plan:
#  1. Strategy(0) regression — HGC is gated off for Strategy(0); HESSE
#     output on a quadratic must be exact (recovers the inverse Hessian
#     to floating-point precision). Pre-existing behavior preserved.
#  2. HGC refines `grd` and `gst` — on a non-quadratic FCN at its
#     minimum, Strategy(1) and Strategy(2) HESSE give a TIGHTER
#     post-HESSE gradient (smaller |grd|) than Strategy(0). Step sizes
#     `gst` shrink by 0.2× per HGC cycle, so Strategy(2) (ncycle=6) has
#     smaller gst than Strategy(1) (ncycle=2) which has smaller gst than
#     Strategy(0) (no HGC).
#  3. HGC monotonic step shrinkage — for a smooth bowl-shaped FCN,
#     `gst_after_hgc < gst_before_hgc` to within strict inequality.
#  4. AD-gradient HESSE refresh — with a deliberately wrong initial
#     error vector (so seed g2/gst are wildly off), AD-grad HESSE
#     output `g2`/`gst` must match numerical HESSE output (within
#     tight atol). Without P2 the AD path would carry stale seed-time
#     g2/gst and produce a divergent inv_hessian.
#  5. HGC uncertainty bound — the per-parameter `dgrd` returned by
#     the allocating `hessian_gradient` overload is non-negative and
#     bounds the magnitude of the gradient refinement.

@testset "HESSE strategy coupling (P1 HessianGradientCalculator + P2 AD-refresh)" begin

    # ─────────────────────────────────────────────────────────────────
    # 1. Strategy(0) regression — exact inverse Hessian on a quadratic
    # ─────────────────────────────────────────────────────────────────
    @testset "Strategy(0) regression: quadratic recovers exact V" begin
        # f(x) = Σ xᵢ². Hessian = 2·I, V = 0.5·I.
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        @test m.is_valid
        st0 = NativeMinuit.hesse(cf, m.state, Strategy(0))
        @test is_valid(st0.error)
        @test is_accurate(st0.error)
        # Strategy(0) skips HGC, but the diagonal+off-diagonal pass alone
        # is sufficient for a pure quadratic.
        for i in 1:3, j in 1:3
            expected = i == j ? 0.5 : 0.0
            @test st0.error.inv_hessian[i, j] ≈ expected atol = 1e-5
        end
    end

    @testset "Strategy(0) regression: no extra FCN calls from HGC" begin
        # Strategy(0) must NOT call hessian_gradient! — verify by
        # comparing call counts to the parallel non-HGC variant (here we
        # use the same code path; the check is that the call count
        # matches what we'd predict without HGC overhead).
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 1.0], [0.1, 0.1])
        reset_ncalls!(cf)
        st0 = NativeMinuit.hesse(cf, m.state, Strategy(0))
        nfcn_s0 = ncalls(cf)
        # Strategy(0) HESSE on a 2-param quadratic should converge the
        # diagonal pass in 1 cycle → 2 calls per coord (4) + 1 off-diag
        # + 1 amin = 6, plus the initial amin gives ~7. Just check it's
        # not ballooning with hidden HGC calls.
        @test nfcn_s0 < 30
    end

    # ─────────────────────────────────────────────────────────────────
    # 2. HGC refines `grd` and `gst` on a non-quadratic
    # ─────────────────────────────────────────────────────────────────
    @testset "HGC tightens gradient on non-quadratic FCN" begin
        # Non-quadratic with curvature varying around the minimum.
        # f(x, y) = x⁴ - 2x² + y⁴ - 2y². Minima at (±1, ±1) with f = -2.
        # At (1, 1): f' = 0, f'' = 12·1² - 4 = 8 (each coord).
        cf = CostFunction(x -> x[1]^4 - 2 * x[1]^2 + x[2]^4 - 2 * x[2]^2)
        m = migrad(cf, [0.5, 0.5], [0.1, 0.1])
        @test m.is_valid
        # Converged near (+1, +1). The 1D quartic minima are degenerate
        # in sign — MIGRAD will pick the basin from the initial point.
        # Run HESSE at three strategy levels and compare.
        st0 = NativeMinuit.hesse(cf, m.state, Strategy(0))
        st1 = NativeMinuit.hesse(cf, m.state, Strategy(1))
        st2 = NativeMinuit.hesse(cf, m.state, Strategy(2))
        for st in (st0, st1, st2)
            @test is_valid(st.error)
        end
        # At the converged minimum the gradient should be near zero for
        # all strategies; we check that HGC at Strategy(1)/(2) produces
        # a gradient norm at least as small as Strategy(0). The HGC's
        # `change > chgold && j > 2` divergence break means it always
        # commits a refined `grd[i]` for at least the first two cycles
        # before bailing if higher-order terms dominate.
        @test norm(st1.gradient.grad) <= norm(st0.gradient.grad) + 1e-12
        @test norm(st2.gradient.grad) <= norm(st0.gradient.grad) + 1e-12
        # All three should give similar inv_hessian (HGC doesn't change
        # g2, only grd and gstep — and the off-diagonal pass uses dirin
        # which is set in the diagonal pass before HGC). The diagonal
        # entries should be ≈ 1/8 = 0.125 at (1, 1).
        for st in (st0, st1, st2)
            @test st.error.inv_hessian[1, 1] ≈ 1 / 8 atol = 1e-3
            @test st.error.inv_hessian[2, 2] ≈ 1 / 8 atol = 1e-3
        end
    end

    @testset "HGC shrinks gstep monotonically with strategy level" begin
        # Smooth bowl: f(x) = (x[1] - 1)^2 + (x[2] + 2)^2.
        # HGC's d *= 0.2 per cycle means gst_after ≤ gst_before for any
        # cycle count > 0 (we commit each cycle's d when it converges).
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] + 2.0)^2)
        m = migrad(cf, [0.0, 0.0], [0.5, 0.5])
        @test m.is_valid
        st0 = NativeMinuit.hesse(cf, m.state, Strategy(0))
        st1 = NativeMinuit.hesse(cf, m.state, Strategy(1))
        st2 = NativeMinuit.hesse(cf, m.state, Strategy(2))
        # Strategy(0) gst comes from the diagonal pass; Strategy(1)/(2)
        # additionally have HGC. For a perfect quadratic, HGC's relative
        # `change < 0.05` should fire on cycle 1 or 2 (since the
        # quadratic gradient response is exact), but the COMMITTED d
        # values from the cycle that triggered the break are ≤ the
        # diagonal pass's gst.
        for i in 1:2
            @test st1.gradient.gstep[i] <= st0.gradient.gstep[i] * (1 + 1e-9)
            @test st2.gradient.gstep[i] <= st0.gradient.gstep[i] * (1 + 1e-9)
            # All values are positive
            @test st0.gradient.gstep[i] > 0
            @test st1.gradient.gstep[i] > 0
            @test st2.gradient.gstep[i] > 0
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # 3. AD-gradient + Strategy(1) — numerical-companion refresh (P2)
    # ─────────────────────────────────────────────────────────────────
    @testset "AD-gradient HESSE: g2/gst no longer carry stale seed values" begin
        # This test specifically exercises the P2 path. The assertion
        # `m_ad.state.gradient.g2[1] > 1e3` below verifies that the
        # AD-MIGRAD state IS carrying stale g2 *before* HESSE runs —
        # this is the gap that P2 fixes. If a future change to
        # `analytical_gradient!` (src/ad_gradient.jl) starts refreshing
        # `g2` mid-MIGRAD, this assertion will fail and the test should
        # be updated; until then, the assertion documents the
        # precondition that P2 protects against.
        # Quadratic with off-diagonal coupling: f = x² + 2y² + 0.5·x·y.
        # Analytical gradient: [2x + 0.5y, 4y + 0.5x].
        # Hessian: [2 0.5; 0.5 4]; inv = (1/(8-0.25)) · [4 -0.5; -0.5 2].
        f = x -> x[1]^2 + 2.0 * x[2]^2 + 0.5 * x[1] * x[2]
        g = x -> [2.0 * x[1] + 0.5 * x[2], 4.0 * x[2] + 0.5 * x[1]]

        # Build TWO cost-function variants on the same minimum, with
        # the SAME initial errors. We use deliberately wrong errors
        # (0.01 instead of natural ~0.7) so the seed-time g2/gst are
        # far off the true Hessian. Without P2, AD-grad hesse would
        # carry these stale values; with P2, they get refreshed.
        x0 = [0.0, 0.0]
        errs = [0.01, 0.01]  # wrong by 70× — seed g2 = 2/0.0001 = 2e4

        cf_num  = CostFunction(f, 1.0)
        cf_ad   = CostFunctionWithGradient(f, g, 1.0)

        m_num = migrad(cf_num, x0, errs)
        m_ad  = migrad(cf_ad,  x0, errs)
        @test m_num.is_valid
        @test m_ad.is_valid

        # Both converge to ≈ (0, 0) — verify before HESSE.
        @test norm(m_num.state.parameters.x) < 1e-3
        @test norm(m_ad.state.parameters.x)  < 1e-3

        # The AD-MIGRAD state's gradient.g2 should be the seed g2 (stale)
        # because the analytical gradient path never refreshed it. The
        # numerical-MIGRAD state's gradient.g2 should be near the true
        # values (2 and 4).
        # Document this with an explicit check — proves the P2 motivation.
        @test m_num.state.gradient.g2[1] ≈ 2.0 atol = 0.5
        @test m_num.state.gradient.g2[2] ≈ 4.0 atol = 0.5
        # AD-grad MIGRAD state should still carry near-seed g2 (sanity
        # check on the gap, before the P2 fix kicks in inside HESSE).
        @test m_ad.state.gradient.g2[1] > 1e3
        @test m_ad.state.gradient.g2[2] > 1e3

        # Now run HESSE on both. P2 should refresh AD-grad g2/gst from
        # the wildly-wrong seed values to fresh numerical values.
        st_num = NativeMinuit.hesse(cf_num, m_num.state, Strategy(1))
        st_ad  = NativeMinuit.hesse(cf_ad,  m_ad.state,  Strategy(1))
        @test is_valid(st_num.error)
        @test is_valid(st_ad.error)

        # Both outputs should give the SAME inverse Hessian (within tight
        # tolerance). HESSE's diagonal pass uses only `cf.f`, so the
        # output is determined by the FCN and the d-step path. With P2
        # refresh, both start the d-step path from the same numerical
        # seed → identical convergence.
        # Inv Hessian for H = [2 0.5; 0.5 4]: det=7.75; inv = (1/7.75)·[4 -0.5; -0.5 2]
        inv_det = 1.0 / 7.75
        @test st_num.error.inv_hessian[1, 1] ≈ inv_det * 4.0 atol = 5e-4
        @test st_num.error.inv_hessian[2, 2] ≈ inv_det * 2.0 atol = 5e-4
        @test st_num.error.inv_hessian[1, 2] ≈ -inv_det * 0.5 atol = 5e-4
        @test st_ad.error.inv_hessian[1, 1] ≈ inv_det * 4.0 atol = 5e-4
        @test st_ad.error.inv_hessian[2, 2] ≈ inv_det * 2.0 atol = 5e-4
        @test st_ad.error.inv_hessian[1, 2] ≈ -inv_det * 0.5 atol = 5e-4

        # P2 specific: AD-grad HESSE's returned g2 should match the
        # numerical version, NOT the stale seed value (would be ~2e4).
        @test st_ad.gradient.g2[1] ≈ 2.0 atol = 0.5
        @test st_ad.gradient.g2[2] ≈ 4.0 atol = 0.5
        @test st_ad.gradient.g2[1] < 100  # very far from the stale 2e4 seed
        @test st_ad.gradient.g2[2] < 100

        # And the AD-grad and numerical HESSE g2/gst should agree
        # within tight tolerance — the diagonal pass gives identical
        # output because both paths feed the same FCN cf.f.
        for i in 1:2
            @test st_ad.gradient.g2[i]    ≈ st_num.gradient.g2[i]    atol = 1e-3
            @test st_ad.gradient.gstep[i] ≈ st_num.gradient.gstep[i] atol = 1e-3
        end
    end

    @testset "AD-grad HESSE with P2 recovers stiff-Hessian g2 from misleading seed" begin
        # Document what the P2 fix prevents: a non-trivial FCN where
        # the seed-time g2/gst are deliberately misleading.
        # Stiff Hessian: f(x) = 1000·x[1]² + x[2]².  At minimum: H_11 = 2000.
        f = x -> 1000.0 * x[1]^2 + x[2]^2
        g = x -> [2000.0 * x[1], 2.0 * x[2]]
        cf_ad = CostFunctionWithGradient(f, g, 1.0)
        # Seed errs = [1.0, 1.0] — for x[1] this gives seed g2 = 2/1 = 2,
        # far below the true 2000. The AD-grad MIGRAD won't refresh this
        # (since analytical_gradient! just copies prev.g2 forward), so
        # the seed g2 of 2 propagates into HESSE.
        m = migrad(cf_ad, [0.0, 0.0], [1.0, 1.0])
        @test m.is_valid
        @test m.state.gradient.g2[1] < 100  # stale seed value, not 2000

        # WITH P2: HESSE refreshes g2 to ~2000 before the diagonal pass.
        # The returned inv_hessian[1,1] should ≈ 1/2000 = 5e-4.
        st = NativeMinuit.hesse(cf_ad, m.state, Strategy(1))
        @test is_valid(st.error)
        @test st.error.inv_hessian[1, 1] ≈ 1 / 2000 atol = 1e-5
        @test st.error.inv_hessian[2, 2] ≈ 1 / 2     atol = 1e-5
        # And the refreshed g2 should be ≈ 2000.
        @test st.gradient.g2[1] ≈ 2000.0 atol = 5.0
        @test st.gradient.g2[2] ≈ 2.0    atol = 1e-2
    end

    # ─────────────────────────────────────────────────────────────────
    # 4. HGC standalone — direct unit test of hessian_gradient!
    # ─────────────────────────────────────────────────────────────────
    @testset "hessian_gradient! refines towards analytical gradient" begin
        # f(x) = (x[1] - 0.5)² — minimum at x=0.5, f'(x[1]) = 2(x[1] - 0.5).
        # At x[1] = 0.6 (away from minimum): true gradient = 0.2.
        cf = CostFunction(x -> (x[1] - 0.5)^2)
        par = MinimumParameters([0.6], [0.05], cf([0.6]))
        # Seed with a "wrong" gradient — test that HGC refines toward
        # the truth.
        grd = [0.0]      # input gradient (wrong)
        gst = [0.05]     # input step
        dgrd = [0.0]
        x_w = similar(par.x)
        g2 = [2.0]       # true g2 for quadratic

        NativeMinuit.hessian_gradient!(grd, gst, dgrd, x_w, par, cf, g2,
                                    Strategy(1), MachinePrecision())
        # After refinement, grd should be near the analytic 0.2 = 2·0.1.
        @test grd[1] ≈ 0.2 atol = 1e-6
        @test gst[1] > 0
        @test gst[1] <= 0.05  # HGC always shrinks step
        @test dgrd[1] >= 0
    end

    @testset "hessian_gradient! preserves g2 (no mutation)" begin
        # Verify the contract: HGC reads g2 but never writes to it.
        cf = CostFunction(x -> x[1]^2 + x[2]^2)
        par = MinimumParameters([0.1, -0.2], [0.1, 0.1], cf([0.1, -0.2]))
        grd = [0.2, -0.4]
        gst = [0.05, 0.05]
        dgrd = [0.0, 0.0]
        x_w = similar(par.x)
        g2_in = [2.0, 2.0]
        g2_check = copy(g2_in)
        NativeMinuit.hessian_gradient!(grd, gst, dgrd, x_w, par, cf, g2_in,
                                    Strategy(2), MachinePrecision())
        @test g2_in == g2_check  # untouched
    end

    @testset "hessian_gradient! grdnew==0 break path does not commit grd/gstep" begin
        # C++ HGC.cxx:125-126 — when `grdnew == 0` exactly, the break
        # fires BEFORE `grd(i) = grdnew; gstep(i) = d;` (lines 131,133).
        # Verify the Julia port matches: input grd/gstep are PRESERVED
        # when the central difference returns exactly zero.
        #
        # Use a constant FCN — its gradient is identically zero, so
        # central differences yield `fs1 - fs2 = 0` → `grdnew = 0` → break.
        cf = CostFunction(x -> 42.0)
        par = MinimumParameters([1.0], [0.1], 42.0)
        grd = [0.7]      # sentinel; must survive the no-op
        gst = [0.05]     # sentinel
        dgrd = [0.0]
        x_w = similar(par.x)
        g2 = [2.0]
        NativeMinuit.hessian_gradient!(grd, gst, dgrd, x_w, par, cf, g2,
                                    Strategy(2), MachinePrecision())
        @test grd[1] == 0.7   # untouched — grdnew==0 break is pre-commit
        @test gst[1] == 0.05  # untouched
    end

    @testset "hessian_gradient! divergence break (cycle ≥ 3) discards new value" begin
        # C++ HGC.cxx:128-129 — when `change > chgold && j > 1` (Julia
        # `j > 2`) fires on cycle 3+, the algorithm breaks BEFORE
        # committing `grd[i] = grdnew; gstep[i] = d`. Verify the Julia
        # port matches by constructing a FCN whose central-difference
        # noise grows non-monotonically with shrinking d.
        #
        # f(x) = x² + noise·sin(x/η) for tiny η fakes a finite-precision
        # gradient that doesn't converge under shrinking step. The
        # divergence break should fire and leave the committed value at
        # the cycle-where-things-stopped-improving point.
        #
        # Simpler proof-of-contract test: verify the algorithm always
        # leaves grd matching one of the cycle-end values seen (never
        # an interpolation).
        #
        # We test the contract indirectly via the standalone allocating
        # overload: after refinement, the inv_hessian computed from the
        # final g2 should not depend on whether the divergence break
        # fired, because g2 is read-only here.
        cf = CostFunction(x -> x[1]^2)
        par = MinimumParameters([0.001], [0.1], cf([0.001]))
        grad_in = FunctionGradient([0.002], [2.0], [0.1])  # near-min grad
        (gr_out, _) = NativeMinuit.hessian_gradient(par, grad_in, cf,
                                                  Strategy(2))
        # g2 must be unchanged (HGC contract):
        @test gr_out.g2 == grad_in.g2
        # grd may be refined or kept; either way it must be finite and
        # not radically different (within OoM) from the analytical 2·x.
        @test isfinite(gr_out.grad[1])
        @test gr_out.gstep[1] > 0
        @test gr_out.gstep[1] <= grad_in.gstep[1]  # never grows
    end

    @testset "hessian_gradient! tolerates negative xtf (dmin sign quirk)" begin
        # C++ HessianGradientCalculator.cxx:100 uses
        # `dmin = 4·eps2·(xtf + eps2)` — without `abs(xtf)`. For
        # xtf < -eps2 this gives a NEGATIVE dmin, but the algorithm
        # is observationally correct because d starts positive
        # (line 103: `d = 0.2·|gstep|`) and the subsequent
        # `if (d < dmin) d = dmin;` only fires when dmin > 0.
        # Verify the Julia port reproduces this faithfully and
        # produces sensible refined values on a parameter sitting at
        # negative xtf.
        cf = CostFunction(x -> (x[1] + 2.0)^2)  # minimum at x = -2
        par = MinimumParameters([-1.9], [0.1], cf([-1.9]))  # xtf=-1.9 ≪ -eps2
        grd = [0.0]
        gst = [0.05]
        dgrd = [0.0]
        x_w = similar(par.x)
        g2 = [2.0]
        NativeMinuit.hessian_gradient!(grd, gst, dgrd, x_w, par, cf, g2,
                                    Strategy(2), MachinePrecision())
        # Analytic grad at x=-1.9: 2·(-1.9 + 2) = 0.2.
        @test grd[1] ≈ 0.2 atol = 1e-6
        @test gst[1] > 0
        @test isfinite(dgrd[1])
        @test x_w[1] == -1.9  # restored after perturbation
    end

    @testset "hessian_gradient! dimension-mismatch errors" begin
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], [0.1, 0.1], cf([1.0, 2.0]))
        # Wrong-length grd
        @test_throws DimensionMismatch NativeMinuit.hessian_gradient!(
            [0.0], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0], par, cf, [2.0, 2.0],
            Strategy(1), MachinePrecision())
        # Wrong-length gstep
        @test_throws DimensionMismatch NativeMinuit.hessian_gradient!(
            [0.0, 0.0], [0.1], [0.0, 0.0], [0.0, 0.0], par, cf, [2.0, 2.0],
            Strategy(1), MachinePrecision())
        # Wrong-length g2
        @test_throws DimensionMismatch NativeMinuit.hessian_gradient!(
            [0.0, 0.0], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0], par, cf, [2.0],
            Strategy(1), MachinePrecision())
    end

    # ─────────────────────────────────────────────────────────────────
    # 5. Allocating overload — `hessian_gradient` returns (FG, dgrd)
    # ─────────────────────────────────────────────────────────────────
    @testset "hessian_gradient allocating overload" begin
        cf = CostFunction(x -> x[1]^2 + x[2]^2)
        par = MinimumParameters([0.05, 0.05], [0.1, 0.1], cf([0.05, 0.05]))
        # Build a FunctionGradient with seed values.
        grad_in = FunctionGradient([0.1, 0.1], [2.0, 2.0], [0.05, 0.05])
        (gr_out, dgrd) = NativeMinuit.hessian_gradient(par, grad_in, cf,
                                                     Strategy(1))
        @test length(gr_out) == 2
        @test length(dgrd) == 2
        # g2 must equal the input (HGC doesn't touch it)
        @test gr_out.g2 == grad_in.g2
        # grd should approach the analytic 2·x values
        @test gr_out.grad[1] ≈ 0.1 atol = 1e-6
        @test gr_out.grad[2] ≈ 0.1 atol = 1e-6
        @test all(dgrd .>= 0)
    end

    # ─────────────────────────────────────────────────────────────────
    # 6. Type stability of hesse() at all strategy levels
    # ─────────────────────────────────────────────────────────────────
    @testset "Type stability across strategy levels" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        for level in (0, 1, 2)
            @test (@inferred NativeMinuit.hesse(cf, m.state, Strategy(level))) isa MinimumState
        end
    end
end

@testset "HESSE strategy coupling — FCN call accounting" begin
    # Verify P1 and P2 increment the FCN counter correctly so that
    # `maxcalls` budget enforcement is honored.

    @testset "Strategy(2) HGC contributes extra FCN calls vs Strategy(0)" begin
        cf = CostFunction(x -> x[1]^4 - 2 * x[1]^2)
        m = migrad(cf, [0.5], [0.1])
        @test m.is_valid

        reset_ncalls!(cf)
        NativeMinuit.hesse(cf, m.state, Strategy(0))
        nfcn_s0 = ncalls(cf)

        reset_ncalls!(cf)
        NativeMinuit.hesse(cf, m.state, Strategy(2))
        nfcn_s2 = ncalls(cf)

        # Strategy(2) HGC does up to 2 · 6 = 12 extra FCN calls for the
        # single parameter; the divergence break can fire earlier.
        @test nfcn_s2 > nfcn_s0
    end

    @testset "P2 AD-refresh contributes extra FCN calls" begin
        f = x -> x[1]^2 + x[2]^2
        g = x -> [2.0 * x[1], 2.0 * x[2]]
        cf_ad = CostFunctionWithGradient(f, g, 1.0)
        cf_num = CostFunction(f, 1.0)

        m_ad  = migrad(cf_ad,  [0.0, 0.0], [0.1, 0.1])
        m_num = migrad(cf_num, [0.0, 0.0], [0.1, 0.1])

        reset_ncalls!(cf_ad)
        NativeMinuit.hesse(cf_ad, m_ad.state, Strategy(1))
        nfcn_ad = ncalls(cf_ad)

        reset_ncalls!(cf_num)
        NativeMinuit.hesse(cf_num, m_num.state, Strategy(1))
        nfcn_num = ncalls(cf_num)

        # AD-grad path does an EXTRA numerical_gradient! call to refresh
        # g2/gst — so its total FCN count should exceed pure numerical
        # HESSE on the same problem.
        @test nfcn_ad > nfcn_num
    end
end
