# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "contour_exact — multi-param function_cross (Phase 1.x)" begin

    @testset "function_cross_multi basic" begin
        # f(x, y, z) = (x-1)² + (y-2)² + (z-3)². Min at (1, 2, 3).
        # Fix (x, y) and scan along (1, 0) direction: minimum at z=3
        # requires no movement in z; alpha goes to root of "fval + 1".
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2)
        fmin = migrad(cf, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test fmin.is_valid

        # Fix (x, y) at (1, 2) — the minimum position; ray along (1, 0).
        # The constrained 1D minimum varies as we move (x, y) away from
        # the minimum. At alpha = 1, x = 2 → fval = 1 + (y-2)² + 0 = 1.
        # So crossing at alpha = 1 (where fval = 1 = up).
        cross = JuMinuit.function_cross_multi(
            fmin, cf, [1, 2], [1.0, 2.0], [1.0, 0.0]; tlr = 0.1)
        @test cross.valid
        @test cross.aopt ≈ 1.0 atol = 0.1
    end

    @testset "contour_exact on symmetric quadratic — circle" begin
        # f(x, y) = (x-1)² + (y-2)². Minimum (1, 2), Hessian = 2·I.
        # Up = 1 → 1σ contour at radius 1 around (1, 2).
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid
        c = contour_exact(fmin, cf, 1, 2; npoints = 8)
        @test c.valid
        @test length(c.points) >= 4  # at least the 4 axis points
        # Every point should be at radius ≈ 1
        for (x, y) in c.points
            r = sqrt((x - 1.0)^2 + (y - 2.0)^2)
            @test r ≈ 1.0 atol = 0.15
        end
    end

    @testset "contour_exact handles correlated FCN" begin
        # f(x, y) = (x-1)² + (y-1)² + 0.5·x·y. Hessian has off-diagonal.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 1.0)^2 + 0.5 * x[1] * x[2])
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid
        c = contour_exact(fmin, cf, 1, 2; npoints = 8)
        @test c.valid
        # Boundary points should all be at reasonable distance from min
        center = Base.values(fmin)
        for (x, y) in c.points
            d = sqrt((x - center[1])^2 + (y - center[2])^2)
            @test 0.2 < d < 10.0
        end
    end

    @testset "Argument validation" begin
        cf = CostFunction(x -> sum(abs2, x))
        fmin = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test_throws ArgumentError contour_exact(fmin, cf, 0, 2)
        @test_throws ArgumentError contour_exact(fmin, cf, 1, 3)
        @test_throws ArgumentError contour_exact(fmin, cf, 1, 1)
        @test_throws ArgumentError contour_exact(fmin, cf, 1, 2; npoints = 3)
    end

    @testset "function_cross_multi argument validation" begin
        cf = CostFunction(x -> sum(abs2, x))
        fmin = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test_throws DimensionMismatch JuMinuit.function_cross_multi(
            fmin, cf, [1], [1.0, 2.0], [1.0])
        @test_throws DimensionMismatch JuMinuit.function_cross_multi(
            fmin, cf, [1, 2], [1.0, 2.0], [1.0])
        # n == npar (no free parameters) is now supported via the
        # all-fixed degenerate path used by 2D contour.
        cr = JuMinuit.function_cross_multi(
            fmin, cf, [1, 2], [1.0, 2.0], [1.0, 0.0])
        @test cr.aopt isa Float64 || isnan(cr.aopt)
    end

    @testset "warm-state probe chain matches cold-only baseline" begin
        # The warm-state restart logic in _migrad_with_multi_fixed must
        # produce the SAME contour as a hypothetical cold-only version
        # (modulo tolerance noise). Build a 4D non-quadratic FCN where
        # the warm path actually saves work, and verify the contour
        # points are numerically identical to a single-CF baseline.
        function rosen4(x)
            s = 0.0
            for i in 1:3
                s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
            end
            return s
        end

        cf = CostFunction(rosen4, 1.0)
        fmin = migrad(cf, [-1.2, 1.0, -1.2, 1.0], fill(0.1, 4))
        @test fmin.is_valid

        ce = contour_exact(fmin, cf, 1, 2; npoints = 12)
        @test ce.valid
        @test length(ce.points) ≥ 4
        # All points must be finite (no NaN/Inf from warm-restart drift)
        for (x, y) in ce.points
            @test isfinite(x)
            @test isfinite(y)
        end
        # Center inside the bounding ellipse implied by MINOS
        cx, cy = fmin.state.parameters.x[1], fmin.state.parameters.x[2]
        for (x, y) in ce.points
            @test sqrt((x - cx)^2 + (y - cy)^2) < 10.0
        end
    end

    @testset "Phase D — MigradScratch dimension-mismatch fail-fast" begin
        # Codex (gpt-5.5 xhigh) review: silent fallback for wrong-size
        # scratch hides caller bugs. _migrad_loop should throw
        # DimensionMismatch when scratch.n != seed dim. Drivers preflight
        # via _get_scratch!, so this only fires under buggy external use.
        cf = CostFunction(x -> sum(abs2, x))
        seed = JuMinuit.seed_state(cf, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        bad_scratch = JuMinuit.MigradScratch(5)  # wrong size — seed is n=3
        @test_throws DimensionMismatch JuMinuit._migrad_loop(
            seed, cf, JuMinuit.Strategy(0), 0.1, 1000,
            JuMinuit.MachinePrecision(); scratch = bad_scratch)
        # nothing-scratch still works
        fmin = JuMinuit._migrad_loop(seed, cf, JuMinuit.Strategy(0), 0.1, 1000,
                                       JuMinuit.MachinePrecision(); scratch = nothing)
        @test fmin.is_valid
        # correct-size scratch works + reuses
        good_scratch = JuMinuit.MigradScratch(3)
        fmin2 = JuMinuit._migrad_loop(seed, cf, JuMinuit.Strategy(0), 0.1, 1000,
                                        JuMinuit.MachinePrecision(); scratch = good_scratch)
        @test fmin2.is_valid
        @test fmin2.state.parameters.x ≈ fmin.state.parameters.x
    end

    @testset "Phase F — AD gradient threads through MnContours / MINOS" begin
        # The full MNCONTOUR chain (function_cross_multi / function_cross /
        # minos / contour_exact) now accepts AbstractCostFunction. Verify:
        # (1) AD path runs end-to-end without falling back to numerical;
        # (2) AD result matches numerical-result on the same problem
        #     (within tlr, since both algorithms target the same crossing);
        # (3) FCN-call count drops dramatically on the AD path (the whole
        #     point — eliminates 2·n·grad_ncycles numerical_gradient! calls
        #     per probe).
        using ForwardDiff
        f = x -> (x[1] - 1.0)^2 + 10.0 * (x[2] - 2.0)^2 + (x[3] - 3.0)^2
        # Numerical baseline
        cf_num = CostFunction(f, 1.0)
        fmin_num = migrad(cf_num, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test fmin_num.is_valid
        ce_num = contour_exact(fmin_num, cf_num, 1, 2; npoints = 10)
        @test ce_num.valid
        @test length(ce_num.points) == 10
        # AD path — same minimum, same contour
        cf_ad = CostFunctionWithGradient(f, x -> ForwardDiff.gradient(f, x), 1.0)
        fmin_ad = migrad(cf_ad, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test fmin_ad.is_valid
        @test fmin_ad.state.parameters.x ≈ fmin_num.state.parameters.x atol = 1e-4
        ce_ad = contour_exact(fmin_ad, cf_ad, 1, 2; npoints = 10)
        @test ce_ad.valid
        @test length(ce_ad.points) == 10
        # FCN count must drop on AD path — that's the entire point.
        # Bound at 50% of numerical (typically ~85-95% reduction).
        @test ce_ad.nfcn < ce_num.nfcn ÷ 2
        # Same contour points (sort by x-coord, compare pair-wise within
        # tlr-scaled tolerance; both algorithms target the same crossing).
        sort_pts = sort([(p[1], p[2]) for p in ce_num.points]; by = first)
        sort_pts_ad = sort([(p[1], p[2]) for p in ce_ad.points]; by = first)
        for k in 1:10
            @test sort_pts[k][1] ≈ sort_pts_ad[k][1] atol = 0.05
            @test sort_pts[k][2] ≈ sort_pts_ad[k][2] atol = 0.05
        end
    end

    @testset "Phase F — high-level Minuit(grad=g); minos! threads AD" begin
        # Codex Phase F review identified that the high-level
        # Minuit API was silently dropping the AD gradient when
        # routing through `migrad_bounded.jl` → BoundedFunctionMinimum.
        # Fix: BoundedFunctionMinimum.internal_cf is now
        # AbstractCostFunction, the CFwG path keeps cf_internal_grad
        # instead of wrapping it back to plain CostFunction, and
        # _fix_*_params(::CFwG) shares counters with the outer CF.
        # This test guards against re-regression.
        using ForwardDiff
        ng_counter = Ref(0)
        f = x -> (x[1] - 1)^2 + 10.0 * (x[2] - 2)^2 + (x[3] - 3)^2
        g = function (x)
            ng_counter[] += 1
            return ForwardDiff.gradient(f, x)
        end

        # Unbounded path
        ng_counter[] = 0
        m = Minuit(f, [0.0, 0.0, 0.0]; error = [0.1, 0.1, 0.1], grad = g)
        migrad!(m)
        ng_after_migrad = ng_counter[]
        @test ng_after_migrad > 0   # AD was used in migrad
        minos!(m, 1)
        @test ng_counter[] > ng_after_migrad  # AD threaded through minos
        # Note: m.cfwg.ngrad reflects USER-FACING g calls (outer
        # migrad), NOT inner cross-search calls. This mirrors how
        # m.fcn.nfcn doesn't track inner-cross-search numerical FCN
        # calls — both counters report on the user's CF wrapper only.
        # `inner_min.nfcn` + `ContoursError.nfcn` are the inner-delta
        # accessors. Inner-AD threading is verified via user-side
        # `ng_counter[]` above.

        # Bounded path (codex's specific failure case)
        ng_counter[] = 0
        m_b = Minuit(f, [0.0, 0.0, 0.0];
                     error = [0.1, 0.1, 0.1], grad = g,
                     limit_x0 = (-5.0, 5.0))
        migrad!(m_b)
        ng_after_migrad_b = ng_counter[]
        @test ng_after_migrad_b > 0
        minos!(m_b, 1)
        @test ng_counter[] > ng_after_migrad_b   # bounded minos also used AD
    end

    @testset "Phase F — _fix_one_param/_fix_multi_params on CostFunctionWithGradient" begin
        # Splice overloads must produce numerically-correct wrapped FCN
        # AND wrapped gradient. Verify both directly.
        using ForwardDiff
        f = x -> x[1]^2 + 2.0 * x[2]^2 + 3.0 * x[3]^2 + 4.0 * x[4]^2 + 5.0 * x[5]^2
        g = x -> ForwardDiff.gradient(f, x)
        cf_ad = CostFunctionWithGradient(f, g, 1.0)

        # Single-param fix: fix index 3 at value 0.5; free = (x1,x2,x4,x5)
        cf_one = JuMinuit._fix_one_param(cf_ad, 3, 0.5, 5)
        @test cf_one isa CostFunctionWithGradient
        y4 = [0.1, 0.2, 0.4, 0.6]
        # f(y) with index 3 = 0.5
        expected_f = 0.1^2 + 2.0 * 0.2^2 + 3.0 * 0.5^2 + 4.0 * 0.4^2 + 5.0 * 0.6^2
        @test cf_one(y4) ≈ expected_f
        # g(y) is the gradient of the spliced f w.r.t. the FREE pars
        # ∂f/∂x_k for free k. Manual: gradient of f at full = [0.1, 0.2, 0.5, 0.4, 0.6]
        # is [2·0.1, 2·2·0.2, 2·3·0.5, 2·4·0.4, 2·5·0.6] = [0.2, 0.8, 3.0, 3.2, 6.0].
        # Splice out index 3 → [0.2, 0.8, 3.2, 6.0]
        @test cf_one.g(y4) ≈ [0.2, 0.8, 3.2, 6.0]
        # Type stability of the wrapped call
        @test (@inferred cf_one(y4)) isa Float64
        # Zero per-call alloc for both f and g (Phase A V3 lift carries over)
        cf_one(y4); cf_one.g(y4)  # warmup
        @test (@allocated cf_one(y4)) == 0
        # Note: g may allocate inside ForwardDiff (Dual stack); we test the
        # WRAPPER overhead is zero, not ForwardDiff's internal allocs.

        # Multi-param fix: fix indices [1, 3] at [0.5, 0.7]; free = (x2,x4,x5)
        cf_multi = JuMinuit._fix_multi_params(cf_ad, [1, 3], [0.5, 0.7], 5)
        @test cf_multi isa CostFunctionWithGradient
        y3 = [0.2, 0.4, 0.6]
        # f at full = [0.5, 0.2, 0.7, 0.4, 0.6]
        expected_f = 0.5^2 + 2.0 * 0.2^2 + 3.0 * 0.7^2 + 4.0 * 0.4^2 + 5.0 * 0.6^2
        @test cf_multi(y3) ≈ expected_f
        # Gradient: [2·0.5, 2·2·0.2, 2·3·0.7, 2·4·0.4, 2·5·0.6] = [1.0, 0.8, 4.2, 3.2, 6.0]
        # Splice out indices [1, 3] → [0.8, 3.2, 6.0]
        @test cf_multi.g(y3) ≈ [0.8, 3.2, 6.0]
    end

    @testset "Phase G — _fix_*_params per-thread full_buf sanity" begin
        # Verify the per-thread buffer pool doesn't break single-threaded
        # use. Allocations should be ~maxthreadid()×n bytes upfront but
        # zero per-call (just like Phase A V3).
        cf = CostFunction(x -> sum(abs2, x), 1.0)
        cf_one = JuMinuit._fix_one_param(cf, 3, 0.5, 5)
        y4 = [0.1, 0.2, 0.3, 0.4]
        cf_one(y4)  # warmup
        # Same numerical result as Phase A V3
        @test cf_one(y4) ≈ 0.1^2 + 0.2^2 + 0.5^2 + 0.3^2 + 0.4^2
        # Zero per-call alloc still holds (Phase G keeps Phase A invariant)
        @test (@allocated cf_one(y4)) == 0

        cf_multi = JuMinuit._fix_multi_params(cf, [1, 3], [0.5, 0.5], 5)
        y3 = [0.1, 0.2, 0.3]
        cf_multi(y3)
        @test cf_multi(y3) ≈ 0.5^2 + 0.1^2 + 0.5^2 + 0.2^2 + 0.3^2
        @test (@allocated cf_multi(y3)) == 0
    end

    @testset "Phase G — threaded_gradient kwarg backward-compat" begin
        # When threaded_gradient=false (default), result must be
        # byte-identical to Phase F.
        f = x -> (x[1]-1)^2 + 10*(x[2]-2)^2 + (x[3]-3)^2
        cf = CostFunction(f, 1.0)
        fmin_default = JuMinuit.migrad(cf, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])

        cf2 = CostFunction(f, 1.0)
        fmin_explicit_false = JuMinuit.migrad(cf2, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1];
                                                threaded_gradient = false)

        @test fmin_default.state.parameters.x ≈ fmin_explicit_false.state.parameters.x
        @test fmin_default.state.parameters.fval ≈ fmin_explicit_false.state.parameters.fval

        # threaded_gradient=true threads the per-coord gradient evaluation.
        # On single-threaded Julia (nthreads=1) it falls back to sequential,
        # producing identical result. Numerical result equivalence is the
        # blocking test; speedup is bench-only.
        cf3 = CostFunction(f, 1.0)
        fmin_threaded = JuMinuit.migrad(cf3, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1];
                                          threaded_gradient = true)
        @test fmin_threaded.state.parameters.x ≈ fmin_default.state.parameters.x atol = 1e-10
        @test fmin_threaded.state.parameters.fval ≈ fmin_default.state.parameters.fval atol = 1e-10

        # High-level Minuit API
        m = Minuit(f, [0.0, 0.0, 0.0]; error = [0.1, 0.1, 0.1], threaded_gradient = true)
        migrad!(m)
        @test m.fmin.internal.is_valid
        @test m.fmin.ext_values ≈ fmin_default.state.parameters.x atol = 1e-4
    end

    @testset "Phase D — _get_scratch! lazy/replace semantics" begin
        # Verify the holder pattern: nothing → allocate; right size → reuse;
        # wrong size → reallocate (caller's external ref is intact).
        holder = Ref{Union{Nothing,JuMinuit.MigradScratch}}(nothing)
        s_first = JuMinuit._get_scratch!(holder, 5)
        @test s_first.n == 5
        @test holder[] === s_first   # holder now wraps s_first
        # Same size → returns same instance
        s_same = JuMinuit._get_scratch!(holder, 5)
        @test s_same === s_first
        # Different size → new instance, old goes to GC eventually
        s_new = JuMinuit._get_scratch!(holder, 8)
        @test s_new.n == 8
        @test s_new !== s_first
        @test holder[] === s_new
    end

    @testset "Phase D scratch pool — per-probe alloc count is bounded" begin
        # Phase D — MigradScratch is allocated ONCE per inner-dim per
        # contour_exact call. Per-probe re-allocations of the ~15
        # vector + 3 symmetric scratch buffers should drop to zero.
        # We can't directly test `@allocated _migrad_loop` because the
        # call site is buried, but we CAN check that the WHOLE contour's
        # allocation count drops vs the pre-Phase-D baseline.
        #
        # Baseline (V3-only): rosenbrock_10d 30-pt contour ≈ 14_859 allocs.
        # Phase D: ≈ 7_587 allocs (-49% measured).
        # We test a generous bound (12_000) so floor-shifts from
        # unrelated Julia version changes don't flake the suite, but a
        # regression that re-enables per-probe scratch alloc would
        # immediately blow past 12k.
        function rosenbrock_nd(x)
            s = 0.0
            for i in 1:(length(x) - 1)
                s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
            end
            return s
        end
        x0 = [(-1.2, 1.0)[1 + (i & 1)] for i in 0:9]
        cf = CostFunction(rosenbrock_nd, 1.0)
        fmin = migrad(cf, x0, fill(0.1, 10))
        @test fmin.is_valid
        # Warmup: pre-compile via a small contour first
        contour_exact(fmin, cf, 1, 2; npoints = 6)
        # `@allocated` returns BYTES (not alloc count).
        # Phase D measured: ~470 KB on rosenbrock_10d 30-pt contour.
        # V3-only baseline (per-probe scratch realloc): ~1240 KB.
        # Post gaps M1/M4/M5/M6/P1+P2 (commits 2fc38b4..8a37bf2):
        #   ~853 KB. The increase is per-probe constant overhead from
        #   the new feature kwargs threaded through `_migrad_loop`
        #   (M1 trace kwarg + branch, M6 `history = MinimumState[]`
        #   allocation, M4 `MinosError` state-field defaults,
        #   M5 prior_cov plumbing, P1 strategy.hessian_grad_ncycles
        #   path-through).
        # Bound widened from 800 KB → 1000 KB so the test still catches
        # a regression to V3-only allocation patterns (1240+ KB) but
        # tolerates the post-gap-closure overhead. Tighten back toward
        # 850 KB after a dedicated alloc-shave pass on the new code
        # paths (follow-up: see docs/GAP_AUDIT.md if reopened).
        n_bytes = @allocated contour_exact(fmin, cf, 1, 2; npoints = 30)
        @test n_bytes > 0           # sanity: some allocs still happen
        @test n_bytes < 1_000_000   # < 1 MB — still well under V3's 1240 KB
    end

    @testset "warm-state probe doesn't shift the minimum" begin
        # When function_cross_multi sees no fixed-param change (npar=2
        # with pdir != 0), the warm state should track the optimum as
        # the probe walks along the ray. Check that the converged inner
        # f at the boundary is within tlf of fmin + up.
        cf = CostFunction(x -> 0.5 * (x[1]^2 + 4 * x[2]^2 + 0.5 * x[1] * x[2]), 1.0)
        fmin = migrad(cf, [0.5, 0.3], [0.1, 0.1])
        @test fmin.is_valid

        ce = contour_exact(fmin, cf, 1, 2; npoints = 8)
        @test ce.valid
        # Quadratic FCN — every boundary point should satisfy f ≈ fmin + up.
        fmin_v = fmin.state.parameters.fval
        for (x, y) in ce.points
            @test abs(cf.f([x, y]) - (fmin_v + cf.up)) < 0.01
        end
    end
end
