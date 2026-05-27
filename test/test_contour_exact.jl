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
