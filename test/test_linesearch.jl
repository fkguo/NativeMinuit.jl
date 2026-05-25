# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "linesearch.jl — MnLineSearch" begin

    @testset "1D quadratic — minimum at slam=1 exactly" begin
        # f(x) = (x - 1)²; minimum at x = 1.
        # Starting at x = 0 with step = [1.0], slam = 1 puts us at the minimum.
        cf = CostFunction(x -> (x[1] - 1.0)^2)
        par = MinimumParameters([0.0], cf([0.0]))  # fval = 1
        step = [1.0]
        # df/dx at x=0 along step = 2(x-1)·1 = -2 at x=0
        gdel = -2.0
        reset_ncalls!(cf)

        pp = line_search(cf, par, step, gdel)
        @test pp.x ≈ 1.0 atol = 1e-6
        @test pp.y ≈ 0.0 atol = 1e-12
        # Should converge in very few calls
        @test ncalls(cf) ≤ 5
    end

    @testset "2D quadratic — minimum in interior" begin
        # f(x, y) = (x - 2)² + (y - 3)²; minimum at (2, 3)
        # Starting at (0, 0); search along (1, 1.5) hits min at slam = 2
        cf = CostFunction(x -> (x[1] - 2.0)^2 + (x[2] - 3.0)^2)
        par = MinimumParameters([0.0, 0.0], cf([0.0, 0.0]))  # fval = 13
        step = [1.0, 1.5]
        # ∇f at origin = (-4, -6); step·∇f = -4 + (-9) = -13
        gdel = -13.0

        pp = line_search(cf, par, step, gdel)
        @test pp.x ≈ 2.0 atol = 1e-3
        @test pp.y ≈ 0.0 atol = 1e-6
    end

    @testset "Rosenbrock — small-step descent" begin
        # f(x, y) = (1 - x)² + 100·(y - x²)²
        # ∂f/∂x = 2(x-1) - 400·x·(y-x²)
        # ∂f/∂y = 200·(y-x²)
        # At (-1.2, 1): ∇f = (-4.4 - 211.2, -88) = (-215.6, -88).
        # Descent step = -∇f scaled small.
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        x0 = [-1.2, 1.0]
        par = MinimumParameters(x0, cf(x0))  # fval = 24.2
        scale = 1e-4
        step = [215.6 * scale, 88.0 * scale]   # = -∇f · scale
        # step · ∇f = -|∇f|² · scale  (descent)
        gdel = -(215.6^2 + 88.0^2) * scale

        pp = line_search(cf, par, step, gdel)
        # Line search must never return a value worse than the start
        # (algorithm tracks fvmin starting from f0).
        @test pp.y ≤ par.fval
        # Descent direction should yield positive slam
        @test pp.x > 0
    end

    @testset "ParabolaPoint construction" begin
        pp = ParabolaPoint(1.5, 2.5)
        @test pp.x == 1.5
        @test pp.y == 2.5
        @test isbits(pp)
    end

    @testset "Workspace dim mismatch throws" begin
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], cf([1.0, 2.0]))
        wrong_step = [1.0]   # wrong length
        @test_throws DimensionMismatch line_search(cf, par, wrong_step, -1.0)

        right_step = [1.0, 1.0]
        wrong_work = zeros(5)
        @test_throws DimensionMismatch line_search(
            cf, par, right_step, -2.0; work_x = wrong_work)
    end

    @testset "Zero-allocation hot path with preallocated workspace" begin
        # 1D quadratic; preallocate workspace; verify the search itself
        # does no heap allocation (modulo the user FCN's own behavior).
        cf = CostFunction(x -> (x[1] - 1.0)^2)
        par = MinimumParameters([0.0], cf([0.0]))
        step = [1.0]
        work_x = zeros(1)
        # Warmup
        line_search(cf, par, step, -2.0; work_x = work_x)
        # Measure — small enough that any allocation is detectable
        alloc = @allocated line_search(cf, par, step, -2.0; work_x = work_x)
        # Some BLAS-or-broadcast paths allocate a tiny temporary in
        # Julia 1.12; allow up to one cache-line. Tighten if/when we
        # see consistent zero.
        @test alloc ≤ 64
    end
end
