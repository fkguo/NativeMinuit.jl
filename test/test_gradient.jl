# SPDX-License-Identifier: LGPL-2.1-or-later

# Analytical-oracle tests for gradient.jl. Cross-checks against C++
# come later when tools/cpp_trace_harness.cxx is up (ROADMAP §3.2).

@testset "gradient.jl" begin

    # ─────────────────────────────────────────────────────────
    @testset "initial_gradient! on quadratic" begin
        # f(x) = Σ xᵢ²; analytical ∇f(1) = (2, 2, ..., 2); g2 = 2
        cf = CostFunction(x -> sum(abs2, x))
        x = ones(3)
        errs = [1.0, 1.0, 1.0]
        par = MinimumParameters(x, cf(x))  # fval = 3
        reset_ncalls!(cf)  # we don't want the construct call counted

        out = FunctionGradient(zeros(3), zeros(3), zeros(3))
        initial_gradient!(out, par, errs, cf.up)
        @test ncalls(cf) == 0  # initial_gradient! never calls FCN

        # Expected at x=1, werr=1: dirin=1, g2 = 2*up/1 = 2,
        # gstep = max(gsmin, 0.1) = 0.1, grd = 2*1 = 2.
        for i in 1:3
            @test out.grad[i] ≈ 2.0 atol = 1e-14
            @test out.g2[i] ≈ 2.0 atol = 1e-14
            @test out.gstep[i] ≈ 0.1 atol = 1e-14
        end

        # NLL case: up = 0.5 halves g2 and grd
        cf_nll = CostFunction(x -> sum(abs2, x), 0.5)
        initial_gradient!(out, par, errs, cf_nll.up)
        for i in 1:3
            @test out.grad[i] ≈ 1.0 atol = 1e-14
            @test out.g2[i] ≈ 1.0 atol = 1e-14
        end
    end

    @testset "initial_gradient dimension checks" begin
        par = MinimumParameters([1.0, 2.0], 5.0)
        out_wrong = FunctionGradient(zeros(3), zeros(3), zeros(3))
        @test_throws DimensionMismatch initial_gradient!(out_wrong, par, [1.0, 1.0], 1.0)
        out_ok = FunctionGradient(zeros(2), zeros(2), zeros(2))
        @test_throws DimensionMismatch initial_gradient!(out_ok, par, [1.0], 1.0)
    end

    # ─────────────────────────────────────────────────────────
    @testset "numerical_gradient on quadratic vs analytical ∇f" begin
        # f(x) = Σ xᵢ²; analytical ∇f(x) = 2x; g2 = 2 uniformly.
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0, 3.0]
        errs = [0.1, 0.1, 0.1]
        par = MinimumParameters(x, errs, cf(x))  # fval = 14
        reset_ncalls!(cf)

        # Cold-start convenience overload
        g = numerical_gradient(par, cf, Strategy(0))
        for i in 1:3
            @test g.grad[i] ≈ 2 * x[i] atol = 1e-7    # central diff at small step
            @test g.g2[i] ≈ 2.0 atol = 1e-6
        end

        # NFcn upper bound: 2 calls per parameter per cycle, ncycle=2 for L0
        # — at worst 2·3·2 = 12; tolerance breaks usually exit earlier.
        @test ncalls(cf) ≤ 2 * length(x) * Strategy(0).grad_ncycles

        # Strategy 1 should not change the answer materially
        reset_ncalls!(cf)
        g1 = numerical_gradient(par, cf, Strategy(1))
        for i in 1:3
            @test g1.grad[i] ≈ 2 * x[i] atol = 1e-8
            @test g1.g2[i] ≈ 2.0 atol = 1e-7
        end
        # Strategy 1 has more cycles allowed but not always used
        @test ncalls(cf) ≤ 2 * length(x) * Strategy(1).grad_ncycles
    end

    @testset "numerical_gradient on Rosenbrock-2 vs analytical" begin
        # f(x,y) = (1-x)² + 100·(y-x²)²
        # ∂f/∂x = -2(1-x) + 100·2·(y-x²)·(-2x) = 2(x-1) - 400·x·(y-x²)
        # ∂f/∂y = 200·(y-x²)
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        x = [0.5, 0.25]
        errs = [0.1, 0.1]
        par = MinimumParameters(x, errs, cf(x))
        reset_ncalls!(cf)

        g = numerical_gradient(par, cf, Strategy(0))

        # Analytical at (0.5, 0.25):
        # y - x² = 0.25 - 0.25 = 0 → ∂f/∂x = 2·(0.5-1) - 0 = -1
        # ∂f/∂y = 200·0 = 0
        # Central-diff truncation is O(step²); with step ~1e-3,
        # error ~2e-6 is expected on Rosenbrock's curvature.
        @test g.grad[1] ≈ -1.0 atol = 1e-5
        @test g.grad[2] ≈ 0.0 atol = 1e-7
    end

    # ─────────────────────────────────────────────────────────
    @testset "numerical_gradient! preserves x_work on exit" begin
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0, 3.0]
        errs = [0.1, 0.1, 0.1]
        par = MinimumParameters(x, errs, cf(x))
        prev = initial_gradient(par, errs, cf)
        x_work = similar(par.x)
        out = FunctionGradient(zeros(3), zeros(3), zeros(3))

        numerical_gradient!(out, x_work, par, prev, cf, Strategy(0))

        # x_work should equal par.x at exit (the algorithm restores)
        @test x_work ≈ par.x atol = 1e-14
    end

    # ─────────────────────────────────────────────────────────
    @testset "Type stability" begin
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], [0.1, 0.1], cf([1.0, 2.0]))
        prev = initial_gradient(par, [0.1, 0.1], cf)
        x_work = similar(par.x)
        out = FunctionGradient(zeros(2), zeros(2), zeros(2))

        @test (@inferred initial_gradient!(out, par, [0.1, 0.1], cf.up)) === out
        @test (@inferred numerical_gradient!(
            out, x_work, par, prev, cf, Strategy(0))) === out
    end

    # ─────────────────────────────────────────────────────────
    @testset "Zero-allocation hot path (after warmup)" begin
        # Hoist literal vectors out of the @allocated expression — vector
        # literals themselves allocate; only the kernel call is measured.
        cf = CostFunction(x -> sum(abs2, x))
        x0 = [1.0, 2.0, 3.0]
        errs = [0.1, 0.1, 0.1]
        par = MinimumParameters(x0, errs, cf(x0))
        prev = initial_gradient(par, errs, cf)
        x_work = similar(par.x)
        out = FunctionGradient(zeros(3), zeros(3), zeros(3))
        strat = Strategy(0)

        # Warmup
        initial_gradient!(out, par, errs, cf.up)
        numerical_gradient!(out, x_work, par, prev, cf, strat)

        # Measure: initial_gradient! is allocation-free.
        @test (@allocated initial_gradient!(out, par, errs, cf.up)) == 0

        # numerical_gradient! allocates only what the user FCN allocates.
        # sum(abs2, x) on a Vector is allocation-free.
        @test (@allocated numerical_gradient!(
            out, x_work, par, prev, cf, strat)) == 0
    end
end
