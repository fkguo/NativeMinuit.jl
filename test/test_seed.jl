# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "seed.jl — MnSeedGenerator (Phase 0 numerical)" begin

    @testset "Quadratic seed at the minimum" begin
        # f(x) = Σ xᵢ². Minimum at origin, ∇f = 2x, g2 = 2.
        cf = CostFunction(x -> sum(abs2, x))
        x0 = [0.0, 0.0, 0.0]
        errs = [0.1, 0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test is_valid(state.parameters)
        @test has_parameters(state)
        @test state.parameters.fval ≈ 0.0 atol = 1e-14
        # Gradient at origin = 0; g2 = 2; inv_hessian diag = 1/2 = 0.5
        for i in 1:3
            @test state.gradient.grad[i] ≈ 0.0 atol = 1e-6
            @test state.gradient.g2[i] ≈ 2.0 atol = 1e-6
            @test state.error.inv_hessian[i, i] ≈ 0.5 atol = 1e-6
        end
        # EDM ≈ 0 at the minimum
        @test estimate_edm(state.gradient, state.error) < 1e-6
        @test !has_negative_g2(state.gradient)
    end

    @testset "Quadratic seed off-center" begin
        # Same FCN; seed at (1, 2, 3) — gradient is 2x = (2, 4, 6)
        cf = CostFunction(x -> sum(abs2, x))
        x0 = [1.0, 2.0, 3.0]
        errs = [0.1, 0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test state.parameters.fval ≈ 14.0 atol = 1e-12
        for i in 1:3
            @test state.gradient.grad[i] ≈ 2 * x0[i] atol = 1e-6
            @test state.gradient.g2[i] ≈ 2.0 atol = 1e-6
        end
        # EDM = 0.5 · (4+16+36) · 0.5 = 14
        @test estimate_edm(state.gradient, state.error) ≈ 14.0 atol = 1e-5
    end

    @testset "Rosenbrock-2 seed" begin
        # Classic starting point (-1.2, 1) for 2D Rosenbrock
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        x0 = [-1.2, 1.0]
        errs = [0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test state.parameters.fval ≈ 24.2 atol = 1e-12
        # Initial state should have positive-def diagonal Hessian
        for i in 1:2
            @test state.error.inv_hessian[i, i] > 0
        end
        @test is_valid(state.error)
    end

    @testset "Strategy ≠ 0 throws" begin
        cf = CostFunction(x -> sum(abs2, x))
        @test_throws ArgumentError seed_state(cf, [1.0, 2.0], [0.1, 0.1], Strategy(1))
        @test_throws ArgumentError seed_state(cf, [1.0, 2.0], [0.1, 0.1], Strategy(2))
    end

    @testset "Dimension mismatch throws" begin
        cf = CostFunction(x -> sum(abs2, x))
        @test_throws DimensionMismatch seed_state(cf, [1.0, 2.0], [0.1])
    end

    @testset "Type stability" begin
        cf = CostFunction(x -> sum(abs2, x))
        # @inferred-clean on the no-negative-g2 path (Quad FCN, g2 = 2 > 0).
        # parallel-review #2 D4 — without this, a Symbol-typed return path
        # could sneak in and be invisible to the non-inferred test.
        @test (@inferred seed_state(cf, [1.0, 2.0], [0.1, 0.1])) isa MinimumState
    end
end
