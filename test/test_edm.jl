# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "edm.jl — Expected Distance to Minimum" begin

    @testset "Analytical 2x2" begin
        # g = (1, 2), V = [[2, 0], [0, 1]] (diagonal)
        # EDM = 0.5 · g' · V · g = 0.5 · (1·2·1 + 2·1·2) = 0.5 · 6 = 3
        grad = FunctionGradient([1.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        err = MinimumError(
            Symmetric(Float64[2.0 0.0; 0.0 1.0], :U), 0.0)

        @test estimate_edm(grad, err) ≈ 3.0 atol = 1e-14

        # In-place variant — same result
        work = zeros(2)
        @test estimate_edm!(work, grad, err) ≈ 3.0 atol = 1e-14
    end

    @testset "Off-diagonal contribution" begin
        # g = (1, 1), V = [[1, 0.5], [0.5, 1]]
        # g'·V·g = 1·1·1 + 1·0.5·1 + 1·0.5·1 + 1·1·1 = 3
        # EDM = 1.5
        grad = FunctionGradient([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        err = MinimumError(
            Symmetric(Float64[1.0 0.5; 0.5 1.0], :U), 0.0)
        @test estimate_edm(grad, err) ≈ 1.5 atol = 1e-14
    end

    @testset "Zero gradient → zero EDM" begin
        n = 4
        grad = FunctionGradient(zeros(n), zeros(n), zeros(n))
        err = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U), 0.0)
        @test estimate_edm(grad, err) == 0.0
    end

    @testset "estimate_edm! is zero-alloc" begin
        n = 10
        grad = FunctionGradient(rand(n), zeros(n), zeros(n))
        err = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U), 0.0)
        work = zeros(n)
        # Warmup
        estimate_edm!(work, grad, err)
        @test (@allocated estimate_edm!(work, grad, err)) == 0
    end

    @testset "Type stability" begin
        n = 3
        grad = FunctionGradient(ones(n), zeros(n), zeros(n))
        err = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U), 0.0)
        work = zeros(n)
        @test (@inferred estimate_edm(grad, err)) isa Float64
        @test (@inferred estimate_edm!(work, grad, err)) isa Float64
    end
end
