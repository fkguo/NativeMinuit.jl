# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test
using LinearAlgebra

@testset "eigen_corr.jl — MnEigen + MnGlobalCorrelationCoeff ports" begin

    @testset "eigenvalues — diagonal matrix" begin
        cov = Symmetric([2.0 0.0; 0.0 5.0])
        eig = eigenvalues(cov)
        @test eig ≈ [2.0, 5.0]
    end

    @testset "eigenvalues — known 2x2" begin
        # V = [[3, 1], [1, 3]] → eigs = 2, 4
        cov = Symmetric([3.0 1.0; 1.0 3.0])
        eig = eigenvalues(cov)
        @test eig ≈ [2.0, 4.0]
    end

    @testset "eigenvalues — sorted ascending" begin
        cov = Symmetric([10.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 5.0])
        eig = eigenvalues(cov)
        @test eig ≈ [1.0, 5.0, 10.0]
        @test issorted(eig)
    end

    @testset "eigenvalues from Minuit (ill-conditioned)" begin
        # f = (a+b-5)² + 0.01·(a-b)² — a+b is well-determined, a-b is
        # weakly determined; eigenvalues of cov should span ~×100.
        f = x -> (x[1] + x[2] - 5.0)^2 + 0.01 * (x[1] - x[2])^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        hesse(m)
        eig = eigenvalues(m)
        @test eig !== nothing
        @test length(eig) == 2
        @test eig[1] > 0   # both eigenvalues should be positive
        @test eig[2] / eig[1] > 50.0   # ill-conditioned (×100 expected)
    end

    @testset "eigenvalues returns nothing before migrad" begin
        f = x -> sum(abs2, x)
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        @test eigenvalues(m) === nothing
    end

    @testset "global_cc — diagonal matrix → all zeros" begin
        # No off-diagonal correlation → ρ_i = 0 for all i.
        cov = Symmetric([1.0 0.0; 0.0 2.0])
        cc, valid = global_cc(cov)
        @test valid
        @test cc ≈ [0.0, 0.0]
    end

    @testset "global_cc — strongly correlated 2x2" begin
        # V = [[1, 0.99], [0.99, 1]] — strong correlation
        cov = Symmetric([1.0 0.99; 0.99 1.0])
        cc, valid = global_cc(cov)
        @test valid
        # ρ_i = sqrt(1 - 1/(V_ii · V⁻¹_ii)); both should be near 0.99
        @test cc[1] ≈ 0.99 atol = 0.01
        @test cc[2] ≈ 0.99 atol = 0.01
    end

    @testset "global_cc from Minuit" begin
        # Strong correlation case
        f = x -> (x[1] + x[2] - 5.0)^2 + 0.01 * (x[1] - x[2])^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        hesse(m)
        result = global_cc(m)
        @test result !== nothing
        cc, valid = result
        @test valid
        # Both parameters near-fully determined by the other
        @test all(c -> c > 0.95, cc)
    end

    @testset "global_cc returns nothing before migrad" begin
        f = x -> sum(abs2, x)
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        @test global_cc(m) === nothing
    end

    @testset "eigenvalues — non-square matrix throws" begin
        @test_throws ArgumentError eigenvalues([1.0 2.0 3.0; 4.0 5.0 6.0])
    end

    @testset "global_cc — non-square matrix throws" begin
        @test_throws ArgumentError global_cc([1.0 2.0 3.0; 4.0 5.0 6.0])
    end

end
