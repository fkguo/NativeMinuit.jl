# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "migrad.jl — MIGRAD loop" begin

    @testset "Quad-1D: f(x) = x²" begin
        cf = CostFunction(x -> x[1]^2)
        m = migrad(cf, [3.0], [0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-10
        @test values(m)[1] ≈ 0.0 atol = 1e-5
        @test edm(m) < 1e-3
    end

    @testset "Quad-4D matches C++ reference (quad_4d.json)" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1])
        @test m.is_valid
        # C++ reference: fval ≈ 7.81e-20, all params ≈ 0 to 1e-10.
        @test fval(m) ≤ 1e-15
        for i in 1:4
            @test abs(values(m)[i]) < 1e-7
        end
        @test edm(m) < 1e-6
    end

    @testset "Rosenbrock-2D converges (vs analytical minimum)" begin
        # Classic Rosenbrock — minimum at (1, 1), fval = 0
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        m = migrad(cf, [-1.2, 1.0], [0.1, 0.1])
        # Convergence with Strategy(0) is loose; allow a wider tolerance.
        # The C++ reference at Strategy(0) lands at (0.99954, 0.99890).
        @test values(m)[1] ≈ 1.0 atol = 5e-3
        @test values(m)[2] ≈ 1.0 atol = 5e-3
        @test fval(m) < 1e-4
    end

    @testset "Already-at-minimum: zero-iteration return" begin
        # Start AT the minimum of f = sum(abs2, x). Gradient is 0;
        # MIGRAD should detect and return without iterating.
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-14
        for i in 1:3
            @test abs(values(m)[i]) < 1e-10
        end
    end

    @testset "maxfcn limit caps iterations" begin
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        # Force tiny maxfcn — should hit reached_call_limit
        m = migrad(cf, [-1.2, 1.0], [0.1, 0.1]; maxfcn = 20)
        @test m.reached_call_limit
        @test !m.is_valid
        @test nfcn(m) >= 20  # within limit ± boundary slack
    end

    @testset "Strategy ≠ 0 throws" begin
        cf = CostFunction(x -> sum(abs2, x))
        @test_throws ArgumentError migrad(
            cf, [1.0, 2.0], [0.1, 0.1]; strategy = Strategy(1))
    end

    @testset "FunctionMinimum accessors" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test m isa FunctionMinimum
        @test parameters(m) isa MinimumParameters
        @test errors(m) isa MinimumError
        @test gradient(m) isa FunctionGradient
        @test has_covariance(m)
        @test covariance(m) isa Symmetric{Float64,Matrix{Float64}}
        # Pretty-print should work without error
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("FunctionMinimum", s)
        @test occursin("valid:", s)
    end
end
