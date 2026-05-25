# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "migrad_bounded.jl — bound-aware MIGRAD" begin

    @testset "Unbounded quadratic via Parameters API" begin
        # f(x, y) = (x-1)² + (y-2)². Minimum at (1, 2), fval = 0.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params = Parameters([
            MinuitParameter("x", 0.0, 0.1),
            MinuitParameter("y", 0.0, 0.1),
        ])
        m = migrad(cf, params)
        @test is_valid(m)
        @test fval(m) ≈ 0.0 atol = 1e-8
        @test m.ext_values[1] ≈ 1.0 atol = 1e-4
        @test m.ext_values[2] ≈ 2.0 atol = 1e-4
        # Errors ≈ 1 for each (V = 0.5·I, σ = √(2·1·0.5) = 1)
        @test m.ext_errors[1] ≈ 1.0 atol = 0.1
        @test m.ext_errors[2] ≈ 1.0 atol = 0.1
    end

    @testset "Bounded parameter — Sin transform" begin
        # f(x) = (x - 0.5)². Minimum at x = 0.5, inside [0, 1].
        # Strategy(0) is quick-and-loose; relax tolerance to match
        # the Phase 0 §3.4 cross-impl Rosenbrock noise level.
        cf = CostFunction(x -> (x[1] - 0.5)^2 + (x[2] - 0.5)^2)
        params = Parameters([
            MinuitParameter("x", 0.3, 0.1; lower = 0.0, upper = 1.0),
            MinuitParameter("y", 0.3, 0.1),  # unbounded
        ])
        m = migrad(cf, params)
        @test is_valid(m)
        @test m.ext_values[1] ≈ 0.5 atol = 0.01
        @test m.ext_values[2] ≈ 0.5 atol = 0.01
        # Bounded x should stay in [0, 1] (transform guarantees)
        @test 0.0 ≤ m.ext_values[1] ≤ 1.0
    end

    @testset "Fixed parameter — kept at initial value" begin
        # f(x, y, z) = (x-1)² + (y-2)² + (z-3)². Fix y=5.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2)
        params = Parameters([
            MinuitParameter("x", 0.0, 0.1),
            MinuitParameter("y", 5.0, 0.1; fixed = true),
            MinuitParameter("z", 0.0, 0.1),
        ])
        m = migrad(cf, params)
        @test is_valid(m)
        @test m.ext_values[1] ≈ 1.0 atol = 1e-4
        @test m.ext_values[2] == 5.0  # fixed: bit-exact
        @test m.ext_values[3] ≈ 3.0 atol = 1e-4
        @test m.ext_errors[2] == 0.0  # fixed has zero error
        # fval at minimum = (1-1)² + (5-2)² + (3-3)² = 9
        @test fval(m) ≈ 9.0 atol = 1e-6
    end

    @testset "External covariance matrix returned" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params = Parameters([
            MinuitParameter("x", 0.0, 0.1),
            MinuitParameter("y", 0.0, 0.1),
        ])
        m = migrad(cf, params)
        cov = ext_covariance(m)
        @test cov !== nothing
        @test size(cov) == (2, 2)
        # For symmetric uncorrelated f, V = 0.5·I → cov = 2·V = I
        @test cov[1, 1] ≈ 1.0 atol = 0.1
        @test cov[2, 2] ≈ 1.0 atol = 0.1
        # Off-diagonal should be ~zero for uncorrelated f
        @test abs(cov[1, 2]) < 0.1
        @test abs(cov[2, 1]) < 0.1
        # Symmetry check: cov[i,j] == cov[j,i] (parallel-review #4 D3
        # blocking — v1 of migrad_bounded read parent(Symmetric{:U})
        # which returned uninitialized zero in the lower triangle,
        # producing an asymmetric matrix).
        @test cov[1, 2] == cov[2, 1]
    end

    @testset "Covariance symmetry on correlated quadratic (D3 regression)" begin
        # f(x, y) = x² + y² + x·y has off-diagonal entries in the
        # Hessian, exposing the upper-vs-lower triangle storage bug
        # (parallel-review #4 D3). The v1 of migrad_bounded.jl read
        # `parent(Symmetric{:U}(...))` which gave 0 in the lower
        # triangle, producing an asymmetric external covariance.
        # The bit-exact magnitude depends on Strategy(0) DFP-approx
        # convergence which is loose for correlated 2D; the strict
        # invariant we MUST hold is symmetry.
        cf = CostFunction(x -> x[1]^2 + x[2]^2 + x[1]*x[2])
        params = Parameters([
            MinuitParameter("x", 1.0, 0.1),
            MinuitParameter("y", 1.0, 0.1),
        ])
        m = migrad(cf, params)
        cov = ext_covariance(m)
        @test cov !== nothing
        # Symmetry MUST hold (the D3 invariant)
        @test cov[1, 2] == cov[2, 1]
        # Off-diagonal must be non-trivially negative (true V[1,2] < 0
        # for this FCN's correlated Hessian); v1 would give 0.
        @test cov[1, 2] < -0.01
    end

    @testset "Argument validation" begin
        cf = CostFunction(x -> sum(abs2, x))
        # All-fixed: should throw
        params_all_fixed = Parameters([
            MinuitParameter("a", 1.0, 0.1; fixed = true),
            MinuitParameter("b", 2.0, 0.1; fixed = true),
        ])
        @test_throws ArgumentError migrad(cf, params_all_fixed)
    end

    @testset "Pretty print" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params = Parameters([
            MinuitParameter("x", 0.0, 0.1; lower = -5.0, upper = 5.0),
            MinuitParameter("y", 0.0, 0.1; fixed = true),
        ])
        m = migrad(cf, params)
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("BoundedFunctionMinimum", s)
        @test occursin("FIXED", s)
        @test occursin("[-5.0, 5.0]", s)
    end
end
