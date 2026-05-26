# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test

@testset "simplex.jl + scan.jl (C++ MnSimplex / MnScan ports)" begin

    @testset "simplex on shifted quadratic" begin
        # 3D shifted quadratic — Nelder-Mead's natural strength.
        f = x -> sum(abs2, x .- [1.0, 2.0, 3.0])
        fm = simplex(f, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test JuMinuit.fval(fm) < 1.0e-5
        @test fm.state.parameters.x ≈ [1.0, 2.0, 3.0] atol = 1e-2
        @test JuMinuit.nfcn(fm) > 0
    end

    @testset "simplex with up=0.5 (NLL)" begin
        # ErrorDef = 0.5 (negative log-likelihood) — final per-param
        # errors should scale as sqrt(up).
        f = x -> 0.5 * sum(abs2, x .- [1.0, 2.0])
        fm = simplex(f, [0.0, 0.0], [0.1, 0.1]; up = 0.5)
        @test JuMinuit.fval(fm) < 1.0e-4
        @test fm.up ≈ 0.5
    end

    @testset "simplex returns FunctionMinimum without covariance" begin
        # Simplex doesn't compute a Hessian; the MinimumError is
        # marked `available=false` so downstream code (m.matrix,
        # eigenvalues, global_cc, minos) sees "no covariance" and
        # behaves accordingly. `hesse_failed` stays FALSE because
        # Hesse was never RUN — "failed" would be misleading
        # (round-2 fix I4).
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        fm = simplex(f, [0.0, 0.0], [0.1, 0.1])
        @test !fm.hesse_failed
        @test !JuMinuit.is_available(fm.state.error)
        @test !JuMinuit.has_covariance(fm)
    end

    @testset "bounded simplex" begin
        # Optimum at (-3, 7) but `a ∈ [0, ∞)` and `b ∈ [0, 5]` force
        # the constrained minimum to (0, 5).
        f = x -> (x[1] + 3)^2 + (x[2] - 7)^2
        params = JuMinuit.Parameters(["a", "b"], [1.0, 1.0], [0.1, 0.1];
                                       limits = [(0.0, NaN), (0.0, 5.0)],
                                       fixed  = [false, false])
        bfm = simplex(f |> JuMinuit.CostFunction, params)
        @test bfm.ext_values[1] ≈ 0.0 atol = 0.1
        @test bfm.ext_values[2] ≈ 5.0 atol = 0.1
    end

    @testset "simplex from Minuit struct" begin
        f = x -> (x[1] - 0.7)^2 + (x[2] - 1.3)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        simplex(m)
        @test m.values ≈ [0.7, 1.3] atol = 1e-2
        @test m.fval < 1.0e-4
        # is_valid should be true for clean simplex convergence
        @test m.fmin !== nothing
    end

    @testset "scan: 1D evaluation" begin
        # Quadratic in 2D, scan along par 1: should see f minimized at x[1]=2.
        f = x -> (x[1] - 2.0)^2 + (x[2] - 3.0)^2
        result = scan(f, [0.0, 0.0], [0.5, 0.5], 1;
                       maxsteps = 11, low = -3.0, high = 7.0)
        # First entry is the central point (x[1]=0, f=4+9=13).
        @test result[1] == (0.0, 13.0)
        # Find the minimum across the grid
        grid = result[2:end]
        fmin_idx = argmin(f for (x, f) in grid)
        x_at_min, f_at_min = grid[fmin_idx]
        @test x_at_min ≈ 2.0 atol = 1.0
        @test f_at_min ≈ 9.0 atol = 1.0   # 0² + 3² = 9 (par 2 at 0)
    end

    @testset "scan: default ±2σ range" begin
        # low=0, high=0 → defaults to value ± 2·errs.
        f = x -> (x[1])^2 + (x[2])^2
        result = scan(f, [1.0, 0.0], [0.5, 0.5], 1; maxsteps = 5)
        # Should span [0, 2].
        grid = result[2:end]
        @test minimum(p[1] for p in grid) ≈ 0.0 atol = 1e-9
        @test maximum(p[1] for p in grid) ≈ 2.0 atol = 1e-9
    end

    @testset "scan: invalid range returns just the central point" begin
        # low > high → empty grid (only central point).
        f = x -> x[1]^2
        result = scan(f, [1.0], [0.1], 1;
                       maxsteps = 5, low = 5.0, high = 0.0)
        @test length(result) == 1
        @test result[1] == (1.0, 1.0)
    end

    @testset "scan: par_idx out of bounds throws" begin
        f = x -> x[1]^2
        @test_throws ArgumentError scan(f, [1.0], [0.1], 5)
        @test_throws ArgumentError scan(f, [1.0], [0.1], 0)
    end

    @testset "scan: maxsteps < 2 throws" begin
        f = x -> x[1]^2
        @test_throws ArgumentError scan(f, [1.0], [0.1], 1; maxsteps = 1)
    end

    @testset "scan: bounded clips against limits" begin
        # par.upper=3 should clip the high end of a scan reaching for ±2σ
        params = JuMinuit.Parameters(["a"], [0.0], [0.5];
                                       limits = [(NaN, 3.0)],
                                       fixed  = [false])
        f = x -> x[1]^2
        cf = JuMinuit.CostFunction(f, 1.0)
        result = scan(cf, params, 1; maxsteps = 5)
        grid = result[2:end]
        @test maximum(p[1] for p in grid) <= 3.0 + 1e-9
    end

    @testset "scan from Minuit struct (Minuit-method)" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        result = scan(m, 1; maxsteps = 5, low = 0.0, high = 2.0)
        @test length(result) == 6   # 1 central + 5 grid
        grid = result[2:end]
        # Minimum should be at x[1]=1.0
        fmin_idx = argmin(p[2] for p in grid)
        @test grid[fmin_idx][1] ≈ 1.0 atol = 0.5
    end

end
