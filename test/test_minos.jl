# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "minos.jl + function_cross.jl" begin

    @testset "MinosError struct" begin
        # min_par_value field stores the parameter value at the minimum,
        # mirroring C++ MinosError::Min() (parallel-review #4 B2 fix).
        e = MinosError(1, 1.5, 0.5, -0.5, true, true, false, false, false, false, 100)
        @test e.par_idx == 1
        @test e.min_par_value == 1.5
        @test e.upper == 0.5
        @test e.lower == -0.5
        @test JuMinuit.is_valid(e)
    end

    @testset "Symmetric quadratic — MINOS ≈ Hesse" begin
        # f(x, y) = (x - 1)² + (y - 2)². Minimum at (1, 2), fval = 0.
        # Hessian is 2·I, so V = 0.5·I, errors = sqrt(2·1·0.5) = 1.0.
        # MINOS should give upper = -lower ≈ 1.0 for each parameter.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid

        e1 = minos(fmin, cf, 1)
        @test JuMinuit.is_valid(e1)
        # Symmetric → upper ≈ -lower
        @test e1.upper ≈ 1.0 atol = 0.1
        @test e1.lower ≈ -1.0 atol = 0.1

        e2 = minos(fmin, cf, 2)
        @test JuMinuit.is_valid(e2)
        @test e2.upper ≈ 1.0 atol = 0.1
        @test e2.lower ≈ -1.0 atol = 0.1
    end

    @testset "All-parameters convenience" begin
        cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0]))
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        errs = minos(fmin, cf)
        @test length(errs) == 2
        @test all(JuMinuit.is_valid, errs)
    end

    @testset "Argument validation" begin
        cf = CostFunction(x -> sum(abs2, x))
        fmin = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test_throws ArgumentError minos(fmin, cf, 0)
        @test_throws ArgumentError minos(fmin, cf, 3)

        # n=1 should throw (cannot fix the only parameter)
        cf1 = CostFunction(x -> x[1]^2)
        fmin1 = migrad(cf1, [1.0], [0.1])
        @test_throws ArgumentError minos(fmin1, cf1, 1)
    end

    @testset "function_cross — direct call" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        # Upper direction along param 1
        cr_up = JuMinuit.function_cross(fmin, cf, 1, +1.0)
        @test cr_up.valid
        @test cr_up.aopt > 0.5
        # Lower direction along param 1
        cr_lo = JuMinuit.function_cross(fmin, cf, 1, -1.0)
        @test cr_lo.valid
        @test cr_lo.aopt > 0.5  # aopt is the magnitude regardless of sign
    end
end
