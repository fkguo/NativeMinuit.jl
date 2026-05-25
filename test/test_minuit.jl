# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "minuit.jl — iminuit-style Minuit wrapper" begin

    @testset "Constructor + property access (no migrad)" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                    names = ["x", "y"], errors = [0.1, 0.2])
        @test m isa Minuit
        @test m.ndim == 2
        @test m.npar == 2
        @test m.values == [1.0, 2.0]  # initial
        @test m.errors == [0.1, 0.2]  # initial
        @test isnan(m.fval)
        @test isnan(m.edm)
        @test m.nfcn == 0
        @test !m.valid
        @test m.covariance === nothing
    end

    @testset "migrad! workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m)
        @test m.valid
        @test m.fval < 1e-8
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
        @test m.covariance isa Matrix{Float64}
        @test size(m.covariance) == (2, 2)
    end

    @testset "Bounded + fixed parameters via Minuit" begin
        m = Minuit(x -> (x[1] - 0.5)^2 + (x[2] - 3.0)^2, [0.3, 5.0];
                    names = ["a", "b"], errors = [0.1, 0.1],
                    limits = [(0.0, 1.0), nothing],
                    fixed = [false, true])
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 0.5 atol = 0.01
        @test m.values[2] == 5.0  # fixed bit-exact
        @test m.errors[2] == 0.0   # fixed → no error
    end

    @testset "minos! workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"])
        migrad!(m)
        minos!(m, 1)
        @test haskey(m.minos_errors, 1)
        e = m.minos_errors[1]
        @test JuMinuit.is_valid(e)
        @test e.upper ≈ 1.0 atol = 0.1
        @test e.lower ≈ -1.0 atol = 0.1

        # By name
        minos!(m, "y")
        @test haskey(m.minos_errors, 2)

        # All free
        m2 = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        migrad!(m2)
        minos!(m2)
        @test length(m2.minos_errors) == 2
    end

    @testset "contour workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"])
        migrad!(m)
        c = contour(m, 1, 2; npoints = 10)
        @test c isa ContoursError
        @test c.valid
        @test length(c.points) == 10

        # By name
        c2 = contour(m, "x", "y"; npoints = 8)
        @test c2.valid
        @test length(c2.points) == 8
    end

    @testset "Pretty print" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                    names = ["x", "y"],
                    limits = [(-5.0, 5.0), nothing])
        # Before migrad
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("not yet minimized", s)
        @test occursin("[-5.0, 5.0]", s)

        # After migrad
        migrad!(m)
        buf2 = IOBuffer()
        show(buf2, MIME"text/plain"(), m)
        s2 = String(take!(buf2))
        @test occursin("valid:", s2)
        @test occursin("fval:", s2)
    end

    @testset "Argument validation" begin
        @test_throws ArgumentError Minuit(x -> 0.0, [1.0, 2.0]; names = ["x"])
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        @test_throws ArgumentError minos!(m, 1)  # no migrad! yet
    end
end
