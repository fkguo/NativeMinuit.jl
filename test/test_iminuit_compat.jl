# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IMinuit.jl drop-in compatibility tests.
#
# IMinuit.jl is the existing Julia binding to Python iminuit; this
# package (JuMinuit) is the clean-room Julia port that aims to be a
# drop-in replacement. These tests assert that the canonical IMinuit.jl
# usage patterns (from its README + tests) work unchanged.
#
# Reference: ~/.julia/packages/IMinuit/83MUo/{test/runtests.jl,
# src/IMinuit.jl}.

@testset "IMinuit.jl drop-in compatibility" begin

    @testset "Minuit(fcn, start; kwds...) — vector start" begin
        # Canonical IMinuit.jl usage from its README
        f(x) = x[1]^2 + (x[2] - 1)^2 + (x[3] - 2)^4
        m = Minuit(f, [1, 1, 0])
        @test m isa Minuit
        @test n_pars(m.params) == 3

        # No-bang `migrad(m)` mutates in place + returns m
        m2 = migrad(m)
        @test m2 === m
        @test m.is_valid
        @test m.values[2] ≈ 1.0 atol = 1e-3   # second param converges to 1
    end

    @testset "Minuit(fcn; x=…, y=…) — keyword-named params" begin
        # IMinuit.jl style: scalar params via kwargs, `fcn` takes
        # positional scalars
        f1(x, y, z) = x^2 + (y - 1)^2 + (z - 2)^4
        m = Minuit(f1; x = 1.0, y = 1.0, z = 0.0)
        @test m isa Minuit
        @test (m.parameters)::Tuple == ("x", "y", "z")
        migrad(m)
        @test m.is_valid
        @test m.values[2] ≈ 1.0 atol = 1e-3
    end

    @testset "IMinuit-style kwargs (singular) + per-param" begin
        # IMinuit.jl typical: `name = [...]`, `error = 0.1*ones(n)`,
        # `fix_a = true`, `limit_b = (0, 50)`.
        f(x) = (x[1] - 2.0)^2 + (x[2] - 1.0)^2
        m = Minuit(f, [1.0, 0.0]; name = ["a", "b"],
                                    error = 0.1 * ones(2),
                                    fix_a = true,
                                    limit_b = (0, 50))
        @test m.parameters == ("a", "b")
        @test m.fixed == [true, false]
        @test m.limits[2] == (0.0, 50.0)
        @test isnan(m.limits[1][1])  # a unbounded
        migrad(m)
        @test m.values[1] == 1.0          # a stays fixed at initial 1.0
        @test m.values[2] ≈ 1.0 atol = 0.01   # b converges to 1 (bounded sub-fit)
    end

    @testset "Property aliases" begin
        cf_inner = CostFunction(x -> sum(abs2, x))
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0]; error = [0.1, 0.1])
        migrad(m)
        # JuMinuit-native. `m.values`/`m.errors` now return write-back
        # `ParameterView`s (iminuit ValueView parity) — an AbstractVector
        # of Float64 rather than a concrete Vector.
        @test m.values isa AbstractVector{Float64}
        @test m.errors isa AbstractVector{Float64}
        @test m.fval ≈ 0.0 atol = 1e-8
        @test m.valid == true
        # IMinuit.jl-compatible aliases
        @test m.is_valid == m.valid
        @test m.ncalls == m.nfcn
        @test m.parameters == ("x0", "x1")
        @test m.fixed == [false, false]
        @test m.errordef == 1.0
        @test m.up == 1.0
        @test m.accurate isa Bool
        # `merrors` exposes MINOS errors keyed by name
        minos(m)
        @test haskey(m.merrors, "x0")
        @test haskey(m.merrors, "x1")
    end

    @testset "args(m) helper" begin
        m = Minuit(x -> (x[1] - 3.0)^2 + (x[2] - 4.0)^2, [0.0, 0.0])
        migrad(m)
        a = args(m)
        @test a isa Vector{Float64}
        @test a[1] ≈ 3.0 atol = 1e-4
        @test a[2] ≈ 4.0 atol = 1e-4
        @test a == m.values
    end

    @testset "matrix(m) helper" begin
        cf = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(cf, [0.0, 0.0])
        migrad(m)
        # Default: free-block covariance
        V = matrix(m)
        @test V isa Matrix{Float64}
        @test size(V) == (2, 2)
        # Symmetric, positive-diagonal
        @test V[1, 1] > 0 && V[2, 2] > 0
        @test V[1, 2] ≈ V[2, 1] atol = 1e-12
        # Correlation form
        C = matrix(m; correlation = true)
        @test all(isapprox.(diag(C), 1.0; atol = 1e-12))
        # Full (with fixed rows/cols if any)
        Vfull = matrix(m; skip_fixed = false)
        @test size(Vfull) == (2, 2)
    end

    @testset "reset(m) + set_precision(m, p)" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 1.0])
        migrad(m)
        @test m.fmin !== nothing
        reset(m)
        @test m.fmin === nothing
        @test isempty(m.minos_errors)
        # After reset, can re-run migrad
        migrad(m)
        @test m.is_valid

        # set_precision changes the MachinePrecision.eps used internally
        m2 = Minuit(x -> sum(abs2, x), [1.0, 1.0])
        old_eps = m2.prec.eps
        set_precision(m2, 1e-10)
        @test m2.prec.eps == 1e-10
        @test m2.prec.eps != old_eps
    end

    @testset "Copy-from-other-fit constructor" begin
        f(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m1 = Minuit(f, [0.0, 0.0]; name = ["a", "b"])
        migrad(m1)
        # Create a new fit using m1's converged values as the new start
        m2 = Minuit(f, m1)
        @test m2.parameters == ("a", "b")
        @test m2.values ≈ m1.values
        @test m2.fmin === nothing   # fresh, not yet minimized
        # Can override per-param via kwargs
        m3 = Minuit(f, m1; fix_a = true)
        @test m3.fixed == [true, false]
    end

    @testset "errordef alias for up (NLL fits)" begin
        # iminuit's `errordef = 0.5` for NLL is the standard convention
        # JuMinuit's native is `up = 0.5`; both should work
        nll(x) = 0.5 * sum(abs2, x .- [1.0, 2.0])
        m1 = Minuit(nll, [0.0, 0.0]; up = 0.5)
        m2 = Minuit(nll, [0.0, 0.0]; errordef = 0.5)
        @test m1.errordef == 0.5
        @test m2.errordef == 0.5
        @test m1.fcn.up == m2.fcn.up == 0.5
    end
end
