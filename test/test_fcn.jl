# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "fcn.jl — CostFunction" begin

    @testset "Construction + defaults" begin
        f = x -> sum(abs2, x)
        cf = CostFunction(f)
        @test cf.f === f
        @test cf.up == 1.0          # default ErrorDef
        @test errordef(cf) == 1.0
        @test ncalls(cf) == 0

        # Explicit up (NLL convention)
        cf_nll = CostFunction(f, 0.5)
        @test errordef(cf_nll) == 0.5

        # Parametric F closure-specialized
        @test typeof(cf).parameters[1] === typeof(f)
        @test typeof(cf).parameters[2] === Float64
    end

    @testset "Call counting" begin
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0, 3.0]
        @test cf(x) == 14.0
        @test ncalls(cf) == 1

        cf(x); cf(x); cf(x)
        @test ncalls(cf) == 4

        reset_ncalls!(cf)
        @test ncalls(cf) == 0

        # reset returns the cf itself for chaining
        @test reset_ncalls!(cf) === cf
    end

    @testset "Return-type enforcement" begin
        # Float64 return — fine
        cf_ok = CostFunction(x -> sum(abs2, x))
        @test cf_ok([1.0, 2.0]) === 5.0

        # Int return is convertible-coercible — should work via ::Float64
        cf_int = CostFunction(x -> 42)
        @test cf_int([0.0]) === 42.0  # convert(Float64, 42) succeeds

        # A user FCN returning something non-convertible would fail at call
        # time; we don't test that here (Julia's MethodError is enough).
    end

    @testset "Type stability + closure specialization" begin
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0, 3.0]
        # The call site must be fully type-stable: a concrete F means the
        # call devirtualizes (ROADMAP §2.3 + Risk #4). Verified here as
        # a Float64 inference from the call operator.
        @test (@inferred cf(x)) === 14.0

        # Calling on different vector types still infers Float64
        @test (@inferred cf(@view x[1:2])) isa Float64
    end

    @testset "Zero-allocation call (after warmup)" begin
        # The Ref{Int} increment + closure call should not heap-allocate
        # when the user FCN itself doesn't. This is a soft canary, not a
        # contract gate (Phase 0 §3.4 gate measures the full MIGRAD iter).
        # NB: Julia 1.10's optimizer leaves a small (~16-byte) closure
        # allocation that 1.12+ elides; marked broken on 1.10 so CI passes
        # while the 1.12 strictness is preserved.
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0, 3.0]
        cf(x); cf(x)  # warmup
        if VERSION >= v"1.12"
            @test (@allocated cf(x)) == 0
        else
            @test_broken (@allocated cf(x)) == 0
        end
    end
end
