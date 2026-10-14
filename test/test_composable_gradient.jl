# SPDX-License-Identifier: LGPL-2.1-or-later
# Exact controls: a capped first probe and isolated vectors across yielding calls.
using Base.Threads

@testset "bounded gradient step controls" begin
    flags = Bool[false, true, true]
    par = MinimumParameters([1.0, 2.0, 3.0], fill(100.0, 3), 14.0)
    cf = CostFunction(x -> sum(abs2, x))
    init = NativeMinuit.initial_gradient(par, par.dirin, cf; has_limits = flags)
    @test init.gstep == [10.0, 0.5, 0.5]
    @test init.g2 == fill(0.0002, 3)
    @test_throws DimensionMismatch NativeMinuit.initial_gradient(par, par.dirin, cf; has_limits = Bool[true])
    @test_throws DimensionMismatch numerical_gradient(par, init, cf, Strategy(0); has_limits = Bool[true])

    # With exact previous derivatives the first cycle converges immediately.
    # gstep=10 makes the unclamped first probe large, so this is diagnostic.
    prev = FunctionGradient(2 .* par.x, fill(1e-12, 3), fill(10.0, 3))
    for threaded in (false, true)
        out = FunctionGradient(zeros(3), zeros(3), zeros(3))
        numerical_gradient!(out, zeros(3), par, prev, cf, Strategy(0);
                            threaded, has_limits = flags)
        @test out.gstep[2:3] == [0.5, 0.5]
        @test out.grad[2:3] == [4.0, 6.0]
        @test out.g2[2:3] == [2.0, 2.0]
    end
    # Both numerical and analytic pre-built seed routes reject invalid flags,
    # including early returns that do not need another derivative evaluation.
    for cost in (CostFunction(x -> sum(abs2, x)),
                 CostFunctionWithGradient(x -> sum(abs2, x), x -> 2 .* x))
        seed = seed_state(cost, [1., 2.], [.1, .1])
        @test_throws DimensionMismatch migrad(cost, seed; has_limits=[true])
        @test_throws DimensionMismatch warm_restart_state(seed, cost; has_limits=[true])
    end
    params = Parameters([
        MinuitParameter("fixed", 2.0, 100.0; lower=0.0, fixed=true),
        MinuitParameter("free", 1.0, 100.0),
        MinuitParameter("lower", 2.0, 100.0; lower=0.0),
        MinuitParameter("upper", 2.0, 100.0; upper=4.0),
        MinuitParameter("both", 2.0, 100.0; lower=0.0, upper=4.0),
    ])
    @test params.ext_of_int == [2, 3, 4, 5]
    @test NativeMinuit._has_limits_internal(params) == [false, true, true, true]
end

@testset "invocation-owned transform and splice scratch" begin
    # @async tasks on the same thread all yield before reading. A thread-indexed
    # pool deterministically replaces earlier inputs with the last input here.
    f(x) = (yield(); sum(i * x[i]^2 for i in eachindex(x)))
    g(x) = (yield(); [2i * x[i] for i in eachindex(x)])
    pars = Parameters([MinuitParameter("a", 1., .1; lower=0.),
                       MinuitParameter("b", 2., .1),
                       MinuitParameter("c", 3., .1; fixed=true)])
    inputs = [[Float64(i), Float64(i+1)] for i in 1:8]
    for cf in (CostFunction(f), CostFunctionWithGradient(f, g))
        wrapped = NativeMinuit._wrap_fcn_internal_to_external(cf, pars)
        expected = [f(NativeMinuit.int_to_ext_vector(pars, x)) for x in inputs]
        tasks = [@async wrapped.f(x) for x in inputs]
        @test fetch.(tasks) == expected
        if cf isa CostFunctionWithGradient
            expected_g = [wrapped.g(x) for x in inputs]
            tasks_g = [@async wrapped.g(x) for x in inputs]
            @test fetch.(tasks_g) == expected_g
        end
        one = NativeMinuit._fix_one_param(cf, 2, 2.0, 3)
        expected_one = [f([x[1], 2., x[2]]) for x in inputs]
        @test fetch.([@async one.f(x) for x in inputs]) == expected_one
        multi = NativeMinuit._fix_multi_params(cf, [2], [2.0], 3)
        @test fetch.([@async multi.f(x) for x in inputs]) == expected_one
        if cf isa CostFunctionWithGradient
            expected_g = [g([x[1], 2., x[2]])[[1,3]] for x in inputs]
            @test fetch.([@async one.g(x) for x in inputs]) == expected_g
            @test fetch.([@async multi.g(x) for x in inputs]) == expected_g
        end
    end
    # A recursive evaluation on the creating task must not overwrite the
    # input of the outer call, and an exception must release its lease.
    nested = Ref{Any}(nothing)
    depth = Ref(0)
    recursive_f = x -> begin
        if depth[] == 0
            depth[] = 1
            try
                nested[].f([10., 11.])
            finally
                depth[] = 0
            end
        end
        sum(abs2, x)
    end
    nested[] = NativeMinuit._fix_one_param(CostFunction(recursive_f), 2, 2., 3)
    @test nested[].f([1., 3.]) == 14.
    fail = Ref(true)
    throwing = NativeMinuit._fix_one_param(CostFunction(x -> begin
        fail[] && error("objective exception")
        sum(abs2, x)
    end), 2, 2., 3)
    @test_throws ErrorException throwing.f([1., 3.])
    fail[] = false
    @test throwing.f([1., 3.]) == 14.
end

@testset "composable gradient scheduling and exact counters" begin
    function witness(threaded; nonfinite=false)
        truth = Threads.Atomic{Int}(0)
        bad = Threads.Atomic{Int}(0)
        f = x -> begin
            Threads.atomic_add!(truth, 1)
            yield()
            val = nonfinite ? Inf : sum(abs2, x)
            isfinite(val) || Threads.atomic_add!(bad, 1)
            val
        end
        cf = CostFunction(f)
        par = MinimumParameters(collect(1.:8.), fill(.1, 8), 204.)
        prev = NativeMinuit.initial_gradient(par, par.dirin, cf)
        out = numerical_gradient(par, prev, cf, Strategy(0))
        NativeMinuit.reset_ncalls!(cf)
        cf.n_nonfinite[] = 0
        truth[] = 0; bad[] = 0
        numerical_gradient!(out, zeros(8), par, prev, cf, Strategy(0); threaded)
        @test cf.nfcn[] == truth[]
        @test cf.n_nonfinite[] == bad[]
        return out
    end
    serial = witness(false)
    worker = fetch(Threads.@spawn witness(true))
    @test worker.grad == serial.grad
    @test worker.g2 == serial.g2
    @test worker.gstep == serial.gstep
    fetch(Threads.@spawn witness(true; nonfinite=true))
    nested = Vector{FunctionGradient}(undef, 4)
    Threads.@threads :static for k in 1:4
        nested[k] = witness(true)
    end
    @test all(g -> g.grad == serial.grad && g.g2 == serial.g2 && g.gstep == serial.gstep, nested)
end

@testset "throwing objectives preserve attempted call counts" begin
    for threaded in (false, true)
        truth = Threads.Atomic{Int}(0)
        bad = Threads.Atomic{Int}(0)
        cf = CostFunction(x -> begin
            Threads.atomic_add!(truth, 1)
            yield()
            x[1] > 1.0 && error("objective exception")
            Threads.atomic_add!(bad, 1)
            Inf
        end)
        par = MinimumParameters(ones(8), fill(.1, 8), 8.)
        prev = NativeMinuit.initial_gradient(par, par.dirin, cf)
        out = FunctionGradient(zeros(8), zeros(8), zeros(8))
        @test_throws Exception numerical_gradient!(out, zeros(8), par, prev, cf,
                                                   Strategy(0); threaded)
        @test cf.nfcn[] == truth[]
        @test cf.nfcn[] > 0
        @test cf.n_nonfinite[] == bad[]
    end
end

@testset "failed mode refinements preserve exact attempted calls" begin
    calls = Threads.Atomic{Int}(0)
    fail = Ref(false)
    f = x -> begin
        Threads.atomic_add!(calls, 1)
        fail[] && error("refinement objective exception")
        sum(abs2, x)
    end
    m = Minuit(f, [1.,2.]; errors=[.1,.1]); migrad!(m)
    X = [1. 2.; 1.001 2.; -1. -2.; -1.001 -2.]
    fail[] = true
    for parallel in (false, true)
        before = calls[]
        modes = @test_logs (:warn, r"per-mode re-fit failed") (:warn, r"per-mode re-fit failed") match_mode=:all find_solution_modes(
            X, m; fvals=:none, whiten=:errors, threshold=.01, parallel,
            refine=true, refine_iterate=1, refine_maxfcn=100)
        @test length(modes) == 2
        @test all(mode -> !mode.refined && !mode.refined_valid, modes)
        @test sum(mode.refined_nfcn for mode in modes) == calls[] - before
        @test all(mode -> mode.refined_nfcn == 1, modes)
    end
end
