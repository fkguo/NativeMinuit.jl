# SPDX-License-Identifier: LGPL-2.1-or-later

using ForwardDiff
using Logging

# Run `thunk` capturing all log records (used by the CheckGradient tests to
# assert presence/absence of the discrepancy warning without being brittle to
# unrelated log messages).
_logs_of(thunk) = (l = Test.TestLogger(); Logging.with_logger(thunk, l); l.logs)
_has_checkgrad_warning(logs) =
    any(r -> r.level == Logging.Warn && occursin("CheckGradient", String(r.message)), logs)

@testset "ad_gradient.jl — analytical-gradient MIGRAD (Phase 2.1 first cut)" begin

    @testset "CostFunctionWithGradient construction" begin
        f = x -> sum(abs2, x)
        g = x -> 2.0 .* x
        cf = CostFunctionWithGradient(f, g, 1.0)
        @test cf.f === f
        @test cf.g === g
        @test cf.up == 1.0
        @test NativeMinuit.ngrad_calls(cf) == 0
        @test ncalls(cf) == 0

        # Calling cf evaluates f and counts
        @test cf([1.0, 2.0]) == 5.0
        @test ncalls(cf) == 1
    end

    @testset "Quad-2D via hand-coded gradient" begin
        # f(x) = (x-1)² + (y-2)²; ∇f = [2(x-1), 2(y-2)]
        cf = CostFunctionWithGradient(
            x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
            x -> [2.0 * (x[1] - 1.0), 2.0 * (x[2] - 2.0)],
        )
        m = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-10
        @test Base.values(m)[1] ≈ 1.0 atol = 1e-4
        @test Base.values(m)[2] ≈ 2.0 atol = 1e-4

        # Gradient call count should be 1 per MIGRAD iteration, much less than
        # the numerical-gradient case (2·n·NCycle per iter = 4-8 per iter for n=2).
        ngrad = NativeMinuit.ngrad_calls(cf)
        @test ngrad > 0
        # FCN call count should also be small (line search + maybe HESSE)
        nfcn_total = ncalls(cf)
        @test nfcn_total > 0
        # Sanity: for a 2D quadratic, total FCN calls should be ≪ what numerical
        # gradient would consume (≥ 30 with central diff per ROADMAP).
        @test nfcn_total < 50
    end

    @testset "Quad-4D via ForwardDiff" begin
        f = x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0])
        cf = CostFunctionWithGradient(f, x -> ForwardDiff.gradient(f, x))
        m = migrad(cf, [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-10
        for (i, target) in enumerate((1.0, 2.0, 3.0, 4.0))
            @test Base.values(m)[i] ≈ target atol = 1e-6
        end
    end

    @testset "Rosenbrock-2D via ForwardDiff" begin
        f = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        cf = CostFunctionWithGradient(f, x -> ForwardDiff.gradient(f, x))
        m = migrad(cf, [-1.2, 1.0], [0.1, 0.1])
        # Strategy(0) cross-impl tolerance from §3.4 lessons
        @test Base.values(m)[1] ≈ 1.0 atol = 1e-2
        @test Base.values(m)[2] ≈ 1.0 atol = 1e-2
        @test fval(m) < 1e-3
    end

    @testset "Gradient dimension mismatch throws" begin
        cf = CostFunctionWithGradient(
            x -> sum(abs2, x),
            x -> [1.0],   # WRONG length (returns 1 element for 2-dim input)
        )
        # The mismatch is detected when analytical_gradient! is called.
        # Best path: catch the DimensionMismatch from the gradient stage.
        @test_throws DimensionMismatch migrad(cf, [1.0, 2.0], [0.1, 0.1])
    end

    @testset "analytical_gradient! direct call" begin
        f = x -> sum(abs2, x)
        cf = CostFunctionWithGradient(f, x -> 2.0 .* x)
        par = MinimumParameters([1.0, 2.0], [0.1, 0.1], f([1.0, 2.0]))
        prev = FunctionGradient(zeros(2), [1.0, 1.0], [1e-3, 1e-3])
        out = FunctionGradient(zeros(2), zeros(2), zeros(2))
        NativeMinuit.analytical_gradient!(out, par, cf, prev)
        @test out.grad ≈ [2.0, 4.0] atol = 1e-12
        # g2 and gstep forwarded from prev
        @test out.g2 == prev.g2
        @test out.gstep == prev.gstep
    end

    @testset "Phase F.2 — CostFunctionAD factory (ForwardDiff extension)" begin
        # ForwardDiff is in [extras] / test target, so loading it here
        # triggers the NativeMinuitForwardDiffExt package extension. The
        # extension defines a concrete `CostFunctionAD(f, up; ...)`
        # method; without ForwardDiff loaded the call would
        # MethodError-pointing-at-stub.
        ext = Base.get_extension(NativeMinuit, :NativeMinuitForwardDiffExt)
        @test ext !== nothing

        # Basic factory usage
        f = x -> (x[1] - 1)^2 + 4 * (x[2] - 2)^2
        cf = CostFunctionAD(f, 1.0)
        @test cf isa CostFunctionWithGradient
        # Gradient matches manual: ∇f = [2(x[1]-1), 8(x[2]-2)]
        @test cf.g([0.0, 0.0]) ≈ [-2.0, -16.0] atol = 1e-12
        @test cf.g([1.0, 2.0]) ≈ [0.0, 0.0] atol = 1e-12

        # End-to-end migrad
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid
        @test fmin.state.parameters.x ≈ [1.0, 2.0] atol = 1e-6

        # chunk_size kwarg path
        cf_chunked = CostFunctionAD(f, 1.0; chunk_size = 2)
        @test cf_chunked.g([0.5, 0.5]) ≈ cf.g([0.5, 0.5])

        # Complex-intermediate FCN (X3872-style): real params,
        # complex amplitude, abs² at the end. ForwardDiff handles via
        # `Complex{Dual}`. This is the "Beyond C++ Minuit2" capability.
        function chi2_complex(par)
            mass, coupling = par
            s = 3.5
            amp = coupling / (s - mass^2 - im * mass * 0.1)
            return abs2(amp)
        end
        cf_c = CostFunctionAD(chi2_complex)
        # Numerical reference (central diff)
        h = 1e-7
        x0 = [1.5, 2.0]
        g_num = [
            (chi2_complex([x0[1]+h, x0[2]]) - chi2_complex([x0[1]-h, x0[2]]))/(2h),
            (chi2_complex([x0[1], x0[2]+h]) - chi2_complex([x0[1], x0[2]-h]))/(2h),
        ]
        @test cf_c.g(x0) ≈ g_num atol = 1e-5

        # Default up=1.0 path
        cf_default = CostFunctionAD(f)
        @test cf_default.up == 1.0
    end

    @testset "CheckGradient discrepancy check (audit §8/§9)" begin
        # C++ MnSeedGenerator.cxx:124-144 — validate the user/AD gradient
        # against a numerical 2-point estimate at the seed; warn (never crash)
        # on disagreement. Default on (C++ FCNGradientBase::CheckGradient()).
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        gcorrect = x -> [2 * (x[1] - 1.0), 2 * (x[2] - 2.0)]
        gwrong = x -> [2 * (x[1] - 1.0) + 5.0, 2 * (x[2] - 2.0)]  # comp-1 off by +5
        x0 = [0.0, 0.0]
        errs = [0.1, 0.1]

        @testset "seed_state warns on a wrong gradient" begin
            cf = CostFunctionWithGradient(f, gwrong)
            logs = _logs_of(() -> NativeMinuit.seed_state(cf, x0, errs))
            @test _has_checkgrad_warning(logs)
        end

        @testset "seed_state is silent on a correct gradient" begin
            cf = CostFunctionWithGradient(f, gcorrect)
            logs = _logs_of(() -> NativeMinuit.seed_state(cf, x0, errs))
            @test !_has_checkgrad_warning(logs)
        end

        @testset "check_gradient=false skips the check (silent + no extra FCN calls)" begin
            cf = CostFunctionWithGradient(f, gwrong; check_gradient = false)
            @test cf.check_gradient == false
            logs = _logs_of(() -> NativeMinuit.seed_state(cf, x0, errs))
            @test !_has_checkgrad_warning(logs)
            # The check is the only seed-time consumer of FCN calls beyond the
            # initial fval; skipping it leaves just that one evaluation.
            @test ncalls(cf) == 1
        end

        @testset "_check_user_gradient return value + diagnostic-only (no state change)" begin
            par = MinimumParameters(x0, errs, f(x0))
            cf_ok = CostFunctionWithGradient(f, gcorrect)
            cf_bad = CostFunctionWithGradient(f, gwrong)
            grad_ok = NativeMinuit.analytical_gradient(par, cf_ok)
            grad_bad = NativeMinuit.analytical_gradient(par, cf_bad)
            @test NativeMinuit._check_user_gradient(par, grad_ok, cf_ok) == true
            local ret
            logs = _logs_of(() -> (ret = NativeMinuit._check_user_gradient(par, grad_bad, cf_bad)))
            @test ret == false
            @test _has_checkgrad_warning(logs)

            # The check must not mutate the supplied gradient (it is diagnostic).
            @test grad_bad.grad == [2 * (x0[1] - 1.0) + 5.0, 2 * (x0[2] - 2.0)]
        end

        @testset "seed state is identical with the check on vs off (diagnostic-only)" begin
            cf_on = CostFunctionWithGradient(f, gcorrect; check_gradient = true)
            cf_off = CostFunctionWithGradient(f, gcorrect; check_gradient = false)
            s_on = NativeMinuit.seed_state(cf_on, x0, errs)
            s_off = NativeMinuit.seed_state(cf_off, x0, errs)
            @test s_on.parameters.x == s_off.parameters.x
            @test s_on.gradient.grad == s_off.gradient.grad
            @test s_on.edm == s_off.edm
            # Only the FCN-call accounting differs (the check costs seed calls).
            @test ncalls(cf_on) > ncalls(cf_off)
        end

        @testset "Minuit(fcn, x0; grad=wrong) warns; correct does not; fit unaffected" begin
            # The audit's end-to-end test: the high-level constructor path
            # (migrad! → migrad_bounded `_wrap_fcn_internal_to_external` → seed).
            m_bad = Minuit(f, x0; grad = gwrong, error = errs)
            logs_bad = _logs_of(() -> migrad!(m_bad))
            @test _has_checkgrad_warning(logs_bad)

            m_ok = Minuit(f, x0; grad = gcorrect, error = errs)
            logs_ok = _logs_of(() -> migrad!(m_ok))
            @test !_has_checkgrad_warning(logs_ok)
            @test m_ok.valid
            @test m_ok.values[1] ≈ 1.0 atol = 1e-5
            @test m_ok.values[2] ≈ 2.0 atol = 1e-5

            # Opt-out silences the warning end-to-end (unbounded path).
            m_off = Minuit(f, x0; grad = gwrong, error = errs, check_gradient = false)
            logs_off = _logs_of(() -> migrad!(m_off))
            @test !_has_checkgrad_warning(logs_off)
        end

        @testset "bounded fit honors the check + the check_gradient=false opt-out" begin
            # Regression: the migrad_bounded `_wrap_fcn_internal_to_external`
            # wrap must forward `check_gradient`. Bounding a parameter routes
            # the seed through the int↔ext wrap; the check runs in internal
            # coordinates and still flags a wrong external gradient.
            m_b_bad = Minuit(f, x0; grad = gwrong, error = errs,
                             limit_x0 = (-10.0, 10.0))
            logs_b_bad = _logs_of(() -> migrad!(m_b_bad))
            @test _has_checkgrad_warning(logs_b_bad)

            m_b_off = Minuit(f, x0; grad = gwrong, error = errs,
                             limit_x0 = (-10.0, 10.0), check_gradient = false)
            logs_b_off = _logs_of(() -> migrad!(m_b_off))
            @test !_has_checkgrad_warning(logs_b_off)
        end
    end
end
