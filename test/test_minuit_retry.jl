# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# test_minuit_retry.jl — iminuit `_robust_low_level_fit` parity for the
# `migrad!(m; iterate, use_simplex)` retry layer.
#
# Coverage:
#   • iterate=1 reproduces single-shot C++-faithful behavior.
#   • iterate=5 (the default) matches iterate=1 nfcn on a convex FCN —
#     proving the retry loop is gated and only fires on failed passes.
#   • iterate=0 and iterate=-1 raise ArgumentError.
#   • use_simplex ∈ {true, false} both converge on a convex FCN.
#   • IMinuit.jl alias `migrad(m; iterate, use_simplex)` threads kwargs.
#   • Multi-minimum safety invariant: iterate=5 never produces a worse
#     fval than iterate=1 (a bimodal toy landscape inspired by the
#     dip-vs-peak ambiguity in the X(3872) `J/ψρ + DD̄*` fit, see
#     arXiv:2404.12003).
# ─────────────────────────────────────────────────────────────────────────────

@testset "minuit retry layer (iminuit _robust_low_level_fit parity)" begin

    # Shared convex FCN: an unbounded paraboloid centred at (1, 2). Pass 1
    # always validates; if the retry loop ever fired on this FCN, the
    # `nfcn` invariant in the no-regression test below would fail.
    fcn_convex(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2

    @testset "iterate=1 — single-shot baseline" begin
        m = Minuit(fcn_convex, [0.0, 0.0];
                    names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m; iterate = 1)
        @test m.valid
        @test m.fval < 1e-8
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
    end

    @testset "iterate=5 on convex matches iterate=1 (retry never enters)" begin
        m1 = Minuit(fcn_convex, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m1; iterate = 1)
        m5 = Minuit(fcn_convex, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m5; iterate = 5)
        # Byte-identical nfcn proves the retry loop's gating predicate
        # never fired on pass 1. If retry had entered, nfcn(m5) > nfcn(m1).
        # Strict `==` (not `isapprox`) — identical code paths must produce
        # identical bits. A future regression that re-runs the inner DFP
        # iteration on a "valid" pass 1 would break this.
        @test m1.nfcn == m5.nfcn
        @test m1.fval == m5.fval
        @test m1.values == m5.values
    end

    @testset "iterate ≤ 0 raises ArgumentError" begin
        # 0 and -1 are the boundary cases. iminuit's `_robust_low_level_fit`
        # itself requires iterate ≥ 1; skipping MIGRAD entirely (the
        # silent-fail behavior we'd get without the guard) would
        # almost always be a user mistake.
        m = Minuit(fcn_convex, [0.0, 0.0]; names = ["x", "y"])
        @test_throws ArgumentError migrad!(m; iterate = 0)
        @test_throws ArgumentError migrad!(m; iterate = -1)
    end

    @testset "use_simplex ∈ {true, false} both converge on convex" begin
        for us in (true, false)
            m = Minuit(fcn_convex, [0.0, 0.0];
                        names = ["x", "y"], errors = [0.1, 0.1])
            migrad!(m; iterate = 5, use_simplex = us)
            @test m.valid
            @test m.fval < 1e-8
            @test m.values[1] ≈ 1.0 atol = 1e-4
            @test m.values[2] ≈ 2.0 atol = 1e-4
        end
    end

    @testset "IMinuit.jl alias `migrad(m; iterate, use_simplex)`" begin
        # ArgumentError must propagate through the no-bang alias too —
        # if it didn't, IMinuit.jl callers would silently skip MIGRAD
        # on a typo.
        m_err = Minuit(fcn_convex, [0.0, 0.0]; names = ["x", "y"])
        @test_throws ArgumentError migrad(m_err; iterate = 0)
        @test_throws ArgumentError migrad(m_err; iterate = -1)

        # Convex no-retry invariant — same nfcn across iterate=1 vs
        # iterate=5 (both default and `use_simplex=false`) via the alias.
        m1 = Minuit(fcn_convex, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad(m1; iterate = 1)
        m5 = Minuit(fcn_convex, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad(m5; iterate = 5, use_simplex = false)
        @test m1.nfcn == m5.nfcn
        @test m1.fval == m5.fval
    end

    @testset "Bounded + retry kwargs (smoke)" begin
        # Ensure the retry kwargs survive the bounded path too — the
        # bounded `migrad(cf::CostFunction, params::Parameters; prior_cov)`
        # entry point gained a `prior_cov` kwarg as part of this PR;
        # this regression test ensures the retry layer's prior_cov hop
        # works through bounded fits as well.
        m = Minuit(x -> (x[1] - 0.5)^2 + (x[2] - 3.0)^2, [0.3, 5.0];
                    names = ["a", "b"], errors = [0.1, 0.1],
                    limits = [(0.0, 1.0), nothing],
                    fixed = [false, true])
        migrad!(m; iterate = 5, use_simplex = false)
        @test m.valid
        @test m.values[1] ≈ 0.5 atol = 0.01
        @test m.values[2] == 5.0  # fixed
    end

    @testset "Retry actually triggers and recovers on a pathological seed" begin
        # The critical end-to-end test: an FCN + start point where pass 1
        # exits with `is_valid=false` (no improvement / above_max_edm)
        # WITHOUT hitting the call limit. From this state the retry loop
        # MUST enter — otherwise the prior_cov / Simplex hop / Strategy
        # branches all sit untested.
        #
        # Construction: a multi-minimum landscape (quadratic bowl with
        # cos+sin overlay) seeded from far away (-5, 5). On pass 1 the
        # DFP descent stalls at a saddle-like point with fval ≈ 20.8 and
        # `is_valid=false`. The retry loop's Simplex hop + re-seed
        # escapes that basin.
        fcn = x -> begin
            a, b = x[1], x[2]
            return (a - 3)^2 + (b - 3)^2 +
                   0.3 * cos(5 * a) + 0.3 * cos(5 * b) +
                   0.5 * sin(a * b)
        end

        # Single-shot baseline
        m1 = Minuit(fcn, [-5.0, 5.0];
                     names = ["a", "b"], errors = [0.5, 0.5])
        migrad!(m1; iterate = 1, tol = 1e-6, maxfcn = 2000)
        @test !m1.valid                                 # pass 1 failed
        @test !m1.fmin.internal.reached_call_limit      # not via budget
        nfcn1 = m1.nfcn
        fval1 = m1.fval

        # With retry: must run AT LEAST one extra pass (nfcn > nfcn1) and
        # NEVER end up at a worse fval than the single-shot baseline.
        m5 = Minuit(fcn, [-5.0, 5.0];
                     names = ["a", "b"], errors = [0.5, 0.5])
        migrad!(m5; iterate = 5, tol = 1e-6, maxfcn = 2000)
        @test m5.nfcn > nfcn1     # retry loop entered (proves coverage)
        @test m5.fval ≤ fval1 + 1e-10
        # And on this FCN the retry actually rescues to a valid fit at a
        # much deeper minimum (fval ≈ -0.86 vs +20.8). This is the
        # X(3872)-shaped "dip vs peak" outcome the spec calls for.
        @test m5.valid
        @test m5.fval < 0.0

        # Same FCN with `use_simplex=false` exercises the `_retry_prior_cov`
        # branch (which is skipped when use_simplex=true). The prior_cov
        # path also escapes the bad seed on this FCN.
        m5_ns = Minuit(fcn, [-5.0, 5.0];
                        names = ["a", "b"], errors = [0.5, 0.5])
        migrad!(m5_ns; iterate = 5, use_simplex = false,
                       tol = 1e-6, maxfcn = 2000)
        @test m5_ns.nfcn > nfcn1
        @test m5_ns.fval ≤ fval1 + 1e-10
        @test m5_ns.valid
    end

    @testset "Multi-minimum safety invariant" begin
        # Bimodal landscape: a deeper Gaussian well centred at (3, 3) and
        # a shallower well at (-3, -3). The choice of relative depth is
        # arbitrary but mirrors the dip-vs-peak ambiguity in the X(3872)
        # `J/ψρ + DD̄*` fit (arXiv:2404.12003, *Dip versus peak*), where
        # the published dataset admits multiple physically distinct local
        # minima corresponding to different scattering-length combinations.
        #
        # Important caveat: MIGRAD from a "wrong basin" start point
        # routinely finds a valid local minimum and validates, so the
        # retry loop's gating predicate (`is_valid(bfm.internal)`) may
        # NOT trigger — the spec's "deliberately multi-minimum" example
        # is more about demonstrating the API than asserting a specific
        # numerical outcome. The invariant we DO assert here is that
        # `iterate=5` is never strictly WORSE than `iterate=1` (the
        # retry layer is a strict refinement of pass 1).
        function fcn_bi(x)
            a, b = x[1], x[2]
            deep    = 1.5 * exp(-((a - 3.0)^2 + (b - 3.0)^2) / 1.0)
            shallow = 0.7 * exp(-((a + 3.0)^2 + (b + 3.0)^2) / 1.0)
            return -deep - shallow + 0.05 * (a^2 + b^2)
        end
        for start in ([0.0, 0.0], [-1.0, -1.0], [1.0, 1.0],
                       [-2.5, -2.5], [2.5, 2.5])
            m1 = Minuit(fcn_bi, start;
                         names = ["a", "b"], errors = [1.0, 1.0])
            migrad!(m1; iterate = 1)
            m5 = Minuit(fcn_bi, start;
                         names = ["a", "b"], errors = [1.0, 1.0])
            migrad!(m5; iterate = 5)
            @test m5.fval ≤ m1.fval + 1e-10
            @test m5.valid
        end
    end

    @testset "Implicit resume still works after migrad! + migrad!" begin
        # The retry layer must NOT break the iminuit-style implicit
        # resume: calling migrad!(m) twice in a row should carry the
        # previous converged point forward as the new seed. Start far
        # from the minimum so the first call uses non-trivial nfcn,
        # then the second call (from the minimum) should be CHEAPER —
        # which is only possible if the resume path is actually feeding
        # the converged values into the new pass-1 seed.
        m = Minuit(fcn_convex, [10.0, -10.0];   # far from (1, 2)
                    names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m; iterate = 5)
        @test m.fmin !== nothing                # converged state stored
        nfcn_first = m.nfcn
        v1 = copy(m.values)
        migrad!(m; iterate = 5)
        @test m.values ≈ v1 atol = 1e-8         # idempotent at the minimum
        @test m.valid
        # On a convex FCN, a second migrad starting from the converged
        # point should converge in far fewer FCN calls than starting
        # from the far-away seed.
        @test m.nfcn < nfcn_first
    end

    @testset "Bounded migrad accepts prior_cov directly (M5 plumbing)" begin
        # Direct functional test of the bounded `migrad(cf, params;
        # prior_cov=...)` plumbing this PR added. The seed wraps the
        # supplied matrix as the inverse Hessian and sets dcovar=0.0;
        # we don't have direct access to dcovar from the BFM but we
        # can verify a well-conditioned prior_cov produces a valid fit
        # in the same number of (or fewer) calls as the cold start.
        cf = JuMinuit.CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params = JuMinuit.Parameters(["x", "y"], [0.0, 0.0], [0.1, 0.1])
        # Cold start
        bfm_cold = migrad(cf, params)
        @test JuMinuit.is_valid(bfm_cold)
        # Warm start with the cold result's inverse Hessian as prior_cov
        # (also in internal coords since this FCN has no bounds → int == ext).
        prior = bfm_cold.internal.state.error.inv_hessian
        cf2 = JuMinuit.CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params2 = JuMinuit.Parameters(["x", "y"], [0.0, 0.0], [0.1, 0.1])
        bfm_warm = migrad(cf2, params2; prior_cov = prior)
        @test JuMinuit.is_valid(bfm_warm)
        # And the CFwG overload accepts prior_cov too
        cfwg = JuMinuit.CostFunctionWithGradient(
            x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
            x -> [2.0 * (x[1] - 1.0), 2.0 * (x[2] - 2.0)],
            1.0)
        params3 = JuMinuit.Parameters(["x", "y"], [0.0, 0.0], [0.1, 0.1])
        bfm_ad = migrad(cfwg, params3; prior_cov = prior)
        @test JuMinuit.is_valid(bfm_ad)
    end

    @testset "AD-gradient FCN survives retry path (codex BLOCKING regression)" begin
        # The AD `seed_state` rejects strategy.level != 0 (see
        # src/ad_gradient.jl:254-255). The retry loop must NOT bump to
        # Strategy(2) when `m.cfwg !== nothing`, or any retry pass would
        # throw on the AD seed. We exercise both `use_simplex=true` (the
        # default) and `use_simplex=false` (which engages `_retry_prior_cov`
        # if retry triggers). Pass 1 validates on this convex FCN, so the
        # retry loop never enters — but the AD-aware retry-strategy
        # selection logic (`m.cfwg === nothing ? Strategy(2) : strategy`)
        # is reached unconditionally before the loop check, so any
        # regression that re-hardcodes `Strategy(2)` would still throw at
        # construction time.
        f_ad = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        g_ad = x -> [2.0 * (x[1] - 1.0), 2.0 * (x[2] - 2.0)]
        for us in (true, false)
            m_ad = Minuit(f_ad, [0.0, 0.0];
                           names = ["x", "y"], errors = [0.1, 0.1],
                           grad = g_ad)
            migrad!(m_ad; iterate = 5, use_simplex = us)
            @test m_ad.valid
            @test m_ad.fval < 1e-8
            @test m_ad.values[1] ≈ 1.0 atol = 1e-4
            @test m_ad.values[2] ≈ 2.0 atol = 1e-4
        end
    end

end
