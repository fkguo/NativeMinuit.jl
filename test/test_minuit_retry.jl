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
        # Ensure the retry kwargs survive the bounded + fixed-parameter path:
        # a bounded fit with the default (faithful, use_simplex=false) retry
        # converges and respects the fix. (Pass 1 validates here, so the loop
        # is a no-op — this is a plumbing smoke test, not a retry-mechanism one.)
        m = Minuit(x -> (x[1] - 0.5)^2 + (x[2] - 3.0)^2, [0.3, 5.0];
                    names = ["a", "b"], errors = [0.1, 0.1],
                    limits = [(0.0, 1.0), nothing],
                    fixed = [false, true])
        migrad!(m; iterate = 5, use_simplex = false)
        @test m.valid
        @test m.values[1] ≈ 0.5 atol = 0.01
        @test m.values[2] == 5.0  # fixed
    end

    @testset "Smooth multimodal FCN: pass-1 reaches deep min (iminuit parity)" begin
        # AUDIT §14 (feat/precision-eps-x4): this testset PREVIOUSLY asserted
        # that pass 1 STALLS here (is_valid=false at fval≈20.8) so the retry
        # loop / opt-in Simplex multistart was needed to reach the deep
        # minimum. That "stall" was a bare-eps ARTIFACT — with the corrected
        # machine precision (eps = 4·eps(Float64), matching C++/iminuit) the
        # numerical-gradient steps are no longer too small, and JuMinuit now
        # converges on pass 1, exactly as iminuit does. Cross-checked against
        # iminuit 2.32.0 (Strategy(0), plain migrad, use_simplex=false):
        #   valid=true, fval=-0.862799, a=b=3.21598, nfcn=70.
        # JuMinuit reaches the IDENTICAL minimum (fval/a/b) in nfcn=82; the
        # call-count differs across implementations by construction (not asserted).
        # So this is now a CONVERGENCE-PARITY test. The genuine "pass-1 stalls →
        # retry enters" coverage lives in the "noisy stall" and "fixed-point"
        # testsets below, whose FCNs stall regardless of step size.
        fcn = x -> begin
            a, b = x[1], x[2]
            return (a - 3)^2 + (b - 3)^2 +
                   0.3 * cos(5 * a) + 0.3 * cos(5 * b) +
                   0.5 * sin(a * b)
        end

        # Pin Strategy(0) (quick mode) so the comparison to iminuit's S0 plain
        # migrad is apples-to-apples. (The high-level default is now Strategy(1)
        # for iminuit parity, see docs/IAM_CONVERGENCE_GAP.md.)
        m1 = Minuit(fcn, [-5.0, 5.0];
                     names = ["a", "b"], errors = [0.5, 0.5],
                     strategy = Strategy(0))
        migrad!(m1; iterate = 1, tol = 1e-6, maxfcn = 2000)
        @test m1.valid                                  # converges (≡ iminuit)
        @test !m1.fmin.internal.reached_call_limit      # not via budget
        @test m1.fval < 0.0                             # reaches the deep well
        @test m1.fval ≈ -0.862799 atol = 1e-4           # iminuit: -0.862799
        @test m1.values[1] ≈ 3.21598 atol = 1e-3        # iminuit: a = 3.21598
        @test m1.values[2] ≈ 3.21598 atol = 1e-3        # iminuit: b = 3.21598
        nfcn1 = m1.nfcn
        fval1 = m1.fval

        # DEFAULT retry (use_simplex=false): pass 1 is already valid, so the
        # retry loop is a faithful NO-OP — it must not spend extra calls or
        # worsen the converged fit (idempotence + safety invariant). This
        # matches iminuit: migrad(iterate=5) on a fit that validated on the
        # first pass returns the same state without further iterations.
        m5 = Minuit(fcn, [-5.0, 5.0];
                     names = ["a", "b"], errors = [0.5, 0.5],
                     strategy = Strategy(0))
        migrad!(m5; iterate = 5, tol = 1e-6, maxfcn = 2000)
        @test m5.valid
        @test m5.nfcn == nfcn1             # valid pass 1 ⇒ retry loop no-op
        @test m5.fval ≈ fval1 atol = 1e-10    # idempotent
        @test m5.fval ≤ fval1 + 1e-10         # safety invariant

        # OPT-IN Simplex multistart (use_simplex=true — JuMinuit extension
        # beyond C++/iminuit): on an already-converged fit it must stay valid
        # at the deep minimum and never worsen the result (safety invariant).
        m5s = Minuit(fcn, [-5.0, 5.0];
                      names = ["a", "b"], errors = [0.5, 0.5],
                      strategy = Strategy(0))
        migrad!(m5s; iterate = 5, use_simplex = true, tol = 1e-6, maxfcn = 2000)
        # On an ALREADY-CONVERGED fit the retry loop is correctly a no-op
        # (gated on `!is_valid`): with the Strategy(1) default + §14 eps=4ε,
        # pass 1 reaches the deep minimum, so `n_passes == 1` and we do NOT
        # assert the retry ran here. (The retry-actually-ran coverage is
        # proven by the dedicated fixed-point + multi-scale testsets below.)
        # What this case must guarantee: still valid, never worsened, deep well.
        @test m5s.n_passes == 1          # retry correctly skipped (pass 1 valid)
        @test m5s.valid
        @test m5s.fval ≤ fval1 + 1e-10   # safety invariant (never worse)
        @test m5s.fval < 0.0             # reaches the deep (negative) well
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
        # The retry loop keeps the user's strategy on the AD path rather
        # than bumping to Strategy(2) (the numerical-path multistart
        # heuristic): `retry_strategy = m.cfwg === nothing ? Strategy(2) :
        # strategy`. The AD `seed_state` now supports all strategy levels,
        # so this is a deliberate choice, not a constraint. We exercise both
        # the faithful default (`use_simplex=false` — plain re-seed at the
        # user strategy) and the opt-in `use_simplex=true` Simplex multistart.
        # Pass 1 validates on this convex FCN (at the Strategy(1) default), so
        # the retry loop never enters; the guard here is that the whole AD
        # retry path — strategy selection + seed + (skipped) loop — runs
        # without error under both settings.
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

    # ─────────────────────────────────────────────────────────────────────
    # Fixed-point detection + multi-scale Simplex escape
    # (PR feat/retry-fixed-point-multiscale)
    # ─────────────────────────────────────────────────────────────────────

    @testset "n_passes diagnostic + iterate=1 is single-shot" begin
        # n_passes reports how many MIGRAD passes ran. iterate=1 → exactly
        # one (single-shot); a convex FCN that validates on pass 1 → one
        # even at iterate=5 (the loop never enters).
        m1 = Minuit(fcn_convex, [0.0, 0.0]; names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m1; iterate = 1)
        @test m1.n_passes == 1
        m5 = Minuit(fcn_convex, [0.0, 0.0]; names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m5; iterate = 5)
        @test m5.n_passes == 1     # convex: retry never entered
        @test m5.nfcn == m1.nfcn   # and no extra FCN calls (regression)
    end

    # A shared landscape: a shallow quadratic bowl around the origin, a tiny
    # |sin(1e3·x)| ripple (whose O(1) gradient keeps the EDM above threshold,
    # so any converged point is flagged INVALID without reaching the call
    # limit), and a deep narrow Gaussian well offset at (0.8, 0.8). Seeded at
    # the origin it exercises the multi-scale escape; seeded at the well it
    # exercises fixed-point detection (the well is a clean, strong — but
    # invalid — attractor that every retry re-converges to).
    fcn_rugged = x -> (
        0.3 * (x[1]^2 + x[2]^2)
        + 1.0e-3 * (abs(sin(1.0e3 * x[1])) + abs(sin(1.0e3 * x[2])))
        - 8.0 * exp(-((x[1] - 0.8)^2 + (x[2] - 0.8)^2) / 0.3)
    )

    @testset "Fixed-point detection stops a deterministically cycling retry" begin
        # Seeded AT the well: pass 1 converges to the well but exits INVALID
        # (the ripple keeps EDM above threshold) without hitting the budget.
        # The well is the only attractor in reach, so every retry pass
        # re-converges to the SAME point — the retry map has cycled.
        # Fixed-point detection must catch the re-visit and stop early,
        # recovering the wasted IAM-style retries (where all 4 passes
        # re-converge to one fval).
        m1 = Minuit(fcn_rugged, [0.8, 0.8]; names = ["x", "y"], errors = [0.05, 0.05])
        migrad!(m1; iterate = 1, tol = 1e-6, maxfcn = 4000)
        @test !m1.valid                              # pass 1 fails to validate
        @test !m1.fmin.internal.reached_call_limit   # but NOT via the budget
        @test m1.n_passes == 1

        m5 = Minuit(fcn_rugged, [0.8, 0.8]; names = ["x", "y"], errors = [0.05, 0.05])
        migrad!(m5; iterate = 5, tol = 1e-6, maxfcn = 4000)
        # Loop entered (≥2 passes) but stopped BEFORE exhausting iterate=5:
        # the cycle was detected. This is the core "stops early ONLY when
        # provably redundant" claim.
        @test 2 <= m5.n_passes < 5
        # The retries cycle and add nothing material, so the best-of-passes
        # selector returns essentially pass 1's result.
        #
        # NOTE: the 2nd-pass-invalid bail (C++ VariableMetricBuilder.cxx
        # :127-132) makes each invalid inner pass return its EARLIER-pass
        # point — matching C++, which returns the pass-1 minimum — so the
        # retry no longer cycles to a BIT-identical fval; it agrees to ~1e-7.
        # The safety invariant (best-of-passes is never WORSE than the
        # single-shot pass 1) still holds exactly, since the selector takes
        # the min over passes and m5's pass 1 equals m1.
        @test m5.fval ≈ m1.fval atol = 1e-5
        @test m5.fval <= m1.fval + 1e-9
    end

    @testset "Multi-scale retry escapes a noisy stall to a deeper basin" begin
        # Seeded at the ORIGIN: single-shot (iterate=1) is trapped in the
        # shallow noisy bowl and never reaches the deep well; the OPT-IN
        # Simplex multistart's growing hop escapes to it — the X(3872)-shaped
        # "the deeper solution is only reached by perturbing out of the
        # stall" outcome. This basin jump is the job of the multistart
        # (use_simplex=true, a JuMinuit extension beyond C++/iminuit), not the
        # faithful plain-re-seed default.
        m1 = Minuit(fcn_rugged, [0.0, 0.0]; names = ["x", "y"], errors = [0.02, 0.02])
        migrad!(m1; iterate = 1, tol = 1e-6, maxfcn = 4000)
        @test !m1.valid          # pass 1 stuck in the shallow noisy bowl
        @test m1.fval > -1.0     # has NOT reached the deep well (≈ −7.6)
        @test m1.n_passes == 1

        m5 = Minuit(fcn_rugged, [0.0, 0.0]; names = ["x", "y"], errors = [0.02, 0.02])
        migrad!(m5; iterate = 5, use_simplex = true, tol = 1e-6, maxfcn = 4000)
        @test m5.n_passes >= 2          # retry layer ran
        # NB: pre-§5 the escape "stopped early" (n_passes < 5). Under the
        # C++-faithful Simplex (audit §5: minedm = 0.1·up, looser than the old
        # 1e-5·up), the growing-hop multistart now uses the full iterate
        # budget to reach the deep well — escape still SUCCEEDS (asserted
        # below), it just isn't early-terminating anymore. So we assert the
        # outcome (deep well reached), not the pass count.
        @test m5.n_passes <= 5          # within the iterate budget
        @test m5.fval < -5.0            # reached the deep well (the real goal)
        @test m5.fval < m1.fval - 1.0   # strictly, substantially deeper
        # Safety invariant holds across the escape too.
        @test m5.fval <= m1.fval + 1e-10
    end

    @testset "retry policy helpers (unit)" begin
        # ── Geometric growth schedule ────────────────────────────────────
        # Pass 2 reproduces the fixed-scale hop (×1); each later pass ×2.
        @test JuMinuit._retry_perturb_factor(2) == 1.0
        @test JuMinuit._retry_perturb_factor(3) == 2.0
        @test JuMinuit._retry_perturb_factor(4) == 4.0
        @test JuMinuit._retry_perturb_factor(5) == 8.0

        # ── Physical range used to cap growth ────────────────────────────
        p_bounded = JuMinuit.MinuitParameter("a", 0.0, 0.1; lower = -2.0, upper = 3.0)
        @test JuMinuit._retry_param_range(p_bounded, 0.1) == 5.0   # span
        p_unbounded = JuMinuit.MinuitParameter("b", 0.0, 0.1)
        @test JuMinuit._retry_param_range(p_unbounded, 0.1) ==
              JuMinuit._RETRY_UNBOUNDED_RANGE_MULT * 0.1
        p_onesided = JuMinuit.MinuitParameter("c", 0.0, 0.1; lower = -1.0)
        @test JuMinuit._retry_param_range(p_onesided, 0.1) ==
              JuMinuit._RETRY_UNBOUNDED_RANGE_MULT * 0.1   # not two-sided → step scale

        # ── Scaled-params construction ───────────────────────────────────
        #   a: unbounded → grows ×factor (range huge, never capped)
        #   b: FIXED     → untouched
        #   c: bounds (−1,1), step 0.1 → factor 4 wants 0.4 but the simplex
        #      edge (10·step) must stay ≤ span 2 ⇒ step capped at 2/10 = 0.2
        #   d: bounds (−0.3,0.3), step 0.1 → cap 0.6/10 = 0.06 is BELOW the
        #      base step, so the floor keeps it at the base step (0.1)
        m = Minuit(x -> sum(abs2, x), [0.0, 0.0, 0.0, 0.0];
                    names = ["a", "b", "c", "d"], errors = [0.1, 0.2, 0.1, 0.1],
                    limits = [nothing, nothing, (-1.0, 1.0), (-0.3, 0.3)],
                    fixed = [false, true, false, false])
        base_errs = [p.error for p in m.params.pars]
        # factor ≤ 1 → returned object is the input unchanged (pass-2 identity).
        @test JuMinuit._retry_scaled_params(m, m.params, 1.0, base_errs) === m.params
        sp = JuMinuit._retry_scaled_params(m, m.params, 4.0, base_errs)
        @test sp.pars[1].error ≈ 0.1 * 4.0   # unbounded: grown ×4 = 0.4
        @test sp.pars[2].error == 0.2        # fixed: unchanged
        @test sp.pars[3].error == 0.2        # capped at span/10 (< grown 0.4)
        @test sp.pars[3].error < 0.1 * 4.0   # capping is observable
        @test sp.pars[4].error == 0.1        # floored back to base step

        # ── Saturation predicate ─────────────────────────────────────────
        # A single tightly-bounded parameter: small factor not saturated,
        # large factor saturated (10·factor·step ≥ span).
        ms = Minuit(x -> x[1]^2, [0.0]; names = ["a"], errors = [0.1],
                     limits = [(-1.0, 1.0)])
        be_s = [p.error for p in ms.params.pars]
        @test !JuMinuit._retry_perturb_saturated(ms, 1.0, be_s)   # 10·1·0.1=1 < 2
        @test JuMinuit._retry_perturb_saturated(ms, 2.0, be_s)    # 10·2·0.1=2 ≥ 2
    end

    @testset "best-of-passes selector enforces the safety invariant" begin
        # Build real BoundedFunctionMinima at known fvals via single-shot fits.
        mlo = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                      names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(mlo; iterate = 1)
        mhi = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + 10.0, [0.0, 0.0];
                      names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(mhi; iterate = 1)
        lo = mlo.fmin
        hi = mhi.fmin
        @test JuMinuit.fval(lo) < JuMinuit.fval(hi)
        # Lower fval wins regardless of argument order.
        @test JuMinuit._retry_select_better(lo, hi) === lo
        @test JuMinuit._retry_select_better(hi, lo) === lo
        # A candidate equal to the incumbent does not displace it (identity).
        @test JuMinuit._retry_select_better(lo, lo) === lo
    end

    @testset "fixed-point predicate: detects revisits, never merges distinct minima" begin
        # Two minima with the SAME fval (≈0) but very different positions.
        ma = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(ma; iterate = 1)
        mb = Minuit(x -> (x[1] - 5.0)^2 + (x[2] - 8.0)^2, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(mb; iterate = 1)
        bfm_a = ma.fmin
        bfm_b = mb.fmin
        base_errs = [0.1, 0.1]
        visited_a = Tuple{Vector{Float64},Float64}[
            (copy(bfm_a.ext_values), JuMinuit.fval(bfm_a))]

        # Re-visiting the SAME converged point → detected.
        @test JuMinuit._retry_is_fixed_point(bfm_a, visited_a, base_errs)
        # A DISTINCT minimum with (near-)equal fval → NOT merged: the
        # position gate disambiguates fval-degenerate minima. This is the
        # "fixed-point detection can't false-positive on a still-progressing
        # search" guarantee.
        @test !JuMinuit._retry_is_fixed_point(bfm_b, visited_a, base_errs)
        # Empty history → nothing to match.
        @test !JuMinuit._retry_is_fixed_point(
            bfm_a, Tuple{Vector{Float64},Float64}[], base_errs)

        # Stress the position gate: a CLOSE but distinct minimum (5% of the
        # value scale away, equal fval) must still NOT be merged — the scale
        # is max(|value|, |user step|), so the window is ~1% of |value|.
        mc = Minuit(x -> (x[1] - 1.05)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                     names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(mc; iterate = 1)
        @test !JuMinuit._retry_is_fixed_point(mc.fmin, visited_a, base_errs)

        # The length scale is the *input* base_errs (the stable user step),
        # NOT the fit's converged ext_errors (which can blow up on an invalid
        # fit and falsely widen the window). Sanity check that base_errs is
        # what drives the scale: with an absurd base step the window grows to
        # ~1% of it (≈10) and the (5,8)-vs-(1,2) gap of 6 then DOES fall
        # inside it. (Real fits pass the small user step, so this can't fire.)
        @test JuMinuit._retry_is_fixed_point(bfm_b, visited_a, [1000.0, 1000.0])
    end

end
