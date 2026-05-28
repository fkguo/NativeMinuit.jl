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
        # previous converged point forward as the new seed.
        m = Minuit(fcn_convex, [0.0, 0.0];
                    names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m; iterate = 5)
        v1 = copy(m.values)
        migrad!(m; iterate = 5)
        @test m.values ≈ v1 atol = 1e-8  # idempotent at the minimum
        @test m.valid
    end

end
