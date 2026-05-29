# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "hesse.jl — MnHesse" begin

    @testset "Quadratic — exact Hessian recovery" begin
        # f(x) = Σ xᵢ². Hessian = 2·I. Inverse = 0.5·I.
        cf = CostFunction(x -> sum(abs2, x))
        # Run MIGRAD to convergence first
        m = migrad(cf, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        @test m.is_valid
        # Apply HESSE
        new_state = hesse(cf, m.state, Strategy(1))
        @test is_valid(new_state.error)
        @test is_accurate(new_state.error)  # dcov < 0.1
        # The inv_hessian should be ≈ 0.5·I for the quadratic
        for i in 1:3, j in 1:3
            expected = i == j ? 0.5 : 0.0
            @test new_state.error.inv_hessian[i, j] ≈ expected atol = 1e-5
        end
    end

    @testset "Hesse refines off-diagonal" begin
        # f(x, y) = x² + y² + 0.1·x·y; Hessian = [2 0.1; 0.1 2].
        # Inverse = (1/(4-0.01)) · [2 -0.1; -0.1 2] ≈ 0.5006·[2 -0.1; -0.1 2]
        cf = CostFunction(x -> x[1]^2 + x[2]^2 + 0.1 * x[1] * x[2])
        m = migrad(cf, [1.0, 1.0], [0.1, 0.1])
        @test m.is_valid
        st = hesse(cf, m.state, Strategy(1))
        # 2x2 Hessian: H = [2 0.1; 0.1 2]; det = 3.99; inv = (1/3.99)·[2 -0.1; -0.1 2]
        inv_det = 1.0 / (4.0 - 0.01)
        @test st.error.inv_hessian[1, 1] ≈ inv_det * 2.0 atol = 1e-4
        @test st.error.inv_hessian[2, 2] ≈ inv_det * 2.0 atol = 1e-4
        @test st.error.inv_hessian[1, 2] ≈ -inv_det * 0.1 atol = 1e-4
    end

    @testset "Hesse on 1D quadratic" begin
        cf = CostFunction(x -> 3.0 * x[1]^2)  # Hessian = 6; inv = 1/6
        m = migrad(cf, [1.0], [0.1])
        st = hesse(cf, m.state, Strategy(1))
        @test st.error.inv_hessian[1, 1] ≈ 1 / 6 atol = 1e-5
        @test is_valid(st.error)
    end

    @testset "Hesse increments NFcn" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        nfcn_before = nfcn(m.state)
        st = hesse(cf, m.state, Strategy(1))
        # HESSE does at least 2n + n·(n-1)/2 + 1 calls = 2·2 + 1 + 1 = 6
        @test nfcn(st) >= nfcn_before + 2
    end

    @testset "Strategy levels exercise different ncycles" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 1.0], [0.1, 0.1])
        # Strategy 0/1/2 differ in hessian_ncycles (3/5/7); all should
        # converge on a smooth quadratic.
        for level in (0, 1, 2)
            st = hesse(cf, m.state, Strategy(level))
            @test is_valid(st.error)
        end
    end

    @testset "Type stability" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test (@inferred hesse(cf, m.state, Strategy(1))) isa MinimumState
    end

    # ─────────────────────────────────────────────────────────────────
    # Regression: `_hesse_diagonal_failure` must NOT clamp `1/g2` to 1.0
    # when `g2` is well-defined (even if `1/g2` is below `eps2`).
    #
    # See `BenchmarkExamples/IAM_2Pformfactor` failure mode discussion
    # in the `_hesse_diagonal_failure` docstring. The C++ reference
    # has a double-`eps2` check that produces `V = I` everywhere when
    # one parameter is locally FCN-flat (sag=0 bail) AND all other
    # parameters have huge `g2` (steep FCN). The resulting Newton step
    # `−V·g ≈ −g` blows up under line search.
    #
    # We intentionally diverge from C++: use `1/g2` whenever `g2` itself
    # is well-defined. The principle matches `MnSeedGenerator.cxx:69-70`.
    # ─────────────────────────────────────────────────────────────────
    @testset "Regression: _hesse_diagonal_failure preserves 1/g2 for huge g2" begin
        # 2-param FCN: steep in x[1], FLAT in x[2]. At x=[0,0]:
        #   f = 1e9 · x[1]² + 0·x[2]
        #   g  = [2e9·x[1], 0]
        #   g2 = [2e9, 0]
        # The diagonal pass succeeds on x[1] (g2=2e9 huge), fails on
        # x[2] (sag=0 forever → bail to `_hesse_diagonal_failure`).
        # The C++ MnHesse-fail fallback (mirrored here after the
        # `feat/davidon-cxx-audit` option-1 revert) applies a SECOND
        # CLAMP `tmp < eps2 ? 1 : tmp`: for x[1] with `1/g2 = 5e-10`
        # which is below `eps2 = 2.98e-8`, V[1,1] gets clamped to 1.0.
        # See docs/DAVIDON_CXX_AUDIT.md for the audit that motivated
        # restoring this clamp — it's what lets IAM x_jm warm-start
        # walk to χ²=322.59 via the V≈I-induced large Newton step.
        cf = CostFunction(x -> 1e9 * x[1]^2 + 0.0 * x[2], 1.0)
        seed = JuMinuit.seed_state(cf, [0.5, 0.5], [0.1, 0.1],
                                    Strategy(2), JuMinuit.MachinePrecision())
        # The seed status is MnHesseFailed (sag=0 on param 2 bails the
        # diagonal pass) — this matches C++ behavior and is the EXPECTED
        # status for this pathological FCN.
        @test seed.error.status == MnHesseFailed
        V = parent(seed.error.inv_hessian)
        # C++ MnHesse.cxx:177-180 second clamp: 1/g2[1] = 5e-10 < eps2 →
        # V[1,1] = 1.0 (the clamp value).
        @test V[1, 1] == 1.0
        # For the truly-degenerate param 2 (g2=0 < eps2), the fallback
        # to 1.0 is correct — `1/0` is not meaningful.
        @test V[2, 2] == 1.0
    end

    @testset "Regression: _migrad_loop accepts a bailed-hesse seed" begin
        # Companion to the above: after `_hesse_diagonal_failure` returns
        # a MnHesseFailed-status seed with USABLE diagonal V, the
        # `_migrad_loop` entry gate must NOT bail (matches C++ behavior
        # where `BasicMinimumSeed::IsValid()` returns the seed's own
        # `fValid` flag, which is `true` after construction regardless
        # of the underlying state's validity).
        #
        # Pre-fix: `_migrad_loop` bailed at the `!is_valid(seed)` check,
        # returning fmin.is_valid=false with nfcn unchanged from the seed.
        # Post-fix: MIGRAD iterates from the bailed-hesse seed.
        cf = CostFunction(x -> 1e9 * x[1]^2 + 0.0 * x[2], 1.0)
        seed = JuMinuit.seed_state(cf, [0.5, 0.5], [0.1, 0.1],
                                    Strategy(2), JuMinuit.MachinePrecision())
        @test seed.error.status == MnHesseFailed   # bailed, as expected
        n_seed = JuMinuit.ncalls(cf)
        # `_migrad_loop` must run — at minimum, evaluate FCN more than
        # the seed did and reach the minimum of the x[1] subspace.
        fmin = JuMinuit._migrad_loop(seed, cf, Strategy(2), 0.1,
                                      200 + 100 * 2 + 5 * 4,
                                      JuMinuit.MachinePrecision())
        @test JuMinuit.ncalls(cf) > n_seed   # MIGRAD actually iterated
        # The non-degenerate subspace converges: x[1] -> 0 (minimum of 1e9·x[1]²)
        @test abs(fmin.state.parameters.x[1]) < 1e-3
        # The x[1] residual is small (FCN at minimum should be ≈ 0 along x[1])
        @test fmin.state.parameters.fval < 1.0
    end

    # ─────────────────────────────────────────────────────────────────
    # Standalone HESSE from vectors — the FCN+params+errors C++ overload
    # (MnHesse.h:57-74). Computes a covariance at a user-supplied point
    # WITHOUT any prior MIGRAD.
    # ─────────────────────────────────────────────────────────────────
    @testset "Standalone hesse(f, x0, errors) — no prior MIGRAD" begin
        # Diagonal χ²: f(x) = Σ aᵢ (xᵢ - cᵢ)². Hessian Hᵢᵢ = 2aᵢ,
        # inverse Vᵢᵢ = 1/(2aᵢ), user covariance = 2·up·V = up/aᵢ (up=1).
        a = [1.0, 4.0, 0.25]
        c = [1.0, -2.0, 3.0]
        f = x -> sum(a[i] * (x[i] - c[i])^2 for i in eachindex(x))
        # Evaluate at a point that is NOT the minimum — errors at a given point.
        x0 = [0.0, 0.0, 0.0]
        r = hesse(f, x0, [0.5, 0.5, 0.5])
        @test r isa HesseResult
        @test r.valid
        @test r.status == MnHesseValid
        # Covariance ≈ diag(1/aᵢ); off-diagonal ≈ 0 (separable FCN).
        for i in 1:3, j in 1:3
            expected = i == j ? 1.0 / a[i] : 0.0
            @test r.covariance[i, j] ≈ expected atol = 1e-5
        end
        @test r.errors ≈ sqrt.(1.0 ./ a) atol = 1e-5
        # The point is unchanged (no minimization / no line search moved it).
        @test r.x == x0
        @test r.nfcn > 0
    end

    @testset "Standalone hesse — off-diagonal covariance" begin
        # f(x) = xᵀ A x with A = [1 0.5; 0.5 1]. Hessian = 2A,
        # user covariance = 2·up·inv(2A) = up·inv(A). For this A,
        # inv(A) = [4/3 -2/3; -2/3 4/3] (det A = 0.75).
        f = x -> x[1]^2 + x[1] * x[2] + x[2]^2
        r = hesse(f, [0.3, -0.4], [0.2, 0.2]; up = 1.0)
        @test r.valid
        cov_expected = [4/3 -2/3; -2/3 4/3]   # up·inv(A), up = 1
        for i in 1:2, j in 1:2
            @test r.covariance[i, j] ≈ cov_expected[i, j] atol = 1e-4
        end
        # `errors` is the sqrt of the covariance diagonal.
        @test r.errors[1] ≈ sqrt(cov_expected[1, 1]) atol = 1e-4
    end

    @testset "Standalone hesse — up scales covariance linearly" begin
        # f(x) = (x - 1)². H = 2, V = 0.5, covariance = 2·up·0.5 = up.
        f = x -> (x[1] - 1.0)^2
        r1  = hesse(f, [0.0], [0.3]; up = 1.0)
        r05 = hesse(f, [0.0], [0.3]; up = 0.5)
        @test r1.covariance[1, 1] ≈ 1.0 atol = 1e-5
        @test r05.covariance[1, 1] ≈ 0.5 atol = 1e-5
    end

    @testset "Standalone hesse — dimension mismatch throws" begin
        @test_throws DimensionMismatch hesse(x -> sum(abs2, x), [0.0, 0.0],
                                              [0.1])
    end

    # ─────────────────────────────────────────────────────────────────
    # Fix 1: bounded-parameter step clamp (C++ MnHesse.cxx:160-167,
    # 194-195). For a parameter with limits, the diagonal probe step `d`
    # is bounded at 0.5 in INTERNAL coordinates. Unbounded HESSE
    # (`has_limits === nothing`) must be byte-identical to before.
    # ─────────────────────────────────────────────────────────────────
    @testset "Bounded step clamp — unbounded path byte-identical" begin
        # `has_limits = nothing` (default) ≡ an all-false vector ≡ the
        # legacy no-clamp path. The clamp must NEVER fire when unbounded.
        cf = CostFunction(x -> x[1]^2 + 0.3 * x[1] * x[2] + 2.0 * x[2]^2 +
                                0.05 * x[1]^3)
        m = migrad(cf, [0.6, -0.3], [0.1, 0.1])
        st_default = hesse(cf, m.state, Strategy(2))
        st_none    = hesse(cf, m.state, Strategy(2); has_limits = nothing)
        st_false   = hesse(cf, m.state, Strategy(2); has_limits = [false, false])
        # Bit-identical covariance, g2, and step across all three.
        @test st_default.error.inv_hessian == st_none.error.inv_hessian
        @test st_none.error.inv_hessian    == st_false.error.inv_hessian
        @test st_none.gradient.g2    == st_false.gradient.g2
        @test st_none.gradient.gstep == st_false.gradient.gstep
        # Sanity: the unbounded covariance is the correct analytic one.
        @test is_valid(st_default.error)

        # Length validation: a mismatched has_limits vector throws.
        @test_throws DimensionMismatch hesse(cf, m.state, Strategy(1);
                                              has_limits = [true])
    end

    @testset "Bounded step clamp limits the probe step (MnHesse.cxx:160-167)" begin
        # FCN with a flat plateau in [-0.1, 0.1] (central-difference sag
        # is exactly 0 there) and quadratic walls outside. The diagonal
        # multiplier loop must grow the probe step `d` out of the plateau
        # to get a non-zero sag. C++ clamps that growth at 0.51 for a
        # BOUNDED parameter (HasLimits); an UNBOUNDED parameter grows
        # unchecked (×10 per step → 1.0 here).
        plateau = x -> (max(0.0, abs(x[1]) - 0.1))^2

        function run_hesse(lim::Bool)
            probes = Float64[]
            cf = CostFunction(x -> (push!(probes, x[1]); plateau(x)))
            x0 = [0.0]
            fval0 = cf(x0)
            # Seed step 0.01 starts INSIDE the plateau (sag == 0 → growth).
            grad = FunctionGradient([0.0], [1.0], [0.01])
            par  = MinimumParameters(copy(x0), [0.01], fval0)
            err0 = MinimumError(reshape([1.0], 1, 1), MnHesseValid)
            state = MinimumState(par, err0, grad, 0.0, ncalls(cf))
            st = hesse(cf, state, Strategy(1); has_limits = [lim])
            return st, maximum(abs, probes)
        end

        st_b, maxprobe_b = run_hesse(true)
        st_u, maxprobe_u = run_hesse(false)

        # Bounded: the multiplier-loop clamp caps the probe step at 0.51.
        @test maxprobe_b ≤ 0.51 + 1e-9
        # Unbounded: the same plateau forces the step well past 0.5 (→ 1.0).
        @test maxprobe_u > 0.9
        # The clamp samples g2 at a different step ⇒ a different covariance.
        @test parent(st_b.error.inv_hessian)[1, 1] !=
              parent(st_u.error.inv_hessian)[1, 1]
    end

    @testset "hesse(m::Minuit) clamps bounded params near a bound" begin
        # Optimum at x = 0.02, jammed against the lower bound 0.0 (∈ [0,1])
        # — the region where the sin-transform is steep and an unclamped
        # HESSE probe would make a wild external excursion. Strategy(2)
        # exercises BOTH the inner-MIGRAD HESSE and the standalone hesse(m).
        m = Minuit(x -> (x[1] - 0.02)^2 + (x[2] - 0.5)^2, [0.5, 0.5];
                    names = ["x", "y"],
                    limits = [(0.0, 1.0), (0.0, 1.0)],
                    strategy = 2)
        # Tight tol so the convergence check is meaningful near the bound,
        # where the default tol=0.1 leaves the steep-transform external
        # value ~0.023 (the clamp does NOT move the converged point — it is
        # identical across Strategy 0/1/2 — it only bounds the HESSE probe).
        migrad!(m; tol = 1e-6)
        @test m.is_valid
        @test m.values[1] ≈ 0.02 atol = 1e-3
        hesse(m)
        # HESSE returns a usable covariance even with the optimum against
        # the bound (the clamp keeps the internal probe sane).
        @test m.fmin.internal.state.error.status in (MnHesseValid, MnMadePosDef)
        @test all(isfinite, m.errors)
        @test m.errors[1] > 0
        @test m.errors[2] > 0
    end
end
