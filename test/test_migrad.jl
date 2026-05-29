# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "migrad.jl — MIGRAD loop" begin

    @testset "Quad-1D: f(x) = x²" begin
        cf = CostFunction(x -> x[1]^2)
        m = migrad(cf, [3.0], [0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-10
        @test values(m)[1] ≈ 0.0 atol = 1e-5
        @test edm(m) < 1e-3
    end

    @testset "Quad-4D matches C++ reference (quad_4d.json)" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1])
        @test m.is_valid
        # C++ reference: fval ≈ 7.81e-20, all params ≈ 0 to 1e-10.
        @test fval(m) ≤ 1e-15
        for i in 1:4
            @test abs(values(m)[i]) < 1e-7
        end
        @test edm(m) < 1e-6
    end

    @testset "Rosenbrock-2D converges (vs analytical minimum)" begin
        # Classic Rosenbrock — minimum at (1, 1), fval = 0
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        m = migrad(cf, [-1.2, 1.0], [0.1, 0.1])
        # Convergence with Strategy(0) is loose; allow a wider tolerance.
        # The C++ reference at Strategy(0) lands at (0.99954, 0.99890).
        @test values(m)[1] ≈ 1.0 atol = 5e-3
        @test values(m)[2] ≈ 1.0 atol = 5e-3
        @test fval(m) < 1e-4
    end

    @testset "Already-at-minimum: zero-iteration return" begin
        # Start AT the minimum of f = sum(abs2, x). Gradient is 0;
        # MIGRAD should detect and return without iterating.
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test m.is_valid
        @test fval(m) ≈ 0.0 atol = 1e-14
        for i in 1:3
            @test abs(values(m)[i]) < 1e-10
        end
    end

    @testset "maxfcn limit caps iterations" begin
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        # Force tiny maxfcn — should hit reached_call_limit
        m = migrad(cf, [-1.2, 1.0], [0.1, 0.1]; maxfcn = 20)
        @test m.reached_call_limit
        @test !m.is_valid
        @test nfcn(m) >= 20  # within limit ± boundary slack
    end

    @testset "Strategy ≥ 1 now accepted (Phase 1 exit gate)" begin
        # Phase 0 locked to Strategy(0) (DR-008). Phase 1 ships the inner
        # MnHesse refinement (`VariableMetricBuilder.cxx:138-173`), so
        # Strategy(1)/(2) no longer throw. See test_migrad's
        # "Strategy(1)/(2) inner-Hesse refinement" set below for behavior.
        cf = CostFunction(x -> sum(abs2, x))
        m1 = migrad(cf, [1.0, 2.0], [0.1, 0.1]; strategy = Strategy(1))
        @test is_valid(m1)
        m2 = migrad(cf, [1.0, 2.0], [0.1, 0.1]; strategy = Strategy(2))
        @test is_valid(m2)
    end

    @testset "FunctionMinimum accessors" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test m isa FunctionMinimum
        @test parameters(m) isa MinimumParameters
        @test errors(m) isa MinimumError
        @test gradient(m) isa FunctionGradient
        @test has_covariance(m)
        @test covariance(m) isa Symmetric{Float64,Matrix{Float64}}
        # Pretty-print should work without error. Phase 3 ships an
        # iminuit-style box; the text contains "Migrad" + "Valid Minimum"
        # for a valid converged result.
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("Migrad", s)
        @test occursin("Valid Minimum", s) || occursin("INVALID", s)
        # Single-line repr still uses the FunctionMinimum prefix
        @test occursin("FunctionMinimum", repr(m))
    end

    @testset "Strategy(1)/(2) inner-Hesse refinement (Phase 1 exit gate)" begin
        # ROADMAP §4 Phase 1 exit: "MnHesse-inside-MIGRAD path works: when
        # Strategy ≥ 1 and Dcovar > 0.05, MIGRAD invokes Hesse internally
        # (VariableMetricBuilder.cxx:138-173). Phase 1 must ship this —
        # it is the iminuit default behavior (Strategy 1)."

        # Simple quadratic CF — converges fast and lets us probe the
        # behavior. Strategy 0 vs Strategy 2 should yield close-to-equal
        # values but Strategy 2 must spend EXTRA FCN calls on Hesse.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        m0 = migrad(cf, [0.0, 0.0], [0.1, 0.1]; strategy = Strategy(0))
        @test is_valid(m0)
        nfcn0 = nfcn(m0)

        # Strategy 2 — always triggers Hesse refinement
        cf2 = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        m2 = migrad(cf2, [0.0, 0.0], [0.1, 0.1]; strategy = Strategy(2))
        @test is_valid(m2)
        # Hesse adds ~2·n + 2·C(n,2) = 2·2 + 2·1 = 6 FCN calls at minimum
        # for a 2-param fit (diagonal + one off-diagonal).
        @test nfcn(m2) > nfcn0  # MUST have invoked Hesse
        # Both should land at same minimum (Hesse should not move the point).
        @test Base.values(m2) ≈ Base.values(m0) atol = 1e-6
        @test fval(m2) ≈ fval(m0) atol = 1e-6

        # Strategy 1 path — Dcovar > 0.05 triggers Hesse. For simple quadratic
        # DFP can land below the threshold (Dcovar ≈ 0). Run on a CF that
        # produces a less-accurate DFP estimate. A correlated CF often does:
        cf_corr = CostFunction(x -> x[1]^2 + x[2]^2 + 0.5 * x[1] * x[2])
        m1 = migrad(cf_corr, [1.0, 1.0], [0.1, 0.1]; strategy = Strategy(1))
        @test is_valid(m1)
        # Either Dcovar ≤ 0.05 (no refinement) or > 0.05 (refinement).
        # Whichever, the result must remain valid.
        @test Base.values(m1)[1] ≈ 0.0 atol = 1e-4
        @test Base.values(m1)[2] ≈ 0.0 atol = 1e-4

        # Argument check: Strategy(2) on a 1-d CF still triggers Hesse path
        # without errors (n=1 Hesse path is well-defined).
        cf1 = CostFunction(x -> (x[1] - 3.0)^2)
        m1d = migrad(cf1, [0.0], [0.1]; strategy = Strategy(2))
        @test is_valid(m1d)
        @test Base.values(m1d)[1] ≈ 3.0 atol = 1e-4
    end

    # ─────────────────────────────────────────────────────────────────
    # Fix 2: 2nd-pass-invalid early bail (C++ VariableMetricBuilder.cxx
    # :127-132). After a non-first inner DFP pass that is invalid, MIGRAD
    # bails instead of burning further HESSE+DFP passes to the call limit.
    # ─────────────────────────────────────────────────────────────────
    @testset "2nd-pass-invalid bail predicate (VariableMetricBuilder.cxx:127-132)" begin
        # White-box, deterministic check that the bail condition matches
        # the C++ guard `ipass > 0 && !min.IsValid()`.
        cf = CostFunction(x -> sum(abs2, x))
        valid_state = migrad(cf, [1.0, 1.0], [0.1, 0.1]).state
        @test is_valid(valid_state)
        invalid_state = MinimumState(2)            # n=2 invalid sentinel
        @test !is_valid(invalid_state)

        edmval = 1.0e-3
        # ipass == 0 → never bails (preserves the first-pass do-while
        # semantics), even with an invalid state and a huge edm.
        @test !JuMinuit._migrad_second_pass_invalid(0, invalid_state, 1.0e9, edmval)
        @test !JuMinuit._migrad_second_pass_invalid(0, valid_state, 1.0e9, edmval)
        # ipass > 0 + invalid inner state → bail.
        @test JuMinuit._migrad_second_pass_invalid(1, invalid_state, 0.0, edmval)
        # ipass > 0 + above-max-edm (edm_corrected > 10·edmval) → bail.
        @test JuMinuit._migrad_second_pass_invalid(1, valid_state, 11 * edmval, edmval)
        # ipass > 0 + valid + edm within 10·edmval → do NOT bail; a
        # converging multi-pass fit must keep iterating.
        @test !JuMinuit._migrad_second_pass_invalid(1, valid_state, 5 * edmval, edmval)
        @test !JuMinuit._migrad_second_pass_invalid(2, valid_state, 9 * edmval, edmval)
    end

    @testset "2nd-pass bail leaves converging Strategy≥1 fits unchanged" begin
        # The bail fires only for an INVALID non-first pass, so a
        # converging Strategy(2) fit (always valid, edm → 0, exercises the
        # inner-HESSE + bail-check path) must be untouched: same minimum
        # and same valid verdict as Strategy(0).
        f = x -> (x[1] - 1.0)^2 + 0.5 * x[1] * x[2] + (x[2] + 2.0)^2
        m0 = migrad(CostFunction(f), [0.0, 0.0], [0.1, 0.1]; strategy = Strategy(0))
        m2 = migrad(CostFunction(f), [0.0, 0.0], [0.1, 0.1]; strategy = Strategy(2))
        @test is_valid(m0)
        @test is_valid(m2)
        @test !m2.reached_call_limit       # converged, not bailed-to-limit
        @test !m2.above_max_edm
        @test Base.values(m2) ≈ Base.values(m0) atol = 1e-5
        @test fval(m2) ≈ fval(m0) atol = 1e-8
    end
end
