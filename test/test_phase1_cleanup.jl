# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# test_phase1_cleanup.jl — focused tests for the four Phase-1 cleanup items
# tracked in docs/GAP_AUDIT.md (M4, M5, M6, P5).
#
# Coverage:
#   M4: MinosError.upper_state / lower_state populated at the ±σ crossing
#       (unbounded and bounded paths).
#   M5: migrad(...; prior_cov=...) reduces total nfcn vs cold start on
#       Rosenbrock-2D at a warm starting point.
#   M6: migrad(...; storage_level=1) populates fmin.states (length ≥ 1),
#       monotonic-improving fvals, and deep-copied snapshots.
#   P5: 2σ MINOS on a quadratic returns errors = √4·1σ = 2·1σ values;
#       the kwarg also threads through `minos(m::Minuit, ...; sigma=k)`.
# ─────────────────────────────────────────────────────────────────────────────

using JuMinuit
using JuMinuit: MinosError, FunctionMinimum, MinimumState
using LinearAlgebra
using Test

@testset "Phase-1 cleanup — GAP_AUDIT M4/M5/M6/P5" verbose = true begin

    # Convex quadratic in 2D: f(x) = (x1-1)² + 2(x2-2)². Analytic 1σ
    # for χ²-like (up=1): ∂²f/∂x1² = 2 → σ_x1 = √(2·1/2) = 1.0;
    # ∂²f/∂x2² = 4 → σ_x2 = √(2·1/4) = 1/√2 ≈ 0.707. Symmetric well,
    # so MINOS upper ≈ -lower at each sigma multiplier.
    quad2(x) = (x[1] - 1.0)^2 + 2.0 * (x[2] - 2.0)^2

    @testset "P5 — minos sigma!=1 (2σ on quadratic)" begin
        cf = CostFunction(quad2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid

        e1 = minos(fmin, cf, 1)
        e2 = minos(fmin, cf, 1; sigma = 2.0)
        @test JuMinuit.is_valid(e1)
        @test JuMinuit.is_valid(e2)
        # 2σ on a quadratic = 2 × 1σ (up_eff = up · sigma² → aopt ≈ k).
        @test isapprox(e2.upper, 2 * e1.upper; atol = 1e-3)
        @test isapprox(e2.lower, 2 * e1.lower; atol = 1e-3)
        # Sign convention preserved across sigma values.
        @test e2.upper > 0 && e2.lower < 0

        # Same for the other parameter (different curvature).
        e1y = minos(fmin, cf, 2)
        e2y = minos(fmin, cf, 2; sigma = 2.0)
        @test isapprox(e2y.upper, 2 * e1y.upper; atol = 1e-3)
        @test isapprox(e2y.lower, 2 * e1y.lower; atol = 1e-3)

        # sigma <= 0 is rejected at the entry points.
        @test_throws ArgumentError minos(fmin, cf, 1; sigma = 0.0)
        @test_throws ArgumentError minos(fmin, cf, 1; sigma = -1.0)
        # The top-level `minos(m::Minuit, ...)` also threads sigma —
        # exercise via the Minuit API path.
        m = Minuit(quad2, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad!(m)
        minos(m, 1; sigma = 2.0)
        e1_m = m.minos_errors[1]
        @test isapprox(e1_m.upper, 2 * e1.upper; atol = 1e-3)
    end

    @testset "M4 — MinosError.upper_state / lower_state (unbounded)" begin
        cf = CostFunction(quad2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin.is_valid

        e = minos(fmin, cf, 1)
        @test JuMinuit.is_valid(e)
        @test e.upper_state !== nothing
        @test e.lower_state !== nothing
        @test length(e.upper_state) == 2
        @test length(e.lower_state) == 2

        # par_idx=1: the scanned coordinate at the upper crossing is
        # min_par_value + upper ≈ 1 + 1 = 2; the OTHER (free) parameter
        # converges back to its minimum (x[2] ≈ 2) since the quadratic
        # has no x[1]·x[2] correlation.
        @test isapprox(e.upper_state[1], e.min_par_value + e.upper; atol = 1e-6)
        @test isapprox(e.upper_state[2], 2.0; atol = 1e-3)
        @test isapprox(e.lower_state[1], e.min_par_value + e.lower; atol = 1e-6)
        @test isapprox(e.lower_state[2], 2.0; atol = 1e-3)

        # A failing side (synthetic invalid via the public constructor)
        # publishes `nothing`. Construct a MinosError using the legacy
        # constructor (which defaults state snapshots to nothing).
        e_legacy = MinosError(1, 0.0, 1.0, -1.0,
                               true, true, false, false, false, false, 10)
        @test e_legacy.upper_state === nothing
        @test e_legacy.lower_state === nothing
    end

    @testset "M4 — bounded path also populates state snapshots" begin
        # 3-parameter quadratic with one bounded parameter — verifies
        # that `MnCross.ext_state` (captured by function_cross_external)
        # propagates through `_minos_external_via_function_cross` into
        # `MinosError.{upper,lower}_state`.
        f3(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2
        m = Minuit(f3, [0.0, 0.0, 0.0];
                    errors = [0.1, 0.1, 0.1],
                    limits = [(-5, 5), nothing, nothing])
        migrad!(m)
        JuMinuit.hesse(m)        # MINOS needs covariance
        @test m.is_valid
        minos!(m, 1)
        e = m.minos_errors[1]
        @test JuMinuit.is_valid(e)
        @test e.upper_state !== nothing
        @test e.lower_state !== nothing
        @test length(e.upper_state) == 3
        # The bounded path's snapshot is the inner-bounded-MIGRAD's
        # converged EXT vector at the crossing — par 2 / par 3 should
        # be near their unconstrained minimum.
        @test isapprox(e.upper_state[2], 2.0; atol = 1e-3)
        @test isapprox(e.upper_state[3], 3.0; atol = 1e-3)
    end

    @testset "M5 — prior_cov reduces nfcn on Rosenbrock-2D warm start" begin
        # Cold-start Rosenbrock-2D from x=[-1, 1] (the classic hard start).
        rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

        cf_cold = CostFunction(rosenbrock)
        fmin_cold = migrad(cf_cold, [-1.0, 1.0], [0.1, 0.1])
        @test fmin_cold.is_valid
        nfcn_cold = fmin_cold.state.nfcn

        # Warm restart: start ~5% off the minimum and supply a prior
        # covariance that's a reasonable approximation. Should converge
        # in dramatically fewer FCN calls.
        prior = Symmetric([0.5  0.4
                            0.4  0.5], :U)
        cf_warm = CostFunction(rosenbrock)
        fmin_warm = migrad(cf_warm, [0.95, 0.95], [0.1, 0.1];
                            prior_cov = prior)
        @test fmin_warm.is_valid
        @test fmin_warm.state.nfcn < nfcn_cold
        # `dcovar = 0.0` for the warm seed (verified indirectly via
        # convergence: the seed's `edm_corrected = edm·(1 + 3·0)` does
        # not get inflated).
        @test fmin_warm.seed.error.dcovar == 0.0

        # Validation: wrong-size matrix throws DimensionMismatch.
        @test_throws DimensionMismatch migrad(cf_cold, [0.0, 0.0], [0.1, 0.1];
                                               prior_cov = zeros(3, 3))
        # Asymmetric matrix throws ArgumentError.
        @test_throws ArgumentError migrad(cf_cold, [0.0, 0.0], [0.1, 0.1];
                                           prior_cov = [1.0 0.5; 0.1 1.0])

        # Storage decoupling: mutating the user's prior must not affect
        # the seed (the seed copies into its own Matrix).
        prior_mut = Matrix{Float64}([1.0 0.0; 0.0 1.0])
        cf3 = CostFunction(rosenbrock)
        fm = migrad(cf3, [0.95, 0.95], [0.1, 0.1]; prior_cov = prior_mut)
        prior_mut[1, 1] = 999.0    # mutate user's matrix
        @test fm.is_valid          # fit already converged; not affected
        @test fm.seed.error.inv_hessian[1, 1] == 1.0   # seed's V is independent
    end

    @testset "M5 — warm_restart_state prior_cov symmetry validation" begin
        # Codex review BLOCKING: `warm_restart_state` originally only
        # shape-checked `prior_cov`. An asymmetric input like
        # `[1 9; 0 1]` would silently get mirrored by `Symmetric(:U)`
        # into `[1 9; 9 1]`. Verify the helper now rejects.
        rosen(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        cf = CostFunction(rosen)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        prev = fmin.state

        # Valid Symmetric input succeeds.
        ok_M = Symmetric([0.5  0.4
                           0.4  0.5], :U)
        ws_ok = JuMinuit.warm_restart_state(prev, CostFunction(rosen);
                                             prior_cov = ok_M)
        @test ws_ok isa JuMinuit.MinimumState

        # Asymmetric input throws (was silently mirroring pre-fix).
        @test_throws ArgumentError JuMinuit.warm_restart_state(
            prev, CostFunction(rosen);
            prior_cov = [1.0 9.0; 0.0 1.0])

        # Wrong-size throws DimensionMismatch.
        @test_throws DimensionMismatch JuMinuit.warm_restart_state(
            prev, CostFunction(rosen);
            prior_cov = zeros(3, 3))
    end

    @testset "M6 — storage_level=1 populates fmin.states" begin
        rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        cf = CostFunction(rosenbrock)

        # Default: empty history, storage_level=0
        fmin_0 = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        @test fmin_0.storage_level == 0
        @test isempty(fmin_0.states)

        # Opt in: per-iteration history.
        fmin_1 = migrad(CostFunction(rosenbrock), [0.0, 0.0], [0.1, 0.1];
                         storage_level = 1)
        @test fmin_1.storage_level == 1
        @test length(fmin_1.states) >= 1
        @test fmin_1.is_valid

        # Final state matches fmin.state by fval (the last snapshot is
        # the converged state).
        @test fmin_1.states[end].parameters.fval ≈ fmin_1.state.parameters.fval

        # Monotonic improvement on a smooth landscape (Rosenbrock from
        # [0,0] descends monotonically through the MIGRAD iterates).
        fvals = [s.parameters.fval for s in fmin_1.states]
        @test issorted(fvals; rev = true) ||
            all(diff(fvals) .<= 1e-8)   # tolerant: tiny numerical jitter

        # Snapshot independence — entries must not alias each other or
        # the final state. Mutating one shouldn't affect another.
        if length(fmin_1.states) >= 2
            s_a = fmin_1.states[1]
            s_b = fmin_1.states[end]
            @test s_a.parameters.x !== s_b.parameters.x
            @test s_a.gradient.grad !== s_b.gradient.grad
            @test parent(s_a.error.inv_hessian) !==
                  parent(s_b.error.inv_hessian)
        end

        # nfcn monotonically grows across the iteration log.
        nfcns = [s.nfcn for s in fmin_1.states]
        @test issorted(nfcns)
    end
end
