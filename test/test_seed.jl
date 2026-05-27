# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "seed.jl — MnSeedGenerator (Phase 0 numerical)" begin

    @testset "Quadratic seed at the minimum" begin
        # f(x) = Σ xᵢ². Minimum at origin, ∇f = 2x, g2 = 2.
        cf = CostFunction(x -> sum(abs2, x))
        x0 = [0.0, 0.0, 0.0]
        errs = [0.1, 0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test is_valid(state.parameters)
        @test has_parameters(state)
        @test state.parameters.fval ≈ 0.0 atol = 1e-14
        # Gradient at origin = 0; g2 = 2; inv_hessian diag = 1/2 = 0.5
        for i in 1:3
            @test state.gradient.grad[i] ≈ 0.0 atol = 1e-6
            @test state.gradient.g2[i] ≈ 2.0 atol = 1e-6
            @test state.error.inv_hessian[i, i] ≈ 0.5 atol = 1e-6
        end
        # EDM ≈ 0 at the minimum
        @test estimate_edm(state.gradient, state.error) < 1e-6
        @test !has_negative_g2(state.gradient)
    end

    @testset "Quadratic seed off-center" begin
        # Same FCN; seed at (1, 2, 3) — gradient is 2x = (2, 4, 6)
        cf = CostFunction(x -> sum(abs2, x))
        x0 = [1.0, 2.0, 3.0]
        errs = [0.1, 0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test state.parameters.fval ≈ 14.0 atol = 1e-12
        for i in 1:3
            @test state.gradient.grad[i] ≈ 2 * x0[i] atol = 1e-6
            @test state.gradient.g2[i] ≈ 2.0 atol = 1e-6
        end
        # EDM = 0.5 · (4+16+36) · 0.5 = 14
        @test estimate_edm(state.gradient, state.error) ≈ 14.0 atol = 1e-5
    end

    @testset "Rosenbrock-2 seed" begin
        # Classic starting point (-1.2, 1) for 2D Rosenbrock
        cf = CostFunction(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
        x0 = [-1.2, 1.0]
        errs = [0.1, 0.1]
        state = seed_state(cf, x0, errs)
        @test state.parameters.fval ≈ 24.2 atol = 1e-12
        # Initial state should have positive-def diagonal Hessian
        for i in 1:2
            @test state.error.inv_hessian[i, i] > 0
        end
        @test is_valid(state.error)
    end

    @testset "Strategy ≥ 1 now produces a seed (Phase 1 exit gate)" begin
        # Phase 1 removed the Strategy(1)/(2) seed-stage throw; the MIGRAD
        # outer loop now ships the inner-Hesse refinement so Strategy ≥ 1
        # works end-to-end. The seed-stage MnHesse bootstrap (Strategy 2
        # path) is intentionally deferred (see src/seed.jl note).
        cf = CostFunction(x -> sum(abs2, x))
        s1 = seed_state(cf, [1.0, 2.0], [0.1, 0.1], Strategy(1))
        @test is_valid(s1.error)
        s2 = seed_state(cf, [1.0, 2.0], [0.1, 0.1], Strategy(2))
        @test is_valid(s2.error)
    end

    @testset "Dimension mismatch throws" begin
        cf = CostFunction(x -> sum(abs2, x))
        @test_throws DimensionMismatch seed_state(cf, [1.0, 2.0], [0.1])
    end

    @testset "Type stability" begin
        cf = CostFunction(x -> sum(abs2, x))
        # @inferred-clean on the no-negative-g2 path (Quad FCN, g2 = 2 > 0).
        # parallel-review #2 D4 — without this, a Symbol-typed return path
        # could sneak in and be invisible to the non-inferred test.
        @test (@inferred seed_state(cf, [1.0, 2.0], [0.1, 0.1])) isa MinimumState
    end

    @testset "warm_restart_state — happy path" begin
        # The contour/MINOS-probe usage pattern: take a converged inner-
        # MIGRAD MinimumState from probe k, build a warm seed for the new
        # cf_fixed at probe k+1, then run migrad on the warm seed.
        rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        cf = CostFunction(rosen, 1.0)
        # First, get a converged state via cold migrad
        fmin = migrad(cf, [-1.2, 1.0], [0.1, 0.1])
        prev = fmin.state
        @test is_valid(prev)
        @test prev.error.dcovar ≥ 0

        # Build a NEW cf (different closure / call counter) — simulates
        # the per-probe cf_fixed in function_cross
        cf_new = CostFunction(rosen, 1.0)

        seed_warm = warm_restart_state(prev, cf_new)
        @test seed_warm !== nothing
        @test is_valid(seed_warm)
        @test seed_warm.parameters.x == prev.parameters.x
        @test seed_warm.error.inv_hessian === prev.error.inv_hessian  # NOT copied
        # Re-eval cost paid: at least 1 FCN call for fval + grad refine.
        @test JuMinuit.ncalls(cf_new) ≥ 1
        # New EDM should be near zero (we're at the minimum)
        @test abs(seed_warm.edm) < 1.0
    end

    @testset "warm_restart_state — fallback returns nothing" begin
        rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        cf = CostFunction(rosen)

        # Invalid sentinel state — should bail out
        invalid = MinimumState(3)  # n=3 sentinel, !is_valid
        @test warm_restart_state(invalid, cf) === nothing

        # Zero-dim state — should also bail
        empty = MinimumState(0)
        @test warm_restart_state(empty, cf) === nothing
    end

    @testset "warm_restart_state — sees a new fixed value" begin
        # Confirm that the warm-seed's fval reflects the NEW cf, not the
        # cached prev.parameters.fval. Use two FCNs with different
        # constants to verify the re-evaluation.
        cf_a = CostFunction(x -> x[1]^2 + x[2]^2, 1.0)
        fmin_a = migrad(cf_a, [1.5, 0.5], [0.1, 0.1])
        @test fmin_a.is_valid

        cf_b = CostFunction(x -> (x[1] - 5.0)^2 + (x[2] - 5.0)^2, 1.0)
        seed_warm = warm_restart_state(fmin_a.state, cf_b)
        @test seed_warm !== nothing
        # At x ≈ origin (the minimum of cf_a), cf_b ≈ 25 + 25 = 50
        @test seed_warm.parameters.fval ≈ 50.0 atol = 1e-3
    end

    @testset "migrad(cf, seed) overload — no-op at converged seed" begin
        # Feed a converged MinimumState back into the new migrad overload;
        # it should return immediately (edm ≤ tol·0.002), ZERO new FCN calls.
        cf = CostFunction(x -> sum(abs2, x))
        fmin0 = migrad(cf, [1.0, -1.0, 0.5], [0.1, 0.1, 0.1])
        @test fmin0.is_valid

        cf_resume = CostFunction(x -> sum(abs2, x))
        fmin1 = migrad(cf_resume, fmin0.state)
        @test fmin1.is_valid
        @test fmin1.state.parameters.fval ≈ fmin0.state.parameters.fval atol = 1e-14
        # At a converged seed, _migrad_loop bails before line search.
        @test JuMinuit.ncalls(cf_resume) == 0
    end

    @testset "migrad(cf, seed) — warm vs cold yields same minimum" begin
        # End-to-end: build a warm seed via warm_restart_state, run
        # migrad(cf, seed), check final answer matches a cold migrad call.
        rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        # Run cold once to get a reference converged state
        cf_ref = CostFunction(rosen)
        fmin_ref = migrad(cf_ref, [-1.2, 1.0], [0.1, 0.1])
        @test fmin_ref.is_valid

        # Warm-restart into the same problem (no fixed parameter changed)
        cf_warm_in = CostFunction(rosen)
        seed_w = warm_restart_state(fmin_ref.state, cf_warm_in)
        @test seed_w !== nothing

        cf_warm_run = CostFunction(rosen)
        # Re-evaluate the FCN once at the warm position so the call counter
        # we test against matches what `_migrad_loop` will observe via
        # warm_restart_state when called with a fresh cf.
        seed_w2 = warm_restart_state(fmin_ref.state, cf_warm_run)
        fmin_warm = migrad(cf_warm_run, seed_w2)
        @test fmin_warm.is_valid
        @test fmin_warm.state.parameters.x ≈ fmin_ref.state.parameters.x atol = 1e-6
        @test fmin_warm.state.parameters.fval ≈ fmin_ref.state.parameters.fval atol = 1e-8
    end

    @testset "migrad(cf, seed) — invalid seed returns invalid result" begin
        cf = CostFunction(x -> sum(abs2, x))
        bad_seed = MinimumState(3)   # invalid sentinel
        fmin = migrad(cf, bad_seed)
        @test !fmin.is_valid
        # Should NOT have entered the DFP loop
        @test JuMinuit.ncalls(cf) == 0
    end

    @testset "warm-restart aliasing guard — prev.error.inv_hessian unchanged" begin
        # Parallel-review I-1: warm_restart_state shares prev.error.inv_hessian
        # storage with the new seed. Safety relies on `_migrad_loop` never
        # writing back into that storage (it copies into ping-pong V_a/V_b
        # buffers; make_posdef allocates a fresh copy). This test runs
        # ~10 sequential warm probes and asserts the original prev
        # inv_hessian is byte-for-byte unchanged, catching any future
        # refactor that introduces in-place mutation.
        function rosen4(x)
            s = 0.0
            for i in 1:3
                s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
            end
            return s
        end
        cf_outer = CostFunction(rosen4, 1.0)
        fmin = migrad(cf_outer, [-1.2, 1.0, -1.2, 1.0], fill(0.1, 4))
        @test fmin.is_valid

        # Snapshot the original storage
        original_M = copy(parent(fmin.state.error.inv_hessian))

        # Run 10 sequential warm-restart + migrad cycles. Each cycle:
        #   1. Build a slightly different cf (varying fixed-param-like shift)
        #   2. warm_restart_state from fmin.state into the new cf
        #   3. migrad(cf, seed_warm)
        #   4. confirm fmin.state's storage is still untouched
        for k in 1:10
            ε = 0.001 * k
            cf_k = CostFunction(x -> rosen4(x .+ ε), 1.0)
            seed_w = JuMinuit.warm_restart_state(fmin.state, cf_k)
            @test seed_w !== nothing
            # Aliasing check at the seed level
            @test parent(seed_w.error.inv_hessian) === parent(fmin.state.error.inv_hessian)
            # Run a full migrad on the warm seed
            _ = migrad(cf_k, seed_w)
            # Critical: prev storage must be byte-identical to its snapshot
            @test parent(fmin.state.error.inv_hessian) == original_M
        end
    end

    @testset "warm-restart preserves has_step_size semantics" begin
        # Parallel-review N-5: dirin scale must match prev's user-error
        # scale, not the numerical-gradient gstep scale. The 2-arg
        # MinimumParameters ctor used inside _migrad_loop already drops
        # dirin to zeros + has_step_size=false, so the typical
        # post-MIGRAD prev state has has_step_size=false; the warm seed
        # must preserve that (NOT silently flip to true by misclaiming
        # gstep as a user-error scale).
        cf = CostFunction(x -> sum(abs2, x))
        fmin = migrad(cf, [1.0, 2.0], [0.5, 0.5])
        # MIGRAD output: 2-arg ctor → has_step_size = false
        @test !fmin.state.parameters.has_step_size
        cf_new = CostFunction(x -> sum(abs2, x))
        seed_w = warm_restart_state(fmin.state, cf_new)
        @test seed_w !== nothing
        # Preserved through warm restart
        @test seed_w.parameters.has_step_size == fmin.state.parameters.has_step_size
        @test all(iszero, seed_w.parameters.dirin)

        # Now exercise the has_step_size = true branch directly by
        # building a seed via seed_state (which DOES set has_step_size).
        cf_seed = CostFunction(x -> sum(abs2, x))
        prev_with_step = seed_state(cf_seed, [1.0, 2.0], [0.5, 0.5])
        @test prev_with_step.parameters.has_step_size
        cf_warm2 = CostFunction(x -> sum(abs2, x))
        seed_w2 = warm_restart_state(prev_with_step, cf_warm2)
        @test seed_w2 !== nothing
        @test seed_w2.parameters.has_step_size
        @test seed_w2.parameters.dirin == prev_with_step.parameters.dirin
    end
end
