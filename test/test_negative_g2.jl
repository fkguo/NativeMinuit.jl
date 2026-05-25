# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "negative_g2.jl — NegativeG2LineSearch" begin

    @testset "has_negative_g2 detector" begin
        # All-positive g2 → false
        g_ok = FunctionGradient([0.1, 0.2], [1.0, 2.0], [1e-3, 1e-3])
        @test !has_negative_g2(g_ok)

        # One negative entry → true
        g_bad = FunctionGradient([0.1, 0.2], [1.0, -0.5], [1e-3, 1e-3])
        @test has_negative_g2(g_bad)

        # Exactly zero counts as non-positive
        g_zero = FunctionGradient([0.1, 0.2], [1.0, 0.0], [1e-3, 1e-3])
        @test has_negative_g2(g_zero)

        # Type stability
        @test (@inferred has_negative_g2(g_ok)) isa Bool
    end

    @testset "Pass-through when g2 is already positive" begin
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0]
        par = MinimumParameters(x, cf(x))
        grad = FunctionGradient([2.0, 4.0], [2.0, 2.0], [1e-3, 1e-3])
        err = MinimumError(2)
        state = MinimumState(par, err, grad, 0.5, 1)
        # No negative g2 — state should pass through unchanged
        new_state = negative_g2_line_search(state, cf, Strategy(0))
        @test new_state === state
    end

    @testset "Triggers + refines on synthetic negative-g2 input" begin
        # Use a positive-definite quadratic and seed it with a manually-
        # constructed FunctionGradient that has a negative g2 entry.
        # After negative_g2 runs, all g2 entries should be positive
        # (the recomputed numerical gradient will give true values).
        cf = CostFunction(x -> sum(abs2, x))  # ∇f = 2x, g2 = 2 uniformly
        x = [1.0, 2.0]
        par = MinimumParameters(x, [0.1, 0.1], cf(x))
        # Manually seed g2 with negative entry to force the search
        bad_grad = FunctionGradient([2.0, 4.0], [-1.0, 2.0], [1e-3, 1e-3])
        err = MinimumError(2)
        state = MinimumState(par, err, bad_grad, 0.5, 1)
        @test has_negative_g2(state.gradient)

        new_state = negative_g2_line_search(state, cf, Strategy(0))
        # After the fix, the recomputed g2 should be positive (the
        # actual second derivative of sum(abs2, x) is 2.0)
        @test !has_negative_g2(new_state.gradient)
        # The error matrix is diagonal 1/g2
        for i in 1:2
            @test new_state.error.inv_hessian[i, i] > 0
        end
        # NFcn should have grown (line search + gradient recompute)
        @test new_state.nfcn > state.nfcn
    end

    @testset "Skip parameters with both grad and g2 negligible" begin
        # If both |grad[i]| < eps and |g2[i]| < eps, the algorithm should
        # skip that parameter and look at others. With ALL params skipped,
        # state passes through with err = diag(1) (the "|g2|<eps2 → 1" rule).
        cf = CostFunction(x -> sum(abs2, x))
        x = [1.0, 2.0]
        par = MinimumParameters(x, cf(x))
        # Both grad ≈ 0 and g2 ≈ 0 for both params — should skip
        grad = FunctionGradient([0.0, 0.0], [-1e-20, -1e-20], [1e-3, 1e-3])
        err = MinimumError(2)
        state = MinimumState(par, err, grad, 0.0, 1)
        @test has_negative_g2(state.gradient)

        new_state = negative_g2_line_search(state, cf, Strategy(0))
        # Both params skipped → no FCN calls beyond the entry — but the
        # function still rebuilds the error matrix.
        for i in 1:2
            # |g2[i]| < eps2 → diag set to 1.0 (C++ NegativeG2.cxx:97)
            @test new_state.error.inv_hessian[i, i] == 1.0
        end
    end

    @testset "Type stability of has_negative_g2 + return type of main" begin
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], cf([1.0, 2.0]))
        grad = FunctionGradient([2.0, 4.0], [2.0, 2.0], [1e-3, 1e-3])
        state = MinimumState(par, MinimumError(2), grad, 0.0, 0)
        @test (@inferred negative_g2_line_search(
            state, cf, Strategy(0))) isa MinimumState
    end
end
