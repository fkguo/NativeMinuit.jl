# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test

@testset "iminuit_compat.jl — Data, chisq, model_fit, helpers" begin

    @testset "Data construction + validation" begin
        d = Data([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.1, 0.1, 0.1])
        @test d.ndata == 3
        @test d.x == [1.0, 2.0, 3.0]
        @test d.y == [2.0, 4.0, 6.0]
        @test d.err == [0.1, 0.1, 0.1]
        @test length(d) == 3

        # Length mismatch throws
        @test_throws ArgumentError Data([1.0, 2.0], [1.0, 2.0, 3.0],
                                          [0.1, 0.1, 0.1])
        # NaN throws
        @test_throws ArgumentError Data([NaN, 1.0], [1.0, 2.0], [0.1, 0.1])
        # Zero error throws
        @test_throws ArgumentError Data([1.0, 2.0], [1.0, 2.0], [0.0, 0.1])
        # Inf throws
        @test_throws ArgumentError Data([Inf, 1.0], [1.0, 2.0], [0.1, 0.1])
    end

    @testset "Data vcat + indexing" begin
        d1 = Data([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])
        d2 = Data([3.0, 4.0], [3.0, 4.0], [0.2, 0.2])
        d = vcat(d1, d2)
        @test d.ndata == 4
        @test d.x == [1.0, 2.0, 3.0, 4.0]
        @test d.err == [0.1, 0.1, 0.2, 0.2]

        # Slice
        ds = d[2:3]
        @test ds.ndata == 2
        @test ds.x == [2.0, 3.0]
    end

    @testset "chisq on Data" begin
        # Linear model y = 2x + 1, perfect data
        d = Data([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], [0.1, 0.1, 0.1])
        model(x, p) = p[1] * x + p[2]
        @test chisq(model, d, [2.0, 1.0]) ≈ 0.0 atol = 1e-14

        # Off-fit: y = 3x + 1 → model returns [1, 4, 7]; residuals
        # (y_data - y_model) = [0, -1, -2]; χ² = (0² + 1² + 2²)/0.01 = 500.
        @test chisq(model, d, [3.0, 1.0]) ≈ 500.0 atol = 1e-9
    end

    @testset "chisq on tuple data" begin
        x = [0.0, 1.0, 2.0]
        y = [1.0, 3.0, 5.0]
        err = [0.1, 0.1, 0.1]
        model(x, p) = p[1] * x + p[2]
        @test chisq(model, (x, y, err), [2.0, 1.0]) ≈ 0.0 atol = 1e-14
        # 2-tuple form: unit errors
        @test chisq(model, (x, y), [2.0, 1.0]) ≈ 0.0 atol = 1e-14
    end

    @testset "model_fit + migrad" begin
        d = Data([0.0, 1.0, 2.0, 3.0, 4.0],
                  [1.0, 3.0, 5.0, 7.0, 9.0],
                  fill(0.1, 5))
        model(x, p) = p[1] * x + p[2]
        m = model_fit(model, d, [1.0, 0.0]; name = ["slope", "icept"],
                       error = [0.1, 0.1])
        migrad(m)
        @test m.values[1] ≈ 2.0 atol = 1e-3
        @test m.values[2] ≈ 1.0 atol = 1e-3
        @test m.fval < 1e-6
    end

    @testset "@model_fit macro" begin
        d = Data([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], [0.1, 0.1, 0.1])
        model(x, p) = p[1] * x + p[2]
        m = @model_fit model d [1.0, 0.0] name=["a","b"] error=[0.1,0.1]
        migrad(m)
        @test m.values ≈ [2.0, 1.0] atol = 1e-3
    end

    @testset "func_argnames" begin
        f1(a, b, c) = a + b + c
        @test func_argnames(f1) == [:a, :b, :c]
        f2(par_x, par_y) = par_x^2 + par_y^2
        @test func_argnames(f2) == [:par_x, :par_y]
    end

    @testset "chi2 / poisson_chi2 / multinominal_chi2" begin
        # chi2: explicit Pearson
        y = [1.0, 2.0, 3.0]
        err = [0.1, 0.1, 0.1]
        ymodel = [1.1, 1.9, 3.0]
        @test chi2(y, err, ymodel) ≈ 2.0 atol = 1e-9
        # Skip bins with err<=0
        @test chi2(y, [0.0, 0.1, 0.1], ymodel) ≈ 1.0 atol = 1e-9   # only 2nd term

        # poisson_chi2: simple match → 0
        @test poisson_chi2([3.0, 4.0, 5.0], [3.0, 4.0, 5.0]) ≈ 0.0 atol = 1e-9
        # Single-bin offset: 2·(μ - n + n·log(n/μ))
        # n=2, μ=2.5 → 2·(2.5 - 2 + 2·log(2/2.5)) = 2·(0.5 + 2·(-0.2231)) = 2·0.0537 ≈ 0.1075
        result = poisson_chi2([2.0], [2.5])
        @test result ≈ 2.0 * (2.5 - 2.0 + 2.0 * log(2.0/2.5)) atol = 1e-9

        # multinominal_chi2: n=μ → 0
        @test multinominal_chi2([3.0, 4.0, 5.0], [3.0, 4.0, 5.0]) ≈ 0.0 atol = 1e-9

        # n=0 should not contribute (0·log(0/μ) = 0 convention)
        @test multinominal_chi2([0.0, 2.0], [1.0, 2.0]) ≈ 0.0 atol = 1e-9

        # Mismatched lengths throw
        @test_throws DimensionMismatch chi2([1.0], [0.1, 0.1], [1.0])
    end

    @testset "mncontour returns point list" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        pts = mncontour(m, 1, 2; numpoints = 8)
        @test length(pts) == 8
        @test all(p -> p isa Tuple{Float64,Float64}, pts)
        # All points within ~1σ of (1, 2)
        @test all(p -> abs(p[1] - 1.0) ≤ 1.5 && abs(p[2] - 2.0) ≤ 1.5, pts)
    end

    @testset "profile = scan with bins default" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        prof = profile(m, 1; bins = 5, low = 0.0, high = 2.0)
        @test length(prof) == 6  # 1 central + 5 grid
        # Verify min at x[1]=1
        grid = prof[2:end]
        @test grid[argmin(p[2] for p in grid)][1] ≈ 1.0 atol = 0.5
    end

    @testset "mnprofile re-minimizes others" begin
        # Quadratic well in both dims; profile-along-x with y free
        # should always show f≈0 (because y can compensate? No — y
        # is at its optimum already; profile fixes x at the grid value
        # and re-min y → still f = (x_grid - 1)² since y can move to 2).
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        mp = mnprofile(m, 1; bins = 5, low = 0.0, high = 2.0)
        @test length(mp) == 5
        # Each profile point should satisfy f_min = (x_grid - 1)²
        for (x_grid, fval) in mp
            @test fval ≈ (x_grid - 1.0)^2 atol = 1e-6
        end
    end

    @testset "scipy throws helpful error" begin
        f = x -> x[1]^2
        m = Minuit(f, [0.0]; name = ["a"], error = [0.1])
        @test_throws ArgumentError scipy(m)
    end

    # ─────────────────────────────────────────────────────────────────
    # Round-2 review fix verification
    # ─────────────────────────────────────────────────────────────────

    @testset "migrad implicit resume: 2nd migrad starts from prior best" begin
        # Verifies carry-forward — after migrad once, calling migrad
        # again should start from the converged point (fewer nfcn).
        f = x -> (x[1] - 5.0)^2 + (x[2] - 7.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        n1 = m.nfcn
        @test m.values ≈ [5.0, 7.0] atol = 1e-6

        # 2nd migrad — should converge essentially instantly because
        # we're already at the minimum.
        migrad(m)
        n2 = m.nfcn - n1
        @test n2 < n1   # less work to redo
        @test m.values ≈ [5.0, 7.0] atol = 1e-6
    end

    @testset "m.params NOT mutated by carry-forward (round-2 B2)" begin
        # m.params is the user's original config — must stay
        # unchanged so reset() returns to initial values.
        f = x -> (x[1] - 5.0)^2
        m = Minuit(f, [0.0]; name = ["a"], error = [0.1])
        x0_original = m.params.pars[1].value
        err_original = m.params.pars[1].error
        migrad(m)
        # After migrad, params.pars[i].value/error must still equal
        # the originals (m.values returns the converged values via
        # m.fmin, NOT via m.params).
        @test m.params.pars[1].value == x0_original
        @test m.params.pars[1].error == err_original
    end

    @testset "migrad resume=false restores initial values" begin
        f = x -> (x[1] - 5.0)^2 + (x[2] - 7.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        # Now move "off" by calling migrad with resume=false. This
        # must restart from m.params (the original [0, 0]).
        migrad(m; resume = false)
        @test m.values ≈ [5.0, 7.0] atol = 1e-6   # still converges
        # Check reset(m) leaves things ready for a fresh migrad
        reset(m)
        @test m.fmin === nothing
        @test m.values == [0.0, 0.0]   # back to constructor's initial
    end

    @testset "minos! after simplex throws helpful error (round-2 I2)" begin
        # simplex produces no covariance → MINOS cannot derive sigma_i.
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        simplex(m)
        @test_throws ArgumentError minos(m, 1)
        # After running hesse(m), it should work.
        hesse(m)
        minos(m, 1)
        @test haskey(m.minos_errors, 1)
    end

    @testset "Inf in err is rejected (round-1 fix verification)" begin
        @test_throws ArgumentError Data([1.0, 2.0], [1.0, 2.0],
                                          [Inf, 0.1])
    end

    @testset "chisq fitrange with stride collapses to first:last" begin
        # IMinuit.jl semantics: fitrange is clamped to first..last
        # (contiguous), even if a stride is passed.
        d = Data([0.0, 1.0, 2.0, 3.0, 4.0],
                  [1.0, 3.0, 5.0, 7.0, 9.0],
                  fill(0.1, 5))
        model(x, p) = p[1] * x + p[2]
        # Stride 2 → indices [2, 4]; but our code clamps to 2:4 (all
        # of 2, 3, 4). Exact match to IMinuit.jl's behavior.
        @test chisq(model, d, [2.0, 1.0]; fitrange = 2:2:4) ≈ 0.0 atol = 1e-12
    end

    @testset "mncontour size= and cl= aliases" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        migrad(m)
        # `size` is iminuit alias for `numpoints`
        pts1 = mncontour(m, 1, 2; numpoints = 8)
        pts2 = mncontour(m, 1, 2; size = 8)
        @test length(pts1) == length(pts2) == 8
        # cl=1.0 is the only currently supported value (Phase 1.x deferred)
        @test_throws ArgumentError mncontour(m, 1, 2; cl = 2.0)
    end

    @testset "simplex(m) ncall alias" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        # ncall is iminuit alias for maxfcn
        simplex(m; ncall = 200)
        # Basin-level accuracy under the C++-faithful EDM goal minedm = 0.1·up
        # (audit §5; was 1e-5·up). This testset checks the ncall→maxfcn alias,
        # not tight convergence — atol relaxed from 1e-2 to match the real goal.
        @test m.values ≈ [1.0, 2.0] atol = 0.1
    end

    @testset "profile/mnprofile size alias" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        migrad(m)
        prof = profile(m, 1; size = 5, low = 0.0, high = 2.0)
        @test length(prof) == 6   # 1 central + 5 grid
        mp = mnprofile(m, 1; size = 5, low = 0.0, high = 2.0)
        @test length(mp) == 5
    end

    @testset "m.values / m.errors / m.limits / m.fixed setters (round-3)" begin
        # iminuit-style live parameter editing.
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        @test m.fmin !== nothing
        # Setting m.values invalidates the prior fit.
        m.values = [5.0, 6.0]
        @test m.fmin === nothing
        @test m.values == [5.0, 6.0]
        # Re-migrad should converge correctly from the new initial.
        migrad(m)
        @test m.values ≈ [1.0, 2.0] atol = 1e-3

        # m.errors setter
        m.errors = [0.5, 0.5]
        @test m.fmin === nothing
        @test m.params.pars[1].error == 0.5

        # m.limits setter
        m.limits = [(0.0, 10.0), (1.0, 5.0)]
        @test m.fmin === nothing
        @test m.params.pars[1].lower == 0.0
        @test m.params.pars[1].upper == 10.0
        @test m.params.pars[2].lower == 1.0
        @test m.params.pars[2].upper == 5.0

        # m.fixed setter
        m.fixed = [true, false]
        @test m.fmin === nothing
        @test m.params.pars[1].fixed == true
        @test m.params.pars[2].fixed == false

        # Length mismatches throw
        @test_throws DimensionMismatch (m.values = [1.0])
        @test_throws DimensionMismatch (m.errors = [0.1, 0.1, 0.1])
        @test_throws DimensionMismatch (m.limits = [(0.0, 1.0)])
        @test_throws DimensionMismatch (m.fixed = [true])
    end

end
