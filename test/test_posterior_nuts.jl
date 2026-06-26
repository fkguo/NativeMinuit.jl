# SPDX-License-Identifier: LGPL-2.1-or-later
#
# NUTS posterior sampler — the AdvancedHMC extension (`sampler = :nuts`).
# The extension deps are declared in the test target, so the extension MUST load
# here; a load or API error FAILS the suite (it is not silently skipped) so CI
# catches a broken extension.

using JuMinuit
using Test
using Statistics
using Random
using AdvancedHMC, LogDensityProblems, LogDensityProblemsAD, TransformVariables, ForwardDiff

@testset "NUTS posterior sampler (AdvancedHMC extension)" begin
    @test Base.get_extension(JuMinuit, :JuMinuitAdvancedHMCExt) !== nothing
    @test isdefined(JuMinuit, :_posterior_sample_nuts)

    @testset "unbounded posterior recovery" begin
        f(x) = ((x[1] - 1.0) / 0.5)^2 + ((x[2] + 0.5) / 0.3)^2
        m = Minuit(f, [0.0, 0.0]; names = ["a", "b"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :nuts, seed = 1, warn = false)
        @test post isa PosteriorSample
        @test post.sampler === :nuts
        A = post.ensemble.samples
        @test mean(A[:, 1]) ≈ 1.0 atol = 0.05
        @test mean(A[:, 2]) ≈ -0.5 atol = 0.04
        @test std(A[:, 1]) ≈ 0.5 rtol = 0.15
        @test std(A[:, 2]) ≈ 0.3 rtol = 0.15
        @test all(isfinite, post.rhat) && maximum(post.rhat) < 1.05
        @test post.logpost_kept ≈ post.loglik_kept                       # flat prior
        @test post.loglik_kept ≈ -post.ensemble.fvals ./ (2 * post.ensemble.up)
    end

    @testset "NUTS is reproducible with a fixed seed" begin
        f(x) = ((x[1] - 1.0) / 0.4)^2 + ((x[2] + 0.5) / 0.3)^2
        m = Minuit(f, [0.0, 0.0]; names = ["a", "b"]); migrad!(m); hesse!(m)
        p1 = posterior_sample(m; sampler = :nuts, seed = 42, warn = false)
        p2 = posterior_sample(m; sampler = :nuts, seed = 42, warn = false)
        @test p1.ensemble.samples == p2.ensemble.samples      # incl. find_good_stepsize RNG
        @test p1.ensemble.fvals == p2.ensemble.fvals
        @test p1.logpost_kept == p2.logpost_kept
    end

    @testset "NUTS honors `thin`" begin
        f(x) = ((x[1] - 1.0) / 0.4)^2
        m = Minuit(f, [0.0]; names = ["x"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :nuts, nsteps = 400, burn = 200,
                                thin = 4, nchains = 2, seed = 1, warn = false)
        @test post.ensemble.thin == 4
        @test length(post) == 2 * ((400 - 200) ÷ 4)        # thinned row count
    end

    @testset "prior log-densities are ForwardDiff-generic (NUTS needs them)" begin
        m = Minuit(x -> (x[1] - 1.0)^2, [1.0]; names = ["x"]); migrad!(m)
        for pr in (normal_prior(m, :x, 0.0, 1.0), uniform_prior(m, :x, -3.0, 3.0),
                   half_normal_prior(m, :x, 1.0), combine_priors(normal_prior(m, :x, 0.0, 1.0)))
            @test all(isfinite, ForwardDiff.gradient(θ -> pr.logdensity(θ), [0.5]))
        end
    end

    @testset "NUTS with an informative Gaussian prior — conjugate posterior" begin
        μL, σL = 1.0, 0.30; μP, σP = 0.0, 0.20
        σpost2 = 1 / (1 / σL^2 + 1 / σP^2); μpost = σpost2 * (μL / σL^2 + μP / σP^2)
        f(x) = ((x[1] - μL) / σL)^2
        m = Minuit(f, [0.5]; names = ["x"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :nuts, prior = normal_prior(m, :x, μP, σP),
                                seed = 6, warn = false)
        vals = post.ensemble.samples[:, 1]
        @test mean(vals) ≈ μpost atol = 0.02
        @test std(vals) ≈ sqrt(σpost2) rtol = 0.12
    end

    @testset "NUTS: a mutating prior cannot corrupt the FCN (separate buffers)" begin
        f(x) = ((x[1] - 1.0) / 0.3)^2
        m = Minuit(f, [1.0]; names = ["x"]); migrad!(m); hesse!(m)
        # returns 0 (≡ flat) but overwrites its argument; the FCN must still be
        # evaluated at the true point, so the posterior recovers N(1, 0.3).
        evil = Prior(θ -> (θ[1] = -999.0; 0.0), :evil, "mutating", [-Inf], [Inf], ["x"], false, [false])
        post = posterior_sample(m; sampler = :nuts, prior = evil, seed = 9, warn = false)
        @test mean(post.ensemble.samples[:, 1]) ≈ 1.0 atol = 0.05
        @test std(post.ensemble.samples[:, 1]) ≈ 0.3 rtol = 0.15
    end

    @testset "JACOBIAN GATE: bounded [0,∞) NUTS matches metropolis" begin
        # If the unconstraining transform's log-Jacobian were wrong, NUTS would
        # sample a biased posterior — so it MUST agree with the independent
        # (transform-free, rejection-based) metropolis sampler.
        g(x) = ((x[1] - 2.0) / 0.5)^2
        m = Minuit(g, [2.0]; names = ["k"], limits = [(0.0, nothing)]); migrad!(m); hesse!(m)
        pn = posterior_sample(m; sampler = :nuts, seed = 2, warn = false)
        pm = posterior_sample(m; sampler = :metropolis, seed = 2, warn = false)
        @test mean(pn.ensemble.samples[:, 1]) ≈ mean(pm.ensemble.samples[:, 1]) atol = 0.06
        @test std(pn.ensemble.samples[:, 1]) ≈ std(pm.ensemble.samples[:, 1]) rtol = 0.15
        @test all(pn.ensemble.samples[:, 1] .>= 0.0)                      # respects the limit
    end

    @testset "JACOBIAN GATE: box [1,3] NUTS matches metropolis" begin
        g(x) = ((x[1] - 2.0) / 0.4)^2
        m = Minuit(g, [2.0]; names = ["p"], limits = [(1.0, 3.0)]); migrad!(m); hesse!(m)
        pn = posterior_sample(m; sampler = :nuts, seed = 3, warn = false)
        pm = posterior_sample(m; sampler = :metropolis, seed = 3, warn = false)
        @test mean(pn.ensemble.samples[:, 1]) ≈ mean(pm.ensemble.samples[:, 1]) atol = 0.05
        @test all(1.0 .<= pn.ensemble.samples[:, 1] .<= 3.0)
    end

    @testset "BinnedNLL: NUTS matches metropolis (AD-generic edge buffer)" begin
        # The binned cost is now ForwardDiff-differentiable (its `_edge_cdf` buffer
        # promotes to the parameter type), so `:nuts` runs on it. The posterior must
        # agree with the gradient-free, transform-free metropolis sampler — the same
        # cross-check as the JACOBIAN GATE tests, now driving AD through `_edge_cdf`.
        edges = collect(0.0:0.5:6.0)
        cdfexp(x, p) = 1 - exp(-p[1] * x)
        probs = [cdfexp(edges[i+1], [0.8]) - cdfexp(edges[i], [0.8])
                 for i in 1:length(edges)-1]
        counts = round.(probs ./ sum(probs) .* 2000)
        c = BinnedNLL(counts, edges, cdfexp)
        m = Minuit(c, [1.0]; limits = [(1e-6, 50.0)], tol = 1e-5); migrad!(m); hesse!(m)

        pn = posterior_sample(m; sampler = :nuts, seed = 7, warn = false)
        pm = posterior_sample(m; sampler = :metropolis, seed = 7, warn = false)
        @test pn isa PosteriorSample && pn.sampler === :nuts
        vn = pn.ensemble.samples[:, 1]
        vm = pm.ensemble.samples[:, 1]
        @test abs(mean(vn) - mean(vm)) < 0.3 * std(vm)     # agree to <0.3 posterior-σ
        @test std(vn) ≈ std(vm) rtol = 0.25
        @test all(vn .> 0.0)                                # respects the [1e-6, 50] limit
    end

    @testset "non-differentiable FCN errors (no finite-difference fallback)" begin
        buf = zeros(Float64, 1)
        hbad(x) = (buf[1] = x[1]; (buf[1] - 1.0)^2)    # Dual → Float64 buffer breaks AD
        m = Minuit(hbad, [1.0]; names = ["x"]); migrad!(m); hesse!(m)
        @test_throws ArgumentError posterior_sample(m; sampler = :nuts, seed = 4, warn = false)
    end

    @testset "best-on-boundary is handled (error or finite restart)" begin
        gb(x) = ((x[1] + 5.0) / 0.3)^2                  # MLE far below 0 ⇒ best pinned at 0
        m = Minuit(gb, [0.1]; names = ["q"], limits = [(0.0, nothing)]); migrad!(m); hesse!(m)
        if m.values[1] == 0.0
            @test_throws ArgumentError posterior_sample(m; sampler = :nuts, seed = 5, warn = false)
        else
            @test posterior_sample(m; sampler = :nuts, seed = 5, warn = false) isa PosteriorSample
        end
    end
end
