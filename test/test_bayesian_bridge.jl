# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test
using Random
using Statistics

_bayes_mean(v::AbstractVector) = sum(v) / length(v)
function _bayes_var(v::AbstractVector)
    μ = _bayes_mean(v)
    return sum((x - μ)^2 for x in v) / (length(v) - 1)
end

@testset "Bayesian bridge — posterior_sample / bayesian" begin
    @test !isdefined(JuMinuit, Symbol("bayesian!"))

    @testset "flat prior reproduces likelihood ensemble path" begin
        f(x) = ((x[1] - 1.0) / 0.25)^2 + ((x[2] + 0.5) / 0.4)^2
        m = Minuit(f, [0.2, 0.0]; names = ["x", "y"])
        migrad!(m)
        hesse!(m)
        ens = mcmc_sample(m; nsteps = 3_000, burn = 500, thin = 5,
                          seed = 11, warn = false)
        post = posterior_sample(m; prior = :flat, nchains = 1,
                                nsteps = 3_000, burn = 500, thin = 5,
                                seed = 11, warn = false)
        @test post isa PosteriorSample
        @test post.prior.name === :flat
        @test post.ensemble.samples == ens.samples
        @test post.ensemble.fvals == ens.fvals
        @test post.loglik_kept ≈ -ens.fvals ./ (2ens.up)
        @test post.logpost_kept == post.loglik_kept
        @test post.nchains == 1
        @test all(post.chain_ids .== 1)
    end

    @testset "flat prior reproduces likelihood path for nonstandard errordef" begin
        f(x) = ((x[1] - 0.3) / 0.2)^2
        m = Minuit(f, [0.0]; names = ["x"], errordef = 2.279)
        migrad!(m)
        hesse!(m)
        ens = mcmc_sample(m; proposal = [0.2], nsteps = 1_000,
                          burn = 200, thin = 10, seed = 123,
                          warn = false)
        post = posterior_sample(m; prior = :flat, proposal = [0.2],
                                nchains = 1, nsteps = 1_000,
                                burn = 200, thin = 10, seed = 123,
                                warn = false)
        @test post.ensemble.samples == ens.samples
        @test post.ensemble.fvals == ens.fvals
        @test post.loglik_kept ≈ -ens.fvals ./ (2ens.up)
    end

    @testset "works with a cost-function object (callable struct, not a Function)" begin
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]; ys = [0.5, 1.0, 1.5, 2.0, 2.5]; es = fill(0.1, 5)
        model(x, p) = p[1] + p[2] * x
        m = Minuit(LeastSquares(xs, ys, es, model), [0.0, 0.0]); migrad!(m); hesse!(m)
        @test !(m.fcn.f isa Function)                       # the FCN is a LeastSquares struct
        @test posterior_sample(m; prior = :flat, seed = 1, warn = false) isa PosteriorSample
        @test bayesian(m; warn = false) isa BayesianReport
        @test posterior_sample(m; sampler = :stretch, seed = 1, warn = false) isa PosteriorSample
    end

    @testset "isconsistent: true for unbounded params, false after a different fit" begin
        m = Minuit(x -> (x[1] - 1.0)^2, [0.0]; names = ["x"]); migrad!(m)
        prob = PosteriorProblem(m)
        @test isconsistent(prob, m)                  # NaN (open) bounds must not break ==
        m2 = Minuit(x -> (x[1] - 5.0)^2, [0.0]; names = ["x"]); migrad!(m2)
        @test !isconsistent(prob, m2)                # a different best ⇒ inconsistent
    end

    @testset "PosteriorProblem freezes the FCN (deepcopy snapshot)" begin
        xs = [0.0, 0.5, 1.0]; ys = [0.5, 1.5, 2.5]; es = fill(0.1, 3); model(x, p) = p[1] + p[2] * x
        c = LeastSquares(xs, ys, es, model)
        m = Minuit(c, [0.0, 0.0]); migrad!(m); hesse!(m)
        prob = PosteriorProblem(m)
        v0 = prob.fcn(prob.best)
        c.data.y .= 100.0                            # mutate the live cost data afterwards
        @test prob.fcn(prob.best) == v0              # the snapshot is unaffected
    end

    @testset "PosteriorProblem freezes the prior too (deepcopy)" begin
        f(x) = ((x[1] - 1.0) / 0.3)^2
        m = Minuit(f, [1.0]; names = ["x"]); migrad!(m); hesse!(m)
        μ = [0.0]
        names = collect(String.(m.parameters))
        pr = Prior(θ -> -0.5 * (θ[1] - μ[1])^2, :mutable, "prior over a mutable μ",
                   [-Inf], [Inf], names, true, [true])
        prob = PosteriorProblem(m; prior = pr)
        v0 = prob.prior.logdensity(prob.best)
        μ[1] = 10.0                                  # mutate the prior's captured state
        @test prob.prior.logdensity(prob.best) == v0 # snapshot unaffected
    end

    @testset "Gaussian likelihood times Gaussian prior matches analytic posterior" begin
        μL, σL = 1.0, 0.30
        μP, σP = 0.0, 0.20
        σpost2 = 1 / (1 / σL^2 + 1 / σP^2)
        μpost = σpost2 * (μL / σL^2 + μP / σP^2)

        f(x) = ((x[1] - μL) / σL)^2
        m = Minuit(f, [0.5]; names = ["x"])
        migrad!(m)
        hesse!(m)
        pr = normal_prior(m, :x, μP, σP)
        post = posterior_sample(m; prior = pr, nchains = 1,
                                nsteps = 22_000, burn = 2_000, thin = 20,
                                seed = 7, warn = false)
        vals = post.ensemble.samples[:, 1]
        @test _bayes_mean(vals) ≈ μpost atol = 0.03
        @test _bayes_var(vals) ≈ σpost2 rtol = 0.25
        ci = credible_interval(post, :x; level = 0.6827)
        @test ci[1] < μpost < ci[2]
        @test posterior_mean(post, :x) ≈ _bayes_mean(vals)
        @test posterior_median(post, :x) ≈ quantile(collect(vals), 0.5)
    end

    @testset "support is Minuit limits intersect prior support" begin
        f(x) = ((x[1] - 0.45) / 0.2)^2
        m = Minuit(f, [0.3]; names = ["g"], limits = [(0.0, nothing)])
        migrad!(m)
        hesse!(m)
        pr = uniform_prior(m, :g, 0.2, 0.8)
        post = posterior_sample(m; prior = pr, proposal = [0.12], scale = 1.0,
                                nchains = 1, nsteps = 8_000, burn = 1_000,
                                thin = 10, seed = 9, warn = false)
        @test all(0.2 .<= post.ensemble.samples[:, 1] .<= 0.8)
        lim = upper_limit(post, :g; level = 0.90)
        @test lim isa CredibleLimit
        @test lim.side === :upper
        @test 0.2 <= lim.limit <= 0.8
    end

    @testset "incompatible support fails loudly" begin
        f(x) = ((x[1] - 1.0) / 0.2)^2
        m = Minuit(f, [0.7]; names = ["x"])
        migrad!(m)
        hesse!(m)
        @test_throws ArgumentError uniform_prior(m, :x, 1.0, 1.0)
        pr = uniform_prior(m, :x, 2.0, 3.0)
        @test_throws ArgumentError PosteriorProblem(m; prior = pr)

        mb = Minuit(f, [0.7]; names = ["x"], limits = [(0.0, 10.0)])
        migrad!(mb)
        hesse!(mb)
        prb = uniform_prior(mb, :x, 0.0, 0.5)  # non-empty support, but excludes MLE
        @test_throws ArgumentError PosteriorProblem(mb; prior = prb)

        names = collect(String.(mb.parameters))
        point_prior = Prior(_ -> 0.0, :point, "point support for regression",
                            [1.0], [1.0], names, true, [true])
        @test_throws ArgumentError PosteriorProblem(mb; prior = point_prior)
    end

    @testset "a fixed parameter outside declared prior support fails loudly" begin
        f(x) = ((x[1] - 1.0) / 0.2)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.7, 2.0]; names = ["x", "y"], fixed = [false, true])
        migrad!(m); hesse!(m)
        names = collect(String.(m.parameters))
        # Declares y ∈ (−∞, 1.0] but y is fixed at 2.0 and the logdensity does
        # not itself enforce it ⇒ must be caught at construction, not sampled.
        bad = Prior(_ -> 0.0, :badfix, "bounded support on a fixed param",
                    [-Inf, -Inf], [Inf, 1.0], names, true, [false, true])
        @test_throws ArgumentError PosteriorProblem(m; prior = bad)
    end

    @testset "a mutating prior cannot corrupt the FCN point" begin
        f(x) = ((x[1] - 1.0) / 0.3)^2
        m = Minuit(f, [0.8]; names = ["x"])
        migrad!(m); hesse!(m)
        # A pathological prior that returns 0 (≡ flat) but overwrites its
        # argument in place. The kernel re-establishes the point before the FCN,
        # so the chain must stay byte-identical to the flat-prior chain.
        evil = Prior(θ -> (θ[1] = -999.0; 0.0), :evil, "mutating prior",
                     [-Inf], [Inf], ["x"], false, [false])
        kw = (; nchains = 1, nsteps = 3_000, burn = 500, thin = 10, seed = 8, warn = false)
        flat = posterior_sample(m; prior = :flat, kw...)
        mut  = posterior_sample(m; prior = evil, kw...)
        @test mut.ensemble.samples == flat.ensemble.samples
    end

    @testset "posterior sampling and bayesian report do not mutate Minuit state" begin
        f(x) = ((x[1] - 1.0) / 0.2)^2
        m = Minuit(f, [0.0]; names = ["x"])
        migrad!(m)
        hesse!(m)
        values0 = collect(m.values)
        errors0 = collect(m.errors)
        nfcn0 = m.nfcn
        post = posterior_sample(m; nchains = 1, nsteps = 1_000, burn = 200,
                                thin = 10, seed = 3, warn = false)
        @test collect(m.values) == values0
        @test collect(m.errors) == errors0
        @test m.nfcn == nfcn0
        report = bayesian(m; nchains = 1, nsteps = 1_000, burn = 200,
                          thin = 10, seed = 3, warn = false)
        @test report isa BayesianReport
        @test report.sample.ensemble.samples == post.ensemble.samples
        @test collect(m.values) == values0
        @test m.nfcn == nfcn0
        @test_throws ArgumentError bayesian(m; interval = :hpd, nchains = 1,
                                            nsteps = 100, burn = 10, thin = 10)

        calls_before_bad_level = m.nfcn
        @test_throws ArgumentError bayesian(m; level = 1.5, nchains = 1,
                                            nsteps = 100, burn = 10, thin = 10)
        @test m.nfcn == calls_before_bad_level
    end

    @testset "combine_priors is disjoint-only in the MVP" begin
        f(x) = (x[1] - 1)^2 + (x[2] + 2)^2
        m = Minuit(f, [0.0, 0.0]; names = ["a", "b"])
        p1 = normal_prior(m, :a, 0.0, 1.0)
        p2 = half_normal_prior(m, :b, 2.0)
        pc = combine_priors(p1, p2)
        @test pc.name === :combined
        @test pc.informative == [true, true]
        @test_throws ArgumentError combine_priors(p1, uniform_prior(m, :a, -1, 1))
    end

    @testset "prior helpers reject non-finite (improper) parameters" begin
        f(x) = (x[1] - 1)^2
        m = Minuit(f, [0.0]; names = ["x"])
        @test_throws ArgumentError normal_prior(m, :x, 0.0, Inf)    # σ = ∞ ⇒ flat
        @test_throws ArgumentError normal_prior(m, :x, Inf, 1.0)    # μ = ∞
        @test_throws ArgumentError uniform_prior(m, :x, -Inf, Inf)  # improper
        @test_throws ArgumentError uniform_prior(m, :x, 0.0, Inf)   # half-infinite
        @test_throws ArgumentError half_normal_prior(m, :x, Inf)    # σ = ∞ ⇒ flat
        @test normal_prior(m, :x, 0.0, 1.0) isa Prior              # finite ⇒ ok
    end

    @testset "Prior rejects invalid support endpoints (+Inf lower / -Inf upper / NaN)" begin
        @test_throws ArgumentError Prior(_ -> 0.0, :bad, "bad", [Inf], [Inf], ["x"], true, [true])
        @test_throws ArgumentError Prior(_ -> 0.0, :bad, "bad", [-Inf], [-Inf], ["x"], true, [true])
        @test_throws ArgumentError Prior(_ -> 0.0, :bad, "bad", [NaN], [1.0], ["x"], true, [true])
        @test Prior(_ -> 0.0, :ok, "ok", [-Inf], [Inf], ["x"], false, [false]) isa Prior   # unbounded ok
        @test Prior(_ -> 0.0, :ok, "ok", [0.0], [1.0], ["x"], true, [true]) isa Prior       # finite box ok
    end

    @testset "non-finite errordef / sampler config is rejected (not silently degenerate)" begin
        f(x) = ((x[1] - 1.0) / 0.3)^2
        m = Minuit(f, [0.0]; names = ["x"]); migrad!(m); hesse!(m)
        @test_throws ArgumentError posterior_sample(m; scale = Inf, warn = false)
        @test_throws ArgumentError posterior_sample(m; overdisperse = Inf, warn = false)
        @test_throws ArgumentError posterior_sample(m; sampler = :stretch, stretch = Inf, warn = false)
        @test_throws ArgumentError posterior_sample(m; proposal = fill(Inf, 1, 1), warn = false)
        @test_throws ArgumentError mcmc_sample(m; scale = Inf, warn = false)
        m.up = Inf                                       # improper target (no likelihood term)
        @test_throws ArgumentError PosteriorProblem(m)
        @test_throws ArgumentError mcmc_sample(m; warn = false)
    end

    @testset "derived intervals are sample-wise quantiles" begin
        f(x) = ((x[1] - 1.0) / 0.3)^2
        m = Minuit(f, [0.2]; names = ["x"])
        migrad!(m)
        hesse!(m)
        post = posterior_sample(m; nchains = 1, nsteps = 4_000, burn = 500,
                                thin = 10, seed = 5, warn = false)
        lo, hi = derived_interval(post, θ -> θ[1]^2; level = 0.80)
        vals = sort!([θ[1]^2 for θ in post])
        @test lo ≈ quantile(vals, 0.1; sorted = true)
        @test hi ≈ quantile(vals, 0.9; sorted = true)
        @test_throws ArgumentError derived_interval(post, θ -> θ[1]; method = :hpd)
    end

    @testset "multi-chain diagnostics and burn-in adaptation smoke" begin
        f(x) = ((x[1] - 1.0) / 0.25)^2
        m = Minuit(f, [0.0]; names = ["x"])
        migrad!(m)
        hesse!(m)
        post = posterior_sample(m; nchains = 2, nsteps = 2_000, burn = 500,
                                thin = 10, seed = 13, scale = 2.0,
                                target_accept = 0.25, warn = false)
        @test post.nchains == 2
        @test length(unique(post.chain_ids)) == 2
        @test length(post.rhat) == 1 && isfinite(post.rhat[1])
        @test length(post.ess) == 1 && post.ess[1] > 0
        @test post.ensemble.scale != 2.0
    end

    @testset "chain starts are genuinely over-dispersed (scale with overdisperse)" begin
        # Starts must be ~`overdisperse`·σ wide, not ~one proposal scale, or the
        # multi-chain split-R̂ is not a real convergence test.
        best = [0.0, 0.0]; steps = [1.0, 2.0]
        lo = [NaN, NaN]; hi = [NaN, NaN]
        meandist(disp) = begin
            rng = Random.MersenneTwister(20)
            s = 0.0
            for _ in 1:4000
                q = JuMinuit._dispersed_start(best, rng, steps, nothing, lo, hi, disp)
                s += abs(q[1])              # coord 1 has σ = 1
            end
            s / 4000
        end
        d_default = meandist(2.0)           # ≈ 2σ·E|N(0,1)| = 2·0.798 ≈ 1.6
        d_tight   = meandist(0.3)           # the old (under-dispersed) magnitude
        @test 1.2 < d_default < 2.1
        @test d_default > 4 * d_tight       # 2.0 vs 0.3 ⇒ ~6.7× wider
    end

    @testset "multi-chain dispersed starts require finite posterior" begin
        f(x) = x[1] < 0 ? Inf : ((x[1] - 0.2) / 0.1)^2
        m = Minuit(f, [0.4]; names = ["x"])
        migrad!(m)
        hesse!(m)
        @test_nowarn posterior_sample(m; nchains = 2, proposal = [1.0],
                                      scale = 10.0, nsteps = 100,
                                      burn = 20, thin = 10, seed = 3,
                                      warn = false)
    end

    @testset "affine-invariant ensemble (sampler = :stretch)" begin
        f(x) = ((x[1] - 1.0) / 0.5)^2 + ((x[2] + 0.5) / 0.3)^2
        m = Minuit(f, [0.0, 0.0]; names = ["a", "b"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :stretch, seed = 7, warn = false)
        @test post isa PosteriorSample
        @test post.sampler === :stretch
        @test post.nchains == 8                       # default nwalkers = max(2·nfree+2, 8)
        @test sort(unique(post.chain_ids)) == collect(1:8)
        # recovers the analytic posterior mean/std
        A = post.ensemble.samples
        @test _bayes_mean(A[:, 1]) ≈ 1.0 atol = 0.06
        @test _bayes_mean(A[:, 2]) ≈ -0.5 atol = 0.05
        @test sqrt(_bayes_var(A[:, 1])) ≈ 0.5 rtol = 0.15
        @test sqrt(_bayes_var(A[:, 2])) ≈ 0.3 rtol = 0.15
        @test all(isfinite, post.rhat) && maximum(post.rhat) < 1.05
        @test all(>(0), post.ess)
        # fvals remain likelihood FCN values; logpost = loglik (flat prior)
        @test post.logpost_kept ≈ post.loglik_kept
        @test post.loglik_kept ≈ -post.ensemble.fvals ./ (2 * post.ensemble.up)
    end

    @testset "ensemble is affine-invariant: recovers strong correlation" begin
        # ρ = 0.95 Gaussian — a single random-walk chain mixes poorly here; the
        # stretch move is affine-invariant and should recover it.
        s = 0.4; rho = 0.95
        Ci = inv([s^2 rho*s^2; rho*s^2 s^2])
        g(x) = x[1]*(Ci[1,1]*x[1] + Ci[1,2]*x[2]) + x[2]*(Ci[2,1]*x[1] + Ci[2,2]*x[2])
        m = Minuit(g, [0.0, 0.0]; names = ["x", "y"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :stretch, seed = 3, warn = false)
        C = cov(post.ensemble.samples)
        @test C[1, 2] / sqrt(C[1, 1] * C[2, 2]) ≈ 0.95 atol = 0.04
    end

    @testset "ensemble: gradient-free works on a non-differentiable FCN" begin
        h(x) = abs(x[1] - 2.0) / 0.3            # kinked, not differentiable at x=2
        m = Minuit(h, [2.0]; names = ["k"]); migrad!(m); hesse!(m)
        post = posterior_sample(m; sampler = :stretch, seed = 5, warn = false)
        @test posterior_median(post, :k) ≈ 2.0 atol = 0.1
    end

    @testset "ensemble: reproducible + validates nwalkers" begin
        f(x) = ((x[1] - 1.0) / 0.4)^2
        m = Minuit(f, [0.0]; names = ["x"]); migrad!(m); hesse!(m)
        p1 = posterior_sample(m; sampler = :stretch, seed = 11, warn = false)
        p2 = posterior_sample(m; sampler = :stretch, seed = 11, warn = false)
        @test p1.ensemble.samples == p2.ensemble.samples
        @test_throws ArgumentError posterior_sample(m; sampler = :stretch, nwalkers = 3)
        @test_throws ArgumentError posterior_sample(m; sampler = :gibbs)
        # odd nwalkers is bumped to even, not rejected
        podd = posterior_sample(m; sampler = :stretch, nwalkers = 7, seed = 1, warn = false)
        @test podd.nchains == 8
    end

    @testset "ensemble: nwalkers must exceed n_free (affine-hull span)" begin
        # The stretch move keeps proposals inside the ensemble's affine hull, so
        # nwalkers ≤ n_free can only sample a subspace — must throw, not warn.
        f(x) = sum(abs2, x .- 1.0)                        # 4 free parameters
        m = Minuit(f, zeros(4); names = ["a", "b", "c", "d"]); migrad!(m); hesse!(m)
        @test_throws ArgumentError posterior_sample(m; sampler = :stretch, nwalkers = 4)
        ok = posterior_sample(m; sampler = :stretch, nwalkers = 10, seed = 1, warn = false)
        @test all(>(0.5), vec(sqrt.(_bayes_var.(eachcol(ok.ensemble.samples)))))  # all 4 dims sampled
    end

    @testset "ensemble throws on a collapsed (rank-deficient) init, not silent point mass" begin
        # A support so tight that no walker can be over-dispersed into it ⇒ the
        # ensemble would collapse onto the best fit; the stretch move could never
        # escape, so construction must fail loudly instead.
        f(x) = sum(abs2, x)
        m = Minuit(f, [0.0, 0.0]; names = ["a", "b"]); migrad!(m); hesse!(m)
        tiny = combine_priors(uniform_prior(m, :a, -1e-300, 1e-300),
                              uniform_prior(m, :b, -1e-300, 1e-300))
        @test_throws ArgumentError posterior_sample(m; sampler = :stretch, prior = tiny,
                                                    nwalkers = 8, seed = 1, warn = false)
    end

    @testset "random-walk-only knobs are ignored by :stretch (no spurious throw)" begin
        f(x) = ((x[1] - 1.0) / 0.4)^2
        m = Minuit(f, [0.0]; names = ["x"]); migrad!(m); hesse!(m)
        @test posterior_sample(m; sampler = :stretch, scale = -1.0, overdisperse = -2.0,
                               seed = 1, warn = false) isa PosteriorSample
    end

    @testset "boundary_active flags a mode sitting at a limit" begin
        fb(x) = ((x[1] + 0.3) / 0.5)^2           # data prefer -0.3, limited ≥ 0 ⇒ best at 0
        m = Minuit(fb, [0.1]; names = ["mu"], limits = [(0.0, nothing)]); migrad!(m); hesse!(m)
        post = posterior_sample(m; prior = :flat, proposal = [0.5], seed = 2, warn = false)
        @test upper_limit(post, :mu; level = 0.90).boundary_active        # detected
        fi(x) = ((x[1] - 2.0) / 0.3)^2           # mode well inside the limit
        mi = Minuit(fi, [2.0]; names = ["g"], limits = [(0.0, nothing)]); migrad!(mi); hesse!(mi)
        @test !any(posterior_sample(mi; prior = :flat, seed = 3, warn = false).boundary_active)
    end

    @testset "_boundary_flags: analytic half-normal at a limit fires (deterministic)" begin
        # Population-limit check (no MCMC noise): a pure half-normal whose mode is
        # exactly on the limit MUST flag; a mode many σ inside must not.
        Random.seed!(0)
        half = reshape(abs.(randn(40_000)) .* 0.5, :, 1)            # mode at lower limit 0
        @test JuMinuit._boundary_flags(half, [1], [0.0], [NaN])[1]
        interior = reshape(3.0 .+ 0.5 .* randn(40_000), :, 1)       # mode 6σ inside
        @test !JuMinuit._boundary_flags(interior, [1], [0.0], [NaN])[1]
        upperpile = reshape(5.0 .- abs.(randn(40_000)) .* 0.5, :, 1)  # mode at upper limit 5
        @test JuMinuit._boundary_flags(upperpile, [1], [NaN], [5.0])[1]
    end

    @testset "degenerate diagnostics fail visibly" begin
        samples = [0.0; 0.0; 1.0; 1.0;;]
        chains = [1, 1, 2, 2]
        rh = JuMinuit._posterior_rhat(samples, chains, [1], 2)
        @test isinf(rh[1])
        same = JuMinuit._posterior_rhat(fill(1.0, 4, 1), chains, [1], 2)
        @test isnan(same[1])
        @test JuMinuit._ess_one([1.0, 1.0, 1.0, 1.0]) == 0.0
    end

    @testset "PosteriorSample constructor validates diagnostic shapes" begin
        samples = reshape([0.0, 1.0], :, 1)
        ens = LikelihoodEnsemble(samples, [0.0, 1.0], ["x"], [true], [0.0],
                                 0.0, 1.0, 0.5, 2, 0, 1, 1.0, :steps, nothing)
        pr = Prior(_ -> 0.0, :flat, "flat", [-Inf], [Inf], ["x"], false, [false])
        @test_throws DimensionMismatch PosteriorSample(ens, pr, :metropolis, 1,
                                                       [1, 1], [0.0, -0.5],
                                                       [0.0, -0.5], Float64[],
                                                       [2.0], [false], String[])
        @test_throws DimensionMismatch PosteriorSample(ens, pr, :metropolis, 1,
                                                       [1, 1], [0.0, -0.5],
                                                       [0.0, -0.5], [NaN],
                                                       Float64[], [false], String[])
        @test_throws DimensionMismatch PosteriorSample(ens, pr, :metropolis, 1,
                                                       [1, 1], [0.0, -0.5],
                                                       [0.0, -0.5], [NaN],
                                                       [2.0], Bool[], String[])
    end
end
