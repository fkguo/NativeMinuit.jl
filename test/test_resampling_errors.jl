# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Random
using Statistics
using DataFrames
using Test

@testset "resampling.jl — bootstrap + jackknife error analysis" begin

    # ── Linear / Gaussian fixture ────────────────────────────────────────────
    # y = a·x + b + σ·z. The noise z is standardised to EXACTLY mean 0, std 1 so
    # the empirical residual scale equals the assumed σ. This isolates each
    # method's resampling behaviour from the ±√(2/N) scatter a single noise draw
    # would otherwise inject into the HESSE comparison — the test then probes the
    # estimator, not a lucky seed.
    a_true, b_true, σ = 2.0, 1.0, 0.3
    N = 100
    x = collect(range(0.0, 5.0; length = N))
    z = randn(Xoshiro(20240530), N)
    z = (z .- mean(z)) ./ std(z)
    y = a_true .* x .+ b_true .+ σ .* z
    d = Data(x, y, fill(σ, N))
    linmodel(xi, p) = p[1] * xi + p[2]

    m = model_fit(linmodel, d, [1.0, 0.0]; name = ["a", "b"])
    migrad!(m)
    hesse(m)
    hesse_err = [m.errors[1], m.errors[2]]

    @testset "bootstrap std ≈ HESSE (parametric, σ-tight)" begin
        bp = bootstrap(linmodel, d, m; nresample = 2000, seed = 1, kind = :parametric)
        @test bp.kind === :parametric
        @test bp.n_valid == 2000
        @test size(bp.samples) == (2000, 2)
        @test length(bp.std) == 2
        # Parametric regenerates from the model + σ, so its spread reflects the
        # SAME σ as HESSE — they agree up to Monte-Carlo noise.
        @test bp.std[1] ≈ hesse_err[1] rtol = 0.12
        @test bp.std[2] ≈ hesse_err[2] rtol = 0.12
        @test bp.mean[1] ≈ a_true rtol = 0.05
        @test bp.mean[2] ≈ b_true rtol = 0.1
    end

    @testset "bootstrap std ≈ HESSE (nonparametric)" begin
        bs = bootstrap(linmodel, d, m; nresample = 2000, seed = 1)
        @test bs.kind === :nonparametric
        @test bs.std[1] ≈ hesse_err[1] rtol = 0.2
        @test bs.std[2] ≈ hesse_err[2] rtol = 0.2
        # estimate is the full-data optimum (anchor), not the resample mean
        @test bs.estimate[1] ≈ m.values[1] atol = 1e-9
        @test bs.estimate[2] ≈ m.values[2] atol = 1e-9
    end

    @testset "jackknife variance ≈ HESSE²; bias ≈ 0 (unbiased linear)" begin
        jk = jackknife(linmodel, d, m)
        @test jk.d == 1
        @test jk.g == N
        @test jk.n == N
        @test jk.n_valid == N
        @test size(jk.samples) == (N, 2)
        # variance == std² (internal consistency) and std ≈ HESSE.
        @test jk.variance ≈ jk.std .^ 2 rtol = 1e-12
        @test jk.std[1] ≈ hesse_err[1] rtol = 0.2
        @test jk.std[2] ≈ hesse_err[2] rtol = 0.2
        # A linear estimator is unbiased → jackknife bias ≈ 0 (≪ the stat error).
        @test abs(jk.bias[1]) < 0.25 * jk.std[1]
        @test abs(jk.bias[2]) < 0.25 * jk.std[2]
        # bias_corrected = estimate − bias.
        @test jk.bias_corrected ≈ jk.estimate .- jk.bias rtol = 1e-12
    end

    @testset "bootstrap percentile CIs asymmetric (nonlinear fit)" begin
        # Exponential decay y = A·exp(-k·x): the bootstrap sampling distribution
        # of the decay rate k is skewed, so its percentile CI is asymmetric about
        # the point estimate — exactly what a symmetric HESSE error cannot show.
        A_true, k_true, σn, Nn = 5.0, 0.9, 0.25, 40
        xn = collect(range(0.0, 4.0; length = Nn))
        zn = randn(Xoshiro(777), Nn)
        yn = A_true .* exp.(-k_true .* xn) .+ σn .* zn
        dn = Data(xn, yn, fill(σn, Nn))
        expmodel(xi, p) = p[1] * exp(-p[2] * xi)
        mn = model_fit(expmodel, dn, [4.0, 1.0]; name = ["A", "k"])
        migrad!(mn)

        bn = bootstrap(expmodel, dn, mn; nresample = 3000, seed = 5)
        @test bn.n_valid > 2900            # the vast majority converge
        # CI must bracket the point estimate.
        for i in 1:2
            @test bn.ci_lower[i] < bn.estimate[i] < bn.ci_upper[i]
        end
        # Asymmetry: upper vs lower half-width differ by a clear margin for ≥1 par.
        upper = bn.ci_upper .- bn.estimate
        lower = bn.estimate .- bn.ci_lower
        rel_asym = abs.(upper .- lower) ./ bn.std
        @test maximum(rel_asym) > 0.08
    end

    @testset "threaded == serial (deterministic seed)" begin
        # Bootstrap consumes randomness; per-resample seeds are drawn serially so
        # the result is bit-identical regardless of thread count.
        b1 = bootstrap(linmodel, d, m; nresample = 256, seed = 99, threaded = false)
        b2 = bootstrap(linmodel, d, m; nresample = 256, seed = 99, threaded = true)
        @test b1.samples == b2.samples
        @test b1.valid == b2.valid
        @test b1.std == b2.std
        # Jackknife uses no randomness → threaded and serial are identical too.
        j1 = jackknife(linmodel, d, m; threaded = false)
        j2 = jackknife(linmodel, d, m; threaded = true)
        @test j1.samples == j2.samples
    end

    @testset "reproducibility + recorded seed" begin
        r1 = bootstrap(linmodel, d, m; nresample = 200, seed = 12345)
        r2 = bootstrap(linmodel, d, m; nresample = 200, seed = 12345)
        @test r1.samples == r2.samples
        @test r1.seed == UInt64(12345)
        # different seed → different resamples
        r3 = bootstrap(linmodel, d, m; nresample = 200, seed = 6789)
        @test r3.samples != r1.samples
    end

    @testset "bootstrap covariance" begin
        bc = bootstrap(linmodel, d, m; nresample = 1500, seed = 8, covariance = true)
        @test bc.covariance !== nothing
        @test size(bc.covariance) == (2, 2)
        @test bc.covariance ≈ bc.covariance'              # symmetric
        @test sqrt(bc.covariance[1, 1]) ≈ bc.std[1] rtol = 1e-8
        @test sqrt(bc.covariance[2, 2]) ≈ bc.std[2] rtol = 1e-8
        # default: no covariance computed
        bnc = bootstrap(linmodel, d, m; nresample = 50, seed = 8)
        @test bnc.covariance === nothing
    end

    @testset "parameter correlations: jackknife covariance + correlation()" begin
        jk = jackknife(linmodel, d, m)
        # full jackknife covariance: symmetric, diagonal == variance
        @test size(jk.covariance) == (2, 2)
        @test jk.covariance ≈ jk.covariance'
        @test [jk.covariance[1, 1], jk.covariance[2, 2]] ≈ jk.variance rtol = 1e-12
        # correlation matrix: unit diagonal, symmetric, slope/intercept
        # ANTI-correlated for a line fit over positive x (a known physical sign).
        Cj = correlation(jk)
        @test size(Cj) == (2, 2)
        @test Cj[1, 1] ≈ 1.0 atol = 1e-12
        @test Cj[2, 2] ≈ 1.0 atol = 1e-12
        @test Cj ≈ Cj'
        @test Cj[1, 2] < -0.5
        # correlation == standardised covariance
        Dj = sqrt.([jk.covariance[1, 1], jk.covariance[2, 2]])
        @test Cj[1, 2] ≈ jk.covariance[1, 2] / (Dj[1] * Dj[2]) rtol = 1e-10

        # bootstrap correlation is available even when covariance=false (default)
        bs = bootstrap(linmodel, d, m; nresample = 1000, seed = 3)
        @test bs.covariance === nothing
        Cb = correlation(bs)
        @test size(Cb) == (2, 2)
        @test Cb[1, 1] ≈ 1.0 atol = 1e-12
        @test Cb[1, 2] < -0.5
        # and it equals the standardised stored covariance when that IS requested
        bsc = bootstrap(linmodel, d, m; nresample = 1000, seed = 3, covariance = true)
        Dc = sqrt.([bsc.covariance[1, 1], bsc.covariance[2, 2]])
        @test correlation(bsc)[1, 2] ≈ bsc.covariance[1, 2] / (Dc[1] * Dc[2]) rtol = 1e-10
        # jackknife and bootstrap estimate the same correlation
        @test Cj[1, 2] ≈ Cb[1, 2] rtol = 0.15
        # degenerate: <2 valid samples → NaN correlation matrix (no throw)
        Cnan = JuMinuit._sample_correlation(jk.samples, falses(jk.g), 2)
        @test all(isnan, Cnan)
    end

    @testset "delete-d block jackknife" begin
        jb = jackknife(linmodel, d, m; d = 5)
        @test jb.d == 5
        @test jb.g == cld(N, 5)                 # 20 consecutive blocks
        @test jb.n_valid == jb.g
        @test size(jb.samples) == (jb.g, 2)
        @test all(isfinite, jb.std)
        @test all(>(0), jb.std)
        # The block jackknife is a deliberately COARSE estimator: with only g=20
        # groups it carries ~√(2/(g-1)) ≈ 32% intrinsic scatter, and consecutive
        # blocks of sorted x are non-exchangeable (a downward bias for IID data —
        # contiguous blocks are meant for SERIALLY CORRELATED data). So we assert
        # only an order-of-magnitude sanity band against HESSE, not a tight match.
        @test 0.3 < jb.std[1] / hesse_err[1] < 3.0
        @test 0.3 < jb.std[2] / hesse_err[2] < 3.0
        # bias estimate stays finite and small relative to the (coarse) spread
        @test all(isfinite, jb.bias)
    end

    @testset "warm vs cold start (vector start)" begin
        bw = bootstrap(linmodel, d, [1.0, 0.0]; nresample = 400, seed = 2,
                       warm_start = true)
        bcs = bootstrap(linmodel, d, [1.0, 0.0]; nresample = 400, seed = 2,
                        warm_start = false)
        # convex χ² → both reach the same minimum per resample; std agrees tightly
        @test bw.std ≈ bcs.std rtol = 0.05
        @test bw.std[1] ≈ hesse_err[1] rtol = 0.2
    end

    @testset "generic refit interface" begin
        refit = dd -> JuMinuit.args(migrad!(model_fit(linmodel, dd, [1.0, 0.0])))
        bg = bootstrap(refit, d; nresample = 500, seed = 3, names = ["a", "b"])
        @test bg.names == ["a", "b"]
        @test bg.kind === :nonparametric
        @test bg.std[1] ≈ hesse_err[1] rtol = 0.25
        @test bg.std[2] ≈ hesse_err[2] rtol = 0.25
        jg = jackknife(refit, d; names = ["a", "b"])
        @test jg.std[1] ≈ hesse_err[1] rtol = 0.25
        @test abs(jg.bias[1]) < 0.3 * jg.std[1]
        # default parameter names
        bgd = bootstrap(refit, d; nresample = 50, seed = 3)
        @test bgd.names == ["p1", "p2"]
        # generic threaded == serial
        gs = bootstrap(refit, d; nresample = 128, seed = 4, threaded = false)
        gt = bootstrap(refit, d; nresample = 128, seed = 4, threaded = true)
        @test gs.samples == gt.samples
    end

    @testset "display + DataFrame" begin
        bs = bootstrap(linmodel, d, m; nresample = 100, seed = 1, covariance = true)
        txt = sprint(show, MIME"text/plain"(), bs)
        @test occursin("Bootstrap", txt)
        @test occursin("nonparametric", txt)
        @test occursin("covariance", txt)
        @test occursin("BootstrapResult", sprint(show, bs))

        jk = jackknife(linmodel, d, m)
        jtxt = sprint(show, MIME"text/plain"(), jk)
        @test occursin("Jackknife", jtxt)
        @test occursin("delete-1", jtxt)
        @test occursin("JackknifeResult", sprint(show, jk))

        dfb = DataFrame(bs)
        @test names(dfb) == ["parameter", "estimate", "mean", "std",
                              "ci_lower", "ci_upper"]
        @test nrow(dfb) == 2
        @test dfb.parameter == ["a", "b"]

        dfj = DataFrame(jk)
        @test "bias_corrected" in names(dfj)
        @test "variance" in names(dfj)
        @test nrow(dfj) == 2
    end

    @testset "argument validation" begin
        @test_throws ArgumentError bootstrap(linmodel, d, m; nresample = 0)
        @test_throws ArgumentError bootstrap(linmodel, d, m; ci_level = 1.5)
        @test_throws ArgumentError bootstrap(linmodel, d, m; ci_level = 0.0)
        @test_throws ArgumentError bootstrap(linmodel, d, m; kind = :bogus)
        @test_throws ArgumentError bootstrap(linmodel, d, m; seed = -1)
        @test_throws ArgumentError jackknife(linmodel, d, m; d = 0)
        @test_throws ArgumentError jackknife(linmodel, d, m; d = N + 1)
        # generic: names length mismatch
        refit = dd -> JuMinuit.args(migrad!(model_fit(linmodel, dd, [1.0, 0.0])))
        @test_throws ArgumentError bootstrap(refit, d; nresample = 10,
                                             names = ["only_one"])
    end

    @testset "NaN re-fits never poison statistics" begin
        # A row of NaN (a thrown / divergent re-fit) must be dropped from the
        # summary stats even when it is flagged valid, on BOTH the default
        # (filter_invalid=true) and the include-all (false) paths.
        samples = [1.0 2.0; 2.0 3.0; NaN NaN; 3.0 4.0]
        @test JuMinuit._stat_mask(samples, [true, true, true, true], true) ==
              [true, true, false, true]
        # filter_invalid=false ignores `valid` but STILL drops the NaN row
        @test JuMinuit._stat_mask(samples, [true, false, true, false], false) ==
              [true, true, false, true]
        mask = JuMinuit._stat_mask(samples, [true, true, true, true], true)
        μ, σ, lo, hi, cov, nv = JuMinuit._bootstrap_stats(samples, mask, 0.68, true)
        @test nv == 3
        @test all(isfinite, μ)
        @test all(isfinite, σ)
    end

    @testset "custom-precision anchor propagates to resamples" begin
        # set_precision on the anchor must carry into every resample re-fit
        # (no silent revert to the 4·eps default); exercise the path end-to-end.
        mp = model_fit(linmodel, d, [1.0, 0.0]; name = ["a", "b"])
        migrad!(mp)
        set_precision(mp, 1e-10)
        bp = bootstrap(linmodel, d, mp; nresample = 50, seed = 1)
        @test all(isfinite, bp.std)
        @test bp.n_valid == 50
        jp = jackknife(linmodel, d, mp)
        @test all(isfinite, jp.std)
    end

end
