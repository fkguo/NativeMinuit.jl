# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Random
using Statistics
using Test

@testset "resampling.jl — interface (i): cost objects" begin

    # ── LeastSquares fixture (shared with the model+Data path) ────────────────
    a_true, b_true, σ = 2.0, 1.0, 0.3
    N = 80
    x = collect(range(0.0, 5.0; length = N))
    z = randn(Xoshiro(20240530), N)
    z = (z .- mean(z)) ./ std(z)
    y = a_true .* x .+ b_true .+ σ .* z
    d = Data(x, y, fill(σ, N))
    linmodel(xi, p) = p[1] * xi + p[2]

    @testset "LeastSquares cost ≡ model+Data (bit-identical)" begin
        # Fitting LeastSquares(d, model) and model_fit(model, d) reach the same
        # optimum, and resampling with the same seed draws the same subsets and
        # re-fits them identically → the θ̂ sample matrices are bit-identical.
        ls = LeastSquares(d, linmodel)
        bs_cost = bootstrap(ls, [1.0, 0.0]; nresample = 300, seed = 42)
        bs_md = bootstrap(linmodel, d, [1.0, 0.0]; nresample = 300, seed = 42)
        @test bs_cost.samples == bs_md.samples
        @test bs_cost.std == bs_md.std
        @test bs_cost.kind === :nonparametric

        jk_cost = jackknife(ls, [1.0, 0.0])
        jk_md = jackknife(linmodel, d, [1.0, 0.0])
        @test jk_cost.samples == jk_md.samples
        @test jk_cost.covariance == jk_md.covariance
    end

    @testset "cost parameter names propagate" begin
        ls = LeastSquares(d, linmodel; name = [:slope, :intercept])
        m = Minuit(ls, [1.0, 0.0]); migrad!(m)
        bs = bootstrap(ls, m; nresample = 100, seed = 1)
        @test bs.names == ["slope", "intercept"]
        jk = jackknife(ls, m)
        @test jk.names == ["slope", "intercept"]
    end

    @testset "LeastSquares cost: bootstrap/jackknife std ≈ HESSE" begin
        ls = LeastSquares(d, linmodel)
        m = Minuit(ls, [1.0, 0.0]); migrad!(m); hesse(m)
        he = [m.errors[1], m.errors[2]]
        bs = bootstrap(ls, m; nresample = 2000, seed = 1)
        @test bs.std[1] ≈ he[1] rtol = 0.2
        @test bs.std[2] ≈ he[2] rtol = 0.2
        jk = jackknife(ls, m)
        @test jk.std[1] ≈ he[1] rtol = 0.2
        @test jk.std[2] ≈ he[2] rtol = 0.2
        # correlation captured (slope/intercept anti-correlated over positive x)
        @test correlation(jk)[1, 2] < -0.5
        @test correlation(bs)[1, 2] < -0.5
    end

    @testset "LeastSquares cost: parametric bootstrap" begin
        ls = LeastSquares(d, linmodel)
        m = Minuit(ls, [1.0, 0.0]); migrad!(m); hesse(m)
        bp = bootstrap(ls, m; nresample = 2000, seed = 1, kind = :parametric)
        @test bp.kind === :parametric
        @test bp.std[1] ≈ m.errors[1] rtol = 0.15
        @test bp.std[2] ≈ m.errors[2] rtol = 0.15
    end

    @testset "threaded == serial (cost path, deterministic)" begin
        # NB: only a genuine parallel check under `julia -t N` (N>1); under the
        # default single thread it still asserts run-to-run reproducibility.
        Threads.nthreads() == 1 &&
            @info "threaded==serial cost test is single-threaded here; run with -t N to exercise parallelism"
        ls = LeastSquares(d, linmodel)
        m = Minuit(ls, [1.0, 0.0]); migrad!(m)
        b1 = bootstrap(ls, m; nresample = 200, seed = 99, threaded = false)
        b2 = bootstrap(ls, m; nresample = 200, seed = 99, threaded = true)
        @test b1.samples == b2.samples
        j1 = jackknife(ls, m; threaded = false)
        j2 = jackknife(ls, m; threaded = true)
        @test j1.samples == j2.samples
    end

    @testset "analytic grad is dropped from resample re-fits" begin
        # A user `grad` is a closure over the ORIGINAL data → invalid for a
        # resampled set; `_fit_kwargs` must not carry it, and the bootstrap must
        # still converge (numerical gradient per resample).
        ls = LeastSquares(d, linmodel)
        gchisq(p) = (r = (d.y .- (p[1] .* d.x .+ p[2])) ./ d.err .^ 2;
                     [-2 * sum(d.x .* r), -2 * sum(r)])
        m = Minuit(ls, [1.0, 0.0]; grad = gchisq); migrad!(m)
        @test m.values[1] ≈ 2.0 rtol = 0.1            # anchor used the analytic grad
        @test !haskey(JuMinuit._fit_kwargs(m), :grad)  # not carried into re-fits
        bs = bootstrap(ls, m; nresample = 100, seed = 1)
        @test bs.n_valid == 100
        @test all(isfinite, bs.std)
    end

    @testset "masked LeastSquares resamples over the active set" begin
        mask = trues(N); mask[1:10] .= false        # drop first 10 points
        ls = LeastSquares(d, linmodel; mask = mask)
        m = Minuit(ls, [1.0, 0.0]); migrad!(m)
        jk = jackknife(ls, m)
        @test jk.n == N - 10                          # only the active points
        @test jk.g == N - 10
        bs = bootstrap(ls, m; nresample = 100, seed = 1)
        @test size(bs.samples, 1) == 100
        @test all(isfinite, bs.std)
    end

    # ── UnbinnedNLL fixture ───────────────────────────────────────────────────
    @testset "UnbinnedNLL cost bootstrap + jackknife" begin
        μ0, s0, Ns = 2.0, 0.7, 400
        gsamp = μ0 .+ s0 .* randn(Xoshiro(123), Ns)
        gpdf(xi, p) = exp(-0.5 * ((xi - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
        un = UnbinnedNLL(gsamp, gpdf; name = [:μ, :σ])
        m = Minuit(un, [0.0, 1.0]; limits = [nothing, (1e-6, 10.0)]); migrad!(m)
        @test m.values[1] ≈ μ0 rtol = 0.1
        @test m.values[2] ≈ s0 rtol = 0.1

        bs = bootstrap(un, m; nresample = 400, seed = 7)
        @test bs.names == ["μ", "σ"]
        @test all(isfinite, bs.std)
        @test all(>(0), bs.std)
        # the bootstrap SE of the mean ≈ s/√N (classic result)
        @test bs.std[1] ≈ s0 / sqrt(Ns) rtol = 0.3

        jk = jackknife(un, m; d = 8)                  # delete-8 block → 50 groups
        @test jk.g == cld(Ns, 8)
        @test all(isfinite, jk.std)

        # parametric is unsupported for a likelihood cost
        @test_throws ArgumentError bootstrap(un, m; kind = :parametric)
    end

    @testset "ExtendedUnbinnedNLL cost bootstrap runs" begin
        λ0, Ns = 1.5, 300
        xexp = -log.(rand(Xoshiro(5), Ns)) ./ λ0
        density(xi, p) = p[2] * p[1] * exp(-p[1] * xi)   # ρ = N·λ·e^{-λx}; p=[λ,N]
        integral(p) = p[2]                                # ∫₀^∞ ρ dx = N
        ext = ExtendedUnbinnedNLL(xexp, density, integral; name = [:λ, :N])
        m = Minuit(ext, [1.0, Float64(Ns)]; limits = [(1e-6, 10.0), (1.0, 1e4)])
        migrad!(m)
        bs = bootstrap(ext, m; nresample = 150, seed = 2)
        @test size(bs.samples) == (150, 2)
        @test all(isfinite, bs.std)
        # Param 1 (rate λ) has genuine bootstrap spread; param 2 (count N) is
        # PINNED: every nonparametric resample draws exactly Ns points, and the
        # extended score ∂(−lnL)/∂N = 1 − n/N fixes N* = n independent of the
        # data, so σ(N) ≈ 0 by construction (≪ HESSE's √N ≈ 17.3) — documented
        # in the bootstrap docstring, asserted here rather than masked away.
        @test bs.std[1] > 0
        @test bs.std[2] < 1.0
    end

    @testset "non-resamplable costs raise a clear error (before any fit)" begin
        # Binned costs carry aggregated counts — point-resampling is undefined.
        n = Float64[12, 25, 31, 18, 7]
        xe = Float64[0, 1, 2, 3, 4, 5]
        cdf(edge, p) = 1 - exp(-p[1] * edge)
        bn = BinnedNLL(n, xe, cdf; name = [:λ])
        @test_throws ArgumentError bootstrap(bn, [0.5])
        @test_throws ArgumentError jackknife(bn, [0.5])
        ebn = ExtendedBinnedNLL(n, xe, (edge, p) -> p[2] * (1 - exp(-p[1] * edge));
                                name = [:λ, :N])
        @test_throws ArgumentError bootstrap(ebn, [0.5, 100.0])

        # CostSum composes data from multiple costs — resampling is ambiguous.
        ls = LeastSquares(d, linmodel; name = [:a, :b])
        gpdf(xi, p) = exp(-0.5 * ((xi - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
        un = UnbinnedNLL(randn(Xoshiro(1), 50), gpdf; name = [:a, :b])
        cs = ls + un
        @test_throws ArgumentError bootstrap(cs, [1.0, 1.0])
        @test_throws ArgumentError jackknife(cs, [1.0, 1.0])
    end

    @testset "cost argument validation" begin
        ls = LeastSquares(d, linmodel)
        @test_throws ArgumentError bootstrap(ls, [1.0, 0.0]; nresample = 0)
        @test_throws ArgumentError bootstrap(ls, [1.0, 0.0]; ci_level = 1.5)
        @test_throws ArgumentError bootstrap(ls, [1.0, 0.0]; kind = :bogus)
        @test_throws ArgumentError bootstrap(ls, [1.0, 0.0]; seed = -1)
        @test_throws ArgumentError jackknife(ls, [1.0, 0.0]; d = 0)
    end

end
