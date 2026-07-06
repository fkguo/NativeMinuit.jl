# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for the likelihood-ensemble MCMC sampler + quantile-band tools
# (src/mcmc.jl): random-walk Metropolis on the exact FCN, marginal
# quantiles/bands of derived quantities, and the plain-text ensemble
# persistence.

using NativeMinuit
using Test
using LinearAlgebra
using Logging
using Random

# Local mean / sample covariance (columns = variables) — same convention
# as test_error_sampling.jl (avoids a Statistics test-dependency).
_mcmc_mean(v::AbstractVector) = sum(v) / length(v)
function _mcmc_cov(X::AbstractMatrix)
    n, p = size(X)
    μ = [_mcmc_mean(@view X[:, j]) for j in 1:p]
    C = zeros(p, p)
    @inbounds for a in 1:p, b in 1:p
        s = 0.0
        for i in 1:n
            s += (X[i, a] - μ[a]) * (X[i, b] - μ[b])
        end
        C[a, b] = s / (n - 1)
    end
    return C
end

@testset "mcmc.jl — likelihood-ensemble MCMC + quantile bands" begin

    # ── Shared Gaussian target: 3-D correlated normal with KNOWN Σ ──────────
    # χ²(x) = (x−μ)ᵀ Σ⁻¹ (x−μ), so the likelihood ∝ exp(−χ²/2) is exactly
    # MvNormal(μ, Σ): every moment of the chain has a closed-form truth.
    μ3 = [1.0, -2.0, 0.5]
    Σ3 = [0.04 0.012 -0.004; 0.012 0.09 0.006; -0.004 0.006 0.0225]
    P3 = inv(Σ3)
    gauss_chi2(x) = (d = x .- μ3; dot(d, P3, d))

    mg = Minuit(gauss_chi2, [0.8, -1.5, 0.3]; names = ["a", "b", "c"])
    migrad!(mg)
    hesse!(mg)
    @test mg.valid

    ens = mcmc_sample(mg; seed = 42)

    @testset "Gaussian target: chain calibration" begin
        # Bookkeeping: defaults 52k/2k/25 ⇒ exactly 2000 kept sets.
        @test ens isa LikelihoodEnsemble
        @test length(ens) == 2000
        @test size(ens.samples) == (2000, 3)
        @test length(ens.fvals) == 2000
        @test ens.names == ["a", "b", "c"]
        @test ens.free == [true, true, true]
        @test ens.up == 1.0
        @test ens.proposal === :hesse
        @test ens.scale == 0.3                  # no adaptation requested
        @test ens.seed == UInt64(42)
        @test ens.fbest ≈ mg.fval atol = 1e-12
        @test all(isfinite, ens.samples)
        @test all(isfinite, ens.fvals)
        # Healthy random-walk acceptance (full-covariance proposal at
        # scale 0.3 is conservative ⇒ high-ish acceptance; the warn
        # thresholds are 0.05 / 0.9).
        @test 0.2 < ens.acceptance < 0.9

        # Sample mean → μ (tolerance ≈ 5 standard errors at n_kept = 2000).
        sm = [_mcmc_mean(@view ens.samples[:, j]) for j in 1:3]
        σ3 = sqrt.(diag(Σ3))
        @test all(abs.(sm .- μ3) ./ σ3 .< 0.12)

        # Sample covariance → Σ (diagonal within 15 %, correlations within 0.12).
        sc = _mcmc_cov(ens.samples)
        @test all(0.85 .< diag(sc) ./ diag(Σ3) .< 1.15)
        for (i, j) in ((1, 2), (1, 3), (2, 3))
            ρ_true = Σ3[i, j] / (σ3[i] * σ3[j])
            ρ_smp = sc[i, j] / sqrt(sc[i, i] * sc[j, j])
            @test abs(ρ_smp - ρ_true) < 0.12
        end

        # Δχ² distribution ~ χ²(3): the volume effect — the chain lives at
        # Δχ² ≈ n_free, NOT inside Δχ² ≤ 1. E[χ²₃] = 3; the PIT fraction
        # below the χ²₃ median must be ≈ 1/2 (uses our own delta_chisq).
        Δ = ens.fvals .- ens.fbest
        @test all(Δ .>= -1e-9)                  # quadratic target: never below min
        @test 2.6 < _mcmc_mean(Δ) < 3.4
        @test 0.45 < _mcmc_mean(Δ .<= delta_chisq(0.5, 3)) < 0.55
        # P(Δχ²₃ ≤ 1) ≈ 0.199: only ~1/5 of likelihood-weighted samples
        # visit the 1σ-joint region (and that is fine — see the docstring).
        @test 0.15 < _mcmc_mean(Δ .<= 1.0) < 0.25

        # The chain never touches the fit or its call counter.
        @test mg.nfcn == (let n0 = mg.nfcn
            mcmc_sample(mg; nsteps = 600, burn = 100, thin = 5, seed = 1)
            mg.nfcn
        end)
        @test collect(mg.values) ≈ [1.0, -2.0, 0.5] atol = 1e-3

        # show smoke test (plain + MIME)
        @test occursin("LikelihoodEnsemble", sprint(show, ens))
        s = sprint(show, MIME"text/plain"(), ens)
        @test occursin("2000 samples × 3 free parameters", s)
        @test occursin("acceptance", s)

        # Collection interface
        @test ens[1] == vec(ens.samples[1, :])
        @test ens[end] == vec(ens.samples[end, :])
        @test eltype(ens) == Vector{Float64}
        @test sum(θ -> θ[1], ens) / length(ens) ≈ sm[1]
    end

    @testset "up = 0.5 (−log L): acceptance uses exp(−Δfcn/(2·up))" begin
        nll(x) = 0.5 * gauss_chi2(x)
        mn = Minuit(nll, [0.8, -1.5, 0.3]; names = ["a", "b", "c"], errordef = 0.5)
        migrad!(mn)
        hesse!(mn)
        @test mn.valid
        ensn = mcmc_sample(mn; nsteps = 26_000, burn = 2_000, thin = 12, seed = 17)
        @test ensn.up == 0.5
        # Same posterior as the χ² form: χ²-equivalent Δ = Δfcn/up ~ χ²(3),
        # and the sample covariance still reproduces Σ.
        Δχ² = (ensn.fvals .- ensn.fbest) ./ ensn.up
        @test 2.6 < _mcmc_mean(Δχ²) < 3.4
        scn = _mcmc_cov(ensn.samples)
        @test all(0.8 .< diag(scn) ./ diag(Σ3) .< 1.2)
    end

    @testset "target_accept: burn-in scale adaptation (both directions)" begin
        # Far-too-large initial scale → adapted down, acceptance lands near target.
        e1 = mcmc_sample(mg; nsteps = 12_000, burn = 4_000, thin = 8,
                         scale = 5.0, target_accept = 0.25, seed = 7, warn = false)
        @test e1.scale < 5.0
        @test 0.12 < e1.acceptance < 0.40
        # Far-too-small initial scale → adapted up.
        e2 = mcmc_sample(mg; nsteps = 12_000, burn = 4_000, thin = 8,
                         scale = 0.005, target_accept = 0.25, seed = 7, warn = false)
        @test e2.scale > 0.005
        @test 0.12 < e2.acceptance < 0.45
        # Adaptation freezes after burn-in: the recorded scale is what ran
        # the kept chain.
        @test e1.scale != 5.0 && e2.scale != 0.005
    end

    @testset "proposal variants" begin
        # Per-coordinate :errors (the classic hand-rolled choice).
        ee = mcmc_sample(mg; nsteps = 12_000, burn = 2_000, thin = 10,
                         proposal = :errors, seed = 3)
        @test ee.proposal === :errors
        sm = [_mcmc_mean(@view ee.samples[:, j]) for j in 1:3]
        @test all(abs.(sm .- μ3) ./ sqrt.(diag(Σ3)) .< 0.25)

        # Explicit per-coordinate σ vector.
        ev = mcmc_sample(mg; nsteps = 12_000, burn = 2_000, thin = 10,
                         proposal = [0.2, 0.3, 0.15], scale = 1.0, seed = 3)
        @test ev.proposal === :steps
        @test all(isfinite, ev.samples)

        # Explicit proposal covariance matrix.
        em = mcmc_sample(mg; nsteps = 12_000, burn = 2_000, thin = 10,
                         proposal = Σ3, scale = 1.0, seed = 3)
        @test em.proposal === :matrix
        @test 0.05 < em.acceptance < 0.95

        # Input validation.
        @test_throws DimensionMismatch mcmc_sample(mg; proposal = [0.1, 0.2])
        @test_throws DimensionMismatch mcmc_sample(mg; proposal = zeros(2, 2))
        @test_throws ArgumentError mcmc_sample(mg; proposal = [0.1, -0.2, 0.3])
        @test_throws ArgumentError mcmc_sample(mg; proposal = :nope)
        # A non-PD explicit covariance would silently freeze the clamped
        # directions — must fail loudly instead.
        @test_throws ArgumentError mcmc_sample(mg;
            proposal = [1.0 0 0; 0 -0.5 0; 0 0 1.0])
    end

    # ── Boundary target (the gS ≥ 0 analogue) ───────────────────────────────
    # χ² = ((g+0.6)/0.4)² + h²  with  g ≥ 0: the unconstrained minimum sits at
    # g = −0.6, so the FIT lands ON the boundary (g* ≈ 0) and the likelihood
    # restricted to g ≥ 0 is a truncated Gaussian N(−0.6, 0.4²)|_{g≥0} with
    # closed-form quantiles:  q_p = −0.6 + 0.4·Φ⁻¹(Φ(1.5) + p·(1−Φ(1.5))) ⇒
    # q16 ≈ 0.0354, q50 ≈ 0.1333, q84 ≈ 0.3204; mean ≈ 0.1755.
    @testset "boundary target: one-sided pile-up, band may exclude best fit" begin
        chib(x) = ((x[1] + 0.6) / 0.4)^2 + x[2]^2
        mb = Minuit(chib, [0.2, 0.1]; names = ["g", "h"],
                    limits = [(0, nothing), nothing])
        migrad!(mb)
        @test mb.values[1] < 1e-4                 # best fit pinned at the boundary

        # At an active boundary both the covariance and the parabolic errors
        # are squeezed/meaningless — use the explicit-σ escape hatch.
        ensb = mcmc_sample(mb; nsteps = 42_000, burn = 2_000, thin = 10,
                           proposal = [0.3, 1.0], scale = 1.0, seed = 7)
        @test ensb.proposal === :steps
        @test 0.1 < ensb.acceptance < 0.6

        g = @view ensb.samples[:, 1]
        @test all(g .>= 0)                        # rejection keeps the box exactly
        @test _mcmc_mean(g) ≈ 0.1755 atol = 0.02      # truncated-Gaussian mean

        q16, q50, q84 = quantiles(ensb, θ -> θ[1])
        @test q16 ≈ 0.0354 atol = 0.02
        @test q50 ≈ 0.1333 atol = 0.025
        @test q84 ≈ 0.3204 atol = 0.04

        # THE boundary property: the marginal 16–84 % interval does NOT
        # contain the best fit (mode at the boundary ≠ median) — by
        # construction, not by failure.
        @test q16 > mb.values[1]

        # The untouched coordinate is a clean unit Gaussian.
        h = @view ensb.samples[:, 2]
        @test abs(_mcmc_mean(h)) < 0.12
        @test 0.85 < sqrt(_mcmc_cov(reshape(collect(h), :, 1))[1, 1]) < 1.15
    end

    @testset "fixed parameters never move" begin
        mf = Minuit(gauss_chi2, [0.8, -1.5, 0.3]; names = ["a", "b", "c"],
                    fixed = [false, true, false])
        migrad!(mf)
        hesse!(mf)
        b0 = mf.values[2]
        ensf = mcmc_sample(mf; nsteps = 6_000, burn = 1_000, thin = 10, seed = 5)
        @test ensf.free == [true, false, true]
        @test all(ensf.samples[:, 2] .== b0)
        @test !all(ensf.samples[:, 1] .== ensf.samples[1, 1])   # free ones DID move
        # Derived-quantity evaluation sees the full vector incl. the fixed value.
        @test quantiles(ensf, θ -> θ[2]; p = (0.5,))[1] == b0
    end

    @testset "quantiles: scalar derived quantities" begin
        # p = (0, 1) are the ensemble extrema, exactly.
        f = θ -> θ[1] - θ[2]
        vals = [f(ens[i]) for i in 1:length(ens)]
        lo, hi = quantiles(ens, f; p = (0, 1))
        @test lo == minimum(vals) && hi == maximum(vals)
        # Default p = 16/50/84 %; median of a−b ≈ μ₁−μ₂ = 3.
        q = quantiles(ens, f)
        @test length(q) == 3
        @test issorted(q)
        @test q[2] ≈ 3.0 atol = 0.05
        # Non-finite values: dropped with a warning; all-NaN throws.
        fnan = θ -> θ[1] > μ3[1] ? NaN : θ[1]
        @test_logs (:warn, r"dropped") match_mode = :any quantiles(ens, fnan)
        @test_logs min_level = Logging.Warn quantiles(ens, fnan; warn = false)
        @test_throws ArgumentError quantiles(ens, θ -> NaN)

        # `f` receives a COPY: a mutating model function (a real-world
        # pattern with preallocated buffers) cannot corrupt the ensemble.
        snap = copy(ens.samples)
        quantiles(ens, θ -> (θ[1] = -777.0; θ[2]))
        @test ens.samples == snap
    end

    @testset "quantile_band: pointwise band of a curve" begin
        # f(x,θ) = θ₁ + θ₂·x is Gaussian at each x with mean μ₁+μ₂x and
        # σ²(x) = Σ₁₁ + 2xΣ₁₂ + x²Σ₂₂ ⇒ analytic 16/50/84 % quantiles
        # (z₀.₈₄ = 0.994458).
        xs = 0.0:0.5:2.0
        B = quantile_band(ens, (x, θ) -> θ[1] + θ[2] * x, xs; p = (0.16, 0.5, 0.84))
        @test size(B) == (length(xs), 3)
        for (i, x) in enumerate(xs)
            mid = μ3[1] + μ3[2] * x
            σx = sqrt(Σ3[1, 1] + 2x * Σ3[1, 2] + x^2 * Σ3[2, 2])
            @test B[i, 2] ≈ mid atol = 5 * σx / sqrt(2000) * 1.6 + 0.01
            half = (B[i, 3] - B[i, 1]) / 2
            @test half ≈ 0.994458 * σx rtol = 0.12
            @test B[i, 1] < B[i, 2] < B[i, 3]
        end

        # curve=true (one call per member, whole curve) ≡ pointwise mode.
        B2 = quantile_band(ens, θ -> [θ[1] + θ[2] * x for x in xs], xs;
                           p = (0.16, 0.5, 0.84), curve = true)
        @test B2 ≈ B rtol = 1e-12

        # Wrong curve length / empty xs are caught.
        @test_throws DimensionMismatch quantile_band(ens, θ -> [θ[1]], xs; curve = true)
        @test_throws ArgumentError quantile_band(ens, (x, θ) -> θ[1], Float64[])

        # Mutation safety in BOTH evaluation modes (f gets row copies).
        snap = copy(ens.samples)
        quantile_band(ens, (x, θ) -> (θ[3] = -1.0; θ[2] * x), [0.5, 1.0])
        quantile_band(ens, θ -> (θ[1] = 99.0; [θ[2] * x for x in xs]), xs; curve = true)
        @test ens.samples == snap

        # Cross-grid-point leak regression (gpt-5.5 r1 Important #1): a
        # mutating f must not let one grid point corrupt the NEXT grid
        # point's view of the same member (the earlier cached-row reuse
        # bug). f returns θ[1]·x then poisons θ[1]; with a reused row the
        # x=2 point would read 2·(poison) instead of 2·a. Output VALUES
        # are asserted here, not just ensemble non-mutation (the previous
        # test mutated θ[3] but returned θ[2]·x, so it could not see it).
        mini = load_ensemble(IOBuffer("# names: a\n0.0 2.0\n0.0 3.0\n"))
        Bm = quantile_band(mini, (x, θ) -> (v = θ[1] * x; θ[1] = 1.0e6; v),
                           [1.0, 2.0]; p = (0.0, 1.0), warn = false)
        @test Bm[1, :] == [2.0, 3.0]      # x=1: a ∈ {2,3}
        @test Bm[2, :] == [4.0, 6.0]      # x=2: 2·a — NOT 2·(a+1e6) from a leaked row

        # Default p is the 16–84 % pair.
        Bd = quantile_band(ens, (x, θ) -> θ[1], [0.0])
        @test size(Bd) == (1, 2)
    end

    @testset "save/load: exact round-trip + hand-rolled compatibility" begin
        mktempdir() do dir
            path = joinpath(dir, "ens.dat")
            save_ensemble(path, ens; comment = "error set demo\nsecond line")
            txt = read(path, String)
            @test startswith(txt, "# NativeMinuit LikelihoodEnsemble v1")
            @test occursin("# error set demo\n", txt)
            @test occursin("# second line\n", txt)
            @test occursin("# cols: fval a b c", txt)

            e2 = load_ensemble(path)
            @test e2.samples == ens.samples         # shortest-repr ⇒ EXACT
            @test e2.fvals == ens.fvals
            @test e2.names == ens.names
            @test e2.free == ens.free
            @test e2.best == ens.best
            @test e2.fbest == ens.fbest
            @test e2.up == ens.up
            @test e2.acceptance == ens.acceptance
            @test (e2.nsteps, e2.burn, e2.thin) == (ens.nsteps, ens.burn, ens.thin)
            @test e2.scale == ens.scale
            @test e2.proposal === ens.proposal
            @test e2.seed == ens.seed

            # A user comment that LOOKS like a metadata line must not shadow
            # the real metadata (save writes metadata first; load takes the
            # first occurrence of each key).
            path2 = joinpath(dir, "ens2.dat")
            save_ensemble(path2, ens; comment = "up: 999\nscale: 777")
            e3 = load_ensemble(path2)
            @test e3.up == ens.up
            @test e3.scale == ens.scale

            # Foreign hand-rolled file (comment header + `χ² p…` rows).
            foreign = """
            # error set B v2 @ rho=-0.295 tied bg2=0; cols: chi2 C12 C32
            360.95 -2.61 -13.30
            361.20 -2.65 -13.25
            362.10 -2.55 -13.40
            """
            ef = load_ensemble(IOBuffer(foreign))
            @test size(ef.samples) == (3, 2)
            @test ef.fvals == [360.95, 361.20, 362.10]
            @test ef.names == ["p1", "p2"]
            @test ef.free == [true, true]
            @test isnan(ef.fbest) && isnan(ef.up)
            @test ef.proposal === :unknown
            @test quantiles(ef, θ -> θ[1]; p = (0.5,))[1] ≈ -2.61
            # Metadata overrides for foreign files.
            ef2 = load_ensemble(IOBuffer(foreign); names = ["C12", "C32"], up = 1.0)
            @test ef2.names == ["C12", "C32"]
            @test ef2.up == 1.0
            @test_throws DimensionMismatch load_ensemble(IOBuffer(foreign); names = ["x"])

            # A parameter name with whitespace cannot round-trip the
            # whitespace-separated format → save fails loudly (gpt-5.5 r1
            # Minor #4) rather than silently splitting it into two names.
            spaced = load_ensemble(IOBuffer("0.0 1.0\n0.0 2.0\n"); names = ["a b"])
            @test_throws ArgumentError save_ensemble(IOBuffer(), spaced)

            # Malformed inputs.
            @test_throws ArgumentError load_ensemble(IOBuffer("# only comments\n"))
            @test_throws ArgumentError load_ensemble(IOBuffer("1.0 2.0\noops 3.0\n"))
            @test_throws ArgumentError load_ensemble(IOBuffer("1.0 2.0\n1.0 2.0 3.0\n"))
            @test_throws ArgumentError load_ensemble(IOBuffer("42.0\n"))
        end
    end

    @testset "reproducibility & input guards" begin
        kw = (; nsteps = 3_000, burn = 500, thin = 5)
        a = mcmc_sample(mg; kw..., seed = 11)
        b = mcmc_sample(mg; kw..., seed = 11)
        @test a.samples == b.samples && a.fvals == b.fvals
        c = mcmc_sample(mg; kw..., seed = 12)
        @test c.samples != a.samples
        # Explicit rng (fresh, equal state ⇒ identical chains).
        r1 = mcmc_sample(mg; kw..., rng = Random.MersenneTwister(99))
        r2 = mcmc_sample(mg; kw..., rng = Random.MersenneTwister(99))
        @test r1.samples == r2.samples
        @test r1.seed === nothing
        @test_throws ArgumentError mcmc_sample(mg; seed = 1, rng = Random.MersenneTwister(1))
        # An out-of-[0, typemax(UInt64)] seed is rejected UP FRONT (stored as
        # UInt64), not after running the whole chain (gpt-5.5 r1 Minor #3 +
        # r2 Minor #7: negative AND oversized).
        @test_throws ArgumentError mcmc_sample(mg; seed = -1)
        @test_throws ArgumentError mcmc_sample(mg; seed = big(2)^80)

        # Argument validation.
        m0 = Minuit(gauss_chi2, [0.8, -1.5, 0.3])
        @test_throws ArgumentError mcmc_sample(m0)              # no migrad! yet
        @test_throws ArgumentError mcmc_sample(mg; nsteps = 0)
        @test_throws ArgumentError mcmc_sample(mg; burn = 60_000)        # ≥ nsteps
        @test_throws ArgumentError mcmc_sample(mg; thin = 0)
        @test_throws ArgumentError mcmc_sample(mg; nsteps = 100, burn = 90, thin = 20)
        @test_throws ArgumentError mcmc_sample(mg; scale = 0.0)
        @test_throws ArgumentError mcmc_sample(mg; adapt_every = 0)
        @test_throws ArgumentError mcmc_sample(mg; target_accept = 1.5)
        @test_throws ArgumentError mcmc_sample(mg; target_accept = 0.25, burn = 50,
                                               adapt_every = 100)

        # Tuning warnings: absurdly large frozen scale ⇒ acceptance ≈ 0 ⇒ warn;
        # warn=false silences.
        @test_logs (:warn, r"acceptance") match_mode = :any mcmc_sample(
            mg; nsteps = 1_500, burn = 100, thin = 5, scale = 500.0, seed = 2)
        @test_logs min_level = Logging.Warn mcmc_sample(
            mg; nsteps = 1_500, burn = 100, thin = 5, scale = 500.0, seed = 2,
            warn = false)
    end

    @testset "unreliable covariance → :errors fallback (warned)" begin
        # An invalid fit (call-limited MIGRAD) has an untrustworthy Σ: the
        # :hesse proposal must warn and fall back to per-coordinate errors —
        # which only affects mixing, not what the chain converges to.
        mq = Minuit(gauss_chi2, [0.8, -1.5, 0.3])
        migrad!(mq; maxfcn = 3)
        @test !mq.valid
        ensq = @test_logs (:warn, r"unreliable|no covariance") match_mode = :any mcmc_sample(
            mq; nsteps = 4_000, burn = 1_000, thin = 10, seed = 4)
        @test ensq.proposal === :errors
        @test all(isfinite, ensq.samples)
        # Even from a junk start the burnt-in chain finds the true mass region.
        sm = [_mcmc_mean(@view ensq.samples[:, j]) for j in 1:3]
        @test all(abs.(sm .- μ3) .< 0.5)
    end
end
