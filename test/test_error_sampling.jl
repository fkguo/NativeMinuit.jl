# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for sampling-based / contour error analysis (src/error_sampling.jl
# + the contour full_points addition in src/contours.jl).

using JuMinuit
using Test
using LinearAlgebra
using Logging
using Random

const _MC = JuMinuit   # internal-symbol access (`JuMinuit._mc_chisq_region`)

# Local sample covariance (columns = variables) — avoids a Statistics
# test-dependency just for one assertion.
function _samplecov(X::AbstractMatrix)
    n, p = size(X)
    μ = [sum(@view X[:, j]) / n for j in 1:p]
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

@testset "error sampling" begin

    # ───────────────────────────────────────────────────────────────────────
    @testset "delta_chisq / chisq_cl values + conventions" begin
        # The canonical Δχ² table (cf. the X(3872) notebook / lecture notes).
        @test delta_chisq(0.6827, 1) ≈ 1.0   atol = 0.01
        @test delta_chisq(0.6827, 2) ≈ 2.30  atol = 0.01
        @test delta_chisq(0.6827, 3) ≈ 3.53  atol = 0.02
        @test delta_chisq(0.6827, 4) ≈ 4.72  atol = 0.02
        @test delta_chisq(0.95,   1) ≈ 3.84  atol = 0.01
        @test delta_chisq(0.95,   2) ≈ 5.99  atol = 0.02
        @test delta_chisq(0.99,   1) ≈ 6.63  atol = 0.02

        # nσ convention (cl ≥ 1): 1→0.6827, 2→0.9545, 3→0.9973.
        @test delta_chisq(1, 1) ≈ 1.0  atol = 1e-6
        @test delta_chisq(2, 1) ≈ 4.0  atol = 1e-6
        @test delta_chisq(3, 1) ≈ 9.0  atol = 1e-6
        @test delta_chisq(1, 2) ≈ 2.30 atol = 0.01   # 1σ ⇒ 0.6827 ⇒ 2.30
        @test delta_chisq(2, 2) ≈ 6.18 atol = 0.02   # 2σ joint, two params

        # 2σ / 3σ probability mass (one param): Δχ² = (nσ)².
        @test delta_chisq(0.9545, 1) ≈ 4.0 atol = 0.02
        @test delta_chisq(0.9973, 1) ≈ 9.0 atol = 0.05

        # chisq_cl is the inverse CDF → probability.
        @test chisq_cl(1.0, 1)  ≈ 0.6827 atol = 1e-3
        @test chisq_cl(2.30, 2) ≈ 0.6827 atol = 1e-3
        @test chisq_cl(3.53, 3) ≈ 0.6827 atol = 2e-3
        @test chisq_cl(3.84, 1) ≈ 0.95   atol = 1e-3

        # Round-trip, probability → Δχ² → probability.
        for p in (0.5, 0.6827, 0.9, 0.95, 0.99), k in (1, 2, 3, 5)
            @test chisq_cl(delta_chisq(p, k), k) ≈ p atol = 1e-6
        end
        # Round-trip, Δχ² → probability → Δχ² (chisq_cl returns a probability
        # in (0,1), which delta_chisq then reads as a probability — NOT nσ).
        for D in (0.5, 1.0, 2.3, 5.0, 9.0), k in (1, 2, 3, 5)
            @test delta_chisq(chisq_cl(D, k), k) ≈ D atol = 1e-6
        end

        # Monotonicity: more parameters ⇒ larger Δχ² at fixed cl.
        @test delta_chisq(0.6827, 1) < delta_chisq(0.6827, 2) < delta_chisq(0.6827, 3)

        # Domain errors.
        @test_throws DomainError delta_chisq(-0.1, 1)
        @test_throws DomainError delta_chisq(0.0, 1)
        @test_throws DomainError delta_chisq(0.68, 0)
        @test_throws DomainError chisq_cl(-1.0, 2)
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "contour full_points (no extra fits)" begin
        # Correlated 3-param quadratic χ²; min 0 at mu, Hessian 2A.
        A  = [4.0 1.0 0.0; 1.0 9.0 2.0; 0.0 2.0 16.0]
        mu = [1.0, -2.0, 0.5]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [0.9, -1.8, 0.6]; names = ["a", "b", "c"], error = [0.5, 0.5, 0.5])
        migrad!(m)
        @test m.valid

        ce = contour_exact(m.fmin.internal, m.fmin.internal_cf, 1, 2; npoints = 12)
        @test ce.valid
        ps = contour_parameter_sets(ce)
        @test ps === ce.full_points
        @test length(ps) == length(ce.points)
        @test all(length(p) == 3 for p in ps)

        up = m.fcn.up
        fmin = m.fval
        for (k, p) in enumerate(ps)
            # The two contour coordinates sit in slots (par_x, par_y) = (1, 2).
            @test p[1] ≈ ce.points[k][1] atol = 1e-9
            @test p[2] ≈ ce.points[k][2] atol = 1e-9
            # Re-evaluating the FCN at the full set returns the contour level.
            @test m.fcn.f(p) ≈ fmin + up rtol = 1e-3
        end

        # The ellipse approximation does NO inner re-minimization → empty.
        ce_ell = contour(m.fmin.internal, m.fmin.internal_cf, 1, 2; npoints = 8)
        @test isempty(contour_parameter_sets(ce_ell))

        # Backward-compatible 7-arg constructor still works (full_points empty).
        ce7 = JuMinuit.ContoursError(1, 2, Tuple{Float64,Float64}[],
                                     ce.minos_x, ce.minos_y, 0, false)
        @test isempty(ce7.full_points)
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "hand-rolled MvNormal reproduces HESSE covariance" begin
        A  = [4.0 1.0 0.0; 1.0 9.0 2.0; 0.0 2.0 16.0]
        mu = [1.0, -2.0, 0.5]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [0.9, -1.8, 0.6]; error = [0.5, 0.5, 0.5])
        migrad!(m)
        Σ = Matrix(JuMinuit.free_covariance(m.fmin))

        # cl ≫ 1σ + adaptive off ⇒ negligible truncation ⇒ sample cov ≈ Σ.
        r = get_contours_samples(m; nsamples = 40_000, cl = 5, adaptive = false, seed = 2024)
        C = _samplecov(r.samples)
        @test size(r.samples) == (r.n_accepted, 3)
        for i in 1:3, j in 1:3
            @test C[i, j] ≈ Σ[i, j] rtol = 0.08 atol = 0.02
        end
    end

    @testset "quadratic: acceptance ≈ cl and marginal shadow" begin
        A  = [4.0 1.0; 1.0 9.0]
        mu = [1.0, -2.0]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [1.0, -2.0]; error = [0.5, 0.5])
        migrad!(m)
        Σ = Matrix(JuMinuit.free_covariance(m.fmin))

        # ndof defaults to n_free = 2 ⇒ δ = 2.30; acceptance → 0.6827.
        r = get_contours_samples(m; nsamples = 40_000, cl = 1, seed = 7)
        @test r.ndof == 2
        @test r.delta ≈ 2.2958 atol = 1e-3
        @test r.acceptance ≈ 0.6827 rtol = 0.03
        @test r.widen_rounds == 0
        @test !r.under_coverage

        # Marginal (min,max) ≈ the joint shadow ±√(δ·Σ_jj); symmetric.
        for j in 1:2
            shadow = sqrt(r.delta * Σ[j, j])
            lo, hi = r.bounds[j]
            @test (r.best[j] - lo) ≈ shadow rtol = 0.08
            @test (hi - r.best[j]) ≈ shadow rtol = 0.08
            @test (hi - r.best[j]) ≈ (r.best[j] - lo) rtol = 0.10   # symmetric
        end
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "nonlinear posterior ⇒ asymmetric region" begin
        # χ²(x) = (log x)² for x>0, +∞ otherwise. Region (Δχ²≤1) = [e⁻¹, e¹]:
        # lower extent 0.632, upper extent 1.718 → strongly asymmetric.
        χ²nl(x) = x[1] > 0 ? (log(x[1]))^2 : Inf
        rng = MersenneTwister(99)
        r = _MC._mc_chisq_region([1.0], χ²nl, 0.0, delta_chisq(1, 1);
                                 Σ = reshape([1.0], 1, 1), proposal = :mvnormal,
                                 nsamples = 40_000, adaptive = false, rng = rng)
        lo, hi = r.bounds[1]
        @test lo ≈ exp(-1) rtol = 0.05
        @test hi ≈ exp(1)  rtol = 0.05
        upper_extent = hi - 1.0
        lower_extent = 1.0 - lo
        @test upper_extent > 1.5 * lower_extent        # clearly asymmetric
        # No samples below 0 survive (Inf rejected).
        @test all(r.samples[:, 1] .> 0)
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "proposal UNDER-COVERAGE: too-tight Σ (the key case)" begin
        # True 1-param χ² with σ_true = 2 ⇒ 1σ region ±2. Feed a too-tight
        # Σ (σ_prop = 0.4) and verify adaptive widening recovers the wide
        # region while the non-adaptive run under-estimates + flags it.
        σ_true = 2.0
        χ²(x) = (x[1] / σ_true)^2
        δ = delta_chisq(1, 1)                       # 1.0
        Σtight = reshape([0.16], 1, 1)              # σ_prop = 0.4

        r_ad = _MC._mc_chisq_region([0.0], χ², 0.0, δ; Σ = Σtight, proposal = :mvnormal,
                                    nsamples = 20_000, adaptive = true,
                                    max_widen_rounds = 10, rng = MersenneTwister(1))
        r_no = _MC._mc_chisq_region([0.0], χ², 0.0, δ; Σ = Σtight, proposal = :mvnormal,
                                    nsamples = 20_000, adaptive = false,
                                    rng = MersenneTwister(1))

        # Adaptive recovers ≈ ±2 and does NOT flag under-coverage.
        @test r_ad.widen_rounds ≥ 1
        @test !r_ad.under_coverage
        @test r_ad.bounds[1][2] ≈ 2.0 rtol = 0.07
        @test r_ad.bounds[1][1] ≈ -2.0 rtol = 0.07

        # Non-adaptive clips short of ±2 and DOES flag under-coverage.
        @test r_no.under_coverage
        @test r_no.bounds[1][2] < r_ad.bounds[1][2]        # narrower than adaptive
        @test (r_ad.bounds[1][2] - r_ad.bounds[1][1]) >
              (r_no.bounds[1][2] - r_no.bounds[1][1])

        # Covariance-FREE range proposal also recovers ±2 (Σ-independent).
        r_uni = _MC._mc_chisq_region([0.0], χ², 0.0, δ; proposal = :uniform,
                                     ranges = [(-6.0, 6.0)], nsamples = 20_000,
                                     adaptive = true, rng = MersenneTwister(3))
        @test !r_uni.under_coverage
        @test r_uni.bounds[1][2] ≈ 2.0  rtol = 0.05
        @test r_uni.bounds[1][1] ≈ -2.0 rtol = 0.05
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "non-positive-definite Σ ⇒ eigen fallback (no throw)" begin
        χ2(x) = x[1]^2 + x[2]^2
        Σnpd = [1.0 2.0; 2.0 1.0]        # eigenvalues 3, −1 → not PD
        r = _MC._mc_chisq_region([0.0, 0.0], χ2, 0.0, delta_chisq(1, 2);
                                 Σ = Σnpd, proposal = :mvnormal, nsamples = 5_000,
                                 adaptive = false, rng = MersenneTwister(5))
        @test r.n_accepted ≥ 1
        @test all(isfinite, r.bounds[1])
        # _mvnormal_factor must still satisfy S·Sᵀ ≈ clamped Σ (neg → 0).
        S = _MC._mvnormal_factor(Σnpd)
        E = eigen(Symmetric(Σnpd))
        Σclamp = E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors'
        @test S * S' ≈ Σclamp atol = 1e-8
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "get_contours_samples: Minuit surface" begin
        A  = [4.0 1.0; 1.0 9.0]
        mu = [1.0, -2.0]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [1.0, -2.0]; names = ["p", "q"], error = [0.5, 0.5])
        migrad!(m)

        # Reproducibility via `seed` (independent of thread count).
        r1 = get_contours_samples(m; nsamples = 5_000, seed = 42)
        r2 = get_contours_samples(m; nsamples = 5_000, seed = 42)
        @test r1.n_accepted == r2.n_accepted
        @test r1.samples == r2.samples
        @test r1.names == ["p", "q"]

        # `paras` restricts the reported bounds; samples stay full-width.
        rp = get_contours_samples(m; nsamples = 3_000, seed = 1, paras = "q")
        @test rp.names == ["q"]
        @test length(rp.bounds) == 1
        @test size(rp.samples, 2) == 2          # still samples all free params

        # Positional IMinuit-style form with explicit χsq.
        rpos = get_contours_samples(m, x -> f(x), nothing, nothing; nsamples = 3_000, seed = 1)
        @test rpos.n_accepted == get_contours_samples(m; χsq = x -> f(x),
                                                      nsamples = 3_000, seed = 1).n_accepted

        # Uniform proposal on the Minuit surface.
        ru = get_contours_samples(m; proposal = :uniform,
                                  ranges = [(0.0, 2.0), (-3.0, -1.0)],
                                  nsamples = 5_000, seed = 9)
        @test ru.proposal === :uniform
        @test ru.n_accepted ≥ 1

        # Mahalanobis diagnostic length matches accepted count.
        rm = get_contours_samples(m; nsamples = 4_000, seed = 4, mahalanobis = true)
        @test length(rm.mahalanobis) == rm.n_accepted

        # :mvnormal without a covariance is an error path only when none
        # exists; with a fit present, the covariance is available.
        @test get_contours_samples(m; nsamples = 100, seed = 1).up == 1.0
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "Phase-H-aware threaded evaluation (consistency)" begin
        A  = [4.0 1.0; 1.0 9.0]
        mu = [1.0, -2.0]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [1.0, -2.0]; error = [0.5, 0.5])
        migrad!(m)
        # Same seed ⇒ identical proposals ⇒ identical result regardless of
        # whether χ² is evaluated serially or across threads.
        rs = get_contours_samples(m; nsamples = 6_000, seed = 11, threaded = false)
        rt = get_contours_samples(m; nsamples = 6_000, seed = 11, threaded = true)
        @test rs.n_accepted == rt.n_accepted
        @test rs.samples == rt.samples
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "warn + steer on unreliable covariance" begin
        # A clean, valid fit must NOT warn.
        f(x) = (x[1] - 1)^2 + (x[2] + 2)^2
        m = Minuit(f, [1.0, -2.0]; error = [0.3, 0.3])
        migrad!(m)
        @test_logs min_level = Logging.Warn get_contours_samples(m; nsamples = 500,
                                                                  seed = 1, warn = true)
        # Uniform proposal never checks the covariance ⇒ also never warns.
        @test_logs min_level = Logging.Warn get_contours_samples(m; proposal = :uniform,
            ranges = [(0.0, 2.0), (-3.0, -1.0)], nsamples = 500, seed = 1, warn = true)

        # An unreliable (non-converged / made-pos-def) covariance must NOT be
        # used silently: get_contours_samples either warns (a covariance
        # exists) or throws the "no covariance" error — never silent.
        m2 = Minuit(f, [50.0, -50.0]; error = [0.3, 0.3])
        migrad!(m2; maxfcn = 2)                      # too few calls → unconverged
        if !m2.valid || m2.fmin.internal.made_pos_def
            if JuMinuit.free_covariance(m2.fmin) !== nothing
                @test_logs (:warn,) match_mode = :any get_contours_samples(m2;
                    nsamples = 300, seed = 1, warn = true)
            else
                @test_throws ArgumentError get_contours_samples(m2; nsamples = 300, seed = 1)
            end
        end
    end

    # ───────────────────────────────────────────────────────────────────────
    @testset "contour_df_samples (DataFrames extension)" begin
        using DataFrames
        A  = [4.0 1.0; 1.0 9.0]
        mu = [1.0, -2.0]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [1.0, -2.0]; names = ["alpha", "beta"], error = [0.5, 0.5])
        migrad!(m)

        df = contour_df_samples(m; nsamples = 5_000, seed = 13)
        @test df isa DataFrame
        @test names(df) == ["alpha", "beta", "delta_chisq"]
        @test nrow(df) ≥ 1
        # The Δχ² column matches the kept-set true Δχ² and respects the cut.
        @test all(df.delta_chisq .<= delta_chisq(1, 2) + 1e-9)
        @test maximum(df.delta_chisq) > 0
    end

    # ───────────────────────────────────────────────────────────────────────
    # Review-hardening regression tests (code-review findings).
    @testset "bounded parameters: proposals respect limits" begin
        # A tight upper limit on x[1] that the unbounded Gaussian proposal
        # would routinely overshoot; out-of-limit draws must be rejected
        # (not fed to the FCN) so every accepted sample obeys the bound.
        f(x) = (x[1] - 1)^2 + (x[2] + 2)^2
        m = Minuit(f, [1.0, -2.0]; names = ["a", "b"], error = [0.5, 0.5],
                   limits = [(nothing, 1.3), nothing])
        migrad!(m)
        r = get_contours_samples(m; nsamples = 20_000, cl = 1, seed = 5)
        @test r.n_accepted ≥ 1
        @test all(r.samples[:, 1] .<= 1.3 + 1e-12)        # bound respected
    end

    @testset "n=2 contour full_points exactly equal the contour points" begin
        A  = [4.0 1.0; 1.0 9.0]
        mu = [1.0, -2.0]
        f(x) = (x .- mu)' * A * (x .- mu)
        m = Minuit(f, [1.0, -2.0]; error = [0.5, 0.5])
        migrad!(m)
        ce = contour_exact(m.fmin.internal, m.fmin.internal_cf, 1, 2; npoints = 8)
        ps = contour_parameter_sets(ce)
        @test length(ps) == length(ce.points)
        for (k, p) in enumerate(ps)
            @test length(p) == 2
            @test p[1] == ce.points[k][1]          # exact (authoritative coords)
            @test p[2] == ce.points[k][2]
        end
    end

    @testset "sampling does not pollute m.nfcn" begin
        f(x) = (x[1] - 1)^2 + (x[2] + 2)^2
        m = Minuit(f, [1.0, -2.0]; error = [0.3, 0.3])
        migrad!(m)
        before = m.nfcn
        get_contours_samples(m; nsamples = 2_000, seed = 1, threaded = false)
        get_contours_samples(m; nsamples = 2_000, seed = 1, threaded = true)
        @test m.nfcn == before                     # χ² evals + safety probe bypass m.fcn counter
    end
end
