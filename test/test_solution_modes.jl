# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for multi-modal solution detection (src/solution_modes.jl):
# clustering Δχ²-accepted samples into distinct solution modes, the WHITENED
# (error-normalized) distance metric, per-mode re-fit + deeper-basin flag, and
# the Clustering.jl-not-loaded fallback to the built-in clusterer.

# Clustering is a WEAKDEP that is ALSO a test dependency (in Project.toml's
# `test` target), so the DBSCAN extension gets standing CI coverage: loading it
# here activates `NativeMinuitClusteringExt`, and the "built-in :components ..."
# testset below then takes its DBSCAN-agreement branch. The complementary
# "not loaded → friendly :dbscan error" path can no longer be reached in-process
# (Clustering is loaded), so it is preserved by a dedicated subprocess test that
# spawns a fresh Julia WITHOUT `using Clustering` (see the subprocess testset
# below). The guard keeps the file runnable standalone even without Clustering;
# (`@eval using` because `using` can't be conditional or live inside a scope.)
if Base.find_package("Clustering") !== nothing
    @eval using Clustering
end

using Random   # MersenneTwister — stable stream, reproducible across Julia versions

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic sample-cloud generator (no Random dependency → reproducible).
# Golden-angle (phyllotaxis) fill: uniform density, so single-linkage connects
# the whole blob; far-apart blobs stay separate. `rad` is per-dimension.
# ─────────────────────────────────────────────────────────────────────────────
function _cloud(center::Vector{Float64}, rad::Vector{Float64}, n::Int)
    d = length(center)
    g = π * (3 - sqrt(5.0))                  # golden angle
    X = Matrix{Float64}(undef, n, d)
    for k in 1:n
        ρ = sqrt(k / n)                       # ∈ (0,1], uniform-area radius
        θ = k * g
        for j in 1:d
            X[k, j] = center[j] + rad[j] * ρ * cos(θ + (j - 1) * 1.7)
        end
    end
    return X
end

# A deterministic matrix with `ncol` columns and the same row count as `S`
# (for the wrong-width validation test; avoids a Random dependency).
_fixed_matrix(nrow::Int, ncol::Int) =
    reshape(collect(range(0.0, 1.0; length = nrow * ncol)), nrow, ncol)

@testset "solution_modes.jl — multi-modal solution detection" begin

    # A clean isotropic quadratic: minimum (1,2), errors ≈ (1,1) (σ=1 per par
    # since Δχ²=1 at 1 unit), covariance ≈ I — a simple whitening metric.
    fq(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2

    @testset "bimodal toy → 2 modes; unimodal → 1" begin
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)
        @test m.valid

        # Two tight, well-separated blobs (σ≈1 ⇒ centres ~6σ apart).
        S2 = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 120),
                   _cloud([5.0, -3.0], [0.1, 0.1], 80))
        modes = find_solution_modes(S2, m)        # default :cov, :components
        @test modes isa SolutionModes
        @test modes isa AbstractVector{SolutionMode}
        @test length(modes) == 2

        # Sorted by χ²: main = the basin at the fit minimum (lowest χ²).
        @test modes[1].index == 1
        @test modes[1].n_points == 120
        @test modes[2].n_points == 80
        @test modes[1].n_points + modes[2].n_points == 200
        @test modes[1].fraction ≈ 120 / 200
        # Representatives land in the right basins.
        @test isapprox(modes[1].representative[1], 1.0; atol = 0.2)
        @test isapprox(modes[1].representative[2], 2.0; atol = 0.2)
        @test isapprox(modes[2].representative[1], 5.0; atol = 0.2)
        @test isapprox(modes[2].representative[2], -3.0; atol = 0.2)
        # main χ² ≤ secondary χ²; Δχ² of secondary is large & positive.
        @test modes[1].fval ≤ modes[2].fval
        # global_best = fit minimum, so the main mode's representative (a sample)
        # sits at Δχ² ≥ 0 and close to it.
        @test 0 ≤ modes[1].delta_fval < 0.5
        @test modes[2].delta_fval > 1.0
        # member_indices partition the rows; per-param ranges bracket the rep.
        @test sort(vcat(modes[1].member_indices, modes[2].member_indices)) == collect(1:200)
        for s in modes, j in 1:2
            lo, hi = s.param_ranges[j]
            @test lo ≤ s.representative[j] ≤ hi
        end

        # Unimodal → exactly 1 mode covering everything.
        S1 = _cloud([1.0, 2.0], [0.1, 0.1], 150)
        modes1 = find_solution_modes(S1, m)
        @test length(modes1) == 1
        @test modes1[1].n_points == 150
        @test modes1[1].fraction ≈ 1.0
    end

    @testset "WHITENED metric resolves tiny-scale separation (naive Euclidean fails)" begin
        # Stiff quadratic: par 1 scale O(1), par 2 scale O(1e-3). Fit errors
        # come out ≈ [1, 1e-3] — exactly the LEC-vs-coupling scale gap.
        fstiff(x) = x[1]^2 + (x[2] / 1e-3)^2
        ms = Minuit(fstiff, [0.5, 5e-4]; names = ["big", "tiny"],
                     errors = [0.5, 1e-4])
        migrad!(ms)
        @test ms.valid
        errs = [ms.errors[i] for i in 1:2]
        @test isapprox(errs[1], 1.0; rtol = 1e-3)
        @test isapprox(errs[2], 1e-3; rtol = 1e-3)

        # Two modes IDENTICAL in `big` (spread 0.2) but separated by 5e-3 = 5σ
        # in `tiny`. In raw coords the `big` spread (0.2) dwarfs the `tiny`
        # separation (5e-3) → the modes overlap. Whitening rescales `tiny` by
        # 1000× → 5σ gap, clearly separated.
        A = _cloud([0.0, 0.0],     [0.2, 2e-4], 100)
        B = _cloud([0.0, 5.0e-3],  [0.2, 2e-4], 100)
        S = vcat(A, B)

        # WITH whitening: both metrics resolve the two modes.
        @test length(find_solution_modes(S, ms; whiten = :errors)) == 2
        @test length(find_solution_modes(S, ms; whiten = :cov)) == 2

        # WITHOUT whitening (naive Euclidean on raw coords, same threshold):
        # the built-in clusterer merges everything into ONE blob — the failure
        # the whitening requirement exists to prevent.
        raw_labels, n_raw = NativeMinuit._connected_components(permutedims(S), 1.0)
        @test n_raw == 1

        # And the whitened clusterer (the path find_solution_modes takes) on the
        # SAME data finds 2 — directly contrasting the two metrics.
        σ = [ms.errors[i] for i in 1:2]
        Zw = NativeMinuit._whiten_samples(S, σ, :errors)
        _, n_white = NativeMinuit._connected_components(Zw, 1.0)
        @test n_white == 2
    end

    @testset "refine=true: per-mode re-fit reaches local minima + deeper-basin flag" begin
        # Genuinely bimodal FCN (mirrors test_minuit_retry.jl's fcn_bi): a deep
        # well near (3,3) [fval ≈ -0.63] and a shallow one near (-3,-3)
        # [fval ≈ +0.14], plus a soft quadratic background.
        function fcn_bi(x)
            a, b = x[1], x[2]
            deep    = 1.5 * exp(-((a - 3.0)^2 + (b - 3.0)^2) / 1.0)
            shallow = 0.7 * exp(-((a + 3.0)^2 + (b + 3.0)^2) / 1.0)
            return -deep - shallow + 0.05 * (a^2 + b^2)
        end

        # Fit lands in the SHALLOW basin (start there, single pass) — the global
        # best the modes are measured against is therefore the WORSE minimum.
        m = Minuit(fcn_bi, [-2.5, -2.5]; names = ["a", "b"], errors = [0.5, 0.5])
        migrad!(m; iterate = 1)
        @test m.valid
        @test m.values[1] < 0          # converged to the shallow (negative) basin

        # Samples around BOTH basins (comparable χ², different physics).
        S = vcat(_cloud([3.0, 3.0],   [0.15, 0.15], 40),    # deep basin
                  _cloud([-3.0, -3.0], [0.15, 0.15], 60))   # shallow basin

        # Without refine: the deep basin still sorts FIRST (lower sample χ²),
        # i.e. the cluster the fit MISSED is flagged as the true main solution.
        modes = find_solution_modes(S, m)
        @test length(modes) == 2
        @test modes[1].representative[1] > 0     # main = deep basin (a ≈ +3)
        @test modes[2].representative[1] < 0     # secondary = shallow basin

        # With refine: each mode re-fits to its own local minimum, and the deep
        # mode is flagged as DEEPER than the global (shallow) best.
        rmodes = find_solution_modes(S, m; refine = true)
        @test length(rmodes) == 2
        @test all(s -> s.refined, rmodes)
        @test all(s -> s.refined_valid, rmodes)
        # Deep mode (main) re-fit ≈ (3,3), strictly below global best ⇒ new_min.
        deep = rmodes[1]
        @test isapprox(deep.refined_values[1], 2.9; atol = 0.2)
        @test isapprox(deep.refined_values[2], 2.9; atol = 0.2)
        @test deep.refined_fval < m.fval
        @test deep.new_min
        @test deep.refined_errors[1] > 0         # genuine errors from the re-fit
        # Shallow mode (secondary) re-fit ≈ (-2.78,-2.78), NOT a new minimum.
        shallow = rmodes[2]
        @test isapprox(shallow.refined_values[1], -2.78; atol = 0.2)
        @test !shallow.new_min
        @test any(s -> s.new_min, rmodes)        # the deeper-basin flag fired
    end

    @testset "built-in :components (no dep) + DBSCAN extension / not-loaded fallback" begin
        # Every test above uses method=:components (the default), so the
        # dependency-free built-in clusterer is already exercised regardless of
        # whether Clustering is loaded. Assert it explicitly, then branch on
        # whether the DBSCAN extension is active. With Clustering in the test
        # target the `else` (DBSCAN-agreement) branch is the one CI runs.
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)
        S = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 50),
                  _cloud([6.0, 6.0], [0.1, 0.1], 50))

        comp = find_solution_modes(S, m; method = :components)
        @test length(comp) == 2

        if isempty(methods(NativeMinuit._dbscan_labels))
            # Clustering NOT loaded → requesting :dbscan must raise a helpful,
            # actionable error (not an opaque MethodError).
            err = try
                find_solution_modes(S, m; method = :dbscan)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Clustering", err.msg)
            @test occursin("using Clustering", err.msg)
        else
            # Clustering loaded → the DBSCAN backend resolves the two modes and
            # agrees with the built-in clusterer on the populations.
            md = find_solution_modes(S, m; method = :dbscan,
                                      min_neighbors = 3, min_size = 5)
            @test length(md) == 2
            @test sort([s.n_points for s in md]) == sort([s.n_points for s in comp])
        end
    end

    @testset "Clustering-not-loaded → helpful :dbscan error (subprocess)" begin
        # Clustering is in the test target (so the DBSCAN-agreement branch above
        # runs in CI), which means the not-loaded branch can't be reached in this
        # process. Spawn a fresh Julia that loads NativeMinuit WITHOUT `using
        # Clustering` and confirm method=:dbscan throws the actionable 'load
        # Clustering' ArgumentError — not a bare MethodError. Mirrors the
        # Optim-bridge not-loaded subprocess test in test_optim_bridge.jl.
        code = """
        using NativeMinuit
        fq(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)
        S = [1.0 2.0; 1.05 1.98; 0.97 2.03; 6.0 6.0; 6.04 5.97; 5.98 6.05]
        try
            find_solution_modes(S, m; method = :dbscan)
            println("NO_ERROR")
        catch e
            msg = sprint(showerror, e)
            ok = e isa ArgumentError && occursin("Clustering", msg) &&
                 occursin("using Clustering", msg)
            println(ok ? "GOT_CLUSTERING_MSG" : "WRONG: " * msg)
        end
        """
        proj = Base.active_project()
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$proj -e $code`
        out = try
            read(ignorestatus(cmd), String)
        catch err
            @warn "could not spawn not-loaded subprocess; skipping" err
            "SKIP"
        end
        if out == "SKIP"
            @test_skip "subprocess unavailable"
        else
            @test occursin("GOT_CLUSTERING_MSG", out)
        end
    end

    @testset "fixed parameters + free-only sample width" begin
        # 3-par FCN with the 3rd parameter FIXED. Clustering must ignore the
        # fixed (zero-variance) dimension and still resolve the two free-space
        # modes; ranges are reported for ALL parameters.
        f3(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2
        m = Minuit(f3, [0.0, 0.0, 3.0]; names = ["x", "y", "z"],
                    errors = [0.1, 0.1, 0.1], fixed = [false, false, true])
        migrad!(m)
        @test m.valid
        @test m.npar == 2 && m.ndim == 3

        # Full (ndim=3) sample vectors; z constant at the fixed value 3.0.
        Sfull = vcat(_cloud([1.0, 2.0, 3.0], [0.1, 0.1, 0.0], 70),
                      _cloud([1.0, 7.0, 3.0], [0.1, 0.1, 0.0], 30))
        mf = find_solution_modes(Sfull, m)
        @test length(mf) == 2
        @test mf[1].n_points == 70
        # Fixed-parameter range is the constant value.
        @test mf[1].param_ranges[3] == (3.0, 3.0)
        @test length(mf[1].representative) == 3

        # Free-only width (npar=2): fixed z spliced back in from the fit.
        Sfree = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 70),
                      _cloud([1.0, 7.0], [0.1, 0.1], 30))
        mfree = find_solution_modes(Sfree, m)
        @test length(mfree) == 2
        @test mfree[1].representative[3] == 3.0          # reconstructed fixed par
        @test mfree[1].param_ranges[3] == (3.0, 3.0)
    end

    @testset "precomputed fvals + min_size noise filtering + input validation" begin
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)

        # Main blob + a 2-point far outlier "cluster".
        main = _cloud([1.0, 2.0], [0.1, 0.1], 100)
        outlier = [9.0 9.0; 9.05 9.02]
        S = vcat(main, outlier)

        # min_size=1 (default): the 2-point region is reported as a mode.
        @test length(find_solution_modes(S, m)) == 2
        # min_size=5: the 2-point region is demoted to noise.
        m3 = find_solution_modes(S, m; min_size = 5)
        @test length(m3) == 1
        @test m3.n_noise == 2

        # Precomputed fvals are used verbatim (representative = min-fval sample).
        fv = [fq(@view S[i, :]) for i in 1:size(S, 1)]
        mp = find_solution_modes(S, m; fvals = fv)
        @test mp[1].fval ≈ minimum(fv[mp[1].member_indices])

        # Wrong fvals length → error.
        @test_throws ArgumentError find_solution_modes(S, m; fvals = [1.0, 2.0])
        # Wrong sample width (3 cols for a 2-par fit) → error.
        @test_throws ArgumentError find_solution_modes(_fixed_matrix(size(S, 1), 3), m)
        # Bad kwargs.
        @test_throws ArgumentError find_solution_modes(S, m; whiten = :bogus)
        @test_throws ArgumentError find_solution_modes(S, m; method = :bogus)
        @test_throws ArgumentError find_solution_modes(S, m; threshold = -1.0)
    end

    @testset "report rendering" begin
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)
        S = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 60),
                  _cloud([6.0, 6.0], [0.1, 0.1], 40))
        modes = find_solution_modes(S, m)
        str = sprint(show, MIME"text/plain"(), modes)
        @test occursin("2 distinct solution", str)
        @test occursin("DIFFERENT physics", str)
        @test occursin("whiten=", str)
        # Per-element compact show.
        @test occursin("SolutionMode", sprint(show, modes[1]))
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Field-geometry regression tests (2026-06 f1(1420) stress-test handoff).
    # 9-parameter two-bowl surface — χ² = min of two quadratic bowls — at the
    # REAL basin geometry of the BESIII K⁰_SK⁰_Sπ⁰ f1(1420) coupled-channel
    # fit (basins ~10–24 fit-σ apart). A scatter cloud spanning both basins is
    # exactly the regime where the fit-local metrics (:cov/:errors) isolate
    # every point (0 modes) and the cloud-scale :sample metric must take over.
    # ─────────────────────────────────────────────────────────────────────────
    P9A = [-11.43, 44.31, -2570.10, 6.95, -84.99, 1.3593, 37.54, 194.72, 74.04]
    P9B = [-12.58, 37.44, -2182.53, 6.14, -91.48, 1.3425, -22.83, -10.07, -77.34]
    SIG9 = [0.13, 1.5, 28.0, 0.09, 1.0, 0.0017, 0.83, 10.0, 0.73]
    NAMES9 = ["c11", "c12", "c22", "h1", "h2", "mf", "d1", "d2", "df"]
    _bowl9(x, c, d0) = d0 + sum(((x[i] - c[i]) / SIG9[i])^2 for i in 1:9)
    f2bowl(x) = min(_bowl9(x, P9A, 30.7), _bowl9(x, P9B, 34.2))

    # Anchors at each basin + the two stress-test clouds (the MersenneTwister
    # stream is stable across Julia versions, so these are deterministic).
    m9A = Minuit(f2bowl, P9A; names = NAMES9, errors = copy(SIG9))
    migrad!(m9A); hesse!(m9A)
    m9B = Minuit(f2bowl, P9B; names = NAMES9, errors = copy(SIG9))
    migrad!(m9B); hesse!(m9B)
    rng9 = MersenneTwister(11)
    # multiplicative 4% scatter: spread ≫ local fit σ in several coordinates
    Smult = vcat([[P9A[j] * (1 + 0.04 * randn(rng9)) for j in 1:9]' for _ in 1:400]...,
                  [[P9B[j] * (1 + 0.04 * randn(rng9)) for j in 1:9]' for _ in 1:400]...)
    # additive 0.3σ blobs: error-scaled scatter around each basin
    Sadd = vcat([[P9A[j] + 0.3 * SIG9[j] * randn(rng9) for j in 1:9]' for _ in 1:400]...,
                 [[P9B[j] + 0.3 * SIG9[j] * randn(rng9) for j in 1:9]' for _ in 1:400]...)

    @testset "two-bowl 9-par geometry: :sample resolves what :cov cannot" begin
        @test m9A.valid && m9A.accurate
        @test m9A.fval ≈ 30.7 atol = 1e-8
        @test m9B.fval ≈ 34.2 atol = 1e-8

        # Fit-local metric on the wide cloud: every point isolated → 0 modes,
        # even with a CLEAN, accurate covariance — and the zero-modes
        # diagnostic reports the NN scale + the concrete fix.
        md_cov = @test_logs (:warn, r"isolated.*whiten=:sample") match_mode = :any begin
            find_solution_modes(Smult, m9A; whiten = :cov, min_size = 10)
        end
        @test length(md_cov) == 0
        @test md_cov.n_noise == 800
        @test md_cov.whiten === :cov

        # Cloud-MAD metric resolves both basins; the :auto default picks it
        # by itself (cloud ≫ fit scale → :sample).
        for kw in ((; whiten = :sample), NamedTuple())
            md = find_solution_modes(Smult, m9A; min_size = 10, kw...)
            @test md.whiten === :sample
            @test length(md) == 2
            @test all(s -> s.n_points >= 300, md)
            @test md.n_noise <= 100
            # one mode per basin: the production constant d₂ flips sign
            d2s = sort([s.representative[8] for s in md])
            @test d2s[1] < 0 < d2s[2]
            # members partition the rows (with noise making up the rest)
            @test sum(s.n_points for s in md) + md.n_noise == 800
        end

        # Error-scaled additive cloud: the fit metric works there, as before.
        md_e = find_solution_modes(Sadd, m9A; whiten = :errors, min_size = 10)
        @test length(md_e) == 2
        @test md_e.whiten === :errors

        # Anchored at the WORSE basin + refine: the basin-A mode re-fits to
        # the deeper minimum and fires new_min (the rescue); basin B re-fits
        # to its own minimum. Refined χ²s hit the exact bowl depths.
        md_r = find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                    refine = true)
        @test md_r.global_best ≈ 34.2 atol = 1e-8
        @test length(md_r) == 2
        @test all(s -> s.refined && s.refined_valid, md_r)
        @test md_r[1].refined_fval ≈ 30.7 atol = 1e-6
        @test md_r[1].new_min
        @test md_r[2].refined_fval ≈ 34.2 atol = 1e-6
        @test !md_r[2].new_min
        @test all(s -> isfinite(s.refined_walltime) && s.refined_walltime >= 0, md_r)
    end

    @testset ":auto gate keeps the fit metric on fit-scale clouds" begin
        # Tight single-basin cloud at the fit's own scale (~0.2 hesse-σ):
        # :auto must stay on :cov — the old, statistically tightest default —
        # and report ONE mode, silently.
        rng = MersenneTwister(5)
        tight = vcat([[P9A[j] + 0.2 * SIG9[j] / sqrt(2) * randn(rng) for j in 1:9]'
                      for _ in 1:300]...)
        md = @test_logs find_solution_modes(tight, m9A)
        @test md.whiten === :cov
        @test length(md) == 1
        @test md[1].n_points == 300

        # Tight unbalanced two-basin cloud (still fit-scale): :auto stays on
        # :cov and resolves both, including the 5% basin.
        rngu = MersenneTwister(13)
        S95 = vcat([[P9A[j] + 0.15 * SIG9[j] * randn(rngu) for j in 1:9]' for _ in 1:950]...,
                    [[P9B[j] + 0.15 * SIG9[j] * randn(rngu) for j in 1:9]' for _ in 1:50]...)
        mdu = find_solution_modes(S95, m9A; min_size = 10)
        @test mdu.whiten === :cov
        @test length(mdu) == 2
        @test sort([s.n_points for s in mdu]) == [50, 950]

        # Degenerate clouds (no spread to measure): :auto resolves to the fit
        # metric without any warning.
        md1 = @test_logs find_solution_modes(reshape(copy(P9A), 1, 9), m9A)
        @test md1.whiten === :cov
        @test length(md1) == 1
        mdid = @test_logs find_solution_modes(vcat(P9A', P9A', P9A'), m9A)
        @test mdid.whiten === :cov
        @test length(mdid) == 1
    end

    @testset "unbalanced WIDE cloud: the zero-modes hint is actionable" begin
        # 700/100 multiplicative cloud (9-par): MAD collapses to the majority
        # basin's own width, so in cloud-σ units the 9-dim blobs are sparse —
        # at threshold=1 single-linkage finds nothing (the documented high-d
        # caveat). The diagnostic must fire, and following its advice (raise
        # the threshold to ~2·median-NN) must actually recover BOTH basins —
        # the "actionable" guarantee is the contract being pinned here.
        rng = MersenneTwister(21)
        Sw = vcat([[P9A[j] * (1 + 0.04 * randn(rng)) for j in 1:9]' for _ in 1:700]...,
                   [[P9B[j] * (1 + 0.04 * randn(rng)) for j in 1:9]' for _ in 1:100]...)
        md0 = @test_logs (:warn, r"Try raising threshold") match_mode = :any begin
            find_solution_modes(Sw, m9A; min_size = 10)
        end
        @test md0.whiten === :sample             # the gate did pick the cloud metric
        @test length(md0) == 0                   # …but the cloud is too sparse at 1σ
        md3 = find_solution_modes(Sw, m9A; min_size = 10, threshold = 3.0)
        @test length(md3) == 2
        @test sort([s.n_points for s in md3]) == [100, 700]
        @test md3.n_noise == 0
    end

    @testset "unbalanced 950/50 with explicit :sample: small basin survives" begin
        # MAD is dominated by the 950-point cluster — the 50-point basin must
        # still come out as a mode (≥ min_size), not be binned as noise.
        f3u(x) = x[1]^2 + x[2]^2 + x[3]^2
        m3 = Minuit(f3u, [0.1, 0.1, 0.1]; names = ["u", "v", "w"],
                     errors = [0.5, 0.5, 0.5])
        migrad!(m3)
        c2 = [8.0, -6.0, 4.0]
        rng = MersenneTwister(7)
        S = vcat([[0.3 * randn(rng) for _ in 1:3]' for _ in 1:950]...,
                  [[c2[j] + 0.3 * randn(rng) for j in 1:3]' for _ in 1:50]...)
        md = find_solution_modes(S, m3; whiten = :sample, min_size = 10)
        @test md.whiten === :sample
        @test length(md) == 2
        ns = sort([s.n_points for s in md])
        @test ns[2] >= 900                       # big cluster intact
        @test ns[1] >= 30                        # small basin survives min_size=10
        @test md.n_noise <= 30
        small = md[1].n_points < md[2].n_points ? md[1] : md[2]
        @test isapprox(small.representative[1], 8.0; atol = 1.0)
    end

    @testset "degenerate coordinates are warned, never silent" begin
        # Constant cloud column under :sample → per-coordinate fit-σ fallback,
        # with a warning naming the coordinate; the modes still resolve.
        f3c(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 7.0)^2
        m3 = Minuit(f3c, [0.0, 0.0, 7.0]; names = ["a", "b", "c"],
                     errors = [0.1, 0.1, 0.1])
        migrad!(m3)
        S2blob = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 60),
                       _cloud([6.0, 6.0], [0.1, 0.1], 40))
        S3 = hcat(S2blob, fill(7.0, 100))
        md = @test_logs (:warn, r"zero cloud spread.*using the fit σ") match_mode = :any begin
            find_solution_modes(S3, m3; whiten = :sample)
        end
        @test md.whiten === :sample
        @test length(md) == 2

        # …and when the fit σ is ALSO degenerate for that coordinate, the
        # warning says the coordinate is excluded outright.
        set_error!(m3, "c", 0.0)                 # drops fmin; errors read from params
        md2 = @test_logs (:warn, r"ALSO degenerate") match_mode = :any begin
            find_solution_modes(S3, m3; whiten = :sample)
        end
        @test length(md2) == 2

        # All-degenerate cloud with EXPLICIT :sample → warned :errors fallback.
        mq = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(mq)
        mdid = @test_logs (:warn, r"zero spread in every coordinate") match_mode = :any begin
            find_solution_modes([1.0 2.0; 1.0 2.0; 1.0 2.0], mq; whiten = :sample)
        end
        @test mdid.whiten === :errors
        @test length(mdid) == 1

        # Dead fit σ under :errors → warned (was a silent 0-contribution).
        mq2 = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(mq2)
        set_error!(mq2, "b", 0.0)
        mdd = @test_logs (:warn, r"contribute NOTHING") match_mode = :any begin
            find_solution_modes(S2blob, mq2; whiten = :errors)
        end
        @test length(mdd) == 2                   # separation in `a` still resolves
    end

    @testset "untrustworthy covariance: :cov falls back with a warning" begin
        # A call-limited MIGRAD ends invalid but still carries a covariance —
        # exactly the silent-degradation case: :cov must warn + fall back.
        fr(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
        mr = Minuit(fr, [-1.2, 1.0]; names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(mr; maxfcn = 20, iterate = 1)
        @test !mr.valid
        @test !mr.accurate
        @test NativeMinuit.matrix(mr; skip_fixed = true) !== nothing
        S = _cloud([1.0, 1.0], [0.05, 0.05], 20)
        md = @test_logs (:warn, r"NOT trustworthy") match_mode = :any begin
            find_solution_modes(S, mr; whiten = :cov)
        end
        @test md.whiten === :errors
    end

    @testset "fvals policies: :none = zero FCN calls, :lazy = K calls" begin
        calls = Ref(0)
        fcount(x) = (calls[] += 1; (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        g(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2   # same function, not counted
        mc = Minuit(fcount, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(mc)
        S = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 60),
                  _cloud([6.0, 6.0], [0.1, 0.1], 40))

        # :none — the headline guarantee: ZERO FCN evaluations.
        calls[] = 0
        mn = find_solution_modes(S, mc; fvals = :none)
        @test calls[] == 0
        @test length(mn) == 2
        @test mn[1].n_points == 60 && mn[2].n_points == 40   # population order
        @test all(s -> isnan(s.fval) && isnan(s.delta_fval), mn)
        # representative = whitened-space medoid: a real member, central
        @test isapprox(mn[1].representative[1], 1.0; atol = 0.1)
        @test isapprox(mn[2].representative[1], 6.0; atol = 0.1)
        for s in mn
            @test any(i -> S[i, :] == s.representative, s.member_indices)
        end

        # :lazy — exactly K evaluations; χ²-sorted; Δχ² against the fit best.
        calls[] = 0
        ml = find_solution_modes(S, mc; fvals = :lazy)
        @test calls[] == 2
        @test length(ml) == 2
        @test ml[1].fval ≈ g(ml[1].representative)
        @test ml[1].fval < ml[2].fval
        @test ml[1].delta_fval == ml[1].fval - mc.fval

        # K == 0 outcome skips the (otherwise default) full evaluation, and
        # the zero-modes diagnostic fires.
        calls[] = 0
        mz = @test_logs (:warn, r"nearest-neighbour") match_mode = :any begin
            find_solution_modes(S, mc; whiten = :errors, threshold = 0.01,
                                 min_size = 5)
        end
        @test length(mz) == 0
        @test calls[] == 0

        # :none + refine: the FCN is used ONLY by the re-fits, and modes are
        # ranked by refined χ².
        calls[] = 0
        mnr = find_solution_modes(S, mc; fvals = :none, refine = true)
        @test all(s -> s.refined && s.refined_valid, mnr)
        @test mnr[1].refined_fval <= mnr[2].refined_fval
        @test mnr[1].refined_fval ≈ 0 atol = 1e-5
        @test all(s -> isnan(s.fval), mnr)
        @test calls[] > 0

        # Invalid fvals symbol → ArgumentError.
        @test_throws ArgumentError find_solution_modes(S, mc; fvals = :bogus)
    end

    @testset "refine budget, strategy/tol overrides, callback" begin
        # Reference: uncapped re-fits of the two 9-par additive-cloud modes.
        md_un = find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                     refine = true)
        nf_un = [s.refined_nfcn for s in md_un]
        @test all(>(100), nf_un)                 # uncapped: O(100) calls/mode

        # refine_maxfcn caps each MIGRAD attempt; reaching the cap also stops
        # the retry loop, so the per-mode cost is bounded by the cap plus at
        # most one MIGRAD iteration of overshoot (~2n+overhead evaluations).
        md_cap = find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                      refine = true, refine_maxfcn = 30,
                                      refine_iterate = 1)
        for s in md_cap
            @test s.refined
            @test s.refined_nfcn <= 30 + 30
            @test !s.refined_valid              # a 30-call budget can't validate 9 pars
        end
        @test maximum(s.refined_nfcn for s in md_cap) < minimum(nf_un)

        # Triage settings (strategy 0, loose tol) reach both minima at a
        # fraction of the default cost — and still validate.
        md_tri = find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                      refine = true, refine_strategy = 0,
                                      refine_tol = 10.0)
        @test all(s -> s.refined_valid, md_tri)
        @test maximum(s.refined_nfcn for s in md_tri) < minimum(nf_un)
        @test md_tri[1].refined_fval ≈ 30.7 atol = 1e-3
        @test md_tri[1].new_min

        # Callback: fires once per finished mode with a self-contained payload
        # (the checkpointing hook for slow FCNs).
        fired = NamedTuple[]
        md_cb = find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                     refine = true,
                                     refine_callback = r -> push!(fired, r))
        @test length(fired) == 2
        @test sort([r.k for r in fired]) == [1, 2]
        @test all(r -> r.K == 2, fired)
        @test all(r -> length(r.representative) == 9, fired)
        @test all(r -> r.refined && r.refined_valid, fired)
        @test all(r -> r.walltime >= 0, fired)
        @test sort([r.n_points for r in fired]) == sort([s.n_points for s in md_cb])

        # A throwing callback is caught + warned; the run completes intact.
        md_throw = @test_logs (:warn, r"refine_callback threw") match_mode = :any begin
            find_solution_modes(Sadd, m9B; whiten = :errors, min_size = 10,
                                 refine = true, refine_callback = _ -> error("boom"))
        end
        @test length(md_throw) == 2
        @test all(s -> s.refined, md_throw)

        # Budget kwarg validation.
        @test_throws ArgumentError find_solution_modes(Sadd, m9B; refine = true,
                                                        refine_maxfcn = 0)
    end

    @testset "rendering: :sample metric line, n/a χ², re-fit wall-time" begin
        mq = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(mq)
        S = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 60),
                  _cloud([6.0, 6.0], [0.1, 0.1], 40))
        md_s = find_solution_modes(S, mq; whiten = :sample)
        @test occursin("whiten=:sample (robust cloud scale)",
                       sprint(show, MIME"text/plain"(), md_s))
        md_n = find_solution_modes(S, mq; fvals = :none)
        str_n = sprint(show, MIME"text/plain"(), md_n)
        @test occursin("χ²=n/a", str_n)
        @test !occursin("Δχ²=NaN", str_n)
        @test occursin("n/a", sprint(show, md_n[1]))
        md_r = find_solution_modes(S, mq; refine = true)
        @test occursin(r"\d+ fcn, [0-9.e+-]+s\)", sprint(show, MIME"text/plain"(), md_r))
    end
end
