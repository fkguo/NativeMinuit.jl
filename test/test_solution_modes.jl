# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for multi-modal solution detection (src/solution_modes.jl):
# clustering Δχ²-accepted samples into distinct solution modes, the WHITENED
# (error-normalized) distance metric, per-mode re-fit + deeper-basin flag, and
# the Clustering.jl-not-loaded fallback to the built-in clusterer.

# Clustering is a WEAKDEP, deliberately kept out of the test deps so the
# "not loaded → built-in works + friendly :dbscan error" path stays testable in
# the default suite. If Clustering IS present in the load path (a dev added it,
# or it lands in the test target later), load it so the DBSCAN extension is
# actually exercised here too. (`@eval using` because `using` can't be
# conditional or live inside a scope.) The ext is also verified manually
# against Clustering 0.15.
if Base.find_package("Clustering") !== nothing
    @eval using Clustering
end

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
        raw_labels, n_raw = JuMinuit._connected_components(permutedims(S), 1.0)
        @test n_raw == 1

        # And the whitened clusterer (the path find_solution_modes takes) on the
        # SAME data finds 2 — directly contrasting the two metrics.
        σ = [ms.errors[i] for i in 1:2]
        Zw = JuMinuit._whiten_samples(S, σ, :errors)
        _, n_white = JuMinuit._connected_components(Zw, 1.0)
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
        # The default suite runs WITHOUT Clustering, so every test above already
        # exercises the dependency-free built-in clusterer. Assert it explicitly,
        # then branch on whether the DBSCAN extension is active.
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
        migrad!(m)
        S = vcat(_cloud([1.0, 2.0], [0.1, 0.1], 50),
                  _cloud([6.0, 6.0], [0.1, 0.1], 50))

        comp = find_solution_modes(S, m; method = :components)
        @test length(comp) == 2

        if isempty(methods(JuMinuit._dbscan_labels))
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
end
