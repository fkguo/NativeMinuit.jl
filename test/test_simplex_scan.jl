# SPDX-License-Identifier: LGPL-2.1-or-later

using NativeMinuit
using Test

@testset "simplex.jl + scan.jl (C++ MnSimplex / MnScan ports)" begin

    @testset "simplex on shifted quadratic" begin
        # 3D shifted quadratic. With the C++-faithful EDM goal minedm = 0.1·up
        # (audit §5; was a 10⁴×-too-tight 1e-5·up), Simplex stops once the
        # simplex spread edm = f(jh)-f(jl) reaches the 0.1 scale — exactly
        # C++/iminuit's rule — locating the minimum to basin accuracy (~0.1),
        # NOT the 1e-5 fval the over-tight goal used to force. The edm band is a
        # regression guard: the old 1e-5·up goal would drive edm ≤ 1e-5 and
        # FAIL `edm > 1e-4`, while `!above_max_edm` guarantees edm ≤ the goal.
        f = x -> sum(abs2, x .- [1.0, 2.0, 3.0])
        fm = simplex(f, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        @test fm.is_valid
        @test !fm.above_max_edm
        @test 1e-4 < fm.state.edm ≤ 0.1          # stopped at the C++ 0.1·up goal
        @test NativeMinuit.fval(fm) < 0.1            # fval at the EDM-goal scale
        @test fm.state.parameters.x ≈ [1.0, 2.0, 3.0] atol = 0.2
        @test NativeMinuit.nfcn(fm) > 0
    end

    @testset "simplex with up=0.5 (NLL)" begin
        # ErrorDef = 0.5 → EDM goal minedm = 0.1·up = 0.05 (audit §5); fval lands
        # at that scale, not 1e-4. The old 1e-5·up = 5e-6 goal → edm ≤ 5e-6.
        f = x -> 0.5 * sum(abs2, x .- [1.0, 2.0])
        fm = simplex(f, [0.0, 0.0], [0.1, 0.1]; up = 0.5)
        @test !fm.above_max_edm
        @test 1e-4 < fm.state.edm ≤ 0.05         # 0.1·up with up=0.5
        @test NativeMinuit.fval(fm) < 0.05
        @test fm.state.parameters.x ≈ [1.0, 2.0] atol = 0.15
        @test fm.up ≈ 0.5
    end

    @testset "simplex returns FunctionMinimum without covariance" begin
        # Simplex doesn't compute a Hessian; the MinimumError is
        # marked `available=false` so downstream code (m.matrix,
        # eigenvalues, global_cc, minos) sees "no covariance" and
        # behaves accordingly. `hesse_failed` stays FALSE because
        # Hesse was never RUN — "failed" would be misleading
        # (round-2 fix I4).
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        fm = simplex(f, [0.0, 0.0], [0.1, 0.1])
        @test !fm.hesse_failed
        @test !NativeMinuit.is_available(fm.state.error)
        @test !NativeMinuit.has_covariance(fm)
    end

    @testset "bounded simplex" begin
        # Optimum at (-3, 7) but `a ∈ [0, ∞)` and `b ∈ [0, 5]` force
        # the constrained minimum to (0, 5).
        f = x -> (x[1] + 3)^2 + (x[2] - 7)^2
        params = NativeMinuit.Parameters(["a", "b"], [1.0, 1.0], [0.1, 0.1];
                                       limits = [(0.0, NaN), (0.0, 5.0)],
                                       fixed  = [false, false])
        bfm = simplex(f |> NativeMinuit.CostFunction, params)
        @test bfm.ext_values[1] ≈ 0.0 atol = 0.1
        @test bfm.ext_values[2] ≈ 5.0 atol = 0.1
    end

    @testset "simplex from Minuit struct" begin
        # C++-faithful EDM goal (0.1·up) → basin-level accuracy (audit §5), not
        # the 1e-2 / 1e-4 the old 10⁴×-tight goal forced. The looser, correct
        # goal also converges cleanly (the old goal frequently tripped
        # `above_max_edm`, which is why this used to assert only `fmin !== nothing`).
        f = x -> (x[1] - 0.7)^2 + (x[2] - 1.3)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        simplex(m)
        @test m.values ≈ [0.7, 1.3] atol = 0.1
        @test m.fval < 0.05
        @test m.fmin !== nothing
        @test m.valid                       # clean simplex convergence
    end

    @testset "scan: 1D evaluation" begin
        # Quadratic in 2D, scan along par 1: should see f minimized at x[1]=2.
        f = x -> (x[1] - 2.0)^2 + (x[2] - 3.0)^2
        result = scan(f, [0.0, 0.0], [0.5, 0.5], 1;
                       maxsteps = 11, low = -3.0, high = 7.0)
        # First entry is the central point (x[1]=0, f=4+9=13).
        @test result[1] == (0.0, 13.0)
        # Find the minimum across the grid
        grid = result[2:end]
        fmin_idx = argmin(f for (x, f) in grid)
        x_at_min, f_at_min = grid[fmin_idx]
        @test x_at_min ≈ 2.0 atol = 1.0
        @test f_at_min ≈ 9.0 atol = 1.0   # 0² + 3² = 9 (par 2 at 0)
    end

    @testset "scan: default ±2σ range" begin
        # low=0, high=0 → defaults to value ± 2·errs.
        f = x -> (x[1])^2 + (x[2])^2
        result = scan(f, [1.0, 0.0], [0.5, 0.5], 1; maxsteps = 5)
        # Should span [0, 2].
        grid = result[2:end]
        @test minimum(p[1] for p in grid) ≈ 0.0 atol = 1e-9
        @test maximum(p[1] for p in grid) ≈ 2.0 atol = 1e-9
    end

    @testset "scan: invalid range returns just the central point" begin
        # low > high → empty grid (only central point).
        f = x -> x[1]^2
        result = scan(f, [1.0], [0.1], 1;
                       maxsteps = 5, low = 5.0, high = 0.0)
        @test length(result) == 1
        @test result[1] == (1.0, 1.0)
    end

    @testset "scan: par_idx out of bounds throws" begin
        f = x -> x[1]^2
        @test_throws ArgumentError scan(f, [1.0], [0.1], 5)
        @test_throws ArgumentError scan(f, [1.0], [0.1], 0)
    end

    @testset "scan: maxsteps < 2 throws" begin
        f = x -> x[1]^2
        @test_throws ArgumentError scan(f, [1.0], [0.1], 1; maxsteps = 1)
    end

    @testset "scan: bounded clips against limits" begin
        # par.upper=3 should clip the high end of a scan reaching for ±2σ
        params = NativeMinuit.Parameters(["a"], [0.0], [0.5];
                                       limits = [(NaN, 3.0)],
                                       fixed  = [false])
        f = x -> x[1]^2
        cf = NativeMinuit.CostFunction(f, 1.0)
        result = scan(cf, params, 1; maxsteps = 5)
        grid = result[2:end]
        @test maximum(p[1] for p in grid) <= 3.0 + 1e-9
    end

    @testset "scan from Minuit struct (Minuit-method)" begin
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        migrad(m)
        result = scan(m, 1; maxsteps = 5, low = 0.0, high = 2.0)
        @test length(result) == 6   # 1 central + 5 grid
        grid = result[2:end]
        # Minimum should be at x[1]=1.0
        fmin_idx = argmin(p[2] for p in grid)
        @test grid[fmin_idx][1] ≈ 1.0 atol = 0.5
    end

    # ─────────────────────────────────────────────────────────────────
    # MnParameterScan best-value retention (MnParameterScan.h:42-43) +
    # iminuit `m.scan()` semantics: leave the Minuit at the best grid
    # point. (Scan tests live here since there is no test_scan.jl.)
    # ─────────────────────────────────────────────────────────────────
    @testset "scan(m, par) retains the best grid point in m" begin
        # Background param x[2] held at its m.params value (0.0); the 1D
        # minimum along x[1] is at 1.5, which is an exact grid point of
        # the [0,3]/20 grid.
        f = x -> (x[1] - 1.5)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; name = ["a", "b"], error = [0.1, 0.1])
        start_val = m.values[1]
        start_f = f(collect(m.values))
        result = scan(m, 1; maxsteps = 21, low = 0.0, high = 3.0)
        # Point-list contract unchanged for existing callers.
        @test length(result) == 22
        # m moved to the best grid point.
        @test m.values[1] != start_val
        @test abs(m.values[1] - 1.5) < abs(start_val - 1.5)
        @test f(collect(m.values)) <= start_f
        # A minimal (covariance-less) fmin is installed: fval / valid read.
        @test m.valid
        @test isfinite(m.fval)
        @test m.fval ≈ f(collect(m.values)) atol = 1e-9
        # Scan computes no Hessian → no covariance.
        @test m.covariance === nothing
        # Consistency with the returned list's lowest-fval entry.
        best = result[argmin(p[2] for p in result)]
        @test m.values[1] ≈ best[1] atol = 1e-12
        @test m.fval ≈ best[2] atol = 1e-9
    end

    @testset "scan retention: improvement vs start on a non-trivial FCN" begin
        # f along x[1] is the double-well (x₁²−1)²; minima at ±1. Start at 0
        # (a local max of the well) → the best grid point strictly lowers f.
        f = x -> (x[1]^2 - 1.0)^2 + 0.5 * (x[2] - 1.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        f_start = f(collect(m.values))
        scan(m, 1; maxsteps = 41, low = -2.0, high = 2.0)
        @test f(collect(m.values)) < f_start
        @test abs(abs(m.values[1]) - 1.0) < 0.2
    end

    @testset "profile does NOT move m (pure diagnostic)" begin
        f = x -> (x[1] - 1.5)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        before = collect(m.values)
        prof = profile(m, 1; bins = 7, low = 0.0, high = 3.0)
        @test length(prof) == 8
        @test collect(m.values) == before    # unchanged
        @test m.fmin === nothing             # no fit installed
    end

    @testset "scan after migrad preserves other params' fit values" begin
        # Minimum at (1, 2). migrad does NOT mutate m.params (stays [0,0]);
        # the fit lives in m.fmin. A post-fit scan must scan around the FIT
        # (x[2]=2), and retention must keep x[2] at 2 — NOT reset it to the
        # constructor 0. (Regression for the scan-around-m.params bug.)
        f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = [0.1, 0.1])
        migrad!(m)
        @test m.values[2] ≈ 2.0 atol = 1e-3
        scan(m, 1; maxsteps = 11, low = 0.0, high = 2.0)
        # x[2] retained at its fit value (NOT the constructor 0).
        @test m.values[2] ≈ 2.0 atol = 1e-3
        # x[1] at its best grid point ≈ 1.0 (an exact node of the [0,2]/10 grid).
        @test m.values[1] ≈ 1.0 atol = 1e-6
        @test m.valid
    end

    @testset "do-while first-round fidelity (iminuit 2.31.3 nfcn parity)" begin
        # C++ SimplexBuilder is a do-while: the first Nelder-Mead round
        # runs unconditionally, even when the INITIAL simplex already
        # satisfies edm ≤ minedm (warm start) or has edm = NaN (the
        # all-NaN pin lives in test_nonfinite_fcn.jl). All nfcn / fval
        # pins are empirical iminuit 2.31.3 values (same FCN, seed,
        # errors=0.1, m.simplex() at default budget/tolerance). fvals
        # are pinned bit-exactly — these FCNs use only +/−/*, so the
        # doubles are platform-deterministic.
        clean = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2

        # Warm seed AT the minimum: initial edm ≈ 0.044 < minedm = 0.1·up
        # ⇒ pre-fix the loop body never ran (nfcn=4 vs iminuit's 6).
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m)
        @test m.nfcn == 6
        @test m.fval === 0.0
        @test m.valid

        # Cold start: the entry guard passes anyway — the path must stay
        # EXACTLY as before the do-while fix (bit-identical to iminuit).
        m = Minuit(clean, [0.0, 0.0]; error = 0.1)
        simplex(m)
        @test m.nfcn == 21
        @test reinterpret(UInt64, m.fval) == 0x3f754d1eb851eb0a # 0.005200500488281143
        @test m.valid

        # NaN wall blocking the minimum: cold path unchanged (422 calls,
        # bit-identical finite incumbent), verdict call-limit-invalid —
        # and the non-finite returns trigger the single end-of-run warn.
        wallf = x -> x[1] > 0.5 ? NaN : (x[1] - 1.0)^2 + x[2]^2
        m = Minuit(wallf, [0.0, 0.0]; error = 0.1)
        @test_logs (:warn, r"non-finite") simplex(m)
        @test m.nfcn == 422
        @test reinterpret(UInt64, m.fval) == 0x3fd7147ae147ae14 # 0.360625
        @test !m.valid
        @test m.fmin.internal.reached_call_limit
        @test !m.fmin.internal.nonfinite_fval
    end

    @testset "pre-builder budget gate (ModularFunctionMinimizer.cxx:78-85)" begin
        # iminuit 2.31.3: m.simplex(ncall=1) bails right after the one
        # seed call — nfcn=1, fval=f(x0), call-limit invalid, errors =
        # the input steps, edm = n·up (InitialGradientCalculator seed:
        # g2 = 2·up/dirin² with V = diag(1/g2) ⇒ EDM ≡ n·up).
        clean = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m; ncall = 1)
        @test m.nfcn == 1
        @test m.fval === 0.0
        @test !m.valid
        @test m.fmin.internal.reached_call_limit
        @test m.fmin.internal.state.edm === 2.0
        @test all(isapprox.(collect(m.errors), 0.1; rtol = 1e-12))

        # Mid-round exhaustion: the builder never aborts inside a round —
        # ncall=2..5 all finish the mandatory round + final centroid
        # (nfcn=6, call-limit invalid); ncall=6 is exactly enough
        # (strict post-ybar `>` ⇒ valid). iminuit-verified boundary.
        for nc in (2, 5)
            m = Minuit(clean, [1.0, 2.0]; error = 0.1)
            simplex(m; ncall = nc)
            @test m.nfcn == 6
            @test !m.valid
            @test m.fmin.internal.reached_call_limit
        end
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m; ncall = 6)
        @test m.nfcn == 6
        @test m.valid
    end

    @testset "degenerate-edm error scaling is unconditional (SimplexBuilder.cxx:218)" begin
        # C++ scales dirin by √(up/edm) with NO edm guard: a constant
        # FCN (edm = 0) must report errors = +Inf — iminuit 2.31.3:
        # valid=True, edm=0.0, errors=[inf, inf], nfcn=7 (the mandatory
        # first round runs even though edm starts at 0). The pre-fix
        # `edm > 0` guard silently kept finite seed-scale errors here.
        m = Minuit(x -> 3.5, [0.0, 0.0]; error = 0.1)
        simplex(m)
        @test m.nfcn == 7
        @test m.valid
        @test m.fval === 3.5
        @test m.fmin.internal.state.edm === 0.0
        @test all(==(Inf), collect(m.errors))

        # Healthy fits: errors match iminuit to 12 digits (warm bowl,
        # same config as the nfcn=6 pin above).
        clean = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m)
        @test isapprox(m.errors[1], 0.9701425001453362; rtol = 1e-12)
        @test isapprox(m.errors[2], 0.48507125007266594; rtol = 1e-12)
    end

    @testset "repeated simplex(m): iminuit-style resume + run-local budget" begin
        # iminuit 2.31.3: a repeat m.simplex() resumes from the CURRENT
        # state — fit values AND the updated per-parameter errors — not
        # the constructor parameters. Warm bowl, errors=0.1:
        #   1st simplex(ncall=6): 6 calls, valid;
        #   2nd simplex(ncall=6): resumes with fit-scale errors, burns 8
        #     calls, call-limit invalid (iminuit displays the CUMULATIVE
        #     nfcn 14 = 6+8; NativeMinuit reports the per-run 8 — display
        #     convention only, the per-run call count is identical), and
        #     errors land 12-digit-identical to iminuit;
        #   repeat at default budget: 16 calls, valid.
        clean = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m; ncall = 6)
        @test m.nfcn == 6
        @test m.valid
        simplex(m; ncall = 6)
        @test m.nfcn == 8
        @test !m.valid
        @test m.fmin.internal.reached_call_limit
        @test isapprox(m.errors[1], 0.8944271909999166; rtol = 1e-12)
        @test isapprox(m.errors[2], 0.5031152949374517; rtol = 1e-12)

        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m)
        simplex(m)
        @test m.nfcn == 16
        @test m.valid
        @test isapprox(m.errors[1], 0.7359621520214286; rtol = 1e-12)
        @test isapprox(m.errors[2], 0.3715887336186622; rtol = 1e-12)

        # Run-local budget at the LOW level: a reused CostFunction's
        # second run gets a fresh per-application budget (C++ constructs
        # `MnFcn` per `MnApplication::operator()`) instead of
        # insta-bailing on the lifetime counter; states carry the
        # per-run nfcn.
        cf = CostFunction(clean, 1.0)
        fm1 = simplex(cf, [1.0, 2.0], [0.1, 0.1]; warn_nonfinite = false)
        fm2 = simplex(cf, [1.0, 2.0], [0.1, 0.1]; warn_nonfinite = false)
        @test fm1.is_valid
        @test fm2.is_valid
        @test fm1.state.nfcn == 6
        @test fm2.state.nfcn == 6
        @test !fm2.reached_call_limit

        # Resume carries SHRUNKEN fit errors as-is (no max() floor) —
        # iminuit 2.31.3, error=1.0 bowl: 1st run ends errors=[1.0,
        # 0.25]; the 2nd MUST seed 0.25 (not max(0.25, 1.0)) to land on
        # iminuit's final errors [0.5, 0.5] with the same 8 calls.
        m = Minuit(clean, [1.0, 2.0]; error = 1.0)
        simplex(m; ncall = 6)
        @test m.nfcn == 8
        @test !m.valid
        @test isapprox(m.errors[1], 1.0;  rtol = 1e-12)
        @test isapprox(m.errors[2], 0.25; rtol = 1e-12)
        simplex(m; ncall = 6)
        @test m.nfcn == 8
        @test !m.valid
        @test m.fmin.internal.reached_call_limit
        @test isapprox(m.errors[1], 0.5; rtol = 1e-12)
        @test isapprox(m.errors[2], 0.5; rtol = 1e-12)
    end

    @testset "maxfcn=0 is the C++ default-budget sentinel; negative throws" begin
        # C++ ModularFunctionMinimizer.cxx:53-54: `if (maxfcn == 0)
        # maxfcn = 200 + 100·npar + 5·npar²` — shared by ALL minimizers.
        # iminuit: m.simplex(ncall=0) runs the default budget (warm bowl
        # → nfcn=6, valid); a negative ncall raises (pybind unsigned
        # conversion) → ArgumentError here.
        clean = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        simplex(m; ncall = 0)
        @test m.nfcn == 6
        @test m.valid
        m = Minuit(clean, [1.0, 2.0]; error = 0.1)
        @test_throws ArgumentError simplex(m; ncall = -5)

        # Same sentinel through the MIGRAD entries (shared
        # _effective_maxfcn helper): maxfcn=0 ≡ default budget.
        rosen = x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        ma = Minuit(rosen, [-1.2, 1.0]; error = 0.1)
        migrad!(ma)
        mb = Minuit(rosen, [-1.2, 1.0]; error = 0.1)
        migrad!(mb; maxfcn = 0)
        @test mb.valid
        @test mb.nfcn == ma.nfcn
        @test mb.fval === ma.fval
    end

end
