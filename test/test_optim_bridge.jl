# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for the `optim(m)` / `minimize_with(m)` alternative-minimizer bridge
# (ext/JuMinuitOptimExt.jl). iminuit's `m.scipy(method=...)` minimises with
# scipy.optimize then the user calls `hesse()`; the JuMinuit analog bridges to
# Optim.jl. These assert: same minimum as MIGRAD on a quadratic, correct
# covariance after `hesse`, bounds honoured via Fminbox, derivative-free
# convergence, fixed-parameter handling, the gradient passthrough, and the
# helpful error when Optim isn't loaded.

using Optim   # activates JuMinuitOptimExt

@testset "optim / minimize_with Optim bridge" begin

    @testset "unbounded quadratic: :lbfgs matches MIGRAD; hesse covariance" begin
        # Separable quadratic: distinct per-parameter minima, χ² (up=1) → σ=1.
        f(x) = (x[1] - 3.0)^2 + (x[2] + 1.0)^2 + (x[3] - 2.0)^2

        m_ref = Minuit(f, [0.0, 0.0, 0.0]); migrad!(m_ref); hesse(m_ref)

        m = Minuit(f, [0.0, 0.0, 0.0])
        out = optim(m; method = :lbfgs)
        @test out === m                              # mutates + returns m
        @test m.values ≈ [3.0, -1.0, 2.0] atol = 1e-5
        @test m.fval ≈ 0.0 atol = 1e-8
        @test m.valid

        # Provisional (pre-hesse) covariance is already the exact diagonal here.
        @test m.covariance !== nothing
        hesse(m)
        @test m.values ≈ m_ref.values atol = 1e-5
        @test m.covariance ≈ m_ref.covariance atol = 1e-4
        @test all(isapprox.(m.errors, 1.0; atol = 1e-3))   # χ² σ = 1
    end

    @testset "correlated quadratic: hesse recovers off-diagonal covariance" begin
        # Cross term couples x1,x2 → genuine off-diagonal covariance that the
        # diagonal seed cannot capture but hesse(m) must.
        f(x) = x[1]^2 + x[2]^2 + x[1] * x[2] - x[1]   # min at (2/3, -1/3)

        m_ref = Minuit(f, [0.0, 0.0]); migrad!(m_ref); hesse(m_ref)

        m = Minuit(f, [0.0, 0.0]); optim(m; method = :lbfgs); hesse(m)
        @test m.values ≈ m_ref.values atol = 1e-5
        @test m.values ≈ [2 / 3, -1 / 3] atol = 1e-4
        @test m.covariance ≈ m_ref.covariance atol = 1e-4
        @test abs(m.covariance[1, 2]) > 1e-2          # genuinely off-diagonal
    end

    @testset "bounded fit respects limits (Fminbox)" begin
        # x1's unconstrained min is 5 but it is capped to [-2, 2] → pins against
        # the upper bound; x2's min 0.5 is interior.
        g(x) = (x[1] - 5.0)^2 + (x[2] - 0.5)^2
        m = Minuit(g, [0.0, 0.2]; limits = [(-2.0, 2.0), (0.0, 1.0)])
        optim(m; method = :lbfgs)
        @test m.values[1] <= 2.0 + 1e-6       # never exceeds the upper bound
        @test m.values[1] > 1.9               # pushed against the active bound
        @test m.values[2] ≈ 0.5 atol = 1e-2   # interior optimum found
    end

    @testset "bounded interior optimum: matches MIGRAD + hesse" begin
        # Tight curvature so 1σ stays well inside the box (no boundary effects).
        g(x) = 100.0 * (x[1] - 0.4)^2 + 100.0 * (x[2] - 0.6)^2
        m_ref = Minuit(g, [0.2, 0.8]; limits = [(0.0, 1.0), (0.0, 1.0)])
        migrad!(m_ref); hesse(m_ref)

        m = Minuit(g, [0.2, 0.8]; limits = [(0.0, 1.0), (0.0, 1.0)])
        optim(m; method = :lbfgs); hesse(m)
        @test m.values ≈ [0.4, 0.6] atol = 1e-4
        @test m.values ≈ m_ref.values atol = 1e-4
        @test all(isfinite, m.errors)
        @test m.errors ≈ m_ref.errors atol = 1e-3
        @test m.covariance ≈ m_ref.covariance atol = 1e-3
    end

    @testset "mixed + one-sided bounds (Fminbox with ±Inf)" begin
        # One free param bounded, one unbounded → Fminbox must handle ±Inf sides.
        g(x) = (x[1] - 5.0)^2 + (x[2] + 3.0)^2
        m = Minuit(g, [0.0, 0.0]; limits = [(-2.0, 2.0), nothing])
        optim(m; method = :lbfgs)
        @test m.values[1] <= 2.0 + 1e-6        # bounded side pinned at the cap
        @test m.values[1] > 1.9
        @test m.values[2] ≈ -3.0 atol = 1e-3   # unbounded side reaches its min

        # One-sided limits: x1 ≥ 1 (interior min 5), x2 ≤ 0 (min 3 → pinned at 0).
        h(x) = (x[1] - 5.0)^2 + (x[2] - 3.0)^2
        m2 = Minuit(h, [2.0, -1.0]; limits = [(1.0, nothing), (nothing, 0.0)])
        optim(m2; method = :lbfgs)
        @test m2.values[1] ≈ 5.0 atol = 1e-2   # lower-bounded, interior optimum
        @test m2.values[2] <= 1e-6             # upper-bounded, pinned at 0
        hesse(m2)
        @test all(isfinite, m2.covariance)
    end

    @testset "Newton converges unbounded (rejected only when bounded)" begin
        f(x) = (x[1] - 2.0)^2 + (x[2] + 1.0)^2
        m = Minuit(f, [0.0, 0.0]); optim(m; method = :newton)
        @test m.values ≈ [2.0, -1.0] atol = 1e-5
    end

    @testset "derivative-free (:neldermead) converges" begin
        f(x) = (x[1] - 1.5)^2 + (x[2] - 2.5)^2 + 0.1
        m = Minuit(f, [0.0, 0.0])
        optim(m; method = :neldermead)
        @test m.values ≈ [1.5, 2.5] atol = 1e-3
        @test m.fval ≈ 0.1 atol = 1e-6
    end

    @testset "method-name aliases + optimizer-object passthrough" begin
        f(x) = sum(abs2, x .- [1.0, 2.0])
        for meth in (:bfgs, :lbfgs, "L-BFGS-B", :conjugategradient, :gradientdescent)
            m = Minuit(f, [0.0, 0.0]); optim(m; method = meth)
            @test m.values ≈ [1.0, 2.0] atol = 1e-3
        end
        # minimize_with with an Optim optimizer object (bypasses the name table)
        m = Minuit(f, [0.0, 0.0]); out = minimize_with(m, Optim.LBFGS())
        @test out === m
        @test m.values ≈ [1.0, 2.0] atol = 1e-4
        # minimize_with by name (alias of optim(m; method=…))
        m = Minuit(f, [0.0, 0.0]); minimize_with(m; method = :bfgs)
        @test m.values ≈ [1.0, 2.0] atol = 1e-4
        # iminuit-style pipe form, drop-in
        m = Minuit(f, [0.0, 0.0]); (m |> optim)
        @test m.values ≈ [1.0, 2.0] atol = 1e-3
    end

    @testset "fixed parameters are held out of the optimisation" begin
        f(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2
        m = Minuit(f, [0.0, 2.0, 0.0]; fix_x1 = true)   # x1 (2nd param) fixed at 2.0
        optim(m; method = :lbfgs)
        @test m.values[2] == 2.0                # fixed → unchanged
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[3] ≈ 3.0 atol = 1e-4
        hesse(m)
        @test m.covariance[2, 2] == 0.0         # fixed param: zero cov row/col
        @test m.covariance[1, 1] > 0.0
    end

    @testset "analytical gradient passthrough (grad=)" begin
        f(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        g(x) = [2 * (x[1] - 1.0), 2 * (x[2] - 2.0)]
        m = Minuit(f, [0.0, 0.0]; grad = g)
        optim(m; method = :lbfgs)
        @test m.values ≈ [1.0, 2.0] atol = 1e-6
        hesse(m)
        @test all(isapprox.(m.errors, 1.0; atol = 1e-3))
    end

    @testset "ncall / tol / options knobs are accepted" begin
        f(x) = sum(abs2, x .- [2.0, -3.0])
        m = Minuit(f, [0.0, 0.0]); optim(m; method = :lbfgs, ncall = 500, tol = 1e-10)
        @test m.values ≈ [2.0, -3.0] atol = 1e-5
        m2 = Minuit(f, [0.0, 0.0])
        optim(m2; method = :lbfgs, options = Optim.Options(g_tol = 1e-12))
        @test m2.values ≈ [2.0, -3.0] atol = 1e-6
    end

    @testset "resume: optim starts from the current values" begin
        # A prior MIGRAD leaves m at the minimum; optim from there stays put.
        f(x) = (x[1] - 4.0)^2 + (x[2] - 5.0)^2
        m = Minuit(f, [0.0, 0.0]); migrad!(m)
        optim(m; method = :lbfgs)
        @test m.values ≈ [4.0, 5.0] atol = 1e-5
    end

    @testset "error paths" begin
        f(x) = sum(abs2, x)
        # Unknown method name → helpful ArgumentError listing supported names.
        m = Minuit(f, [1.0, 1.0])
        @test_throws ArgumentError optim(m; method = :no_such_method)
        # Derivative-free + box limits is unsupported (Fminbox needs first-order).
        mb = Minuit(f, [0.5, 0.5]; limits = [(0.0, 1.0), (0.0, 1.0)])
        @test_throws ArgumentError optim(mb; method = :neldermead)
        # Second-order (Newton) + box limits is also rejected by Fminbox.
        mb2 = Minuit(f, [0.5, 0.5]; limits = [(0.0, 1.0), (0.0, 1.0)])
        @test_throws ArgumentError optim(mb2; method = :newton)
        # All-fixed → nothing to optimise.
        mf = Minuit(f, [1.0, 1.0]; fix_x0 = true, fix_x1 = true)
        @test_throws ArgumentError optim(mf)
    end

    @testset "extension is loaded; helpful message exists" begin
        # With Optim loaded the dispatch resolves to the extension module.
        @test JuMinuit._optim_bridge_ext() isa Module
        # The not-loaded message is helpful regardless of current load state.
        msg = JuMinuit._OPTIM_BRIDGE_NOT_LOADED
        @test occursin("Optim", msg)
        @test occursin("using Optim", msg)
    end

    @testset "Optim-not-loaded → helpful error (subprocess)" begin
        # Can't exercise the not-loaded branch in-process (Optim is loaded in
        # the test target), so spawn a fresh Julia that loads JuMinuit WITHOUT
        # Optim and confirm optim(m) throws a 'load Optim' message — not a bare
        # MethodError.
        code = """
        using JuMinuit
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        try
            optim(m)
            println("NO_ERROR")
        catch e
            msg = sprint(showerror, e)
            println(occursin("Optim", msg) ? "GOT_OPTIM_MSG" : "WRONG: " * msg)
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
            @test occursin("GOT_OPTIM_MSG", out)
        end
    end
end
