# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Phase 0 §3.4 Criterion 4: Aqua + type-stability checks.
#
# - Aqua.test_all: project-quality checks (compat bounds, stale deps,
#   piracy, persistent tasks). Ambiguities check disabled by default —
#   known to flag stdlib false positives.
# - `@inferred` on every public entry point: the Julia-stdlib way of
#   asserting type-stability — each call must have a concretely inferrable
#   return type for the closure type the user supplied.
# - JET `@report_opt` (below): a NARROW optimization-analysis guard on the
#   low-level `migrad` entry points, scoped via `target_modules=(JuMinuit,)`.
#   `@inferred` only checks the top-level return type; this walks the call
#   graph and catches a silent FCN-call-site dispatch regression the perf
#   claim can't afford (ROADMAP risk #4). The narrow, JuMinuit-scoped target
#   avoids the noisy cross-version false positives (e.g. Julia 1.10's
#   BLAS.hemv! drift) that forced the earlier broad JET scan's removal — and
#   1.10, the worst offender, is no longer supported.

using Aqua
using RecipesBase

# JET reaches deep into Julia's compiler internals and is FRAGILE across Julia
# point releases (e.g. JET 0.11.3 fails to *precompile* on Julia 1.12.x —
# `MethodError: add_active_gotos!(…, ::Compiler.GenericDomTree{true})`), which
# would abort the WHOLE `Pkg.test()` at the dependency-precompile stage, before
# any JuMinuit test runs. JET is a dev-only optimization diagnostic, so it is
# NOT in the `[targets] test` deps: load it opportunistically and SKIP its check
# when unavailable, instead of letting a tooling/Julia-version incompatibility
# block CI and releases. Aqua + the `@inferred` block below (the real
# type-stability gate) always run. A developer who wants the JET check adds JET
# to the test environment (`]add JET`) and it runs automatically.
const HAS_JET = try
    @eval using JET
    true
catch err
    @info "JET unavailable — skipping the JET opt-analysis guard (dev-only, " *
          "Julia-version-fragile tool; add JET to the env to enable it)" exception = err
    false
end

@testset "Aqua + type-stability (§3.4 Criterion 4)" begin
    @testset "Aqua quality checks" begin
        # `treat_as_own = [RecipesBase.apply_recipe]`: our `plot_recipes.jl`
        # recipe for the `get_contours_samples` return — a Base `NamedTuple`
        # (kept as the documented public return) — is an intentional, reviewed
        # extension of `RecipesBase.apply_recipe`. Aqua's piracy heuristic
        # exempts a Symbol *value* type-param (so `Val{:sym}` recipes pass) but
        # not a `NamedTuple`'s tuple-of-symbols param, so it would otherwise
        # flag this one recipe. The recipes on our own result types
        # (BootstrapResult, JackknifeResult, SolutionModes/SolutionMode) are
        # non-pirating and need no exemption.
        Aqua.test_all(JuMinuit; ambiguities = false,
                      piracies = (treat_as_own = [RecipesBase.apply_recipe],))
    end

    @testset "@inferred on public entry points" begin
        # Standard test closure: a clean quadratic so the inner MIGRAD
        # converges quickly. Type-stability of the call graph is the
        # contract; the numerical result is incidental here.
        f(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        cf = CostFunction(f)
        x0 = [0.0, 0.0]
        errs = [0.1, 0.1]

        # migrad — vector-start (unbounded path)
        @test (@inferred migrad(cf, x0, errs)) isa FunctionMinimum

        # migrad — bounded path via Parameters
        params = Parameters([
            MinuitParameter("a", 0.0, 0.1; lower = -5.0, upper = 5.0),
            MinuitParameter("b", 0.0, 0.1),
        ])
        @test (@inferred migrad(cf, params)) isa BoundedFunctionMinimum

        # Build a converged fmin for the downstream tests.
        fmin = migrad(cf, x0, errs)

        # hesse — standalone HESSE on a converged state
        @test (@inferred JuMinuit.hesse(cf, fmin.state, Strategy(1))) isa
                JuMinuit.MinimumState

        # minos — asymmetric error on one parameter
        @test (@inferred JuMinuit.minos(fmin, cf, 1)) isa MinosError

        # contour — ellipse approximation
        @test (@inferred JuMinuit.contour(fmin, cf, 1, 2)) isa ContoursError

        # contour_exact — multi-param function_cross
        @test (@inferred JuMinuit.contour_exact(fmin, cf, 1, 2;
                                                  npoints = 8)) isa ContoursError

        # function_cross — single-param MINOS-style root find
        @test (@inferred JuMinuit.function_cross(fmin, cf, 1, +1.0)) isa
                JuMinuit.MnCross

        # function_cross_multi — multi-param variant
        @test (@inferred JuMinuit.function_cross_multi(
                fmin, cf, [1, 2], [0.0, 0.0], [1.0, 0.5])) isa JuMinuit.MnCross

        # estimate_edm — used inside MIGRAD's hot loop; must be inferable
        # for the inner-loop type-stability contract.
        @test (@inferred JuMinuit.estimate_edm(fmin.state.gradient,
                                                fmin.state.error)) isa Float64

        # Minuit wrapper — high-level user-facing constructor + migrad!
        m = Minuit(f, x0; name = ["a", "b"], errors = errs)
        @test (@inferred migrad!(m)) isa Minuit
        @test (@inferred minos!(m, 1)) isa Minuit
    end

    # JET opt-analysis (hot-path devirtualization). The `@report_opt` MACROS
    # cannot be parsed when JET is absent, so the check lives in a separate file
    # that is only `include`d (hence only parsed) when JET actually loaded.
    if HAS_JET
        include("test_jet_optanalysis.jl")
    else
        @testset "JET opt-analysis (skipped — JET unavailable)" begin
            @test_skip true
        end
    end
end
