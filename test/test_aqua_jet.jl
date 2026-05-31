# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Phase 0 §3.4 Criterion 4: Aqua + type-stability checks.
#
# - Aqua.test_all: project-quality checks (compat bounds, stale deps,
#   piracy, persistent tasks). Ambiguities check disabled by default —
#   known to flag stdlib false positives.
# - `@inferred` on every public entry point: the Julia-stdlib way of
#   asserting type-stability. Each call must have a concretely
#   inferrable return type for the closure type the user supplied.
#   Replaced the earlier JET dependency (issue: noisy cross-version
#   false positives like Julia 1.10's BLAS.hemv! signature drift;
#   only covered one entry point; external dep maintenance).

using Aqua
using RecipesBase

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
end
