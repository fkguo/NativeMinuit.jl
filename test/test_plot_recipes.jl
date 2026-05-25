# SPDX-License-Identifier: LGPL-2.1-or-later

using RecipesBase

@testset "plot_recipes.jl — Phase 2.3" begin

    # Test the recipe machinery without requiring Plots.jl / Makie.jl
    # to be loaded. RecipesBase.apply_recipe extracts the recipe's
    # output series for direct inspection.

    @testset "ContoursError → closed polygon recipe" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour(fmin, cf, 1, 2; npoints = 8)
        @test c.valid

        # Apply the recipe; result is a Vector of RecipeData.
        kwargs = Dict{Symbol,Any}()
        recipes = RecipesBase.apply_recipe(kwargs, c)
        @test length(recipes) >= 1
        # The recipe returns (xs, ys) via the last expression — check that
        # the points are closed (first == last to close the polygon).
        rd = recipes[1]
        xs, ys = rd.args
        @test length(xs) == 9   # 8 contour points + 1 wrap-around
        @test xs[1] == xs[end]
        @test ys[1] == ys[end]
    end

    @testset "MinosError → scatter+yerror recipe" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        e = minos(fmin, cf, 1)
        @test JuMinuit.is_valid(e)

        kwargs = Dict{Symbol,Any}()
        recipes = RecipesBase.apply_recipe(kwargs, e)
        @test length(recipes) >= 1
        rd = recipes[1]
        xs, ys = rd.args
        @test xs == [1]
        @test ys[1] ≈ e.min_par_value
    end

    @testset "Vector{MinosError} → multi-param error-bar recipe" begin
        cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0]))
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        errs = minos(fmin, cf)
        @test length(errs) == 2

        kwargs = Dict{Symbol,Any}()
        recipes = RecipesBase.apply_recipe(kwargs, errs)
        @test length(recipes) >= 1
        rd = recipes[1]
        xs, ys = rd.args
        @test xs == [1, 2]
        @test length(ys) == 2
    end

    @testset "FunctionMinimum recipe" begin
        cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0]))
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])

        kwargs = Dict{Symbol,Any}()
        recipes = RecipesBase.apply_recipe(kwargs, fmin)
        @test length(recipes) >= 1
        rd = recipes[1]
        xs, ys = rd.args
        @test xs == [1, 2]
        @test ys ≈ Base.values(fmin)
    end

    @testset "BoundedFunctionMinimum recipe" begin
        cf = CostFunction(x -> sum(abs2, x .- [0.5, 1.0]))
        params = Parameters([
            MinuitParameter("a", 0.0, 0.1; lower = 0.0, upper = 1.0),
            MinuitParameter("b", 0.0, 0.1; fixed = true),
        ])
        m = migrad(cf, params)

        kwargs = Dict{Symbol,Any}()
        recipes = RecipesBase.apply_recipe(kwargs, m)
        @test length(recipes) >= 1
        rd = recipes[1]
        xs, ys = rd.args
        @test xs == [1, 2]
        @test ys ≈ m.ext_values
        # markershape attribute should distinguish fixed parameter
        @test haskey(rd.plotattributes, :markershape)
    end
end
