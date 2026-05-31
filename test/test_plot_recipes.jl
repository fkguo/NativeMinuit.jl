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

# ─────────────────────────────────────────────────────────────────────────────
# Error-analysis recipes: MC Δχ² samples, bootstrap/jackknife, multimodal modes.
# All exercised through `RecipesBase.apply_recipe` — no Plots/Makie backend.
# ─────────────────────────────────────────────────────────────────────────────

# Deterministic golden-angle blob (no Random dependency); `rad` is scalar.
function _eacloud(c::Vector{Float64}, rad::Float64, n::Int)
    g = π * (3 - sqrt(5.0))
    X = Matrix{Float64}(undef, n, 2)
    for k in 1:n
        ρ = sqrt(k / n)
        θ = k * g
        X[k, 1] = c[1] + rad * ρ * cos(θ)
        X[k, 2] = c[2] + rad * ρ * sin(θ)
    end
    return X
end

@testset "plot_recipes.jl — error-analysis recipes" begin

    @testset "get_contours_samples → Δχ² sample scatter" begin
        f2(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f2, [0.0, 0.0]; names = ["a", "b"], error = [0.5, 0.5])
        migrad!(m)
        r = get_contours_samples(m; nsamples = 1500, seed = 1)
        @test r.n_accepted >= 1

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
        @test length(recipes) == 1
        rd = recipes[1]
        @test rd.plotattributes[:seriestype] === :scatter
        # default pair = first two free parameters; coloured by per-sample Δχ².
        @test rd.plotattributes[:marker_z] == r.delta_chisq_values
        @test rd.plotattributes[:xguide] == "a"
        @test rd.plotattributes[:yguide] == "b"
        xs, ys = rd.args
        @test xs == view(r.samples, :, 1)
        @test ys == view(r.samples, :, 2)
        @test length(xs) == size(r.samples, 1) == r.n_accepted

        # `vars` override (by name) swaps the axes and is consumed, not leaked.
        kw = Dict{Symbol,Any}(:vars => ("b", "a"))
        rd2 = RecipesBase.apply_recipe(kw, r)[1]
        @test rd2.args[1] == view(r.samples, :, 2)
        @test rd2.args[2] == view(r.samples, :, 1)
        @test !haskey(rd2.plotattributes, :vars)
        @test !haskey(kw, :vars)

        # `vars` override by integer index, too.
        rd3 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => (2, 1)), r)[1]
        @test rd3.plotattributes[:xguide] == "b"
        @test rd3.plotattributes[:yguide] == "a"
    end

    @testset "get_contours_samples → 1-free-parameter degradation" begin
        f2(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(f2, [0.0, 0.0]; names = ["a", "b"], error = [0.5, 0.5],
                   fixed = [false, true])
        migrad!(m)
        r = get_contours_samples(m; nsamples = 1500, seed = 3)
        @test size(r.samples, 2) == 1            # only one free parameter

        rd = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)[1]
        @test rd.plotattributes[:seriestype] === :scatter
        @test rd.plotattributes[:yguide] == "Δχ²"   # value-vs-Δχ² fallback
        xs, ys = rd.args
        @test xs == view(r.samples, :, 1)
        @test ys == r.delta_chisq_values
    end

    # Shared resampling fixture: a noisy-enough linear fit so the resampled
    # distributions are non-degenerate (no Random dependency — a deterministic
    # wiggle on top of the line).
    Nfit = 30
    xfit = collect(range(0.0, 5.0; length = Nfit))
    yfit = 2.0 .* xfit .+ 1.0 .+ 0.25 .* sinpi.(xfit ./ 2)
    dfit = Data(xfit, yfit, fill(0.3, Nfit))
    linmodel(xi, p) = p[1] * xi + p[2]
    mfit = model_fit(linmodel, dfit, [1.0, 0.0]; name = ["slope", "intercept"])
    migrad!(mfit)

    @testset "BootstrapResult → histogram + estimate/CI lines" begin
        bs = bootstrap(linmodel, dfit, mfit; nresample = 80, seed = 2)
        @test bs isa BootstrapResult

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), bs)
        @test length(recipes) == 3            # estimate vline + CI vline + histogram

        hi = findfirst(rd -> get(rd.plotattributes, :seriestype, nothing) === :histogram,
                       recipes)
        @test hi !== nothing
        hist = recipes[hi]
        # default parameter = first varying column (slope); finite rows only.
        @test hist.args[1] == JuMinuit._finite_col(bs.samples, 1)
        @test hist.plotattributes[:xguide] == "slope"

        # the two reference lines mark the estimate and the percentile CI.
        vlines = filter(rd -> get(rd.plotattributes, :seriestype, nothing) === :vline,
                        recipes)
        @test length(vlines) == 2
        est = vlines[findfirst(rd -> rd.plotattributes[:label] == "estimate", vlines)]
        @test est.args[1] == [bs.estimate[1]]
        ci = vlines[findfirst(rd -> occursin("CI", rd.plotattributes[:label]), vlines)]
        @test ci.args[1] == [bs.ci_lower[1], bs.ci_upper[1]]

        # `par` override (by name) selects a different column.
        bd = RecipesBase.apply_recipe(Dict{Symbol,Any}(:par => "intercept"), bs)
        hj = findfirst(rd -> get(rd.plotattributes, :seriestype, nothing) === :histogram, bd)
        @test bd[hj].plotattributes[:xguide] == "intercept"
        @test bd[hj].args[1] == JuMinuit._finite_col(bs.samples, 2)
    end

    @testset "JackknifeResult → histogram + estimate/mean lines" begin
        jk = jackknife(linmodel, dfit, mfit)
        @test jk isa JackknifeResult

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), jk)
        @test length(recipes) == 3
        hi = findfirst(rd -> get(rd.plotattributes, :seriestype, nothing) === :histogram,
                       recipes)
        @test hi !== nothing
        @test recipes[hi].args[1] == JuMinuit._finite_col(jk.samples, 1)

        vlines = filter(rd -> get(rd.plotattributes, :seriestype, nothing) === :vline,
                        recipes)
        @test length(vlines) == 2
        labels = [rd.plotattributes[:label] for rd in vlines]
        @test "estimate" in labels
        @test any(l -> occursin("θ̄", l), labels)
    end

    @testset "SolutionModes → colour-per-mode scatter (with samples)" begin
        fq(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], error = [0.1, 0.1])
        migrad!(m)
        S = vcat(_eacloud([1.0, 2.0], 0.2, 60), _eacloud([5.0, 6.0], 0.2, 40))
        modes = find_solution_modes(S, m)
        @test length(modes) == 2

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes, S)
        @test length(recipes) == 4            # 2 modes × (cluster scatter + rep star)

        scatters = filter(rd -> get(rd.plotattributes, :markershape, nothing) !== :star5,
                          recipes)
        stars = filter(rd -> get(rd.plotattributes, :markershape, nothing) === :star5,
                       recipes)
        @test length(scatters) == 2 && length(stars) == 2
        # one coloured series per mode, in rank order.
        @test scatters[1].plotattributes[:seriescolor] == 1
        @test scatters[2].plotattributes[:seriescolor] == 2
        @test scatters[1].plotattributes[:label] == "main"
        @test scatters[2].plotattributes[:label] == "mode 2"
        # each cluster scatter holds exactly that mode's member rows.
        @test length(scatters[1].args[1]) == modes[1].n_points
        @test length(scatters[2].args[1]) == modes[2].n_points
        # full-width samples ⇒ the representative is marked directly.
        @test stars[1].args == ([modes[1].representative[1]], [modes[1].representative[2]])
        @test stars[2].args == ([modes[2].representative[1]], [modes[2].representative[2]])
        # representative stars carry no extra legend entry.
        @test stars[1].plotattributes[:primary] == false
        @test scatters[1].plotattributes[:xguide] == "a"
        @test scatters[1].plotattributes[:yguide] == "b"

        # `vars` override flows to both the scatter columns and the rep marker.
        rec_v = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => (2, 1)), modes, S)
        sc = filter(rd -> get(rd.plotattributes, :markershape, nothing) !== :star5, rec_v)
        @test sc[1].args[1] == view(S, modes[1].member_indices, 2)
        @test sc[1].args[2] == view(S, modes[1].member_indices, 1)
    end

    @testset "SolutionModes alone → bounding boxes + reps" begin
        fq(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m = Minuit(fq, [0.0, 0.0]; names = ["a", "b"], error = [0.1, 0.1])
        migrad!(m)
        S = vcat(_eacloud([1.0, 2.0], 0.2, 50), _eacloud([5.0, 6.0], 0.2, 50))
        modes = find_solution_modes(S, m)
        @test length(modes) == 2

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes)
        @test length(recipes) == 4            # 2 modes × (box shape + rep star)
        shapes = filter(rd -> get(rd.plotattributes, :seriestype, nothing) === :shape,
                        recipes)
        @test length(shapes) == 2
        # each box is a closed 5-vertex rectangle from param_ranges.
        bxs, bys = shapes[1].args
        @test length(bxs) == 5 && length(bys) == 5
        @test bxs[1] == bxs[end] && bys[1] == bys[end]
        xlo, xhi = modes[1].param_ranges[1]
        @test minimum(bxs) == xlo && maximum(bxs) == xhi
        @test shapes[1].plotattributes[:label] == "main"

        # single SolutionMode → its own box + representative star.
        sm = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes[1])
        @test any(rd -> get(rd.plotattributes, :seriestype, nothing) === :shape, sm)
        star = sm[findfirst(rd -> get(rd.plotattributes, :markershape, nothing) === :star5, sm)]
        @test star.args == ([modes[1].representative[1]], [modes[1].representative[2]])
    end

    @testset "SolutionModes with free-width samples → centroid markers" begin
        # 3-parameter fit, 3rd fixed; cluster on free-only (npar=2) sample width.
        f3(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + (x[3] - 3.0)^2
        m = Minuit(f3, [0.0, 0.0, 3.0]; names = ["x", "y", "z"],
                   error = [0.1, 0.1, 0.1], fixed = [false, false, true])
        migrad!(m)
        Sfree = vcat(_eacloud([1.0, 2.0], 0.2, 60), _eacloud([1.0, 7.0], 0.2, 40))
        modes = find_solution_modes(Sfree, m)
        @test length(modes) == 2
        @test length(modes.param_names) == 3            # full external names
        @test size(Sfree, 2) == 2                       # free-only sample width

        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes, Sfree)
        @test length(recipes) == 4
        # free-width ⇒ generic axis labels and centroid (not representative) stars.
        scatters = filter(rd -> get(rd.plotattributes, :markershape, nothing) !== :star5,
                          recipes)
        @test scatters[1].plotattributes[:xguide] == "parameter 1"
        stars = filter(rd -> get(rd.plotattributes, :markershape, nothing) === :star5,
                       recipes)
        cx = sum(view(Sfree, modes[1].member_indices, 1)) / modes[1].n_points
        @test stars[1].args[1][1] ≈ cx
    end

    @testset "SolutionModes → 1-free-parameter degradation" begin
        # A 1-free-parameter fit yields width-1 samples; find_solution_modes
        # accepts them (ncol == ndim == 1), so the modes recipes must degrade to
        # a 1-D layout rather than erroring on a missing 2nd column (a crash the
        # default vars=(1,2) would otherwise cause).
        f1(x) = (x[1] - 1.0)^2
        m = Minuit(f1, [0.0]; names = ["a"], error = [0.1])
        migrad!(m)
        S = reshape(vcat(fill(1.0, 50), fill(5.0, 50)) .+ 0.01 .* sin.(1:100), 100, 1)
        @test size(S, 2) == 1                          # single free parameter
        modes = find_solution_modes(S, m)
        @test length(modes) >= 1
        nm = length(modes)

        # with samples: 1-D clusters along x, separated on y by mode index.
        recs  = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes, S)
        @test length(recs) == 2 * nm                   # per mode: cluster scatter + rep star
        scs   = filter(rd -> get(rd.plotattributes, :markershape, nothing) !== :star5, recs)
        stars = filter(rd -> get(rd.plotattributes, :markershape, nothing) === :star5, recs)
        @test length(scs) == nm && length(stars) == nm
        @test scs[1].args[1] == view(S, modes[1].member_indices, 1)
        @test all(==(1.0), scs[1].args[2])             # mode 1 drawn at y = 1
        @test scs[1].plotattributes[:yguide] == "mode"
        @test stars[1].args == ([modes[1].representative[1]], [1.0])

        # modes alone: 1-D range segments at y = mode index + rep stars.
        only_m = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes)
        @test length(only_m) == 2 * nm
        segs = filter(rd -> get(rd.plotattributes, :seriestype, nothing) === :path, only_m)
        @test length(segs) == nm
        lo, hi = modes[1].param_ranges[1]
        @test segs[1].args == ([lo, hi], [1.0, 1.0])

        # single SolutionMode: 1-D range segment at y = 0 + rep star.
        sm  = RecipesBase.apply_recipe(Dict{Symbol,Any}(), modes[1])
        seg = sm[findfirst(rd -> get(rd.plotattributes, :seriestype, nothing) === :path, sm)]
        @test seg.args[2] == [0.0, 0.0]
        star = sm[findfirst(rd -> get(rd.plotattributes, :markershape, nothing) === :star5, sm)]
        @test star.args == ([modes[1].representative[1]], [0.0])
    end
end
