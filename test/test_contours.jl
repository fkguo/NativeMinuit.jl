# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "contours.jl — contour (Phase 1 first cut: ellipse)" begin

    @testset "Symmetric uncorrelated quadratic → circle" begin
        # f(x, y) = (x-1)² + (y-2)². Hessian = 2·I, V = 0.5·I.
        # σ_x = σ_y = 1, ρ = 0. Contour is unit circle around (1, 2).
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour_ellipse(fmin, cf, 1, 2; npoints = 16)
        @test c.valid
        @test length(c.points) == 16
        # Every point should be approximately at distance 1 from the minimum
        for (x, y) in c.points
            d = sqrt((x - 1.0)^2 + (y - 2.0)^2)
            @test d ≈ 1.0 atol = 0.05
        end
    end

    @testset "Correlated quadratic — ellipse with non-zero ρ" begin
        # f(x, y) = x² + y² + 2·0.5·x·y (correlated with ρ=0.5 in H)
        # H = [[2, 1], [1, 2]], V = inv(H) = (1/3)·[[2, -1], [-1, 2]]
        # σ_x = σ_y = sqrt(2/3), ρ = -1/2
        cf = CostFunction(x -> x[1]^2 + x[2]^2 + x[1]*x[2])
        fmin = migrad(cf, [1.0, 1.0], [0.1, 0.1])
        c = contour_ellipse(fmin, cf, 1, 2; npoints = 32)
        @test c.valid
        @test length(c.points) == 32
        # Points should be on the ellipse — verify mean radius ≈ σ
        σ_eff = sqrt(2/3)
        radii = [sqrt(x^2 + y^2) for (x, y) in c.points]
        # The mean radius for a correlated ellipse with σ_x=σ_y=σ
        # and |ρ|=0.5 is bounded by σ·sqrt(1±0.5) ≈ σ·{0.71, 1.22}
        @test minimum(radii) > 0.5 * σ_eff
        @test maximum(radii) < 2.0 * σ_eff
    end

    @testset "Asymmetric MINOS displacement-sign selector (C2 regression)" begin
        # When ρ ≠ 0, the y-displacement sign can flip relative to sin(θ).
        # v1 of contour selected `e_y` by `sign(sin θ)` and could pick the
        # wrong asymmetric radius. The fix uses the actual displacement
        # sign `ρ·cos θ + sqrt(1−ρ²)·sin θ`.
        #
        # Use an asymmetric FCN: f = (x-1)² + (y-1)² + x·y (slightly
        # correlated; symmetric MINOS makes the sign-blind bug invisible).
        # Here we just verify the contour is well-formed and respects
        # the analytical Hessian-based ellipse approximation.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 1.0)^2 + 0.5 * x[1] * x[2])
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour_ellipse(fmin, cf, 1, 2; npoints = 36)
        @test c.valid
        # Every point should be at a sane distance from the minimum (no
        # wild sign-flip excursions)
        center_x = Base.values(fmin)[1]
        center_y = Base.values(fmin)[2]
        for (x, y) in c.points
            d = sqrt((x - center_x)^2 + (y - center_y)^2)
            @test 0.1 < d < 5.0
        end
    end

    @testset "Argument validation" begin
        cf = CostFunction(x -> sum(abs2, x))
        fmin = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test_throws ArgumentError contour_ellipse(fmin, cf, 0, 2)
        @test_throws ArgumentError contour_ellipse(fmin, cf, 1, 3)
        @test_throws ArgumentError contour_ellipse(fmin, cf, 1, 1)  # same param
        @test_throws ArgumentError contour_ellipse(fmin, cf, 1, 2; npoints = 3)  # too few points
    end

    @testset "ContoursError struct + accessors" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour_ellipse(fmin, cf, 1, 2; npoints = 10)
        @test c isa ContoursError
        @test c.par_x == 1
        @test c.par_y == 2
        @test c.nfcn > 0
        @test c.minos_x isa MinosError
        @test c.minos_y isa MinosError
    end

    @testset "0.5.0 rename: deprecated `contour` forwards to contour_ellipse" begin
        # `contour` is deliberately NOT exported (Plots.contour collision) but
        # the qualified deprecated alias must keep old code working, for both
        # the low-level (fmin, cf) and the Minuit-level signatures.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        ce_dep = JuMinuit.contour(fmin, cf, 1, 2; npoints = 8)
        ce_new = contour_ellipse(fmin, cf, 1, 2; npoints = 8)
        @test ce_dep isa ContoursError
        @test ce_dep.points == ce_new.points
        @test :contour ∉ names(JuMinuit)   # not exported (collision fix)

        m = Minuit(p -> (p[1] - 1.0)^2 + (p[2] - 2.0)^2, [0.0, 0.0];
                   names = ["x", "y"])
        migrad!(m)
        cm_dep = JuMinuit.contour(m, 1, 2; npoints = 8)
        cm_new = contour_ellipse(m, "x", "y"; npoints = 8)
        @test cm_dep.points == cm_new.points
    end
end

@testset "contour_grid — iminuit-style 2D FCN grid slice" begin

    @testset "values, orientation, destructuring" begin
        # 2 free params: F[i,j] must be EXACTLY fcn(xs[i], ys[j]).
        f2(p) = (p[1] - 1.0)^2 + 2.0 * (p[2] + 0.5)^2 + 0.3 * p[1] * p[2]
        m = Minuit(f2, [0.0, 0.0]; names = ["a", "b"])
        migrad!(m)
        g = contour_grid(m, "a", "b"; size = 21, bound = 2)
        @test g isa ContourGrid
        @test g.name_x == "a" && g.name_y == "b"
        @test length(g.x) == 21 && length(g.y) == 21
        @test size(g.fval) == (21, 21)
        @test g.up == 1.0
        @test !g.subtracted
        # x-major orientation: fval[i, j] = FCN(x[i], y[j])
        @test g.fval[3, 17] ≈ f2([g.x[3], g.y[17]]) rtol = 1e-12
        @test g.fval[17, 3] ≈ f2([g.x[17], g.y[3]]) rtol = 1e-12
        # iminuit-style destructuring
        xs, ys, F = g
        @test xs === g.x && ys === g.y && F === g.fval
        # central marker = best-fit values
        @test g.value_x ≈ m.values[1] && g.value_y ≈ m.values[2]
        # bound=k → value ± k·σ
        @test g.x[1] ≈ m.values[1] - 2 * m.errors[1] atol = 1e-9
        @test g.x[end] ≈ m.values[1] + 2 * m.errors[1] atol = 1e-9
    end

    @testset "slice vs profile: 2 free params ⇒ identical; correlated 3rd ⇒ conditional" begin
        # f = p·A·p with A = [1 0 0.9; 0 1 0; 0.9 0 1]  (up = 1):
        #   marginal  σ_x = √((A⁻¹)₁₁) = √(1/0.19) ≈ 2.2942  (HESSE/MINOS/mncontour)
        #   slice     σ_x (y,z pinned at 0) = 1                (= √(1−ρ²)·σ_x, ρ=0.9)
        # This is the documented "a slice is NOT a confidence region" caveat,
        # verified quantitatively.
        f3(p) = p[1]^2 + p[2]^2 + p[3]^2 + 1.8 * p[1] * p[3]
        m = Minuit(f3, [0.5, 0.5, 0.5]; names = ["x", "y", "z"])
        migrad!(m)
        @test m.errors[1] ≈ sqrt(1 / 0.19) rtol = 1e-3

        # fine explicit grid so resolution doesn't blur the crossing
        g = contour_grid(m, "x", "y";
                          bound = ((-1.5, 1.5), (-0.2, 0.2)), size = 121,
                          subtract_min = true)
        @test g.subtracted
        @test minimum(g.fval) == 0.0
        jy = argmin(abs.(g.y .- m.values[2]))
        row = @view g.fval[:, jy]
        xex = maximum(abs(g.x[i] - m.values[1])
                      for i in eachindex(g.x) if row[i] <= 1.0)
        # conditional σ_x = 1.0 (grid spacing 3/120 = 0.025)
        @test 0.93 <= xex <= 1.01
        # …while the PROFILE extent at Δχ²=1 (the C++ MnContours curve,
        # cl = chisq_cl(1,2) ≈ 0.3935 in mncontour's joint-cl language) is
        # the marginal σ_x ≈ 2.294:
        pts = mncontour(m, "x", "y"; numpoints = 16, cl = JuMinuit.chisq_cl(1.0, 2))
        xprof = maximum(abs(p[1] - m.values[1]) for p in pts)
        @test xprof ≈ sqrt(1 / 0.19) rtol = 0.02
        # the slice is ≈ √(1−ρ²) = 0.436 of the profile — far tighter
        @test xex < 0.55 * xprof
    end

    @testset "explicit grid, argument validation, fixed params" begin
        f2(p) = (p[1] - 1.0)^2 + (p[2] + 1.0)^2 + p[3]^2
        m = Minuit(f2, [0.0, 0.0, 0.0]; names = ["a", "b", "c"],
                   fixed = [false, false, true])
        migrad!(m)
        # explicit grid axes override size/bound
        gx = collect(0.0:0.25:2.0)
        gy = collect(-2.0:0.5:0.0)
        g = contour_grid(m, 1, 2; grid = (gx, gy))
        @test g.x == gx && g.y == gy
        @test size(g.fval) == (length(gx), length(gy))
        # fixed parameter `c` stays pinned in every evaluation
        @test g.fval[1, 1] ≈ f2([gx[1], gy[1], m.values[3]]) rtol = 1e-12
        # validation
        @test_throws ArgumentError contour_grid(m, 1, 1)            # same par
        @test_throws ArgumentError contour_grid(m, 1, "c")          # fixed par
        @test_throws ArgumentError contour_grid(m, 1, 2; size = 1)  # size < 2
        @test_throws ArgumentError contour_grid(m, 1, 2; grid = (gx,))  # bad grid
    end

    @testset "limits clip the numeric bound" begin
        f2(p) = (p[1] - 0.5)^2 + (p[2] - 0.5)^2
        m = Minuit(f2, [0.4, 0.4]; names = ["a", "b"],
                   limits = [(0.0, 1.0), nothing])
        migrad!(m)
        g = contour_grid(m, "a", "b"; size = 11, bound = 50)  # huge bound
        @test g.x[1] >= 0.0 && g.x[end] <= 1.0   # clipped to a's limits
        @test g.y[end] > 10.0                     # b unclipped
    end
end
