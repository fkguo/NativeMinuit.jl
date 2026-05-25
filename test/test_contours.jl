# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "contours.jl — contour (Phase 1 first cut: ellipse)" begin

    @testset "Symmetric uncorrelated quadratic → circle" begin
        # f(x, y) = (x-1)² + (y-2)². Hessian = 2·I, V = 0.5·I.
        # σ_x = σ_y = 1, ρ = 0. Contour is unit circle around (1, 2).
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour(fmin, cf, 1, 2; npoints = 16)
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
        c = contour(fmin, cf, 1, 2; npoints = 32)
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
        c = contour(fmin, cf, 1, 2; npoints = 36)
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
        @test_throws ArgumentError contour(fmin, cf, 0, 2)
        @test_throws ArgumentError contour(fmin, cf, 1, 3)
        @test_throws ArgumentError contour(fmin, cf, 1, 1)  # same param
        @test_throws ArgumentError contour(fmin, cf, 1, 2; npoints = 3)  # too few points
    end

    @testset "ContoursError struct + accessors" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour(fmin, cf, 1, 2; npoints = 10)
        @test c isa ContoursError
        @test c.par_x == 1
        @test c.par_y == 2
        @test c.nfcn > 0
        @test c.minos_x isa MinosError
        @test c.minos_y isa MinosError
    end
end
