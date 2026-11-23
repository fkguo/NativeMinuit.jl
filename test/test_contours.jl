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

    @testset "minos_x/minos_y are external — bounded + fixed parameter" begin
        # The high-level `contour_ellipse(m, …)` runs the contour on the
        # INTERNAL cost function, so its two axis MinosErrors are born with
        # internal indices and (for a bounded parameter) arcsin-coordinate
        # errors. They are documented as the errors along `par_x`/`par_y`,
        # so they must be published in the same external frame as `points`.
        # Needs BOTH a bound (int ≠ ext values) and a fixed parameter
        # (int ≠ ext indices) — the other contour tests have neither.
        #
        # χ²(p2, p3) = 1 + 3·(p2−2)² − 0.5·(p2−2) + 2·(p3−3)² at p1 = 0,
        # so the Δχ²=1 half-widths are 1/√3 and 1/√2 and the two axes are
        # uncorrelated.
        o3(p) = (p[1]-1)^2 + 3*(p[2]-2)^2 + 2*(p[3]-3)^2 +
                0.5*(p[1]-1)*(p[2]-2)
        m = Minuit(o3, [0.0, 0.0, 0.0]; errors = fill(0.1, 3),
                   limits = [nothing, (-9.0, 9.0), nothing])
        m.fixed[1] = true
        migrad!(m)
        ce = contour_ellipse(m, 2, 3; npoints = 12)
        @test ce.valid

        # Indices: the MinosErrors describe the same parameters as the axes.
        @test ce.minos_x.par_idx == ce.par_x == 2
        @test ce.minos_y.par_idx == ce.par_y == 3

        # Values: the external frame, i.e. `m.values` — NOT the internal
        # arcsin coordinate (≈ 0.233 for par 2, which is what the raw
        # low-level contour reports).
        @test ce.minos_x.min_par_value == m.values[2]
        @test ce.minos_y.min_par_value == m.values[3]

        # Errors: signed DISTANCES, so the endpoints must be converted and
        # re-differenced. The internal-coordinate error for par 2 is ≈ 0.067
        # — two orders of magnitude away from the external 1/√3.
        @test ce.minos_x.upper ≈ 1/sqrt(3) rtol = 0.02
        @test ce.minos_x.lower ≈ -1/sqrt(3) rtol = 0.02
        @test ce.minos_y.upper ≈ 1/sqrt(2) rtol = 0.02
        @test ce.minos_y.lower ≈ -1/sqrt(2) rtol = 0.02
        # Near-quadratic ⇒ MINOS ≈ HESSE, in the same frame as `m.errors`.
        @test ce.minos_x.upper ≈ m.errors[2] rtol = 0.02
        @test ce.minos_y.upper ≈ m.errors[3] rtol = 0.02

        # Crossing snapshots: full external vectors, fixed parameter
        # carried at its static value, scanned slot == min + error.
        for (me, idx) in ((ce.minos_x, 2), (ce.minos_y, 3))
            for (err, st) in ((me.upper, me.upper_state),
                              (me.lower, me.lower_state))
                @test st !== nothing
                @test length(st) == 3
                @test st[1] == 0.0                    # the fixed parameter
                @test st[idx] ≈ me.min_par_value + err atol = 1e-10
                # The defining MINOS condition, evaluated in the USER's
                # frame with the USER's function: FCN = fval + up.
                @test o3(st) - m.fval ≈ m.up atol = 5e-3
            end
        end

        # The ellipse GEOMETRY is unchanged: `contour_ellipse` still builds
        # it from the raw internal MinosErrors and the internal covariance,
        # and only the published axis errors are externalized. Pin that by
        # mapping the low-level (internal) ellipse point-by-point.
        ix = m.params.int_of_ext[2]
        iy = m.params.int_of_ext[3]
        ce_int = contour_ellipse(m.fmin.internal, m.fmin.internal_cf, ix, iy;
                                 npoints = 12)
        @test ce.points == [(NativeMinuit.int_to_ext_value(m.params, ix, px),
                             NativeMinuit.int_to_ext_value(m.params, iy, py))
                            for (px, py) in ce_int.points]
        # …and the raw internal errors really are the other frame.
        @test ce_int.minos_x.par_idx == ix == 1
        @test ce_int.minos_x.upper < 0.1        # arcsin coordinate
    end

    @testset "minos_x/minos_y externalization — transform edge cases" begin
        o3(p) = (p[1]-1)^2 + 3*(p[2]-2)^2 + 2*(p[3]-3)^2 +
                0.5*(p[1]-1)*(p[2]-2)

        # An upper-bound-only parameter uses `ext = upper + 1 − √(int²+1)`,
        # which DECREASES with the internal coordinate: the internal +σ
        # crossing is the external LOWER one. The two sides (values, flags,
        # states) must be swapped so `upper ≥ 0 ≥ lower` still holds.
        mu = Minuit(o3, [0.0, 0.0, 0.0]; errors = fill(0.1, 3),
                    limits = [nothing, (nothing, 9.0), nothing])
        migrad!(mu)
        ceu = contour_ellipse(mu, 2, 3; npoints = 8)
        @test ceu.minos_x.par_idx == 2
        @test ceu.minos_x.min_par_value == mu.values[2]
        @test ceu.minos_x.upper >= 0 >= ceu.minos_x.lower
        @test ceu.minos_x.upper ≈ 1/sqrt(3) rtol = 0.03
        @test ceu.minos_x.lower ≈ -1/sqrt(3) rtol = 0.03
        @test o3(ceu.minos_x.upper_state) - mu.fval ≈ mu.up atol = 1e-2
        @test o3(ceu.minos_x.lower_state) - mu.fval ≈ mu.up atol = 1e-2

        # A lower-bound-only parameter is orientation-PRESERVING.
        ml = Minuit(o3, [0.0, 0.0, 0.0]; errors = fill(0.1, 3),
                    limits = [nothing, (-9.0, nothing), nothing])
        migrad!(ml)
        cel = contour_ellipse(ml, 2, 3; npoints = 8)
        @test cel.minos_x.upper >= 0 >= cel.minos_x.lower
        @test cel.minos_x.upper ≈ 1/sqrt(3) rtol = 0.03
        @test o3(cel.minos_x.upper_state) - ml.fval ≈ ml.up atol = 1e-2

        # All-free unbounded: the identity transform must be a BIT-exact
        # no-op on the published values (the endpoint re-differencing
        # reproduces them only to within one ULP, so it is skipped).
        mf = Minuit(o3, [0.0, 0.0, 0.0]; errors = fill(0.1, 3))
        migrad!(mf)
        cef = contour_ellipse(mf, 1, 2; npoints = 8)
        raw = NativeMinuit.minos(mf.fmin.internal, mf.fmin.internal_cf, 1)
        @test cef.minos_x.min_par_value === raw.min_par_value
        @test cef.minos_x.upper === raw.upper
        @test cef.minos_x.lower === raw.lower
        @test cef.minos_x.upper_state == raw.upper_state
    end

    @testset "0.5.0 rename: deprecated `contour` forwards to contour_ellipse" begin
        # `contour` is deliberately NOT exported (Plots.contour collision) but
        # the qualified deprecated alias must keep old code working, for both
        # the low-level (fmin, cf) and the Minuit-level signatures.
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        ce_dep = NativeMinuit.contour(fmin, cf, 1, 2; npoints = 8)
        ce_new = contour_ellipse(fmin, cf, 1, 2; npoints = 8)
        @test ce_dep isa ContoursError
        @test ce_dep.points == ce_new.points
        @test :contour ∉ names(NativeMinuit)   # not exported (collision fix)

        m = Minuit(p -> (p[1] - 1.0)^2 + (p[2] - 2.0)^2, [0.0, 0.0];
                   names = ["x", "y"])
        migrad!(m)
        cm_dep = NativeMinuit.contour(m, 1, 2; npoints = 8)
        cm_new = contour_ellipse(m, "x", "y"; npoints = 8)
        @test cm_dep.points == cm_new.points
    end
end

@testset "contour_grid — iminuit-style 2D FCN grid slice" begin

    @testset "callback replacement takes effect at the next grid point" begin
        fit = Ref{Minuit}()
        seen = Tuple{Int,Float64,Float64}[]
        replacement = Ref{Any}()
        objective(label) = function (x)
            push!(seen, (label, x[1], x[2]))
            if length(seen) == 1
                fit[].fcn = replacement[]
            end
            return Float64(label)
        end
        m = Minuit(objective(1), [0.0, 0.0])
        fit[] = m
        first = m.fcn
        replacement[] = CostFunction(objective(2), 0.5)
        @test typeof(first) == typeof(replacement[])
        g = contour_grid(m, 1, 2; grid = ([3.0, 4.0], [5.0, 6.0]))
        @test seen == [(1, 3.0, 5.0), (2, 4.0, 5.0),
                       (2, 3.0, 6.0), (2, 4.0, 6.0)]
        @test g.fval == [1.0 2.0; 2.0 2.0]
        @test (ncalls(first), ncalls(m.fcn), g.up) == (1, 3, 0.5)
    end

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
        pts = mncontour(m, "x", "y"; numpoints = 16, cl = NativeMinuit.chisq_cl(1.0, 2))
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
