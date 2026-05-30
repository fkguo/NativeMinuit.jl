# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "parameters.jl — MinuitParameter + Parameters" begin

    @testset "MinuitParameter construction" begin
        p = MinuitParameter("x", 1.0, 0.1)
        @test p.name == "x"
        @test p.value == 1.0
        @test p.error == 0.1
        @test isnan(p.lower) && isnan(p.upper)
        @test !p.fixed
        @test !has_limits(p)
        @test JuMinuit.bound_kind(p) == JuMinuit.NoBounds

        # With double bounds
        p2 = MinuitParameter("y", 0.5, 0.1; lower = -1.0, upper = 1.0)
        @test has_lower_limit(p2) && has_upper_limit(p2)
        @test JuMinuit.bound_kind(p2) == JuMinuit.BothBounds

        # With upper only
        p3 = MinuitParameter("z", 0.0, 0.1; upper = 5.0)
        @test !has_lower_limit(p3) && has_upper_limit(p3)
        @test JuMinuit.bound_kind(p3) == JuMinuit.UpperOnly

        # With lower only
        p4 = MinuitParameter("w", 0.0, 0.1; lower = -5.0)
        @test has_lower_limit(p4) && !has_upper_limit(p4)
        @test JuMinuit.bound_kind(p4) == JuMinuit.LowerOnly

        # Fixed flag
        p5 = MinuitParameter("k", 1.0, 0.1; fixed = true)
        @test is_fixed(p5)

        # Bad bounds throw
        @test_throws ArgumentError MinuitParameter("x", 0.0, 0.1; lower = 1.0, upper = 0.0)
    end

    @testset "Parameters: vector construction + index maps" begin
        pars = [
            MinuitParameter("x", 1.0, 0.1),                # ext 1, int 1
            MinuitParameter("y", 2.0, 0.1; fixed = true),  # ext 2, fixed
            MinuitParameter("z", 0.5, 0.1; lower = -1, upper = 1),  # ext 3, int 2
        ]
        P = Parameters(pars)
        @test n_pars(P) == 3
        @test n_free(P) == 2
        @test length(P) == 3

        @test P.ext_of_int == [1, 3]
        @test P.int_of_ext == [1, 0, 2]
        @test ext_index(P, "x") == 1
        @test ext_index(P, "y") == 2
        @test ext_index(P, "z") == 3
        @test_throws KeyError ext_index(P, "not_here")

        @test is_fixed(P, 2)
        @test !is_fixed(P, 1)

        # Duplicate names rejected
        @test_throws ArgumentError Parameters([
            MinuitParameter("x", 0.0, 0.1),
            MinuitParameter("x", 1.0, 0.1),
        ])
    end

    @testset "Parameters: keyword-array convenience constructor" begin
        P = Parameters(
            ["a", "b", "c"], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1];
            limits = [(NaN, NaN), (-5.0, 5.0), (NaN, NaN)],
            fixed = [false, false, true],
        )
        @test n_pars(P) == 3
        @test n_free(P) == 2
        @test JuMinuit.bound_kind(P.pars[2]) == JuMinuit.BothBounds
        @test is_fixed(P, 3)

        # Dimension-mismatch guards
        @test_throws DimensionMismatch Parameters(
            ["a"], [1.0, 2.0], [0.1, 0.1])
        @test_throws DimensionMismatch Parameters(
            ["a", "b"], [1.0, 2.0], [0.1, 0.1]; limits = [(NaN, NaN)])
    end

    @testset "int ↔ ext value conversion" begin
        # All-unbounded: identity
        pars = [
            MinuitParameter("a", 1.0, 0.1),
            MinuitParameter("b", 2.0, 0.1),
        ]
        P = Parameters(pars)
        @test int_to_ext_value(P, 1, 5.0) == 5.0
        @test int_to_ext_value(P, 2, -3.0) == -3.0
        @test ext_to_int_value(P, 1, 5.0) == 5.0
        @test dint2ext_value(P, 1, 5.0) == 1.0

        # Bounded — round-trip far from bounds
        pars_b = [MinuitParameter("x", 0.0, 0.1; lower = -5.0, upper = 5.0)]
        Pb = Parameters(pars_b)
        for ext in (-3.0, 0.0, 3.0)
            int = ext_to_int_value(Pb, 1, ext)
            @test int_to_ext_value(Pb, 1, int) ≈ ext atol = 1e-12
        end
    end

    @testset "int ↔ ext vector conversion (with fixed params)" begin
        pars = [
            MinuitParameter("x", 1.0, 0.1),                              # ext 1, free
            MinuitParameter("y", 9.0, 0.1; fixed = true),                # ext 2, fixed
            MinuitParameter("z", 2.0, 0.1; lower = -5.0, upper = 5.0),   # ext 3, free, bounded
        ]
        P = Parameters(pars)
        # Internal vector has 2 entries (x, z); external has 3 (x, y_fixed, z)
        int_vec = [1.5, 0.0]   # z=0.0 internal → ext = 0.0 (mid of [-5,5])
        ext_vec = int_to_ext_vector(P, int_vec)
        @test length(ext_vec) == 3
        @test ext_vec[1] == 1.5
        @test ext_vec[2] == 9.0  # fixed param value preserved
        @test ext_vec[3] ≈ 0.0 atol = 1e-12

        # Round-trip
        new_int = ext_to_int_vector(P, ext_vec)
        @test length(new_int) == 2
        @test new_int[1] ≈ int_vec[1] atol = 1e-12
        @test new_int[2] ≈ int_vec[2] atol = 1e-12

        # Dimension guards
        @test_throws DimensionMismatch int_to_ext_vector(P, [1.0])
        @test_throws DimensionMismatch ext_to_int_vector(P, [1.0])
    end

    @testset "int_to_ext_vector! (in-place) matches allocating form" begin
        # Mix every bound kind so the buffer-reusing hot-path variant is
        # exercised across free / fixed / two-sided / lower-only /
        # upper-only parameters.
        pars = [
            MinuitParameter("free", 1.0, 0.1),                            # ext 1, free
            MinuitParameter("fix",  9.0, 0.1; fixed = true),              # ext 2, fixed
            MinuitParameter("both", 0.5, 0.1; lower = -2.0, upper = 3.0), # ext 3, two-sided
            MinuitParameter("lo",   4.0, 0.1; lower = 1.0),               # ext 4, lower-only
            MinuitParameter("hi",  -4.0, 0.1; upper = 2.0),               # ext 5, upper-only
        ]
        P = Parameters(pars)
        @test n_pars(P) == 5
        @test n_free(P) == 4

        for int_vec in ([0.3, 0.1, 1.7, -0.6], [-1.2, 2.0, 0.0, 3.3], zeros(4))
            ref = int_to_ext_vector(P, int_vec)            # allocating reference
            buf = fill(NaN, n_pars(P))
            out = JuMinuit.int_to_ext_vector!(buf, P, int_vec)
            @test out === buf                              # writes in place, returns buffer
            @test out == ref                               # bit-identical to allocating form
            @test out[2] == 9.0                            # fixed entry preserved
        end

        # ext-length guard (new in the in-place method) + int-length guard
        @test_throws DimensionMismatch JuMinuit.int_to_ext_vector!(zeros(4), P, zeros(n_free(P)))
        @test_throws DimensionMismatch JuMinuit.int_to_ext_vector!(zeros(n_pars(P)), P, [1.0])

        # The in-place transform itself must allocate nothing after warm-up,
        # otherwise the hot-path buffer reuse buys nothing.
        _alloc_inplace(b, p, v) = @allocated JuMinuit.int_to_ext_vector!(b, p, v)
        buf = Vector{Float64}(undef, n_pars(P))
        iv = [0.3, 0.1, 1.7, -0.6]
        _alloc_inplace(buf, P, iv)                         # compile
        @test _alloc_inplace(buf, P, iv) == 0
    end

    @testset "initial_int_values + initial_int_errors" begin
        # Mix of bounded, unbounded, fixed
        pars = [
            MinuitParameter("a", 1.0, 0.1),
            MinuitParameter("b", 0.0, 0.1; lower = -1.0, upper = 1.0),
            MinuitParameter("c", 5.0, 0.1; fixed = true),
        ]
        P = Parameters(pars)

        int_vals = initial_int_values(P)
        @test length(int_vals) == 2
        @test int_vals[1] == 1.0  # unbounded: identity
        # b is at the midpoint → asin(0) = 0
        @test int_vals[2] ≈ 0.0 atol = 1e-12

        int_errs = initial_int_errors(P)
        @test length(int_errs) == 2
        @test int_errs[1] == 0.1  # unbounded: identity
        # For Sin at midpoint (v=0, ext=0, werr=0.1), the two-sided C++
        # formula (parallel-review #2 B4) computes:
        #   vplu = asin(0.1) - 0  ≈  0.10017
        #   vmin = asin(-0.1) - 0 ≈ -0.10017
        #   int_err = 0.5·(|vplu| + |vmin|) ≈ 0.10017
        # The old Taylor approximation gave exactly 0.1 at the midpoint,
        # which was a coincidence (the linear limit).
        @test int_errs[2] ≈ asin(0.1) atol = 1e-12
    end

    @testset "initial_int_errors C++ parity near a bound" begin
        # At ext = 0.99 of a [-1, 1] interval, the Taylor approx blows
        # up but the C++ two-sided formula clamps gracefully.
        pars = [
            MinuitParameter("near_upper", 0.99, 0.1; lower = -1.0, upper = 1.0),
        ]
        P = Parameters(pars)
        int_errs = initial_int_errors(P)
        # Manual C++ trace:
        #   sav = 0.99
        #   sav_plus = min(0.99 + 0.1, 1.0) = 1.0  (clamped)
        #   var_plus = asin(2·(1.0 - (-1.0))/(2) - 1) = asin(1) = π/2 (clamped)
        #   sav_minus = 0.99 - 0.1 = 0.89
        #   var_minus = asin(0.89) (≈ 1.0974)
        #   var = asin(0.99) (≈ 1.4289)
        #   vplu = π/2 - 1.4289 ≈ 0.1419
        #   vmin = 1.0974 - 1.4289 ≈ -0.3315
        #   int_err ≈ 0.5·(0.1419 + 0.3315) ≈ 0.237
        # Old Taylor at v ≈ 1.4289 would give 0.1/cos(1.4289)·2 ≈ 0.71 — wildly off.
        @test 0.2 < int_errs[1] < 0.3
    end
end
