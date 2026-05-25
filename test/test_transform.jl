# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "transform.jl — bound transformations" begin

    prec = MachinePrecision()

    # ──────────────────────────────────────────────────────
    @testset "Sin (both bounds)" begin
        L, U = -1.0, 3.0
        # int2ext on a few canonical points
        @test JuMinuit.sin_int2ext(0.0, L, U) ≈ L + 0.5 * (U - L) * 1.0
        @test JuMinuit.sin_int2ext(Float64(π) / 2, L, U) ≈ U
        @test JuMinuit.sin_int2ext(-Float64(π) / 2, L, U) ≈ L

        # Round-trip ext → int → ext
        for ext in (-0.999, 0.0, 1.5, 2.999)
            v = JuMinuit.sin_ext2int(ext, L, U, prec)
            @test JuMinuit.sin_int2ext(v, L, U) ≈ ext atol = 1e-12
        end

        # Clamping near limits: |v| comes out STRICTLY INSIDE (-π/2, π/2)
        # by the `distnn = 8·√eps2` clamp at sin_ext2int.
        v_lo = JuMinuit.sin_ext2int(L + 1e-20, L, U, prec)
        v_hi = JuMinuit.sin_ext2int(U - 1e-20, L, U, prec)
        # distnn = 8·√eps2 ≈ 1.4e-3; clamped value is π/2 ∓ distnn
        @test -Float64(π) / 2 < v_lo < -Float64(π) / 2 + 1e-2  # just inside lower
        @test  Float64(π) / 2 - 1e-2 < v_hi <  Float64(π) / 2  # just inside upper

        # Derivative
        @test JuMinuit.sin_dint2ext(0.0, L, U) ≈ 0.5 * (U - L)
        @test JuMinuit.sin_dint2ext(Float64(π) / 2, L, U) ≈ 0.0 atol = 1e-15
        @test JuMinuit.sin_dint2ext(Float64(π), L, U) ≈ -0.5 * (U - L) atol = 1e-12
    end

    # ──────────────────────────────────────────────────────
    @testset "SqrtUp (upper bound only)" begin
        U = 5.0

        # int2ext: v=0 → upper; |v|→∞ → -∞
        @test JuMinuit.sqrtup_int2ext(0.0, U) == U + 1 - sqrt(1)
        @test JuMinuit.sqrtup_int2ext(0.0, U) == U  # exact when v² + 1 = 1
        @test JuMinuit.sqrtup_int2ext(3.0, U) ≈ U + 1 - sqrt(10)
        @test JuMinuit.sqrtup_int2ext(1000.0, U) < -990

        # Round-trip
        for ext in (-100.0, 0.0, 4.0, 4.999)
            v = JuMinuit.sqrtup_ext2int(ext, U, prec)
            # Note: ext2int returns 0 when ext is too close to upper
            if ext < U - 1
                @test JuMinuit.sqrtup_int2ext(v, U) ≈ ext atol = 1e-12
            end
        end

        # Derivative — NEGATIVE for v > 0
        @test JuMinuit.sqrtup_dint2ext(0.0, U) == 0.0
        @test JuMinuit.sqrtup_dint2ext(1.0, U) < 0
        @test JuMinuit.sqrtup_dint2ext(-1.0, U) > 0
        @test JuMinuit.sqrtup_dint2ext(1.0, U) ≈ -1 / sqrt(2)
    end

    # ──────────────────────────────────────────────────────
    @testset "SqrtLow (lower bound only)" begin
        L = -2.0

        @test JuMinuit.sqrtlow_int2ext(0.0, L) == L  # v² + 1 = 1
        @test JuMinuit.sqrtlow_int2ext(3.0, L) ≈ L - 1 + sqrt(10)
        @test JuMinuit.sqrtlow_int2ext(1000.0, L) > 990

        for ext in (-1.999, 0.0, 5.0, 100.0)
            v = JuMinuit.sqrtlow_ext2int(ext, L, prec)
            if ext > L + 1
                @test JuMinuit.sqrtlow_int2ext(v, L) ≈ ext atol = 1e-12
            end
        end

        # Derivative — POSITIVE for v > 0 (opposite sign of SqrtUp)
        @test JuMinuit.sqrtlow_dint2ext(0.0, L) == 0.0
        @test JuMinuit.sqrtlow_dint2ext(1.0, L) > 0
        @test JuMinuit.sqrtlow_dint2ext(-1.0, L) < 0
        @test JuMinuit.sqrtlow_dint2ext(1.0, L) ≈ 1 / sqrt(2)
        # Sign opposition vs SqrtUp at same v:
        @test JuMinuit.sqrtlow_dint2ext(1.0, L) == -JuMinuit.sqrtup_dint2ext(1.0, 0.0)
    end

    # ──────────────────────────────────────────────────────
    @testset "bound_kind classifier" begin
        @test JuMinuit.bound_kind(NaN, NaN) == JuMinuit.NoBounds
        @test JuMinuit.bound_kind(-1.0, 1.0) == JuMinuit.BothBounds
        @test JuMinuit.bound_kind(NaN, 1.0) == JuMinuit.UpperOnly
        @test JuMinuit.bound_kind(-1.0, NaN) == JuMinuit.LowerOnly
        # Sanity guard
        @test_throws ArgumentError JuMinuit.bound_kind(1.0, 0.0)  # lower > upper
    end

    # ──────────────────────────────────────────────────────
    @testset "Dispatch int2ext / ext2int / dint2ext" begin
        # NoBounds — identity
        @test JuMinuit.int2ext(JuMinuit.NoBounds, 1.5, NaN, NaN) == 1.5
        @test JuMinuit.ext2int(JuMinuit.NoBounds, 1.5, NaN, NaN, prec) == 1.5
        @test JuMinuit.dint2ext(JuMinuit.NoBounds, 1.5, NaN, NaN) == 1.0

        # BothBounds — Sin
        @test JuMinuit.int2ext(JuMinuit.BothBounds, 0.0, -1.0, 3.0) ==
            JuMinuit.sin_int2ext(0.0, -1.0, 3.0)

        # UpperOnly — SqrtUp
        @test JuMinuit.int2ext(JuMinuit.UpperOnly, 0.0, NaN, 5.0) ==
            JuMinuit.sqrtup_int2ext(0.0, 5.0)

        # LowerOnly — SqrtLow
        @test JuMinuit.int2ext(JuMinuit.LowerOnly, 0.0, -2.0, NaN) ==
            JuMinuit.sqrtlow_int2ext(0.0, -2.0)

        # All dispatch paths are type-stable
        @test (@inferred JuMinuit.int2ext(JuMinuit.BothBounds, 0.0, -1.0, 3.0)) isa Float64
        @test (@inferred JuMinuit.dint2ext(JuMinuit.UpperOnly, 1.0, NaN, 5.0)) isa Float64
    end
end
