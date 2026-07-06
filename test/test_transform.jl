# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "transform.jl — bound transformations" begin

    prec = MachinePrecision()

    # ──────────────────────────────────────────────────────
    @testset "Sin (both bounds)" begin
        L, U = -1.0, 3.0
        # int2ext on a few canonical points
        @test NativeMinuit.sin_int2ext(0.0, L, U) ≈ L + 0.5 * (U - L) * 1.0
        @test NativeMinuit.sin_int2ext(Float64(π) / 2, L, U) ≈ U
        @test NativeMinuit.sin_int2ext(-Float64(π) / 2, L, U) ≈ L

        # Round-trip ext → int → ext
        for ext in (-0.999, 0.0, 1.5, 2.999)
            v = NativeMinuit.sin_ext2int(ext, L, U, prec)
            @test NativeMinuit.sin_int2ext(v, L, U) ≈ ext atol = 1e-12
        end

        # Clamping near limits: |v| comes out STRICTLY INSIDE (-π/2, π/2)
        # by the `distnn = 8·√eps2` clamp at sin_ext2int.
        v_lo = NativeMinuit.sin_ext2int(L + 1e-20, L, U, prec)
        v_hi = NativeMinuit.sin_ext2int(U - 1e-20, L, U, prec)
        # distnn = 8·√eps2 ≈ 1.4e-3; clamped value is π/2 ∓ distnn
        @test -Float64(π) / 2 < v_lo < -Float64(π) / 2 + 1e-2  # just inside lower
        @test  Float64(π) / 2 - 1e-2 < v_hi <  Float64(π) / 2  # just inside upper

        # Derivative
        @test NativeMinuit.sin_dint2ext(0.0, L, U) ≈ 0.5 * (U - L)
        @test NativeMinuit.sin_dint2ext(Float64(π) / 2, L, U) ≈ 0.0 atol = 1e-15
        @test NativeMinuit.sin_dint2ext(Float64(π), L, U) ≈ -0.5 * (U - L) atol = 1e-12
    end

    # ──────────────────────────────────────────────────────
    @testset "SqrtUp (upper bound only)" begin
        U = 5.0

        # int2ext: v=0 → upper; |v|→∞ → -∞
        @test NativeMinuit.sqrtup_int2ext(0.0, U) == U + 1 - sqrt(1)
        @test NativeMinuit.sqrtup_int2ext(0.0, U) == U  # exact when v² + 1 = 1
        @test NativeMinuit.sqrtup_int2ext(3.0, U) ≈ U + 1 - sqrt(10)
        @test NativeMinuit.sqrtup_int2ext(1000.0, U) < -990

        # Round-trip (codex parallel-review #3 A4-extra). C++ Ext2int
        # clamps to 0 only when yy² < 1 ⟺ (upper-ext+1)² < 1 ⟺
        # ext > upper (out of domain). For ext ≤ upper the round-trip
        # is exact. Earlier `if ext < U - 1` skip was over-restrictive.
        for ext in (-100.0, 0.0, 4.0, 4.999, U)
            v = NativeMinuit.sqrtup_ext2int(ext, U, prec)
            @test NativeMinuit.sqrtup_int2ext(v, U) ≈ ext atol = 1e-12
        end
        # At the bound exactly: ext = U → v = 0 (since yy² = 1 hits the
        # `yy2 < 1` branch via `<` strict → so v = sqrt(0) = 0). Round-trip OK.
        @test NativeMinuit.sqrtup_ext2int(U, U, prec) == 0.0
        @test NativeMinuit.sqrtup_int2ext(0.0, U) == U
        # Out-of-domain: ext > U clamps to 0
        @test NativeMinuit.sqrtup_ext2int(U + 0.1, U, prec) == 0.0

        # Derivative — NEGATIVE for v > 0
        @test NativeMinuit.sqrtup_dint2ext(0.0, U) == 0.0
        @test NativeMinuit.sqrtup_dint2ext(1.0, U) < 0
        @test NativeMinuit.sqrtup_dint2ext(-1.0, U) > 0
        @test NativeMinuit.sqrtup_dint2ext(1.0, U) ≈ -1 / sqrt(2)
    end

    # ──────────────────────────────────────────────────────
    @testset "SqrtLow (lower bound only)" begin
        L = -2.0

        @test NativeMinuit.sqrtlow_int2ext(0.0, L) == L  # v² + 1 = 1
        @test NativeMinuit.sqrtlow_int2ext(3.0, L) ≈ L - 1 + sqrt(10)
        @test NativeMinuit.sqrtlow_int2ext(1000.0, L) > 990

        # Round-trip — symmetric to SqrtUp. Clamp only when ext < L
        # (yy = ext - L + 1, yy² < 1 ⟺ ext ∈ (L-2, L) actually but
        # symmetric to upper's case: ext < L out-of-domain).
        for ext in (-1.999, L, -1.5, 0.0, 5.0, 100.0)
            v = NativeMinuit.sqrtlow_ext2int(ext, L, prec)
            @test NativeMinuit.sqrtlow_int2ext(v, L) ≈ ext atol = 1e-12
        end
        # At the bound: ext = L → v = 0
        @test NativeMinuit.sqrtlow_ext2int(L, L, prec) == 0.0
        @test NativeMinuit.sqrtlow_int2ext(0.0, L) == L
        # Out-of-domain: ext < L clamps
        @test NativeMinuit.sqrtlow_ext2int(L - 0.1, L, prec) == 0.0

        # Derivative — POSITIVE for v > 0 (opposite sign of SqrtUp)
        @test NativeMinuit.sqrtlow_dint2ext(0.0, L) == 0.0
        @test NativeMinuit.sqrtlow_dint2ext(1.0, L) > 0
        @test NativeMinuit.sqrtlow_dint2ext(-1.0, L) < 0
        @test NativeMinuit.sqrtlow_dint2ext(1.0, L) ≈ 1 / sqrt(2)
        # Sign opposition vs SqrtUp at same v:
        @test NativeMinuit.sqrtlow_dint2ext(1.0, L) == -NativeMinuit.sqrtup_dint2ext(1.0, 0.0)
    end

    # ──────────────────────────────────────────────────────
    @testset "int2ext_error — Phase 1.x D5 two-sided" begin
        # Unbounded: identity
        @test NativeMinuit.int2ext_error(NativeMinuit.NoBounds, 1.0, 0.1, NaN, NaN) == 0.1

        # Double-bounded at midpoint v=0, [-1, 1], err=0.1
        # ui = 0, du1 = sin_int2ext(0.1, -1, 1) - 0 = sin(0.1) ≈ 0.0998
        # du2 = -du1 by symmetry, avg = |du1| ≈ 0.0998
        result = NativeMinuit.int2ext_error(NativeMinuit.BothBounds, 0.0, 0.1, -1.0, 1.0)
        @test 0.09 < result < 0.11

        # Saturation: err > 1 → du1 clamped to (upper - lower) = 2
        result_sat = NativeMinuit.int2ext_error(NativeMinuit.BothBounds, 0.0, 2.0, -1.0, 1.0)
        @test result_sat > 1.0
        @test result_sat < 2.0  # 0.5 · (2.0 + |something|)

        # SqrtUp (upper-only) at v=0.5, err=0.3, upper=5
        result_up = NativeMinuit.int2ext_error(NativeMinuit.UpperOnly, 0.5, 0.3, NaN, 5.0)
        @test result_up > 0

        # SqrtLow at v=-0.5, err=0.3, lower=-5
        result_lo = NativeMinuit.int2ext_error(NativeMinuit.LowerOnly, -0.5, 0.3, -5.0, NaN)
        @test result_lo > 0

        # Near-bound case: parameter very close to the upper limit.
        # The Jacobian-only approach would under-report; two-sided
        # captures the nonlinear remapping.
        result_near = NativeMinuit.int2ext_error(NativeMinuit.BothBounds, 1.4, 0.2, -1.0, 1.0)
        @test result_near > 0
        # Jacobian-only would give |dint2ext| · 0.2 = cos(1.4) ≈ 0.17 · 0.2 = 0.034
        # The two-sided formula gives something larger (the actual range
        # of external displacement, accounting for one-sided saturation).
        # Magnitude check only.
        @test result_near < 1.0  # bounded by the range scaling
    end

    # ──────────────────────────────────────────────────────
    @testset "bound_kind classifier" begin
        @test NativeMinuit.bound_kind(NaN, NaN) == NativeMinuit.NoBounds
        @test NativeMinuit.bound_kind(-1.0, 1.0) == NativeMinuit.BothBounds
        @test NativeMinuit.bound_kind(NaN, 1.0) == NativeMinuit.UpperOnly
        @test NativeMinuit.bound_kind(-1.0, NaN) == NativeMinuit.LowerOnly
        # Sanity guard
        @test_throws ArgumentError NativeMinuit.bound_kind(1.0, 0.0)  # lower > upper
    end

    # ──────────────────────────────────────────────────────
    @testset "Dispatch int2ext / ext2int / dint2ext" begin
        # NoBounds — identity
        @test NativeMinuit.int2ext(NativeMinuit.NoBounds, 1.5, NaN, NaN) == 1.5
        @test NativeMinuit.ext2int(NativeMinuit.NoBounds, 1.5, NaN, NaN, prec) == 1.5
        @test NativeMinuit.dint2ext(NativeMinuit.NoBounds, 1.5, NaN, NaN) == 1.0

        # BothBounds — Sin
        @test NativeMinuit.int2ext(NativeMinuit.BothBounds, 0.0, -1.0, 3.0) ==
            NativeMinuit.sin_int2ext(0.0, -1.0, 3.0)

        # UpperOnly — SqrtUp
        @test NativeMinuit.int2ext(NativeMinuit.UpperOnly, 0.0, NaN, 5.0) ==
            NativeMinuit.sqrtup_int2ext(0.0, 5.0)

        # LowerOnly — SqrtLow
        @test NativeMinuit.int2ext(NativeMinuit.LowerOnly, 0.0, -2.0, NaN) ==
            NativeMinuit.sqrtlow_int2ext(0.0, -2.0)

        # All dispatch paths are type-stable
        @test (@inferred NativeMinuit.int2ext(NativeMinuit.BothBounds, 0.0, -1.0, 3.0)) isa Float64
        @test (@inferred NativeMinuit.dint2ext(NativeMinuit.UpperOnly, 1.0, NaN, 5.0)) isa Float64
    end
end
