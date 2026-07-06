# SPDX-License-Identifier: LGPL-2.1-or-later

using LinearAlgebra: Symmetric

@testset "covariance_squeeze.jl — MnCovarianceSqueeze" begin
    prec = MachinePrecision()

    @testset "squeeze_symmetric — pure drop row/col" begin
        # Construct a 3×3 symmetric matrix and drop the middle row/col
        M = Float64[1 2 3; 0 4 5; 0 0 6]
        S = Symmetric(M, :U)
        Sq = NativeMinuit.squeeze_symmetric(S, 2)
        @test size(Sq) == (2, 2)
        # Remaining entries: (1,1), (1,3), (3,1)=(1,3), (3,3) → new indices (1,1), (1,2), (2,1), (2,2)
        @test Sq[1, 1] == 1.0
        @test Sq[1, 2] == 3.0
        @test Sq[2, 2] == 6.0
        @test Sq[2, 1] == 3.0  # symmetric read

        # Drop first row/col
        Sq1 = NativeMinuit.squeeze_symmetric(S, 1)
        @test Sq1[1, 1] == 4.0
        @test Sq1[1, 2] == 5.0
        @test Sq1[2, 2] == 6.0

        # Drop last row/col
        Sq3 = NativeMinuit.squeeze_symmetric(S, 3)
        @test Sq3[1, 1] == 1.0
        @test Sq3[1, 2] == 2.0
        @test Sq3[2, 2] == 4.0
    end

    @testset "squeeze_symmetric — error cases" begin
        S = Symmetric([1.0 2.0; 0.0 3.0], :U)
        @test_throws ArgumentError NativeMinuit.squeeze_symmetric(S, 0)
        @test_throws ArgumentError NativeMinuit.squeeze_symmetric(S, 3)
        # 1x1 cannot squeeze
        S1 = Symmetric(reshape([1.0], 1, 1), :U)
        @test_throws ArgumentError NativeMinuit.squeeze_symmetric(S1, 1)
    end

    @testset "squeeze_error — invert/squeeze/re-invert round trip" begin
        # Set up a 3×3 well-conditioned inverse-Hessian (V).
        # H = inv(V) is the Hessian. Squeezing param 2 from V should
        # produce a 2×2 V' equal to inv(squeeze(H, 2)).
        V = Symmetric(Float64[
            2.0  0.1  0.05;
            0.1  3.0  0.2;
            0.05 0.2  1.5
        ], :U)
        err = MinimumError(V, 0.001)
        sq = NativeMinuit.squeeze_error(err, 2; prec)
        @test size(sq) == (2, 2)
        @test is_valid(sq)
        @test sq.dcovar == 0.001  # preserved per C++

        # Independent oracle: build H = inv(V), drop row/col 2, invert back
        H_full = inv(Matrix(V))
        H_squeezed = H_full[[1, 3], [1, 3]]
        V_expected = inv(H_squeezed)
        for i in 1:2, j in 1:2
            @test sq.inv_hessian[i, j] ≈ V_expected[i, j] atol = 1e-10
        end
    end

    @testset "squeeze_error — first inversion (V→H) failure → VALID diagonal (audit §12)" begin
        # Rank-1 inverse-Hessian V = v·v' is singular, so the V→H inversion
        # (squeeze_error Step 1) fails. C++ `MnCovarianceSqueeze` does NOT bail
        # to a failure status here: that inversion lives inside `err.Hessian()`
        # (BasicMinimumError.cxx:20-35), which on failure returns the diagonal
        # Hessian diag(1/V[i,i]); MnCovarianceSqueeze then squeezes that diagonal
        # and re-inverts it cleanly, yielding the VALID
        # `MinimumError(squeezed, err.Dcovar())` whose diagonal is diag(V[i,i]).
        # Pre-fix NativeMinuit mis-tagged this MnInvertFailed (audit §12).
        v = [1.0, 2.0, 3.0]
        Vsing_mat = v * v'                              # rank-1 ⇒ singular
        Vsing = Symmetric(0.5 * (Vsing_mat + Vsing_mat'), :U)
        err = MinimumError(Vsing, 0.5)
        sq = NativeMinuit.squeeze_error(err, 2; prec)
        @test size(sq) == (2, 2)
        # C++-aligned status: VALID, not MnInvertFailed.
        @test is_valid(sq)
        @test !invert_failed(sq)
        @test sq.status == MnHesseValid
        # dcovar preserved from input (C++ line 86 `MinimumError(squeezed, Dcovar)`).
        @test sq.dcovar == 0.5
        # Diagonal = original V diagonal with row/col 2 dropped: [V[1,1], V[3,3]].
        @test sq.inv_hessian[1, 1] == Vsing[1, 1]       # 1.0
        @test sq.inv_hessian[2, 2] == Vsing[3, 3]       # 9.0
        @test sq.inv_hessian[1, 2] == 0.0
    end

    @testset "squeeze_error — back-inversion (squeezed H→V) failure → MnInvertFailed" begin
        # The OTHER fallback (C++ MnCovarianceSqueeze.cxx:76-84), unchanged by
        # the §12 fix: V→H succeeds but the squeezed Hessian is singular, so the
        # back-inversion (Step 3) fails → MnInvertFailed.
        #
        # Triggering this reliably needs an EXACTLY-singular squeezed Hessian:
        # a generic indefinite V leaves the recovered H only *near*-singular
        # after the V→H round-trip's float error, which Bunch–Kaufman then
        # inverts rather than rejecting. An involution V (V² = I ⇒ inv(V)=V is
        # recovered exactly, no float drift) sidesteps that. Here
        # V = diag-swap; recovering H=V and dropping row/col 2 gives the exactly
        # singular [[1,0],[0,0]].
        V = Symmetric(Float64[1 0 0; 0 0 1; 0 1 0], :U)
        err = MinimumError(V, 0.25)
        sq = NativeMinuit.squeeze_error(err, 2; prec)
        @test size(sq) == (2, 2)
        @test invert_failed(sq)
        @test sq.status == MnInvertFailed
    end
end
