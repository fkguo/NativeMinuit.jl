# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "posdef.jl — MnPosDef" begin

    @testset "Already pos-def: returns same matrix" begin
        # Strongly diagonal-dominant matrix — should pass through
        M = Float64[4.0 1.0 0.0;
                    1.0 5.0 0.5;
                    0.0 0.5 6.0]
        err = MinimumError(Symmetric(M, :U), 0.001)
        @test is_posdef_enough(err)

        new_err = make_posdef(err)
        # When already pos-def, original status preserved
        @test new_err.status == MnHesseValid
        @test new_err.dcovar == 0.001
        # Matrix essentially unchanged
        for i in 1:3, j in 1:3
            @test new_err.inv_hessian[i, j] ≈ err.inv_hessian[i, j] atol = 1e-14
        end
    end

    @testset "n=1 fast path" begin
        # Positive 1×1: returned unchanged
        e1_ok = MinimumError(Symmetric(fill(2.5, 1, 1), :U), 0.01)
        @test make_posdef(e1_ok) === e1_ok

        # Near-zero 1×1: clamped to 1.0, status = MnMadePosDef
        e1_bad = MinimumError(Symmetric(fill(1e-20, 1, 1), :U), 0.01)
        out = make_posdef(e1_bad)
        @test out.inv_hessian[1, 1] == 1.0
        @test out.status == MnMadePosDef
        # C++ tag ctor forces dcovar = 1.0, NOT the incoming 0.01 (audit §11a).
        @test out.dcovar == 1.0
    end

    @testset "n=1 boundary M==eps falls through to valid+posdef (codex §11 follow-up)" begin
        # C++ MnPosDef.cxx:37/41 use strict `<` / `>`, so a 1×1 matrix with
        # `M[1,1] == eps` exactly takes NEITHER n=1 early return → it falls
        # through to the eigenvalue gate, which forces valid+pos-def while
        # preserving the incoming dcovar. v1's `else return err` returned the
        # matrix with the incoming (stale) status, diverging at this boundary.
        eps = JuMinuit.MachinePrecision().eps
        err = MinimumError(Symmetric(fill(eps, 1, 1), :U), 0.5, MnMadePosDef, true)
        out = make_posdef(err)
        @test out.status == MnHesseValid    # forced valid (old `else` kept MnMadePosDef)
        @test JuMinuit.is_pos_def(out)
        @test out.dcovar == 0.5             # incoming dcovar preserved
    end

    @testset "Negative diagonal: dg-only fix → MnHesseValid (matches C++)" begin
        # Two-parameter matrix with one negative diagonal — dg-add path
        # fixes it. Per C++ MnPosDef.cxx:85-86 the eigenvalue-clean return
        # forces valid+pos-def (MnHesseValid), NOT MnMadePosDef — that flag is
        # only set when the eigenvalue-based padd at line 103 fires.
        M = Float64[-0.5 0.1; 0.1 2.0]
        err = MinimumError(Symmetric(M, :U), 0.0)
        @test !is_posdef_enough(err)

        new_err = make_posdef(err)
        @test new_err.status == MnHesseValid  # C++ MnPosDef.cxx:85-86 semantic
        @test new_err.dcovar == 0.0           # incoming dcovar preserved at the gate
        # All diagonals now positive
        for i in 1:2
            @test new_err.inv_hessian[i, i] > 0
        end
        # Eigenvalues now positive
        evs = JuMinuit.sym_eigvals(new_err.inv_hessian)
        @test all(λ -> λ > 0, evs)
    end

    @testset "Indefinite eigenvalues: padd applied" begin
        # Construct a matrix with positive diagonals but negative min
        # eigenvalue (one off-diagonal too large): e.g.
        # [1 5; 5 1] has eigenvalues -4, 6.
        M = Float64[1.0 5.0; 5.0 1.0]
        err = MinimumError(Symmetric(M, :U), 0.0)
        @test !is_posdef_enough(err)
        new_err = make_posdef(err)
        @test new_err.status == MnMadePosDef
        @test new_err.dcovar == 1.0   # padd path forces dcovar = 1.0 (audit §11a)
        evs = JuMinuit.sym_eigvals(new_err.inv_hessian)
        @test all(λ -> λ > 0, evs)
    end

    @testset "Eigenvalue gate drops stale status, keeps dcovar (audit §11b)" begin
        # A strongly pos-def matrix still carrying a STALE MnMadePosDef status +
        # a custom dcovar (as it could after an earlier MnPosDef event within the
        # same MIGRAD iteration). C++ MnPosDef.cxx:85-86 returns the
        # (matrix, dcovar) ctor → valid+pos-def, so the stale status MUST be
        # dropped while the incoming dcovar is preserved.
        M = Float64[4.0 1.0; 1.0 5.0]
        err = MinimumError(Symmetric(M, :U), 0.3, MnMadePosDef, true)
        @test is_posdef_enough(err)
        new_err = make_posdef(err)
        @test new_err.status == MnHesseValid     # stale MnMadePosDef dropped
        @test JuMinuit.is_pos_def(new_err)
        @test new_err.dcovar == 0.3              # incoming dcovar preserved
    end

    @testset "MinimumState overload" begin
        params = MinimumParameters([1.0, 2.0], 0.5)
        grad = FunctionGradient([0.0, 0.0], [1.0, 1.0], [1e-3, 1e-3])
        # Bad inverse-Hessian — diagonal-only fix, returns MnHesseValid
        err = MinimumError(Symmetric(Float64[-1.0 0.0; 0.0 -1.0], :U), 0.0)
        state = MinimumState(params, err, grad, 1.0, 42)
        new_state = make_posdef(state)
        # Same params/grad/edm/nfcn; new error
        @test new_state.parameters === state.parameters
        @test new_state.gradient === state.gradient
        @test new_state.edm == state.edm
        @test new_state.nfcn == state.nfcn
        @test new_state.error.status == MnHesseValid  # dg-add only, no padd
        # Now use a truly indefinite case to exercise the MnMadePosDef path:
        err_indef = MinimumError(Symmetric(Float64[1.0 5.0; 5.0 1.0], :U), 0.0)
        state_indef = MinimumState(params, err_indef, grad, 1.0, 42)
        new_state_indef = make_posdef(state_indef)
        @test new_state_indef.error.status == MnMadePosDef
    end

    @testset "Type stability" begin
        err = MinimumError(Symmetric(Float64[2.0 0.0; 0.0 3.0], :U), 0.0)
        @test (@inferred make_posdef(err)) isa MinimumError
        @test (@inferred is_posdef_enough(err)) isa Bool
    end

    @testset "make_posdef! — bit-identical in-place variant" begin
        # `make_posdef!` (hot path inside MnHesse) mutates the matrix in place
        # and returns whether MnMadePosDef would apply. It MUST be bit-for-bit
        # identical to the allocating `make_posdef` on every code path —
        # otherwise the in-place hesse refactor would silently shift results.
        upper(M) = [M[i, j] for j in 1:size(M, 1) for i in 1:j]
        function parity(M::Matrix{Float64}; dcov = 0.3)
            ref = make_posdef(MinimumError(Symmetric(copy(M), :U), dcov))
            ref_made = is_made_pos_def(ref)
            # default (self-allocated scratch)
            M1 = copy(M); made1 = JuMinuit.make_posdef!(Symmetric(M1, :U))
            # pooled scratch path
            n = size(M, 1)
            M2 = copy(M)
            made2 = JuMinuit.make_posdef!(Symmetric(M2, :U);
                                 p_buf = Matrix{Float64}(undef, n, n),
                                 s_buf = Vector{Float64}(undef, n))
            (ref = upper(parent(ref.inv_hessian)), ref_made = ref_made,
             m1 = upper(M1), made1 = made1, m2 = upper(M2), made2 = made2)
        end
        cases = Dict(
            "already-posdef" => Float64[4.0 1.0 0.0; 1.0 5.0 0.5; 0.0 0.5 6.0],
            "negative-diag"  => Float64[-0.5 0.1; 0.1 2.0],
            "indefinite"     => Float64[1.0 5.0; 5.0 1.0],
            "n1-ok"          => fill(2.5, 1, 1),
            "n1-clamp"       => fill(1e-20, 1, 1),
        )
        for (name, M) in cases
            r = parity(M)
            @test r.m1 == r.ref        # bit-identical matrix (self-scratch)
            @test r.m2 == r.ref        # bit-identical matrix (pooled scratch)
            @test r.made1 == r.ref_made
            @test r.made2 == r.ref_made
        end
        # return value semantics + in-place mutation actually happened
        Mneg = Float64[1.0 5.0; 5.0 1.0]
        S = Symmetric(copy(Mneg), :U)
        @test JuMinuit.make_posdef!(S) === true            # padd applied → MnMadePosDef
        @test parent(S) != Mneg                   # mutated in place
        Sok = Symmetric(Float64[4.0 1.0; 1.0 5.0], :U)
        @test JuMinuit.make_posdef!(Sok) === false         # gate passed → not made-posdef
        @test (@inferred JuMinuit.make_posdef!(Symmetric(Float64[2.0 0.0; 0.0 3.0], :U))) isa Bool

        # caller-supplied scratch is size-validated before the @inbounds loop:
        # an undersized buffer must raise, not silently corrupt the heap.
        S3 = Symmetric(Float64[1.0 5.0; 5.0 1.0], :U)
        @test_throws DimensionMismatch JuMinuit.make_posdef!(S3; p_buf = Matrix{Float64}(undef, 1, 1))
        @test_throws DimensionMismatch JuMinuit.make_posdef!(S3; s_buf = Vector{Float64}(undef, 1))
        @test parent(S3) == Float64[1.0 5.0; 5.0 1.0]   # untouched (threw pre-mutation)
    end
end
