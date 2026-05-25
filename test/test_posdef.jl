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
    end

    @testset "Negative diagonal: dg-only fix → MnHesseValid (matches C++)" begin
        # Two-parameter matrix with one negative diagonal — dg-add path
        # fixes it. Per C++ MnPosDef.cxx:86 the eigenvalue-clean return
        # preserves original status (not MnMadePosDef — that flag is
        # only set when eigenvalue-based padd at line 103 fires).
        M = Float64[-0.5 0.1; 0.1 2.0]
        err = MinimumError(Symmetric(M, :U), 0.0)
        @test !is_posdef_enough(err)

        new_err = make_posdef(err)
        @test new_err.status == MnHesseValid  # C++ MnPosDef.cxx:86 semantic
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
        evs = JuMinuit.sym_eigvals(new_err.inv_hessian)
        @test all(λ -> λ > 0, evs)
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
end
