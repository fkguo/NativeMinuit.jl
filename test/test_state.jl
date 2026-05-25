# SPDX-License-Identifier: LGPL-2.1-or-later

using LinearAlgebra: Symmetric

@testset "state.jl" begin

    # ─────────────────────────────────────────────────────────
    @testset "CovStatus enum" begin
        # All 5 variants distinct
        all_variants = (MnHesseValid, MnHesseFailed, MnMadePosDef,
                        MnInvertFailed, MnNotPosDef)
        @test length(unique(Int.(all_variants))) == 5

        # Integer values per design
        @test Int(MnHesseValid)   == 0
        @test Int(MnHesseFailed)  == 1
        @test Int(MnMadePosDef)   == 2
        @test Int(MnInvertFailed) == 3
        @test Int(MnNotPosDef)    == 4

        # @enum is isbits → zero-alloc enum dispatch
        @test isbits(MnHesseValid)
    end

    # ─────────────────────────────────────────────────────────
    @testset "MinimumParameters" begin
        # Invalid sentinel
        p_inv = MinimumParameters(5)
        @test length(p_inv) == 5
        @test p_inv.fval == 0.0
        @test !is_valid(p_inv)
        @test !has_step_size(p_inv)
        @test all(p_inv.x .== 0)
        @test all(p_inv.dirin .== 0)

        # Valid with explicit fval, no step
        x = [1.0, 2.0, 3.0]
        p_xfv = MinimumParameters(x, 0.5)
        @test length(p_xfv) == 3
        @test p_xfv.fval == 0.5
        @test is_valid(p_xfv)
        @test !has_step_size(p_xfv)
        @test p_xfv.x === x  # shared by reference (ROADMAP §2.2)
        @test all(p_xfv.dirin .== 0)

        # Fully valid
        dirin = [0.1, 0.1, 0.1]
        p_full = MinimumParameters(x, dirin, 0.5)
        @test is_valid(p_full)
        @test has_step_size(p_full)
        @test p_full.x === x        # shared
        @test p_full.dirin === dirin  # shared

        # Dimension mismatch throws
        @test_throws DimensionMismatch MinimumParameters([1.0, 2.0],
                                                          [0.1, 0.1, 0.1], 0.0)

        # Shared-by-reference semantic: mutating the input vector
        # changes the wrapper's view (no defensive copy)
        x_mut = [1.0, 2.0]
        p_mut = MinimumParameters(x_mut, 0.0)
        x_mut[1] = 99.0
        @test p_mut.x[1] == 99.0

        # Type stability
        @test (@inferred MinimumParameters(3)) isa MinimumParameters
        @test (@inferred MinimumParameters(x, 1.0)) isa MinimumParameters
        @test (@inferred MinimumParameters(x, dirin, 1.0)) isa MinimumParameters
    end

    # ─────────────────────────────────────────────────────────
    @testset "FunctionGradient" begin
        # Invalid sentinel
        g_inv = FunctionGradient(4)
        @test length(g_inv) == 4
        @test !is_valid(g_inv)
        @test !is_analytical(g_inv)
        @test all(g_inv.grad .== 0)

        # Full constructor
        grad = [0.1, 0.2, 0.3]
        g2 = [1.0, 1.0, 1.0]
        gstep = [1e-3, 1e-3, 1e-3]
        g = FunctionGradient(grad, g2, gstep)
        @test length(g) == 3
        @test is_valid(g)
        @test !is_analytical(g)
        @test g.grad === grad  # shared by reference
        @test g.g2 === g2
        @test g.gstep === gstep

        # Analytical flag
        g_ana = FunctionGradient(grad, g2, gstep; analytical = true)
        @test is_analytical(g_ana)
        @test is_valid(g_ana)

        # Dimension mismatch
        @test_throws DimensionMismatch FunctionGradient([1.0],
                                                         [1.0, 2.0], [1e-3])
    end

    # ─────────────────────────────────────────────────────────
    @testset "MinimumError" begin
        # Invalid sentinel
        e_inv = MinimumError(3)
        @test size(e_inv) == (3, 3)
        @test e_inv.dcovar == 0.0
        @test e_inv.status == MnHesseValid
        @test !is_available(e_inv)
        @test !is_valid(e_inv)

        # Valid Hesse error
        M = Float64[1.0 0.1 0.0;
                    0.1 2.0 0.2;
                    0.0 0.2 3.0]
        e_ok = MinimumError(M, 0.001)
        @test e_ok.status == MnHesseValid
        @test e_ok.dcovar == 0.001
        @test is_available(e_ok)
        @test is_valid(e_ok)
        @test is_accurate(e_ok)
        @test is_pos_def(e_ok)
        @test !is_made_pos_def(e_ok)
        @test !hesse_failed(e_ok)
        @test !invert_failed(e_ok)
        @test e_ok.inv_hessian isa Symmetric{Float64,Matrix{Float64}}

        # Symmetric inv_hessian: writing the wrapper reads symmetrically
        @test e_ok.inv_hessian[1, 2] == e_ok.inv_hessian[2, 1]

        # Failure modes — predicates match C++ BasicMinimumError tag-ctor
        # semantics (parallel-review #2 A1). Per BasicMinimumError.h:55-75:
        # tag (fValid, fPosDef, fDCovar):
        #   MnHesseFailed   (false, false, 1.0)
        #   MnMadePosDef    (true,  false, 1.0)
        #   MnInvertFailed  (false, true,  1.0)
        #   MnNotPosDef     (false, false, 1.0)
        e_hesse_failed = MinimumError(M, MnHesseFailed)
        @test hesse_failed(e_hesse_failed)
        @test !is_valid(e_hesse_failed)   # fValid=false per C++
        @test !is_pos_def(e_hesse_failed) # fPosDef=false per C++
        @test e_hesse_failed.dcovar == 1.0
        @test !is_accurate(e_hesse_failed) # dcov ≥ 0.1

        e_invert = MinimumError(M, MnInvertFailed)
        @test invert_failed(e_invert)
        @test !is_valid(e_invert)
        @test is_pos_def(e_invert)        # fPosDef=true per C++
        @test e_invert.dcovar == 1.0

        e_posdef = MinimumError(M, MnMadePosDef)
        @test is_made_pos_def(e_posdef)
        @test is_valid(e_posdef)          # fValid=true per C++
        @test !is_pos_def(e_posdef)       # fPosDef=false per C++
        @test e_posdef.dcovar == 1.0
        @test !is_accurate(e_posdef)

        e_notpd = MinimumError(M, MnNotPosDef)
        @test !is_pos_def(e_notpd)
        @test !is_valid(e_notpd)
        @test e_notpd.dcovar == 1.0

        # Already-Symmetric input passes through (no double-wrap)
        Ms = Symmetric(M, :U)
        e_sym = MinimumError(Ms, 0.001)
        @test e_sym.inv_hessian isa Symmetric{Float64,Matrix{Float64}}
    end

    # ─────────────────────────────────────────────────────────
    @testset "MinimumState" begin
        # Invalid sentinel
        s_inv = MinimumState(3)
        @test length(s_inv) == 3
        @test !is_valid(s_inv)
        @test !has_parameters(s_inv)
        @test !has_covariance(s_inv)

        # Scalar-only (Simplex/Scan style)
        s_scalar = MinimumState(1.23, 0.01, 17)
        @test fval(s_scalar) == 1.23
        @test edm(s_scalar) == 0.01
        @test nfcn(s_scalar) == 17
        @test !is_valid(s_scalar)

        # Params-only
        params = MinimumParameters([1.0, 2.0], 5.0)
        s_p = MinimumState(params, 0.001, 42)
        @test has_parameters(s_p)
        @test !has_covariance(s_p)
        @test fval(s_p) == 5.0
        @test edm(s_p) == 0.001
        @test nfcn(s_p) == 42

        # Full state
        M = Float64[1.0 0.1; 0.1 2.0]
        err = MinimumError(M, 0.001)
        grad = FunctionGradient([0.0, 0.0], [1.0, 2.0], [1e-3, 1e-3])
        s = MinimumState(params, err, grad, 1e-10, 73)
        @test is_valid(s)
        @test has_parameters(s)
        @test has_covariance(s)
        @test fval(s) == 5.0
        @test edm(s) == 1e-10
        @test nfcn(s) == 73

        # Shared-by-reference: same vector across param / state lifetime
        @test s.parameters === params
        @test s.error === err
        @test s.gradient === grad
    end
end
