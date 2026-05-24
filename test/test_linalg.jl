# SPDX-License-Identifier: LGPL-2.1-or-later

using LinearAlgebra
using LinearAlgebra: BLAS

@testset "linalg.jl" begin

    @testset "SYMMETRIC_UPLO convention" begin
        @test JuMinuit.SYMMETRIC_UPLO === :U
    end

    # ─────────────────────────────────────────────────────────
    @testset "sym_rank1_update!" begin
        # Known small case: S = I + α*x*x'
        S = Symmetric(Matrix{Float64}(I, 3, 3), :U)
        x = [1.0, 2.0, 3.0]
        α = 0.5
        expected = Matrix(I(3)) + α * x * x'

        JuMinuit.sym_rank1_update!(S, α, x)
        # Element-wise compare (Symmetric reads both triangles)
        for i in 1:3, j in 1:3
            @test S[i, j] ≈ expected[i, j] atol = 1e-14
        end

        # Stacking two rank-1 updates equals a rank-2 update
        S2 = Symmetric(zeros(3, 3), :U)
        y = [-1.0, 0.5, 2.0]
        β = -1.0
        JuMinuit.sym_rank1_update!(S2, α, x)
        JuMinuit.sym_rank1_update!(S2, β, y)
        expected2 = α * x * x' + β * y * y'
        for i in 1:3, j in 1:3
            @test S2[i, j] ≈ expected2[i, j] atol = 1e-14
        end

        # Type stability
        S3 = Symmetric(zeros(4, 4), :U)
        z = rand(4)
        @test (@inferred JuMinuit.sym_rank1_update!(S3, 0.3, z)) === S3
    end

    # ─────────────────────────────────────────────────────────
    @testset "sym_mul!" begin
        S = Symmetric(Float64[2.0 1.0 0.0;
                              1.0 3.0 1.0;
                              0.0 1.0 2.0], :U)
        x = [1.0, 2.0, 3.0]
        y = zeros(3)

        # Default: y = S * x
        JuMinuit.sym_mul!(y, S, x)
        @test y ≈ S * x

        # y = α S x
        JuMinuit.sym_mul!(y, S, x, 2.0)
        @test y ≈ 2.0 * (S * x)

        # y = α S x + β y_init
        y_init = [1.0, -1.0, 2.0]
        y2 = copy(y_init)
        JuMinuit.sym_mul!(y2, S, x, 2.0, 3.0)
        @test y2 ≈ 2.0 * (S * x) + 3.0 * y_init

        # Type stability
        y3 = zeros(3)
        @test (@inferred JuMinuit.sym_mul!(y3, S, x)) === y3
    end

    # ─────────────────────────────────────────────────────────
    @testset "Zero-allocation hot kernels" begin
        # The two BLAS-backed primitives must be zero-alloc — they live
        # in MIGRAD's inner loop (§3.4 criterion 3 prerequisite).
        n = 10
        S = Symmetric(Matrix(I(n)) + rand(n, n) * 0.1, :U)
        sym_S = Symmetric(parent(S) + parent(S)', :U)  # ensure sym + posdef
        x = rand(n)
        y = zeros(n)

        # Warmup (compile)
        JuMinuit.sym_mul!(y, sym_S, x)
        JuMinuit.sym_rank1_update!(sym_S, 0.1, x)

        # Measure
        @test (@allocated JuMinuit.sym_mul!(y, sym_S, x)) == 0
        @test (@allocated JuMinuit.sym_mul!(y, sym_S, x, 2.0, 3.0)) == 0
        @test (@allocated JuMinuit.sym_rank1_update!(sym_S, 0.1, x)) == 0
    end

    # ─────────────────────────────────────────────────────────
    @testset "sym_invert!" begin
        # Known: M^{-1} for a diagonally-dominant 3x3
        M_orig = Float64[4.0 1.0 0.0;
                         1.0 3.0 1.0;
                         0.0 1.0 2.0]
        S = Symmetric(copy(M_orig), :U)
        JuMinuit.sym_invert!(S)
        # Verify: S now holds inv(M_orig); product is I.
        product = Symmetric(M_orig, :U) * S
        @test isapprox(product, I(3); atol = 1e-12)

        # Round trip: inverting twice returns the original (within precision)
        S2 = Symmetric(copy(M_orig), :U)
        JuMinuit.sym_invert!(S2)
        JuMinuit.sym_invert!(S2)
        for i in 1:3, j in 1:3
            @test S2[i, j] ≈ M_orig[i, j] atol = 1e-10
        end

        # Singular: 0 diagonal triggers
        S_sing = Symmetric(zeros(2, 2), :U)
        @test_throws LinearAlgebra.SingularException JuMinuit.sym_invert!(S_sing)

        # throw_on_fail=false returns S without throwing on singular
        S_sing2 = Symmetric(zeros(2, 2), :U)
        @test JuMinuit.sym_invert!(S_sing2; throw_on_fail = false) === S_sing2
    end

    # ─────────────────────────────────────────────────────────
    @testset "sym_eigvals" begin
        # 2×2 with known analytical eigenvalues
        S = Symmetric(Float64[4.0 1.0; 1.0 3.0], :U)
        evs = JuMinuit.sym_eigvals(S)
        @test length(evs) == 2
        @test issorted(evs)
        # λ = (7 ± √5) / 2
        @test evs[1] ≈ (7 - sqrt(5)) / 2 atol = 1e-12
        @test evs[2] ≈ (7 + sqrt(5)) / 2 atol = 1e-12

        # 3×3 identity — all ones
        S_I = Symmetric(Matrix{Float64}(I, 3, 3), :U)
        @test JuMinuit.sym_eigvals(S_I) ≈ ones(3) atol = 1e-14
    end

    # ─────────────────────────────────────────────────────────
    @testset "Convention sanity: :U-stored matrix reads symmetric" begin
        # Construct an asymmetric Matrix; wrap with Symmetric(:, :U).
        # Lower triangle of underlying Matrix is ignored — Symmetric
        # mirrors the upper into both positions on access.
        M = Float64[1.0 2.0 3.0;
                    99.0 4.0 5.0;     # 99 should be ignored
                    99.0 99.0 6.0]    # 99s should be ignored
        S = Symmetric(M, :U)
        @test S[1, 2] == 2.0
        @test S[2, 1] == 2.0  # symmetric access
        @test S[1, 3] == 3.0
        @test S[3, 1] == 3.0
        @test S[2, 3] == 5.0
        @test S[3, 2] == 5.0
    end
end
