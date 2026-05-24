# SPDX-License-Identifier: LGPL-2.1-or-later

using LinearAlgebra

@testset "davidon.jl — DFP update" begin

    # ─────────────────────────────────────────────────────────
    @testset "Additive rank-1 semantic (NOT branched)" begin
        # This is the Opus-review blocking #1 check. Construct dx, dg, V
        # such that delgam > gvg, so the rank-1 correction fires.
        # Verify the result is rank-2 base + rank-1 correction, NOT
        # one branch alone.
        n = 3
        V0 = Symmetric(Float64[
            1.0  0.1  0.0;
            0.1  2.0  0.2;
            0.0  0.2  1.5
        ], :U)
        dx = [1.0, 2.0, 1.5]
        dg = [0.3, 0.4, 0.5]
        prev_dc = 0.0

        # Compute scalars
        delgam = dot(dx, dg)
        vg = V0 * dg
        gvg = dot(dg, vg)
        @test delgam > gvg  # confirm we hit the rank-1 additive branch

        # Run the Julia DFP update
        V = Symmetric(copy(parent(V0)), :U)
        vg_work = zeros(n)
        vUpd_work = Symmetric(zeros(n, n), :U)
        new_dc, status = JuMinuit.davidon_update!(
            V, dx, dg, prev_dc, vg_work, vUpd_work)
        @test status === :updated

        # Independently compute the expected V_new with explicit
        # additive formula.
        expected_update =
            (dx * dx') / delgam - (vg * vg') / gvg +
            gvg * ((dx / delgam - vg / gvg) * (dx / delgam - vg / gvg)')
        V_expected = Symmetric(parent(V0) + expected_update, :U)

        for i in 1:n, j in 1:n
            @test V[i, j] ≈ V_expected[i, j] atol = 1e-12
        end

        # SANITY: a "branched" implementation that ONLY did the rank-1
        # would give a different answer. Verify our additive result
        # differs from such an alternative.
        wrong_update_rank1_only =
            gvg * ((dx / delgam - vg / gvg) * (dx / delgam - vg / gvg)')
        V_wrong = Symmetric(parent(V0) + wrong_update_rank1_only, :U)
        @test !isapprox(V_expected, V_wrong; atol = 1e-6)  # they're really different
    end

    # ─────────────────────────────────────────────────────────
    @testset "Rank-2-only when delgam ≤ gvg" begin
        # Construct a case where δ < γ so the rank-1 branch does NOT
        # fire. Result should be the pure rank-2 DFP base.
        n = 3
        V0 = Symmetric(Float64[
            5.0  0.0  0.0;
            0.0  5.0  0.0;
            0.0  0.0  5.0
        ], :U)
        dx = [0.1, 0.05, 0.02]
        dg = [1.0, 1.0, 1.0]   # makes gvg ~ 5*(1+1+1) = 15 ≫ δ
        delgam = dot(dx, dg)
        vg = V0 * dg
        gvg = dot(dg, vg)
        @test delgam < gvg  # confirm rank-1 does NOT fire

        V = Symmetric(copy(parent(V0)), :U)
        vg_work = zeros(n)
        vUpd_work = Symmetric(zeros(n, n), :U)
        _, status = JuMinuit.davidon_update!(V, dx, dg, 0.0, vg_work, vUpd_work)
        @test status === :updated

        expected = (dx * dx') / delgam - (vg * vg') / gvg
        V_expected = Symmetric(parent(V0) + expected, :U)
        for i in 1:n, j in 1:n
            @test V[i, j] ≈ V_expected[i, j] atol = 1e-12
        end
    end

    # ─────────────────────────────────────────────────────────
    @testset "Degenerate cases — unchanged matrix" begin
        n = 2
        V0 = Symmetric(Matrix{Float64}(I, n, n), :U)
        dx = [1.0, 0.0]

        # delgam = 0 (dg ⊥ dx)
        dg_zero = [0.0, 1.0]
        V = Symmetric(copy(parent(V0)), :U)
        vg_w = zeros(n)
        vUpd_w = Symmetric(zeros(n, n), :U)
        new_dc, status = (@test_logs (:warn, r"delgam = 0") JuMinuit.davidon_update!(
            V, dx, dg_zero, 0.5, vg_w, vUpd_w))
        @test status === :unchanged_delgam_zero
        @test new_dc == 0.5
        @test V ≈ V0  # untouched

        # gvg ≤ 0 (with V having a negative eigenvalue — pathological but defensible)
        V_bad = Symmetric(Float64[-1.0 0.0; 0.0 -1.0], :U)
        dg = [1.0, 1.0]
        V2 = Symmetric(copy(parent(V_bad)), :U)
        new_dc2, status2 = (@test_logs (:warn, r"gvg ≤ 0") JuMinuit.davidon_update!(
            V2, dx, dg, 0.3, vg_w, vUpd_w))
        @test status2 === :unchanged_gvg_nonpositive
        @test new_dc2 == 0.3
    end

    # ─────────────────────────────────────────────────────────
    @testset "Zero-allocation hot path (after warmup)" begin
        n = 5
        V = Symmetric(Matrix{Float64}(I, n, n) + rand(n, n) * 0.1, :U)
        # Make V symmetric explicitly (rand was asymmetric)
        Mp = parent(V)
        for i in 1:n, j in i+1:n
            Mp[i, j] = Mp[j, i]
        end

        dx = rand(n)
        dg = rand(n) .+ 0.1
        vg_work = zeros(n)
        vUpd_work = Symmetric(zeros(n, n), :U)

        # Warmup
        JuMinuit.davidon_update!(V, dx, dg, 0.0, vg_work, vUpd_work)

        # Reset V to a fresh known matrix (the warmup mutated it)
        copyto!(parent(V), Matrix{Float64}(I, n, n))

        # Measure (hopefully no @warn fires, since random dx · random
        # dg should be nonzero generically)
        @test (@allocated JuMinuit.davidon_update!(V, dx, dg, 0.0, vg_work, vUpd_work)) == 0
    end

    # ─────────────────────────────────────────────────────────
    @testset "Allocating convenience wrapper" begin
        n = 3
        M = Matrix{Float64}(I, n, n) .+ 0.1  # broadcast scalar add
        err0 = MinimumError(Symmetric(M, :U), 0.5)
        dx = [0.1, 0.2, 0.3]
        dg = [0.5, 0.4, 0.3]
        err_new = JuMinuit.davidon_update(err0, dx, dg)
        @test err_new isa MinimumError
        @test is_available(err_new)
        @test err_new.status == MnHesseValid
        # Original is untouched (deep-copy semantic)
        @test parent(err0.inv_hessian) == M
    end
end
