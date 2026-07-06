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
        new_dc, status = NativeMinuit.davidon_update!(
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
        _, status = NativeMinuit.davidon_update!(V, dx, dg, 0.0, vg_work, vUpd_work)
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
        new_dc, status = (@test_logs (:warn, r"delgam = 0") NativeMinuit.davidon_update!(
            V, dx, dg_zero, 0.5, vg_w, vUpd_w))
        @test status === :unchanged_delgam_zero
        @test new_dc == 0.5
        @test V ≈ V0  # untouched

        # gvg ≤ 0 (with V having a negative eigenvalue — pathological but defensible)
        V_bad = Symmetric(Float64[-1.0 0.0; 0.0 -1.0], :U)
        dg = [1.0, 1.0]
        V2 = Symmetric(copy(parent(V_bad)), :U)
        new_dc2, status2 = (@test_logs (:warn, r"gvg ≤ 0") NativeMinuit.davidon_update!(
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

        # Measure through a function barrier: at @testset scope the non-const
        # bindings dispatch dynamically on Julia 1.11 (a spurious box the old
        # "BLAS-syr! temporary" comment mis-attributed — verified 0 through a
        # barrier on 1.11). The real in-function alloc is 0 on both 1.11/1.12.
        _a6(f, a, b, c, d, e, g) = @allocated f(a, b, c, d, e, g)
        # Warmup the barrier (mutates V), then reset to a fresh known matrix.
        _a6(NativeMinuit.davidon_update!, V, dx, dg, 0.0, vg_work, vUpd_work)
        copyto!(parent(V), Matrix{Float64}(I, n, n))
        @test _a6(NativeMinuit.davidon_update!, V, dx, dg, 0.0, vg_work, vUpd_work) == 0
    end

    # ─────────────────────────────────────────────────────────
    @testset "Allocating convenience wrapper" begin
        n = 3
        M = Matrix{Float64}(I, n, n) .+ 0.1  # broadcast scalar add
        err0 = MinimumError(Symmetric(M, :U), 0.5)
        dx = [0.1, 0.2, 0.3]
        dg = [0.5, 0.4, 0.3]
        err_new = NativeMinuit.davidon_update(err0, dx, dg)
        @test err_new isa MinimumError
        @test is_available(err_new)
        @test err_new.status == MnHesseValid
        # Original is untouched (deep-copy semantic)
        @test parent(err0.inv_hessian) == M
    end

    # ─────────────────────────────────────────────────────────
    # dcov against independent C++-equivalent oracle (parallel-review
    # D1 — previously missing; the lack of this test let the sum_sym
    # signed-vs-absolute bug slip through).
    # ─────────────────────────────────────────────────────────
    @testset "new_dcov matches C++ sum_of_elements formula (absolute sums)" begin
        n = 3
        V0 = Symmetric(Float64[
            2.0  0.1 -0.05;
            0.1  3.0  0.2;
           -0.05 0.2  1.5
        ], :U)
        dx = [0.3, -0.4, 0.5]   # mixed signs deliberately
        dg = [0.1,  0.2, 0.3]
        prev_dc = 0.7

        # ── Independent oracle: build the update matrix from C++ math ──
        delgam = dot(dx, dg)
        vg = parent(V0) * dg  # use parent to ensure plain matvec
        gvg = dot(dg, vg)
        # Pure rank-2 base
        update = (dx * dx') / delgam - (vg * vg') / gvg
        # Rank-1 additive if applicable
        if delgam > gvg
            u = dx ./ delgam .- vg ./ gvg
            update .+= gvg .* (u * u')
        end
        # sum_upd = Σ |update[i,j]| over upper+diagonal (matches mndasum)
        sum_upd_expected = sum(abs(update[i, j]) for j in 1:n for i in 1:j)
        # V_new = V0 + update; sum its |entries|
        V_new_expected = parent(V0) + update
        sum_V_expected = sum(abs(V_new_expected[i, j]) for j in 1:n for i in 1:j)
        expected_dcov = 0.5 * (prev_dc + sum_upd_expected / sum_V_expected)

        # ── Actual ──
        V = Symmetric(copy(parent(V0)), :U)
        vg_work = zeros(n)
        vUpd_work = Symmetric(zeros(n, n), :U)
        new_dc, status = NativeMinuit.davidon_update!(
            V, dx, dg, prev_dc, vg_work, vUpd_work)
        @test status === :updated
        @test new_dc ≈ expected_dcov atol = 1e-12
    end
end
