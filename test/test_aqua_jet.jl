# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Phase 0 §3.4 Criterion 4: Aqua + JET clean.
#
# - Aqua.test_all: project-quality checks (compat bounds, stale deps,
#   piracy, persistent tasks). Ambiguities check disabled by default —
#   known to flag stdlib false positives.
# - JET.report_call on the public `migrad(::Function, ::Vector{Float64},
#   ::Vector{Float64})` entry point: no errors detected.

using Aqua
using JET

@testset "Aqua + JET (§3.4 Criterion 4)" begin
    @testset "Aqua quality checks" begin
        Aqua.test_all(JuMinuit; ambiguities = false)
    end

    @testset "JET clean on public migrad" begin
        # JET on Julia 1.10 incorrectly flags `LinearAlgebra.BLAS.hemv!`
        # because the method signature on 1.10 differs from 1.12 (Julia's
        # stdlib BLAS bindings were refactored between minor versions).
        # The function itself dispatches correctly at runtime — this is
        # purely a JET-inference false positive on 1.10. Skip the check
        # on 1.10 to keep CI green while preserving the strict 1.12+ JET
        # gate (test_aqua_jet.jl is the §3.4 Criterion 4 contract).
        if VERSION >= v"1.12"
            f(x) = sum(abs2, x)
            report = JET.report_call(
                JuMinuit.migrad, (typeof(f), Vector{Float64}, Vector{Float64}),
            )
            @test isempty(JET.get_reports(report))
        else
            @test_skip "JET inference false positives on Julia < 1.12 (BLAS.hemv! signature drift)"
        end
    end
end
