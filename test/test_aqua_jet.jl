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
        f(x) = sum(abs2, x)
        report = JET.report_call(
            JuMinuit.migrad, (typeof(f), Vector{Float64}, Vector{Float64}),
        )
        @test isempty(JET.get_reports(report))
    end
end
