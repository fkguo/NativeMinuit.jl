# SPDX-License-Identifier: LGPL-2.1-or-later
# Included only after JET loads. The independent pinned diagnostic environment
# executes this file in CI; optional execution in Pkg.test is additional evidence.

@testset "JET narrow optimization diagnostics" begin
    g(x) = (x[1] - 1.0)^2 + 100.0 * (x[2] - x[1]^2)^2
    gx0 = [0.0, 0.0]
    gerrs = [0.1, 0.1]
    @test isempty(JET.get_reports(
        JET.@report_opt target_modules = (NativeMinuit,) migrad(g, gx0, gerrs)))
    @test isempty(JET.get_reports(
        JET.@report_opt target_modules = (NativeMinuit,) migrad(CostFunction(g), gx0, gerrs)))

    # Negative control: an Any-valued callable must produce a dispatch report.
    # This also detects an empty/stub analyzer masquerading as a passing guard.
    dynamic_call(xs::Vector{Any}) = xs[1](xs[2])
    reports = JET.get_reports(JET.@report_opt dynamic_call(Any[sin, 1.0]))
    @test any(r -> r isa JET.RuntimeDispatchReport, reports)
end
