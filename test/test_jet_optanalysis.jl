# SPDX-License-Identifier: LGPL-2.1-or-later
#
# JET optimization-analysis guard (hot-path devirtualization). Included from
# test_aqua_jet.jl ONLY when JET successfully loaded (`HAS_JET`), because the
# `@report_opt` macros below cannot be parsed when JET is absent. JET is a
# dev-only, Julia-version-fragile tool kept out of the default `[targets] test`
# deps — see the `HAS_JET` note in test_aqua_jet.jl.
#
# The "C++-comparable performance" claim rests on the FCN call site being
# devirtualized: a user closure must specialize into `CostFunction{F}` so every
# FCN call in MIGRAD's inner loop is a static call, not a runtime dispatch.
# `@inferred` only checks the top-level return type; JET's optimization analysis
# walks the whole call graph and flags ANY runtime dispatch — catching a SILENT
# perf regression (numerically identical, just slow) the rest of the suite
# cannot see (ROADMAP risk #4). `target_modules=(NativeMinuit,)` scopes the check to
# our own code (ignoring LinearAlgebra/Base internals), keeping it
# false-positive-free across Julia versions. On failure, run e.g.
#   JET.@report_opt target_modules=(NativeMinuit,) migrad(g, gx0, gerrs)
# to see the offending dispatch site(s).

@testset "JET opt-analysis — hot-path devirtualization (regression guard)" begin
    g(x) = (x[1] - 1.0)^2 + 100.0 * (x[2] - x[1]^2)^2   # a raw user closure
    gx0 = [0.0, 0.0]
    gerrs = [0.1, 0.1]
    @test isempty(JET.get_reports(
        @report_opt target_modules = (NativeMinuit,) migrad(g, gx0, gerrs)))
    @test isempty(JET.get_reports(
        @report_opt target_modules = (NativeMinuit,) migrad(CostFunction(g), gx0, gerrs)))
end
