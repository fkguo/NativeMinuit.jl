# SPDX-License-Identifier: LGPL-2.1-or-later
# Run with julia --project=benchmark benchmark/compare_cpp.jl [--julia-record FILE] [--cpp-record FILE]
using Pkg
Pkg.activate(@__DIR__; io=devnull)
include(joinpath(@__DIR__, "comparison_evidence.jl"))
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(comparison_main(ARGS))
end
