# SPDX-License-Identifier: LGPL-2.1-or-later
# Reads supplied measurements without silently combining incomparable workloads.
using Pkg
Pkg.activate(@__DIR__; io=devnull)
include(joinpath(@__DIR__, "comparison_evidence.jl"))
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(comparison_main(ARGS; languages=["julia","cpp","python"]))
end
