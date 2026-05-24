# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test

@testset "JuMinuit.jl" verbose = true begin
    include("test_precision.jl")
    include("test_strategy.jl")
end
