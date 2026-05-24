# SPDX-License-Identifier: LGPL-2.1-or-later

using JuMinuit
using Test

@testset "JuMinuit.jl" verbose = true begin
    include("test_precision.jl")
    include("test_strategy.jl")
    include("test_state.jl")
    include("test_fcn.jl")
    include("test_linalg.jl")
    include("test_gradient.jl")
end
