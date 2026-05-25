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
    include("test_davidon.jl")
    include("test_edm.jl")
    include("test_posdef.jl")
    include("test_linesearch.jl")
    include("test_negative_g2.jl")
    include("test_seed.jl")
    include("test_migrad.jl")
    include("test_cpp_oracle.jl")
    include("test_aqua_jet.jl")
    include("test_transform.jl")
    include("test_parameters.jl")
    include("test_hesse.jl")
    include("test_covariance_squeeze.jl")
    include("test_minos.jl")
    include("test_contours.jl")
    include("test_migrad_bounded.jl")
    include("test_minuit.jl")
end
