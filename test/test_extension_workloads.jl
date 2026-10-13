# SPDX-License-Identifier: LGPL-2.1-or-later
using Test, NativeMinuit, ForwardDiff, DataFrames, Clustering, Optim

@testset "Extension precompile operations execute without suppression" begin
    fd = Base.get_extension(NativeMinuit, :NativeMinuitForwardDiffExt)
    cf, minimum, fit = fd._precompile_workload()
    @test cf isa CostFunctionWithGradient
    @test is_valid(minimum)
    @test fit.valid
    @test has_closed_interval(fit.minos_errors[1])
    @test ncalls(cf) > 0

    de = Base.get_extension(NativeMinuit, :NativeMinuitDataFramesExt)
    frame, data = de._precompile_workload()
    @test data isa Data
    @test data.x == frame.x
    @test data.y == frame.y
    @test data.err == frame.e

    ce = Base.get_extension(NativeMinuit, :NativeMinuitClusteringExt)
    @test ce._precompile_workload() == [1, 1, 1, 2, 2, 2]

    oe = Base.get_extension(NativeMinuit, :NativeMinuitOptimExt)
    optimized = oe._precompile_workload()
    @test optimized isa Minuit
    @test optimized.valid
    @test optimized.nfcn > 0
end
