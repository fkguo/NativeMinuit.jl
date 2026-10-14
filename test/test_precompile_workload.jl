# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "Precompile workload units execute without suppression" begin
    workloads = NativeMinuit._PRECOMPILE_WORKLOADS
    @test length(workloads) == 34
    @test length(unique(first.(workloads))) == 34
    for (name, workload) in workloads
        @testset "$name" begin
            # An exception is a test error, never a skipped workload.
            @test workload() !== nothing
        end
    end
end
