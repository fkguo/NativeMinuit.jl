# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "MachinePrecision" begin
    p = MachinePrecision()
    @test p.eps == eps(Float64)
    @test p.eps2 ≈ 2 * sqrt(eps(Float64))
    @test p.eps2 > p.eps  # √eps > eps for eps < 1

    # User override (e.g. stochastic FCN with reduced precision)
    p2 = MachinePrecision(1e-12)
    @test p2.eps == 1e-12
    @test p2.eps2 ≈ 2 * sqrt(1e-12)

    # isbits sanity — zero-alloc when used in hot loops
    @test isbits(p)
    @test isbits(p2)
    @test sizeof(MachinePrecision) == 2 * sizeof(Float64)  # eps + eps2

    # Type stability of constructor
    @test (@inferred MachinePrecision()) isa MachinePrecision
    @test (@inferred MachinePrecision(1e-15)) isa MachinePrecision
end
