# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "MachinePrecision" begin
    p = MachinePrecision()
    # C++ MnMachinePrecision.cxx:26 — fEpsMac = 4·numeric_limits::epsilon().
    @test p.eps == 4 * eps(Float64)
    @test p.eps2 ≈ 2 * sqrt(4 * eps(Float64))
    @test p.eps2 > p.eps  # √eps > eps for eps < 1

    # User override (e.g. stochastic FCN with reduced precision) — the ×4
    # applies ONLY to the bare default; an explicit value is taken verbatim.
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

@testset "MnMachinePrecision C++ parity (audit §14)" begin
    # C++ Minuit2 MnMachinePrecision.cxx:26-27:
    #   fEpsMac = 4. * std::numeric_limits<double>::epsilon();  // ≈ 8.88e-16
    #   fEpsMa2 = 2. * std::sqrt(fEpsMac);                      // ≈ 5.96e-8
    # The default MachinePrecision() must carry the ×4. It was previously a
    # bare eps(Float64), making eps2 2× too small vs C++/iminuit and shifting
    # every eps2-gated step size engine-wide (audit §14, MAJOR).
    p = MachinePrecision()

    # eps == C++ fEpsMac (exact: 4 is a power of two, so no rounding).
    @test p.eps == 4 * eps(Float64)
    @test p.eps ≈ 8.881784197001252e-16

    # eps2 == C++ fEpsMa2 = 2·√(4·eps) = 4·√eps ≈ 5.96e-8.
    @test p.eps2 ≈ 2 * sqrt(4 * eps(Float64))
    @test p.eps2 ≈ 4 * sqrt(eps(Float64))
    @test p.eps2 ≈ 5.960464477539063e-8

    # The ×4 on eps propagates to EXACTLY ×2 on eps2 relative to the old
    # bare-eps default — the intended fidelity correction (~2× per audit §14).
    @test p.eps2 ≈ 2 * MachinePrecision(eps(Float64)).eps2

    # The default equals the explicit 4·eps constructor; the user-supplied
    # path is untouched and bypasses the ×4 (explicit value taken verbatim).
    @test p.eps == MachinePrecision(4 * eps(Float64)).eps
    @test p.eps2 == MachinePrecision(4 * eps(Float64)).eps2
    @test MachinePrecision(eps(Float64)).eps == eps(Float64)
end
