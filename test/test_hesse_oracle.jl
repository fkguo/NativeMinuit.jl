# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Standalone HESSE numerical agreement with C++ Minuit2.
#
# Tests the path "Strategy(0) MIGRAD → hesse(cf, state)" where the
# user explicitly calls `hesse` after a quick (DFP-only) MIGRAD.
# The Strategy(1+) inner-Hesse path is already covered by the
# main C++ oracle tests (test_cpp_oracle.jl); this file specifically
# verifies the standalone `hesse(cf, state)` entry point that the
# user may call directly.
#
# Reference JSON: tools/cpp_trace_harness.cxx :: run_hesse_case
# dumps the EXTERNAL covariance matrix (2·up·V) after MnHesse runs
# on a Strategy(0) MIGRAD result.

using JSON

const HESSE_CASES = Dict{String,Any}(
    "quad_4d" => (
        fcn   = x -> sum(abs2, x),
        x0    = [1.0, 1.0, 1.0, 1.0],
        errs0 = [0.1, 0.1, 0.1, 0.1],
        # Pure quadratic — HESSE recovers the exact V = (2·H)^{-1} = 0.5·I.
        # Floating-point precision agreement.
        atol_cov = 1e-12,
    ),
    "rosenbrock_2d" => (
        fcn   = x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
        x0    = [-1.2, 1.0],
        errs0 = [0.1, 0.1],
        # Strategy(0) MIGRAD stops earlier than the Strategy(1+) path,
        # so the standalone-HESSE evaluation point itself differs from
        # the C++ reference by ~3e-5 in params; that drift gets
        # amplified by 100× in the cov (which scales as inv(H_min)
        # and rosenbrock's H is ill-conditioned). Measured cov |diff|
        # on this case ≈ 0.14 (~3% rel of cov scale ~4).
        atol_cov = 2e-1,
    ),
    "quad_2d_shifted" => (
        fcn   = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
        x0    = [0.0, 0.0],
        errs0 = [0.1, 0.1],
        atol_cov = 1e-8,
    ),
)

@testset "C++ HESSE oracle parity (standalone hesse(cf, state))" begin
    for (name, c) in HESSE_CASES
        path = joinpath(@__DIR__, "reference_data", "$(name)_hesse.json")
        @testset "$name standalone HESSE" begin
            isfile(path) || (@warn "Oracle missing: $path"; continue)
            ref = JSON.parsefile(path)

            cf = CostFunction(c.fcn)
            # Strategy(0) MIGRAD: V is DFP approximation, not numerical.
            fmin = migrad(cf, c.x0, c.errs0; strategy = Strategy(0))
            @test NativeMinuit.is_valid(fmin)

            # Standalone HESSE refinement on the converged state.
            refined = NativeMinuit.hesse(cf, fmin.state, Strategy(1))

            # Verify converged params unchanged by HESSE (HESSE doesn't
            # move x; any diff vs C++ here is the Strategy(0) MIGRAD
            # EDM-stopping drift, same as in test_cpp_oracle.jl).
            for (i, p_ref) in enumerate(ref["params"])
                @test refined.parameters.x[i] ≈ Float64(p_ref) atol = 1e-4
            end

            # Compare the EXTERNAL covariance (2·up·V) element-by-element
            # to C++ MnHesse output (also external).
            n = length(ref["params"])
            ref_cov_flat = Float64.(ref["covariance_upper"])
            ref_cov = zeros(n, n)
            k = 1
            for i in 1:n, j in i:n
                ref_cov[i, j] = ref_cov[j, i] = ref_cov_flat[k]
                k += 1
            end
            jl_cov = 2.0 * cf.up * collect(refined.error.inv_hessian)
            for i in 1:n, j in 1:n
                @test isapprox(jl_cov[i, j], ref_cov[i, j];
                                atol = c.atol_cov)
            end
        end
    end
end
