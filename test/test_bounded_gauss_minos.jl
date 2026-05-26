# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Bounded Gauss-LL MINOS — Phase 1 完成判据 #5 explicitly asks for
# numerical agreement with C++ Minuit2 on a bounded fit. The
# unbounded Quad4F case (in test_minos_oracle.jl) already meets the
# 1e-8 bar; this file closes the bounded half.
#
# Setup (mirrors tools/cpp_trace_harness.cxx :: run_minos_case for
# `bounded_gauss_ll`):
#   - Data: 200 events drawn from N(μ=2, σ=1), seeded with
#     C++ mt19937_64(0xCAFEF00D). The data array is reconstructed
#     bit-identically from `bounded_gauss_ll_data.json` so the Julia
#     FCN is the same function the C++ side fit.
#   - FCN: `Σᵢ (log σ + ½(xᵢ−μ)²/σ²)`, with `up = 0.5` (NLL convention).
#   - Bound: `σ ∈ [0.1, ∞)` (lower-only — exercises the SqrtLow transform).
#   - Strategy(1) (iminuit default).

using JSON

@testset "C++ MINOS oracle — bounded Gauss-LL (Phase 1 完成判据 #5)" begin
    data_path  = joinpath(@__DIR__, "reference_data", "bounded_gauss_ll_data.json")
    minos_path = joinpath(@__DIR__, "reference_data", "bounded_gauss_ll_minos.json")
    isfile(data_path)  || error("missing oracle: $data_path")
    isfile(minos_path) || error("missing oracle: $minos_path")

    data = Float64.(JSON.parsefile(data_path)["data"])
    @test length(data) == 200

    # The exact Gaussian NLL FCN — identical to GaussNLL in the C++ harness.
    function gauss_nll(p::AbstractVector)
        μ, σ = p[1], p[2]
        σ <= 0 && return 1e30   # Soft barrier (matches C++ harness)
        s = 0.0
        @inbounds for x in data
            d = x - μ
            s += log(σ) + 0.5 * d * d / (σ * σ)
        end
        return s
    end

    # Drive the fit through the Minuit wrapper so we get
    # external-coordinate MINOS errors (the user-facing form, matching
    # C++ MnMinos and iminuit).
    m = Minuit(gauss_nll, [1.0, 2.0]; name = ["p0", "p1"],
                                       errors = [0.1, 0.1],
                                       limit_p1 = (0.1, nothing),
                                       up = 0.5)
    migrad(m; strategy = Strategy(1))
    @test m.is_valid

    ref = JSON.parsefile(minos_path)

    # ── Function value + external param values ─────────────────────
    @test m.fval ≈ ref["fval"] atol = 1e-4
    @test m.values[1] ≈ ref["params"][1] atol = 1e-3   # μ
    @test m.values[2] ≈ ref["params"][2] atol = 1e-3   # σ

    # ── MINOS asymmetric errors (external) vs C++ MnMinos ──────────
    # Gauss NLL is well-conditioned (Hessian ≈ diag(N/σ², 2N/σ²)),
    # so MINOS is close to linear and Julia↔C++ should agree at
    # ~1e-3. (Tighter than rosen2 because FCN is smoother; looser
    # than pure quadratic because of the σ ∈ [0.1, ∞) SqrtLow
    # transform's nonlinear remapping near the minimum.)
    minos(m)
    for (i, ref_entry) in enumerate(ref["minos"])
        @testset "par $(i-1) ($(i == 1 ? "μ" : "σ"))" begin
            @test haskey(m.minos_errors, i)
            err = m.minos_errors[i]
            @test JuMinuit.is_valid(err)
            @test err.upper_valid == ref_entry["upper_valid"]
            @test err.lower_valid == ref_entry["lower_valid"]
            @test err.upper ≈ ref_entry["upper"] atol = 1e-3
            @test err.lower ≈ ref_entry["lower"] atol = 1e-3
            @test err.min_par_value ≈ ref_entry["min_value"] atol = 1e-3
        end
    end
end
