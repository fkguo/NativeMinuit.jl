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

    # Tolerances. The bounded Gauss NLL has a well-conditioned Hessian
    # (≈ diag(N/σ², 2N/σ²)) and only one parameter is on a SqrtLow
    # transform — so Julia↔C++ agreement is much tighter than the
    # rosenbrock case. Local measurements (Apple M3, OpenBLAS 0.3.29):
    #   fval        |diff| ≈ 3e-7
    #   μ           |diff| ≈ 9e-6
    #   σ           |diff| ≈ 5e-5
    #   μ MINOS U/L |diff| ≈ 1.1e-4
    #   σ MINOS U/L |diff| ≈ 5e-5
    # Tolerances picked to give ~3-5× headroom over the worst local
    # drift plus ~2× margin for CI machines (Gauss is well-conditioned
    # so the 3× rosen10d-cov CI factor shouldn't apply here).
    atol_fval  = 1e-6
    atol_val   = 1e-4
    atol_minos = 5e-4

    # ── Function value + external param values ─────────────────────
    @test m.fval ≈ ref["fval"] atol = atol_fval
    @test m.values[1] ≈ ref["params"][1] atol = atol_val   # μ
    @test m.values[2] ≈ ref["params"][2] atol = atol_val   # σ

    # ── MINOS asymmetric errors (external) vs C++ MnMinos ──────────
    minos(m)
    for (i, ref_entry) in enumerate(ref["minos"])
        @testset "par $(i-1) ($(i == 1 ? "μ" : "σ"))" begin
            @test haskey(m.minos_errors, i)
            err = m.minos_errors[i]
            @test NativeMinuit.is_valid(err)
            @test err.upper_valid == ref_entry["upper_valid"]
            @test err.lower_valid == ref_entry["lower_valid"]
            @test err.upper ≈ ref_entry["upper"] atol = atol_minos
            @test err.lower ≈ ref_entry["lower"] atol = atol_minos
            @test err.min_par_value ≈ ref_entry["min_value"] atol = atol_minos
        end
    end
end
