# SPDX-License-Identifier: LGPL-2.1-or-later
# Real X3872 dip-fit benchmark: numerical vs AD vs threaded numerical.
# Reproduces the fit from BenchmarkExamples/X3872_dip/Xdip_published.ipynb
# (resonance-amplitude dip-structure fit with 3 free parameters).
#
# Source of the published fit:
#   V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
#   "How does the X(3872) show up in e+e- collisions: Dip versus peak",
#   Phys. Rev. D 109 (2024) 11, L111501, arXiv:2404.12003.
#   https://inspirehep.net/literature/2778938

using JuMinuit
using DataFrames, CSV, QuadGK
using LinearAlgebra, Random, Statistics, BenchmarkTools

BLAS.set_num_threads(1)
println("Threads: nthreads=", Threads.nthreads(), " maxthreadid=", Threads.maxthreadid())

const X3872_DIR = @__DIR__   # script-relative, portable across machines
cd(X3872_DIR)
include(joinpath(X3872_DIR, "hadronmasses.jl"))

const mρ = 0.750 * unit_choice
const mω = 0.782 * unit_choice

# Gauss–Legendre quadrature (n=128 for adequate accuracy on resolution
# convolution; original notebook uses 64-128 depending on cell).
function quadgauss(f, x, w)
    res = zero(f(x[1]))
    @inbounds for i in eachindex(x)
        res += f(x[i]) * w[i]
    end
    return res
end

const QX, QW = gauss(128, 10, -10)

# Load data (3-column CSV: w [MeV], y [pb], err [pb])
const data_df = DataFrame(CSV.File("data.csv", header=["w", "y", "err"]))
println("Loaded ", nrow(data_df), " data points")

# Complex sqrt on the cut along the positive real axis
function xsqrt(x)
    imag(x) >= 0 ? sqrt(x + 0im) : -sqrt(x - 0im)
end

λ(x, y, z) = x^2 + y^2 + z^2 - 2x*y - 2y*z - 2z*x
qsq(E, m1, m2) = λ(E^2, m1^2, m2^2) / (4E^2)

# Single-channel scattering length matrix with constrained DD̄* effective.
# Original notebook fixes a22eff = (-6.39 + 11.74im) fm; we follow.
function t11_lo_constrained(e, a11, a22, a22eff; mrho)
    Σ2 = md0 + mdstar0
    μ2 = md0 * mdstar0 / Σ2
    k1 = xsqrt(qsq(e, mjψ, mrho))
    k2 = xsqrt(2μ2 * (e - md0 - mdstar0))
    return -8π * Σ2 * (1/a22 - 1im*k2) / (1/a11 - 1im*k1) / (1/a22eff - 1im*k2)
end

# resolution kernel (Gaussian smearing, σ=1.7 MeV per notebook)
resolution(x, σ) = exp(-0.5*(x/σ)^2) / (sqrt(2π)*σ)

# Model 1 (fixed a22eff): par = [p0, r, a22] in physical units
function model1(x, par; mrho)
    p0, r, a22 = par
    a22 = a22 / ħc                                  # fm → MeV⁻¹
    a22eff = (-6.39 + 11.74im) / ħc
    inv_a22eff = 1 / a22eff
    k1 = real(xsqrt(qsq(x, mjψ, mrho)))
    a11 = 1 / (k1 / imag(inv_a22eff) * (real(inv_a22eff) - 1/a22))
    amp2 = p0^2 * abs2(1 + r * t11_lo_constrained(x, a11, a22, a22eff; mrho))
    return amp2
end

# Resolution convolution via fixed Gauss-Legendre quadrature
function res_model1(x, σ, par; mrho)
    quadgauss(y -> model1(x - y, par; mrho) * resolution(y, σ), QX, QW)
end
res_model1(x, par; mrho) = res_model1(x, 1.7, par; mrho)

# Use the Model 4 rho-mass complex pole from the notebook
const mrho_model4 = 775.0 - 75.0im

# Chi-square (manual — sidesteps IMinuit-compat Data) so we don't pull
# unnecessary deps.
function chi2_x3872(par)
    s = 0.0
    @inbounds for i in 1:nrow(data_df)
        x = data_df.w[i]
        y = data_df.y[i]
        ε = data_df.err[i]
        pred = res_model1(x, par; mrho = mrho_model4)
        s += ((pred - y) / ε)^2
    end
    return s
end

# Starting point per notebook: (3., 0.0001, -4)
const PAR0 = [3.0, 0.0001, -4.0]
const ERRS = [0.1, 0.01, 1.0]

println("\nχ²(par0) = ", chi2_x3872(PAR0))

# Per-call FCN cost
println("\n=== Per-call FCN cost (real X3872) ===")
b0 = @benchmark chi2_x3872($PAR0) samples=50 evals=1
println("  median = ", round(median(b0).time/1000; digits=1), " μs/call, allocs=", b0.allocs)

# === 3-path comparison ===
println("\n=== migrad wall time, 5-round median ===")

function run_5(label, factory)
    factory()  # warmup
    times = Float64[]
    for _ in 1:5
        t0 = time_ns()
        factory()
        push!(times, (time_ns() - t0) / 1e9)
        sleep(0.3)
    end
    sort!(times)
    println("  $label: median=$(round(times[3]*1000; digits=1)) ms, min=$(round(times[1]*1000; digits=1)) max=$(round(times[end]*1000; digits=1))")
    return times[3]
end

# (a) numerical
fn_num() = (cf = CostFunction(chi2_x3872, 1.0); migrad(cf, PAR0, ERRS); cf)
t_num = run_5("numerical 1T", fn_num)

# (b) AD
using ForwardDiff
fn_ad() = (cf = CostFunctionAD(chi2_x3872, 1.0); migrad(cf, PAR0, ERRS); cf)
t_ad = run_5("AD (ForwardDiff)", fn_ad)

# (c) threaded numerical (only meaningful with -t N)
if Threads.nthreads() > 1
    fn_th() = (cf = CostFunction(chi2_x3872, 1.0); migrad(cf, PAR0, ERRS; threaded_gradient=true); cf)
    t_th = run_5("threaded numerical", fn_th)
else
    println("  threaded numerical: SKIPPED (need julia -t N, N>1)")
    t_th = NaN
end

println("\n=== Summary (5-round median, X3872 dip fit, n=3) ===")
println("  numerical 1T:        ", round(t_num*1000; digits=1), " ms")
println("  AD (ForwardDiff):    ", round(t_ad*1000; digits=1), " ms (", round(t_num/t_ad; digits=2), "× vs num)")
if !isnan(t_th)
    println("  threaded numerical:  ", round(t_th*1000; digits=1), " ms (", round(t_num/t_th; digits=2), "× vs num)")
end

# ─────────────────────────────────────────────────────────────────────
# CORRECTNESS CROSS-CHECK — all paths must converge to the same minimum.
# Critical for trusting the wall-time comparison. If a path silently
# converges to a DIFFERENT minimum (e.g., from a thread-unsafe FCN
# giving corrupted gradients → algorithm slides to wrong basin), the
# wall-time numbers are meaningless. See IAM bench for a real example
# where IAM internal `St4_00!` mutates module-level `const c_*` buffers
# → threading races → DIFFERENT minimum.
# ─────────────────────────────────────────────────────────────────────
println("\n=== Correctness cross-check ===")
cf_n = CostFunction(chi2_x3872, 1.0)
fm_n = migrad(cf_n, PAR0, ERRS)
cf_a = CostFunctionAD(chi2_x3872, 1.0)
fm_a = migrad(cf_a, PAR0, ERRS)
println("  numerical 1T: x=$(round.(fm_n.state.parameters.x; digits=5)) fval=$(round(fm_n.state.parameters.fval; sigdigits=5))")
println("  AD:           x=$(round.(fm_a.state.parameters.x; digits=5)) fval=$(round(fm_a.state.parameters.fval; sigdigits=5))")
if Threads.nthreads() > 1
    cf_t = CostFunction(chi2_x3872, 1.0)
    fm_t = migrad(cf_t, PAR0, ERRS; threaded_gradient=true)
    println("  threaded 8T:  x=$(round.(fm_t.state.parameters.x; digits=5)) fval=$(round(fm_t.state.parameters.fval; sigdigits=5))")
    Δf_na = abs(fm_n.state.parameters.fval - fm_a.state.parameters.fval)
    Δf_nt = abs(fm_n.state.parameters.fval - fm_t.state.parameters.fval)
    ok_na = Δf_na < 0.01
    ok_nt = Δf_nt < 0.01
    println("\n  cross-check verdict:")
    println("    numerical vs AD:       Δfval=$(round(Δf_na; sigdigits=3)) → $(ok_na ? "✓ PASS" : "⚠ DIFFERENT MINIMA")")
    println("    numerical vs threaded: Δfval=$(round(Δf_nt; sigdigits=3)) → $(ok_nt ? "✓ PASS — X3872 FCN is thread-safe" : "⚠ DIFFERENT MINIMA (FCN not thread-safe)")")
end
