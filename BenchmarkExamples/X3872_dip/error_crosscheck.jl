# SPDX-License-Identifier: LGPL-2.1-or-later
#
# X(3872) dip fit — ERROR-ANALYSIS CROSS-CHECK (demonstration, NOT a unit test).
#
#   Compares three independent error estimates on the real published fit:
#     • HESSE              — analytic, symmetric  ±σ
#     • MINOS              — analytic, asymmetric +σ_up / −σ_lo (profile likelihood)
#     • parametric bootstrap — sampling-based ±1σ percentile interval
#
# Run (needs the bench env with CSV/QuadGK/DataFrames):
#     julia --project=scripts BenchmarkExamples/X3872_dip/error_crosscheck.jl
#
# WHY PARAMETRIC bootstrap (and why NO jackknife / nonparametric here):
#   the published dataset is only 4 points for 3 free parameters (1 DOF).
#   Nonparametric bootstrap (resample points with replacement) and delete-1
#   jackknife are NOT statistically meaningful at this size — a delete-1 fit
#   has 3 points / 3 params (0 DOF, exactly determined). The PARAMETRIC
#   bootstrap regenerates y_i* = model(x_i; p̂) + σ_i·z_i* from the fitted model
#   and the quoted per-point σ; it is valid at any N and shares MINOS's
#   likelihood assumptions, so the two should agree. (The bootstrap-vs-MINOS
#   agreement is exercised as a pass/fail regression test on a larger fixture
#   in test/test_resampling_errors.jl.)
#
# Published fit: V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
#   "How does the X(3872) show up in e+e- collisions: Dip versus peak",
#   Phys. Rev. D 109 (2024) L111501, arXiv:2404.12003.

using JuMinuit
using DataFrames, CSV, QuadGK
using LinearAlgebra, Random, Statistics

const X3872_DIR = @__DIR__
include(joinpath(X3872_DIR, "hadronmasses.jl"))

# ── model (mirrors BenchmarkExamples/X3872_dip/bench.jl) ─────────────────────
function quadgauss(f, x, w)
    res = zero(f(x[1]))
    @inbounds for i in eachindex(x)
        res += f(x[i]) * w[i]
    end
    return res
end
const QX, QW = gauss(128, 10, -10)

xsqrt(x) = imag(x) >= 0 ? sqrt(x + 0im) : -sqrt(x - 0im)
λ(x, y, z) = x^2 + y^2 + z^2 - 2x*y - 2y*z - 2z*x
qsq(E, m1, m2) = λ(E^2, m1^2, m2^2) / (4E^2)

function t11_lo_constrained(e, a11, a22, a22eff; mrho)
    Σ2 = md0 + mdstar0
    μ2 = md0 * mdstar0 / Σ2
    k1 = xsqrt(qsq(e, mjψ, mrho))
    k2 = xsqrt(2μ2 * (e - md0 - mdstar0))
    return -8π * Σ2 * (1/a22 - 1im*k2) / (1/a11 - 1im*k1) / (1/a22eff - 1im*k2)
end

resolution(x, σ) = exp(-0.5 * (x / σ)^2) / (sqrt(2π) * σ)

function model1(x, par; mrho)
    p0, r, a22 = par
    a22 = a22 / ħc
    a22eff = (-6.39 + 11.74im) / ħc
    inv_a22eff = 1 / a22eff
    k1 = real(xsqrt(qsq(x, mjψ, mrho)))
    a11 = 1 / (k1 / imag(inv_a22eff) * (real(inv_a22eff) - 1 / a22))
    return p0^2 * abs2(1 + r * t11_lo_constrained(x, a11, a22, a22eff; mrho))
end

res_model1(x, σ, par; mrho) =
    quadgauss(y -> model1(x - y, par; mrho) * resolution(y, σ), QX, QW)
res_model1(x, par; mrho) = res_model1(x, 1.7, par; mrho)

const mrho_model4 = 775.0 - 75.0im

# ── data + fit ───────────────────────────────────────────────────────────────
data_df = DataFrame(CSV.File(joinpath(X3872_DIR, "data.csv"), header = ["w", "y", "err"]))
xmodel(x, par) = res_model1(x, par; mrho = mrho_model4)
d = Data(Float64.(data_df.w), Float64.(data_df.y), Float64.(data_df.err))
PAR0 = [3.0, 0.0001, -4.0]
pnames = ["p0", "r", "a22"]

m = model_fit(xmodel, d, PAR0; name = pnames)
migrad!(m)
hesse(m)
minos!(m)

ndata = nrow(data_df)
println("\n", "="^74)
println("X(3872) dip fit — error-analysis cross-check")
println("="^74)
println("data points = $ndata,  free params = 3,  DOF = $(ndata - 3)")
println("fit:  p0 = ", round(m.values[1]; sigdigits = 5),
        "   r = ", round(m.values[2]; sigdigits = 5),
        "   a22 = ", round(m.values[3]; sigdigits = 5),
        "   (χ²_min = ", round(m.fval; sigdigits = 4), ")")

# ── parametric bootstrap (valid at small N) ──────────────────────────────────
bp = bootstrap(xmodel, d, m; nresample = 2000, seed = 2024,
               kind = :parametric, ci_level = 0.68)
println("parametric bootstrap: $(bp.n_valid)/2000 resamples converged\n")

# ── side-by-side comparison ──────────────────────────────────────────────────
fmt(x) = string(round(x; sigdigits = 3))
println(rpad("param", 7), rpad("value", 13), rpad("HESSE ±", 13),
        rpad("MINOS +up / −lo", 24), "param-bootstrap 1σ +up / −lo")
println("-"^84)
for i in 1:3
    e = m.merrors[pnames[i]]
    bu = bp.ci_upper[i] - bp.estimate[i]
    bl = bp.estimate[i] - bp.ci_lower[i]
    println(rpad(pnames[i], 7),
            rpad(fmt(m.values[i]), 13),
            rpad("±" * fmt(m.errors[i]), 13),
            rpad("+" * fmt(e.upper) * " / " * fmt(e.lower), 24),
            "+" * fmt(bu) * " / -" * fmt(bl))
end
println("-"^84)
println("""
Reading: all three methods share the same central values and the same overall
error SCALE, and both MINOS and the bootstrap reveal ASYMMETRY that the
symmetric HESSE ± cannot (e.g. a22 and p0 both skew downward in both methods) —
an independent, sampling-based corroboration of the analytic error analysis on
a real physics fit. They do NOT agree to the percent in the deep tails (e.g.
a22's MINOS −$(round(-m.merrors["a22"].lower;sigdigits=2)) vs bootstrap
−$(round(m.values[3]-bp.ci_lower[3];sigdigits=2))): with only $ndata points for
3 parameters (1 DOF) the χ² profile is far from parabolic, so the small-sample
asymmetric tails are genuinely method-dependent — treat them with care. (For a
tight, well-conditioned bootstrap-vs-MINOS agreement see the regression test in
test/test_resampling_errors.jl.) Nonparametric bootstrap and jackknife are
omitted entirely: with $ndata points / 3 params a delete-1 or
resample-with-replacement fit is degenerate (≤ 0 effective DOF).""")
