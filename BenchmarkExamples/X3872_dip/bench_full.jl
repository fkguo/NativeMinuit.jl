# SPDX-License-Identifier: LGPL-2.1-or-later
#
# X(3872) dip-fit FULL benchmark — migrad + minos + mncontour, comparing
# JuMinuit (numerical / AD / threaded numerical / threaded AD) vs
# IMinuit.jl (PyCall → Python iminuit). Includes correctness cross-check
# at each stage.
#
# Source of the published fit:
#   V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
#   "How does the X(3872) show up in e+e- collisions: Dip versus peak",
#   Phys. Rev. D 109 (2024) 11, L111501, arXiv:2404.12003.
#   https://inspirehep.net/literature/2778938
#
# Run with:  julia -t 8 --project=../../scripts BenchmarkExamples/X3872_dip/bench_full.jl

using JuMinuit
using IMinuit
using DataFrames, CSV, QuadGK
using LinearAlgebra, Random, Statistics, BenchmarkTools
using ForwardDiff

BLAS.set_num_threads(1)
println("Threads: nthreads=", Threads.nthreads(), " maxthreadid=", Threads.maxthreadid())

const X3872_DIR = @__DIR__
cd(X3872_DIR)
include(joinpath(X3872_DIR, "hadronmasses.jl"))

const mρ = 0.750 * unit_choice
const mω = 0.782 * unit_choice

function quadgauss(f, x, w)
    res = zero(f(x[1]))
    @inbounds for i in eachindex(x)
        res += f(x[i]) * w[i]
    end
    return res
end

const QX, QW = gauss(128, 10, -10)

const data_df = DataFrame(CSV.File("data.csv", header=["w", "y", "err"]))
println("Loaded ", nrow(data_df), " data points")

function xsqrt(x)
    imag(x) >= 0 ? sqrt(x + 0im) : -sqrt(x - 0im)
end

λ(x, y, z) = x^2 + y^2 + z^2 - 2x*y - 2y*z - 2z*x
qsq(E, m1, m2) = λ(E^2, m1^2, m2^2) / (4E^2)

function t11_lo_constrained(e, a11, a22, a22eff; mrho)
    Σ2 = md0 + mdstar0
    μ2 = md0 * mdstar0 / Σ2
    k1 = xsqrt(qsq(e, mjψ, mrho))
    k2 = xsqrt(2μ2 * (e - md0 - mdstar0))
    return -8π * Σ2 * (1/a22 - 1im*k2) / (1/a11 - 1im*k1) / (1/a22eff - 1im*k2)
end

resolution(x, σ) = exp(-0.5*(x/σ)^2) / (sqrt(2π)*σ)

function model1(x, par; mrho)
    p0, r, a22 = par
    a22 = a22 / ħc
    a22eff = (-6.39 + 11.74im) / ħc
    inv_a22eff = 1 / a22eff
    k1 = real(xsqrt(qsq(x, mjψ, mrho)))
    a11 = 1 / (k1 / imag(inv_a22eff) * (real(inv_a22eff) - 1/a22))
    amp2 = p0^2 * abs2(1 + r * t11_lo_constrained(x, a11, a22, a22eff; mrho))
    return amp2
end

function res_model1(x, σ, par; mrho)
    quadgauss(y -> model1(x - y, par; mrho) * resolution(y, σ), QX, QW)
end
res_model1(x, par; mrho) = res_model1(x, 1.7, par; mrho)

const mrho_model4 = 775.0 - 75.0im

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

const PAR0 = [3.0, 0.0001, -4.0]
const ERRS = [0.1, 0.01, 1.0]

println("\nχ²(par0) = ", chi2_x3872(PAR0))

println("\n=== Per-call FCN cost (real X3872) ===")
b0 = @benchmark chi2_x3872($PAR0) samples=50 evals=1
println("  median = ", round(median(b0).time/1000; digits=1), " μs/call, allocs=", b0.allocs)

# ─────────────────────────────────────────────────────────────────────
# Timing helper — N rounds, return median wall-time (s)
# ─────────────────────────────────────────────────────────────────────
function time_runs(factory; n_rounds=5, warmup=true)
    warmup && factory()
    times = Float64[]
    for _ in 1:n_rounds
        GC.gc()
        t0 = time_ns()
        factory()
        push!(times, (time_ns() - t0) / 1e9)
        sleep(0.2)
    end
    sort!(times)
    return (med = times[(n_rounds+1) ÷ 2],
            min = times[1], max = times[end])
end

fmt_ms(t) = isnan(t) ? "n/a" : "$(round(t*1000; digits=1)) ms"

# ─────────────────────────────────────────────────────────────────────
# Stage 1: migrad — all schemes
# ─────────────────────────────────────────────────────────────────────
println("\n┌─ Stage 1: MIGRAD ───────────────────────────────────────────")

results = Dict{String, NamedTuple}()
fits = Dict{String, Any}()

# (a) JuMinuit numerical sequential — high-level Minuit API
function build_jm_num()
    m = JuMinuit.Minuit(chi2_x3872, PAR0; error=ERRS)
    JuMinuit.migrad!(m)
    JuMinuit.hesse(m)
    return m
end
r = time_runs(build_jm_num; n_rounds=5)
results["jm_num"] = r
fits["jm_num"] = build_jm_num()
println("│  jm_num     migrad+hesse: med=", fmt_ms(r.med),
        "  fval=", round(fits["jm_num"].fmin.internal.state.parameters.fval; sigdigits=6),
        "  is_valid=", fits["jm_num"].fmin.internal.is_valid)

# (b) JuMinuit AD
function build_jm_ad()
    m = JuMinuit.Minuit(chi2_x3872, PAR0; error=ERRS,
                        grad = par -> ForwardDiff.gradient(chi2_x3872, par))
    JuMinuit.migrad!(m)
    JuMinuit.hesse(m)
    return m
end
r = time_runs(build_jm_ad; n_rounds=5)
results["jm_ad"] = r
fits["jm_ad"] = build_jm_ad()
println("│  jm_ad      migrad+hesse: med=", fmt_ms(r.med),
        "  fval=", round(fits["jm_ad"].fmin.internal.state.parameters.fval; sigdigits=6),
        "  is_valid=", fits["jm_ad"].fmin.internal.is_valid)

# (c) JuMinuit threaded numerical
if Threads.nthreads() > 1
    function build_jm_th_num()
        m = JuMinuit.Minuit(chi2_x3872, PAR0; error=ERRS, threaded_gradient=true)
        JuMinuit.migrad!(m)
        JuMinuit.hesse(m)
        return m
    end
    r = time_runs(build_jm_th_num; n_rounds=5)
    results["jm_th_num"] = r
    fits["jm_th_num"] = build_jm_th_num()
    println("│  jm_th_num  migrad+hesse: med=", fmt_ms(r.med),
            "  fval=", round(fits["jm_th_num"].fmin.internal.state.parameters.fval; sigdigits=6),
            "  is_valid=", fits["jm_th_num"].fmin.internal.is_valid)

    # (d) JuMinuit threaded AD
    function build_jm_th_ad()
        m = JuMinuit.Minuit(chi2_x3872, PAR0; error=ERRS,
                            grad = par -> ForwardDiff.gradient(chi2_x3872, par),
                            threaded_gradient=true)
        JuMinuit.migrad!(m)
        JuMinuit.hesse(m)
        return m
    end
    r = time_runs(build_jm_th_ad; n_rounds=5)
    results["jm_th_ad"] = r
    fits["jm_th_ad"] = build_jm_th_ad()
    println("│  jm_th_ad   migrad+hesse: med=", fmt_ms(r.med),
            "  fval=", round(fits["jm_th_ad"].fmin.internal.state.parameters.fval; sigdigits=6),
            "  is_valid=", fits["jm_th_ad"].fmin.internal.is_valid)
else
    results["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
    results["jm_th_ad"]  = (med=NaN, min=NaN, max=NaN)
    println("│  jm_th_*:   SKIPPED (need julia -t N>1)")
end

# (e) IMinuit (Python iminuit via PyCall)
function build_iminuit()
    m = IMinuit.Minuit(chi2_x3872, PAR0; error=ERRS)
    IMinuit.migrad(m)
    IMinuit.hesse(m)
    return m
end
r = time_runs(build_iminuit; n_rounds=5)
results["iminuit"] = r
fits["iminuit"] = build_iminuit()
im_fval = fits["iminuit"].fval
println("│  iminuit    migrad+hesse: med=", fmt_ms(r.med),
        "  fval=", round(im_fval; sigdigits=6),
        "  is_valid=", fits["iminuit"].valid)

println("└─")

# ─────────────────────────────────────────────────────────────────────
# Cross-check: minima agree
#
# EXPECTED on this fit: `jm_ad` (and `jm_th_ad`) land ~Δx=0.0149 from
# `jm_num` at a marginally deeper fval, so the 1e-3 threshold below
# prints "⚠ MISMATCH" for the AD path. This is NOT a regression — it is
# inherent to this weakly-constrained fit and is documented in full in
# `docs/AD_OFFSET_X3872.md`. In short:
#   • The numerical and AD MIGRAD seeds use different diagonal 2nd-deriv
#     estimates (g2): numerical refines g2 by finite differences; the AD
#     path uses the rough 2·up/dirin² estimate. This is C++ Minuit2-
#     faithful (MnSeedGenerator.cxx:60 vs :119-122) — iminuit matches
#     jm_num here only because iminuit is also numerical.
#   • The X(3872) χ² minimum is a flat, degenerate valley (MINOS cannot
#     close most 1σ contours), so MIGRAD's edm<goal stop is satisfied at
#     trajectory-dependent points. The two seeds → two valid valley-floor
#     minima.
#   • The offset is only 0.5–0.7% of the 1σ parabolic error — physically
#     negligible. The AD gradient itself is exact (chunk-invariant, agrees
#     with finite-difference to ~1e-8).
# A "⚠ MISMATCH" here is therefore informational. On a well-constrained
# fit, a mismatch WOULD signal a real bug — the check is kept as-is.
# ─────────────────────────────────────────────────────────────────────
println("\n=== Cross-check: minima ===")
ref_x   = fits["jm_num"].fmin.internal.state.parameters.x
ref_fv  = fits["jm_num"].fmin.internal.state.parameters.fval
println("  jm_num (reference):  x=", round.(ref_x; sigdigits=6), " fval=", round(ref_fv; sigdigits=6))
for (lab, m) in fits
    lab == "jm_num" && continue
    if lab == "iminuit"
        x_l   = collect(IMinuit.args(m))
        fv_l  = m.fval
    else
        x_l   = m.fmin.internal.state.parameters.x
        fv_l  = m.fmin.internal.state.parameters.fval
    end
    Δx = maximum(abs.(x_l .- ref_x))
    Δf = abs(fv_l - ref_fv)
    verdict = (Δx < 1e-3 && Δf < 1e-3) ? "✓ MATCH" : "⚠ MISMATCH"
    println("  $lab: Δx=", round(Δx; sigdigits=3),
            " Δfval=", round(Δf; sigdigits=3), " → ", verdict)
end
println("  (note: jm_ad ⚠ MISMATCH is EXPECTED & negligible here — flat",
        " degenerate valley + C++-faithful AD seed g2; see docs/AD_OFFSET_X3872.md)")

# ─────────────────────────────────────────────────────────────────────
# Stage 2: MINOS — all parameters, all schemes (reuses pre-fit Minuit
# from Stage 1; we time only the minos call.)
# ─────────────────────────────────────────────────────────────────────
println("\n┌─ Stage 2: MINOS (all free params) ──────────────────────────")
minos_times = Dict{String, NamedTuple}()
minos_errs = Dict{String, Vector{Tuple{Float64,Float64}}}()  # (lower, upper) per param

function run_minos_jm(label, build_fit)
    r = time_runs(function ()
        m = build_fit()
        JuMinuit.minos!(m)
        return m
    end; n_rounds=3)
    minos_times[label] = r
    m = build_fit(); JuMinuit.minos!(m)
    errs = [(m.minos_errors[i].lower, m.minos_errors[i].upper) for i in 1:length(PAR0)]
    minos_errs[label] = errs
    println("│  $label  minos: med=", fmt_ms(r.med),
            "  errs=", [round.(e; sigdigits=4) for e in errs])
end

run_minos_jm("jm_num   ", build_jm_num)
run_minos_jm("jm_ad    ", build_jm_ad)
if Threads.nthreads() > 1
    run_minos_jm("jm_th_num", build_jm_th_num)
    run_minos_jm("jm_th_ad ", build_jm_th_ad)
else
    minos_times["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
    minos_times["jm_th_ad "] = (med=NaN, min=NaN, max=NaN)
end

# IMinuit minos
r = time_runs(function ()
    m = build_iminuit()
    IMinuit.minos(m)
    return m
end; n_rounds=3)
minos_times["iminuit  "] = r
m_im = build_iminuit(); IMinuit.minos(m_im)
# IMinuit param names are auto-generated as "x0","x1","x2"
errs_im = Tuple{Float64,Float64}[]
for i in 0:(length(PAR0)-1)
    me = m_im.merrors["x$i"]
    push!(errs_im, (me.lower, me.upper))
end
minos_errs["iminuit  "] = errs_im
println("│  iminuit   minos: med=", fmt_ms(r.med),
        "  errs=", [round.(e; sigdigits=4) for e in errs_im])
println("└─")

# Cross-check MINOS errors
println("\n=== Cross-check: MINOS errors (ref=jm_num) ===")
ref = minos_errs["jm_num   "]
for (lab, errs) in minos_errs
    lab == "jm_num   " && continue
    max_dev = 0.0
    for i in eachindex(ref)
        max_dev = max(max_dev, abs(ref[i][1] - errs[i][1]), abs(ref[i][2] - errs[i][2]))
    end
    verdict = max_dev < 1e-3 ? "✓ MATCH" : "⚠ DEV $(round(max_dev; sigdigits=3))"
    println("  $lab: ", verdict)
end

# ─────────────────────────────────────────────────────────────────────
# Stage 3: MNCONTOUR (par1 = p0, par2 = a22 — physically interesting)
# ─────────────────────────────────────────────────────────────────────
const MNC_NPTS = 20
const MNC_P1, MNC_P2 = 1, 2   # (p0, r) — well-constrained pair; (1,3)
                              # fails on a22 (hard direction → MnContours
                              # "unable to find first two points")
println("\n┌─ Stage 3: MNCONTOUR (par ", MNC_P1, " vs par ", MNC_P2, ", npts=", MNC_NPTS, ") ────")
mnc_times = Dict{String, NamedTuple}()
mnc_pts = Dict{String, Vector{Tuple{Float64,Float64}}}()

function run_mnc_jm(label, build_fit)
    r = time_runs(function ()
        m = build_fit()
        JuMinuit.mncontour(m, MNC_P1, MNC_P2; numpoints=MNC_NPTS)
    end; n_rounds=3)
    mnc_times[label] = r
    m = build_fit()
    pts = JuMinuit.mncontour(m, MNC_P1, MNC_P2; numpoints=MNC_NPTS)
    mnc_pts[label] = pts
    println("│  $label  mncontour: med=", fmt_ms(r.med))
end

run_mnc_jm("jm_num   ", build_jm_num)
run_mnc_jm("jm_ad    ", build_jm_ad)
if Threads.nthreads() > 1
    run_mnc_jm("jm_th_num", build_jm_th_num)
    run_mnc_jm("jm_th_ad ", build_jm_th_ad)
else
    mnc_times["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
    mnc_times["jm_th_ad "] = (med=NaN, min=NaN, max=NaN)
end

# IMinuit mncontour
imnames = ["x$(MNC_P1-1)", "x$(MNC_P2-1)"]
r = time_runs(function ()
    m = build_iminuit()
    IMinuit.mncontour(m, imnames[1], imnames[2]; numpoints=MNC_NPTS)
end; n_rounds=3)
mnc_times["iminuit  "] = r
m_im = build_iminuit()
arr = IMinuit.mncontour(m_im, imnames[1], imnames[2]; numpoints=MNC_NPTS)
# arr is Matrix{Float64} (npts+1, 2) closed; drop last
mnc_pts["iminuit  "] = size(arr,1) > 1 ?
    [(arr[i,1], arr[i,2]) for i in 1:size(arr,1)-1] :
    Tuple{Float64,Float64}[]
println("│  iminuit   mncontour: med=", fmt_ms(r.med))
println("└─")

# Cross-check contour: compare centroid + max radius
function centroid_radius(pts)
    isempty(pts) && return (NaN, NaN, NaN)
    n = length(pts)
    cx = sum(p[1] for p in pts) / n
    cy = sum(p[2] for p in pts) / n
    rmax = maximum(sqrt((p[1]-cx)^2 + (p[2]-cy)^2) for p in pts)
    return (cx, cy, rmax)
end
println("\n=== Cross-check: mncontour (centroid + max radius) ===")
ref_c = centroid_radius(mnc_pts["jm_num   "])
println("  jm_num (ref): cx=", round(ref_c[1]; sigdigits=5),
        " cy=", round(ref_c[2]; sigdigits=5),
        " rmax=", round(ref_c[3]; sigdigits=4),
        " npts=", length(mnc_pts["jm_num   "]))
for (lab, pts) in mnc_pts
    lab == "jm_num   " && continue
    if isempty(pts)
        println("  $lab: EMPTY (algorithm terminated early)")
        continue
    end
    c = centroid_radius(pts)
    Δc = sqrt((c[1]-ref_c[1])^2 + (c[2]-ref_c[2])^2)
    Δr = abs(c[3] - ref_c[3])
    verdict = (Δc < 0.05*ref_c[3] && Δr < 0.1*ref_c[3]) ? "✓ MATCH" : "~ approximate"
    println("  $lab: cx=", round(c[1]; sigdigits=5),
            " cy=", round(c[2]; sigdigits=5),
            " rmax=", round(c[3]; sigdigits=4),
            " npts=", length(pts),
            " Δcentroid=", round(Δc; sigdigits=3),
            " Δrmax=", round(Δr; sigdigits=3),
            " → ", verdict)
end

# ─────────────────────────────────────────────────────────────────────
# Final summary table
# ─────────────────────────────────────────────────────────────────────
println("\n╔════════════════════════════════════════════════════════════════════════╗")
println("║  X(3872) dip fit — JuMinuit vs IMinuit (median wall-time, n=3 params) ║")
println("╠══════════════╦══════════════╦══════════════╦══════════════════════════╣")
println("║  scheme      ║  migrad+hesse║  +minos(all) ║  mncontour(20 pts)       ║")
println("╠══════════════╬══════════════╬══════════════╬══════════════════════════╣")
for lab in ["jm_num   ", "jm_ad    ", "jm_th_num", "jm_th_ad ", "iminuit  "]
    key_m = strip(lab)
    println("║  ", lab, "  ║  ",
            rpad(fmt_ms(results[key_m].med), 12), "║  ",
            rpad(fmt_ms(minos_times[lab].med), 12), "║  ",
            rpad(fmt_ms(mnc_times[lab].med), 24), "║")
end
println("╚══════════════╩══════════════╩══════════════╩══════════════════════════╝")
println("Threads: ", Threads.nthreads(),
        " · FCN per-call: ", round(median(b0).time/1000; digits=1), " μs",
        " · ", nrow(data_df), " data points")
