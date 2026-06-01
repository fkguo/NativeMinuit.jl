# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IAM 2π form-factor FULL benchmark — migrad + minos + mncontour,
# comparing JuMinuit (numerical sequential) vs IMinuit.jl (PyCall →
# Python iminuit). Includes correctness cross-check at each stage.
#
# NOTE on schemes:
# - threaded numerical/AD: SKIPPED because IAM's `St4_00!` mutates a
#   module-level `const c_00_4 = zeros(ComplexF64, ...)` buffer — Phase H
#   `verify_threading=true` raises ThreadSafetyError. Demonstrating that
#   detection is itself part of this script.
# - AD (ForwardDiff): probed at the end; expected to fail because of
#   non-generic `Float64` annotations in src/amplitudes.jl etc.
#
# Run with:  julia -t 8 --project=../../scripts BenchmarkExamples/IAM_2Pformfactor/bench_full.jl

using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
println("Threads: nthreads=", Threads.nthreads(), " maxthreadid=", Threads.maxthreadid())

const IAM_DIR = @__DIR__
cd(IAM_DIR)

using CSV, DataFrames
using StaticArrays
using QuadGK
using Interpolations
using JuMinuit
using IMinuit
using ForwardDiff
using BenchmarkTools

const unit = 1.0
const fpi = 92.21unit
const mpic = 139.57018unit
const mpi0 = 134.9766unit
const meta = 547.862unit
const mkc = 493.677unit
const mk0 = 497.614unit
const mpi = (2mpic + mpi0)/3
const mk = (mkc + mk0)/2
const μ = 770.0unit
const ϵ = eps()

struct TwoBodyChannel{T<:AbstractFloat}
    m1::T
    m2::T
end
qon(s, m1, m2) = sqrt((s - (m1+m2)^2) * (s - (m1-m2)^2))/(2sqrt(s))
const ππ = TwoBodyChannel(mpi, mpi)
const KK = TwoBodyChannel(mk, mk)
const ηη = TwoBodyChannel(meta, meta)
const πη = TwoBodyChannel(mpi, meta)
const Kπ = TwoBodyChannel(mk, mpi)
const Kη = TwoBodyChannel(mk, meta)

include(joinpath(IAM_DIR, "src", "init_const.jl"))
include(joinpath(IAM_DIR, "src", "amplitudes.jl"))
include(joinpath(IAM_DIR, "src", "tmatrix.jl"))
include(joinpath(IAM_DIR, "src", "unitarity_modification.jl"))
include(joinpath(IAM_DIR, "src", "phaseshifts.jl"))

const lecr0 = [0.56e-3, 1.21e-3, -2.79e-3, -0.36e-3, 1.4e-3, 0.07e-3, -0.44e-3, 0.78e-3]
const paras0 = collect(lecr0)   # 8 LECs; πη normalization c dropped (paper-faithful, arXiv:2011.00921)
println("\nIAM setup loaded. n_pars = $(length(paras0))  (L6 fixed in each build → 7 free)")

data_GKPRY_ππ00_df = DataFrame(CSV.File("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))
data_GKPRY_ππ11_df = DataFrame(CSV.File("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))
data_GKPRY_ππ20_df = DataFrame(CSV.File("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))

data_ππ00 = JuMinuit.Data(data_GKPRY_ππ00_df)
data_ππ11 = JuMinuit.Data(data_GKPRY_ππ11_df)
data_ππ20 = JuMinuit.Data(data_GKPRY_ππ20_df)

println("Loaded data: ππ00=$(length(data_ππ00.x)) pts, ππ11=$(length(data_ππ11.x)) pts, ππ20=$(length(data_ππ20.x)) pts")

function chisq_ps(dist::Function, data::JuMinuit.Data, par; fitrange = ())
    fitrange = (isempty(fitrange) ? (1:data.ndata) : fitrange)
    res = 0.0
    @inbounds for i = fitrange[1]:fitrange[end]
        res += (sind(data.y[i] - dist(data.x[i], par)) / (data.err[i] * π/180))^2
    end
    return res
end

function chi2_iam(pars)
    p8 = @views pars[1:8]
    return chisq_ps(δ00_0, data_ππ00, p8) +
           chisq_ps(δ11,   data_ππ11, p8) +
           chisq_ps(δ20,   data_ππ20, p8)
end

println("\nχ²(par0) = ", chi2_iam(paras0))

b0 = @benchmark chi2_iam($paras0) samples=10 evals=1
println("FCN per call: ", round(median(b0).time/1e6; digits=2), " ms (allocs=", b0.allocs, ")")

const errs0 = fill(1e-6, 8)

# ─────────────────────────────────────────────────────────────────────
function time_runs(factory; n_rounds=3, warmup=true)
    warmup && factory()
    times = Float64[]
    for _ in 1:n_rounds
        GC.gc()
        t0 = time_ns()
        factory()
        push!(times, (time_ns() - t0) / 1e9)
        sleep(0.5)
    end
    sort!(times)
    return (med = times[(n_rounds+1) ÷ 2],
            min = times[1], max = times[end])
end

fmt_ms(t) = isnan(t) ? "n/a" : (t < 1.0 ? "$(round(t*1000; digits=1)) ms" : "$(round(t; digits=2)) s")

# ─────────────────────────────────────────────────────────────────────
# Phase H demo — verify_threading rejects the racey IAM FCN
# ─────────────────────────────────────────────────────────────────────
println("\n┌─ Phase H demo: verify_threading on IAM ─────────────────────")
print("│  is_thread_safe(chi2_iam, paras0)  = ")
flush(stdout)
safe_flag = JuMinuit.is_thread_safe(JuMinuit.CostFunction(chi2_iam, 1.0), paras0)
println(safe_flag, " ", safe_flag ? "(safe ✓)" : "(racey ✗ — as expected)")
println("└─")

# ─────────────────────────────────────────────────────────────────────
# Stage 1: migrad — sequential only (threaded path blocked by Phase H)
# ─────────────────────────────────────────────────────────────────────
println("\n┌─ Stage 1: MIGRAD ───────────────────────────────────────────")
results = Dict{String, NamedTuple}()
fits = Dict{String, Any}()

# (a) JuMinuit numerical sequential
function build_jm_num()
    m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0)
    JuMinuit.fix!(m, 6)            # L6 fixed (2L6+L8 degeneracy; paper)
    JuMinuit.migrad!(m)
    JuMinuit.hesse(m)
    return m
end
r = time_runs(build_jm_num; n_rounds=3)
results["jm_num"] = r
fits["jm_num"] = build_jm_num()
println("│  jm_num     migrad+hesse: med=", fmt_ms(r.med),
        "  fval=", round(fits["jm_num"].fmin.internal.state.parameters.fval; sigdigits=6),
        "  is_valid=", fits["jm_num"].fmin.internal.is_valid)

# (b) IMinuit
function build_iminuit()
    m = IMinuit.Minuit(chi2_iam, paras0; error=errs0)
    m.fixed["x5"] = true          # L6 fixed (0-based index 5; matches JuMinuit fix!(m,6))
    IMinuit.migrad(m)
    IMinuit.hesse(m)
    return m
end
r = time_runs(build_iminuit; n_rounds=3)
results["iminuit"] = r
fits["iminuit"] = build_iminuit()
println("│  iminuit    migrad+hesse: med=", fmt_ms(r.med),
        "  fval=", round(fits["iminuit"].fval; sigdigits=6),
        "  is_valid=", fits["iminuit"].valid)

# (c) JuMinuit AD — likely fails (non-generic in src/)
ad_ok = false
try
    function build_jm_ad()
        m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0,
                            grad = par -> ForwardDiff.gradient(chi2_iam, par))
        JuMinuit.migrad!(m)
        JuMinuit.hesse(m)
        return m
    end
    local r_ad = time_runs(build_jm_ad; n_rounds=2)
    results["jm_ad"] = r_ad
    fits["jm_ad"] = build_jm_ad()
    global ad_ok = true
    println("│  jm_ad      migrad+hesse: med=", fmt_ms(r_ad.med),
            "  fval=", round(fits["jm_ad"].fmin.internal.state.parameters.fval; sigdigits=6),
            "  is_valid=", fits["jm_ad"].fmin.internal.is_valid)
catch e
    println("│  jm_ad      FAILED: ", typeof(e),
            " — ", first(split(string(e), "\n"), 1))
    results["jm_ad"] = (med=NaN, min=NaN, max=NaN)
end

# (d) threaded paths
println("│  jm_th_*    SKIPPED — Phase H rejects (IAM St4_00! is racey)")
results["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
results["jm_th_ad"]  = (med=NaN, min=NaN, max=NaN)
println("└─")

# ─────────────────────────────────────────────────────────────────────
# Cross-check minima
# ─────────────────────────────────────────────────────────────────────
println("\n=== Cross-check: minima ===")
ref_x = fits["jm_num"].fmin.internal.state.parameters.x
ref_f = fits["jm_num"].fmin.internal.state.parameters.fval
println("  jm_num (ref): fval=", round(ref_f; digits=4))
for (lab, m) in fits
    lab == "jm_num" && continue
    if lab == "iminuit"
        x_l  = collect(IMinuit.args(m))
        fv_l = m.fval
    else
        x_l  = m.fmin.internal.state.parameters.x
        fv_l = m.fmin.internal.state.parameters.fval
    end
    Δf = abs(fv_l - ref_f)
    Δx_rel = maximum(abs.(x_l .- ref_x) ./ max.(abs.(ref_x), 1e-12))
    verdict = (Δf < 1e-2) ? "✓ MATCH" : "⚠ MISMATCH"
    println("  $lab: fval=", round(fv_l; digits=4),
            " Δf=", round(Δf; sigdigits=3),
            " max(Δx/x)=", round(Δx_rel; sigdigits=3),
            " → ", verdict)
end

# ─────────────────────────────────────────────────────────────────────
# Stage 2: MINOS on one slow parameter only (IAM is ~10 ms/call)
# Pick param 1 (first LEC, magnitude ~6e-4) — representative.
# ─────────────────────────────────────────────────────────────────────
println("\n┌─ Stage 2: MINOS (par 1 only — IAM is expensive) ────────────")
minos_times = Dict{String, NamedTuple}()
minos_errs = Dict{String, Tuple{Float64,Float64}}()

# JuMinuit
r = time_runs(function()
    m = build_jm_num()
    JuMinuit.minos!(m, 1)
    return m
end; n_rounds=2)
minos_times["jm_num   "] = r
m = build_jm_num(); JuMinuit.minos!(m, 1)
minos_errs["jm_num   "] = (m.minos_errors[1].lower, m.minos_errors[1].upper)
println("│  jm_num    minos(par1): med=", fmt_ms(r.med),
        "  err=", round.(minos_errs["jm_num   "]; sigdigits=4))

# IMinuit (may refuse to run MINOS if the fmin is invalid — iminuit
# raises RuntimeError("Function minimum is not valid"). Catch and
# report rather than crash the entire bench.)
try
    local rr = time_runs(function()
        local mm = build_iminuit()
        IMinuit.minos(mm, "x0")
        return mm
    end; n_rounds=2)
    minos_times["iminuit  "] = rr
    local m_im = build_iminuit(); IMinuit.minos(m_im, "x0")
    local me = m_im.merrors["x0"]
    minos_errs["iminuit  "] = (me.lower, me.upper)
    println("│  iminuit   minos(par1): med=", fmt_ms(rr.med),
            "  err=", round.(minos_errs["iminuit  "]; sigdigits=4))
catch e
    minos_times["iminuit  "] = (med=NaN, min=NaN, max=NaN)
    minos_errs["iminuit  "] = (NaN, NaN)
    println("│  iminuit   minos(par1): REFUSED — ",
            occursin("not valid", string(e)) ? "fmin invalid (is_valid=false)" :
                                                first(split(string(e), "\n"), 1))
end

if ad_ok
    r = time_runs(function()
        m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0,
                            grad = par -> ForwardDiff.gradient(chi2_iam, par))
        JuMinuit.migrad!(m); JuMinuit.hesse(m)
        JuMinuit.minos!(m, 1)
        return m
    end; n_rounds=2)
    minos_times["jm_ad    "] = r
    m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0,
                        grad = par -> ForwardDiff.gradient(chi2_iam, par))
    JuMinuit.migrad!(m); JuMinuit.hesse(m); JuMinuit.minos!(m, 1)
    minos_errs["jm_ad    "] = (m.minos_errors[1].lower, m.minos_errors[1].upper)
    println("│  jm_ad     minos(par1): med=", fmt_ms(r.med),
            "  err=", round.(minos_errs["jm_ad    "]; sigdigits=4))
else
    minos_times["jm_ad    "] = (med=NaN, min=NaN, max=NaN)
    println("│  jm_ad     SKIPPED (AD failed in migrad)")
end
minos_times["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
minos_times["jm_th_ad "] = (med=NaN, min=NaN, max=NaN)
println("└─")

# Cross-check MINOS
println("\n=== Cross-check: MINOS errors ===")
ref = minos_errs["jm_num   "]
for (lab, e) in minos_errs
    lab == "jm_num   " && continue
    dev = max(abs(e[1]-ref[1]), abs(e[2]-ref[2])) / max(abs(ref[1]), abs(ref[2]))
    verdict = dev < 0.1 ? "✓ MATCH" : "~ DEV $(round(dev*100; digits=1))%"
    println("  $lab: ", verdict)
end

# ─────────────────────────────────────────────────────────────────────
# Stage 3: MNCONTOUR (par 1 vs par 2) — small npts because IAM is slow
# ─────────────────────────────────────────────────────────────────────
const MNC_NPTS = 8
println("\n┌─ Stage 3: MNCONTOUR (par 1 vs par 2, npts=", MNC_NPTS, ") ────")
mnc_times = Dict{String, NamedTuple}()
mnc_pts = Dict{String, Vector{Tuple{Float64,Float64}}}()

# JuMinuit
r = time_runs(function()
    m = build_jm_num()
    JuMinuit.mncontour(m, 1, 2; numpoints=MNC_NPTS)
end; n_rounds=2)
mnc_times["jm_num   "] = r
m = build_jm_num()
pts = JuMinuit.mncontour(m, 1, 2; numpoints=MNC_NPTS)
mnc_pts["jm_num   "] = pts
println("│  jm_num    mncontour: med=", fmt_ms(r.med))

# IMinuit (same fmin-invalid issue may bite mncontour)
try
    local rr = time_runs(function()
        local mm = build_iminuit()
        IMinuit.mncontour(mm, "x0", "x1"; numpoints=MNC_NPTS)
    end; n_rounds=2)
    mnc_times["iminuit  "] = rr
    local m_im = build_iminuit()
    local arr = IMinuit.mncontour(m_im, "x0", "x1"; numpoints=MNC_NPTS)
    mnc_pts["iminuit  "] = size(arr,1) > 1 ?
        [(arr[i,1], arr[i,2]) for i in 1:size(arr,1)-1] :
        Tuple{Float64,Float64}[]
    println("│  iminuit   mncontour: med=", fmt_ms(rr.med),
            "  npts=", length(mnc_pts["iminuit  "]))
catch e
    mnc_times["iminuit  "] = (med=NaN, min=NaN, max=NaN)
    mnc_pts["iminuit  "] = Tuple{Float64,Float64}[]
    println("│  iminuit   mncontour: REFUSED — ",
            first(split(string(e), "\n"), 1))
end

if ad_ok
    r = time_runs(function()
        m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0,
                            grad = par -> ForwardDiff.gradient(chi2_iam, par))
        JuMinuit.migrad!(m); JuMinuit.hesse(m)
        JuMinuit.mncontour(m, 1, 2; numpoints=MNC_NPTS)
    end; n_rounds=2)
    mnc_times["jm_ad    "] = r
    m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0,
                        grad = par -> ForwardDiff.gradient(chi2_iam, par))
    JuMinuit.migrad!(m); JuMinuit.hesse(m)
    mnc_pts["jm_ad    "] = JuMinuit.mncontour(m, 1, 2; numpoints=MNC_NPTS)
    println("│  jm_ad     mncontour: med=", fmt_ms(r.med))
else
    mnc_times["jm_ad    "] = (med=NaN, min=NaN, max=NaN)
end
mnc_times["jm_th_num"] = (med=NaN, min=NaN, max=NaN)
mnc_times["jm_th_ad "] = (med=NaN, min=NaN, max=NaN)
println("└─")

# Cross-check contour
function centroid_radius(pts)
    isempty(pts) && return (NaN, NaN, NaN)
    n = length(pts)
    cx = sum(p[1] for p in pts) / n
    cy = sum(p[2] for p in pts) / n
    rmax = maximum(sqrt((p[1]-cx)^2 + (p[2]-cy)^2) for p in pts)
    return (cx, cy, rmax)
end
println("\n=== Cross-check: mncontour ===")
ref_c = centroid_radius(mnc_pts["jm_num   "])
println("  jm_num (ref): cx=", round(ref_c[1]; sigdigits=4),
        " cy=", round(ref_c[2]; sigdigits=4),
        " rmax=", round(ref_c[3]; sigdigits=4))
for (lab, pts) in mnc_pts
    lab == "jm_num   " && continue
    if isempty(pts)
        println("  $lab: EMPTY (algorithm terminated early)")
        continue
    end
    c = centroid_radius(pts)
    Δc = sqrt((c[1]-ref_c[1])^2 + (c[2]-ref_c[2])^2)
    Δr_rel = abs(c[3] - ref_c[3]) / ref_c[3]
    verdict = (Δr_rel < 0.2) ? "✓ MATCH" : "~ DEV"
    println("  $lab: cx=", round(c[1]; sigdigits=4),
            " cy=", round(c[2]; sigdigits=4),
            " rmax=", round(c[3]; sigdigits=4),
            " Δrmax/r=", round(Δr_rel*100; digits=1), "%",
            " → ", verdict)
end

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
println("\n╔════════════════════════════════════════════════════════════════════════╗")
println("║  IAM 9-LEC fit — JuMinuit vs IMinuit (median wall-time, expensive FCN)║")
println("╠══════════════╦══════════════╦══════════════╦══════════════════════════╣")
println("║  scheme      ║  migrad+hesse║  +minos(p1)  ║  mncontour(8 pts)        ║")
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
        " · FCN per-call: ", round(median(b0).time/1e6; digits=2), " ms")
println("Phase H verdict: threaded paths refused (FCN not thread-safe)")
