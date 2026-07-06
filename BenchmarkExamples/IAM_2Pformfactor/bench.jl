# SPDX-License-Identifier: LGPL-2.1-or-later
# Real IAM 2π form-factor benchmark — numerical vs AD vs threaded.
# Sources the original notebook setup; replaces `using IMinuit` with
# `using NativeMinuit` (drop-in compatible API).

using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
println("Threads: nthreads=", Threads.nthreads(), " maxthreadid=", Threads.maxthreadid())

# Need to cd to IAM dir for relative ./datajl paths in setup
const IAM_DIR = @__DIR__   # script-relative, portable across machines
cd(IAM_DIR)

# Mimic notebook environment but use NativeMinuit instead of IMinuit
using CSV, DataFrames
using StaticArrays
using QuadGK
using Interpolations
using NativeMinuit   # was: using IMinuit
using ForwardDiff

# Constants from notebook (must precede src/ includes that reference them)
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

# TwoBodyChannel (from notebook cell 13) — required by src/amplitudes.jl
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

# Source files (relative to IAM_DIR)
include(joinpath(IAM_DIR, "src", "init_const.jl"))
include(joinpath(IAM_DIR, "src", "amplitudes.jl"))
include(joinpath(IAM_DIR, "src", "tmatrix.jl"))
include(joinpath(IAM_DIR, "src", "unitarity_modification.jl"))
include(joinpath(IAM_DIR, "src", "phaseshifts.jl"))

# Note: many of these may already be in src/init_const.jl; redefining
# only if not — the const-redefinition warnings are harmless on
# repeated re-run.

# Best-fit reference LECs from Gomez Nicola, Pelaez
const lecr0 = [0.56e-3, 1.21e-3, -2.79e-3, -0.36e-3, 1.4e-3, 0.07e-3, -0.44e-3, 0.78e-3]
const paras0 = [lecr0..., 1e-4]
println("\nIAM setup loaded. n_pars = $(length(paras0))")

# ─────────────────────────────────────────────────────────────────────────
# Load all data files (paste from notebook cell 5-10)
# ─────────────────────────────────────────────────────────────────────────
data_GKPRY_ππ00_df = DataFrame(CSV.File("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))
data_GKPRY_ππ11_df = DataFrame(CSV.File("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))
data_GKPRY_ππ20_df = DataFrame(CSV.File("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat",
    header = [:w, :δ, :err], delim=' ', ignorerepeated=true))

# Data(df) is now supported directly via NativeMinuitDataFramesExt
# (drop-in IMinuit.jl compatibility — extension auto-activates when
# `using DataFrames` is in scope).
data_ππ00 = NativeMinuit.Data(data_GKPRY_ππ00_df)
data_ππ11 = NativeMinuit.Data(data_GKPRY_ππ11_df)
data_ππ20 = NativeMinuit.Data(data_GKPRY_ππ20_df)

println("Loaded data: ππ00=$(length(data_ππ00.x)) pts, ππ11=$(length(data_ππ11.x)) pts, ππ20=$(length(data_ππ20.x)) pts")

# ─────────────────────────────────────────────────────────────────────────
# Simplified chi-square — just the 3 ππ channels (subset of full notebook)
# This still exercises the full FCN chain (amplitude / T-matrix /
# phase-shift extraction) but skips the loading of less-stable data
# channels that the notebook excludes via various weight functions.
# ─────────────────────────────────────────────────────────────────────────
# From notebook cell 40 — sind-based residual to handle 360°-periodicity
function chisq_ps(dist::Function, data::NativeMinuit.Data, par; fitrange = ())
    fitrange = (isempty(fitrange) ? (1:data.ndata) : fitrange)
    res = 0.0
    @inbounds for i = fitrange[1]:fitrange[end]
        res += (sind(data.y[i] - dist(data.x[i], par)) / (data.err[i] * π/180))^2
    end
    return res
end

function chi2_iam(pars)
    p8 = @views pars[1:8]   # matches notebook cell 47 convention
    return chisq_ps(δ00_0, data_ππ00, p8) +
           chisq_ps(δ11,   data_ππ11, p8) +
           chisq_ps(δ20,   data_ππ20, p8)
end

println("\nχ²(par0) = ", chi2_iam(paras0))

# Per-call cost
using BenchmarkTools
b0 = @benchmark chi2_iam($paras0) samples=20 evals=1
println("\nFCN per call: ", round(median(b0).time/1000; digits=1), " μs (allocs=", b0.allocs, ")")

# ─────────────────────────────────────────────────────────────────────────
# 3-path comparison
# ─────────────────────────────────────────────────────────────────────────
println("\n=== migrad wall time (5-round median) ===")

function run_5(label, factory; n_rounds=5)
    try
        factory()  # warmup
        times = Float64[]
        for _ in 1:n_rounds
            t0 = time_ns()
            factory()
            push!(times, (time_ns() - t0) / 1e9)
            sleep(0.3)
        end
        sort!(times)
        med = times[(n_rounds+1) ÷ 2]
        println("  $label: median=$(round(med*1000; digits=1)) ms, min=$(round(times[1]*1000; digits=1)) max=$(round(times[end]*1000; digits=1))")
        return med
    catch e
        println("  $label: FAILED — $(typeof(e)): ", first(split(string(e), "\n"), 1))
        return NaN
    end
end

errs0 = fill(1e-6, 9)

fn_num() = (cf = NativeMinuit.CostFunction(chi2_iam, 1.0); migrad(cf, paras0, errs0); cf)
t_num = run_5("numerical 1T", fn_num; n_rounds=3)

# threaded numerical — Phase G path, works on any FCN
if Threads.nthreads() > 1
    fn_th() = (cf = NativeMinuit.CostFunction(chi2_iam, 1.0); migrad(cf, paras0, errs0; threaded_gradient=true); cf)
    t_th = run_5("threaded numerical", fn_th; n_rounds=3)
else
    println("  threaded numerical: SKIPPED (need julia -t N)")
    t_th = NaN
end

# AD — may fail on IAM because of non-generic Float64-restrictions
# in amplitude/T-matrix internals
fn_ad() = (cf = NativeMinuit.CostFunctionAD(chi2_iam, 1.0); migrad(cf, paras0, errs0); cf)
t_ad = run_5("AD (ForwardDiff)", fn_ad; n_rounds=3)

println("\n=== Summary (3-round median, IAM 3-channel ππ fit, n=9) ===")
println("  numerical 1T:        ", round(t_num*1000; digits=1), " ms")
if !isnan(t_th)
    println("  threaded numerical:  ", round(t_th*1000; digits=1), " ms (", round(t_num/t_th; digits=2), "× vs num)")
end
if !isnan(t_ad)
    println("  AD (ForwardDiff):    ", round(t_ad*1000; digits=1), " ms (", round(t_num/t_ad; digits=2), "× vs num)")
else
    println("  AD (ForwardDiff):    FAILED (non-generic types in IAM internals)")
end

# ─────────────────────────────────────────────────────────────────────────
# CORRECTNESS CROSS-CHECK — all paths must converge to the same minimum.
# Tolerance is tlr × |par_scale|; for IAM LECs (10⁻⁴–10⁻³) we use 1e-7.
# ─────────────────────────────────────────────────────────────────────────
println("\n=== Correctness cross-check: minima from each path ===")
cf_n = NativeMinuit.CostFunction(chi2_iam, 1.0)
fm_n = migrad(cf_n, paras0, errs0)
println("  numerical 1T: fval=", round(fm_n.state.parameters.fval; digits=6),
        ", x=", round.(fm_n.state.parameters.x; sigdigits=6))
println("    is_valid=", fm_n.is_valid, ", edm=", round(fm_n.state.edm; sigdigits=3))

if Threads.nthreads() > 1
    cf_t = NativeMinuit.CostFunction(chi2_iam, 1.0)
    fm_t = migrad(cf_t, paras0, errs0; threaded_gradient=true)
    println("  threaded 8T:  fval=", round(fm_t.state.parameters.fval; digits=6),
            ", x=", round.(fm_t.state.parameters.x; sigdigits=6))

    # Param-level identity (within MIGRAD tlr)
    Δx = maximum(abs.(fm_n.state.parameters.x .- fm_t.state.parameters.x))
    Δf = abs(fm_n.state.parameters.fval - fm_t.state.parameters.fval)
    println("\n  cross-check (num vs threaded):")
    println("    max|Δx| = ", round(Δx; sigdigits=3),
            "  (relative to param scale ~1e-3: ", round(Δx/1e-3; digits=3), "×)")
    println("    |Δfval| = ", round(Δf; sigdigits=3))
    if Δx < 1e-7 && Δf < 1e-3
        println("    → ✓ PASS — both paths converged to same minimum")
    else
        println("    → ⚠ DIFFERENT minima — investigate (likely different basins)")
    end
end
