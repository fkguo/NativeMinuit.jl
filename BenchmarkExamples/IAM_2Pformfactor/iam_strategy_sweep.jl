# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IAM strategy sweep — PAPER-FAITHFUL 7-free ππ fit (arXiv:2011.00921):
#   free LECs L1,L2,L3,L4,L5,L7,L8 ; L6 FIXED (in ππ/Kπ/KKbar only 2L6+L8
#   enters → the paper fixes L6); the πη normalization c is NOT a ππ-fit
#   parameter, so it is dropped (was the vestigial flat 9th param).
#
# Reports S=0/1/2 fval/valid (single-shot iterate=1 AND default retry iterate=5)
# for BOTH NativeMinuit (native) and iminuit (PyCall), plus a default-config
# (S=1 + retry) migrad+hesse timing. Same cold seed & FCN as bench_full.jl.
#
# Run in a throwaway env (the repo `scripts` env can't resolve — NativeMinuit/IMinuit
# unregistered + Manifest gitignored). From the repo root:
#   E=/tmp/iamsweep; mkdir -p $E
#   julia --project=$E -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.add(["PyCall","CSV","DataFrames","StaticArrays","QuadGK","Interpolations"])'
#   julia --project=$E BenchmarkExamples/IAM_2Pformfactor/iam_strategy_sweep.jl
#
using LinearAlgebra, Printf
BLAS.set_num_threads(1)

const IAM_DIR = @__DIR__
cd(IAM_DIR)

using CSV, DataFrames, StaticArrays, QuadGK, Interpolations
using NativeMinuit
using PyCall

# ---- IAM model setup ----
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
struct TwoBodyChannel{T<:AbstractFloat}; m1::T; m2::T; end
qon(s, m1, m2) = sqrt((s - (m1+m2)^2) * (s - (m1-m2)^2))/(2sqrt(s))
const ππ = TwoBodyChannel(mpi, mpi); const KK = TwoBodyChannel(mk, mk)
const ηη = TwoBodyChannel(meta, meta); const πη = TwoBodyChannel(mpi, meta)
const Kπ = TwoBodyChannel(mk, mpi); const Kη = TwoBodyChannel(mk, meta)
include(joinpath(IAM_DIR, "src", "init_const.jl"))
include(joinpath(IAM_DIR, "src", "amplitudes.jl"))
include(joinpath(IAM_DIR, "src", "tmatrix.jl"))
include(joinpath(IAM_DIR, "src", "unitarity_modification.jl"))
include(joinpath(IAM_DIR, "src", "phaseshifts.jl"))
# 8 LECs L1..L8 (GomezNicola uChPT seed); the πη normalization c is NOT included
# (this is a ππ-only fit). L6 is fixed below (2L6+L8 degeneracy — see the paper).
const lecr0 = [0.56e-3, 1.21e-3, -2.79e-3, -0.36e-3, 1.4e-3, 0.07e-3, -0.44e-3, 0.78e-3]
const paras0 = collect(lecr0)
data00 = NativeMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
data11 = NativeMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
data20 = NativeMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
function chisq_ps(dist::Function, data::NativeMinuit.Data, par; fitrange=())
    fitrange = (isempty(fitrange) ? (1:data.ndata) : fitrange)
    res = 0.0
    @inbounds for i = fitrange[1]:fitrange[end]
        res += (sind(data.y[i] - dist(data.x[i], par)) / (data.err[i] * π/180))^2
    end
    return res
end
chi2_iam(pars) = (p8 = @views pars[1:8]; chisq_ps(δ00_0, data00, p8) + chisq_ps(δ11, data11, p8) + chisq_ps(δ20, data20, p8))
const errs0  = fill(1e-6, 8)
const NPAR   = length(paras0)            # 8
const PNAMES = ["x$(i-1)" for i in 1:NPAR]
const L6IDX  = 6                          # L6 (1-based); fixed → 7 free

# ---- iminuit via PyCall (direct); L6 fixed ----
const iminuit = pyimport("iminuit")
py"""
def __iam_run(f, start, errors, names, strat, iters):
    import iminuit
    m = iminuit.Minuit(f, *start, name=list(names))
    for n, e in zip(names, errors):
        m.errors[n] = float(e)
    m.fixed[names[5]] = True            # L6 (0-based index 5)
    m.strategy = int(strat)
    m.migrad(iterate=int(iters))
    return (float(m.fval), bool(m.valid))
def __iam_default(f, start, errors, names):
    import iminuit
    m = iminuit.Minuit(f, *start, name=list(names))
    for n, e in zip(names, errors):
        m.errors[n] = float(e)
    m.fixed[names[5]] = True
    m.strategy = 1
    m.migrad(); m.hesse()
    return (float(m.fval), bool(m.valid))
"""
function im_run(S, iter)
    objective(args...) = chi2_iam(Float64[args...])
    cb = pyfunction(objective, ntuple(_ -> Float64, NPAR)...)
    res = py"__iam_run"(cb, collect(Float64, paras0), collect(Float64, errs0), PNAMES, S, iter)
    return (Float64(res[1]), Bool(res[2]))
end
function im_default()
    objective(args...) = chi2_iam(Float64[args...])
    cb = pyfunction(objective, ntuple(_ -> Float64, NPAR)...)
    res = py"__iam_default"(cb, collect(Float64, paras0), collect(Float64, errs0), PNAMES)
    return (Float64(res[1]), Bool(res[2]))
end

# ---- NativeMinuit (native); L6 fixed ----
function jm_run(S, iter)
    m = NativeMinuit.Minuit(chi2_iam, paras0; error=errs0, strategy=S)
    NativeMinuit.fix!(m, L6IDX)
    NativeMinuit.migrad!(m; iterate=iter)
    return (m.fmin.internal.state.parameters.fval, m.fmin.internal.is_valid)
end
function jm_default()
    m = NativeMinuit.Minuit(chi2_iam, paras0; error=errs0)   # default S=1
    NativeMinuit.fix!(m, L6IDX)
    NativeMinuit.migrad!(m); NativeMinuit.hesse(m)
    return m
end

# ---- logging ----
const LOG = open(joinpath(IAM_DIR, "iam_strategy_sweep.log"), "w")
say(s) = (println(s); flush(stdout); println(LOG, s); flush(LOG))

say("="^80)
say("IAM strategy sweep — PAPER-FAITHFUL 7-free fit (L6 fixed; no πη c)")
say("  Julia $(VERSION) | iminuit $(iminuit.__version__) | Python $(PyCall.pyversion)")
say("  FCN: 3 ππ datasets (00/11/20), pars[1:8]; cold seed paras0; errs0=1e-6")
say("  free: L1,L2,L3,L4,L5,L7,L8 (7) ; L6 FIXED = $(paras0[L6IDX])")
say("  χ²(paras0) = $(round(chi2_iam(paras0); digits=4))")
say("="^80)

# warmup
jm_run(1, 1); im_run(1, 1)

say("")
say(@sprintf("%-28s | %-26s | %-26s", "config", "NativeMinuit (fval, valid)", "iminuit (fval, valid)"))
say("-"^84)
for (label, iter) in (("single-shot (iterate=1)", 1), ("default retry (iterate=5)", 5))
    say("$label:")
    for S in (0, 1, 2)
        jf, jv = jm_run(S, iter)
        ifv, ivd = im_run(S, iter)
        say(@sprintf("  S=%d                        | %12.4f  valid=%-5s | %12.4f  valid=%-5s",
                     S, jf, string(jv), ifv, string(ivd)))
    end
end

# ---- default-config timing (S=1 + retry, migrad+hesse) ----
jm_default(); im_default()   # warm
jmm = jm_default()
jm_t = minimum([(GC.gc(); @elapsed jm_default()) for _ in 1:3])
jm_fv = jmm.fmin.internal.state.parameters.fval; jm_vd = jmm.fmin.internal.is_valid
imr = im_default()
im_t = minimum([(GC.gc(); @elapsed im_default()) for _ in 1:3])
say("")
say("default S=1 + retry, migrad+hesse (min of 3 rounds):")
say(@sprintf("  NativeMinuit : %7.2f s   fval=%.4f  valid=%s", jm_t, jm_fv, string(jm_vd)))
say(@sprintf("  iminuit  : %7.2f s   fval=%.4f  valid=%s", im_t, Float64(imr[1]), string(Bool(imr[2]))))
say("")
say("DONE")
close(LOG)
