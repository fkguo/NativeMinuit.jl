# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IAM strategy sweep — current-code fvals at S=0/1/2 for BOTH JuMinuit (native)
# and iminuit (via PyCall directly), single-shot (iterate=1) AND each library's
# default retry (iterate=5). Same cold seed `paras0`, same FCN (3 ππ datasets,
# pars[1:8]) as bench_full.jl / test_convergence_gap.jl.
#
# Set ENV `FIX9=1` to FIX the unused/flat 9th parameter (the FCN uses only
# pars[1:8]); this tests whether that flat direction is what stalls Strategy 2.
#
# Run in a throwaway env (the repo `scripts` env can't resolve — JuMinuit/IMinuit
# are unregistered + its Manifest is gitignored). From the repo root:
#   E=/tmp/iamsweep; mkdir -p $E
#   julia --project=$E -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.add(["PyCall","CSV","DataFrames","StaticArrays","QuadGK","Interpolations"])'
#   [FIX9=1] julia --project=$E BenchmarkExamples/IAM_2Pformfactor/iam_strategy_sweep.jl
#
using LinearAlgebra, Printf
BLAS.set_num_threads(1)

const IAM_DIR = @__DIR__
cd(IAM_DIR)

using CSV, DataFrames, StaticArrays, QuadGK, Interpolations
using JuMinuit
using PyCall

# ---- IAM model setup (verbatim from test_convergence_gap.jl) ----
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
const lecr0 = [0.56e-3, 1.21e-3, -2.79e-3, -0.36e-3, 1.4e-3, 0.07e-3, -0.44e-3, 0.78e-3]
const paras0 = [lecr0..., 1e-4]
data00 = JuMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
data11 = JuMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
data20 = JuMinuit.Data(DataFrame(CSV.File("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat", header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
function chisq_ps(dist::Function, data::JuMinuit.Data, par; fitrange=())
    fitrange = (isempty(fitrange) ? (1:data.ndata) : fitrange)
    res = 0.0
    @inbounds for i = fitrange[1]:fitrange[end]
        res += (sind(data.y[i] - dist(data.x[i], par)) / (data.err[i] * π/180))^2
    end
    return res
end
chi2_iam(pars) = (p8 = @views pars[1:8]; chisq_ps(δ00_0, data00, p8) + chisq_ps(δ11, data11, p8) + chisq_ps(δ20, data20, p8))
const errs0 = fill(1e-6, 9)
const NPAR  = length(paras0)
const PNAMES = ["x$(i-1)" for i in 1:NPAR]
const FIX9 = get(ENV, "FIX9", "0") == "1"   # fix the unused/flat 9th param?

# ---- iminuit via PyCall (direct) ----
const iminuit = pyimport("iminuit")
py"""
def __iam_run(f, start, errors, names, strat, iters, fix9):
    import iminuit
    m = iminuit.Minuit(f, *start, name=list(names))
    for n, e in zip(names, errors):
        m.errors[n] = float(e)
    if fix9:
        m.fixed[names[-1]] = True
    m.strategy = int(strat)
    m.migrad(iterate=int(iters))
    return (float(m.fval), bool(m.valid))
"""
function im_run(S, iter)
    objective(args...) = chi2_iam(Float64[args...])
    cb = pyfunction(objective, ntuple(_ -> Float64, NPAR)...)
    res = py"__iam_run"(cb, collect(Float64, paras0), collect(Float64, errs0), PNAMES, S, iter, FIX9)
    return (Float64(res[1]), Bool(res[2]))
end

# ---- JuMinuit (native) ----
function jm_run(S, iter)
    m = JuMinuit.Minuit(chi2_iam, paras0; error=errs0, strategy=S)
    FIX9 && JuMinuit.fix!(m, NPAR)
    JuMinuit.migrad!(m; iterate=iter)
    return (m.fmin.internal.state.parameters.fval, m.fmin.internal.is_valid)
end

# ---- logging ----
const LOG = open(joinpath(IAM_DIR, "iam_strategy_sweep.log"), "w")
say(s) = (println(s); flush(stdout); println(LOG, s); flush(LOG))

say("="^78)
say("IAM strategy sweep — JuMinuit (native) vs iminuit (PyCall), S=0/1/2")
say("  Julia $(VERSION) | iminuit $(iminuit.__version__) | Python $(PyCall.pyversion)")
say("  FCN: 3 ππ datasets (00/11/20), pars[1:8]; cold seed paras0; errs0=1e-6")
say("  9th parameter: $(FIX9 ? "FIXED (8 free)" : "FREE & unused (9 free, flat)")")
say("  χ²(paras0) = $(round(chi2_iam(paras0); digits=4))")
say("="^78)

# warmup (JIT both paths)
jm_run(1, 1); im_run(1, 1)

say("")
say(@sprintf("%-30s | %-26s | %-26s", "config", "JuMinuit (fval, valid)", "iminuit (fval, valid)"))
say("-"^86)
for (label, iter) in (("single-shot (iterate=1)", 1), ("default retry (iterate=5)", 5))
    say("$label:")
    for S in (0, 1, 2)
        jf, jv = jm_run(S, iter)
        iv_fval, iv_valid = im_run(S, iter)
        say(@sprintf("  S=%d                          | %12.4f  valid=%-5s | %12.4f  valid=%-5s",
                     S, jf, string(jv), iv_fval, string(iv_valid)))
    end
end
say("")
say("DONE")
close(LOG)
