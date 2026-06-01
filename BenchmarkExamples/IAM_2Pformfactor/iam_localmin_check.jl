# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Local-conditioning check: seed BOTH backends NEAR a JuMinuit minimum and see
# whether they converge to the SAME point. If they agree locally, the cold-start
# JuMinuit/iminuit split is a far-from-minimum / multi-basin conditioning effect,
# not a systematic algorithm difference. (7-free paper setup: L6 fixed.)
#
using LinearAlgebra, Printf
BLAS.set_num_threads(1)
const IAM_DIR = @__DIR__
cd(IAM_DIR)
using CSV, DataFrames, StaticArrays, QuadGK, Interpolations
using JuMinuit
using PyCall

const unit = 1.0
const fpi = 92.21unit; const mpic = 139.57018unit; const mpi0 = 134.9766unit
const meta = 547.862unit; const mkc = 493.677unit; const mk0 = 497.614unit
const mpi = (2mpic + mpi0)/3; const mk = (mkc + mk0)/2; const μ = 770.0unit; const ϵ = eps()
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
const paras0 = collect(lecr0)
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
const NPAR = 8
const PNAMES = ["x$(i-1)" for i in 1:NPAR]
const L6IDX = 6
const FREE  = [1,2,3,4,5,7,8]
# paper Fit-2 per-LEC uncertainties (×1e-3); L6 fixed (placeholder error)
const PROPERR = [0.02e-3, 0.03e-3, 0.03e-3, 0.02e-3, 0.61e-3, 1e-6, 0.12e-3, 0.29e-3]

const iminuit = pyimport("iminuit")
py"""
def __run(f, start, errors, names, strat):
    import iminuit
    m = iminuit.Minuit(f, *start, name=list(names))
    for n,e in zip(names, errors): m.errors[n] = float(e)
    m.fixed[names[5]] = True
    m.strategy = int(strat)
    m.migrad()
    return (float(m.fval), bool(m.valid), [float(v) for v in m.values])
"""
function im_from(x0, err0; S=1)
    obj(args...) = chi2_iam(Float64[args...])
    cb = pyfunction(obj, ntuple(_->Float64, NPAR)...)
    r = py"__run"(cb, collect(Float64,x0), collect(Float64,err0), PNAMES, S)
    return (Float64(r[1]), Bool(r[2]), Float64.(r[3]))
end
function jm_from(x0, err0; S=1)
    m = JuMinuit.Minuit(chi2_iam, collect(Float64,x0); error=collect(Float64,err0), strategy=S)
    JuMinuit.fix!(m, L6IDX)
    JuMinuit.migrad!(m)
    return (m.fmin.internal.state.parameters.fval, m.fmin.internal.is_valid, collect(Float64, m.values))
end

const LOG = open(joinpath(IAM_DIR, "iam_localmin_check.log"), "w")
say(s) = (println(s); flush(stdout); println(LOG, s); flush(LOG))
cmp(a,b) = maximum(abs(a[i]-b[i]) for i in FREE)

say("="^74)
say("IAM local-conditioning check (7-free, L6 fixed) | iminuit $(iminuit.__version__)")
say("="^74)

say("\nStep 1 — JuMinuit minimum from the COLD seed:")
jf, jv, xstar = jm_from(paras0, fill(1e-6, 8))
say(@sprintf("  fval=%.4f  valid=%s", jf, string(jv)))
say("  xstar(free L1..L8) = $(round.(xstar; sigdigits=5))")

say("\nStep 2a — seed BOTH exactly AT xstar (per-LEC errors):")
jf2, jv2, jx = jm_from(xstar, PROPERR);  if2, iv2, ix = im_from(xstar, PROPERR)
say(@sprintf("  JuMinuit: fval=%.4f valid=%s", jf2, string(jv2)))
say(@sprintf("  iminuit : fval=%.4f valid=%s", if2, string(iv2)))
say(@sprintf("  max|Δparam|(free) = %.3e ; Δfval = %.3e", cmp(jx,ix), jf2-if2))

say("\nStep 2b — seed BOTH at xstar + 0.5·σ perturbation:")
xseed = xstar .+ 0.5 .* PROPERR
jf3, jv3, jx3 = jm_from(xseed, PROPERR);  if3, iv3, ix3 = im_from(xseed, PROPERR)
say(@sprintf("  JuMinuit: fval=%.4f valid=%s", jf3, string(jv3)))
say(@sprintf("  iminuit : fval=%.4f valid=%s", if3, string(iv3)))
say(@sprintf("  max|Δparam|(free) = %.3e ; Δfval = %.3e", cmp(jx3,ix3), jf3-if3))

say("\nStep 2c — seed BOTH at xstar + 2·σ perturbation (bigger kick):")
xseed2 = xstar .+ 2.0 .* PROPERR
jf4, jv4, jx4 = jm_from(xseed2, PROPERR);  if4, iv4, ix4 = im_from(xseed2, PROPERR)
say(@sprintf("  JuMinuit: fval=%.4f valid=%s", jf4, string(jv4)))
say(@sprintf("  iminuit : fval=%.4f valid=%s", if4, string(iv4)))
say(@sprintf("  max|Δparam|(free) = %.3e ; Δfval = %.3e", cmp(jx4,ix4), jf4-if4))
say("\nDONE")
close(LOG)
