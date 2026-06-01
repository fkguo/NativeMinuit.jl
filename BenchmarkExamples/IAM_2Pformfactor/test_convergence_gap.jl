# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IAM convergence-gap regression guard (docs/dev/IAM_CONVERGENCE_GAP.md).
#
# Asserts that the high-level `Minuit(chi2_iam, paras0)` default path reaches
# the deep basin (fval ≤ 410) from the cold `paras0` seed — i.e. that the
# constructor default strategy stays at the iminuit-parity level 1, under
# which JuMinuit's single MIGRAD descends to ≈330.75 (deeper than iminuit's
# 409.89). A regression of the default back to Strategy(0) would land at
# ≈613 and fail this test.
#
# This lives in BenchmarkExamples (not test/) because the IAM FCN needs
# CSV/DataFrames/QuadGK/Interpolations + the data files, which are not
# JuMinuit test dependencies.
#
# IMPORTANT — run it against the checkout you intend to test. The repo's
# `scripts` environment `dev`s JuMinuit at a FIXED absolute path (the primary
# checkout). Running `--project=scripts` from a *git worktree* therefore
# silently loads the OTHER checkout's JuMinuit, not this one. To test THIS
# checkout, point the env at it first and clear the stale precompile cache:
#
#   julia --project=scripts -e 'using Pkg; Pkg.develop(path=pwd())'
#   rm -rf ~/.julia/compiled/v*/JuMinuit*/
#   julia -t 8 --project=scripts BenchmarkExamples/IAM_2Pformfactor/test_convergence_gap.jl
#
# (Any environment that provides the IAM deps AND resolves JuMinuit to this
# checkout works.) The first @test below — `m.strategy == Strategy(1)` — fails
# loudly if the wrong JuMinuit is loaded, so a stale-checkout run cannot
# silently "pass" with the pre-fix Strategy(0) default.
#
using LinearAlgebra, Test
BLAS.set_num_threads(1)

const IAM_DIR = @__DIR__
cd(IAM_DIR)

using CSV, DataFrames, StaticArrays, QuadGK, Interpolations
using JuMinuit

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
const paras0 = collect(lecr0)   # 8 LECs; πη normalization c dropped (paper-faithful)
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
const errs0 = fill(1e-6, 8)

@testset "IAM cold-start convergence gap (docs/dev/IAM_CONVERGENCE_GAP.md)" begin
    # Paper-faithful 7-free setup: L6 fixed (only 2L6+L8 enters ππ/Kπ/KKbar;
    # arXiv:2011.00921). The πη normalization c is dropped (not a ππ-fit param).
    fixL6!(m) = (JuMinuit.fix!(m, 6); m)

    # The constructor default must be Strategy(1) (numerical FCN).
    m_probe = JuMinuit.Minuit(chi2_iam, paras0; error = errs0)
    @test m_probe.strategy == JuMinuit.Strategy(1)

    # Default migrad!(m) (iterate=5) reaches the deep basin (≈360), beating
    # iminuit's cold-start result. Pre-0.3.0 (Strategy(0) default) this was ≈613.
    m = fixL6!(JuMinuit.Minuit(chi2_iam, paras0; error = errs0))
    JuMinuit.migrad!(m)
    fv = m.fmin.internal.state.parameters.fval
    @info "IAM default migrad! fval" fval = fv n_passes = m.n_passes
    @test fv ≤ 410.0

    # Single-shot (iterate=1) does NOT reach the deep basin on this
    # ill-conditioned surface (≈500); only the default retry does. So we only
    # assert it descended from the cold seed (informational fval logged).
    m1 = fixL6!(JuMinuit.Minuit(chi2_iam, paras0; error = errs0))
    JuMinuit.migrad!(m1; iterate = 1)
    fv1 = m1.fmin.internal.state.parameters.fval
    @info "IAM single-shot (iterate=1, default S=1) fval" fval = fv1
    @test fv1 < chi2_iam(paras0)     # made progress from the cold seed (χ²≈1269)

    # Strategy(0) with the faithful default retry also reaches the deep basin.
    # (The old retry — Simplex hop + unconditional S=2 bump — stuck at ≈613; see
    # docs/dev/IAM_CONVERGENCE_GAP.md "Closing the S=0 retry gap".)
    m0 = fixL6!(JuMinuit.Minuit(chi2_iam, paras0; error = errs0, strategy = 0))
    JuMinuit.migrad!(m0)
    fv0 = m0.fmin.internal.state.parameters.fval
    @info "IAM S=0 default retry (plain re-seed) fval" fval = fv0 n_passes = m0.n_passes
    @test fv0 ≤ 410.0
end
