# SPDX-License-Identifier: LGPL-2.1-or-later
#
# `find_deeper_minimum` on the IAM ππ fit — WORKED EXAMPLE (demonstration, not a
# unit test). Shows the basin-hopping search escaping a shallow cold-start basin,
# in BOTH of the physically relevant setups, side by side:
#
#   CASE 1 — L6ʳ FIXED (the paper-faithful setup). L6ʳ is poorly constrained by
#            these ππ data — in NLO ChPT it enters the ππ/Kπ/KK̄ sector largely via
#            the combination 2L6ʳ+L8ʳ, and the πη data that would separate it are
#            sparse — so the paper FIXES it to its input value 0.07×10⁻³ (7 free).
#   CASE 2 — L6ʳ FREE (8 LECs). Releasing L6ʳ gives the search one more direction.
#
# Both cases start from the SAME cold Strategy-1 MIGRAD and use the SAME
# data-resampling `find_deeper_minimum`; the only difference is whether L6ʳ is
# fixed. `find_deeper_minimum` honours the fixed flag (v0.4.0), so CASE 1 keeps
# L6ʳ pinned throughout the search.
#
# What the comparison shows (see the printed LEC table): the two runs land in
# DIFFERENT deep basins — the LEC sets differ across the board, not just in L6ʳ —
# and BOTH sit far from the paper's input LECs. `find_deeper_minimum` optimises χ²;
# it does not enforce physical priors. So this illustrates the TOOL (escaping a
# shallow basin, with vs. without a fixed parameter), not a recommended physical
# fit — judge physicality separately and do error analysis at the minimum you adopt.
#
# Run (needs CSV/DataFrames/StaticArrays/QuadGK + NativeMinuit; slow — each round is
# many full MIGRADs on an ~11 ms FCN; stops on convergence):
#     julia --project=. BenchmarkExamples/IAM_2Pformfactor/find_deeper_minimum_demo.jl
#   Tunable env (defaults): IAM_NDISC=20  IAM_SEED=1  IAM_MAXROUNDS=40
#
# Physics: GKPY Roy-equation ππ phase shifts; IAM with SU(3) NLO LECs.

using LinearAlgebra, Random, Statistics, Printf
BLAS.set_num_threads(1)
const NDISC = parse(Int, get(ENV, "IAM_NDISC", "20"))
const SEED  = parse(Int, get(ENV, "IAM_SEED", "1"))
const MAXR  = parse(Int, get(ENV, "IAM_MAXROUNDS", "40"))

const IAM_DIR = @__DIR__
cd(IAM_DIR)
using CSV, DataFrames, StaticArrays, QuadGK, NativeMinuit

# ── constants + model (mirror error_crosscheck.jl / bench.jl setup) ──────────
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
const L6FIX = [false, false, false, false, false, true, false, false]   # L6 (index 6) fixed

_load(f) = NativeMinuit.Data(DataFrame(CSV.File(f, header=[:w,:δ,:err], delim=' ', ignorerepeated=true)))
d00 = _load("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat")
d11 = _load("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat")
d20 = _load("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat")
const δfuns = (δ00_0, δ11, δ20)
const dats  = (d00, d11, d20)
const ndata = d00.ndata + d11.ndata + d20.ndata
const nm = ["L$i" for i in 1:8]

function chi2_8(lec)
    s = 0.0
    for c in 1:3
        d = dats[c]; δ = δfuns[c]
        @inbounds for i in 1:d.ndata
            s += (sind(d.y[i] - δ(d.x[i], lec)) / (d.err[i]*π/180))^2
        end
    end
    return s
end

struct IAMPoint; chan::Int; x::Float64; y::Float64; err::Float64; end
const pts = IAMPoint[]
for c in 1:3, i in 1:dats[c].ndata
    push!(pts, IAMPoint(c, dats[c].x[i], dats[c].y[i], dats[c].err[i]))
end

# `refit(subpts, start)` for the resampling dispatch: fit a bootstrap subset,
# warm-started from `start`, optionally with L6 fixed (so the discovery is
# consistent with the parent fit). Returns NaNs for an invalid fit.
function make_refit(fixL6::Bool)
    (subpts, start) -> begin
        chi2r(lec) = begin
            s = 0.0
            @inbounds for p in subpts
                s += (sind(p.y - δfuns[p.chan](p.x, lec)) / (p.err*π/180))^2
            end
            s
        end
        ms = Minuit(chi2r, collect(start); names = nm, errors = fill(1e-6, 8),
                    fixed = fixL6 ? L6FIX : fill(false, 8), strategy = NativeMinuit.Strategy(1))
        migrad!(ms)
        ms.valid ? collect(ms.values) : fill(NaN, 8)
    end
end

# Run one case: cold Strategy-1 fit → find_deeper_minimum (data-resampling).
function run_case(label::String, fixL6::Bool)
    nfree = fixL6 ? 7 : 8
    m = Minuit(chi2_8, collect(lecr0); names = nm, errors = fill(1e-6, 8),
               fixed = fixL6 ? L6FIX : fill(false, 8), strategy = 1)
    migrad!(m); hesse(m)
    χcold = m.fval
    t = @elapsed mdeep = find_deeper_minimum(m, make_refit(fixL6), pts;
                                             n_discovery = NDISC, max_rounds = MAXR,
                                             seed = SEED, verbose = true)
    return (; label, nfree, χcold, χdeep = mdeep.fval, dof = ndata - nfree,
            L6 = mdeep.values[6], L8 = mdeep.values[8], valid = mdeep.valid,
            params = collect(mdeep.values), t)
end

println("\n", "="^88)
println("find_deeper_minimum on the IAM ππ fit — L6 FIXED vs L6 FREE  (data-resampling)")
println("="^88)
println("data points = $ndata;  seed = $SEED, n_discovery = $NDISC, max_rounds(backstop) = $MAXR\n")

println(">>> CASE 1: L6 FIXED at 0.07×10⁻³ (paper-faithful, 7 free LECs)")
c1 = run_case("L6 fixed", true)
println("\n>>> CASE 2: L6 FREE (8 LECs)")
c2 = run_case("L6 free", false)

@printf("\n%s\n", "="^88)
@printf("%-10s  %8s  %8s  %9s  %8s   %12s\n", "case", "cold χ²", "deep χ²", "χ²/dof", "Δχ²", "L6 (×1e-3)")
@printf("%s\n", "-"^88)
for c in (c1, c2)
    @printf("%-10s  %8.2f  %8.2f  %9.3f  %8.2f   %+11.3f%s\n",
            c.label, c.χcold, c.χdeep, c.χdeep/c.dof, c.χcold - c.χdeep,
            c.L6*1e3, c.label == "L6 fixed" ? " (fixed)" : " (free)")
end
@printf("%s\n", "="^88)
@printf("L6 free reaches χ²=%.1f; the best L6-fixed basin found is χ²=%.1f  (Δ=%.1f).\n",
        c2.χdeep, c1.χdeep, c1.χdeep - c2.χdeep)
println("These are DIFFERENT deep basins — the LEC sets differ across the board, not just in")
println("L6 — and BOTH sit far from the paper's input LECs. find_deeper_minimum optimises χ²;")
println("it does NOT enforce physical priors. Releasing L6 just opens one more direction, so")
println("the search can reach a different, deeper basin; whether either basin is physical is a")
println("separate judgement. Fix the LECs your physics requires (here L6 — sparsely constrained")
println("by the πη data), search WITHIN that constraint, and assess physicality separately.")
println("Error analysis (HESSE/MINOS) belongs at the minimum you adopt, not the cold basin.")
