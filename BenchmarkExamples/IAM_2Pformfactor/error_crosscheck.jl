# SPDX-License-Identifier: LGPL-2.1-or-later
#
# IAM 3-channel ππ phase-shift fit — A CAUTIONARY error-analysis case study on a
# MULTI-BASIN surface (demonstration, NOT a unit test).
#
#   Companion/contrast to BenchmarkExamples/X3872_dip (a clean single-basin fit
#   where bootstrap/MINOS agree). IAM is ill-conditioned and strongly
#   MULTI-BASIN, and it teaches two honest lessons:
#
#   LESSON 1 — find the true minimum first. A naive Strategy-1 MIGRAD lands at
#   χ²≈379; Strategy-2 / other starts reach χ²≈310; and resample-driven basin
#   discovery (`find_solution_modes(...; refine=true)`, which flags a re-fit
#   DEEPER than the current best via `new_min`) drops further still. PHASE 1
#   automates "restart → adopt any deeper basin → repeat". Error analysis at the
#   shallow local minimum is meaningless. (The package's `find_deeper_minimum`
#   is the general-purpose version of this loop for objectives with resamplable
#   data unavailable.)
#
#   LESSON 2 — naive resampling is UNRELIABLE here, and there is no clean rescue.
#   At the true minimum we compute the LOCAL error methods (HESSE, MINOS, the
#   MC-Δχ² region from `get_contours_samples`) and, for contrast, plain
#   bootstrap + jackknife. On this surface every resample re-fits into a
#   possibly-DIFFERENT basin, so the resampling "error" is dominated by the
#   distance BETWEEN basins and is inflated by large factors (often orders of
#   magnitude, the jackknife especially) — it is not a 1σ error.
#   `find_solution_modes` confirms the resamples scatter across many basins.
#   (Tempting fixes don't hold up: keeping only the best-fit-basin resamples and
#   taking their std selects on the very σ you're measuring → truncation-biased.
#   The honest conclusion is to TRUST THE LOCAL METHODS at the true minimum.)
#
#   The χ² uses a phase-shift `sind(δ_data − δ_model)` metric over THREE channels;
#   resampling goes through the package's generic `bootstrap(refit, data)` /
#   `jackknife(refit, data)` (they apply the correct bootstrap/jackknife scaling
#   and drop invalid re-fits). We fit the 8 active LECs (bench.jl's vestigial 9th
#   parameter dropped; this is NOT the paper's L6-fixed 7-parameter fit).
#
# Run (needs CSV/DataFrames/StaticArrays/QuadGK + NativeMinuit; the resampling is
# slow — each re-fit is a full MIGRAD on an ~11 ms FCN):
#     julia --project=. BenchmarkExamples/IAM_2Pformfactor/error_crosscheck.jl
#   Tunable env (defaults): IAM_NSTART=12 IAM_NDISC=20 IAM_NBOOT=40 IAM_MC=4000 IAM_SKIP_MINOS=
#
# Physics: GKPY Roy-equation ππ phase shifts; IAM with SU(3) NLO LECs
# (Gomez Nicola & Pelaez).

using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)

const NSTART     = parse(Int, get(ENV, "IAM_NSTART", "12"))
const NDISC      = parse(Int, get(ENV, "IAM_NDISC", "20"))   # discovery resamples / round
const NBOOT      = parse(Int, get(ENV, "IAM_NBOOT", "40"))   # bootstrap resamples
const MC_NS      = parse(Int, get(ENV, "IAM_MC", "4000"))
const SKIP_MINOS = haskey(ENV, "IAM_SKIP_MINOS")

const IAM_DIR = @__DIR__
cd(IAM_DIR)
using CSV, DataFrames, StaticArrays, QuadGK, NativeMinuit

# ── constants + model (mirror bench.jl setup) ────────────────────────────────
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

# refit a resampled subset, warm-started from `start`, at the given Strategy.
# Returns NaNs for an INVALID fit so the package resamplers drop it.
function iam_refit(subpts, start, strat)
    function chi2r(lec)
        s = 0.0
        @inbounds for p in subpts
            s += (sind(p.y - δfuns[p.chan](p.x, lec)) / (p.err*π/180))^2
        end
        return s
    end
    fm = migrad(NativeMinuit.CostFunction(chi2r, 1.0), start, fill(1e-6, 8);
                strategy = NativeMinuit.Strategy(strat))
    return NativeMinuit.is_valid(fm) ? collect(fm.state.parameters.x) : fill(NaN, 8)
end

println("\n", "="^92)
println("IAM 3-channel ππ fit — error analysis on a multi-basin surface (cautionary)")
println("="^92)
println("data points = $ndata,  free LECs = 8,  DOF = $(ndata - 8)\n")

# ── PHASE 1: find the true minimum (multi-start + adopt-deeper-basin loop) ────
println("PHASE 1 — global minimisation")
seeds = [collect(lecr0)]
for j in 1:(NSTART-1)
    push!(seeds, lecr0 .* (1 .+ 0.5 .* sin.(j .* (1:8))))
end
best_f = Inf; p_best = collect(lecr0)
for st in seeds
    ms = Minuit(chi2_8, collect(st); names = nm, errors = fill(1e-6, 8), strategy = 2)
    migrad!(ms)
    ms.valid && ms.fval < best_f && (global best_f = ms.fval; global p_best = collect(ms.values))
end
println("  multi-start ($NSTART × Strategy 2): best χ² = ", round(best_f; digits=3))
m = Minuit(chi2_8, p_best; names = nm, errors = fill(1e-6, 8), strategy = 2)
migrad!(m); hesse(m)
p_star = collect(m.values)

converged = false
for iter in 1:6
    rows = [iam_refit(pts[rand(Xoshiro(900 + iter*100 + k), 1:ndata, ndata)], p_star, 1) for k in 1:NDISC]
    finite = [r for r in rows if all(isfinite, r)]
    length(finite) >= 2 || (println("  round $iter: too few valid resamples → STABLE."); global converged = true; break)
    disc = reduce(vcat, (r' for r in finite))
    md = find_solution_modes(disc, m; refine = true)
    deeper = [x for x in md if x.new_min && x.refined_valid]
    if isempty(deeper)
        println("  round $iter: none of $(length(finite)) valid resamples found a deeper basin → STABLE.")
        global converged = true
        break
    end
    bn = deeper[argmin([x.refined_fval for x in deeper])]
    println("  round $iter: deeper basin found (χ² ", round(bn.refined_fval; digits=3),
            " < ", round(m.fval; digits=3), ") → adopt + re-minimise.")
    global m = Minuit(chi2_8, collect(bn.refined_values); names = nm, errors = fill(1e-6, 8), strategy = 2)
    migrad!(m); hesse(m)
    global p_star = collect(m.values)
end
println("  ", converged ? "STABLE best fit" : "best fit after 6 rounds (cap hit — may not be global)",
        ": χ² = ", round(m.fval; digits=3), "  (χ²/dof = ", round(m.fval/(ndata-8); digits=2),
        ")  valid = ", m.valid)

# ── PHASE 2: LOCAL errors at the minimum, vs naive (unreliable) resampling ───
println("\nPHASE 2 — error analysis at the best fit")
hesse_sig = collect(m.errors)
n_minos_valid = -1
minosA = fill((NaN, NaN), 8)
if !SKIP_MINOS
    minos!(m)
    global n_minos_valid = count(i -> NativeMinuit.is_valid(m.merrors[nm[i]]), 1:8)
    global minosA = [(m.merrors[nm[i]].upper, m.merrors[nm[i]].lower) for i in 1:8]
end
mc = get_contours_samples(m; nsamples = MC_NS, cl = 1, seed = 2024)
mc_sig = [std(@view mc.samples[:, i]) for i in 1:8]

refit2(sp) = iam_refit(sp, p_star, 2)
bs = bootstrap(refit2, pts; nresample = NBOOT, seed = 2024, names = nm)   # correct bootstrap SE
jk = jackknife(refit2, pts; names = nm)                                    # correct ((g-1)/g) jackknife SE
# DIAGNOSTIC: how many basins did the (valid) resamples scatter into?
_vrows = vec(all(isfinite, bs.samples; dims = 2))
n_basins = count(_vrows) >= 2 ? length(find_solution_modes(bs.samples[_vrows, :], m)) : count(_vrows)

println("  MINOS: ", SKIP_MINOS ? "skipped" : "$n_minos_valid/8 valid",
        " | MC: $(mc.n_accepted) accepted, under_coverage=$(mc.under_coverage)",
        " | bootstrap: $(bs.n_valid)/$NBOOT valid, scattered over $n_basins candidate basin(s)",
        " | jackknife: $(jk.n_valid)/$ndata valid\n")

# ── table: LOCAL methods (trustworthy) vs RAW resampling (inflated) ──────────
fmt(x) = isnan(x) ? "—" : (a = abs(x); (a != 0 && (a < 1e-3 || a >= 1e4)) ?
         string(round(x; sigdigits=3)) : string(round(x; digits=6)))
println("  LOCAL (trust these) ............................   RAW resampling (inflated)")
println(rpad("LEC",5), rpad("value",12), rpad("HESSE",10), rpad("MINOS +up/−lo",22),
        rpad("MC-Δχ²",10), rpad("boot(RAW)",11), "jack(RAW)")
println("-"^92)
for i in 1:8
    mstr = SKIP_MINOS ? "—" : "+$(fmt(minosA[i][1]))/$(fmt(minosA[i][2]))"
    println(rpad(nm[i],5), rpad(fmt(m.values[i]),12), rpad(fmt(hesse_sig[i]),10),
            rpad(mstr,22), rpad(fmt(mc_sig[i]),10), rpad(fmt(bs.std[i]),11), fmt(jk.std[i]))
end
println("-"^92)
for (lbl, v) in (("MC-Δχ²", mc_sig), ("boot(RAW)", bs.std), ("jack(RAW)", jk.std))
    r = filter(isfinite, v ./ hesse_sig)
    isempty(r) || println("  $(rpad(lbl,10)) / HESSE : median ", round(median(r); digits=2),
                          "   max ", round(maximum(r); digits=1))
end
println("""
Reading (see the ratio lines above): at the true minimum the LOCAL methods track
each other — MC-Δχ²/HESSE median ≈ 1 — and MINOS, when it validates here, adds the
asymmetric 1σ. The RAW resampling does NOT measure a 1σ on this multi-basin surface,
and the two methods fail DIFFERENTLY:
  • bootstrap (~few× HESSE) is a BASIN MIXTURE, √(within-basin² + between-basin²):
    inflated by re-fits landing in other basins, but still the same order as HESSE.
  • jackknife (often ~10²× HESSE) is CATEGORICALLY invalid here. Delete-1 assumes a
    SMOOTH estimator, but a multi-basin argmin is not — deleting one point can flip
    the basin, and the jackknife's √(g−1)≈$(round(Int, sqrt(ndata-1)))× scaling turns
    those few jumps into orders of magnitude. It is a variance of basin LABELS, not
    a standard error.
There is no clean resampling 1σ on a multi-basin surface (basin-selecting then
taking the survivors' std selects on the σ you are measuring → biased low). The
DEFENSIBLE recipe: (1) find the true minimum first (PHASE 1 / `find_deeper_minimum`);
(2) trust the LOCAL methods (HESSE / MINOS / MC-Δχ²); and where genuinely distinct
solutions exist, report PER-MODE local errors — `find_solution_modes(…; refine=true)`
gives each mode its own HESSE/MINOS — rather than one merged resampled σ.""")
