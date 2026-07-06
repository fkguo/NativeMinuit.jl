# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Theory-vs-data overlays for the find_deeper_minimum_demo fits, plus the physical
# reference. Produces two PNGs in this directory:
#
#   iam_fdm_deep_basins.png  — the two DEEP basins find_deeper_minimum reaches
#       (L6 free, χ²≈255  vs  L6 fixed, χ²≈272). Both describe the data, but both
#       are non-physical (LECs far from the paper input) and show spurious narrow
#       phase features — see find_deeper_minimum_demo.jl and its LEC table.
#   iam_physical_fit.png     — a PHYSICAL fit (L6 fixed at the paper value 0.07e-3,
#       MIGRAD from the paper's input LECs, NOT basin-hopped). Higher χ² than the
#       deep basins but physically sensible: clean, smooth phase shifts.
#
# δ₀⁰ is plotted with a proper phase UNWRAP (the 0.5·atand principal value wraps at
# ±90°); δ₁¹/δ₀² use the package's continuity-aware functions.
#
# Run: julia --project=. BenchmarkExamples/IAM_2Pformfactor/plot_fits_vs_data.jl
#      (needs CSV/DataFrames/StaticArrays/QuadGK/Plots/LaTeXStrings + NativeMinuit)

using LinearAlgebra, Printf
const IAM_DIR = @__DIR__
cd(IAM_DIR)
using CSV, DataFrames, StaticArrays, QuadGK, NativeMinuit
using Plots, LaTeXStrings
gr(); default(framestyle = :box, minorticks = 5, fg_color_legend = :lightgray, guidefontsize = 11)

const unit=1.0; const fpi=92.21unit; const mpic=139.57018unit; const mpi0=134.9766unit
const meta=547.862unit; const mkc=493.677unit; const mk0=497.614unit
const mpi=(2mpic+mpi0)/3; const mk=(mkc+mk0)/2; const μ=770.0unit; const ϵ=eps()
struct TwoBodyChannel{T<:AbstractFloat}; m1::T; m2::T; end
qon(s,m1,m2)=sqrt((s-(m1+m2)^2)*(s-(m1-m2)^2))/(2sqrt(s))
const ππ=TwoBodyChannel(mpi,mpi); const KK=TwoBodyChannel(mk,mk); const ηη=TwoBodyChannel(meta,meta)
const πη=TwoBodyChannel(mpi,meta); const Kπ=TwoBodyChannel(mk,mpi); const Kη=TwoBodyChannel(mk,meta)
for f in ("init_const.jl","amplitudes.jl","tmatrix.jl","unitarity_modification.jl","phaseshifts.jl")
    include(joinpath(IAM_DIR,"src",f))
end
const lecr0 = [0.56e-3, 1.21e-3, -2.79e-3, -0.36e-3, 1.4e-3, 0.07e-3, -0.44e-3, 0.78e-3]
const L6FIX = [false,false,false,false,false,true,false,false]
_load(f)=NativeMinuit.Data(DataFrame(CSV.File(f,header=[:w,:δ,:err],delim=' ',ignorerepeated=true)))
const d00=_load("./datajl/pipi/pipi00_Roy-GKPY_PRD83_074004.dat")
const d11=_load("./datajl/pipi/pipi11_Roy-GKPY_PRD83_074004.dat")
const d20=_load("./datajl/pipi/pipi20_Roy-GKPY_PRD83_074004.dat")
const δfuns=(δ00_0,δ11,δ20); const dats=(d00,d11,d20)
const nm=["L$i" for i in 1:8]
chi2_8(lec)=(s=0.0; for c in 1:3; d=dats[c]; δ=δfuns[c]; @inbounds for i in 1:d.ndata; s+=(sind(d.y[i]-δ(d.x[i],lec))/(d.err[i]*π/180))^2; end; end; s)

# find_deeper_minimum results (seed=1, n_discovery=20), from find_deeper_minimum_demo.jl
const P_FREE = [0.0008413723943717318, 0.001315878288298748, -0.003482519227202564,
                0.00029283367767295827, -0.0014041579878597986, -0.0004065704815191333,
                -0.0005944030414758117, 0.0007941959289395085]               # L6 free,  χ²≈255
const P_FIX  = [0.0022857838317941817, 0.0010714473146124974, -0.005617924467477749,
                -0.00033981486976600246, 0.0003850541239538042, 7.0e-5,
                0.00010264682979175501, 0.0007333473333771101]               # L6 fixed, χ²≈272

# physical reference: the cold L6-fixed MIGRAD from the paper's input LECs (Strategy 1,
# NOT basin-hopped) — the physically-sensible local minimum find_deeper_minimum starts
# from. Higher χ² than the deep basins, but clean (single 90° wrap) and physical LECs.
mphys = Minuit(chi2_8, collect(lecr0); names=nm, errors=fill(1e-6,8), fixed=L6FIX, strategy=1)
migrad!(mphys); hesse(mphys)
const P_PHYS = collect(mphys.values)

unwrap(raw::Vector{Float64}; thresh=100.0) = begin
    out=copy(raw); off=0.0
    @inbounds for i in 2:length(raw)
        d=raw[i]-raw[i-1]; d<-thresh && (off+=180.0); d>thresh && (off-=180.0); out[i]=raw[i]+off
    end; out
end
const wv = collect(range(310.0, 1200.0; length=900))
δ00u(p) = unwrap([δ00_0(w,p) for w in wv])      # I=0 S-wave: unwrap the ±90° principal value
δ11u(p) = unwrap([δ11_0(w,p) for w in wv])      # I=1 P-wave: same
# `u11`: unwrap δ11 (clean — for the physical fit) vs the package `δ11` continuity
# convention (for the non-physical deep basins, whose δ11 genuinely misbehaves — the
# unwrap there lands on the wrong branch, so we show the honest package curve).
curves!(plt, p; ls, lab="", u11=false) = begin
    plot!(plt, wv, δ00u(p);                                  color=:dodgerblue, lw=2, ls=ls, label=lab)
    plot!(plt, wv, u11 ? δ11u(p) : [δ11(w,p) for w in wv];   color=:darkorange, lw=2, ls=ls, label="")
    plot!(plt, wv, [δ20(w,p) for w in wv];                   color=:green3,     lw=2, ls=ls, label="")
end
data!(plt) = begin
    @plt_data!(d00, label=L"\delta_0^0", marker=(:circle,:dodgerblue,4), msc=:dodgerblue)
    @plt_data!(d11, label=L"\delta_1^1", marker=(:circle,:darkorange,4), msc=:darkorange)
    @plt_data!(d20, label=L"\delta_0^2", marker=(:circle,:green3,4), msc=:green3,
               xlab=L"\sqrt{s}\ \mathrm{[MeV]}", ylab=L"\delta_J^I\ (\pi\pi\!\to\!\pi\pi)\ \mathrm{[deg]}")
    hline!(plt, [0], color=:black, label=:none)
end

# ── Figure 1: the two deep basins ────────────────────────────────────────────
p1 = plot(size=(760,520), legend=:topleft, legendfontsize=8)
curves!(p1, P_FREE; ls=:solid, u11=false); curves!(p1, P_FIX; ls=:dash, u11=false); data!(p1)
plot!(p1, [NaN],[NaN]; color=:black, lw=2, ls=:solid, label="L6 free  (χ²="*string(round(chi2_8(P_FREE);digits=0))*")")
plot!(p1, [NaN],[NaN]; color=:black, lw=2, ls=:dash,  label="L6 fixed (χ²="*string(round(chi2_8(P_FIX);digits=0))*")")
title!(p1, "IAM "*L"\pi\pi"*" — find_deeper_minimum DEEP basins (non-physical): L6 free vs fixed"; titlefontsize=8)
savefig(p1, "iam_fdm_deep_basins.png")

# ── Figure 2: the physical reference fit ─────────────────────────────────────
p2 = plot(size=(760,520), legend=:topleft, legendfontsize=8)
curves!(p2, P_PHYS; ls=:solid, u11=true); data!(p2)
plot!(p2, [NaN],[NaN]; color=:black, lw=2, ls=:solid,
      label="physical fit, L6 fixed (χ²="*string(round(mphys.fval;digits=0))*")")
title!(p2, "IAM "*L"\pi\pi"*" — PHYSICAL fit (L6 fixed at 0.07×10⁻³, from paper LECs)"; titlefontsize=8)
savefig(p2, "iam_physical_fit.png")

@printf("saved iam_fdm_deep_basins.png  (free χ²=%.1f, fixed χ²=%.1f)\n", chi2_8(P_FREE), chi2_8(P_FIX))
@printf("saved iam_physical_fit.png     (physical L6-fixed χ²=%.1f, χ²/dof=%.2f)\n", mphys.fval, mphys.fval/(85-7))
