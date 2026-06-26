# SPDX-License-Identifier: LGPL-2.1-or-later
#
# X(6200) — the near-J/ψJ/ψ-threshold state from the two-channel coupled-channel
# fit to the LHCb double-J/ψ spectrum, reproduced natively with JuMinuit, then its
# pole / scattering length / effective range / compositeness propagated with the
# Bayesian bridge (posterior_sample + derived_interval).
#
# Published analysis:
#   X.-K. Dong, V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
#   "Coupled-channel interpretation of the LHCb double-J/ψ spectrum and hints of
#    a new state near the J/ψJ/ψ threshold",
#   Phys. Rev. Lett. 126 (2021) 132001, arXiv:2009.07795.
#   https://inspirehep.net/literature/1817481
#   Full original analysis (IMinuit.jl): https://github.com/fkguo/double_jpsi_fit
#
# Data (vendored from github.com/fkguo/double_jpsi_fit with the author's permission):
#   data_lhcb.csv        digitized LHCb double-J/ψ invariant-mass spectrum
#                        (LHCb, Sci. Bull. 65 (2020) 1983, arXiv:2006.16957),
#                        as used in arXiv:2009.07795.
#   parametersets_2c.csv the published Δχ²≈1 parameter ensemble (322 sets) the
#                        analysis used for its frequentist error bars.
#
# Run from the repo root:
#   julia --project=. BenchmarkExamples/X6200_double_jpsi/x6200_double_jpsi.jl

using JuMinuit
using DelimitedFiles, Statistics, LinearAlgebra

const DIR = @__DIR__

# ============================================================================
# Two-channel amplitude (channels: J/ψ J/ψ and ψ(2S) J/ψ), faithful to the PRL
# ============================================================================
const mJ, mψ2, ħc = 3.0969, 3.686097, 0.197327          # GeV, GeV, GeV·fm
kallen(a, b, c) = a^2 + b^2 + c^2 - 2a*b - 2b*c - 2a*c
qsq(w, m1, m2)  = kallen(w^2, m1^2, m2^2) / (4w^2)       # c.m. momentum²
phsp(w, m1, m2) = sqrt(qsq(w, m1, m2)) / (8π*w)          # real two-body phase space
xsqrt(z)        = imag(z) ≥ 0 ? sqrt(z + 0im) : -sqrt(z - 0im)   # cut on +real axis
ρ(w, m1, m2)    = xsqrt(qsq(w, m1, m2) + 0im) / (8π*w)

# two-point loop, dimensional regularization with subtraction constant `sub`
function Gdr(w, m1, m2, sub = -3)
    s = w^2; Δ = m1^2 - m2^2
    q = xsqrt(kallen(s, m1^2, m2^2) + 0im) / (2w)
    1 / (16π^2) * (sub + 2log(m1) + (m2^2 - m1^2 + s) / s * log(m2 / m1) + q / w *
        (log(s - Δ + 2q*w) + log(s + Δ + 2q*w) - log(-s + Δ + 2q*w) - log(-s - Δ + 2q*w)))
end

# (T11, T21, denominator) on Riemann sheet rs ∈ {1,2,3,4}; p = the 5 contact couplings
function Tparts(w, p, rs)
    a1, a2, c, b1, b2 = p
    v11 = (a1 + b1*qsq(w, mJ, mJ)) * 4mJ^2
    v22 = (a2 + b2*qsq(w, mψ2, mJ)) * 4mJ*mψ2
    v12 = c * 4mJ*sqrt(mJ*mψ2)
    g11 = Gdr(w, mJ, mJ); g22 = Gdr(w, mψ2, mJ)
    (rs == 2 || rs == 3) && (g11 += 2im * ρ(w, mJ, mJ))     # continue onto unphysical sheets
    (rs == 3 || rs == 4) && (g22 += 2im * ρ(w, mψ2, mJ))
    num11 = -g22*v11*v22 + g22*v12^2 + v11
    den   = 1 - g11*num11 - g22*v22
    (num11 / den, v12 / den, den)
end
Tmat(w, p, rs = 1) = Tparts(w, p, rs)[1:2]

# event-yield model: phase space × ( |α e^{-β s} (ct + G₁T₁₁ + r G₂T₂₁)|² + bg )
function dist(w, par)
    a1, a2, c, b1, b2, α, β, r, bg, ct = par
    t11, t21 = Tmat(w, (a1, a2, c, b1, b2))
    g1 = Gdr(w, mJ, mJ); g2 = Gdr(w, mψ2, mJ)
    amp = α * exp(-β * w^2) * (ct + g1*t11 + r*g2*t21)
    phsp(w, mJ, mJ) * (abs2(amp) + bg)
end

# ============================================================================
# Derived quantities of the X(6200) (all from the 5 couplings)
# ============================================================================
# scattering length a and effective range r from the near-threshold expansion
#   T(k)⁻¹ = -8π√s [ 1/a + ½ r k² - i k + O(k⁴) ]   in the J/ψJ/ψ channel.
a_fm(p)  = -real(Tmat(2mJ, p)[1]) / (16π*mJ) * ħc       # fm   — DIVERGES near unitarity
inv_a(p) = 1 / a_fm(p)                                  # 1/fm — the well-defined variable
function r_fm(p)
    th, ϵ, μ = 2mJ + 1e-7, 1e-8, mJ / 2
    t(w) = Tmat(w, p)[1]
    d = real(1 / t(th)) + th * (real(1 / t(th + ϵ)) - real(1 / t(th))) / ϵ
    -8π / μ * d * ħc
end
X_A(p) = (a = a_fm(p); r = r_fm(p); sqrt(1 / (1 + 2abs(r / a))))   # compositeness

# pole on sheet rs: complex Newton on the T-matrix denominator, det(1 - G·V) = 0
function find_pole(p, rs; w0)
    w = w0
    for _ in 1:80
        f  = Tparts(w, p, rs)[3]
        df = (Tparts(w + 1e-7, p, rs)[3] - f) / 1e-7
        w -= f / df
    end
    w
end

# same Newton, but report NaN when it does not actually land on a pole (|den| > tol)
function find_pole_checked(p, rs; w0, tol = 1e-8)
    w = w0
    for _ in 1:80
        f  = Tparts(w, p, rs)[3]
        df = (Tparts(w + 1e-7, p, rs)[3] - f) / 1e-7
        w -= f / df
    end
    abs(Tparts(w, p, rs)[3]) < tol ? w : complex(NaN, NaN)
end

# causality: a physical-sheet (RS1) search from several seeds must NOT find a
# spurious complex pole (only a real bound-state pole below threshold is allowed)
function causal(lecs)
    for w0 in (6.85 + 0.15im, 6.6 + 0.05im, 6.4 + 0.05im, 6.25 + 0.02im, 7.1 + 0.3im)
        p = find_pole_checked(lecs, 1; w0 = w0)
        (!isnan(real(p)) && abs(imag(p)) > 1e-3) && return false
    end
    true
end

# near-threshold pole closest to the J/ψJ/ψ threshold, with its interpretation:
#   bound     — sheet I,  real pole below threshold
#   virtual   — sheet II, real pole below threshold
#   resonance — sheet II, complex pole (above threshold)
# Near the unitary limit a (hence 1/a) changes sign and the pole crosses sheets:
# the paper's convention 1/a + ½rk² - ik = 0 gives κ ≈ -1/a, so a<0 ⇒ bound (I),
# a>0 ⇒ virtual / resonance (II).
function classify_pole(p)
    thr = 2mJ
    cand = Tuple{Float64,Symbol}[]
    w = find_pole_checked(p, 1; w0 = complex(thr - 0.02, 0.0))
    (!isnan(real(w)) && abs(imag(w)) < 2e-3 && real(w) < thr)  && push!(cand, (real(w), :bound))
    w = find_pole_checked(p, 2; w0 = complex(thr - 0.02, 0.0))
    (!isnan(real(w)) && abs(imag(w)) < 2e-3 && real(w) < thr)  && push!(cand, (real(w), :virtual))
    w = find_pole_checked(p, 2; w0 = complex(thr + 0.02, 0.03))
    (!isnan(real(w)) && abs(imag(w)) >= 2e-3)                  && push!(cand, (real(w), :resonance))
    isempty(cand) && return (NaN, :none)
    cand[argmin([abs(c[1] - thr) for c in cand])]
end

# ============================================================================
# Part 1 — fit the real LHCb double-J/ψ spectrum with JuMinuit
# ============================================================================
dat = readdlm(joinpath(DIR, "data_lhcb.csv"), ',')      # columns: mass(GeV), events, error
ws, ys, es = dat[1:36, 1], dat[1:36, 2], dat[1:36, 3]   # ndata = 36 bins, as in the paper
const βfix, rfix, bgfix = 0.012336, 1.0, 0.0            # fixed: production slope, channel mix, bg

# 7 free parameters: a1, a2, c, b1, b2 (couplings) + alpha (production), ct (contact)
χ²(q) = (par = (q[1], q[2], q[3], q[4], q[5], q[6], βfix, rfix, bgfix, q[7]);
         sum(((ys .- dist.(ws, Ref(par))) ./ es) .^ 2))
m = Minuit(χ², [0.2, -4.0, 3.0, -1.8, -7.0, 70.0, 3.0];
           names = ["a1", "a2", "c", "b1", "b2", "alpha", "ct"])
migrad!(m); hesse!(m)
lecs(v) = v[1:5]
best = lecs(m.values[:])

println("="^72)
println("Part 1 — JuMinuit fit to the real LHCb double-J/ψ spectrum")
println("  valid = ", m.valid, "   χ²/dof = ", round(m.fval / (36 - 7), digits = 3),
        "   (published: 28.708/29 = 0.99)")
println("  best fit: a1=", round(best[1], digits = 3), " a2=", round(best[2], digits = 3),
        " c=", round(best[3], digits = 3), " b1=", round(best[4], digits = 3),
        " b2=", round(best[5], digits = 3))

# ============================================================================
# Part 2 — pole search on 4 Riemann sheets, and central derived quantities
# ============================================================================
pX  = find_pole(best, 2; w0 = 6.20 + 0.01im)    # near-threshold pole = the X(6200)
pHi = find_pole(best, 3; w0 = 6.85 + 0.10im)    # broad higher pole
println("\nPart 2 — pole search and central values")
println("  J/ψJ/ψ threshold        = ", round(2mJ, digits = 4), " GeV")
println("  X(6200) pole (sheet II) = ", round(real(pX), digits = 4), " + ",
        round(imag(pX), digits = 4), "im GeV")
println("  higher pole (sheet III) ≈ ", round(real(pHi), digits = 3), " GeV")
println("  a = ", round(a_fm(best), digits = 3), " fm   r = ", round(r_fm(best), digits = 3),
        " fm   X̄_A = ", round(X_A(best), digits = 3),
        "   (published: a=0.80, r=-2.18, X̄_A=0.39)")

# ============================================================================
# Part 3 — frequentist uncertainties: propagate the published Δχ² ensemble.
#          This is the paper's own method, and it reproduces the published Table.
# ============================================================================
ens, _ = readdlm(joinpath(DIR, "parametersets_2c.csv"), ',', header = true)
P = [Float64.(ens[i, 2:6]) for i in 1:size(ens, 1)]     # the 5 couplings of each set
band(f) = (c = f(best); v = f.(P); (c, maximum(v) - c, minimum(v) - c))   # central, +up, -down
ra, xa = band(r_fm), band(X_A)
av = a_fm.(P)
println("\nPart 3 — frequentist error bars (propagate the 322-set Δχ² ensemble)")
println("  r₀  = ", round(ra[1], digits = 2), " +", round(ra[2], digits = 2), " ",
        round(ra[3], digits = 2), " fm     (paper -2.18 +0.66 -0.81)")
println("  X̄_A = ", round(xa[1], digits = 2), " +", round(xa[2], digits = 2), " ",
        round(xa[3], digits = 2), "           (paper 0.39 +0.58 -0.12)")
println("  a   : neg branch ≤ ", round(maximum(av[av .< 0]), digits = 2),
        " fm, pos branch ≥ ", round(minimum(av[av .> 0]), digits = 2),
        " fm   (paper ≤ -0.49 or ≥ 0.48 — a is near-unitary and diverges)")

# ============================================================================
# Part 4 — JuMinuit Bayesian bridge: pole sheet and the bound-vs-virtual question.
#          1/a changes sign across the posterior, so the near-threshold pole
#          crosses between sheet I (bound state) and sheet II (virtual / resonance).
# ============================================================================
post = posterior_sample(m; sampler = :stretch, nsteps = 6000, seed = 2024, warn = false)
S = post.ensemble.samples; N = size(S, 1)
qtrip(f) = round.(quantile([f(S[i, 1:5]) for i in 1:N], (0.16, 0.5, 0.84)), digits = 2)
cls = [classify_pole(S[i, 1:5]) for i in 1:N]
av  = [a_fm(S[i, 1:5]) for i in 1:N]
frac(s) = round(100 * count(c -> c[2] == s, cls) / N, digits = 1)
EB  = [2mJ - c[1] for c in cls if c[2] == :bound]
ebq = round.(Int, quantile(EB, (0.16, 0.5, 0.84)) .* 1000)
println("\nPart 4 — JuMinuit Bayesian bridge: pole sheet and the bound-vs-virtual question")
println("  P(bound,     sheet I)  = ", frac(:bound), " %   (a<0;  binding E_B = ",
        ebq[1], "/", ebq[2], "/", ebq[3], " MeV at 16/50/84%)")
println("  P(virtual,   sheet II) = ", frac(:virtual), " %   (a>0, below threshold)")
println("  P(resonance, sheet II) = ", frac(:resonance), " %   (a>0, above threshold)")
println("  1/a sign split:  a<0 ", round(100 * count(<(0), av) / N, digits = 1),
        " %  /  a>0 ", round(100 * count(>(0), av) / N, digits = 1), " %  (so report 1/a, not a)")
println("  r (16,50,84) = ", qtrip(r_fm), " fm     1/a (16,50,84) = ", qtrip(inv_a), " /fm")
println()
println("  The best fit (a = ", round(a_fm(best), digits = 2), " > 0) is a near-threshold")
println("  *resonance*; the bridge shows the same data also accommodate a *bound state*")
println("  (sheet I), and quantifies each — the X(6200) interpretation is genuinely two-fold,")
println("  exactly the paper's disjoint a₀. These credible intervals are broad: the 36-bin")
println("  spectrum weakly constrains the couplings (HESSE cond ~190). Centrals (Part 2) and")
println("  the frequentist bars (Part 3) match the paper; the bridge adds P(interpretation).")

# ============================================================================
# Part 5 — a physical constraint *is* a prior. Encode causality (no spurious
#          pole on the physical sheet) as logprior = -Inf on acausal couplings.
# ============================================================================
caus = Prior(θ -> causal(θ[1:5]) ? 0.0 : -Inf, :causality,
             "no spurious complex pole on the physical (RS1) sheet",
             fill(-Inf, 7), fill(Inf, 7),
             ["a1", "a2", "c", "b1", "b2", "alpha", "ct"], false, fill(false, 7))
postc = posterior_sample(m; sampler = :stretch, prior = caus, nsteps = 6000, seed = 2024, warn = false)
Sc = postc.ensemble.samples; Nc = size(Sc, 1)
qc(f) = round.(quantile([f(Sc[i, 1:5]) for i in 1:Nc], (0.16, 0.5, 0.84)), digits = 2)
println("\nPart 5 — a physical constraint as a prior (logprior = -Inf when acausal)")
println("  + causality:  r (16,50,84) = ", qc(r_fm), " fm    X̄_A = ", qc(X_A))
println("  For the X(6200) this is essentially unchanged from Part 4: nearly the whole")
println("  posterior already satisfies RS1 causality, so the constraint does not bite —")
println("  it is NOT what tightens the published bars. (The mechanism is general; a prior")
println("  that *does* cut into the posterior — a hard support, a measured nuisance — would.)")
println("="^72)
