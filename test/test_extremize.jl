# SPDX-License-Identifier: LGPL-2.1-or-later
using JuMinuit
using Test
using LinearAlgebra
using Logging

# extremize / profile_band — Δχ²-region extremization of derived quantities
# and the pointwise profile-likelihood band. The validation targets:
#
#   1. ANALYTIC (projection theorem): on a linear-Gaussian fit the interval
#      of f = cᵀθ over {χ² ≤ χ²_min + Δχ²} is exactly f̂ ± √(Δχ²·cᵀCc) and
#      the tangency point is θ̂ ± √Δχ²·Cc/σ_f — digit-level checks.
#   2. MULTI-CORRIDOR REGRESSION: a single-seed extremization stops at a
#      local tangency and silently under-covers when the region is two
#      disconnected low-χ² corridors; a seed pool finds the far corridor and
#      the per-seed diagnostics expose the difference. (The real-world
#      under-extremization incident this API is built to prevent.)
#   3. MINOS CONSISTENCY: for f(θ) = θᵢ the construction IS MINOS — the
#      intervals must agree (each method has its own root/stop tolerance).

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Linear model y = a + b·x on 10 points, σ = 1, with FIXED "noise" so the
# test is fully deterministic. Analytic: θ̂ = C Xᵀy, C = (XᵀX)⁻¹.
const _ex_x = collect(0.0:9.0)
const _ex_y = 1.0 .+ 2.0 .* _ex_x .+
              [0.3, -0.5, 0.1, 0.7, -0.2, -0.6, 0.4, 0.0, -0.3, 0.2]
_ex_chi2(θ) = sum(((_ex_y[i] - θ[1] - θ[2] * _ex_x[i]))^2 for i in eachindex(_ex_x))
const _ex_X = hcat(ones(length(_ex_x)), _ex_x)
const _ex_C = inv(_ex_X' * _ex_X)
const _ex_θ̂ = _ex_C * (_ex_X' * _ex_y)

function _ex_linear_fit()
    m = Minuit(_ex_chi2, [0.0, 0.0]; errors = [0.1, 0.1])
    migrad!(m)
    hesse!(m)
    return m
end

# Two degenerate low-χ² corridors at θ₁ ≈ ±1 (both at χ² = 0), separated by
# a barrier of height K. Δχ² ≤ 1 cuts each corridor at θ₁² = 1 ∓ 1/√K (the
# θ₂ degree of freedom relaxes to θ₂ = θ₁ at the extremum), so the exact
# extremes of f = θ₁ are ±√(1 + 1/√K) (global) and the +corridor alone
# spans [√(1 − 1/√K), √(1 + 1/√K)].
const _ex_K = 200.0
_ex_chi2c(θ) = _ex_K * (θ[1]^2 - 1)^2 + (θ[2] - θ[1])^2
const _ex_edge_in = sqrt(1 - 1 / sqrt(_ex_K))
const _ex_edge_out = sqrt(1 + 1 / sqrt(_ex_K))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytic linear-Gaussian target (projection theorem)
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize — analytic linear target (projection theorem)" begin
    m = _ex_linear_fit()
    @test m.valid
    @test collect(m.values) ≈ _ex_θ̂ atol = 1e-8

    # f = a + b·x₀ at an extrapolation point (x₀ outside the data — the
    # quantity is neither a fit parameter nor a fitted observation), at a
    # data-range point, and AT x₀ = 0 (where f ≡ the parameter a, so the
    # interval must equal a's 1σ parameter error). cl spans the nσ and the
    # probability conventions.
    for x0 in (15.0, -3.0, 0.0), cl in (1, 2, 0.95)
        c = [1.0, x0]
        σf = sqrt(c' * _ex_C * c)
        f̂ = c' * _ex_θ̂
        δ = delta_chisq(cl, 1)
        hw = sqrt(δ) * σf                       # analytic half-width
        r = extremize(m, θ -> θ[1] + θ[2] * x0; cl = cl)

        # Projection theorem, relative to the half-width (observed accuracy
        # ~2e-5; the tolerance leaves a ~10× margin).
        @test r.lo ≈ f̂ - hw atol = 3e-4 * hw
        @test r.hi ≈ f̂ + hw atol = 3e-4 * hw

        # The endpoints are attained at FEASIBLE points on the boundary,
        # and the bound is the documented one. The fcn_* diagnostics carry
        # the same values (so feasibility is checkable, not trusted).
        @test r.bound ≈ m.fval + δ * m.up rtol = 1e-12
        @test _ex_chi2(r.plo) <= r.bound + 1e-9
        @test _ex_chi2(r.phi) <= r.bound + 1e-9
        @test r.diagnostics.fcn_min ≈ _ex_chi2(r.plo) rtol = 1e-12
        @test r.diagnostics.fcn_max ≈ _ex_chi2(r.phi) rtol = 1e-12

        # Tangency points (the Lagrange solution): θ̂ ± √δ·Cc/σ_f. NB f is
        # STATIONARY along the boundary at the tangency, so a position
        # error ε costs only O(ε²) in f — the point converges as √(f-tol)
        # and is checked an order looser than the value.
        @test r.phi ≈ _ex_θ̂ .+ sqrt(δ) .* (_ex_C * c) ./ σf atol = 1e-2
        @test r.plo ≈ _ex_θ̂ .- sqrt(δ) .* (_ex_C * c) ./ σf atol = 1e-2

        # Construction property and bookkeeping.
        @test r.fbest ≈ f̂ atol = 1e-8
        @test r.lo <= r.fbest <= r.hi
        @test r.delta ≈ δ
        @test r.up == 1.0
    end

    # cl as a probability ≈ the matching nσ (0.6827 ≈ 1σ).
    f15 = θ -> θ[1] + θ[2] * 15.0
    r1 = extremize(m, f15)
    rp = extremize(m, f15; cl = 0.6827)
    @test rp.hi ≈ r1.hi rtol = 1e-4
    @test rp.lo ≈ r1.lo rtol = 1e-4

    # Explicit `delta` override ≡ the equivalent cl; cl is recorded as NaN.
    r2 = extremize(m, f15; cl = 2)
    rd = extremize(m, f15; delta = delta_chisq(2, 1))
    @test rd.hi ≈ r2.hi rtol = 1e-6
    @test rd.lo ≈ r2.lo rtol = 1e-6
    @test isnan(rd.cl) && rd.delta == r2.delta

    # −lnL parity: NLL = χ²/2 with up = 0.5 is the same statistical problem
    # — the interval must match the χ² fit's (the penalty excess and the
    # bound are both up-normalized).
    nll(θ) = _ex_chi2(θ) / 2
    mn = Minuit(nll, [0.0, 0.0]; errors = [0.1, 0.1], errordef = 0.5)
    migrad!(mn)
    rn = extremize(mn, f15)
    @test rn.up == 0.5
    @test rn.lo ≈ r1.lo rtol = 1e-6
    @test rn.hi ≈ r1.hi rtol = 1e-6

    # winner == 0 has two readings, disambiguated by naccepted: here f's
    # unconstrained MINIMUM sits exactly at the best fit, so the min side's
    # winner is legitimately the best-fit value WITH accepted fits (healthy),
    # while the max side is a regular boundary tangency.
    fsq = θ -> (θ[1] - _ex_θ̂[1])^2 + (θ[2] - _ex_θ̂[2])^2
    rsq = extremize(m, fsq)
    @test rsq.lo ≈ 0.0 atol = 1e-10
    @test rsq.diagnostics.winner_min == 0
    @test rsq.diagnostics.naccepted_min > 0     # NOT a failure
    @test rsq.diagnostics.winner_max == 1
    @test occursin("genuinely extremal", sprint(show, MIME"text/plain"(), rsq))

    # An mcmc_sample ensemble is a ready-made seed bank (the cross-feature
    # interlock): its f-extreme members — and the ensemble itself, which
    # iterates as a collection of full parameter vectors — feed `seeds`.
    ens = mcmc_sample(m; nsteps = 600, burn = 100, thin = 50, seed = 7)
    ext = sort(collect(ens); by = f15)
    r_ens = extremize(m, f15; seeds = [ext[1], ext[end]])
    @test r_ens.lo ≈ r1.lo rtol = 1e-4          # unimodal target: same answer
    @test r_ens.hi ≈ r1.hi rtol = 1e-4
    @test length(r_ens.diagnostics.min) == 3    # best fit + the two extremes
    r_all = extremize(m, f15; seeds = ens)      # whole ensemble as the pool
    @test r_all.lo ≈ r1.lo rtol = 1e-4

    # The input fit is not mutated (extremize talks to the RAW FCN and its
    # own Minuit clones only).
    vals = collect(m.values)
    nc = ncalls(m.fcn)
    extremize(m, f15)
    @test collect(m.values) == vals
    @test ncalls(m.fcn) == nc
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-corridor §-regression: seed coverage is load-bearing
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize — two-corridor under-extremization regression" begin
    mc = Minuit(_ex_chi2c, [0.9, 0.9]; errors = [0.05, 0.1])
    migrad!(mc)
    @test mc.valid && mc.values[1] > 0          # converged into the +corridor
    fc(θ) = θ[1]

    # Single (default, best-fit) seed: the penalty fit cannot cross the
    # χ² ≈ K barrier — it stops at the +corridor's INNER edge and silently
    # reports a too-narrow lower endpoint. This is the documented failure
    # mode, and it must be VISIBLE in the result, not hidden.
    r1 = extremize(mc, fc)
    @test r1.lo ≈ _ex_edge_in atol = 1e-3       # local tangency only…
    @test r1.lo > 0                             # …nowhere near the far corridor
    @test r1.hi ≈ _ex_edge_out atol = 1e-3
    @test r1.diagnostics.winner_min == 1

    # A seed in the far corridor recovers the true global extremum, and the
    # per-seed records expose exactly what each seed found (the audit trail
    # that distinguishes "covered" from "lucky").
    r2 = extremize(mc, fc; seeds = [[-0.9, -0.9]])
    @test r2.lo ≈ -_ex_edge_out atol = 1e-3     # found the far corridor
    @test r2.hi ≈ r1.hi atol = 1e-6             # +side unchanged
    d = r2.diagnostics
    @test d.winner_min == 2                     # the POOL seed won the min side
    @test length(d.min) == 2 && d.naccepted_min == 2
    @test d.min[1].seed == 1 && d.min[1].accepted && d.min[1].f > 0.9
    @test d.min[2].seed == 2 && d.min[2].accepted && d.min[2].f < -1.0
    @test all(rec.nfcn > 0 for rec in d.min)

    # Matrix seed pool (rows = seeds, e.g. ensemble members) ≡ vector form.
    r2m = extremize(mc, fc; seeds = [-0.9 -0.9])
    @test r2m.lo == r2.lo
    # Single-vector convenience form ≡ one-seed pool.
    r2v = extremize(mc, fc; seeds = [-0.9, -0.9])
    @test r2v.lo == r2.lo

    # LIMITS are honoured: a lower bound at −1.01 cuts the far corridor's
    # extremum (analytically at −√(1+1/√K) ≈ −1.0348) off at the bound.
    ml = Minuit(_ex_chi2c, [0.9, 0.9]; errors = [0.05, 0.1],
                limits = [(-1.01, 2.0), nothing])
    migrad!(ml)
    rl = extremize(ml, fc; seeds = [[-0.9, -0.9]])
    @test rl.plo[1] >= -1.01 - 1e-9             # never leaves the bound
    @test rl.lo ≈ -1.01 atol = 1e-4             # extremum AT the bound
    @test _ex_chi2c(rl.plo) <= rl.bound + 1e-9

    # FIXED parameters stay pinned (the region is defined with them fixed;
    # NB the fixed fit has its own, higher χ²_min, so its interval is NOT
    # nested in the free fit's — only the pinning is asserted).
    mf = Minuit(_ex_chi2c, [0.9, 0.9]; errors = [0.05, 0.1],
                fixed = [false, true])
    migrad!(mf)
    rf = extremize(mf, fc; seeds = [[-0.9, 0.9]])
    @test rf.plo[2] == 0.9 && rf.phi[2] == 0.9
    @test _ex_chi2c(rf.plo) <= rf.bound + 1e-9
    @test rf.hi ≈ _ex_edge_out atol = 5e-3      # ≈ free edge (small θ₂ cost)

    # The boundary pull-back is strictly LOCAL (unit-level check on the same
    # corridor geometry, bound = 1, anchor = the +corridor best fit (1,1)):
    # • From a point just outside the far corridor's OUTER edge the segment
    #   toward the anchor re-enters that corridor immediately — the pull-back
    #   lands on the corridor's own boundary (feasible, fcn ≤ bound).
    # • From a point just outside the far corridor's INNER (anchor-facing)
    #   edge the segment first climbs the χ² ≈ K barrier — there is no local
    #   feasible step, and the pull-back must REFUSE (return nothing) rather
    #   than bisect across the barrier and hand the candidate back to the
    #   anchor's corridor.
    anchor = [1.0, 1.0]
    pout = fill(-(_ex_edge_out + 1e-3), 2)        # outer edge, slightly outside
    proj = JuMinuit._project_to_bound(_ex_chi2c, pout, anchor, 1.0)
    @test proj !== nothing
    @test proj.fcn <= 1.0
    @test 1.0 - proj.fcn < 1e-3                   # ON the boundary, not interior
    @test proj.θ[1] ≈ -_ex_edge_out atol = 2e-3   # stayed on the FAR corridor
    @test _ex_chi2c(proj.θ) == proj.fcn
    pin = fill(-(_ex_edge_in - 1e-3), 2)          # inner edge, slightly outside
    @test _ex_chi2c(pin) > 1.0                    # (sanity: genuinely infeasible)
    @test JuMinuit._project_to_bound(_ex_chi2c, pin, anchor, 1.0) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. MINOS consistency: f(θ) = θᵢ reproduces the MINOS interval
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize — MINOS consistency on the Rosenbrock valley" begin
    ros(θ) = (1 - θ[1])^2 + 100 * (θ[2] - θ[1]^2)^2
    mr = Minuit(ros, [0.5, 0.5]; errors = [0.1, 0.1])
    migrad!(mr)
    hesse!(mr)
    @test mr.valid
    minos!(mr)

    # x: the profile χ² is EXACTLY (1−x)² (y compensates as y = x²), so the
    # Δχ²=1 interval is [0, 2] up to the anchor offset χ²_min ≈ 0. extremize
    # must hit the analytic answer; MINOS agrees within its own cross-search
    # tolerance (~1e-2·σ here — extremize is the tighter of the two).
    rx = extremize(mr, θ -> θ[1])
    @test rx.lo ≈ 0.0 atol = 1e-3
    @test rx.hi ≈ 2.0 atol = 1e-3
    ex = mr.merrors["x0"]
    @test rx.lo ≈ mr.values[1] + ex.lower atol = 1e-2
    @test rx.hi ≈ mr.values[1] + ex.upper atol = 1e-2

    # y: genuinely asymmetric MINOS errors (the test has teeth), agreement
    # at the cross-search tolerance.
    ry = extremize(mr, θ -> θ[2])
    ey = mr.merrors["x1"]
    @test abs(ey.upper) > 2 * abs(ey.lower)     # clear asymmetry (ratio ≈ 2.9)
    @test ry.lo ≈ mr.values[2] + ey.lower atol = 2e-3
    @test ry.hi ≈ mr.values[2] + ey.upper atol = 2e-3

    # cl = 2 ⇒ Δχ² = 4 ⇒ x ∈ [−1, 3] analytically; minos!(…; sigma = 2)
    # must agree. The +side is a LONG curved-valley traverse (x: 1 → 3 with
    # y tracking x²) — the case the warm-restarted penalty rounds exist for.
    mr2 = Minuit(ros, [0.5, 0.5]; errors = [0.1, 0.1])
    migrad!(mr2)
    minos!(mr2; sigma = 2)
    ex2 = mr2.merrors["x0"]
    rx2 = extremize(mr, θ -> θ[1]; cl = 2)
    @test rx2.delta ≈ 4.0 atol = 1e-9
    @test rx2.lo ≈ -1.0 atol = 1e-3
    @test rx2.hi ≈ 3.0 atol = 1e-3
    @test rx2.lo ≈ mr2.values[1] + ex2.lower atol = 1e-2
    @test rx2.hi ≈ mr2.values[1] + ex2.upper atol = 1e-2
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. profile_band — analytic band on a correlated 3-parameter Gaussian
# ─────────────────────────────────────────────────────────────────────────────

@testset "profile_band — analytic band, warm sweep, clamping" begin
    # χ² = (θ−μ)ᵀA(θ−μ) with strong correlations ⇒ C = A⁻¹ (up = 1), and
    # for the curve family f(x, θ) = θ₁ + θ₂x + θ₃x² the pointwise band is
    # analytic: c(x) = (1, x, x²), band(x) = c·μ ± √(c'Cc).
    A = [2.0 0.8 0.3; 0.8 1.5 0.5; 0.3 0.5 1.0]
    μ = [0.5, -1.0, 2.0]
    chi3(θ) = (θ .- μ)' * A * (θ .- μ)
    m3 = Minuit(chi3, [0.0, 0.0, 0.0]; errors = fill(0.1, 3))
    migrad!(m3)
    hesse!(m3)
    C3 = inv(A)
    fcurve(x, θ) = θ[1] + θ[2] * x + θ[3] * x^2
    xs = range(-1.0, 2.0; length = 9)

    band = profile_band(m3, fcurve, xs)
    @test band isa ProfileBand
    @test band.x == collect(Float64, xs)
    @test band.nfail == 0
    @test length(band.lo) == length(xs) == length(band.diagnostics)
    for (i, x) in enumerate(xs)
        c = [1.0, x, x^2]
        σf = sqrt(c' * C3 * c)
        f̂ = c' * μ
        # pointwise projection theorem (observed accuracy ~3e-4·hw)
        @test band.lo[i] ≈ f̂ - σf atol = 2e-3 * σf
        @test band.hi[i] ≈ f̂ + σf atol = 2e-3 * σf
        # the band contains the best-fit curve by construction
        @test band.lo[i] <= band.fbest[i] <= band.hi[i]
        @test band.fbest[i] ≈ f̂ atol = 1e-6
        # extremal parameter vectors are feasible
        @test chi3(band.plo[i]) <= band.bound + 1e-9
        @test chi3(band.phi[i]) <= band.bound + 1e-9
        # per-point diagnostics: every group accepted, stored-edge FCN
        # values certify feasibility
        di = band.diagnostics[i]
        @test di.x == band.x[i]
        @test !di.failed_lo && !di.failed_hi
        @test di.accepted_lo > 0 && di.accepted_hi > 0
        @test di.nfits_lo >= di.accepted_lo
        @test di.fcn_lo ≈ chi3(band.plo[i]) rtol = 1e-12
        @test di.fcn_hi ≈ chi3(band.phi[i]) rtol = 1e-12
        @test di.fcn_lo <= band.bound + 1e-9
    end

    # Sweep variants must agree with the default on this smooth problem.
    b1 = profile_band(m3, fcurve, xs; passes = 1, warm = false)
    @test maximum(abs.(b1.lo .- band.lo)) < 2e-3
    @test maximum(abs.(b1.hi .- band.hi)) < 2e-3
    brev = profile_band(m3, fcurve, reverse(collect(xs)))
    @test brev.lo ≈ reverse(band.lo) atol = 2e-3
    bnc = profile_band(m3, fcurve, xs; include_best = false)
    @test bnc.nfail == 0 && !any(isnan, bnc.lo) && !any(isnan, bnc.hi)
    @test maximum(abs.(bnc.lo .- band.lo)) < 2e-3

    # A cl = 2 band is √4/√1 = 2× wider (linear-Gaussian exactness).
    b2 = profile_band(m3, fcurve, xs; cl = 2)
    for (i, x) in enumerate(xs)
        c = [1.0, x, x^2]
        σf = sqrt(c' * C3 * c)
        @test b2.hi[i] - b2.lo[i] ≈ 4 * σf atol = 8e-3 * σf
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Failure accounting, robustness, validation, display
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize/profile_band — failure path, robustness, validation" begin
    m = _ex_linear_fit()
    f15 = θ -> θ[1] + θ[2] * 15.0

    # A vanishing penalty weight makes every penalty optimum run far outside
    # the region (the boundary overshoot scales as 1/λ), so every candidate
    # fails the acceptance gate DETERMINISTICALLY and the machinery must
    # take the documented failure path: warn, fall back to the best-fit
    # value, report winner 0 and naccepted 0.
    rfail = @test_logs (:warn, r"no penalty fit was accepted on the min side") (:warn, r"no penalty fit was accepted on the max side") match_mode = :any extremize(m, f15; lambda = 1e-12)
    @test rfail.lo == rfail.hi == rfail.fbest
    @test rfail.diagnostics.winner_min == 0 && rfail.diagnostics.winner_max == 0
    @test rfail.diagnostics.naccepted_min == 0 == rfail.diagnostics.naccepted_max
    @test occursin("side FAILED", sprint(show, MIME"text/plain"(), rfail))

    # Same trigger on the band: with include_best = false the failed edges
    # are NaN and every (point, side, pass) group is counted in nfail.
    xs5 = range(0.0, 4.0; length = 5)
    fb(x, θ) = θ[1] + θ[2] * x
    bfail = @test_logs (:warn, r"extremization group") match_mode = :any profile_band(m, fb, xs5; lambda = 1e-12, include_best = false, passes = 2)
    @test bfail.nfail == 2 * 2 * length(xs5)          # sides × passes × points
    @test all(isnan, bfail.lo) && all(isnan, bfail.hi)
    @test all(p === nothing for p in bfail.plo)
    @test all(d.failed_lo && d.failed_hi for d in bfail.diagnostics)
    # …and with the default clamp the band degrades to the best-fit curve
    # (finite, still flagged as failed in the diagnostics).
    bclamp = @test_logs (:warn, r"extremization group") match_mode = :any profile_band(m, fb, xs5; lambda = 1e-12)
    @test bclamp.lo ≈ bclamp.fbest atol = 1e-12
    @test bclamp.hi ≈ bclamp.fbest atol = 1e-12
    @test bclamp.nfail > 0

    # An FCN with a hard domain wall INSIDE the Δχ² range: probing
    # θ₁ < 1.93 throws, while the unconstrained Δχ² = 1 crossing would be
    # at θ₁ = 1.9 (σ = 0.1). The throw-guard must map those probes to
    # infeasible — not abort — so the min endpoint stops AT the wall: the
    # FCN's domain edge acts as a hard constraint. The free (+) side is
    # untouched and lands on the analytic 2 + σ.
    wallchi2(θ) = θ[1] < 1.93 ? throw(DomainError(θ[1], "domain wall")) :
                  (θ[1] - 2.0)^2 / 0.01 + θ[2]^2
    mg = Minuit(wallchi2, [2.0, 0.0]; errors = [0.01, 0.1])
    migrad!(mg)
    @test mg.valid
    rg = extremize(mg, θ -> θ[1])
    @test rg.diagnostics.naccepted_min > 0
    @test 1.93 - 1e-9 <= rg.plo[1] <= 1.95      # stopped at/inside the wall
    @test 1.93 - 1e-9 <= rg.lo <= 1.95
    @test rg.hi ≈ 2.1 atol = 5e-3               # analytic 2 + √(Δχ²)·σ
    @test rg.lo <= rg.fbest <= rg.hi

    # Argument validation.
    munfit = Minuit(_ex_chi2, [0.0, 0.0]; errors = [0.1, 0.1])
    @test_throws ArgumentError extremize(munfit, f15)            # no fit yet
    # All-fixed: migrad! itself refuses such a fit, so extremize can only
    # ever see it unfitted — still a clear ArgumentError, not a deep crash.
    mallfix = Minuit(_ex_chi2, [1.0, 2.0]; errors = [0.1, 0.1],
                     fixed = [true, true])
    @test_throws ArgumentError extremize(mallfix, f15)
    @test_throws ArgumentError extremize(m, f15; lambda = 0.0)
    @test_throws ArgumentError extremize(m, f15; accept_tol = -1.0)
    @test_throws ArgumentError extremize(m, f15; rounds = 0)
    @test_throws ArgumentError extremize(m, f15; delta = 0.0)
    # Non-finite control values break the bounded-region / finite-penalty
    # semantics and must be rejected up front.
    @test_throws ArgumentError extremize(m, f15; lambda = Inf)
    @test_throws ArgumentError extremize(m, f15; accept_tol = NaN)
    @test_throws ArgumentError extremize(m, f15; delta = Inf)
    @test_throws ArgumentError extremize(m, f15; delta = NaN)
    @test_throws ArgumentError extremize(m, f15; seeds = [[1.0]])       # length
    @test_throws ArgumentError extremize(m, f15; seeds = [[NaN, 1.0]])  # finite
    @test_throws ArgumentError profile_band(m, fb, Float64[])           # empty
    @test_throws ArgumentError profile_band(m, fb, xs5; passes = 0)
    @test_throws ArgumentError profile_band(m, fb, xs5; rounds = 0)

    # An invalid input fit is allowed but loudly flagged (the region anchor
    # m.fval may not be the true minimum).
    minv = Minuit(_ex_chi2, [0.0, 0.0]; errors = [0.1, 0.1])
    migrad!(minv; maxfcn = 3)
    @test !minv.valid
    @test_logs (:warn, r"NOT valid") match_mode = :any extremize(minv, f15)

    # Display forms (compact + MIME) render and carry the key facts.
    r = extremize(m, f15)
    @test occursin("ExtremizeResult", repr(r))
    plain = sprint(show, MIME"text/plain"(), r)
    @test occursin("extremize [full]: f ∈ [", plain)
    @test occursin("winner seed 1", plain)
    band = profile_band(m, fb, xs5)
    @test occursin("ProfileBand(5 points)", repr(band))
    bplain = sprint(show, MIME"text/plain"(), band)
    @test occursin("pointwise profile envelope", bplain)
    @test occursin("group failures: 0", bplain)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Expensive-FCN features (f1(1420) field report): mode=:directional, the
#    f-failure contract (P4), iterate pass-through, and the on_unit hook (P5).
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize — mode=:directional (fast linear-direction crossing)" begin
    m = _ex_linear_fit()
    # Exact in the linear-Gaussian limit: directional must reproduce the
    # projection-theorem interval (and the :full result) to high accuracy, at a
    # tiny fraction of the FCN cost.
    ncall = Ref(0)
    counting(θ) = (ncall[] += 1; _ex_chi2(θ))
    mc = Minuit(counting, [0.0, 0.0]; errors = [0.1, 0.1]); migrad!(mc); hesse!(mc)
    for x0 in (15.0, -3.0, 0.0), cl in (1, 2)
        c = [1.0, x0]
        σf = sqrt(c' * _ex_C * c); f̂ = c' * _ex_θ̂; hw = sqrt(delta_chisq(cl, 1)) * σf
        fcall = Ref(0)
        fx(θ) = (fcall[] += 1; θ[1] + θ[2] * x0)
        ncall[] = 0
        r = extremize(mc, fx; cl = cl, mode = :directional)
        @test r.mode === :directional
        @test r.lo ≈ f̂ - hw atol = 1e-3 * hw
        @test r.hi ≈ f̂ + hw atol = 1e-3 * hw
        @test r.lo <= r.fbest <= r.hi
        # endpoints feasible on the boundary
        @test _ex_chi2(r.plo) <= r.bound + 1e-6
        @test _ex_chi2(r.phi) <= r.bound + 1e-6
        # cost ceiling: ≤ n_free + ~30 of each of FCN and f (field report budget)
        @test ncall[] <= mc.npar + 60
        @test fcall[] <= mc.npar + 30
        # directional diagnostics present
        @test haskey(r.diagnostics, :gCg) && r.diagnostics.gCg > 0
        @test haskey(r.diagnostics, :alpha_lo) && haskey(r.diagnostics, :alpha_hi)
    end

    # user-supplied grad_f reproduces the numerical-gradient result
    rnum = extremize(mc, θ -> θ[1] + θ[2] * 15.0; mode = :directional)
    rana = extremize(mc, θ -> θ[1] + θ[2] * 15.0; mode = :directional,
                     grad_f = θ -> [1.0, 15.0])
    @test rana.lo ≈ rnum.lo atol = 1e-6
    @test rana.hi ≈ rnum.hi atol = 1e-6

    # :full and :directional agree on this (linear) target
    rfull = extremize(mc, θ -> θ[1] + θ[2] * 15.0)
    @test rana.lo ≈ rfull.lo atol = 1e-3 * (rfull.hi - rfull.lo)
    @test rana.hi ≈ rfull.hi atol = 1e-3 * (rfull.hi - rfull.lo)

    # seeds are ignored (with a warning) in directional mode
    @test_logs (:warn, r"seeds.* ignored") match_mode = :any extremize(
        mc, θ -> θ[1]; mode = :directional, seeds = [[0.9, 1.9]])

    # a flat-along-C direction (∇fᵀC∇f = 0) is rejected, not silently wrong
    @test_throws ArgumentError extremize(mc, θ -> 1.0; mode = :directional)
    @test_throws ArgumentError extremize(mc, θ -> θ[1]; mode = :bogus)

    # non-finite f at the boundary crossings (finite only in a tiny ball around
    # θ̂, so the gradient still succeeds) collapses both sides onto the best fit
    # — but DETECTABLY: a warning fires and the f_failed_* flags are set, so
    # lo == hi == fbest is never mistaken for a genuinely tight interval. A
    # throwing f at the crossing is equally safe (no crash).
    mthat = collect(mc.values)
    fcross(θ) = norm(θ .- mthat) < 1e-3 ? (θ[1] + θ[2] * 15.0) : NaN
    rcol = @test_logs (:warn, r"non-finite at the") match_mode = :any extremize(
        mc, fcross; mode = :directional)
    @test rcol.diagnostics.f_failed_lo && rcol.diagnostics.f_failed_hi
    @test rcol.lo == rcol.hi == rcol.fbest
    fthrow(θ) = norm(θ .- mthat) < 1e-3 ? (θ[1] + θ[2] * 15.0) : throw(DomainError(θ))
    @test Logging.with_logger(Logging.NullLogger()) do
        try; extremize(mc, fthrow; mode = :directional); true; catch; false; end
    end

    # a bounded free parameter ⇒ directional warns (it ignores limits)
    mb = Minuit(_ex_chi2, [0.0, 0.0]; errors = [0.1, 0.1], limits = [(-5.0, 5.0), nothing])
    migrad!(mb); hesse!(mb)
    @test_logs (:warn, r"bounded free parameters") match_mode = :any extremize(
        mb, θ -> θ[1] + θ[2] * 15.0; mode = :directional)
end

@testset "extremize — f-failure contract (P4): non-finite f is safe" begin
    m = _ex_linear_fit()
    that = collect(m.values)
    f15(θ) = θ[1] + θ[2] * 15.0
    rclean = extremize(m, f15)

    # f that returns NaN at ~30% of probes (but is finite at the anchor): the
    # call must neither error nor silently bias the endpoints, and the
    # rejections are tallied. A failing side may legitimately *warn*
    # (`naccepted == 0`) — that is the opposite of a silent bias — so we assert
    # no THROW (the safety contract), not no-warning. A deterministic failure
    # pattern keyed on θ keeps it stable. Logs are muted to keep the suite quiet.
    bad(θ) = (isapprox(θ, that; atol = 1e-9) ? f15(θ) :
              (hash(round.(θ; digits = 6)) % 10 < 3 ? NaN : f15(θ)))
    local r = nothing
    @test Logging.with_logger(Logging.NullLogger()) do
        try; r = extremize(m, bad); true; catch; false; end
    end
    @test isfinite(r.lo) && isfinite(r.hi)
    @test r.lo <= r.fbest <= r.hi               # brackets the best fit
    @test r.lo <= rclean.fbest <= r.hi          # brackets the true best-fit value
    # NOT collapsed to [fbest, fbest]: a non-finite-f region may narrow the
    # interval, but with only ~30% of probes failing it must retain a
    # substantial width (≥ 30% of the clean interval) — this is the assertion
    # that actually rules out the silent-centre-collapse failure mode.
    @test (r.hi - r.lo) >= 0.3 * (rclean.hi - rclean.lo)
    nnf = sum(rec.f_nonfinite for rec in vcat(r.diagnostics.min, r.diagnostics.max))
    @test nnf > 0                                # rejections recorded, not hidden
    @test all(haskey(rec, :f_nonfinite) for rec in r.diagnostics.min)

    # f that THROWS (a different failure mode) is equally safe (no throw escapes).
    thrower(θ) = (isapprox(θ, that; atol = 1e-9) ? f15(θ) :
                  (hash(round.(θ; digits = 6)) % 10 < 3 ? throw(DomainError(θ)) : f15(θ)))
    @test Logging.with_logger(Logging.NullLogger()) do
        try; extremize(m, thrower); true; catch; false; end
    end
end

@testset "extremize/profile_band — iterate pass-through + on_unit hook (P5)" begin
    m = _ex_linear_fit()
    f15(θ) = θ[1] + θ[2] * 15.0

    # iterate must be ≥ 1; iterate=1 forbids the per-MIGRAD retry and is the
    # cheapest setting (still a valid, best-fit-bracketing interval).
    @test_throws ArgumentError extremize(m, f15; iterate = 0)
    r1 = extremize(m, f15; rounds = 1, iterate = 1, strategy = 0)
    @test r1.lo <= r1.fbest <= r1.hi
    @test isfinite(r1.lo) && isfinite(r1.hi)

    # on_unit fires once per completed penalty-MIGRAD unit with the documented
    # record; count > 0 and every payload carries the unit key.
    units = NamedTuple[]
    extremize(m, f15; rounds = 1, on_unit = u -> push!(units, u))
    @test !isempty(units)
    for u in units
        @test u.side === :lower || u.side === :upper
        @test haskey(u, :seed) && haskey(u, :round) && haskey(u, :stage)
        @test haskey(u, :cur) && length(u.cur) == 2
        @test haskey(u, :nfcn) && haskey(u, :fcn) && haskey(u, :valid)
    end

    # A throwing on_unit callback must PROPAGATE — even when it throws one of
    # the domain-failure types the internal MIGRAD guard catches (DomainError ∈
    # _EXTREMIZE_CATCH). A swallowed checkpoint exception would be a silent
    # data-loss bug; the hook is deliberately fired outside that catch.
    @test_throws DomainError extremize(
        m, f15; rounds = 1, on_unit = u -> throw(DomainError(0.0, "checkpoint failed")))

    # profile_band forwards both; the band hook record additionally carries the
    # grid point.
    fb(x, θ) = θ[1] + θ[2] * x
    bunits = NamedTuple[]
    band = profile_band(m, fb, 0.0:2.0:8.0; passes = 1, iterate = 1,
                        on_unit = u -> push!(bunits, u))
    @test band.nfail == 0
    @test !isempty(bunits)
    @test all(haskey(u, :x) && haskey(u, :point) && haskey(u, :pass) for u in bunits)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Directional with a fixed parameter; ExtremizeResult back-compat ctor.
# ─────────────────────────────────────────────────────────────────────────────

@testset "extremize — directional fixed params + back-compat constructor" begin
    # A FIXED parameter must stay pinned and contribute nothing to the
    # direction (its gradient/Cg slot is zero); the interval still forms.
    mfix = Minuit(_ex_chi2, [0.0, 0.0]; errors = [0.1, 0.1], fixed = [false, true])
    migrad!(mfix); hesse!(mfix)
    rfix = extremize(mfix, θ -> θ[1] + θ[2] * 15.0; mode = :directional)
    @test rfix.diagnostics.grad[2] == 0.0                 # fixed slot zeroed
    @test rfix.plo[2] == mfix.values[2] == rfix.phi[2]    # fixed coord pinned
    @test rfix.lo <= rfix.fbest <= rfix.hi
    @test rfix.lo < rfix.hi                                # non-degenerate

    # Pre-0.5.3 positional construction (10 args, no `mode`) still works and
    # defaults to :full — the exported result type stays backward-compatible.
    diag0 = (min = NamedTuple[], max = NamedTuple[], winner_min = 0,
             winner_max = 0, naccepted_min = 0, naccepted_max = 0,
             fcn_min = NaN, fcn_max = NaN)
    rc = ExtremizeResult(1.0, 2.0, [0.0], [0.0], 1.5, 3.0, 1.0, 1.0, 1.0, diag0)
    @test rc.mode === :full
    @test rc.lo == 1.0 && rc.hi == 2.0 && rc.fbest == 1.5
end
