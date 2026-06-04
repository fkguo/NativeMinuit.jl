# SPDX-License-Identifier: LGPL-2.1-or-later
using JuMinuit
using Test
using Logging

# As of v0.4.0 ALL find_deeper_minimum overloads return a `Minuit` (check `.valid`)
# and route every fit through the high-level Minuit path, so parameter LIMITS and
# FIXED parameters are honoured throughout the search.

@testset "find_deeper_minimum — perturbation (escape a local basin)" begin
    # Double well in x[1] (minima ≈ ±1), tilted by +0.4·x[1] so the x[1]≈−1 well
    # is DEEPER (f ≈ −0.41) than the x[1]≈+1 well (f ≈ +0.39); x[2] is a simple
    # quadratic. A plain MIGRAD started on the +1 side stays in the shallow basin;
    # find_deeper_minimum must escape to the deep one.
    f(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + x[2]^2
    shallow = 0.4    # ≈ f at the +1 well

    m1 = find_deeper_minimum(f, [1.0, 0.5], [0.3, 0.3]; n_restarts = 80, perturb = 2.0, seed = 1)
    @test m1 isa Minuit
    @test m1.valid
    @test m1.values[1] < 0                        # escaped to the deeper (−1) well
    @test m1.fval < shallow - 0.4                 # and it is genuinely deeper

    # reproducible (same seed ⇒ same result)
    m2 = find_deeper_minimum(f, [1.0, 0.5], [0.3, 0.3]; n_restarts = 80, perturb = 2.0, seed = 1)
    @test collect(m2.values) ≈ collect(m1.values)

    # the AbstractCostFunction form agrees with the bare-callable form
    mc = find_deeper_minimum(CostFunction(f, 1.0), [1.0, 0.5], [0.3, 0.3];
                             n_restarts = 80, perturb = 2.0, seed = 1)
    @test collect(mc.values) ≈ collect(m1.values)

    # A throwing FCN must NOT abort the search: log(x[1]) throws for x[1] ≤ 0,
    # and the wide jitter will probe there — those restarts are skipped.
    g(x) = (log(x[1]))^2 + (x[1] - 2.0)^2 + x[2]^2
    mg = find_deeper_minimum(g, [2.0, 0.0], [0.5, 0.5]; n_restarts = 40, perturb = 1.0, seed = 3)
    @test mg.valid
    @test mg.values[1] > 0                        # converged in the valid domain

    # m::Minuit convenience overload agrees with the bare-callable form (same seed)
    mref = Minuit(f, [1.0, 0.5]; errors = [0.3, 0.3], strategy = 1)
    m_viaM = find_deeper_minimum(mref; n_restarts = 80, perturb = 2.0, seed = 1)
    @test m_viaM isa Minuit
    @test m_viaM.values[1] * m1.values[1] > 0     # same (deep) basin
    @test mref.fmin === nothing || mref.values[1] > 0   # input `mref` not mutated (still shallow/unfit)

    # argument validation (call the core directly with a fitted 1-param Minuit)
    mv = Minuit(x -> (x[1] - 1)^2, [0.0]; errors = [0.1]); migrad!(mv)
    @test_throws ArgumentError find_deeper_minimum(mv; n_restarts = 0)
    @test_throws ArgumentError find_deeper_minimum(mv; perturb = 0.0)
    @test_throws ArgumentError find_deeper_minimum(mv; max_rounds = 0)
    @test_throws ArgumentError find_deeper_minimum(mv; min_improvement = -1.0)

    # all-params-fixed ⇒ a clear ArgumentError (not an opaque low-level migrad error)
    maf = Minuit(f, [1.0, 0.5]; errors = [0.3, 0.3], fixed = [true, true])
    @test_throws ArgumentError find_deeper_minimum(maf; n_restarts = 4)

    # deprecated v0.3.1 name still forwards (depwarn visibility is flag-dependent)
    md = find_global_minimum(f, [1.0, 0.5], [0.3, 0.3]; n_restarts = 80, perturb = 2.0, seed = 1)
    @test collect(md.values) ≈ collect(m1.values)
end

@testset "find_deeper_minimum — perturbation honours LIMITS and FIXED" begin
    # ── Bounded: the deep well at x[1]≈−1 is OUTSIDE [0,2]; the search must stay
    # in bounds and therefore CANNOT reach it. ────────────────────────────────
    f(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + x[2]^2
    mb = Minuit(f, [1.0, 0.5]; errors = [0.3, 0.3], limits = [(0.0, 2.0), nothing], strategy = 1)
    rb = find_deeper_minimum(mb; n_restarts = 120, perturb = 3.0, seed = 1)
    @test rb.valid
    @test 0.0 <= rb.values[1] <= 2.0              # stayed in bounds (did NOT escape to −1)
    @test rb.limits[1] == (0.0, 2.0)              # the bound survived

    # ── Fixed: x[2] fixed at 2.0 while its unconstrained optimum is 3.0. If the
    # fix were dropped, an adoption MIGRAD would pull x[2] toward 3.0. ─────────
    h(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + (x[2] - 3.0)^2
    mf = Minuit(h, [1.0, 2.0]; errors = [0.3, 0.3], fixed = [false, true], strategy = 1)
    rf = find_deeper_minimum(mf; n_restarts = 80, perturb = 2.0, seed = 1)
    @test rf.valid
    @test rf.fixed[2] == true                     # fix preserved
    @test rf.values[2] == 2.0                     # fixed param never moved (would be ~3.0 if un-fixed)
    @test rf.values[1] < 0                        # the FREE param still escaped to the deep well

    # `name=` (singular, iminuit-style) accepted as an alias for `names=` on the
    # bare overloads (else it would leak to the core as a MethodError).
    mn = find_deeper_minimum(x -> (x[1]-1)^2 + (x[2]-2)^2, [0.0, 0.0], [0.3, 0.3];
                             name = ["a", "b"], n_restarts = 8, max_rounds = 2, seed = 1)
    @test mn isa Minuit
    @test collect(mn.parameters) == ["a", "b"]
end

@testset "find_deeper_minimum — resampling projects OOB candidates (FCN throws past the bound)" begin
    # The FCN is undefined for x[1] ≤ 0 — the very reason x[1] is bounded. A user
    # `refit` that returns an OUT-OF-BOUNDS candidate must NOT crash the search:
    # the discovery projection clamps it before find_solution_modes scores the FCN.
    fb(x) = (log(x[1]))^2 + (x[1] - 2.0)^2 + x[2]^2          # DomainError for x[1] ≤ 0
    m = Minuit(fb, [2.0, 0.0]; errors = [0.3, 0.3], limits = [(1e-6, nothing), nothing], strategy = 1)
    migrad!(m); hesse(m)
    data = collect(1.0:10.0)
    refit_oob = (sub, st) -> [-0.5, 0.01 * (sum(sub)/length(sub) - 5.5)]   # x[1] = −0.5 (out of bounds)
    m_out = with_logger(NullLogger()) do                     # must not throw a DomainError
        find_deeper_minimum(m, refit_oob, data; n_discovery = 6, seed = 1)
    end
    @test m_out isa Minuit
    @test m_out.values[1] >= 1e-6                             # stayed in bounds
end

@testset "find_deeper_minimum — resampling dispatches" begin
    # Single-basin fixture: chi2(p)=Σ(y_i-p[1])²/0.01, data all at y=2 ⇒ every
    # resample converges to p≈2 ⇒ suitability check fires.
    pts_sb = fill(2.0, 40)
    function chi2_sb(p, d = pts_sb)
        s = 0.0; for y in d; s += (y - p[1])^2 / 0.01; end; return s
    end
    refit_sb = (subdata, start) -> begin
        fm = migrad(CostFunction(p -> chi2_sb(p, subdata), 1.0), start, [0.1]; strategy = Strategy(1))
        JuMinuit.is_valid(fm) ? collect(Float64, values(fm)) : fill(NaN, length(start))
    end
    m_sb = Minuit(chi2_sb, [1.9]; errors = [0.1]); migrad!(m_sb); hesse(m_sb)

    @testset "suitability check — single-basin warns, returns a fitted clone" begin
        m_out = @test_logs((:warn, r"No deeper basin|no deeper basin"), min_level = Logging.Warn,
                           find_deeper_minimum(m_sb, refit_sb, pts_sb; n_discovery = 10, seed = 1))
        @test m_out isa Minuit
        @test m_out.fval ≈ m_sb.fval atol = 1e-6   # unchanged (no adoption)
        @test collect(m_out.values) ≈ collect(m_sb.values)
    end

    @testset "argument validation — resampling dispatch" begin
        @test_throws ArgumentError find_deeper_minimum(m_sb, refit_sb, pts_sb; n_discovery = 1)
        @test_throws ArgumentError find_deeper_minimum(m_sb, refit_sb, pts_sb; max_rounds = 0)
        @test_throws ArgumentError find_deeper_minimum(m_sb, refit_sb, pts_sb; min_improvement = -0.1)
        @test_throws ArgumentError find_deeper_minimum(m_sb, refit_sb, [2.0]; n_discovery = 2)
    end

    @testset "fresh-start delegates to pre-fitted (single-basin, warns once)" begin
        cf_sb = CostFunction(chi2_sb, 1.0)
        m_pre   = @test_logs((:warn, r"No deeper basin|no deeper basin"), min_level = Logging.Warn,
                             find_deeper_minimum(m_sb, refit_sb, pts_sb; n_discovery = 10, seed = 2))
        m_fresh = @test_logs((:warn, r"No deeper basin|no deeper basin"), min_level = Logging.Warn,
                             find_deeper_minimum(cf_sb, [1.9], [0.1], refit_sb, pts_sb; n_discovery = 10, seed = 2))
        @test m_pre isa Minuit && m_fresh isa Minuit
        @test m_pre.fval ≈ m_fresh.fval atol = 1e-4
    end

    @testset "plain-callable wrapper for resampling" begin
        m_plain = @test_logs((:warn, r"No deeper basin|no deeper basin"), min_level = Logging.Warn,
                             find_deeper_minimum(chi2_sb, [1.9], [0.1], refit_sb, pts_sb; n_discovery = 10, seed = 3))
        @test m_plain isa Minuit
    end

    @testset "dispatch disambiguator" begin
        @test_throws ArgumentError find_deeper_minimum(m_sb, [1.0], [0.1])
        @test_throws ArgumentError find_deeper_minimum(m_sb, [1.0], [0.1], refit_sb, pts_sb)
    end

    @testset "refit returning wrong-length vector is filtered" begin
        refit_bad = (subdata, start) -> fill(NaN, length(start) + 1)
        m_out = @test_logs((:warn, r"valid resample"), min_level = Logging.Warn,
                           find_deeper_minimum(m_sb, refit_bad, pts_sb; n_discovery = 4, seed = 4))
        @test m_out isa Minuit
        @test m_out.fval ≈ m_sb.fval atol = 1e-6
    end
end

@testset "find_deeper_minimum — resampling ADOPTION path (deeper basin found)" begin
    # Drives the FULL adoption path: discovery → find_solution_modes refine →
    # new_min → rebuild Minuit → migrad!+hesse → loop. Tilted double well; start
    # shallow; refit returns DEEP-well candidates so adoption fires.
    f(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + x[2]^2
    m = Minuit(f, [1.0, 0.3]; errors = [0.2, 0.2], strategy = 2); migrad!(m); hesse(m)
    m.ndata = 17
    @test m.values[1] > 0
    data = collect(1.0:10.0)
    refit_deep = (subdata, start) -> [-1.0 + 0.01*(sum(subdata)/length(subdata) - 5.5),
                                       0.0 + 0.01*(sum(subdata)/length(subdata) - 5.5)]
    m_deep = find_deeper_minimum(m, refit_deep, data; n_discovery = 12, seed = 1)
    @test m_deep isa Minuit
    @test m_deep.valid
    @test m_deep.values[1] < 0                    # escaped to the DEEP well
    @test m_deep.fval < m.fval - 0.5
    @test m_deep.ndata == 17                      # ndata carried through the rebuild
    @test m_deep !== m                            # a NEW (cloned) Minuit
    m_deep2 = find_deeper_minimum(m, refit_deep, data; n_discovery = 12, seed = 1)
    @test collect(m_deep2.values) ≈ collect(m_deep.values)
end

@testset "find_deeper_minimum — resampling adoption honours a FIXED parameter" begin
    # x[2] fixed at 2.0; its unconstrained optimum is 3.0. Even though the search
    # adopts a deeper x[1] basin, x[2] must stay pinned at 2.0 (refine pins it AND
    # the adoption clone keeps the fix) — NOT drift to 3.0.
    h(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + (x[2] - 3.0)^2
    m = Minuit(h, [1.0, 2.0]; errors = [0.2, 0.2], fixed = [false, true], strategy = 2)
    migrad!(m); hesse(m)
    @test m.fixed[2] == true && m.values[2] == 2.0
    data = collect(1.0:10.0)
    # refit returns deep-well x[1] with x[2] held at the fixed value
    refit_w = (subdata, start) -> [-1.0 + 0.01*(sum(subdata)/length(subdata) - 5.5), 2.0]
    m_deep = with_logger(NullLogger()) do
        find_deeper_minimum(m, refit_w, data; n_discovery = 12, seed = 1)
    end
    @test m_deep.values[1] < 0                    # deep basin adopted
    @test m_deep.fixed[2] == true                 # fix survived the adoption rebuild
    @test m_deep.values[2] == 2.0                 # pinned (would drift toward 3.0 if un-fixed)
end

@testset "find_deeper_minimum — AD-gradient parity + check_gradient" begin
    fq(x) = (x[1] - 2.0)^2 + (x[2] + 1.0)^2
    gq(x) = [2 * (x[1] - 2.0), 2 * (x[2] + 1.0)]
    m_ad = Minuit(fq, [0.0, 0.0]; errors = [0.3, 0.3], grad = gq, strategy = 1); migrad!(m_ad)
    @test m_ad.cfwg !== nothing

    # perturbation m::Minuit keeps the AD gradient (clones route through m.cfwg)
    r = find_deeper_minimum(m_ad; n_restarts = 10, perturb = 0.4, seed = 1)
    @test r isa Minuit && r.valid
    @test collect(r.values) ≈ [2.0, -1.0] atol = 1e-3
    @test r.cfwg !== nothing                      # gradient survived the perturbation clones

    # fresh-start resampling keeps cf.g for a CostFunctionWithGradient
    refit_ad = (subdata, start) -> begin
        fm2 = migrad(CostFunction(fq, 1.0), start, [0.3, 0.3]; strategy = JuMinuit.Strategy(1))
        JuMinuit.is_valid(fm2) ? collect(Float64, values(fm2)) : fill(NaN, length(start))
    end
    data_ad = fill(1.0, 16)
    m_out = with_logger(NullLogger()) do
        find_deeper_minimum(m_ad.cfwg, [0.0, 0.0], [0.3, 0.3], refit_ad, data_ad; n_discovery = 8, seed = 1)
    end
    @test m_out isa Minuit && m_out.cfwg !== nothing

    # check_gradient=false preserved through fresh-start
    m_cg = Minuit(fq, [0.0, 0.0]; errors = [0.3, 0.3], grad = gq, check_gradient = false)
    @test m_cg.cfwg.check_gradient == false
    m_out_cg = with_logger(NullLogger()) do
        find_deeper_minimum(m_cg.cfwg, [0.0, 0.0], [0.3, 0.3], refit_ad, data_ad; n_discovery = 8, seed = 1)
    end
    @test m_out_cg.cfwg !== nothing && m_out_cg.cfwg.check_gradient == false

    # check_gradient=false preserved through an ACTUAL adoption rebuild
    fw(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + x[2]^2
    gw(x) = [4 * x[1] * (x[1]^2 - 1) + 0.4, 2 * x[2]]
    m_w = Minuit(fw, [1.0, 0.3]; errors = [0.2, 0.2], grad = gw, check_gradient = false, strategy = 2)
    migrad!(m_w); hesse(m_w)
    refit_w = (subdata, start) -> [-1.0 + 0.01*(sum(subdata)/length(subdata) - 5.5),
                                    0.0 + 0.01*(sum(subdata)/length(subdata) - 5.5)]
    data_w = collect(1.0:10.0)
    m_w_deep = with_logger(NullLogger()) do
        find_deeper_minimum(m_w, refit_w, data_w; n_discovery = 12, seed = 1)
    end
    @test m_w_deep.values[1] < 0
    @test m_w_deep.cfwg !== nothing && m_w_deep.cfwg.check_gradient == false
end

@testset "find_deeper_minimum — max_rounds is a backstop; convergence is the stop" begin
    # The stop criterion is convergence (a round finding no deeper basin). A
    # max_rounds cap hit WHILE still improving must WARN (never silently truncate),
    # and must still return the deeper basin it did reach.
    f(x) = (x[1]^2 - 1)^2 + 0.4 * x[1] + x[2]^2

    # perturbation: round 1 escapes to the deep well (improves); max_rounds=1 caps it.
    mc = @test_logs (:warn, r"not converged") match_mode = :any begin
        find_deeper_minimum(f, [1.0, 0.5], [0.3, 0.3];
                            n_restarts = 80, perturb = 2.0, max_rounds = 1, seed = 1)
    end
    @test mc.values[1] < 0                       # still returned the deeper basin

    # resampling: round 1 adopts the deep well; max_rounds=1 caps mid-descent.
    m = Minuit(f, [1.0, 0.3]; errors = [0.2, 0.2], strategy = 2); migrad!(m); hesse(m)
    data = collect(1.0:10.0)
    refit_deep = (sub, st) -> [-1.0 + 0.01*(sum(sub)/length(sub) - 5.5),
                                0.0 + 0.01*(sum(sub)/length(sub) - 5.5)]
    mr = @test_logs (:warn, r"not converged") match_mode = :any begin
        find_deeper_minimum(m, refit_deep, data; n_discovery = 12, max_rounds = 1, seed = 1)
    end
    @test mr.values[1] < 0

    # at the (high) default it converges → NO cap warning emitted.
    lg = Test.TestLogger(min_level = Logging.Warn)
    md = with_logger(lg) do
        find_deeper_minimum(m, refit_deep, data; n_discovery = 12, seed = 1)
    end
    @test md.values[1] < 0
    @test !any(occursin("not converged", string(r.message)) for r in lg.logs)
end
