# SPDX-License-Identifier: LGPL-2.1-or-later
#
# P6 — non-finite FCN values (find_solution_modes stress-test handoff,
# finding F7): "can a trial point with non-finite FCN value ever become
# the incumbent?"
#
# Answer + contract pinned by these tests:
#   • A non-finite (NaN/+Inf) trial can NEVER displace a finite
#     incumbent — IEEE `<` comparisons reject it, exactly like C++
#     Minuit2 / iminuit (verified against iminuit 2.31: NaN wall →
#     stall at the seed with finite fval, valid=False).
#   • The incumbent fval CAN be non-finite when the FCN is non-finite
#     at the very first evaluation (the seed), or via a -Inf trial that
#     legitimately wins the comparisons (iminuit accepts -Inf too).
#     In that case the minimum must be INVALID with the explicit
#     `nonfinite_fval` reason — before this fix, a seed-NaN fit at
#     Strategy 0 returned `is_valid = true` with `fval = NaN`, and at
#     Strategy ≥ 1 the run CRASHED with a raw LAPACK
#     `ArgumentError("matrix contains Infs or NaNs")` out of
#     make_posdef!/hesse (iminuit never crashes: fval=nan, valid=False).
#   • Every public migrad entry warns ONCE at the end when any
#     non-finite value was returned (iminuit warns per occurrence via
#     MnPrint, invisible at default print level); inner/warm-restart
#     probes stay silent.

using JuMinuit
using Test
using LinearAlgebra
using Logging

# Quadratic bowl Σᵢ (xᵢ − cᵢ)² with c = (c1, 0, …, 0), returning
# `wallval` for x[1] > wall — mirrors the field FCN (model undefined
# beyond a physical boundary / matrix near-singularity).
function _wall_fcn(wall::Float64, wallval::Float64; c1::Float64 = 0.0)
    return function (x)
        x[1] > wall && return wallval
        s = (x[1] - c1)^2
        @inbounds for i in 2:length(x)
            s += x[i]^2
        end
        return s
    end
end

@testset "non-finite FCN handling (P6 / handoff F7)" begin

    @testset "line_search: non-finite trial never displaces finite incumbent" begin
        for wallval in (NaN, Inf)
            f = _wall_fcn(0.05, wallval)            # min at 0, wall just past it
            cf = CostFunction(f, 1.0)
            x0 = [-0.5, 0.0]
            f0 = cf(x0)                              # 0.25, finite
            par = MinimumParameters(copy(x0), f0)
            step = [1.0, 0.0]                        # crosses the wall at slam ≳ 0.55
            gdel = -1.0                              # descent direction
            pp = line_search(cf, par, step, gdel)
            @test isfinite(pp.y)
            @test pp.y <= f0
            @test isfinite(pp.x)
        end
    end

    @testset "seed-time non-finite fval ⇒ invalid + explicit reason (low-level)" begin
        # wall at -1.0 ⇒ the FCN is `wallval` everywhere MIGRAD looks,
        # including the seed — the non-finite value IS the incumbent.
        for wallval in (NaN, Inf, -Inf), strat in (0, 1, 2)
            f = _wall_fcn(-1.0, wallval)
            # Regression 1 (F7): at Strategy 0 this returned is_valid=true
            # with fval=NaN. Regression 2: at Strategy ≥ 1 this THREW
            # ArgumentError from LAPACK eigvals inside make_posdef!/hesse.
            fm = migrad(f, [0.0, 0.0], [0.1, 0.1];
                        strategy = Strategy(strat), maxfcn = 300,
                        warn_nonfinite = false)
            @test !is_valid(fm)
            @test fm.nonfinite_fval
            @test nonfinite_fval(fm)
            @test !isfinite(fval(fm))
            @test fm.n_nonfinite_calls > 0
            @test n_nonfinite_calls(fm) == fm.n_nonfinite_calls
        end
    end

    @testset "AD/CFwG path: seed-NaN ⇒ invalid + flagged, no throw" begin
        f = _wall_fcn(-1.0, NaN)
        cfwg = CostFunctionWithGradient(f, x -> 2 .* x, 1.0;
                                        check_gradient = false)
        fm = migrad(cfwg, [0.0, 0.0], [0.1, 0.1];
                    strategy = Strategy(1), maxfcn = 300,
                    warn_nonfinite = false)
        @test !is_valid(fm)
        @test fm.nonfinite_fval
        @test fm.n_nonfinite_calls > 0
    end

    @testset "Minuit front end: seed-NaN ⇒ valid=false, ONE warning, no crash" begin
        for strat in (0, 1, 2)
            f = _wall_fcn(-1.0, NaN)
            m = Minuit(f, [0.0, 0.0]; error = 0.1, strategy = strat)
            # Exactly ONE aggregate warning for the whole migrad! run —
            # despite the iminuit-style iterate=5 retry loop.
            @test_logs (:warn, r"non-finite") migrad!(m; maxfcn = 300)
            @test !m.valid
            @test isnan(m.fval)
            @test m.fmin.internal.nonfinite_fval
            @test m.fmin.internal.n_nonfinite_calls > 0
        end
    end

    @testset "NaN wall blocking the path: stall stays finite + invalid (iminuit parity)" begin
        # True minimum beyond the wall (c1=1, wall at 0.5): every line
        # search crosses into NaN. iminuit 2.31 observable: fval stays
        # at the finite seed value 1.0, valid=False. The non-finite
        # returns are COUNTED but the final fval is finite ⇒ no
        # nonfinite_fval flag.
        for strat in (0, 1)
            f = _wall_fcn(0.5, NaN; c1 = 1.0)
            m = Minuit(f, [0.0, 0.0]; error = 0.1, strategy = strat)
            @test_logs (:warn, r"non-finite") migrad!(m; maxfcn = 300)
            @test !m.valid
            @test isfinite(m.fval)
            @test m.fval <= 1.0 + 1e-12          # never worse than the seed
            @test !m.fmin.internal.nonfinite_fval
            @test m.fmin.internal.n_nonfinite_calls > 0
        end
    end

    @testset "+Inf wall: rejected like NaN, finite incumbent kept" begin
        f = _wall_fcn(0.5, Inf; c1 = 1.0)
        fm = migrad(f, [0.0, 0.0], [0.1, 0.1];
                    strategy = Strategy(1), maxfcn = 300,
                    warn_nonfinite = false)
        @test isfinite(fval(fm))
        @test !fm.nonfinite_fval
        @test fm.n_nonfinite_calls > 0
    end

    @testset "VALID fit that brushed non-finite values stays silent" begin
        # Log-domain likelihoods routinely push predicted counts to 0 on
        # exploratory trial steps (→ ±Inf/NaN) yet converge fine. iminuit
        # is silent on these at default print level — so are we. The
        # count stays queryable. (This exact fit is in the precompile
        # workload; a warning here would pollute precompilation output.)
        n = [2.0, 5.0, 3.0]
        xe = [-1.0, -0.3, 0.3, 1.0]
        scdf = (x, p) -> p[3] / (1.0 + exp(-(x - p[1]) / p[2]))
        m = Minuit(ExtendedBinnedNLL(n, xe, scdf), [0.0, 0.5, 10.0])
        @test_logs migrad!(m)                 # asserts NO log records
        @test m.valid
        @test m.fmin.internal.n_nonfinite_calls > 0
        @test !m.fmin.internal.nonfinite_fval
    end

    @testset "clean fit: zero non-finite calls, no flag, no warning" begin
        f = x -> (x[1] - 1.0)^2 + 4.0 * (x[2] - 2.0)^2
        m = Minuit(f, [0.0, 0.0]; error = 0.1)
        @test_logs migrad!(m)                     # asserts NO log records
        @test m.valid
        @test !m.fmin.internal.nonfinite_fval
        @test m.fmin.internal.n_nonfinite_calls == 0
        @test nonfinite_calls(m.fcn) == 0
    end

    @testset "seed-state migrad overload (inner-probe path) stays silent" begin
        f = _wall_fcn(-1.0, NaN)
        cf = CostFunction(f, 1.0)
        seed = seed_state(cf, [0.0, 0.0], [0.1, 0.1], Strategy(0),
                          MachinePrecision())
        # warm-restart/cross-search entry: counts but never warns
        fm = @test_logs migrad(cf, seed; strategy = Strategy(0), maxfcn = 300)
        @test !is_valid(fm)
        @test fm.nonfinite_fval
    end

    @testset "non-finite counter on the FCN wrappers" begin
        f = _wall_fcn(0.0, NaN)
        cf = CostFunction(f, 1.0)
        @test nonfinite_calls(cf) == 0
        cf([1.0, 0.0])                            # NaN region
        cf([-1.0, 0.0])                           # finite region
        @test ncalls(cf) == 2
        @test nonfinite_calls(cf) == 1
        reset_ncalls!(cf)
        @test ncalls(cf) == 0 && nonfinite_calls(cf) == 0

        cfwg = CostFunctionWithGradient(f, x -> 2 .* x, 1.0;
                                        check_gradient = false)
        cfwg([1.0, 0.0])
        @test nonfinite_calls(cfwg) == 1
        reset_ncalls!(cfwg)
        @test nonfinite_calls(cfwg) == 0
    end

    @testset "make_posdef family: non-finite matrices never reach LAPACK" begin
        prec = MachinePrecision()
        for bad in (NaN, Inf, -Inf)
            # in-place variant (the strategy ≥ 1 hesse path that crashed)
            S = Symmetric([1.0 bad; 0.0 1.0], :U)
            @test JuMinuit.make_posdef!(S, prec) == true   # no throw; tag applies

            # n=1 fall-through (previously reached the eigenvalue gate)
            S1 = Symmetric(fill(bad, 1, 1), :U)
            @test JuMinuit.make_posdef!(S1, prec) == true

            # allocating variant (MIGRAD-loop recovery path)
            err = MinimumError(Symmetric([bad 0.0; 0.0 1.0], :U), 0.0)
            err2 = make_posdef(err, prec)
            @test is_made_pos_def(err2)

            # quick-check helper
            @test !is_posdef_enough(err, prec)
        end
        # `make_posdef!` is unchanged for finite input
        Sok = Symmetric([2.0 0.1; 0.0 3.0], :U)
        @test JuMinuit.make_posdef!(Sok, prec) == false
    end

    @testset "standalone hesse on a NaN-straddling FCN: invalid, no throw" begin
        # f finite AT x0 but NaN one probe step away — the wall sits
        # tight against the seed so the HESSE x±d probes MUST cross it
        # whatever step-refinement d ends up at.
        f = _wall_fcn(1e-8, NaN)
        r = hesse(f, [0.0, 0.0], [0.1, 0.1])
        @test r.valid == false
    end

    @testset "hesse! after a NaN-flagged fit keeps the sticky invalid reason" begin
        f = _wall_fcn(-1.0, NaN)
        m = Minuit(f, [0.0, 0.0]; error = 0.1, strategy = 1)
        with_logger(NullLogger()) do
            migrad!(m; maxfcn = 300)
            hesse!(m)
        end
        @test !m.valid
        @test m.fmin.internal.nonfinite_fval
    end

    @testset "display + serialization surface the reason" begin
        f = _wall_fcn(-1.0, NaN)
        fm = migrad(f, [0.0, 0.0], [0.1, 0.1];
                    strategy = Strategy(0), maxfcn = 300,
                    warn_nonfinite = false)
        @test occursin("fval non-finite", sprint(show, MIME("text/plain"), fm))
        d = JuMinuit.to_dict(fm)
        @test d["nonfinite_fval"] === true
        @test d["n_nonfinite_calls"] >= 1
        @test d["valid"] === false

        m = Minuit(f, [0.0, 0.0]; error = 0.1, strategy = 0)
        with_logger(NullLogger()) do
            migrad!(m; maxfcn = 300)
        end
        @test occursin("FCN value finite", JuMinuit._checklist_text(m))
        db = JuMinuit.to_dict(m.fmin)
        @test db["nonfinite_fval"] === true
    end
end
