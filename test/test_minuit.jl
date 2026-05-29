# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "minuit.jl — iminuit-style Minuit wrapper" begin

    @testset "Constructor + property access (no migrad)" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                    names = ["x", "y"], errors = [0.1, 0.2])
        @test m isa Minuit
        @test m.ndim == 2
        @test m.npar == 2
        @test m.values == [1.0, 2.0]  # initial
        @test m.errors == [0.1, 0.2]  # initial
        @test isnan(m.fval)
        @test isnan(m.edm)
        @test m.nfcn == 0
        @test !m.valid
        @test m.covariance === nothing
    end

    @testset "Default strategy = 1 (iminuit Minuit-class parity)" begin
        # Regression for docs/IAM_CONVERGENCE_GAP.md: the high-level
        # Minuit(fcn, x0) constructor must default to Strategy(1) — the
        # iminuit `Minuit` class default and C++ Minuit2 `MnStrategy()`
        # default — so a bare `migrad!(m)` is drop-in-equivalent to
        # iminuit's `m.migrad()`. (The low-level migrad(cf, …) keeps its
        # own Strategy(0) default, pinned to the C++ oracle.)
        m_num = Minuit(x -> sum(abs2, x), [1.0, 2.0]; error = [0.1, 0.1])
        @test m_num.strategy == Strategy(1)
        @test m_num.strategy.level == 1

        # The named-parameter constructor agrees.
        m_kw = Minuit(x -> sum(abs2, x); a = 1.0, b = 2.0)
        @test m_kw.strategy == Strategy(1)

        # The AD (`grad=`) default is ALSO Strategy(1) — iminuit applies
        # strategy 1 regardless of whether a gradient is supplied, and the
        # AD seed_state now supports all strategy levels (no asymmetry).
        m_ad = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                       error = [0.1, 0.1],
                       grad = x -> 2 .* x)
        @test m_ad.strategy == Strategy(1)
        # …and a default AD fit runs end-to-end at S=1 (would throw under
        # the old "Phase 0 supports Strategy(0) only" AD-seed guard).
        migrad!(m_ad)
        @test m_ad.valid
        @test m_ad.values[1] ≈ 0.0 atol = 1e-4
        @test m_ad.values[2] ≈ 0.0 atol = 1e-4

        # An explicit strategy is always respected over the default.
        m_s0 = Minuit(x -> sum(abs2, x), [1.0, 2.0]; strategy = Strategy(0))
        @test m_s0.strategy == Strategy(0)
        m_s2 = Minuit(x -> sum(abs2, x), [1.0, 2.0]; strategy = 2)
        @test m_s2.strategy == Strategy(2)

        # AD at S=2 (seed-time MnHesse bootstrap path) also runs.
        m_ad2 = Minuit(x -> sum(abs2, x .- 1.0), [0.0, 0.0];
                        error = [0.1, 0.1], grad = x -> 2 .* (x .- 1.0),
                        strategy = 2)
        migrad!(m_ad2; iterate = 1)
        @test m_ad2.valid
        @test m_ad2.values[1] ≈ 1.0 atol = 1e-4
    end

    @testset "AbstractFit / Fit / ArrayFit (IMinuit.jl drop-in types)" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        # Minuit is a concrete subtype of the AbstractFit supertype.
        @test Minuit <: AbstractFit
        @test m isa AbstractFit
        # Fit / ArrayFit are aliases of Minuit (same type, not distinct).
        @test Fit === Minuit
        @test ArrayFit === Minuit
        @test m isa Fit
        @test m isa ArrayFit
        # Type annotations that IMinuit.jl user code uses must dispatch.
        f_take(::AbstractFit) = :abstract
        f_take_arr(::ArrayFit) = :array
        @test f_take(m) === :abstract
        @test f_take_arr(m) === :array
        # Keyword/scalar-arg construction (IMinuit.jl `Fit` form) yields
        # the same type as the array form.
        mk = Minuit(x -> sum(abs2, x); a = 1.0, b = 2.0)
        @test mk isa AbstractFit
        @test typeof(mk) === typeof(m)
    end

    @testset "migrad! workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"], errors = [0.1, 0.1])
        migrad!(m)
        @test m.valid
        @test m.fval < 1e-8
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
        @test m.covariance isa Matrix{Float64}
        @test size(m.covariance) == (2, 2)
    end

    @testset "Bounded + fixed parameters via Minuit" begin
        m = Minuit(x -> (x[1] - 0.5)^2 + (x[2] - 3.0)^2, [0.3, 5.0];
                    names = ["a", "b"], errors = [0.1, 0.1],
                    limits = [(0.0, 1.0), nothing],
                    fixed = [false, true])
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 0.5 atol = 0.01
        @test m.values[2] == 5.0  # fixed bit-exact
        @test m.errors[2] == 0.0   # fixed → no error
    end

    @testset "minos! workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"])
        migrad!(m)
        minos!(m, 1)
        @test haskey(m.minos_errors, 1)
        e = m.minos_errors[1]
        @test JuMinuit.is_valid(e)
        @test e.upper ≈ 1.0 atol = 0.1
        @test e.lower ≈ -1.0 atol = 0.1

        # By name
        minos!(m, "y")
        @test haskey(m.minos_errors, 2)

        # All free
        m2 = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        migrad!(m2)
        minos!(m2)
        @test length(m2.minos_errors) == 2
    end

    # ─────────────────────────────────────────────────────────────────
    # Fix 3: MnMinos n-scaled default call budget (C++ MnMinos.cxx
    # :111-114). With no explicit `maxcall`, the per-cross-search budget
    # is 2·(nvar+1)·(200+100·nvar+5·nvar²), not the legacy hardcoded 1000.
    # An explicit `maxcall` must still be respected.
    # ─────────────────────────────────────────────────────────────────
    @testset "MINOS default budget is n-scaled (MnMinos.cxx:111-114)" begin
        # (a) The default per-cross-search budget equals the C++/iminuit
        #     formula (the value the `maxcall == 0` path forwards).
        for nvar in (1, 2, 5, 9)
            @test JuMinuit._minos_default_maxcalls(nvar) ==
                  2 * (nvar + 1) * (200 + 100 * nvar + 5 * nvar^2)
        end
        @test JuMinuit._minos_default_maxcalls(9) == 30100   # the audit's figure
        @test JuMinuit._minos_default_maxcalls(2) > 1000     # beats legacy default

        # (b) Fixed parameters are excluded from nvar (matches C++
        #     MnUserParameterState::VariableParameters()).
        mf = Minuit(x -> sum(abs2, x), [1.0, 2.0, 3.0];
                    names = ["a", "b", "c"], fixed = [false, true, false])
        @test JuMinuit.n_free(mf.params) == 2

        # (c) Default (maxcall=0 sentinel) succeeds with the n-scaled
        #     budget; an explicit tiny `maxcall` is RESPECTED — it
        #     truncates the cross-search (fcn_limit) rather than being
        #     ignored in favor of the n-scaled default.
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"])
        migrad!(m)
        minos!(m, 1)                       # default budget
        minos!(m, 2; maxcall = 1)          # explicit tiny override
        @test JuMinuit.is_valid(m.minos_errors[1])
        @test !m.minos_errors[1].upper_fcn_limit
        @test !m.minos_errors[1].lower_fcn_limit
        @test !JuMinuit.is_valid(m.minos_errors[2])
        @test m.minos_errors[2].upper_fcn_limit || m.minos_errors[2].lower_fcn_limit
    end

    @testset "contour workflow" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"])
        migrad!(m)
        c = contour(m, 1, 2; npoints = 10)
        @test c isa ContoursError
        @test c.valid
        @test length(c.points) == 10

        # By name
        c2 = contour(m, "x", "y"; npoints = 8)
        @test c2.valid
        @test length(c2.points) == 8
    end

    @testset "Pretty print" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                    names = ["x", "y"],
                    limits = [(-5.0, 5.0), nothing])
        # Before migrad
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("not yet minimized", s)
        @test occursin("[-5.0, 5.0]", s)

        # After migrad — Phase 3 C1 Unicode table format
        migrad!(m)
        buf2 = IOBuffer()
        show(buf2, MIME"text/plain"(), m)
        s2 = String(take!(buf2))
        # Header line carries fval / edm / nfcn / status
        @test occursin("fval=", s2)
        @test occursin("nfcn=", s2)
        @test occursin("Valid", s2)
        # Unicode box-drawing characters present
        @test occursin("┌", s2)
        @test occursin("┤", s2)
        @test occursin("└", s2)
        # Column headers. Phase: the feat/jupyter-rich-output overhaul
        # MERGED the old "Hesse ±" / "Minos −" / "Minos +" columns into a
        # single "Value" column (value ± uncertainty, or asymmetric MINOS
        # superscript/subscript when present), so those three headers are
        # gone by design — see test_display.jl for the merged-cell content.
        for col in ("Name", "Value", "Limit −", "Limit +", "Fixed")
            @test occursin(col, s2)
        end
        @test !occursin("Hesse ±", s2)
        @test !occursin("Minos −", s2)
    end

    @testset "C1 (a) at-limit warning detection" begin
        # Force a parameter to sit on a tight lower bound: fit
        # (x-0.5)² with x ∈ [0.3, 10]. The minimum is at 0.5, the lower
        # bound is 0.3 away — well within 1σ (Hesse err ≈ 1.0).
        cf = x -> (x[1] - 0.5)^2
        m = Minuit(cf, [1.0]; name = ["a"], limit_a = (0.3, 10.0))
        migrad(m)
        @test m.is_valid
        # At-limit detector should flag `a` (lower edge within 1σ)
        @test 1 in JuMinuit._at_limit_indices(m)
        # Warning visible in text/plain output
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        s = String(take!(buf))
        @test occursin("⚠", s)
        @test occursin("`a`", s)
        @test occursin("lower limit", s)
        @test occursin("unreliable", s)

        # Negative case: parameter NOT at limit
        cf2 = x -> (x[1] - 5.0)^2
        m2 = Minuit(cf2, [4.0]; name = ["a"], limit_a = (-100.0, 100.0))
        migrad(m2)
        @test isempty(JuMinuit._at_limit_indices(m2))
        buf2 = IOBuffer()
        show(buf2, MIME"text/plain"(), m2)
        s2 = String(take!(buf2))
        @test !occursin("⚠", s2)
    end

    @testset "MINOS sign + magnitude on real bounded fits (review round-2)" begin
        # Round-2 review (both codex + Opus) caught that the round-1
        # fix mistook Jacobian-sign inversion (intrinsic for UpperOnly
        # — sqrtup's d(ext)/d(int) is negative throughout) for
        # saturation. The proper fix is a per-kind swap of upper/lower
        # int steps before conversion. These tests assert the FIX is
        # correct (not just "wrong-signed values aren't published"),
        # using real migrad + minos on interior-optimum bounded fits
        # — the basic HEP use case (one-sided positivity constraint).

        @testset "UpperOnly interior optimum: real fit" begin
            # f(x, y) = (x - 8)² + (y - 2)²; x ∈ (-∞, 10], optimum
            # at x=8 well-interior to upper bound. Hesse σ_x = 1.
            # MINOS at 1σ should give upper ≈ +1 (toward x=9), lower
            # ≈ −1 (toward x=7); both must be valid.
            m = Minuit(x -> (x[1] - 8.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(nothing, 10.0), nothing])
            migrad(m)
            @test m.is_valid
            @test m.values[1] ≈ 8.0 atol = 1e-3
            minos(m, 1)
            e = m.minos_errors[1]
            @test e.upper_valid
            @test e.lower_valid
            @test e.upper > 0           # sign convention
            @test e.lower < 0
            @test e.upper ≈ 1.0 atol = 0.05   # magnitude
            @test e.lower ≈ -1.0 atol = 0.05
            @test !e.upper_fcn_limit
            @test !e.lower_fcn_limit
        end

        @testset "LowerOnly interior optimum: real fit" begin
            # Mirror: x ∈ [0, ∞), optimum at x=8 well-interior to
            # lower bound. Same expected MINOS errors.
            m = Minuit(x -> (x[1] - 8.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(0.0, nothing), nothing])
            migrad(m)
            @test m.is_valid
            minos(m, 1)
            e = m.minos_errors[1]
            @test e.upper_valid
            @test e.lower_valid
            @test e.upper > 0
            @test e.lower < 0
            @test e.upper ≈ 1.0 atol = 0.05
            @test e.lower ≈ -1.0 atol = 0.05
        end

        @testset "BothBounds interior optimum: real fit" begin
            # x ∈ [-100, 100]. Standard Sin transform; should match
            # unbounded MINOS within EDM noise.
            m = Minuit(x -> (x[1] - 8.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(-100.0, 100.0), nothing])
            migrad(m)
            @test m.is_valid
            minos(m, 1)
            e = m.minos_errors[1]
            @test e.upper_valid
            @test e.lower_valid
            @test e.upper > 0
            @test e.lower < 0
            @test e.upper ≈ 1.0 atol = 0.05
            @test e.lower ≈ -1.0 atol = 0.05
        end

        @testset "UpperOnly saturated: param at upper bound" begin
            # Fit pushes x toward upper bound: optimum is at x=12 but
            # upper=10. MINOS upper side saturates AT the bound; lower
            # side stays interior. Two-param form (MINOS needs n>1).
            m = Minuit(x -> (x[1] - 12.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(nothing, 10.0), nothing])
            migrad(m)
            @test m.is_valid
            @test m.values[1] ≈ 10.0 atol = 1e-2   # at upper bound
            minos(m, 1)
            e = m.minos_errors[1]
            # Upper saturated: par_limit raised (NOT fcn_limit — round-3
            # I-4 separates them: par_limit = "hit a bound", fcn_limit =
            # "exhausted budget"). The published `upper` value equals
            # the PHYSICAL bound_distance, matching iminuit. For this
            # at-the-bound case bound_distance is essentially 0.
            # `upper_valid=true` under new semantics: saturating against
            # a bound is a clean MINOS termination (matches iminuit).
            bound_distance = m.params.pars[1].upper - m.values[1]
            @test e.upper_valid
            @test e.upper_par_limit
            @test !e.upper_fcn_limit
            @test e.upper ≈ bound_distance atol = 1e-6
            # Lower side: search goes away from bound, finds interior
            # crossing (or also saturates if x has bumped 1σ toward the
            # bound). Sign must be ≤ 0.
            @test e.lower <= 0
        end

        @testset "LowerOnly saturated: param at lower bound (round-6 mirror)" begin
            # Mirror of the UpperOnly saturated test, but with the
            # optimum below the lower bound so x converges AT the
            # bound. Round-6 polish: ensures the LowerOnly path of
            # the at-limit publish-bound_distance fix is exercised.
            m = Minuit(x -> (x[1] - 4.0)^2 + (x[2] - 2.0)^2, [10.0, 0.0];
                        name = ["x", "y"], limits = [(6.0, nothing), nothing])
            migrad(m)
            @test m.is_valid
            @test m.values[1] ≈ 6.0 atol = 1e-2   # at lower bound
            minos(m, 1)
            e = m.minos_errors[1]
            # Lower saturated: par_limit raised, lower_err equals the
            # PHYSICAL bound_distance (negative — par.lower − ext_min).
            # `lower_valid=true` under new semantics.
            bound_distance = m.params.pars[1].lower - m.values[1]   # ≤ 0
            @test e.lower_valid
            @test e.lower_par_limit
            @test !e.lower_fcn_limit
            @test e.lower ≈ bound_distance atol = 1e-6
            # Upper side stays interior (search goes away from bound).
            @test e.upper >= 0
        end

        @testset "BothBounds saturated: both sides clip (round-6 mirror)" begin
            # Bounds at ±0.5 around optimum at 0 with σ ≈ 1 — both
            # MINOS sides need to saturate. Tests the BothBounds
            # par_limit case symmetrically.
            m = Minuit(x -> x[1]^2 + (x[2] - 2.0)^2, [0.2, 0.0];
                        name = ["x", "y"], limits = [(-0.5, 0.5), nothing])
            migrad(m)
            @test m.is_valid
            minos(m, 1)
            e = m.minos_errors[1]
            up_bound = m.params.pars[1].upper - m.values[1]   # > 0
            lo_bound = m.params.pars[1].lower - m.values[1]   # < 0
            # Both sides should be `valid` under new semantics (clean
            # crossing OR at-limit termination both count). Switch on
            # `par_limit` to distinguish the saturated vs interior case.
            @test e.upper_valid
            @test e.lower_valid
            if e.upper_par_limit
                @test e.upper ≈ up_bound atol = 1e-6
            else
                @test e.upper <= up_bound + 1e-9
            end
            if e.lower_par_limit
                @test e.lower ≈ lo_bound atol = 1e-6
            else
                @test e.lower >= lo_bound - 1e-9
            end
        end

        @testset "UpperOnly L300/aulim overshoot (round-4 codex BLOCKING)" begin
            # Bound JUST PAST 1σ — search will L300-extend, the probe
            # will clamp ext_val to the bound for aopt > aulim, but
            # _cross_core records the unclamped aopt. Without the
            # round-4 fix, MINOS returned `valid=true` with
            # `upper > bound_distance` — a physics-wrong asymmetric
            # error (user sees an error larger than the bound allows).
            m = Minuit(x -> (x[1] - 8.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(nothing, 8.995), nothing])
            migrad(m); minos(m, 1)
            e = m.minos_errors[1]
            bound_distance = 8.995 - m.values[1]
            @test bound_distance > 0   # sanity: optimum inside the bound
            # The PHYSICAL invariant: any published upper error must
            # fit inside the bound. Pre-fix: upper ≈ 1.39, distance ≈
            # 0.997 → violation. Post-fix: upper ≤ bound_distance, and
            # the side is `valid` under new semantics whether it hit
            # the bound (par_limit=true) or found an interior crossing.
            @test e.upper_valid
            @test e.upper <= bound_distance + 1e-9
            # Lower side untouched (bound nowhere near).
            @test e.lower_valid
            @test e.lower ≈ -1.0 atol = 0.05
        end

        @testset "UpperOnly partial truncation: bound inside 1σ (round-3 I-1)" begin
            # The C++-faithful behavior: bound INSIDE the 1σ HESSE
            # interval but NOT at the converged minimum. Round-3
            # reviewers both flagged this as the missing par_limit
            # propagation case. Setup: x ∈ (-∞, 8.5] with optimum at
            # x=8 (well-interior); σ_x ≈ 1 → upper 1σ at x=9 would be
            # past the bound at 8.5.
            m = Minuit(x -> (x[1] - 8.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
                        name = ["x", "y"],
                        limits = [(nothing, 8.5), nothing])
            migrad(m)
            @test m.is_valid
            minos(m, 1)
            e = m.minos_errors[1]
            # Lower side unaffected (bound far from lower direction).
            @test e.lower_valid
            @test e.lower ≈ -1.0 atol = 0.05
            # Upper side: bound cuts into the 1σ interval. Should be
            # `valid` under new semantics (either succeeded at truncated
            # value with par_limit=false, OR saturated against the bound
            # with par_limit=true — both count as clean termination).
            # The SILENT failure (round-3 BLOCKING) would have valid=false
            # AND par_limit=false AND fcn_limit=false.
            @test e.upper_valid
            # No false fcn_limit (the budget wasn't the issue — the
            # bound was).
            @test !e.upper_fcn_limit
        end
    end

    @testset "at-limit warning labels lower/upper correctly (review BLOCKING #2)" begin
        # `has_limits(p) = has_lower_limit(p) || has_upper_limit(p)`,
        # so the prior `if has_limits(p) ... elseif has_upper_limit(p)`
        # path made the elseif/else branches unreachable. Regression:
        # a LowerOnly param sitting at its lower bound must print
        # "is at its **lower** limit", not "upper".
        cf = x -> (x[1] - 0.5)^2
        m_lo = Minuit(cf, [0.6]; name = ["a"], limit_a = (0.45, nothing))
        migrad(m_lo)
        @test m_lo.is_valid
        @test 1 in JuMinuit._at_limit_indices(m_lo)
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_lo)
        s = String(take!(buf))
        @test occursin("lower limit", s)
        @test !occursin("upper limit", s)

        # Symmetric: UpperOnly at upper bound.
        cf2 = x -> (x[1] - 5.0)^2
        m_up = Minuit(cf2, [4.9]; name = ["b"], limit_b = (nothing, 5.05))
        migrad(m_up)
        @test 1 in JuMinuit._at_limit_indices(m_up)
        buf2 = IOBuffer()
        show(buf2, MIME"text/plain"(), m_up)
        s2 = String(take!(buf2))
        @test occursin("upper limit", s2)
        @test !occursin("lower limit", s2)
    end

    @testset "hesse(m) clears prior sticky flags (review IMPORTANT)" begin
        # Both reviewers flagged that previous `hesse(m)` made
        # `made_pos_def` and `is_valid` sticky across calls. Test:
        # a fresh successful `hesse(m)` should CLEAR an artificially
        # injected prior `made_pos_def = true` / `is_valid = false`.
        f = x -> sum(abs2, x .- [1.0, 2.0])
        m = Minuit(f, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad(m)
        bfm = m.fmin
        # Inject artificial prior failure flags.
        fm_bad = JuMinuit.FunctionMinimum(
            bfm.internal.state, bfm.internal.seed, bfm.internal.up;
            is_valid = false,           # was failed
            hesse_failed = true,        # was failed
            made_pos_def = true,        # was perturbed
        )
        m.fmin = JuMinuit.BoundedFunctionMinimum(
            fm_bad, bfm.params, bfm.ext_values, bfm.ext_errors,
            bfm.ext_covariance, bfm.internal_cf,
        )
        # Now run hesse — should recover.
        JuMinuit.hesse(m)
        @test m.fmin.internal.hesse_failed == false   # cleared
        @test m.fmin.internal.made_pos_def == false   # cleared (new HESSE pos-def)
        @test m.is_valid == true                       # recovered
    end

    @testset "C1 (c) HTML repr (IJulia / Pluto)" begin
        cf = x -> sum(abs2, x .- [1.0, 2.0])
        m = Minuit(cf, [0.0, 0.0]; names = ["a", "b"])
        # Before migrad
        buf = IOBuffer()
        show(buf, MIME"text/html"(), m)
        @test occursin("not yet minimized", String(take!(buf)))

        # After migrad
        migrad(m)
        buf = IOBuffer()
        show(buf, MIME"text/html"(), m)
        s = String(take!(buf))
        # HTML structure
        @test occursin("<table", s)
        @test occursin("<thead", s)
        @test occursin("<tbody", s)
        @test occursin("</table>", s)
        # Column headers and the validity checklist. The old per-error
        # columns ("Hesse ±", "Minos −", "Minos +") were merged into a
        # single "Value" column by the feat/jupyter-rich-output overhaul,
        # and the single status badge became a per-flag checklist whose
        # first chip is "Valid minimum" (see test_display.jl).
        @test occursin("Value", s)
        @test occursin("Valid", s)
        @test !occursin("Hesse ±", s)
        @test !occursin("Minos −", s)

        # at-limit + HTML: yellow warning div appears
        m_lim = Minuit(x -> (x[1] - 0.5)^2, [1.0];
                       name = ["a"], limit_a = (0.3, 10.0))
        migrad(m_lim)
        buf = IOBuffer()
        show(buf, MIME"text/html"(), m_lim)
        s_lim = String(take!(buf))
        @test occursin("⚠", s_lim)
        @test occursin("<code>a</code>", s_lim)

        # HTML escape — parameter name with `<`, `&`, `"` must be
        # entity-encoded (review IMPORTANT). Without this, IJulia /
        # Pluto would render injected markup. Realistic for HEP users
        # naming params with operators like "a<b" for slicing categories.
        m_unsafe = Minuit(x -> (x[1] - 1.0)^2, [0.0];
                          name = ["a<b&\"q"])
        migrad(m_unsafe)
        buf = IOBuffer()
        show(buf, MIME"text/html"(), m_unsafe)
        s_unsafe = String(take!(buf))
        # Raw special characters must NOT appear in the HTML output.
        @test !occursin("a<b", s_unsafe)
        @test !occursin("a&q", s_unsafe)
        @test !occursin("a\"q", s_unsafe)
        # The escaped versions DO.
        @test occursin("a&lt;b", s_unsafe)
        @test occursin("&amp;", s_unsafe)
        @test occursin("&quot;", s_unsafe)
    end

    @testset "Argument validation" begin
        @test_throws ArgumentError Minuit(x -> 0.0, [1.0, 2.0]; names = ["x"])
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0])
        @test_throws ArgumentError minos!(m, 1)  # no migrad! yet
        @test_throws ArgumentError JuMinuit.hesse(m)  # no migrad! yet
    end

    @testset "hesse(m) refreshes the covariance (was placeholder)" begin
        # Task #36 — `hesse(m::Minuit)` used to be a no-op. After the
        # fix, a Strategy(0) MIGRAD followed by `hesse(m)` should give
        # a covariance numerically close to a Strategy(2) MIGRAD's
        # output (both end with a full numerical-HESSE pass).
        cf_fn = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        m_s0 = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad(m_s0; strategy = Strategy(0))
        cov_s0_dfp = collect(m_s0.covariance)   # DFP estimate

        JuMinuit.hesse(m_s0; strategy = Strategy(1))
        cov_s0_hesse = collect(m_s0.covariance)   # numerical HESSE
        @test m_s0.is_valid
        # Strategy(2) MIGRAD also ends with numerical HESSE.
        m_s2 = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad(m_s2; strategy = Strategy(2))
        cov_s2 = collect(m_s2.covariance)
        @test m_s2.is_valid

        # The hesse(m) refresh should match Strategy(2) MIGRAD's cov
        # element-by-element (both compute numerical 2nd-derivative
        # Hessian at the same converged minimum). Pure quadratic FCN
        # so the inverse Hessian is exact in both paths.
        @test cov_s0_hesse ≈ cov_s2 atol = 1e-8

        # Bounded variant: ensure the int↔ext transform round-trips
        # in hesse(m) too.
        m_bnd = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1],
                                          limit_x0 = (-5.0, 5.0))
        migrad(m_bnd; strategy = Strategy(0))
        cov_bnd_pre = collect(m_bnd.covariance)
        JuMinuit.hesse(m_bnd; strategy = Strategy(1))
        @test m_bnd.is_valid
        cov_bnd_post = collect(m_bnd.covariance)
        # Bounded path: covariance shape preserved, diagonals positive.
        @test size(cov_bnd_post) == size(cov_bnd_pre)
        @test all(diag(cov_bnd_post) .> 0)
        # External errors get refreshed via int2ext_error.
        @test all(m_bnd.errors .> 0)

        # hesse(m) returns m for chaining.
        m_chain = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad(m_chain)
        @test JuMinuit.hesse(m_chain) === m_chain
    end

    @testset "hesse(m) works when m was built with grad=" begin
        # Regression — hesse(::AbstractCostFunction, ...) used to be
        # narrowly typed to ::CostFunction, so Minuit constructed with
        # an analytical gradient (CostFunctionWithGradient internally)
        # raised MethodError on `hesse(m)`. HESSE only calls `cf(x)`
        # — not the gradient — so it should accept either flavor.
        cf_fn = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
        cf_grad = x -> [2*(x[1] - 1.0), 2*(x[2] - 2.0)]
        m = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1], grad = cf_grad)
        migrad(m; strategy = Strategy(0))
        @test JuMinuit.hesse(m) === m   # no MethodError
        @test m.is_valid
        # Cov matches the numerical-gradient path on a pure quadratic.
        cov_grad = collect(m.covariance)
        m_num = Minuit(cf_fn, [0.0, 0.0]; errors = [0.1, 0.1])
        migrad(m_num; strategy = Strategy(0))
        JuMinuit.hesse(m_num)
        cov_num = collect(m_num.covariance)
        @test cov_grad ≈ cov_num atol = 1e-8
    end
end
