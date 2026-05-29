# SPDX-License-Identifier: LGPL-2.1-or-later

# Per-parameter mutator tests (gap M3) — `fix!`, `release!`, `set_value!`,
# `set_error!`, `set_limits!`, `remove_limits!`. Mirrors the surface of
# C++ `MnUserParameters` (`reference/Minuit2_cpp/inc/Minuit2/MnUserParameters.h:75-95`)
# and validates the iminuit-idiomatic profile-likelihood pattern.

@testset "Per-parameter mutators (gap M3)" begin

    # Quadratic FCN used across this file: minimum at (1, 2) for the
    # unconstrained two-parameter case, fval ≈ 0. Default parameter
    # names are "x0", "x1" (iminuit convention).
    quad2() = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                      errors = [0.1, 0.1])

    @testset "fix! / release! — by index and by name" begin
        m = quad2()
        @test m.fixed == [false, false]

        fix!(m, 1)
        @test m.fixed == [true, false]
        @test m.params.pars[1].fixed
        @test !m.params.pars[2].fixed

        release!(m, 1)
        @test m.fixed == [false, false]

        # By name (default names are "x0", "x1")
        fix!(m, "x0")
        @test m.fixed[1] && !m.fixed[2]

        release!(m, "x0")
        @test m.fixed == [false, false]

        # Chaining: each mutator returns m.
        @test fix!(m, 2) === m
        @test release!(m, 2) === m
    end

    @testset "fix! drops fmin and clears minos_errors" begin
        m = quad2()
        migrad!(m)
        @test m.valid
        minos!(m, 1)
        @test haskey(m.minos_errors, 1)

        fix!(m, 1)
        @test m.fmin === nothing
        @test isempty(m.minos_errors)
    end

    @testset "release! also drops fmin / minos_errors" begin
        m = quad2()
        fix!(m, 1)
        migrad!(m)
        @test m.valid
        release!(m, 1)
        @test m.fmin === nothing
        @test isempty(m.minos_errors)
    end

    @testset "Round-trip: fix → migrad → release → migrad → values change" begin
        m = quad2()
        # Fix x1=0 (off-minimum). First migrad converges on x0 only:
        # x0 → 1.0, x1 stays at 0.0.
        fix!(m, 2)
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] == 0.0
        # Release x1 and re-migrad: both float, converge to (1, 2).
        release!(m, 2)
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
        # The values DID change — round-trip is observable.
        @test m.values[2] != 0.0
    end

    @testset "Profile-likelihood scan idiom (by name)" begin
        # Canonical iminuit pattern: fix a nuisance parameter to scan
        # its conditional minimum, then release and refit.
        m = quad2()
        fix!(m, "x1")
        set_value!(m, "x1", 1.5)   # scan point
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] == 1.5    # held at scan point

        release!(m, "x1")
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
    end

    @testset "set_value! — by index and by name" begin
        m = quad2()
        set_value!(m, 1, 3.0)
        @test m.values[1] == 3.0
        @test m.params.pars[1].value == 3.0
        # Other params untouched.
        @test m.values[2] == 0.0

        set_value!(m, "x1", -4.5)
        @test m.values[2] == -4.5

        # After migrad, set_value drops fmin.
        migrad!(m)
        @test m.valid
        set_value!(m, 1, 2.0)
        @test m.fmin === nothing
        @test m.values[1] == 2.0   # falls through to params.pars[1].value
    end

    @testset "set_error! — by index and by name" begin
        m = quad2()
        set_error!(m, 1, 0.5)
        @test m.errors[1] == 0.5
        @test m.params.pars[1].error == 0.5
        @test m.errors[2] == 0.1

        set_error!(m, "x1", 0.7)
        @test m.params.pars[2].error == 0.7
    end

    @testset "set_limits! — followed by migrad respects bound" begin
        # FCN minimum at x[1]=10. Constrain x[1] ∈ [0, 5]: post-migrad
        # x[1] sits at the upper bound, NOT at the unconstrained minimum.
        m = Minuit(x -> (x[1] - 10.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    errors = [0.1, 0.1])
        set_limits!(m, 1, 0.0, 5.0)
        @test m.params.pars[1].lower == 0.0
        @test m.params.pars[1].upper == 5.0
        migrad!(m)
        @test m.valid
        @test m.values[1] <= 5.0 + 1e-6
        @test m.values[1] ≈ 5.0 atol = 1e-3   # saturated at the bound
        # x2 is unconstrained, but the EDM stop — with x1 pinned at the bound,
        # where its Sin-transform internal gradient → 0 — leaves x2 ~3.8e-4
        # short of 2.0. iminuit 2.32.0 (Strategy 1) lands at the IDENTICAL
        # value 2.0003789700 (verified). The old atol=1e-4 was tuned to
        # JuMinuit's pre-fix bare-eps over-tightness (audit §14 fix
        # feat/precision-eps-x4 makes eps2 match C++/iminuit).
        @test m.values[2] ≈ 2.0 atol = 1e-3   # ≡ iminuit's 2.00037897

        # One-sided via nothing.
        m2 = Minuit(x -> (x[1] - 10.0)^2, [0.0]; errors = [0.1])
        set_limits!(m2, 1, nothing, 3.0)
        @test isnan(m2.params.pars[1].lower)
        @test m2.params.pars[1].upper == 3.0
        migrad!(m2)
        @test m2.values[1] ≈ 3.0 atol = 1e-3

        # By name.
        m3 = quad2()
        set_limits!(m3, "x0", -1.0, 0.5)
        @test m3.params.pars[1].lower == -1.0
        @test m3.params.pars[1].upper == 0.5

        # Inf is normalized to "absent bound" (NaN sentinel).
        m4 = quad2()
        set_limits!(m4, 1, -Inf, 5.0)
        @test isnan(m4.params.pars[1].lower)
        @test m4.params.pars[1].upper == 5.0

        # Invalid range raises (delegated to MinuitParameter ctor).
        m5 = quad2()
        @test_throws ArgumentError set_limits!(m5, 1, 5.0, 0.0)
        @test_throws ArgumentError set_limits!(m5, 1, 1.0, 1.0)
    end

    @testset "remove_limits! — clears both bounds" begin
        m = Minuit(x -> (x[1] - 10.0)^2, [0.5]; errors = [0.1],
                    limits = [(0.0, 5.0)])
        @test m.params.pars[1].lower == 0.0
        @test m.params.pars[1].upper == 5.0
        # Confirm bounded migrad first saturates at the upper bound.
        migrad!(m)
        @test m.values[1] ≈ 5.0 atol = 1e-3

        remove_limits!(m, 1)
        @test isnan(m.params.pars[1].lower)
        @test isnan(m.params.pars[1].upper)
        @test m.fmin === nothing   # cache dropped

        # Re-migrad now finds the unconstrained minimum.
        migrad!(m)
        @test m.values[1] ≈ 10.0 atol = 1e-3

        # By name.
        m2 = Minuit(x -> sum(abs2, x), [1.0, 2.0];
                     limits = [(-5.0, 5.0), (-5.0, 5.0)])
        remove_limits!(m2, "x0")
        @test isnan(m2.params.pars[1].lower)
        @test isnan(m2.params.pars[1].upper)
        # x1 bounds untouched.
        @test m2.params.pars[2].lower == -5.0
        @test m2.params.pars[2].upper == 5.0
    end

    @testset "Bounds checking — out-of-range index / unknown name" begin
        m = quad2()
        @test_throws BoundsError fix!(m, 0)
        @test_throws BoundsError fix!(m, 99)
        @test_throws BoundsError release!(m, -1)
        @test_throws BoundsError set_value!(m, 3, 1.0)
        @test_throws BoundsError set_error!(m, 100, 0.1)
        @test_throws BoundsError set_limits!(m, 5, 0.0, 1.0)
        @test_throws BoundsError remove_limits!(m, 99)

        # Unknown name → KeyError (from ext_index).
        @test_throws KeyError fix!(m, "nonexistent")
        @test_throws KeyError set_value!(m, "missing", 1.0)

        # Bool is <: Integer in Julia — explicitly rejected so users
        # get a clear diagnostic instead of a confusing BoundsError
        # silently triggered by `true` indexing into pars[1].
        @test_throws ArgumentError fix!(m, true)
        @test_throws ArgumentError set_value!(m, false, 1.0)
    end

    @testset "Value/error validation — NaN/Inf/negative" begin
        # set_value!: NaN and ±Inf are footguns — they "converge"
        # silently in migrad. Reject like iminuit's Python wrapper.
        m = quad2()
        @test_throws ArgumentError set_value!(m, 1, NaN)
        @test_throws ArgumentError set_value!(m, 1, Inf)
        @test_throws ArgumentError set_value!(m, 1, -Inf)
        @test_throws ArgumentError set_value!(m, "x0", NaN)

        # set_error!: NaN, ±Inf, and negative all rejected. Zero is
        # allowed (fixed parameters have error == 0).
        @test_throws ArgumentError set_error!(m, 1, NaN)
        @test_throws ArgumentError set_error!(m, 1, Inf)
        @test_throws ArgumentError set_error!(m, 1, -0.1)
        @test_throws ArgumentError set_error!(m, "x1", -1.0)
        set_error!(m, 1, 0.0)   # zero is allowed
        @test m.params.pars[1].error == 0.0
    end

    @testset "Bulk setters route through per-parameter mutators" begin
        # Confirm the existing bulk-setter semantics still hold after
        # the refactor (`m.values=...` etc. share the per-parameter
        # `_build_*_par` validation helpers).
        m = quad2()
        migrad!(m)
        @test m.valid

        # m.values = [...]
        m.values = [3.0, 4.0]
        @test m.fmin === nothing
        @test m.values == [3.0, 4.0]

        # m.errors = [...]
        m.errors = [0.3, 0.4]
        @test m.errors == [0.3, 0.4]

        # m.fixed = [...]
        m.fixed = [true, false]
        @test m.fixed == [true, false]
        m.fixed = [false, false]

        # m.limits = [...]
        m.limits = [(0.0, 10.0), nothing]
        @test m.params.pars[1].lower == 0.0
        @test m.params.pars[1].upper == 10.0
        @test isnan(m.params.pars[2].lower)
        @test isnan(m.params.pars[2].upper)

        # Bulk-setter length validation still throws.
        @test_throws DimensionMismatch (m.values = [1.0])
        @test_throws DimensionMismatch (m.errors = [0.1, 0.1, 0.1])
        @test_throws DimensionMismatch (m.fixed = [true])
        @test_throws DimensionMismatch (m.limits = [(0.0, 1.0)])
    end

    @testset "Bulk setters are exception-atomic" begin
        # Codex-review blocking finding: if a later element of a bulk
        # assignment fails validation, the EARLIER elements must NOT
        # already be committed, and `m.fmin` must NOT already be cleared
        # — `m` must look exactly as it did before the assignment.
        # Pre-M3 semantics were atomic because the old code built
        # `new_pars` fully and only committed via a single setfield!.
        # The refactor must preserve that.
        m = quad2()
        migrad!(m)
        @test m.valid
        old_values = copy(m.values)
        old_fmin = m.fmin

        # values: NaN in slot 2 must reject without modifying slot 1
        # or dropping fmin.
        @test_throws ArgumentError (m.values = [3.0, NaN])
        @test m.values == old_values
        @test m.fmin === old_fmin

        # errors: negative in slot 2.
        @test_throws ArgumentError (m.errors = [0.5, -0.1])
        @test m.errors == [m.fmin.ext_errors[1], m.fmin.ext_errors[2]]
        @test m.fmin === old_fmin

        # limits: invalid range (lo == up) in slot 2.
        @test_throws ArgumentError (m.limits = [(0.0, 10.0), (1.0, 1.0)])
        @test m.fmin === old_fmin
        # No bounds were applied — params still match the original
        # (no-limits) configuration.
        @test isnan(m.params.pars[1].lower)
        @test isnan(m.params.pars[1].upper)
        @test isnan(m.params.pars[2].lower)
        @test isnan(m.params.pars[2].upper)
    end

    @testset "Parameters internal index maps rebuilt after fix!/release!" begin
        # The `ext_of_int` / `int_of_ext` caches inside `Parameters`
        # must reflect the new free/fixed partition after every
        # fix!/release! — otherwise downstream (migrad, hesse, minos)
        # will index into the wrong slot.
        m = Minuit(x -> sum(abs2, x .- [1.0, 2.0, 3.0]), [0.0, 0.0, 0.0])
        @test n_free(m.params) == 3
        fix!(m, 2)
        @test n_free(m.params) == 2
        @test m.params.ext_of_int == [1, 3]   # the two free parameters
        @test m.params.int_of_ext[2] == 0      # x1 is fixed
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] == 0.0   # fixed
        @test m.values[3] ≈ 3.0 atol = 1e-4

        release!(m, 2)
        @test n_free(m.params) == 3
        @test m.params.ext_of_int == [1, 2, 3]
        migrad!(m)
        @test m.valid
        @test m.values ≈ [1.0, 2.0, 3.0] atol = 1e-4
    end
end
