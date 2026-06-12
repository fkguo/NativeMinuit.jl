# SPDX-License-Identifier: LGPL-2.1-or-later

# Write-back parameter-view tests — indexed assignment through
# `m.values` / `m.errors` / `m.fixed` / `m.limits` (iminuit
# ValueView/LimitView parity) plus the one-sided limit setters
# `set_upper_limit!` / `set_lower_limit!` (C++ `MnUserParameters::
# SetUpperLimit` / `SetLowerLimit`).
#
# Closes the silent-no-op bug: `m.fixed["x"] = true` used to mutate a
# throwaway copy of the property and was lost. The views now route
# writes through the PR #2 per-parameter mutators, so the canonical
# iminuit profile-scan idiom works in place.

@testset "Write-back parameter views + one-sided limits" begin

    # Quadratic FCN, minimum at (1, 2). Parameters named "x", "y".
    quad2() = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                      names = ["x", "y"], errors = [0.1, 0.1])

    @testset "Read compatibility (views behave like the old Vectors)" begin
        m = quad2()
        # AbstractVector, not a concrete Vector (the deliberate type change).
        @test m.values isa AbstractVector{Float64}
        @test m.errors isa AbstractVector{Float64}
        @test m.fixed isa AbstractVector{Bool}
        @test eltype(m.limits) == Tuple{Float64,Float64}
        @test length(m.values) == 2
        @test size(m.errors) == (2,)

        # Equality against a plain Vector, both directions.
        @test m.values == [0.0, 0.0]
        @test [0.0, 0.0] == m.values
        @test m.errors == [0.1, 0.1]
        @test m.fixed == [false, false]
        @test [false, false] == m.fixed

        # collect / copy produce a plain Array (decoupled from m).
        @test collect(m.errors) == [0.1, 0.1]
        @test collect(m.errors) isa Vector{Float64}
        cp = copy(m.values)
        @test cp isa Vector{Float64}
        @test cp == [0.0, 0.0]

        # Broadcast lowers to a plain Array.
        @test (m.values .+ 1.0) == [1.0, 1.0]
        @test (m.errors .* 2) == [0.2, 0.2]
        @test (m.values .+ 1.0) isa Vector{Float64}

        # Iteration.
        @test [v for v in m.values] == [0.0, 0.0]
        @test sum(m.errors) ≈ 0.2

        # Element read by Int and by name.
        @test m.values[1] == 0.0
        @test m.values["y"] == 0.0
        @test m.errors["x"] == 0.1
        @test m.fixed["x"] == false
        @test m.limits["x"] isa Tuple{Float64,Float64}
    end

    @testset "show does not leak the wrapped Minuit" begin
        m = quad2()
        # Two-arg show matches a plain Vector's compact form.
        @test repr(m.values) == repr([0.0, 0.0])
        # MIME show: summary names the view type, never dumps the object.
        io = IOBuffer()
        show(io, MIME"text/plain"(), m.fixed)
        out = String(take!(io))
        @test occursin("ParameterView", out)
        @test !occursin("Minuit(", out)      # no constructor-style dump
        @test !occursin("CostFunction", out)
    end

    @testset "m.fixed[k] = bool — the silent-no-op bug fix" begin
        m = quad2()
        # By name: the canonical iminuit idiom that used to be lost.
        m.fixed["x"] = true
        @test m.params.pars[1].fixed == true
        @test m.fixed == [true, false]
        # By index: release.
        m.fixed[1] = false
        @test m.params.pars[1].fixed == false
        @test m.fixed == [false, false]
        # Round-trip by name.
        m.fixed["y"] = true
        @test m.fixed[2]
        m.fixed["y"] = false
        @test !m.fixed[2]
    end

    @testset "m.values[k] / m.errors[k] write-back (Int and String)" begin
        m = quad2()
        m.values["x"] = 1.5
        @test m.values[1] == 1.5
        @test m.params.pars[1].value == 1.5
        m.values[2] = -3.0
        @test m.values["y"] == -3.0

        m.errors["x"] = 0.5
        @test m.errors[1] == 0.5
        m.errors[2] = 0.25
        @test m.params.pars[2].error == 0.25

        # Whole-vector assignment still routes through the setproperty!
        # bulk path (unchanged by the views).
        m.values = [4.0, 5.0]
        @test m.values == [4.0, 5.0]
        m.errors = [0.7, 0.8]
        @test m.errors == [0.7, 0.8]

        # set_value! validation still applies through the view.
        @test_throws ArgumentError (m.values["x"] = NaN)
        @test_throws ArgumentError (m.errors[1] = -0.1)
    end

    @testset "m.limits[k] = one-sided tuples / nothing" begin
        m = quad2()
        # Lower-only.
        m.limits["x"] = (0.0, nothing)
        @test has_lower_limit(m.params.pars[1])
        @test !has_upper_limit(m.params.pars[1])
        @test m.params.pars[1].lower == 0.0
        @test isnan(m.params.pars[1].upper)

        # Upper-only.
        m.limits["x"] = (nothing, 5.0)
        @test !has_lower_limit(m.params.pars[1])
        @test has_upper_limit(m.params.pars[1])
        @test m.params.pars[1].upper == 5.0

        # Both.
        m.limits["x"] = (-1.0, 5.0)
        @test has_lower_limit(m.params.pars[1])
        @test has_upper_limit(m.params.pars[1])

        # Remove via nothing.
        m.limits["x"] = nothing
        @test !has_limits(m.params.pars[1])

        # By index, and read-back as a tuple.
        m.limits[2] = (0.0, 10.0)
        @test m.params.pars[2].lower == 0.0
        @test m.params.pars[2].upper == 10.0
        @test m.limits[2] == (0.0, 10.0)

        # ±Inf normalizes to "absent" like the explicit setters.
        m.limits[2] = (-Inf, 4.0)
        @test !has_lower_limit(m.params.pars[2])
        @test m.params.pars[2].upper == 4.0

        # Invalid range still rejected (delegated to MinuitParameter ctor).
        @test_throws ArgumentError (m.limits[2] = (5.0, 0.0))
        @test_throws ArgumentError (m.limits[2] = (1.0, 1.0))

        # A scalar (fat-fingered, missing the tuple) gives a clear error,
        # not a cryptic destructuring BoundsError.
        @test_throws ArgumentError (m.limits[2] = 5.0)
        @test_throws ArgumentError (m.limits["x"] = (1.0, 2.0, 3.0))
    end

    @testset "set_upper_limit! / set_lower_limit! — C++ one-sided semantics" begin
        # C++ SetUpperLimit sets the upper bound AND clears the lower
        # (fLoLimValid=false). Start two-sided to observe the clear.
        m = quad2()
        set_limits!(m, "x", -10.0, 10.0)
        @test has_lower_limit(m.params.pars[1]) && has_upper_limit(m.params.pars[1])
        @test set_upper_limit!(m, "x", 3.0) === m       # returns m (chains)
        @test !has_lower_limit(m.params.pars[1])         # lower cleared
        @test has_upper_limit(m.params.pars[1])
        @test m.params.pars[1].upper == 3.0

        # SetLowerLimit sets lower AND clears upper.
        set_limits!(m, "y", -10.0, 10.0)
        set_lower_limit!(m, 2, -2.0)                     # by index
        @test has_lower_limit(m.params.pars[2])
        @test !has_upper_limit(m.params.pars[2])
        @test m.params.pars[2].lower == -2.0

        # By index and by name both resolve.
        m2 = quad2()
        set_upper_limit!(m2, 1, 4.0)
        @test m2.params.pars[1].upper == 4.0

        # Finiteness validation (a one-sided Inf bound is a footgun —
        # use remove_limits! to drop a bound instead).
        @test_throws ArgumentError set_upper_limit!(m2, 1, NaN)
        @test_throws ArgumentError set_upper_limit!(m2, 1, Inf)
        @test_throws ArgumentError set_lower_limit!(m2, 1, -Inf)

        # Index / name error paths (shared _check_par_index / ext_index).
        @test_throws BoundsError set_upper_limit!(m2, 99, 1.0)
        @test_throws BoundsError set_lower_limit!(m2, 0, 1.0)
        @test_throws KeyError set_lower_limit!(m2, "nope", 1.0)

        # An upper-only bound actually constrains migrad.
        mb = Minuit(x -> (x[1] - 10.0)^2, [0.0]; names = ["a"], errors = [0.1])
        set_upper_limit!(mb, "a", 3.0)
        migrad!(mb)
        @test mb.values[1] ≈ 3.0 atol = 1e-3
    end

    @testset "View index error paths" begin
        m = quad2()
        @test_throws BoundsError m.values[99]
        @test_throws BoundsError m.fixed[0]
        @test_throws BoundsError (m.values[99] = 1.0)
        @test_throws KeyError m.values["nope"]
        @test_throws KeyError (m.fixed["nope"] = true)
        @test_throws KeyError (m.limits["nope"] = (0.0, 1.0))
    end

    @testset "View writes invalidate the cached fit (PR #2 staleness rule)" begin
        m = quad2()
        migrad!(m)
        minos!(m, 1)
        @test m.valid
        @test haskey(m.minos_errors, 1)
        # Writing through a view drops fmin + minos, like the mutators.
        m.values["x"] = 0.5
        @test m.fmin === nothing
        @test isempty(m.minos_errors)

        for mutate! in (m -> (m.fixed[1] = true),
                        m -> (m.errors[1] = 0.9),
                        m -> (m.limits["x"] = (0.0, 5.0)),
                        m -> (set_upper_limit!(m, "x", 5.0)))
            mm = quad2()
            migrad!(mm)
            @test mm.valid
            mutate!(mm)
            @test mm.fmin === nothing
        end
    end

    @testset "Reads reflect post-fit values, then fall back after a write" begin
        m = quad2()
        migrad!(m)
        @test m.valid
        # Post-fit reads come from fmin.ext_values.
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
        @test m.values == m.fmin.ext_values
        @test m.errors == m.fmin.ext_errors
        # A view write resets fmin; reads now reflect the new initial value.
        m.values["x"] = 7.0
        @test m.fmin === nothing
        @test m.values[1] == 7.0
    end

    @testset "Profile-scan idiom end-to-end (the canonical iminuit pattern)" begin
        # Fix a nuisance param off-minimum, fit, observe the constrained
        # minimum; release, refit, observe the fit CHANGE — all through
        # indexed views. This is the line the silent-no-op bug broke.
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["x", "y"], errors = [0.1, 0.1])
        m.fixed["y"] = true
        m.values["y"] = 0.0          # held off the true minimum (y* = 2)
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] == 0.0      # y held at the scan point
        fixed_fval = m.fval
        @test fixed_fval ≈ 4.0 atol = 1e-3   # (1-1)² + (0-2)²

        m.fixed["y"] = false
        migrad!(m)
        @test m.valid
        @test m.values[2] ≈ 2.0 atol = 1e-4   # y now floats to the minimum
        @test m.fval < fixed_fval             # the fit improved → observable
    end

    @testset "issue #38: m.params reflects the fit (parity with m.values)" begin
        m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["a", "b"], errors = [0.1, 0.1],
                    limits = [(-5.0, 5.0), nothing])

        # Before any fit, `m.params` is the constructor-time config.
        @test m.params.pars[1].value == 0.0
        @test m.params.pars[2].value == 0.0
        @test m.params.pars[1].error == 0.1

        migrad!(m)
        minos!(m)
        @test m.valid

        # The whole bug: `m.params.pars[i]` must show the FITTED value/error,
        # exactly agreeing with the `m.values` / `m.errors` views.
        for i in 1:2
            @test m.params.pars[i].value == m.values[i]
            @test m.params.pars[i].error == m.errors[i]
        end
        @test m.params.pars[1].value > 0.9               # not the 0.0 initial
        @test m.params.pars[2].value ≈ 2.0 atol = 1e-6

        # Structure (names, bounds, fixed flags, index maps) is untouched.
        @test m.params.pars[1].name == "a"
        @test (m.params.pars[1].lower, m.params.pars[1].upper) == (-5.0, 5.0)
        @test isnan(m.params.pars[2].lower) && isnan(m.params.pars[2].upper)
        @test all(!p.fixed for p in m.params.pars)
        @test m.params.ext_of_int == [1, 2]

        # A fixed parameter overlays its (unmoved) value and keeps the flag.
        mf = quad2()
        mf.fixed["y"] = true
        mf.values["y"] = 0.5
        migrad!(mf)
        @test mf.params.pars[2].fixed
        @test mf.params.pars[2].value == mf.values[2] == 0.5

        # `reset(m)` drops the fit → `m.params` returns to the initial config.
        reset(m)
        @test m.fmin === nothing
        @test m.params.pars[1].value == 0.0
        @test m.params.pars[1].error == 0.1

        # A re-`migrad!` after a fit must still seed from the user's original
        # step (the retry length scale reads the raw config, not the overlay):
        # converges to the same optimum, byte-equal to a cold fit.
        m2 = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                     names = ["a", "b"], errors = [0.1, 0.1],
                     limits = [(-5.0, 5.0), nothing])
        migrad!(m2)
        v_cold = collect(m2.values)
        migrad!(m2)               # resume: must not drift
        @test collect(m2.values) ≈ v_cold atol = 1e-8
    end
end
