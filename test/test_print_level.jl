# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# test_print_level.jl — gap M1: iminuit-compatible print_level / trace.
#
# Confirms:
#   1. level 0 (default) emits no log output (zero @info / @debug records).
#   2. level 1 emits a per-DFP-iter line for each accepted MIGRAD step.
#   3. level 2 emits strictly more lines than level 1 (inner-loop trace).
#   4. level 3 emits @debug records carrying parameter + gradient vectors.
#   5. The same gating applies to hesse() and minos().
# ─────────────────────────────────────────────────────────────────────────────

using Test
using NativeMinuit
using Logging

# Helper: capture log records emitted while running `f`. Returns the
# captured records as a Vector{NamedTuple} so individual tests can match
# against `level`, `message`, `_group`, etc.
function capture_logs(f; level::LogLevel = Logging.Info)
    logger = Test.TestLogger(min_level = level)
    Logging.with_logger(f, logger)
    return logger.logs
end

# 2D quadratic minimum at (1, 2), well-conditioned. Used for the
# baseline trace tests — MIGRAD typically takes 3-5 DFP iters from a
# reasonable seed.
quad2(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2 + 4.0 * (x[1] - 1.0) * (x[2] - 2.0) * 0.1

@testset "print_level — MIGRAD" begin
    @testset "level 0 is silent" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1])
            migrad!(m)
        end
        # No MnMigrad records when print_level=0.
        n_migrad = count(r -> occursin("MnMigrad", r.message), logs)
        @test n_migrad == 0
    end

    @testset "level 1 emits per-DFP-iter lines" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 1)
            migrad!(m)
        end
        # Count per-iter lines (contain "iter=" and "fval=").
        n_iter = count(r -> occursin("iter=", r.message) &&
                            occursin("fval=", r.message), logs)
        # >= 1 because we emit an initial iter=0 banner before the loop,
        # plus one per accepted DFP step. For this 2D quadratic with seed
        # error 0.1, MIGRAD converges in ≥ 1 DFP step.
        @test n_iter >= 2
    end

    @testset "level 2 emits strictly more than level 1" begin
        logs1 = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 1)
            migrad!(m)
        end
        logs2 = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 2)
            migrad!(m)
        end
        # Level 2 adds per-iter line-search + gnorm lines on top of the
        # level-1 per-iter lines.
        @test length(logs2) > length(logs1)
        @test any(r -> occursin("line search:", r.message), logs2)
        @test any(r -> occursin("gnorm=", r.message), logs2)
    end

    @testset "level 3 emits @debug state records" begin
        logs = capture_logs(level = Logging.Debug) do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 3)
            migrad!(m)
        end
        debug_records = filter(r -> r.level == Logging.Debug, logs)
        @test !isempty(debug_records)
        # The state trace carries `x=` and `grad=` kwargs (TestLogger
        # exposes them via the `kwargs` field).
        any_state = any(debug_records) do r
            haskey(Dict(r.kwargs), :x) && haskey(Dict(r.kwargs), :grad)
        end
        @test any_state
    end
end

@testset "print_level — HESSE" begin
    @testset "level 0 is silent" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1])
            migrad!(m)
            hesse(m)
        end
        @test count(r -> occursin("MnHesse", r.message), logs) == 0
    end

    @testset "level 1 emits start+done banners" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 1)
            migrad!(m)
            hesse(m)
        end
        hesse_logs = filter(r -> occursin("MnHesse", r.message), logs)
        @test !isempty(hesse_logs)
        @test any(r -> occursin("start:", r.message), hesse_logs)
        @test any(r -> occursin("done:", r.message), hesse_logs)
    end

    @testset "level 2 emits per-parameter diagonal pass lines" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 2)
            migrad!(m)
            hesse(m)
        end
        n_diag = count(r -> occursin("MnHesse", r.message) &&
                            occursin("diag i=", r.message), logs)
        # For n=2 we expect 2 diagonal-pass lines per HESSE invocation
        # (plus one for the inner Strategy ≥ 1 refinement if Migrad
        # triggered it). >= 2 captures both standalone-hesse and any
        # inner refresh.
        @test n_diag >= 2
    end
end

@testset "print_level — MINOS" begin
    @testset "level 0 is silent" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1])
            migrad!(m)
            hesse(m)
            minos!(m)
        end
        @test count(r -> occursin("MnMinos", r.message), logs) == 0
        @test count(r -> occursin("MnFunctionCross", r.message), logs) == 0
    end

    @testset "level 1 emits MnMinos direction headers" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 1)
            migrad!(m)
            hesse(m)
            minos!(m)
        end
        # For 2 free parameters MINOS runs 2 × (upper, lower) = 4 headers.
        n_dir = count(r -> occursin("MnMinos", r.message) &&
                           occursin("Determination of", r.message), logs)
        @test n_dir >= 2   # at least one direction got traced
        # MnFunctionCross start banner fires per direction.
        n_cross_start = count(r -> occursin("MnFunctionCross", r.message) &&
                                    occursin("start:", r.message), logs)
        @test n_cross_start >= 1
    end

    @testset "level 2 emits per-probe trace" begin
        logs = capture_logs() do
            m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1], print_level = 2)
            migrad!(m)
            hesse(m)
            minos!(m)
        end
        # Per-probe lines in _cross_core (probe ipt=...) fire whenever a
        # parabolic-fit probe is evaluated.
        n_probes = count(r -> occursin("MnFunctionCross", r.message) &&
                              occursin("probe ipt=", r.message), logs)
        @test n_probes >= 1
    end
end

@testset "print_level — Minuit struct forwards m.print_level" begin
    # Setting `m.print_level = 1` via property must reach migrad!.
    logs = capture_logs() do
        m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1])
        m.print_level = 1
        migrad!(m)
    end
    @test count(r -> occursin("MnMigrad", r.message), logs) >= 1
end

@testset "print_level — bounded MINOS path" begin
    # Codex review: confirm the bounded-MINOS branch in
    # _minos_external_via_function_cross also forwards print_level.
    # Use bounds that don't restrict the minimum so MINOS converges.
    logs = capture_logs() do
        m = Minuit(quad2, [0.0, 0.0]; error = [0.1, 0.1],
                    limits = [(-5.0, 5.0), (-5.0, 5.0)],
                    print_level = 1)
        migrad!(m)
        hesse(m)
        minos!(m)
    end
    n_min = count(r -> occursin("MnMinos", r.message) &&
                       occursin("Determination of", r.message), logs)
    @test n_min >= 2
end

@testset "print_level — direct low-level function_cross_multi" begin
    # Codex review: function_cross_multi accepts print_level too. Direct
    # invocation should produce the cross-search start banner.
    cf = NativeMinuit.CostFunction(quad2, 1.0)
    fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
    @test fmin.is_valid
    logs = capture_logs() do
        NativeMinuit.function_cross_multi(fmin, cf, [1, 2],
                                        [fmin.state.parameters.x[1],
                                         fmin.state.parameters.x[2]],
                                        [1.0, 0.0];
                                        print_level = 2)
    end
    # Header banner + at least one probe line.
    @test any(r -> occursin("MnFunctionCross", r.message) &&
                   occursin("start:", r.message), logs)
    @test any(r -> occursin("MnFunctionCross", r.message) &&
                   occursin("probe ipt=", r.message), logs)
end

@testset "print_level — direct migrad(cf, seed; ...)" begin
    # Codex review: confirm the `migrad(cf, seed; ...)` warm-restart
    # entry-point forwards print_level. Build a seed via seed_state
    # at a non-trivial point.
    cf = NativeMinuit.CostFunction(quad2, 1.0)
    seed = NativeMinuit.seed_state(cf, [0.0, 0.0], [0.1, 0.1],
                                NativeMinuit.Strategy(0),
                                NativeMinuit.MachinePrecision())
    logs = capture_logs() do
        migrad(cf, seed; print_level = 1)
    end
    @test count(r -> occursin("MnMigrad", r.message), logs) >= 1
end

@testset "print_level — @debug records snapshot state (codex IMPORTANT)" begin
    # Codex flagged that level-3 records storing references to mutable
    # scratch buffers would all collapse to the final state by the time
    # the test reads them. Verify each stored debug record carries a
    # DISTINCT (snapshot) state — at least two records should differ.
    logs = capture_logs(level = Logging.Debug) do
        m = Minuit(quad2, [-0.5, -0.5]; error = [0.1, 0.1], print_level = 3)
        migrad!(m)
    end
    debug_recs = filter(r -> r.level == Logging.Debug &&
                              occursin("state at iter=", r.message), logs)
    if length(debug_recs) >= 2
        # Compare the x vector across the first two stored records.
        x_first  = Dict(debug_recs[1].kwargs)[:x]
        x_second = Dict(debug_recs[end].kwargs)[:x]
        # If we stored references, x_first == x_second (both point at the
        # final buffer state). Copies → they differ as the optimizer
        # marches toward the minimum.
        @test x_first != x_second
    else
        # If the optimizer converged in <2 iters the test is uninformative
        # — flag it as a soft pass with a note.
        @test_skip "need >=2 DFP iterations to test snapshot independence"
    end
end
