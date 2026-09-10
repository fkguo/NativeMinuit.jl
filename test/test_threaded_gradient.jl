# SPDX-License-Identifier: LGPL-2.1-or-later

using Logging   # @test_logs min_level=Logging.Warn in the :auto tests

# Phase H test helper — module-level shared mutable scratch buffer
# (mimics the IAM `const c_00_4 = zeros(...)` anti-pattern). Needs to be
# module-level (not inside the @testset's `if Threads.nthreads()>1` block)
# because Julia doesn't allow `const` declarations inside local scopes.
const _PHASE_H_RACEY_BUF = zeros(Float64, 3)
function _PHASE_H_RACEY_CHI2(par)
    _PHASE_H_RACEY_BUF[1] = par[1]^2
    _PHASE_H_RACEY_BUF[2] = par[2]^2
    _PHASE_H_RACEY_BUF[3] = par[1] * par[2]
    sleep(0.0001)  # widen race window for reliable detection
    return _PHASE_H_RACEY_BUF[1] + _PHASE_H_RACEY_BUF[2] + 2 * _PHASE_H_RACEY_BUF[3]
end

# Thread-UNSAFE FCN with a NON-degenerate minimum, for the `:auto` correctness
# test: a shared module-level buffer (the classic anti-pattern) raced by the
# threaded gradient. Serial minimum is par[i] = i (χ² = 0); under a real race
# the threaded gradient disagrees, so `is_thread_safe` returns false and
# `:auto` must fall back to the serial gradient and STILL find par[i] = i.
const _AUTO_RACEY_BUF = zeros(Float64, 6)
function _AUTO_RACEY_CHI2(par)
    @inbounds for i in eachindex(par)
        _AUTO_RACEY_BUF[i] = par[i] - Float64(i)
    end
    sleep(0.0002)  # widen the race window so concurrent calls corrupt the buffer
    s = 0.0
    @inbounds for i in eachindex(par)
        s += _AUTO_RACEY_BUF[i]^2
    end
    return s
end

# Ground truth for the call-counter exactness test at the bottom of this file:
# the FCN counts its own invocations with an ATOMIC, so the tally is exact under
# any thread interleaving, and the fit's `m.nfcn` must equal it. n = 300 makes
# the per-coordinate tasks of the threaded gradient long enough that concurrent
# `+= 1` on a plain Ref lost increments on every measured run at 4 and 8 threads
# (and on most runs at 2).
const _NFCN_TRUTH = Threads.Atomic{Int}(0)
const _NFCN_TARGET = collect(range(-1.0, 1.0; length = 300))
function _NFCN_COUNTED_CHI2(x)
    Threads.atomic_add!(_NFCN_TRUTH, 1)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += (x[i] - _NFCN_TARGET[i])^2 * (1 + 0.1i)
    end
    return s
end

@testset "numerical_gradient! — Phase 2.2 threaded path" begin

    @testset "Threaded result matches serial — Quad-4D" begin
        cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]))
        par = MinimumParameters([0.5, 1.5, 2.5, 3.5], [0.1, 0.1, 0.1, 0.1], cf([0.5, 1.5, 2.5, 3.5]))
        prev = NativeMinuit.initial_gradient(par, par.dirin, cf)
        strategy = Strategy(0)

        # Serial
        out_serial = FunctionGradient(zeros(4), zeros(4), zeros(4))
        x_work_serial = similar(par.x)
        numerical_gradient!(out_serial, x_work_serial, par, prev, cf, strategy;
                             threaded = false)

        # Threaded — same FCN, fresh state
        cf2 = CostFunction(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]))
        par2 = MinimumParameters(copy(par.x), copy(par.dirin), par.fval)
        prev2 = NativeMinuit.initial_gradient(par2, par2.dirin, cf2)
        out_threaded = FunctionGradient(zeros(4), zeros(4), zeros(4))
        x_work_threaded = similar(par2.x)
        numerical_gradient!(out_threaded, x_work_threaded, par2, prev2, cf2, strategy;
                             threaded = true)

        # Threaded result should match serial to bit precision (same FCN
        # evaluations, just dispatched across threads).
        for i in 1:4
            @test out_threaded.grad[i] ≈ out_serial.grad[i] atol = 1e-12
            @test out_threaded.g2[i] ≈ out_serial.g2[i] atol = 1e-12
            @test out_threaded.gstep[i] ≈ out_serial.gstep[i] atol = 1e-12
        end
    end

    @testset "Threaded falls back to serial when nthreads == 1" begin
        # If Threads.nthreads() == 1, the threaded path should still
        # work (just serially in effect). Verifies no crash on
        # single-thread systems.
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], [0.1, 0.1], cf([1.0, 2.0]))
        prev = NativeMinuit.initial_gradient(par, par.dirin, cf)
        out = FunctionGradient(zeros(2), zeros(2), zeros(2))
        # threaded=true; if nthreads==1 we skip the threaded branch
        numerical_gradient!(out, similar(par.x), par, prev, cf, Strategy(0);
                             threaded = true)
        @test out.grad[1] ≈ 2.0 atol = 1e-6
        @test out.grad[2] ≈ 4.0 atol = 1e-6
    end

    @testset "Phase H — is_thread_safe + verify_threading auto-check" begin
        # Thread-safe FCN — pure function, no shared mutable state
        cf_safe = CostFunction(x -> sum(abs2, x), 1.0)
        @test is_thread_safe(cf_safe, [1.0, 2.0, 3.0]) === true

        # `migrad(..., threaded_gradient=true, verify_threading=true)`
        # passes silently for a safe FCN
        fmin = migrad(cf_safe, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1];
                       threaded_gradient = true, verify_threading = true)
        @test fmin.is_valid
        # Convergence to origin (within tlr)
        @test all(abs.(fmin.state.parameters.x) .< 0.01)

        # Bypass switch — verify_threading=false skips the check
        fmin2 = migrad(cf_safe, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1];
                        threaded_gradient = true, verify_threading = false)
        @test fmin2.is_valid

        # threaded_gradient=false → verify auto-skips (no work to verify)
        fmin3 = migrad(cf_safe, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1];
                        threaded_gradient = false)
        @test fmin3.is_valid

        # Minuit high-level API also wires verify_threading + defaults
        # it true when threading is on
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0, 3.0];
                    error = [0.1, 0.1, 0.1], threaded_gradient = true)
        @test m.verify_threading === true   # auto-defaulted to threaded value
        migrad!(m)
        @test m.fmin.internal.is_valid

        # Thread-unsafe FCN exhibits race ONLY when nthreads > 1.
        # The is_thread_safe API short-circuits to `true` on single-thread
        # Julia (nothing to race with), so this test only exercises the
        # full machinery when run via `julia -t N>1`.
        if Threads.nthreads() > 1
            cf_racey = CostFunction(_PHASE_H_RACEY_CHI2, 1.0)
            n = 8  # n >= nthreads so all threads have work
            @test is_thread_safe(cf_racey, fill(1.5, n)) === false
            @test_throws ThreadSafetyError migrad(
                cf_racey, fill(1.5, n), fill(0.1, n);
                threaded_gradient = true, verify_threading = true)
        end
    end
end

@testset "threaded_gradient = :auto (3-way mode)" begin

    # ── Validation + field storage (any nthreads) ──────────────────────────
    @testset "validation + construction" begin
        m = Minuit(x -> sum(abs2, x), [1.0, 2.0]; error = [0.1, 0.1],
                    threaded_gradient = :auto)
        @test m.threaded_gradient === :auto
        @test m.verify_threading === false           # :auto never uses the strict throw
        @test m._auto_threads[] === nothing          # not probed at construction
        @test Minuit(x -> sum(abs2, x), [1.0]; threaded_gradient = true).verify_threading === true
        @test Minuit(x -> sum(abs2, x), [1.0]; threaded_gradient = false).threaded_gradient === false
        @test_throws ArgumentError Minuit(x -> sum(abs2, x), [1.0];
                                           threaded_gradient = :sometimes)
    end

    # ── nthreads == 1: serial, no probe, no warning ────────────────────────
    if Threads.nthreads() == 1
        @testset ":auto on single-thread Julia — serial, no probe" begin
            m = Minuit(x -> sum(abs2, x .- 1.0), zeros(3); error = fill(0.1, 3),
                        threaded_gradient = :auto)
            @test NativeMinuit._use_threads(m) === false
            @test m._auto_threads[] === nothing       # never probed
            @test_logs min_level = Logging.Warn migrad!(m)   # no warnings
            @test m.fmin !== nothing
            @test all(abs.(m.fmin.ext_values .- 1.0) .< 1e-3)
        end
    end

    # ── nthreads > 1: the real machinery ───────────────────────────────────
    if Threads.nthreads() > 1
        @testset ":auto + safe FCN → threads, bit-identical to serial" begin
            target = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            f = x -> sum(abs2, x .- target)
            m_auto = Minuit(f, zeros(6); error = fill(0.1, 6),
                             threaded_gradient = :auto)
            @test NativeMinuit._use_threads(m_auto) === true   # safe → selects threading
            @test m_auto._auto_threads[] === true          # and memoizes the result
            migrad!(m_auto)

            m_false = Minuit(f, zeros(6); error = fill(0.1, 6),
                              threaded_gradient = false)
            migrad!(m_false)

            # :auto (threaded) fit is bit-identical to the serial fit
            @test m_auto.fmin.ext_values == m_false.fmin.ext_values
            @test fval(m_auto.fmin.internal) == fval(m_false.fmin.internal)
            @test all(abs.(m_auto.fmin.ext_values .- target) .< 1e-3)
        end

        @testset ":auto + UNSAFE FCN → warn + serial result (correct, no throw)" begin
            m_auto = Minuit(_AUTO_RACEY_CHI2, zeros(6); error = fill(0.1, 6),
                             threaded_gradient = :auto)
            # KEY correctness test: warns once, does NOT throw, and returns the
            # CORRECT serial minimum (par[i] = i), not the racy wrong one.
            @test_logs (:warn, r"not thread-safe") match_mode = :any migrad!(m_auto)
            @test m_auto._auto_threads[] === false       # probed, found unsafe
            @test m_auto.fmin !== nothing
            @test all(i -> abs(m_auto.fmin.ext_values[i] - Float64(i)) < 1e-3, 1:6)

            m_false = Minuit(_AUTO_RACEY_CHI2, zeros(6); error = fill(0.1, 6),
                              threaded_gradient = false)
            migrad!(m_false)
            @test m_auto.fmin.ext_values ≈ m_false.fmin.ext_values atol = 1e-8
        end

        @testset ":auto probe is memoized — runs once, not per call" begin
            f = x -> sum(abs2, x .- 1.0)
            m = Minuit(f, zeros(4); error = fill(0.1, 4), threaded_gradient = :auto)
            @test NativeMinuit._use_threads(m) === true       # first call probes
            @test m._auto_threads[] === true
            # Poison the cache: a re-probe of this SAFE FCN would return true,
            # so getting back the poisoned `false` proves no re-probe happened.
            m._auto_threads[] = false
            @test NativeMinuit._use_threads(m) === false
            # Repeated migrad! passes reuse the cache (no re-probe / no throw).
            m._auto_threads[] = nothing
            migrad!(m)
            cached = m._auto_threads[]
            @test cached === true
            migrad!(m)                                     # second fit
            @test m._auto_threads[] === cached             # unchanged → not re-probed
        end

        @testset "back-compat: true still throws on an unsafe FCN (Minuit API)" begin
            m_true = Minuit(_AUTO_RACEY_CHI2, fill(1.5, 6); error = fill(0.1, 6),
                             threaded_gradient = true)
            @test_throws ThreadSafetyError migrad!(m_true)
        end
    end
end

@testset "FCN call counter is exact under the threaded gradient" begin
    # `cf.nfcn` / `cf.n_nonfinite` are plain (non-atomic) Refs. Up to v0.7.2 the
    # threaded gradient loop incremented them from every worker thread through
    # the counting call operator and LOST increments — on this exact problem
    # (Julia 1.12, `julia -t 8`): m.nfcn = 49514 against 49653 true calls —
    # while the numerics stayed bit-identical to the serial fit. The threaded
    # branch now tallies its calls per coordinate and folds them into the
    # shared counters after the parallel loop. This pins `m.nfcn == true call
    # count` on both paths, plus threaded/serial `fval` bit-identity.
    #
    # Single-threaded Julia: `numerical_gradient!` gates its parallel loop on
    # `Threads.nthreads() > 1`, so `threaded_gradient = true` runs the serial
    # branch and the "threaded" half below is NOT discriminating there — it
    # only re-checks the serial path and passes trivially. The race is
    # exercised under `julia -t N>1` (CI runs the suite with 4 threads).
    n = length(_NFCN_TARGET)
    function fit_counted(threaded)
        _NFCN_TRUTH[] = 0
        m = Minuit(_NFCN_COUNTED_CHI2, zeros(n); errors = fill(0.1, n),
                   threaded_gradient = threaded)
        migrad!(m)
        return (nfcn = m.nfcn, truth = _NFCN_TRUTH[], fval = m.fval, valid = m.valid)
    end

    serial = fit_counted(false)
    @test serial.valid
    @test serial.nfcn == serial.truth

    threaded = fit_counted(true)      # verify_threading defaults to true: its
    @test threaded.valid              # probe calls are counted on both sides
    @test threaded.nfcn == threaded.truth
    @test threaded.fval == serial.fval          # numerics untouched, bit-identical
end
