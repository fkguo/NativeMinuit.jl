# SPDX-License-Identifier: LGPL-2.1-or-later

# Structured coordinate containers (issue #43). A `ComponentVector` is the
# reference implementation, but nothing here is ComponentArrays-specific: the
# contract is that any `AbstractVector{Float64}` whose `similar`/`copy`/
# broadcast preserve its structure survives to every point where NativeMinuit
# calls back into user code, and into the results the user reads.
#
# Two failure modes these tests are built to catch:
#
#   1. A path that quietly hands the user's objective a bare `Array` — the
#      original bug. Every check below COUNTS the calls that arrive with the
#      wrong container instead of latching a flag, so a single bad call among
#      hundreds still fails (a `saw_ok[] = ...` assignment only records the
#      LAST call and can pass while most calls are wrong).
#   2. Container preserved but numbers changed. Every structured fit is
#      compared against the identical fit run on a plain `Vector`.

using ComponentArrays
using Optim
using AdvancedHMC, LogDensityProblems, LogDensityProblemsAD, TransformVariables
using ForwardDiff

# Count callbacks that did NOT arrive with the expected axes. Returns
# (n_bad, n_total) so a test can also assert the path was exercised at all.
function _axis_probe(expected_axes)
    bad = Ref(0)
    tot = Ref(0)
    f = p -> begin
        tot[] += 1
        (p isa ComponentVector && ComponentArrays.getaxes(p) == expected_axes) ||
            (bad[] += 1)
        (p.a - 1.0)^2 + 3 * (p.b - 2.0)^2 + 0.5 * (p.a - 1.0) * (p.b - 2.0)
    end
    return f, bad, tot
end

# The same objective written against a plain vector, for numeric parity.
_flat_obj(p) = (p[1] - 1.0)^2 + 3 * (p[2] - 2.0)^2 + 0.5 * (p[1] - 1.0) * (p[2] - 2.0)
_struct_obj(p) = (p.a - 1.0)^2 + 3 * (p.b - 2.0)^2 + 0.5 * (p.a - 1.0) * (p.b - 2.0)

@testset "structured parameter containers" begin
    start = ComponentArray(a = 0.0, b = 0.0)
    expected_axes = ComponentArrays.getaxes(start)
    flat_start = [0.0, 0.0]
    errs = [0.1, 0.1]

    @testset "active vector is the low-level state" begin
        result = @inferred migrad(_struct_obj, start, errs)

        @test result.state.parameters.x isa ComponentVector
        @test result.state.gradient.grad isa ComponentVector
        @test ComponentArrays.getaxes(result.state.parameters.x) == expected_axes
        @test ComponentArrays.getaxes(result.state.gradient.grad) == expected_axes
        @test values(result).a ≈ 1.0 atol = 1e-4
        @test values(result).b ≈ 2.0 atol = 1e-4

        # Numeric parity with the flat run: same optimizer, same trajectory.
        flat = migrad(_flat_obj, flat_start, errs)
        @test collect(result.state.parameters.x) == collect(flat.state.parameters.x)
        @test result.state.parameters.fval == flat.state.parameters.fval
        @test result.state.nfcn == flat.state.nfcn
    end

    @testset "standalone HESSE errors keep coordinate axes" begin
        result = @inferred hesse(_struct_obj, start, errs)

        @test result.x isa ComponentVector
        @test result.errors isa ComponentVector
        @test typeof(result.errors) == typeof(result.x)
        @test result isa HesseResult{typeof(result.x),typeof(result.state)}
        @test ComponentArrays.getaxes(result.x) == expected_axes
        @test ComponentArrays.getaxes(result.errors) == expected_axes
        # Internal consistency of the returned fields (errors belong to
        # THIS covariance) …
        @test collect(result.errors) ≈
              sqrt.([result.covariance[i, i] for i in axes(result.covariance, 1)])
        # … and numeric parity with the identical flat run: the container
        # must not change a single bit of the numbers, and the flat run
        # must keep its dense container (the other half of the contract).
        flat = hesse(_flat_obj, flat_start, errs)
        @test flat.x isa Vector{Float64}
        @test flat.errors isa Vector{Float64}
        @test collect(result.x) == collect(flat.x)
        @test collect(result.errors) == collect(flat.errors)
        @test result.covariance == flat.covariance
        @test result.nfcn == flat.nfcn
    end

    @testset "standalone HESSE failure path keeps the container" begin
        # A flat objective has zero curvature everywhere: the diagonal
        # pass cannot grow the sag and HESSE fails. The failure result
        # flows through the same `HesseResult(state, up)` constructor, so
        # it must carry the caller's coordinate container too.
        result = hesse(p -> 0.0, start, errs)
        @test !result.valid
        @test result.status == MnHesseFailed
        @test result.errors isa ComponentVector
        @test typeof(result.errors) == typeof(result.x)
        @test ComponentArrays.getaxes(result.errors) == expected_axes
    end

    @testset "HesseResult{X,S} compatibility constructor" begin
        result = hesse(_struct_obj, start, errs)
        X, S = typeof(result.x), typeof(result.state)
        # Exact-type passthrough: `E` is inferred from `errors`.
        reconstructed = HesseResult{X,S}(
            result.x, result.covariance, result.errors,
            result.edm, result.nfcn, result.status, result.valid,
            result.state)
        @test reconstructed.errors === result.errors
        # Pre-#48 calling pattern: dense errors alongside a structured x
        # (the field was `Vector{Float64}` regardless of `X`).
        dense = HesseResult{X,S}(
            result.x, result.covariance, collect(result.errors),
            result.edm, result.nfcn, result.status, result.valid,
            result.state)
        @test dense.errors isa Vector{Float64}
        @test dense isa HesseResult{X,S}
        # The default constructor's convert-to-field-type semantics are
        # preserved (an `Int` edm / `Int32` nfcn converted, as they were
        # before the error-container type parameter existed).
        converted = HesseResult{X,S}(
            result.x, result.covariance, result.errors,
            0, Int32(result.nfcn), result.status, result.valid,
            result.state)
        @test converted.edm === 0.0
        @test converted.nfcn === result.nfcn
    end

    @testset "the caller's x0 is never mutated" begin
        x0 = ComponentArray(a = 0.3, b = 0.4)
        snapshot = copy(getdata(x0))
        er = [0.1, 0.1]
        er_snapshot = copy(er)
        m = Minuit(_struct_obj, x0; errors = er)
        migrad!(m)
        @test getdata(x0) == snapshot
        @test er == er_snapshot
    end

    # ── every callback path receives the user's container ────────────────────
    # Each entry drives one public operation and asserts ZERO bare-Array
    # callbacks. `mnprofile`/`contour_grid`/`optim`/the threaded probe reach
    # the objective through paths that do not go through the bounded wrapper,
    # so they are the ones most likely to regress.
    @testset "no path hands the objective a bare Array" begin
        cases = [
            ("unbounded",      (f) -> Minuit(f, start; errors = errs),
                               (m) -> migrad!(m)),
            ("bounded",        (f) -> Minuit(f, start; errors = errs,
                                             limits = [(-5.0, 5.0), nothing]),
                               (m) -> migrad!(m)),
            ("fixed",          (f) -> (q = Minuit(f, start; errors = errs);
                                       q.fixed[2] = true; q),
                               (m) -> migrad!(m)),
            ("hesse",          (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); hesse!(m))),
            ("minos",          (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); minos!(m))),
            ("minos bounded",  (f) -> Minuit(f, start; errors = errs,
                                             limits = [(-5.0, 5.0), nothing]),
                               (m) -> (migrad!(m); minos!(m))),
            ("mncontour",      (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); mncontour(m, 1, 2; size = 6))),
            ("contour_grid",   (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); contour_grid(m, 1, 2; size = 4))),
            ("profile",        (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); profile(m, 1; size = 4))),
            ("mnprofile",      (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m); mnprofile(m, 1; size = 4))),
            ("scan",           (f) -> Minuit(f, start; errors = errs),
                               (m) -> NativeMinuit.scan(m, 1; maxsteps = 7)),
            ("simplex",        (f) -> Minuit(f, start; errors = errs),
                               (m) -> NativeMinuit.simplex(m)),
            ("optim",          (f) -> Minuit(f, start; errors = errs),
                               (m) -> NativeMinuit.optim(m; method = :lbfgs)),
            ("extremize",      (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m);
                                       NativeMinuit.extremize(m, p -> p[1] + p[2]))),
            ("mcmc_sample",    (f) -> Minuit(f, start; errors = errs),
                               (m) -> (migrad!(m);
                                       mcmc_sample(m; nsteps = 300, burn = 50, thin = 5))),
        ]
        for (label, build, run) in cases
            f, bad, tot = _axis_probe(expected_axes)
            m = build(f)
            run(m)
            @test (label, bad[]) == (label, 0)
            @test tot[] > 0            # guards against a vacuously-zero count
        end
    end

    @testset "threaded_gradient=:auto probe" begin
        # The probe only runs with >1 thread; assert the call path either way
        # so a single-threaded CI run still exercises the construction.
        f, bad, tot = _axis_probe(expected_axes)
        m = Minuit(f, start; errors = errs)
        migrad!(m; threaded_gradient = :auto)
        @test bad[] == 0
        @test m.valid
    end

    @testset "user gradient receives the original axes" begin
        gbad = Ref(0)
        gtot = Ref(0)
        gradient = p -> begin
            gtot[] += 1
            (p isa ComponentVector && ComponentArrays.getaxes(p) == expected_axes) ||
                (gbad[] += 1)
            ComponentArray(a = 2 * (p.a - 1.0) + 0.5 * (p.b - 2.0),
                           b = 6 * (p.b - 2.0) + 0.5 * (p.a - 1.0))
        end
        f, bad, _ = _axis_probe(expected_axes)
        m = Minuit(f, start; errors = errs, grad = gradient)
        migrad!(m)
        minos!(m)
        @test bad[] == 0
        @test gbad[] == 0
        @test gtot[] > 0
        @test m.ngrad > 0
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
    end

    # ── results carry the container ──────────────────────────────────────────
    @testset "results carry the axes" begin
        m = Minuit(_struct_obj, start; errors = errs)
        migrad!(m)
        @test m.fcn.f === _struct_obj      # no callback wrapper is installed
        @test m.params.values isa ComponentVector
        @test m.fmin.ext_values isa ComponentVector
        @test m.fmin.ext_errors isa ComponentVector
        @test ComponentArrays.getaxes(m.fmin.ext_values) == expected_axes
        @test ComponentArrays.getaxes(m.fmin.ext_errors) == expected_axes
        @test m.fmin.ext_values.a ≈ 1.0 atol = 1e-4
        @test m.fmin.ext_values.b ≈ 2.0 atol = 1e-4
        @test copy(m.values) isa ComponentVector
        @test copy(m.errors) isa ComponentVector

        minos!(m)
        e = m.merrors["x0"]
        @test e.upper_state isa ComponentVector
        @test e.lower_state isa ComponentVector
        @test ComponentArrays.getaxes(e.upper_state) == expected_axes

        r = NativeMinuit.extremize(m, p -> p[1] + p[2])
        @test r.plo isa ComponentVector
        @test r.phi isa ComponentVector
    end

    @testset "MCMC ensemble hands structured rows to user functions" begin
        m = Minuit(_struct_obj, start; errors = errs)
        migrad!(m)
        ens = mcmc_sample(m; nsteps = 300, burn = 50, thin = 5)
        @test ens.best isa ComponentVector
        @test ens[1] isa ComponentVector
        @test ComponentArrays.getaxes(ens[1]) == expected_axes
        @test first(ens) isa ComponentVector           # iteration protocol too

        bad = Ref(0); tot = Ref(0)
        g = θ -> begin
            tot[] += 1
            (θ isa ComponentVector && ComponentArrays.getaxes(θ) == expected_axes) ||
                (bad[] += 1)
            θ.a + θ.b
        end
        q = quantiles(ens, g)
        @test bad[] == 0
        @test tot[] == length(ens)
        @test length(q) == 3
        @test issorted(q)

        # Rows are copies: a mutating user function must not corrupt the
        # ensemble (the container rebuild must not become a view).
        snapshot = copy(ens.samples)
        quantiles(ens, θ -> (θ .= 0.0; 0.0))
        @test ens.samples == snapshot

        bad[] = 0; tot[] = 0
        h = θ -> begin
            tot[] += 1
            (θ isa ComponentVector) || (bad[] += 1)
            [θ.a, 2 * θ.a]
        end
        quantile_band(ens, h, [1.0, 2.0]; curve = true)
        @test bad[] == 0
        @test tot[] == length(ens)
    end

    # ── the low-level kernel APIs, called directly on a cost function ────────
    # The high-level `Minuit` path wraps the objective, so it is structured-safe
    # for a different reason than these are. A user driving the kernel directly
    # (`migrad(f, x0, errs)` then `minos(fm, CostFunction(f), i)`) gets the
    # container from the splice buffers instead, which is a separate mechanism
    # and needs its own coverage.
    @testset "low-level kernel API" begin
        f, bad, tot = _axis_probe(expected_axes)
        cf = CostFunction(f)

        fm = migrad(cf, start, errs)
        @test fm.state.parameters.x isa ComponentVector

        me = minos(fm, cf, 1)
        @test me.upper > 0
        ce = contour_exact(fm, cf, 1, 2; npoints = 6)
        @test length(ce.points) == 6
        sm = NativeMinuit.simplex(cf, start, errs)
        @test isfinite(sm.state.parameters.fval)

        @test bad[] == 0
        @test tot[] > 0

        # Same three, on plain vectors, must give identical numbers.
        ff, _, _ = _axis_probe(expected_axes)   # unused counters; flat objective below
        cff = CostFunction(_flat_obj)
        fmf = migrad(cff, flat_start, errs)
        @test collect(fm.state.parameters.x) == collect(fmf.state.parameters.x)
        @test minos(fmf, cff, 1).upper == me.upper
        @test collect(contour_exact(fmf, cff, 1, 2; npoints = 6).points) ==
              collect(ce.points)
        @test NativeMinuit.simplex(cff, flat_start, errs).state.parameters.fval ==
              sm.state.parameters.fval
    end

    @testset "sampling and resampling helpers" begin
        f, bad, tot = _axis_probe(expected_axes)
        m = Minuit(f, start; errors = errs)
        migrad!(m)

        @test args(m) isa ComponentVector           # IMinuit.jl-compatible accessor

        get_contours_samples(m; nsamples = 60)
        @test bad[] == 0

        # find_solution_modes evaluates the raw objective on each sample row.
        S = [1.0 2.0; 1.01 2.01; 0.99 1.99; 1.005 2.005; 0.995 1.995]
        bad[] = 0
        find_solution_modes(S, m)
        @test bad[] == 0

        # The per-sample evaluation loop rebuilds each row onto the container.
        # Its parallel branch must agree EXACTLY with the serial one and with a
        # direct evaluation — a wrong row there is a silent wrong χ², not an
        # error.
        #
        # HONEST LIMIT: with `Threads.nthreads() == 1` the parallel branch
        # degenerates to a serial loop, so `par == ser` is trivially true and
        # this check only pins the row-fill logic, not the concurrency. It
        # discriminates from 2 threads up; the original defect needed 8 to show
        # 18/61 corrupted rows. Run the suite with `-t auto` to get the real
        # signal.
        big = hcat(1.0 .+ 0.02 .* range(-1, 1; length = 64),
                   2.0 .+ 0.02 .* range(1, -1; length = 64))
        par = NativeMinuit._eval_fvals(m, big, true)
        ser = NativeMinuit._eval_fvals(m, big, false)
        direct = [_struct_obj(ComponentArray(a = big[i, 1], b = big[i, 2]))
                  for i in 1:size(big, 1)]
        @test par == ser
        @test ser == direct

        # Bootstrap / jackknife re-fit a model that reads named components.
        data = NativeMinuit.Data(collect(1.0:6.0), 2 .* collect(1.0:6.0) .+ 1.0,
                                 fill(0.1, 6))
        linmodel(x, p) = p.a * x + p.b
        bs = NativeMinuit.bootstrap(linmodel, data, start; nresample = 5)
        @test bs.n_valid >= 1
        jk = NativeMinuit.jackknife(linmodel, data, start)
        @test jk !== nothing
    end

    # The NUTS path assembles its full vector inside the AdvancedHMC extension,
    # separately from the Metropolis/stretch samplers, and pushes ForwardDiff
    # Duals through it — so the container has to survive a non-Float64 eltype.
    @testset "NUTS posterior keeps the axes" begin
        f, bad, tot = _axis_probe(expected_axes)
        m = Minuit(f, start; errors = errs)
        migrad!(m)
        ps = posterior_sample(m; sampler = :nuts, nsteps = 120, burn = 40)
        @test bad[] == 0
        @test tot[] > 0
        @test length(ps.ensemble) > 0
        @test ps.ensemble.best isa ComponentVector
    end

    # The MIGRAD AD paths, separately from the NUTS one above: the ForwardDiff
    # extension differentiates the objective at the container itself, so the
    # Dual-eltype calls it makes must arrive structured too — the probe counts
    # every call, Float64 and Dual alike.
    @testset "CostFunctionAD keeps the axes" begin
        f, bad, tot = _axis_probe(expected_axes)
        result = migrad(NativeMinuit.CostFunctionAD(f), start, errs)
        @test result.is_valid
        @test bad[] == 0
        @test tot[] > 0
        @test result.state.parameters.x isa ComponentVector
        @test result.state.gradient.grad isa ComponentVector
        @test ComponentArrays.getaxes(result.state.parameters.x) == expected_axes

        # Numeric parity with the identical AD fit on a plain vector.
        flat = migrad(NativeMinuit.CostFunctionAD(_flat_obj), flat_start, errs)
        @test collect(result.state.parameters.x) == collect(flat.state.parameters.x)
        @test result.state.parameters.fval == flat.state.parameters.fval
        @test result.state.nfcn == flat.state.nfcn
    end

    @testset "user ForwardDiff gradient keeps the axes" begin
        f, bad, tot = _axis_probe(expected_axes)
        gbad = Ref(0)
        gtot = Ref(0)
        g = p -> begin
            gtot[] += 1
            (p isa ComponentVector && ComponentArrays.getaxes(p) == expected_axes) ||
                (gbad[] += 1)
            ForwardDiff.gradient(_struct_obj, p)
        end
        m = Minuit(f, start; errors = errs, grad = g)
        migrad!(m)
        hesse!(m)
        minos!(m)
        @test m.valid
        @test bad[] == 0
        @test tot[] > 0
        @test gbad[] == 0
        @test gtot[] > 0
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test m.values[2] ≈ 2.0 atol = 1e-4
    end

    @testset "profile_band endpoints keep the axes" begin
        nested = ComponentArray(slope = 0.0, intercept = 0.0, offset = 0.0)
        obj(p) = (p.slope - 1.0)^2 + (p.intercept - 2.0)^2 + (p.offset - 3.0)^2
        m = Minuit(obj, nested; errors = fill(0.1, 3))
        migrad!(m)
        band = NativeMinuit.profile_band(m, (x, p) -> p.slope * x + p.intercept,
                                         [0.5, 1.0])
        stored = filter(!isnothing, vcat(band.plo, band.phi))
        @test !isempty(stored)
        @test all(v -> v isa ComponentVector, stored)
    end

    @testset "bounded MINOS snapshot keeps the axes" begin
        m = Minuit(_struct_obj, start; errors = errs,
                   limits = [(-5.0, 5.0), nothing])
        migrad!(m)
        minos!(m)
        e = m.merrors["x0"]
        @test e.upper_state isa ComponentVector
        @test ComponentArrays.getaxes(e.upper_state) == expected_axes
    end

    # ── the internal coordinate space is labelled honestly ───────────────────
    # A bound makes the stored number an arcsin-transformed coordinate and a
    # fixed parameter shortens the vector; in neither case does slot `i` still
    # denote the parameter the user's label names, so the internal workspace
    # must NOT wear those labels. Only the all-free unbounded case, where
    # internal ≡ external, keeps the container.
    @testset "internal coordinates are not mislabelled" begin
        m_free = Minuit(_struct_obj, start; errors = errs)
        migrad!(m_free)
        @test m_free.fmin.internal.state.parameters.x isa ComponentVector
        @test ComponentArrays.getaxes(m_free.fmin.internal.state.parameters.x) ==
              expected_axes
        # internal ≡ external here, so the labelled value IS the fitted value
        @test m_free.fmin.internal.state.parameters.x.a ≈ m_free.values[1]

        m_bnd = Minuit(_struct_obj, start; errors = errs,
                       limits = [(-5.0, 5.0), nothing])
        migrad!(m_bnd)
        xb = m_bnd.fmin.internal.state.parameters.x
        @test !(xb isa ComponentVector)
        @test xb isa Vector{Float64}
        # The transformed coordinate genuinely differs from the external value,
        # which is exactly why it must not carry the label `a`.
        @test !isapprox(xb[1], m_bnd.values[1]; atol = 1e-8)
        @test m_bnd.values[1] ≈ 1.0 atol = 1e-2

        m_fix = Minuit(_struct_obj, start; errors = errs)
        m_fix.fixed[2] = true
        migrad!(m_fix)
        xf = m_fix.fmin.internal.state.parameters.x
        @test !(xf isa ComponentVector)
        @test length(xf) == 1                       # reduced free space
        # The full external result is still structured and full-length.
        @test m_fix.fmin.ext_values isa ComponentVector
        @test length(m_fix.fmin.ext_values) == 2
    end

    # ── numbers must match the equivalent flat fit exactly ───────────────────
    @testset "numeric parity with a plain-Vector fit" begin
        for (label, limits, fixed) in [
            ("unbounded", nothing, nothing),
            ("bounded",   [(-5.0, 5.0), nothing], nothing),
            ("fixed",     nothing, [false, true]),
        ]
            ms = Minuit(_struct_obj, start; errors = errs,
                        limits = limits, fixed = fixed)
            mf = Minuit(_flat_obj, flat_start; errors = errs,
                        limits = limits, fixed = fixed)
            migrad!(ms); migrad!(mf)
            @test (label, ms.fval) == (label, mf.fval)
            @test collect(ms.values) == collect(mf.values)
            @test collect(ms.errors) == collect(mf.errors)
            @test ms.nfcn == mf.nfcn
        end

        # MINOS and contours too — these route through the reduced-coordinate
        # probe machinery, where the container must NOT leak into the inner
        # dense space and perturb it.
        ms = Minuit(_struct_obj, start; errors = errs)
        mf = Minuit(_flat_obj, flat_start; errors = errs)
        migrad!(ms); migrad!(mf)
        minos!(ms); minos!(mf)
        for k in ("x0", "x1")
            @test ms.merrors[k].lower == mf.merrors[k].lower
            @test ms.merrors[k].upper == mf.merrors[k].upper
        end
        cs = mncontour(ms, 1, 2; size = 8)
        cf = mncontour(mf, 1, 2; size = 8)
        @test collect(cs) == collect(cf)
    end

    @testset "nested axes are generic" begin
        nested_start = ComponentArray(
            line = (slope = 0.0, intercept = 0.0),
            offset = 0.0,
        )
        objective(p) = (p.line.slope - 1.0)^2 +
                       (p.line.intercept - 2.0)^2 +
                       (p.offset - 3.0)^2

        m = Minuit(objective, nested_start; errors = fill(0.1, 3))
        migrad!(m)
        minos!(m)

        @test m.valid
        @test m.fmin.ext_values.line.slope ≈ 1.0 atol = 1e-4
        @test m.fmin.ext_values.line.intercept ≈ 2.0 atol = 1e-4
        @test m.fmin.ext_values.offset ≈ 3.0 atol = 1e-4
        @test m.merrors["x0"].upper_state isa ComponentVector
    end

    @testset "copy construction and mutation keep the container" begin
        m = Minuit(_struct_obj, start; errors = errs)
        migrad!(m)

        m_copy = Minuit(_struct_obj, m)
        migrad!(m_copy)
        @test m_copy.fmin.ext_values isa ComponentVector
        @test ComponentArrays.getaxes(m_copy.fmin.ext_values) == expected_axes

        set_value!(m_copy, 1, 0.5)
        @test NativeMinuit._init_params(m_copy).values isa ComponentVector
        migrad!(m_copy)
        @test m_copy.valid
        # Loose absolute tolerance: MIGRAD stops at `tol`, not at machine
        # precision. The sharp check is the flat-run comparison below, which
        # must agree to the last bit.
        @test m_copy.values[1] ≈ 1.0 atol = 1e-2
        # Same construction sequence on the flat side (fit → copy-construct →
        # fit → set_value → fit), so the two trajectories are comparable.
        mf0 = Minuit(_flat_obj, flat_start; errors = errs)
        migrad!(mf0)
        mf_copy = Minuit(_flat_obj, mf0)
        migrad!(mf_copy)
        set_value!(mf_copy, 1, 0.5)
        migrad!(mf_copy)
        @test collect(m_copy.values) == collect(mf_copy.values)

        set_limits!(m_copy, 1, -4.0, 4.0)
        migrad!(m_copy)
        @test m_copy.valid
        @test m_copy.fmin.ext_values isa ComponentVector
    end

    # ── containers other than ComponentVector still behave ───────────────────
    # The contract is on `similar`, not on ComponentArrays. A view and a range
    # have no structure to preserve, so they must simply keep working.
    @testset "unstructured AbstractVector x0 still works" begin
        backing = zeros(5)
        v = view(backing, 2:3)
        m = Minuit(_flat_obj, v; errors = errs)
        migrad!(m)
        @test m.valid
        @test m.values[1] ≈ 1.0 atol = 1e-4
        @test backing == zeros(5)          # the view's backing is untouched

        mr = Minuit(_flat_obj, range(0.0, 0.0; length = 2); errors = errs)
        migrad!(mr)
        @test mr.valid
        @test mr.values[1] ≈ 1.0 atol = 1e-4

        mi = Minuit(_flat_obj, [0, 0]; errors = errs)   # Int x0
        migrad!(mi)
        @test mi.valid
        @test mi.values[1] ≈ 1.0 atol = 1e-4
    end

    # ── canonical errors vector carries the values container's axes ──────────
    # Issue #45 splits `Parameters` into metadata + two canonical number
    # vectors. `errors` is allocated as `similar(values, Float64)`, so a
    # structured donor must stamp its concrete container type AND axes onto
    # BOTH canonical vectors — not just `values`.
    @testset "canonical errors vector carries the values axes" begin
        # Record path: the `values=` keyword donates only the container/axes;
        # the numbers come from the records.
        recs = [MinuitParameter("a", 0.7, 0.2),
                MinuitParameter("b", -1.3, 0.4)]
        P = Parameters(recs; values = start)
        v = getfield(P, :values)
        e = getfield(P, :errors)
        @test typeof(e) == typeof(v)
        @test v isa ComponentVector
        @test e isa ComponentVector
        @test ComponentArrays.getaxes(v) == expected_axes
        @test ComponentArrays.getaxes(e) == ComponentArrays.getaxes(v)
        # The numbers are the records', not the donor's.
        @test collect(v) == [0.7, -1.3]
        @test collect(e) == [0.2, 0.4]

        # After a fit, the fit-overlaid `m.params` property must keep the
        # config's container family on both canonical vectors.
        m = Minuit(_struct_obj, start; errors = errs)
        migrad!(m)
        ov = m.params
        @test ov.values isa ComponentVector
        @test ov.errors isa ComponentVector
        @test ComponentArrays.getaxes(ov.values) == expected_axes
        @test ComponentArrays.getaxes(ov.errors) == expected_axes
        @test collect(ov.values) == collect(m.values)
        @test collect(ov.errors) == collect(m.errors)
    end
end
