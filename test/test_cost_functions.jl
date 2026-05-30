# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Tests for the Julia-native cost type family (src/cost_functions.jl).
#
# Organised by the three design requirements:
#   Req 1 — Julia-native shape: errordef TRAIT dispatch, composition via
#           `+`, masking via BitVector, type-parameterised on the model.
#   Req 2 — ONE χ² kernel: LeastSquares is bit-identical to / reaches the
#           same minimum as chisq / model_fit.
#   Req 3 — IMinuit.jl function-style untouched (its own test files stay
#           green; a couple of re-assertions here guard the chisq refactor).
#
# All data are fixed arrays (no RNG) so the analytic-MLE checks are
# reproducible. Unbinned tests use a Gaussian pdf (always positive, ∫=1
# analytically, no erf needed); binned tests use the exponential cdf.

# Local mean (avoid a Statistics test-dep that isn't in the test target).
_mean(v) = sum(v) / length(v)

@testset "cost_functions.jl — Julia-native cost family" begin

    # Shared fixtures ----------------------------------------------------
    # Linear model in the JuMinuit/IMinuit.jl convention model(x, par).
    linmodel(x, p) = p[1] * x + p[2]
    # Gaussian pdf, p = [μ, σ]; positive everywhere, analytically normalised.
    _inv_sqrt2pi = 1 / sqrt(2π)
    gpdf(x, p) = _inv_sqrt2pi / p[2] * exp(-0.5 * ((x - p[1]) / p[2])^2)

    xlin = [0.0, 1.0, 2.0, 3.0, 4.0]
    ylin = [1.1, 2.9, 5.2, 6.8, 9.1]
    elin = fill(0.1, 5)

    # Gaussian sample (well inside ±∞ so truncation is negligible).
    gsamp = [-1.2, -0.5, 0.1, 0.3, 0.7, -0.8, 1.1, 0.4,
             -0.2, 0.9, 0.0, 0.6, -0.4, 1.3, -0.9, 0.2]

    # ───────────────────────────────────────────────────────────────────
    # Req 2 — dedup with chisq (the single χ² kernel)
    # ───────────────────────────────────────────────────────────────────
    @testset "Req 2: LeastSquares ≡ chisq (bit-identical)" begin
        d = Data(xlin, ylin, elin)
        ls = LeastSquares(xlin, ylin, elin, linmodel)
        ls_fromdata = LeastSquares(d, linmodel)
        for par in ([2.0, 1.0], [1.7, 0.9], [3.0, -1.0])
            ref = chisq(linmodel, d, par)
            # `===` proves bit-identical (same compiled `_chisq_core`).
            @test ls(par) === ref
            @test ls_fromdata(par) === ref
        end
        # ...and equal to the explicit residual sum.
        par = [1.9, 1.05]
        manual = sum(((ylin .- linmodel.(xlin, Ref(par))) ./ elin) .^ 2)
        @test ls(par) ≈ manual
    end

    @testset "Req 2: Minuit(LeastSquares) ≡ model_fit (same minimum)" begin
        d = Data(xlin, ylin, elin)
        m_cost = Minuit(LeastSquares(d, linmodel), [1.0, 0.0]); migrad!(m_cost)
        m_mf   = model_fit(linmodel, d, [1.0, 0.0]);            migrad!(m_mf)
        @test collect(m_cost.values) ≈ collect(m_mf.values) atol = 1e-8
        @test m_cost.fval ≈ m_mf.fval atol = 1e-8
        # `Minuit(cost,x0)` auto-extracts up=1 and the data count, like model_fit.
        @test m_cost.fcn.up == 1.0
        @test m_cost.ndata == d.ndata == m_mf.ndata
    end

    # ───────────────────────────────────────────────────────────────────
    # Req 1 — errordef TRAIT dispatch (not a stored field)
    # ───────────────────────────────────────────────────────────────────
    @testset "Req 1: errordef trait dispatch" begin
        @test errordef(LeastSquares(xlin, ylin, elin, linmodel)) === 1.0
        @test errordef(UnbinnedNLL(gsamp, gpdf)) === 0.5
        @test errordef(ExtendedUnbinnedNLL(gsamp, gpdf, p -> 1.0)) === 0.5
        @test errordef(BinnedNLL([1.0, 2.0], [0.0, 1.0, 2.0], (x, p) -> x)) === 0.5
        @test errordef(ExtendedBinnedNLL([1.0, 2.0], [0.0, 1.0, 2.0], (x, p) -> x)) === 0.5
        # Dispatch is on the TYPE — no instance needed.
        @test errordef(UnbinnedNLL(Float64[], gpdf)) === 0.5
    end

    # ───────────────────────────────────────────────────────────────────
    # UnbinnedNLL
    # ───────────────────────────────────────────────────────────────────
    @testset "UnbinnedNLL: value, errordef, MLE recovery" begin
        c = UnbinnedNLL(gsamp, gpdf)
        # value == −Σ log pdf
        par = [0.0, 1.0]
        @test c(par) ≈ -sum(log.(gpdf.(gsamp, Ref(par))))
        # `log=true` path: pass log-density, same objective
        clog = UnbinnedNLL(gsamp, (x, p) -> log(gpdf(x, p)); log = true)
        @test clog(par) ≈ c(par)

        # MLE: μ̂ = mean, σ̂ = population std
        μ̂ = _mean(gsamp)
        σ̂ = sqrt(_mean((gsamp .- μ̂) .^ 2))
        m = Minuit(c, [0.0, 1.0]; limits = [nothing, (1e-6, 10.0)], tol = 1e-5)
        migrad!(m)
        @test m.errordef == 0.5
        @test m.fcn.up == 0.5
        @test m.values[1] ≈ μ̂ atol = 1e-4
        @test m.values[2] ≈ σ̂ atol = 1e-4
    end

    # ───────────────────────────────────────────────────────────────────
    # BinnedNLL + ExtendedBinnedNLL (reuse the LR-χ² kernels ×0.5)
    # ───────────────────────────────────────────────────────────────────
    @testset "BinnedNLL: histogram fit + value identity" begin
        edges = collect(0.0:0.5:6.0)
        λ0 = 0.8
        cdfexp(x, p) = 1 - exp(-p[1] * x)
        probs = [cdfexp(edges[i+1], [λ0]) - cdfexp(edges[i], [λ0])
                 for i in 1:length(edges)-1]
        counts = round.(probs ./ sum(probs) .* 2000)

        c = BinnedNLL(counts, edges, cdfexp)
        m = Minuit(c, [1.0]; limits = [(1e-6, 50.0)], tol = 1e-5); migrad!(m)
        @test m.errordef == 0.5
        @test m.values[1] ≈ λ0 atol = 0.02
        @test m.ndata == length(counts)

        # value ≡ 0.5·multinominal_chi2(n, μ), μ scaled to the total
        par = collect(m.values)
        p = [cdfexp(edges[i+1], par) - cdfexp(edges[i], par) for i in 1:length(edges)-1]
        μ = sum(counts) .* p ./ sum(p)
        @test c(par) ≈ 0.5 * multinominal_chi2(counts, μ) atol = 1e-9

        # length(xe) must be length(n)+1
        @test_throws ArgumentError BinnedNLL([1.0, 2.0], [0.0, 1.0], cdfexp)
    end

    @testset "ExtendedBinnedNLL: fits normalisation, value identity" begin
        edges = collect(0.0:0.5:6.0)
        λ0 = 0.8
        cdfexp(x, p) = 1 - exp(-p[1] * x)
        probs = [cdfexp(edges[i+1], [λ0]) - cdfexp(edges[i], [λ0])
                 for i in 1:length(edges)-1]
        counts = round.(probs ./ sum(probs) .* 2000)
        Ntrue = sum(counts)

        scdf(x, p) = p[2] * (1 - exp(-p[1] * x))   # p[2] = expected total
        c = ExtendedBinnedNLL(counts, edges, scdf)
        m = Minuit(c, [1.0, 1900.0];
                   limits = [(1e-6, 50.0), (1.0, 1e7)], tol = 1e-5)
        migrad!(m)
        @test m.errordef == 0.5
        @test m.values[1] ≈ λ0 atol = 0.02
        @test m.values[2] ≈ Ntrue rtol = 0.05

        par = collect(m.values)
        μ = [scdf(edges[i+1], par) - scdf(edges[i], par) for i in 1:length(edges)-1]
        @test c(par) ≈ 0.5 * poisson_chi2(counts, μ) atol = 1e-9
    end

    @testset "ExtendedUnbinnedNLL: extended MLE (N̂=n, λ̂=n/Σx)" begin
        xexp = [0.2, 0.5, 0.9, 1.3, 2.1, 0.7, 1.8, 0.4, 1.1, 0.6, 0.3, 1.5]
        density(x, p) = p[2] * p[1] * exp(-p[1] * x)   # p=[λ,N]; ρ=N·λe^{-λx}
        integral(p) = p[2]                              # ∫₀^∞ ρ dx = N
        c = ExtendedUnbinnedNLL(xexp, density, integral)
        # value ≡ μ − Σ log ρ
        par = [1.0, 12.0]
        @test c(par) ≈ par[2] - sum(log.(density.(xexp, Ref(par))))
        n = length(xexp)
        m = Minuit(c, [1.0, 10.0];
                   limits = [(1e-6, 50.0), (1e-6, 1e6)], tol = 1e-6)
        migrad!(m)
        @test m.errordef == 0.5
        @test m.values[1] ≈ n / sum(xexp) atol = 1e-3   # λ̂
        @test m.values[2] ≈ n atol = 1e-3               # N̂ = sample size
    end

    # ───────────────────────────────────────────────────────────────────
    # Req 1 — composition by operator overloading (CostSum)
    # ───────────────────────────────────────────────────────────────────
    @testset "CostSum: param union by name + FCN = scaled sum" begin
        lsq = LeastSquares(xlin, ylin, elin, linmodel; name = [:a, :b])
        nll = UnbinnedNLL(gsamp, gpdf; name = [:mu, :sig])
        s = lsq + nll
        @test s isa CostSum
        @test parameter_names(s) == [:a, :b, :mu, :sig]
        @test errordef(s) === 1.0          # χ² common scale
        @test JuMinuit._cost_ndata(s) == 5 + length(gsamp)

        # FCN = Σ c_k(sub_k)/errordef(c_k): LSQ unchanged, NLL doubled.
        par = [1.9, 1.05, 0.1, 0.7]
        expect = lsq([1.9, 1.05]) / 1.0 + nll([0.1, 0.7]) / 0.5
        @test s(par) === expect
    end

    @testset "CostSum: shared parameter genuinely shared" begin
        # Two datasets sharing the slope `a`: yA = a x + b, yB = a x + c.
        xA = [0.0, 1.0, 2.0, 3.0]; yA = 2.0 .* xA .+ 1.0
        xB = [0.0, 1.0, 2.0, 3.0]; yB = 2.0 .* xB .+ 5.0
        cA = LeastSquares(xA, yA, fill(0.1, 4), linmodel; name = [:a, :b])
        cB = LeastSquares(xB, yB, fill(0.1, 4), linmodel; name = [:a, :c])
        s = cA + cB
        @test parameter_names(s) == [:a, :b, :c]   # union, first-appearance order
        m = Minuit(s, [1.0, 0.0, 0.0]); migrad!(m)
        @test m.parameters == ("a", "b", "c")      # names flow into Minuit
        @test m.values[1] ≈ 2.0 atol = 1e-4        # shared slope
        @test m.values[2] ≈ 1.0 atol = 1e-4        # intercept A
        @test m.values[3] ≈ 5.0 atol = 1e-4        # intercept B
        @test m.errordef == 1.0
    end

    @testset "CostSum: mixed-type simultaneous fit reaches joint minimum" begin
        # χ² piece (errordef-1) + NLL piece (errordef-0.5), no shared params.
        lsq = LeastSquares(xlin, ylin, elin, linmodel; name = [:a, :b])
        nll = UnbinnedNLL(gsamp, gpdf; name = [:mu, :sig])
        s = lsq + nll
        m = Minuit(s, [1.0, 0.0, 0.0, 1.0];
                   limits = [nothing, nothing, nothing, (1e-6, 10.0)], tol = 1e-6)
        migrad!(m)
        # Each block reaches its own optimum (the pieces are independent).
        m_ls = Minuit(LeastSquares(xlin, ylin, elin, linmodel), [1.0, 0.0]); migrad!(m_ls)
        μ̂ = _mean(gsamp); σ̂ = sqrt(_mean((gsamp .- μ̂) .^ 2))
        @test m.values[1] ≈ m_ls.values[1] atol = 1e-3
        @test m.values[2] ≈ m_ls.values[2] atol = 1e-3
        @test m.values[3] ≈ μ̂ atol = 1e-3
        @test m.values[4] ≈ σ̂ atol = 1e-3
    end

    @testset "CostSum: flattening + unnamed-cost error" begin
        a = LeastSquares(xlin, ylin, elin, linmodel; name = [:a, :b])
        b = UnbinnedNLL(gsamp, gpdf; name = [:mu, :sig])
        c = ExtendedUnbinnedNLL([0.5, 1.0], (x, p) -> p[1], p -> 1.0; name = [:k])
        s = (a + b) + c
        @test length(s.costs) == 3            # flat, not nested
        @test parameter_names(s) == [:a, :b, :mu, :sig, :k]
        # Composing an unnamed cost is rejected with a helpful message.
        @test_throws ArgumentError (a + UnbinnedNLL(gsamp, gpdf))
    end

    # ───────────────────────────────────────────────────────────────────
    # Req 1 — masking (BitVector, no data copy)
    # ───────────────────────────────────────────────────────────────────
    @testset "Masking changes an LSQ fit (drops an outlier)" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 3.0, 5.0, 7.0, 200.0]      # last point is a gross outlier
        ye = fill(1.0, 5)
        m_all = Minuit(LeastSquares(x, y, ye, linmodel), [1.0, 0.0]); migrad!(m_all)
        keep = BitVector([true, true, true, true, false])
        m_msk = Minuit(LeastSquares(x, y, ye, linmodel; mask = keep), [1.0, 0.0];
                       tol = 1e-6)
        migrad!(m_msk)
        @test !isapprox(m_all.values[1], m_msk.values[1]; atol = 1.0)
        @test m_msk.values[1] ≈ 2.0 atol = 1e-3    # clean slope without outlier
        @test m_msk.values[2] ≈ 1.0 atol = 1e-3
        @test m_msk.ndata == 4                      # masked count

        # mask == all-true reproduces the unmasked fit
        m_full = Minuit(LeastSquares(x, y, ye, linmodel; mask = trues(5)), [1.0, 0.0])
        migrad!(m_full)
        @test collect(m_full.values) ≈ collect(m_all.values) atol = 1e-8
    end

    @testset "Masking changes an NLL fit" begin
        c_all = UnbinnedNLL(gsamp, gpdf)
        keep = BitVector([fill(true, 15); false])   # drop last sample
        c_msk = UnbinnedNLL(gsamp, gpdf; mask = keep)
        m_all = Minuit(c_all, [0.0, 1.0]; limits = [nothing, (1e-6, 10.0)], tol = 1e-6)
        m_msk = Minuit(c_msk, [0.0, 1.0]; limits = [nothing, (1e-6, 10.0)], tol = 1e-6)
        migrad!(m_all); migrad!(m_msk)
        @test !isapprox(m_all.values[1], m_msk.values[1]; atol = 1e-4)
        # masked μ̂ equals the mean over the kept sample
        @test m_msk.values[1] ≈ _mean(gsamp[1:15]) atol = 1e-4

        # bad mask length is rejected
        @test_throws ArgumentError UnbinnedNLL(gsamp, gpdf; mask = trues(3))
    end

    # ───────────────────────────────────────────────────────────────────
    # errordef flows into MINOS scaling (needs ≥2 free params)
    # ───────────────────────────────────────────────────────────────────
    @testset "errordef(cost) flows into MINOS" begin
        # Gaussian UnbinnedNLL (errordef 0.5) vs the identical closure with
        # an explicit up=0.5 — MINOS must agree to machine precision, since
        # the only thing that differs is HOW up was set.
        c = UnbinnedNLL(gsamp, gpdf)
        m_cost = Minuit(c, [0.0, 1.0]; limits = [nothing, (1e-6, 10.0)], tol = 1e-6)
        migrad!(m_cost); minos!(m_cost)

        nllc(p) = -sum(log.(gpdf.(gsamp, Ref(p))))
        m_man = Minuit(nllc, [0.0, 1.0]; up = 0.5,
                       limits = [nothing, (1e-6, 10.0)], tol = 1e-6)
        migrad!(m_man); minos!(m_man)

        for k in ("x0", "x1")
            @test m_cost.merrors[k].lower ≈ m_man.merrors[k].lower atol = 1e-6
            @test m_cost.merrors[k].upper ≈ m_man.merrors[k].upper atol = 1e-6
        end

        # A χ² cost (errordef 1) gives the SAME MINOS as a chisq closure with up=1.
        d = Data(xlin, ylin, elin)
        m_ls = Minuit(LeastSquares(d, linmodel), [1.0, 0.0]); migrad!(m_ls); minos!(m_ls)
        m_chi = Minuit(p -> chisq(linmodel, d, p), [1.0, 0.0]); migrad!(m_chi); minos!(m_chi)
        for k in ("x0", "x1")
            @test m_ls.merrors[k].lower ≈ m_chi.merrors[k].lower atol = 1e-6
            @test m_ls.merrors[k].upper ≈ m_chi.merrors[k].upper atol = 1e-6
        end
    end

    # ───────────────────────────────────────────────────────────────────
    # Req 3 — IMinuit.jl function-style behaviour preserved by the refactor
    # ───────────────────────────────────────────────────────────────────
    @testset "Req 3: chisq behaviour unchanged after kernel extraction" begin
        d = Data([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], [0.1, 0.1, 0.1])
        @test chisq(linmodel, d, [2.0, 1.0]) ≈ 0.0 atol = 1e-14
        @test chisq(linmodel, d, [3.0, 1.0]) ≈ 500.0 atol = 1e-9
        # tuple form
        @test chisq(linmodel, ([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]), [2.0, 1.0]) ≈ 0.0 atol = 1e-14
        # fitrange (stride collapses to first:last)
        d2 = Data([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 5.0, 7.0, 9.0], fill(0.1, 5))
        @test chisq(linmodel, d2, [2.0, 1.0]; fitrange = 2:2:4) ≈ 0.0 atol = 1e-12
    end

    # ───────────────────────────────────────────────────────────────────
    # Review follow-ups: NLL log(0) safeguard + cost copy-constructor
    # ───────────────────────────────────────────────────────────────────
    @testset "UnbinnedNLL: pdf=0 gives a finite (not Inf) term" begin
        # A pdf that is exactly 0 at one sample must not Inf/NaN the cost
        # (iminuit _TINY_FLOAT parity). Normal values are unaffected.
        zpdf(x, p) = x == 0.0 ? 0.0 : gpdf(x, p)
        c = UnbinnedNLL([0.0, 0.5, -0.5], zpdf)
        v = c([0.0, 1.0])
        @test isfinite(v)
        # the value-identity test still holds for a strictly-positive pdf
        cpos = UnbinnedNLL(gsamp, gpdf)
        @test cpos([0.0, 1.0]) ≈ -sum(log.(gpdf.(gsamp, Ref([0.0, 1.0]))))
    end

    @testset "Binned NLLs: degenerate μ≤0 stays finite (no DomainError)" begin
        # A non-monotonic cdf drives a negative per-bin probability — without
        # the _NLL_TINY clamp this throws DomainError(log(<0)) and hard-crashes
        # MIGRAD. With the clamp it is a large-but-finite penalty.
        weird_cdf(x, p) = x <= 0 ? 0.0 : x <= 1 ? 0.6 : x <= 2 ? 0.5 : 1.0
        edges = [0.0, 1.0, 2.0, 3.0]
        counts = [5.0, 2.0, 3.0]                  # middle bin: p = 0.5-0.6 = -0.1
        @test isfinite(BinnedNLL(counts, edges, weird_cdf)([1.0]))
        @test isfinite(ExtendedBinnedNLL(counts, edges, weird_cdf)([1.0]))
        # the value-identity (vs 0.5·*_chi2) is unaffected for valid cdfs —
        # already covered in the BinnedNLL / ExtendedBinnedNLL testsets above.
    end

    @testset "Minuit(cost, fit) copy-constructor carries errordef" begin
        d = Data(xlin, ylin, elin)
        m1 = Minuit(LeastSquares(d, linmodel), [1.0, 0.0]; name = ["a", "b"])
        migrad!(m1)
        # Rebuild a DIFFERENT-errordef cost from the old fit: errordef must
        # come from the new cost (0.5), not be inherited from m1 (1.0).
        un = UnbinnedNLL(gsamp, gpdf; name = ["mu", "sig"])
        m2 = Minuit(un, m1)
        @test m2.fcn.up == 0.5                  # errordef(UnbinnedNLL), not m1's 1.0
        @test m2.parameters == ("a", "b")       # names reused from m1
        @test collect(m2.values) ≈ collect(m1.values)  # starts at m1's latest
        @test m2.fmin === nothing               # fresh, not yet minimised
        # same-type rebuild keeps errordef 1
        m3 = Minuit(LeastSquares(d, linmodel), m1)
        @test m3.fcn.up == 1.0
        @test m3.ndata == d.ndata
    end

    # show -----------------------------------------------------------------
    @testset "show methods" begin
        s = repr(LeastSquares(xlin, ylin, elin, linmodel; name = [:a, :b]))
        @test occursin("LeastSquares", s) && occursin("errordef=1.0", s)
        cs = LeastSquares(xlin, ylin, elin, linmodel; name = [:a, :b]) +
             UnbinnedNLL(gsamp, gpdf; name = [:mu, :sig])
        @test occursin("CostSum", repr(cs))
    end
end
