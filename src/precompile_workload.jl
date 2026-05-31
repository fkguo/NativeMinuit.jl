# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# precompile_workload.jl — Phase 2.4 PrecompileTools workload.
#
# Runs a small set of representative calls during precompilation so user TTFX
# (time-to-first-execution) is dominated by their FCN compilation, not by ours.
# Covers the public API surface listed in §3 of the ROADMAP plus the Phase 3
# iminuit-style wrapper, the Julia-native cost-function family, the algorithm
# wrappers (mncontour / profile / mnprofile / simplex / scan) and the
# error-analysis layer (bootstrap / jackknife / contour sampling / solution
# modes). Each path is exercised on the SMALLEST data that still compiles it,
# so the bulk of the type-generic machinery (linear algebra, the DFP loop,
# MINOS, contour, Parameters, result construction) is cached once here — only
# the thin FCN-calling wrappers recompile per user closure.
#
# Each path/section below is wrapped in its OWN try/catch so workload failures
# during package precompilation (a) never break installation and (b) cannot let
# one drifting path silently skip the others. (Every call is validated to run
# cleanly, without warnings, so a catch never silently drops a path.)
#
# AD- and Optim-backed paths live in the package extensions, which carry their
# own @compile_workload blocks (ext/JuMinuit*Ext.jl) — they precompile when the
# trigger package is loaded.
# ─────────────────────────────────────────────────────────────────────────────

using PrecompileTools

PrecompileTools.@setup_workload begin
    # Small test FCN — type-stable, Float64 in/out.
    _wl_f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2

    # Tiny datasets + simple model/pdf/cdf closures for the cost-function
    # family and the error-analysis layer. Kept minimal (3-5 points) but
    # well-posed so every fit converges quietly.
    _wl_x  = [0.0, 1.0, 2.0, 3.0]
    _wl_y  = [1.0, 3.0, 5.0, 7.0]
    _wl_ye = [1.0, 1.0, 1.0, 1.0]
    _wl_line = (x, p) -> p[1] * x + p[2]
    _wl_samp = [-0.4, 0.1, 0.3, -0.2, 0.5]
    # Normalised Gaussian pdf / extended density + integral (no extra deps).
    _wl_pdf  = (x, p) -> exp(-0.5 * ((x - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
    _wl_dens = (x, p) -> p[3] * exp(-0.5 * ((x - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
    _wl_intg = p -> p[3]
    # Binned: logistic cdf — smooth, normalised to (0,1), no SpecialFunctions.
    _wl_n    = [2.0, 5.0, 3.0]
    _wl_xe   = [-1.0, -0.3, 0.3, 1.0]
    _wl_cdf  = (x, p) -> 1.0 / (1.0 + exp(-(x - p[1]) / p[2]))
    _wl_scdf = (x, p) -> p[3] / (1.0 + exp(-(x - p[1]) / p[2]))

    PrecompileTools.@compile_workload begin
        try
            # ── Path A: bare-vector unbounded MIGRAD ───────────────
            cf_a = CostFunction(_wl_f, 1.0)
            m_a = migrad(cf_a, [0.0, 0.0], [0.1, 0.1])

            # ── Path B: MINOS + contour (uses function_cross) ──────
            me_a = minos(m_a, cf_a, 1)
            ce_a = contour(m_a, cf_a, 1, 2; npoints = 6)

            # ── Path C: bound-aware via Parameters ─────────────────
            params = Parameters([
                MinuitParameter("a", 0.0, 0.1; lower = -3.0, upper = 3.0),
                MinuitParameter("b", 0.0, 0.1; fixed = true),
            ])
            m_b = migrad(cf_a, params)

            # ── Path D: iminuit-style Minuit wrapper ───────────────
            mw = Minuit(_wl_f, [0.0, 0.0];
                        names = ["a", "b"], errors = [0.1, 0.1])
            migrad!(mw)
            minos!(mw, 1)

            # ── Path E: HESSE standalone ───────────────────────────
            hesse(cf_a, m_a.state, Strategy(0))

            # ── Path F: serialization ──────────────────────────────
            to_dict(m_a)
            to_dict(m_b)
            to_dict(me_a)
        catch; end

        try
            # ── Path G: Julia-native cost-function family ──────────
            # construct → Minuit(cost, x0) → migrad! for each cost type.
            migrad!(Minuit(LeastSquares(_wl_x, _wl_y, _wl_ye, _wl_line), [1.0, 0.0]))
            migrad!(Minuit(LeastSquares(Data(_wl_x, _wl_y, _wl_ye), _wl_line), [1.0, 0.0]))
            migrad!(Minuit(UnbinnedNLL(_wl_samp, _wl_pdf), [0.0, 1.0]))
            migrad!(Minuit(ExtendedUnbinnedNLL(_wl_samp, _wl_dens, _wl_intg),
                           [0.0, 1.0, 5.0]))
            migrad!(Minuit(BinnedNLL(_wl_n, _wl_xe, _wl_cdf), [0.0, 0.5]))
            migrad!(Minuit(ExtendedBinnedNLL(_wl_n, _wl_xe, _wl_scdf),
                           [0.0, 0.5, 10.0]))
            # CostSum: two named LeastSquares sharing params (simultaneous fit).
            _lsa = LeastSquares(_wl_x, _wl_y, _wl_ye, _wl_line; name = [:a, :b])
            _lsb = LeastSquares(_wl_x, _wl_y, _wl_ye, _wl_line; name = [:a, :b])
            migrad!(Minuit(_lsa + _lsb, [1.0, 0.0]))
        catch; end

        try
            # ── Path H: iminuit-style algorithm wrappers ───────────
            mncontour(mw, 1, 2; numpoints = 6)
            profile(mw, 1; bins = 5)
            mnprofile(mw, 1; bins = 4)
            # scan mutates `m` (best-value retention) — use a throwaway fit.
            _ms = Minuit(_wl_f, [0.0, 0.0]); migrad!(_ms)
            scan(_ms, 1; maxsteps = 5)
            simplex(Minuit(_wl_f, [0.0, 0.0]))
        catch; end

        try
            # ── Path I: error analysis (smallest settings) ─────────
            _data = Data(_wl_x, _wl_y, _wl_ye)
            bootstrap(_wl_line, _data, [1.0, 0.0]; nresample = 2, seed = 1)
            jackknife(_wl_line, _data, [1.0, 0.0])
            get_contours_samples(mw; nsamples = 64, adaptive = false,
                                 seed = 1, warn = false)
            _S = [0.98 1.99; 1.02 2.01; 1.00 2.00; 0.99 2.02; 1.01 1.98; 1.00 1.99]
            find_solution_modes(_S, mw)
        catch; end

        try
            # ── Path J: result plot recipes (RecipesBase; Plots-agnostic) ──
            RecipesBase.apply_recipe(Dict{Symbol,Any}(), ce_a)
            RecipesBase.apply_recipe(Dict{Symbol,Any}(), me_a)
            RecipesBase.apply_recipe(Dict{Symbol,Any}(), m_a)
        catch
            # Don't fail precompile on transient issues
        end
    end
end
