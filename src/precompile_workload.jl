# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# precompile_workload.jl — Phase 2.4 PrecompileTools workload.
#
# Runs a small set of representative MIGRAD / HESSE / MINOS / contour /
# bounded MIGRAD calls during precompilation so user TTFX
# (time-to-first-execution) is dominated by their FCN compilation, not
# by ours. Targets the public API surface listed in §3 of the ROADMAP
# plus the Phase 3 iminuit-style wrapper.
#
# Wrapped in a try/catch so workload failures during package
# precompilation don't break installation.
# ─────────────────────────────────────────────────────────────────────────────

using PrecompileTools

PrecompileTools.@setup_workload begin
    # Small test FCN — type-stable, Float64 in/out.
    _wl_f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2

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
        catch
            # Don't fail precompile on transient issues
        end
    end
end
