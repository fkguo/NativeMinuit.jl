# SPDX-License-Identifier: LGPL-2.1-or-later

# Tests for the Jupyter-first rich output (src/display.jl): round-to-
# uncertainty formatting, asymmetric MINOS notation, the validity
# checklist, the correlation-matrix heatmap, strong-correlation warnings,
# the χ²/ndf + p-value header, the LaTeX export, and graceful degradation.

using JuMinuit
using Test

const _D = JuMinuit   # reach the un-exported display helpers

html(m) = (io = IOBuffer(); show(io, MIME"text/html"(), m); String(take!(io)))
plain(m) = (io = IOBuffer(); show(io, MIME"text/plain"(), m); String(take!(io)))

@testset "display.jl — Jupyter-first rich output" begin

    # ── B: round-to-uncertainty value formatting ─────────────────────────────
    @testset "B: _format_value_uncertainty rounding" begin
        # Error sets the significant figures; value matches the decimal place.
        @test _D._format_value_uncertainty(2.5478, 0.1234) == "2.55 ± 0.12"
        # Common power-of-ten factored out for small magnitudes.
        @test _D._format_value_uncertainty(1.7e-4, 3e-5)   == "(1.7 ± 0.3)e-4"
        @test _D._format_value_uncertainty(123.456, 1.2)   == "123.5 ± 1.2"
        # One significant figure on a 0.5-style error.
        @test _D._format_value_uncertainty(0.0, 0.5)       == "0.0 ± 0.5"

        # Graceful degradation: no usable uncertainty → value alone, no "±".
        for baderr in (0.0, -1.0, Inf, -Inf, NaN)
            s = _D._format_value_uncertainty(100.0, baderr)
            @test !occursin("±", s)
            @test occursin("100", s)
        end
        # Non-finite value never throws and yields no "±".
        @test !occursin("±", _D._format_value_uncertainty(NaN, 1.0))
        @test !occursin("±", _D._format_value_uncertainty(Inf, 1.0))
    end

    # ── D: compact asymmetric MINOS notation ─────────────────────────────────
    @testset "D: _format_value_minos notation per mode" begin
        t = _D._format_value_minos(8.0, -1.05, 1.12; mode = :text)
        @test occursin("+", t) && occursin("/", t) && occursin("−", t)

        h = _D._format_value_minos(8.0, -1.05, 1.12; mode = :html)
        @test occursin("<sup>+", h)
        @test occursin("<sub>−", h)

        l = _D._format_value_minos(8.0, -1.05, 1.12; mode = :latex)
        @test occursin("\\num{", l)
        @test occursin("^{+", l) && occursin("_{-", l)

        # Degrades to a plain value when neither side is a usable error.
        @test !occursin("sup", _D._format_value_minos(8.0, 0.0, 0.0; mode = :html))
    end

    # ── E: self-contained χ² p-value ─────────────────────────────────────────
    @testset "E: _chi2_pvalue against known χ² survival values" begin
        @test isapprox(_D._chi2_pvalue(10.0, 10), 0.4405, atol = 1e-3)
        @test isapprox(_D._chi2_pvalue(20.0, 10), 0.0293, atol = 1e-3)
        @test isapprox(_D._chi2_pvalue(2.0, 10),  0.9963, atol = 1e-3)
        @test isapprox(_D._chi2_pvalue(3.84, 1),  0.0500, atol = 1e-3)  # 1-dof 95%
        # Monotone decreasing in χ² and within [0, 1].
        ps = [_D._chi2_pvalue(c, 5) for c in 1.0:1.0:30.0]
        @test all(0.0 .<= ps .<= 1.0)
        @test issorted(ps; rev = true)
        # Invalid input → NaN, never throws.
        @test isnan(_D._chi2_pvalue(5.0, 0))
        @test isnan(_D._chi2_pvalue(-1.0, 5))
    end

    # ── A quadratic χ² fit with a strongly-correlated parameter pair ─────────
    # A line fit on x far from the origin makes intercept/slope nearly
    # perfectly anticorrelated (ρ ≈ −1), exercising the heatmap, the
    # strong-correlation warning, AND the χ²/ndf line (model_fit sets ndata).
    line(x, p) = p[1] + p[2] * x
    xs = [100.0, 101.0, 102.0, 103.0, 104.0]
    ys = [2.0, 2.1, 2.0, 2.2, 2.1]
    dat = Data(xs, ys, fill(0.05, 5))
    mc = model_fit(line, dat, [0.0, 0.0]; names = ["a", "b"])
    migrad!(mc)

    @testset "E: χ²/ndf + p-value line (ndata auto-populated by model_fit)" begin
        @test mc.ndata == 5
        s = plain(mc)
        @test occursin("χ²/ndf", s)
        @test occursin("p =", s)
        h = html(mc)
        @test occursin("χ²/ndf", h)
    end

    @testset "A: correlation-matrix heatmap (HTML)" begin
        h = html(mc)
        @test occursin("Correlation matrix", h)
        @test occursin("<table", h)
        @test occursin("rgb(", h)          # color-graded cells
        # Row/col headers are the free-parameter names.
        @test occursin("a", h) && occursin("b", h)
    end

    @testset "F: strong-correlation (near-degeneracy) warning" begin
        @test !isempty(_D._strong_corr_pairs(mc))
        @test occursin("strongly correlated", plain(mc))
        @test occursin("strongly correlated", html(mc))
        # No spurious warning for a well-conditioned, independent fit.
        miquad = Minuit(p -> (p[1] - 1.0)^2 + (p[2] - 2.0)^2, [0.0, 0.0];
                        names = ["u", "v"])
        migrad!(miquad)
        @test isempty(_D._strong_corr_pairs(miquad))
        @test !occursin("strongly correlated", html(miquad))
    end

    # ── C: validity checklist ────────────────────────────────────────────────
    @testset "C: validity checklist chips" begin
        checks = _D._validity_checks(mc)
        @test length(checks) == 5
        @test all(c -> c.status in (:ok, :warn, :bad), checks)
        labels = [c.label for c in checks]
        @test "Valid minimum" in labels
        @test "EDM below goal" in labels
        @test "Covariance accurate" in labels
        h = html(mc)
        @test occursin("Valid minimum", h)
        @test occursin("#1a7f37", h)       # green chip for a passing check
        p = plain(mc)
        @test occursin("[✓ Valid minimum]", p)
    end

    # ── D + sup/sub in the live HTML table after MINOS ───────────────────────
    @testset "D: asymmetric MINOS renders as sup/sub in the table" begin
        mq = Minuit(p -> (p[1] - 8.0)^2 + (p[2] - 2.0)^2, [7.0, 1.0];
                    names = ["x", "y"])
        migrad!(mq)
        minos!(mq)
        h = html(mq)
        @test occursin("<sup>+", h)
        @test occursin("<sub>−", h)
        # Without MINOS, the Value column shows the symmetric Hesse "± ".
        mh = Minuit(p -> (p[1] - 8.0)^2 + (p[2] - 2.0)^2, [7.0, 1.0];
                    names = ["x", "y"])
        migrad!(mh)
        @test occursin("±", html(mh))
        @test !occursin("<sup>+", html(mh))
    end

    # ── G: LaTeX export ──────────────────────────────────────────────────────
    @testset "G: to_latex(m) booktabs + siunitx" begin
        m3 = Minuit(p -> sum(abs2, p .- [1.0, 2.0, 3.0]), [0.0, 0.0, 0.0];
                    names = ["p1", "p2", "p3"])
        migrad!(m3)
        lx = to_latex(m3)
        @test occursin("\\begin{tabular}", lx)
        @test occursin("\\toprule", lx) && occursin("\\bottomrule", lx)
        @test occursin("\\num{", lx)
        # Round-trips the parameter count (one \num value cell per parameter).
        @test count("\\num{", lx) >= 3
        @test count(" \\\\\n", lx) == 4   # header row + 3 parameter rows

        # Asymmetric MINOS becomes \num{x}^{+hi}_{-lo}.
        mq = Minuit(p -> (p[1] - 8.0)^2 + (p[2] - 2.0)^2, [7.0, 1.0];
                    names = ["x", "y"])
        migrad!(mq); minos!(mq)
        lxm = to_latex(mq)
        @test occursin("^{+", lxm) && occursin("_{-", lxm)

        # Options: plain numbers + \hline rules + table float.
        lp = to_latex(m3; siunitx = false, booktabs = false,
                       caption = "Fit", label = "tab:fit")
        @test !occursin("\\num{", lp)
        @test occursin("\\hline", lp)
        @test occursin("\\begin{table}", lp) && occursin("\\caption{Fit}", lp)

        # to_latex(::MinosError) for inline use.
        e = mq.merrors["x"]
        es = to_latex(e)
        @test occursin("^{+", es) && occursin("_{-", es)
    end

    # ── Security: HTML-escape user-controlled parameter names everywhere ─────
    @testset "HTML injection safety (param names escaped)" begin
        nasty = "a<b&\"x"
        mu = Minuit(p -> (p[1] - 1.0)^2 + (p[2] - 2.0)^2, [0.0, 0.0];
                    names = [nasty, "z"])
        migrad!(mu)
        h = html(mu)
        # Raw special chars must not survive into the markup.
        @test !occursin("a<b", h)
        @test !occursin("a&\"", h)
        # Escaped forms appear (table cell AND heatmap header).
        @test occursin("a&lt;b", h)
        @test occursin("&amp;", h)
        @test occursin("&quot;", h)
        # LaTeX export escapes its own specials (underscore is the classic one).
        ml = Minuit(p -> (p[1] - 1.0)^2, [0.0]; names = ["m_pi"])
        migrad!(ml)
        @test occursin("m\\_pi", to_latex(ml))
    end

    # ── Graceful degradation at every stage ──────────────────────────────────
    @testset "graceful degradation" begin
        # Not yet minimized: both reprs render, no fit-only sections.
        m0 = Minuit(x -> sum(abs2, x), [1.0, 2.0]; names = ["x", "y"])
        @test occursin("not yet minimized", plain(m0))
        @test occursin("not yet minimized", html(m0))
        @test !occursin("Correlation matrix", html(m0))

        # Single free parameter: no heatmap, no correlation warning.
        m1 = Minuit(x -> (x[1] - 3.0)^2, [0.0]; names = ["p"])
        migrad!(m1)
        @test isnothing(matrix(m1; correlation = true)) ||
              size(matrix(m1; correlation = true), 1) < 2
        @test !occursin("Correlation matrix", html(m1))
        @test !occursin("strongly correlated", html(m1))
        @test occursin("Valid minimum", html(m1))   # checklist still shown

        # No covariance (pre-migrad): heatmap/warning helpers no-op cleanly.
        m2 = Minuit(x -> (x[1] - 1.0)^2, [0.0])
        @test matrix(m2; correlation = true) === nothing
        @test isempty(_D._strong_corr_pairs(m2))

        # ndata unknown → no χ²/ndf line; likelihood fit (errordef ≠ 1) with
        # ndata set still suppresses it.
        @test _D._chi2_summary(m1) === nothing          # ndata unset
        ml = Minuit(x -> (x[1] - 1.0)^2, [0.0]); ml.errordef = 0.5
        migrad!(ml); ml.ndata = 10
        @test _D._chi2_summary(ml) === nothing          # likelihood, not χ²
        @test !occursin("χ²/ndf", plain(ml))

        # Fixed parameter renders (value-only cell, no error) without error.
        mf = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2, [0.0, 0.0];
                    names = ["a", "b"], fixed = [true, false])
        migrad!(mf)
        @test occursin("yes", plain(mf))
        @test occursin("(fixed)", to_latex(mf))
    end

    # ── ndata property plumbing ──────────────────────────────────────────────
    @testset "ndata property: set / clear / coerce / introspect" begin
        m = Minuit(x -> (x[1] - 1.0)^2, [0.0])
        @test m.ndata === nothing
        m.ndata = 42
        @test m.ndata == 42
        m.ndata = 7.0          # coerced to Int
        @test m.ndata === 7
        m.ndata = nothing
        @test m.ndata === nothing
        @test :ndata in propertynames(m)
    end

    # ── Pathological magnitude gaps stay bounded (code-review findings) ───────
    @testset "extreme value/error magnitude gaps are bounded" begin
        # A tiny-but-finite error against an O(1) value must NOT explode into a
        # ~300-digit fixed-point string; it stays short and never throws.
        for (v, e) in ((1.0, 1e-300), (1.0, 1e300), (1e300, 1.0), (1e-300, 1e-302))
            s = _D._format_value_uncertainty(v, e)
            @test s isa String
            @test length(s) < 64
        end
        # to_latex(::MinosError) with a non-converged (non-finite) side falls
        # back to the central value — no Inf/NaN leaks into the LaTeX.
        bad = JuMinuit.MinosError(1, 2.5, Inf, -0.3, false, true,
                                  false, false, false, false, 0)
        ls = to_latex(bad)
        @test !occursin("Inf", ls) && !occursin("NaN", ls)
        @test !occursin("^{+", ls)
        @test occursin("2.5", ls)
        # A well-formed MinosError still renders the asymmetric form.
        good = JuMinuit.MinosError(1, 2.5, 0.32, -0.28, true, true,
                                   false, false, false, false, 0)
        @test occursin("^{+", to_latex(good))

        # Subnormal / non-finite errors must never throw (was InexactError
        # via round(Int, Inf) when exp10(oerr-2) underflowed).
        for e in (5e-324, 1e-320, floatmin(Float64), Inf, NaN)
            @test _D._format_value_uncertainty(1.0, e) isa String
        end
        @test _D._format_value_minos(1.0, -1e-310, 2e-310; mode = :html) isa String

        # _format_value_minos with one non-finite side degrades to a clean
        # symmetric string (no "inf"/"nan" inside the sup/sub markup).
        h = _D._format_value_minos(8.0, -Inf, 1.1; mode = :html)
        @test !occursin("inf", lowercase(h))
        @test !occursin("<sub>", h)        # degraded, not asymmetric

        # _corr_cell_style must not throw on a non-finite ρ.
        @test _D._corr_cell_style(NaN, false) isa String
        @test _D._corr_cell_style(Inf, false) isa String
    end

    # ── PDG 950-999 rounding carry ───────────────────────────────────────────
    @testset "PDG 950-999 carry" begin
        # An error mantissa in [0.950, 0.999) rounds UP into the next decade
        # and keeps 2 sig figs (0.96 → "1"), with the value tracking the place.
        @test _D._format_value_uncertainty(5.0, 0.96) == "5 ± 1"
        @test _D._format_value_uncertainty(5.0, 9.7)  == "5 ± 10"
        # A "normal" 2-sig-fig error is unchanged.
        @test _D._format_value_uncertainty(5.0, 0.12) == "5.00 ± 0.12"
    end
end
