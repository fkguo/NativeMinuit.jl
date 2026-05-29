# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# display.jl — Jupyter-first rich output for `Minuit`.
#
# JuMinuit is used mainly from Jupyter/Pluto, so the `text/html` repr is a
# primary UX surface. This file holds the building blocks the `show`
# methods (in minuit.jl) assemble:
#
#   • round-to-uncertainty value formatting        (_format_value_uncertainty)
#   • compact asymmetric MINOS notation            (_format_value_minos)
#   • a self-contained χ² p-value (no SpecialFunctions dependency)
#   • a per-flag validity checklist                (_validity_checks)
#   • a correlation-matrix heatmap (HTML)          (_render_heatmap_html)
#   • strong-correlation near-degeneracy warnings  (_strong_corr_pairs)
#   • a publication-ready LaTeX export             (to_latex)
#
# These are inspired by Python iminuit's repr but go beyond it (merged
# value±error column, heatmap, χ²/ndf p-value, LaTeX). The `show` methods
# call into here; everything is defined in the JuMinuit module namespace.
# ─────────────────────────────────────────────────────────────────────────────

# ── Low-level number formatting ──────────────────────────────────────────────

# Plain "%.4g" with explicit non-finite spellings (used as the graceful
# fallback when there is no usable uncertainty to round against).
function _fmt_g(x::Float64)
    isnan(x) && return "nan"
    isinf(x) && return x < 0 ? "-inf" : "inf"
    return @sprintf("%.4g", x)
end

# Fixed-point format with exactly `d` decimals. Built at runtime because
# Printf's `@sprintf` macro cannot take a dynamic precision.
function _fixed_decimals(x::Float64, d::Integer)
    d = max(Int(d), 0)
    return Printf.format(Printf.Format("%.$(d)f"), x)
end

# Strip trailing zeros shared by ALL of `vstr` and `estrs` simultaneously,
# so the value and error(s) keep a common decimal precision. This turns
# e.g. ("1.70", ("0.30",)) into ("1.7", ("0.3",)) but leaves
# ("2.55", ("0.12",)) untouched. A dangling decimal point left behind
# (e.g. "2.") is removed.
function _strip_common_zeros(vstr::AbstractString, estrs::Tuple{Vararg{AbstractString}})
    v = String(vstr)
    es = String[String(e) for e in estrs]
    strippable(s) = occursin('.', s) && endswith(s, '0')
    while strippable(v) && all(strippable, es)
        v = chop(v)
        es = map(chop, es)
    end
    droptdot(s) = endswith(s, '.') ? chop(s) : s
    return droptdot(v), Tuple(droptdot.(es))
end

"""
    _round_to_uncertainty(value, errs; factor=true) -> (e10, vstr, estrs)

HEP round-to-uncertainty core. The smallest positive, finite error in
`errs` sets the precision via the PDG rounding rule (look at the three
leading digits of the error: 100–354 → keep two significant figures,
otherwise one). The central `value` is rounded to the same decimal place.

When `factor=true` and the value is very small or very large, a common
power-of-ten `10^e10` is factored out so the mantissas read cleanly
(`1.7e-4 ± 3e-5` → `e10=-4`, `"1.7"`, `("0.3",)`). With `factor=false`
the numbers stay at full scale (`e10=0`), which is what the LaTeX export
wants so `siunitx` can decide the presentation.

Returns the chosen exponent `e10`, the value mantissa string, and a tuple
of error mantissa strings (all at the same decimal precision).
"""
function _round_to_uncertainty(value::Float64, errs::Tuple{Vararg{Float64}};
                               factor::Bool = true)
    emin = Inf
    for e in errs
        (isfinite(e) && e > 0) && (emin = min(emin, e))
    end
    # No usable error → just print everything at 4 significant figures.
    isfinite(emin) || return (0, _fmt_g(value), map(_fmt_g, errs))

    oerr = floor(Int, log10(emin))
    vmag = (isfinite(value) && value != 0) ? abs(value) : emin
    ovalue = floor(Int, log10(vmag))

    # Extreme magnitudes (near the Float64 exponent limits, including
    # subnormals) or a value/error gap too wide to share one fixed-point
    # frame → fall back to independent %g formatting. This preserves BOTH
    # numbers (e.g. "1 ± 1e-300"), and crucially avoids the leading-digit
    # extraction below, where `exp10(oerr - 2)` would underflow to 0.0 for a
    # subnormal error and `round(Int, emin / 0.0)` would throw on `Inf`.
    if oerr < -290 || oerr > 290 || ovalue < -290 || ovalue > 290 ||
       abs(ovalue - oerr) > 18
        return (0, _fmt_g(value), map(_fmt_g, errs))
    end

    # PDG rounding on the three leading digits of the error:
    #   100-354 → 2 sig figs; 355-949 → 1 sig fig; 950-999 → round UP into the
    #   next decade (carry) and keep 2 sig figs (e.g. 0.96 → "1.0", not 0.96).
    lead3 = round(Int, emin / exp10(oerr - 2))    # in [100, 1000]
    if lead3 >= 1000                              # float carry in the extraction
        oerr += 1
        lead3 = 100
    end
    if lead3 >= 950                               # PDG 950-999 carry
        oerr += 1
        nsig = 2
    elseif lead3 >= 355
        nsig = 1
    else                                          # 100..354
        nsig = 2
    end
    p = oerr - (nsig - 1)                         # absolute place of last sig digit

    e10 = 0
    if factor
        eerr = floor(Int, log10(emin))
        # Factor a common power of ten when either the value or the
        # (precision-setting) error is far from O(1). Picking the exponent
        # from whichever is LARGER in magnitude keeps the dominant number's
        # mantissa O(1) and bounds the string length even in the pathological
        # case of a huge error paired with a small value.
        if ovalue < -2 || ovalue > 4 || eerr < -2 || eerr > 4
            e10 = max(ovalue, eerr)
        end
    end

    # Cap the decimal count: a value many orders larger than its error would
    # otherwise demand a meaninglessly long fixed-point string (far past
    # Float64 precision). 20 decimals is generous yet keeps display bounded.
    d = clamp(e10 - p, 0, 20)
    scale = exp10(e10)
    # A non-finite value / error has no mantissa to render — spell it out
    # ("nan"/"inf") rather than letting Printf emit "Inf"/"NaN".
    fmt(x) = isfinite(x) ? _fixed_decimals(x / scale, d) : _fmt_g(x)
    vstr = fmt(value)
    estrs = map(fmt, errs)
    return (e10, _strip_common_zeros(vstr, estrs)...)
end

"""
    _format_value_uncertainty(value, err) -> String

Format `value ± err` the HEP-standard way, where the error sets the
significant figures (1–2 on the error, value rounded to the same decimal
place). Examples:

```
_format_value_uncertainty(2.5478, 0.1234) == "2.55 ± 0.12"
_format_value_uncertainty(1.7e-4, 3e-5)   == "(1.7 ± 0.3)e-4"
```

Degrades gracefully: if `err` is zero, negative, `Inf`, or `NaN` (or the
value is non-finite) there is no uncertainty to round against, so the
value alone is returned at 4 significant figures.
"""
function _format_value_uncertainty(value::Real, err::Real)
    v = Float64(value)
    e = Float64(err)
    (isfinite(v) && isfinite(e) && e > 0) || return _fmt_g(v)
    e10, vstr, estrs = _round_to_uncertainty(v, (e,))
    estr = estrs[1]
    return e10 == 0 ? string(vstr, " ± ", estr) :
                      string("(", vstr, " ± ", estr, ")e", e10)
end

"""
    _format_value_minos(value, lower, upper; mode=:text) -> String

Compact asymmetric-MINOS rendering. `lower` is the (negative) downward
excursion and `upper` the (positive) upward one in Minuit's convention;
both are taken in magnitude here. `mode`:

  • `:text`  → `value +hi/−lo`            (or `(…)eN` when factored)
  • `:html`  → `value<sup>+hi</sup><sub>−lo</sub>`
  • `:latex` → `\\num{value}^{+hi}_{-lo}`

Asymmetric notation requires BOTH sides to be finite. If a side failed to
converge (non-finite), this degrades to a symmetric form on whichever side
is a usable (finite, positive) error, and to a bare value if neither is —
so `inf`/`nan` can never leak into the sup/sub (or LaTeX) markup.
"""
function _format_value_minos(value::Real, lower::Real, upper::Real;
                             mode::Symbol = :text)
    v = Float64(value)
    hi = abs(Float64(upper))
    lo = abs(Float64(lower))
    if !(isfinite(v) && isfinite(hi) && isfinite(lo) && (hi > 0 || lo > 0))
        e = isfinite(hi) && hi > 0 ? hi : (isfinite(lo) && lo > 0 ? lo : NaN)
        return _format_value_uncertainty(v, e)
    end
    e10, vstr, estrs = _round_to_uncertainty(v, (hi, lo))
    histr, lostr = estrs
    if mode === :html
        core = string(vstr, "<sup>+", histr, "</sup><sub>−", lostr, "</sub>")
        return e10 == 0 ? core : string("(", core, ")×10<sup>", e10, "</sup>")
    elseif mode === :latex
        core = string("\\num{", vstr, "}^{+", histr, "}_{-", lostr, "}")
        return e10 == 0 ? core : string(core, " \\times 10^{", e10, "}")
    else
        core = string(vstr, " +", histr, "/−", lostr)
        return e10 == 0 ? core : string("(", core, ")e", e10)
    end
end

# Pick the merged "Value" cell for parameter-row tuple `r` (from
# `_param_row_data`): asymmetric MINOS when both sides exist, else the
# symmetric Hesse error, else the bare value (fixed / no covariance).
function _value_cell(r; mode::Symbol = :text)
    if r.minos_lo !== nothing && r.minos_hi !== nothing
        return _format_value_minos(r.value, r.minos_lo, r.minos_hi; mode = mode)
    elseif r.hesse !== nothing && isfinite(r.hesse) && r.hesse > 0
        # The symmetric string contains no HTML-special characters, so it
        # is safe to emit verbatim in both modes.
        return _format_value_uncertainty(r.value, r.hesse)
    else
        return _fmt_cell(r.value)
    end
end

# ── E: χ²/ndf + p-value (self-contained, no SpecialFunctions dep) ────────────

# ln Γ(x) via the Lanczos approximation (g = 7, 8-term coefficient set),
# with the reflection formula for x < 0.5. Source: Lanczos, SIAM J. Numer.
# Anal. 1 (1964) 86; coefficient set as tabulated in Numerical Recipes 3e
# §6.1 / Boost.Math. Accurate to ~15 digits for the a = ndf/2 arguments we
# feed it.
function _lgamma(x::Float64)
    g = 7.0
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(π / sin(π * x)) - _lgamma(1.0 - x)
    end
    x -= 1.0
    a = c[1]
    t = x + g + 0.5
    @inbounds for i in 2:length(c)
        a += c[i] / (x + (i - 1))
    end
    return 0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end

# Regularized lower incomplete gamma P(a,x) by series. Numerical Recipes
# 3e §6.2 (`gser`).
function _gamma_p_series(a::Float64, x::Float64)
    gln = _lgamma(a)
    ap = a
    s = 1.0 / a
    del = s
    for _ in 1:1000
        ap += 1.0
        del *= x / ap
        s += del
        abs(del) < abs(s) * 1e-15 && break
    end
    return s * exp(-x + a * log(x) - gln)
end

# Regularized upper incomplete gamma Q(a,x) by modified-Lentz continued
# fraction. Numerical Recipes 3e §6.2 (`gcf`).
function _gamma_q_cf(a::Float64, x::Float64)
    gln = _lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in 1:1000
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) < 1e-15 && break
    end
    return exp(-x + a * log(x) - gln) * h
end

# Upper-tail regularized incomplete gamma Q(a,x) = 1 − P(a,x).
function _gamma_q(a::Float64, x::Float64)
    (x < 0.0 || a <= 0.0) && return NaN
    x == 0.0 && return 1.0
    return x < a + 1.0 ? 1.0 - _gamma_p_series(a, x) : _gamma_q_cf(a, x)
end

"""
    _chi2_pvalue(chi2, ndf) -> Float64

Fit p-value: the probability that a χ² random variable with `ndf` degrees
of freedom exceeds the observed `chi2`, i.e. the survival function
`Q(ndf/2, chi2/2)`. Returns `NaN` for non-positive `ndf` or invalid input.
"""
function _chi2_pvalue(chi2::Real, ndf::Integer)
    (ndf <= 0 || !isfinite(chi2) || chi2 < 0) && return NaN
    return _gamma_q(ndf / 2.0, Float64(chi2) / 2.0)
end

# Header summary for a χ² fit, or `nothing` when not applicable (no fit
# yet, unknown ndata, a likelihood fit with errordef ≠ 1, or ndf ≤ 0).
function _chi2_summary(m::Minuit)
    (m.fmin === nothing || m.ndata === nothing) && return nothing
    m.fcn.up == 1.0 || return nothing
    ndf = m.ndata - n_free(m.params)
    ndf > 0 || return nothing
    chi2 = m.fval
    return (chi2 = chi2, ndf = ndf, ratio = chi2 / ndf,
            p = _chi2_pvalue(chi2, ndf))
end

# ── C: validity checklist ────────────────────────────────────────────────────

# One entry per health check, each `status` ∈ (:ok, :warn, :bad). A failure
# that merely makes errors unreliable is :warn (amber); a failure that
# invalidates the result is :bad (red).
function _validity_checks(m::Minuit)
    bfm = m.fmin
    bfm === nothing && return NamedTuple{(:label, :status), Tuple{String, Symbol}}[]
    internal = bfm.internal
    cov_ok = m.accurate && !internal.made_pos_def && !internal.hesse_failed
    atlimit_ok = isempty(_at_limit_indices(m))
    return [
        (label = "Valid minimum",       status = m.is_valid ? :ok : :bad),
        (label = "EDM below goal",      status = internal.above_max_edm ? :warn : :ok),
        (label = "Below call limit",    status = internal.reached_call_limit ? :bad : :ok),
        (label = "Covariance accurate", status = cov_ok ? :ok : :warn),
        (label = "No params at limit",  status = atlimit_ok ? :ok : :warn),
    ]
end

_check_glyph(s::Symbol) = s === :ok ? "✓" : s === :warn ? "⚠" : "✗"
_check_color(s::Symbol) = s === :ok ? "#1a7f37" : s === :warn ? "#bf8700" : "#cf222e"

function _checklist_text(m::Minuit)
    checks = _validity_checks(m)
    return join((string("[", _check_glyph(c.status), " ", c.label, "]")
                 for c in checks), " ")
end

function _render_checklist_html(io::IO, m::Minuit)
    checks = _validity_checks(m)
    isempty(checks) && return
    print(io, """<div style="margin:0.3em 0">""")
    for c in checks
        color = _check_color(c.status)
        print(io,
            """<span style="display:inline-block;border:1px solid """, color,
            """;border-radius:10px;padding:0 8px;margin:1px 3px;color:""", color,
            """">""", _check_glyph(c.status), " ", _html_escape(c.label), "</span>")
    end
    print(io, "</div>")
end

# ── A + F: correlation heatmap + strong-correlation warnings ─────────────────

# Names of the free (non-fixed) parameters in the internal ordering used
# by the free-covariance / correlation matrix.
function _free_param_names(m::Minuit)
    p = m.params
    return String[p.pars[p.ext_of_int[k]].name for k in 1:n_free(p)]
end

# Off-diagonal free-parameter pairs whose |correlation| exceeds `threshold`.
function _strong_corr_pairs(m::Minuit; threshold::Real = 0.95)
    C = matrix(m; correlation = true)
    (C === nothing || size(C, 1) < 2) && return Tuple{String, String, Float64}[]
    names = _free_param_names(m)
    out = Tuple{String, String, Float64}[]
    n = size(C, 1)
    for j in 2:n, i in 1:(j - 1)
        ρ = C[i, j]
        isfinite(ρ) && abs(ρ) > threshold && push!(out, (names[i], names[j], ρ))
    end
    return out
end

# Format a correlation coefficient to 2 decimals with a Unicode minus.
function _fmt_rho(ρ::Float64)
    isfinite(ρ) || return "─"
    s = @sprintf("%.2f", abs(ρ))
    return ρ < 0 ? string("−", s) : s
end

# Cell background/foreground for a correlation heatmap entry: red for
# positive, blue for negative, intensity ∝ |ρ|, neutral grey on the
# diagonal. White text once the cell is dark enough for contrast.
function _corr_cell_style(ρ::Float64, diag::Bool)
    # Neutral grey on the diagonal AND for a non-finite ρ (a NaN off-diagonal
    # from a broken covariance would otherwise throw in `round(Int, …)`).
    (diag || !isfinite(ρ)) && return "background:#e8e8e8;color:#444"
    a = clamp(abs(ρ), 0.0, 1.0)
    fade = round(Int, 255 * (1 - a))
    bg = ρ >= 0 ? string("rgb(255,", fade, ",", fade, ")") :
                  string("rgb(", fade, ",", fade, ",255)")
    fg = a > 0.6 ? "#fff" : "#000"
    return string("background:", bg, ";color:", fg)
end

function _render_heatmap_html(io::IO, m::Minuit)
    C = matrix(m; correlation = true)
    (C === nothing || size(C, 1) < 2) && return
    names = _free_param_names(m)
    n = size(C, 1)
    print(io, """<div style="margin-top:0.6em">""")
    print(io, """<div style="font-weight:bold;margin-bottom:0.2em">Correlation matrix</div>""")
    print(io, """<table style="border-collapse:collapse;font-family:monospace;font-size:0.9em">""")
    print(io, """<tr><th style="padding:2px 6px"></th>""")
    for nm in names
        print(io, """<th style="border:1px solid #d0d7de;padding:2px 6px;background:#f6f8fa">""",
              _html_escape(nm), "</th>")
    end
    print(io, "</tr>")
    for i in 1:n
        print(io, "<tr>")
        print(io, """<th style="border:1px solid #d0d7de;padding:2px 6px;background:#f6f8fa;text-align:left">""",
              _html_escape(names[i]), "</th>")
        for j in 1:n
            ρ = C[i, j]
            print(io, """<td style="border:1px solid #d0d7de;padding:2px 6px;text-align:right;""",
                  _corr_cell_style(ρ, i == j), "\">", _fmt_rho(ρ), "</td>")
        end
        print(io, "</tr>")
    end
    print(io, "</table></div>")
end

function _render_corr_warning_html(io::IO, m::Minuit)
    pairs = _strong_corr_pairs(m)
    isempty(pairs) && return
    print(io, """<div style="color:#bf8700;margin-top:0.4em">""")
    for (a, b, ρ) in pairs
        print(io, "⚠ <code>", _html_escape(a), "</code> ↔ <code>", _html_escape(b),
              "</code> strongly correlated (ρ = ", _fmt_rho(ρ),
              ") — fit may be poorly conditioned.<br>")
    end
    print(io, "</div>")
end

function _render_corr_warning_text(io::IO, m::Minuit)
    for (a, b, ρ) in _strong_corr_pairs(m)
        println(io, "⚠ `", a, "` ↔ `", b, "` strongly correlated (ρ = ",
                _fmt_rho(ρ), ") — fit may be poorly conditioned.")
    end
end

# ── G: LaTeX export ──────────────────────────────────────────────────────────

# Escape the LaTeX-special characters that can appear in a user parameter
# name so the emitted table compiles.
function _latex_escape(s::AbstractString)
    out = IOBuffer()
    for c in s
        if c in ('_', '%', '&', '#', '$', '{', '}')
            print(out, '\\', c)
        elseif c == '~'
            print(out, "\\textasciitilde{}")
        elseif c == '^'
            print(out, "\\textasciicircum{}")
        elseif c == '\\'
            print(out, "\\textbackslash{}")
        else
            print(out, c)
        end
    end
    return String(take!(out))
end

function _latex_value_cell(r, num)
    v = Float64(r.value)
    if r.minos_lo !== nothing && r.minos_hi !== nothing
        _, vstr, estrs = _round_to_uncertainty(v,
            (abs(Float64(r.minos_hi)), abs(Float64(r.minos_lo))); factor = false)
        histr, lostr = estrs
        return string("\$", num(vstr), "^{+", histr, "}_{-", lostr, "}\$")
    elseif r.hesse !== nothing && isfinite(r.hesse) && r.hesse > 0
        _, vstr, estrs = _round_to_uncertainty(v, (Float64(r.hesse),); factor = false)
        return string("\$", num(vstr), " \\pm ", num(estrs[1]), "\$")
    else
        return string("\$", num(_fmt_g(v)), "\$", r.fixed ? "~(fixed)" : "")
    end
end

"""
    to_latex(m::Minuit; siunitx=true, booktabs=true,
             caption=nothing, label=nothing) -> String

Render the fitted parameters of `m` as a publication-ready LaTeX table.

Defaults to a `booktabs` rule set with `siunitx` `\\num{}` numbers.
Asymmetric MINOS errors (when `minos!` has run) are written as
`\\num{x}^{+hi}_{-lo}`; otherwise a symmetric `\\num{x} \\pm \\num{e}` is
used. Numbers are rounded to the uncertainty (1–2 significant figures on
the error). Fixed parameters show the value alone, tagged `(fixed)`.

Set `siunitx=false` for plain numbers (no `\\num{}`), `booktabs=false`
for `\\hline` rules, and pass `caption`/`label` to wrap the `tabular` in a
`table` float.

Requires `\\usepackage{booktabs}` and `\\usepackage{siunitx}` in the
document preamble (unless the corresponding option is disabled).

# Example

```julia
m = Minuit(p -> (p[1]-1)^2 + (p[2]-2)^2, [0.0, 0.0]; names=["mass", "width"])
migrad!(m)
print(to_latex(m))
```

emits (numbers rounded to the uncertainty; trailing zeros stripped):

```latex
\\begin{tabular}{l c}
\\toprule
Parameter & Value \\\\
\\midrule
mass & \$\\num{1} \\pm \\num{1}\$ \\\\
width & \$\\num{2} \\pm \\num{1}\$ \\\\
\\bottomrule
\\end{tabular}
```
"""
function to_latex(m::Minuit; siunitx::Bool = true, booktabs::Bool = true,
                  caption = nothing, label = nothing)
    num(s) = siunitx ? string("\\num{", s, "}") : s
    rule_top = booktabs ? "\\toprule" : "\\hline"
    rule_mid = booktabs ? "\\midrule" : "\\hline"
    rule_bot = booktabs ? "\\bottomrule" : "\\hline"
    float = caption !== nothing || label !== nothing
    io = IOBuffer()
    if float
        println(io, "\\begin{table}")
        println(io, "\\centering")
    end
    println(io, "\\begin{tabular}{l c}")
    println(io, rule_top)
    println(io, "Parameter & Value \\\\")
    println(io, rule_mid)
    for i in 1:n_pars(m.params)
        r = _param_row_data(m, i)
        println(io, _latex_escape(r.name), " & ", _latex_value_cell(r, num), " \\\\")
    end
    println(io, rule_bot)
    println(io, "\\end{tabular}")
    caption !== nothing && println(io, "\\caption{", caption, "}")
    label !== nothing && println(io, "\\label{", label, "}")
    float && println(io, "\\end{table}")
    return String(take!(io))
end

"""
    to_latex(e::MinosError; value=e.min_par_value, siunitx=true) -> String

Render a single asymmetric MINOS result as `\\num{value}^{+hi}_{-lo}`
(no surrounding math delimiters), suitable for dropping into running text.

If either MINOS side failed to converge (a non-finite `upper`/`lower`),
there is no meaningful asymmetric error, so the central value alone is
returned (`\\num{value}`).
"""
function to_latex(e::MinosError; value::Real = e.min_par_value, siunitx::Bool = true)
    num(s) = siunitx ? string("\\num{", s, "}") : s
    (isfinite(e.upper) && isfinite(e.lower)) || return num(_fmt_g(Float64(value)))
    _, vstr, estrs = _round_to_uncertainty(Float64(value),
        (abs(e.upper), abs(e.lower)); factor = false)
    histr, lostr = estrs
    return string(num(vstr), "^{+", histr, "}_{-", lostr, "}")
end
