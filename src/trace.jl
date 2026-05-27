# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# trace.jl — print_level / iteration trace helpers (gap M1).
#
# Wired through `_migrad_loop`, `hesse`, `_cross_core`, `minos` to mirror
# iminuit's `m.print_level = N` debugging interface. Mirrors C++ MnPrint.cxx
# + MnTraceObject.cxx but uses Julia stdlib Logging idioms — no global TLS
# state, `print_level` threads through as a plain kwarg.
#
# Levels (match iminuit / C++ Minuit2 conventions):
#   0 — silent (default).
#   1 — per OUTER iteration of the inner loop: (iter, fval, edm, dcovar,
#       ncalls). For MIGRAD this is each accepted DFP step plus the
#       Strategy ≥ 1 HESSE refinement banner. For HESSE this is a header
#       + a final summary. For MINOS this is each ±σ direction header
#       + the `function_cross` start banner + non-convergence warnings.
#   2 — adds inner-loop diagnostics: line-search outcomes, pos-def
#       events, HESSE per-parameter diagonal pass, HESSE per-pair
#       off-diagonal, MINOS per-probe events inside `_cross_core`.
#   3 — adds full parameter + gradient vectors via @debug.
#
# Levels 1, 2 emit via `@info`; level 3 via `@debug`. The default Julia
# ConsoleLogger sends @info to stderr and filters @debug — wrap a call
# in `with_logger(ConsoleLogger(stderr, Logging.Debug))` to capture
# level-3 output (this is what test/test_print_level.jl does).
#
# All call sites guard with `print_level >= N` BEFORE invoking these
# helpers, so the cost when print_level == 0 is a single comparison +
# branch (≈1ns); the formatting work is gated.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _trace_iter(level, prefix, iter, fval, edm, dcovar, ncalls)

Level-1 per-iteration one-liner. Mirrors C++ `MnPrint::Oneline`
(reference/Minuit2_cpp/src/MnPrint.cxx:200-210):

```
[MnMigrad] iter=   3  fval=12.345678  edm=1.234e-3  dcovar=0.04  ncalls=42
```

`iter < 0` suppresses the iteration prefix (one-shot summary lines).
"""
@inline function _trace_iter(level::Integer, prefix::AbstractString,
                              iter::Integer, fval::Real, edm::Real,
                              dcovar::Real, ncalls::Integer)
    level >= 1 || return nothing
    iter_str = iter >= 0 ? (@sprintf "iter=%4d  " iter) : ""
    msg = @sprintf("[%s] %sfval=%.10g  edm=%.10g  dcovar=%.4g  ncalls=%d",
                   prefix, iter_str, Float64(fval), Float64(edm),
                   Float64(dcovar), Int(ncalls))
    @info msg
    return nothing
end

"""
    _trace_info(level, prefix, msg; min_level=1)

Conditional `@info` emit at `min_level` or higher. Use for headers
(`min_level=1`) and inner-loop step descriptions (`min_level=2`).
"""
@inline function _trace_info(level::Integer, prefix::AbstractString,
                              msg::AbstractString; min_level::Integer = 1)
    level >= min_level || return nothing
    @info "[$prefix] $msg"
    return nothing
end

"""
    _trace_warn(level, prefix, msg)

Level-1 warning (anomalous-but-non-fatal events: pos-def fixup, new
minimum during MINOS scan, etc.).
"""
@inline function _trace_warn(level::Integer, prefix::AbstractString,
                              msg::AbstractString)
    level >= 1 || return nothing
    @warn "[$prefix] $msg"
    return nothing
end

"""
    _trace_state(level, prefix, iter, x, grad)

Level-3 full-state trace via `@debug`. Emits the parameter and
gradient vectors keyed by `x`/`grad`. Capture by configuring the
active logger at `Logging.Debug` level (e.g. via
`Logging.with_logger(ConsoleLogger(stderr, Logging.Debug))`).

The `x` and `grad` arguments alias mutable scratch buffers inside
`_migrad_loop` — by the time a logger like `Test.TestLogger` reads a
stored record, the buffer may have been overwritten by a later DFP
iteration (Phase D's ping-pong scratch + `numerical_gradient!`'s
in-place fills). To give each record an immutable snapshot we
`copy(x)` / `copy(grad)` AT EMIT TIME inside the `@debug` kwarg
list — `@debug`'s lazy kwarg evaluation means the copies allocate
only when (a) `print_level >= 3` AND (b) the active logger accepts
`Debug` (so the typical level-0/1/2 path and a level-3 user with a
default `ConsoleLogger(min=Info)` both pay zero).
"""
@inline function _trace_state(level::Integer, prefix::AbstractString,
                               iter::Integer,
                               x::AbstractVector{<:Real},
                               grad::AbstractVector{<:Real})
    level >= 3 || return nothing
    @debug "[$prefix] state at iter=$iter" x=copy(x) grad=copy(grad)
    return nothing
end
