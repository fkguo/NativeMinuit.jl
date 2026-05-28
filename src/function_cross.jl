# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# function_cross.jl — MnFunctionCross.
#
# Mirrors reference/Minuit2_cpp/src/MnFunctionCross.cxx:25-512, including
# the L300/L460/L500 control-flow (extension, linear extrapolation,
# parabolic root-find) — see Phase 1.x A3/A4 work.
#
# Given a converged minimum (state, fmin), a parameter index i, and a
# scan direction, find the value `α` such that:
#
#     min_{x_{-i}} f(x_i = x_min_i + α·step_i, x_{-i}) = fmin + up
#
# where `up` is the ErrorDef (1.0 for χ², 0.5 for NLL) and step_i is
# a step in parameter i (positive or negative). The minimization at
# each α is over all OTHER parameters with x_i FIXED.
#
# The algorithm is a parabolic root-find with up to 15 inner-MIGRAD
# iterations:
#
#   1. Initial MIGRAD with x_i = x_min_i + step_i (α = 1).
#   2. Quadratic estimate of α at aim: `√(up/(f - fmin)) - 1`.
#   3. Iterate: MIGRAD at new α, parabolic update, until either
#      (a) `|f - aim| < tlf AND |Δα| < tla` → converged,
#      (b) iteration cap or call cap hit,
#      (c) new lower minimum discovered.
#
# Bounded fits are supported via the internal-coord CF wrap in
# `migrad_bounded.jl` / `Minuit.minos!`; this file works in whatever
# coordinate frame the caller provides. The `par_limit` flag in
# `MnCross` is reserved but not raised — see the docstring of
# `function_cross` below for the known-limitation note.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MnCross

Result of `function_cross`. Mirrors C++ `MnCross`
(`reference/Minuit2_cpp/inc/Minuit2/MnCross.h`).

# Fields

- `state::MinimumState` — the state at the crossing (or current best
  if invalid).
- `aopt::Float64` — the step multiplier at the crossing; `NaN` if
  invalid.
- `nfcn::Int` — cumulative FCN calls made by `function_cross`.
- `valid::Bool` — `true` if a crossing was found within tolerance.
- `new_min::Bool` — `true` if a lower minimum was discovered during the
  scan (Phase 1+ should restart MIGRAD here).
- `fcn_limit::Bool` — `true` if the call budget was exhausted.
- `par_limit::Bool` — `true` if a parameter bound was hit (Phase 1+
  only; always `false` in first cut).
- `ext_state::Union{Nothing,Vector{Float64}}` — the inner-bounded-MIGRAD's
  converged EXTERNAL parameter vector at the crossing (bounded path
  only; always `nothing` for unbounded `function_cross`, and `nothing`
  for invalid results). Used by [`MinosError`](@ref) M4 snapshot fields.
"""
struct MnCross
    state::MinimumState
    aopt::Float64
    nfcn::Int
    valid::Bool
    new_min::Bool
    fcn_limit::Bool
    par_limit::Bool
    # M4 (bounded path): the inner-bounded-MIGRAD's converged EXTERNAL
    # parameter vector at the crossing (with par_idx held at the trial
    # ext value). Populated by `function_cross_external` so the caller
    # (`_minos_external_via_function_cross`) can publish a full ext
    # snapshot via `MinosError.{upper,lower}_state`. `nothing` for the
    # unbounded path (caller assembles ext from `state.parameters.x`
    # via `_assemble_crossing_state`) and for invalid results.
    ext_state::Union{Nothing,Vector{Float64}}
end

MnCross(state::MinimumState, aopt::Real, nfcn::Integer; valid=true,
         new_min=false, fcn_limit=false, par_limit=false,
         ext_state::Union{Nothing,Vector{Float64}} = nothing) =
    MnCross(state, Float64(aopt), Int(nfcn), valid, new_min,
            fcn_limit, par_limit, ext_state)

# ─────────────────────────────────────────────────────────────────────────────
# Parabola helpers — Phase 1.x A3/A4 (parallel-review #4 A3/A4).
#
# Mirror C++ `MnParabolaFactory` + `MnParabola` (used by
# `MnFunctionCross.cxx:357-392`). A parabola is `f(x) = A·x² + B·x + C`,
# and the L500 loop solves `f(x) = aim` for the next probe, choosing
# the root with positive slope.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _parabola_fit3(a, f) -> (A, B, C)

Fit a parabola `A·x² + B·x + C` through three points `(a[i], f[i])`,
i = 1, 2, 3. Mirrors `MnParabolaFactory` (Lagrange form).

The points must be distinct in `a`; behavior on near-coincident points
follows the C++ flow (the caller's `dfda <= 0` check rules out the
worst pathologies before this is called).
"""
@inline function _parabola_fit3(a::AbstractVector{<:Real},
                                 f::AbstractVector{<:Real})
    a0, a1, a2 = Float64(a[1]), Float64(a[2]), Float64(a[3])
    f0, f1, f2 = Float64(f[1]), Float64(f[2]), Float64(f[3])
    # Divided differences (numerically symmetric on the three labels):
    #   A = ((f1-f0)/(a1-a0) - (f2-f1)/(a2-a1)) / (a0 - a2)
    d01 = (f1 - f0) / (a1 - a0)
    d12 = (f2 - f1) / (a2 - a1)
    A = (d01 - d12) / (a0 - a2)
    B = d01 - A * (a0 + a1)
    C = f0 - A * a0 * a0 - B * a0
    return A, B, C
end

"""
    _parabola_solve_for_aim(A, B, C, aim, prec) -> Union{Nothing,Tuple{Float64,Float64}}

Solve `A·x² + B·x + C = aim` and return `(x_root, slope)` where the
root is selected by positive slope (`f'(x) = 2A·x + B`). Returns
`nothing` if the discriminant is negative (curvature wrong, no real
root). Mirrors `MnFunctionCross.cxx:365-394`.
"""
@inline function _parabola_solve_for_aim(A::Float64, B::Float64, C::Float64,
                                          aim::Float64, prec::MachinePrecision)
    determ = B * B - 4.0 * A * (C - aim)
    determ < prec.eps && return nothing
    rt = sqrt(determ)
    x1 = (-B + rt) / (2.0 * A)
    x2 = (-B - rt) / (2.0 * A)
    s1 = B + 2.0 * x1 * A
    s2 = B + 2.0 * x2 * A
    # Pick the root with positive slope (function increasing through aim)
    if s2 > 0.0
        return x2, s2
    else
        return x1, s1
    end
end

"""
    _three_point_classify(a, f, aim) -> (ibest, iworst, ileft, iright, iout, noless)

Categorize three (α, f) probes around `aim` per C++
`MnFunctionCross.cxx:303-322` (initial noless count) and
`MnFunctionCross.cxx:412-443` (ileft/iright/iout/ibest inside L500
loop). Indices are 1-based.

- `noless` — number of points with `f < aim`.
- `ileft`/`iright` — left- and right-side anchors (low-side / high-side
  of aim). 0 if absent on that side.
- `iout` — the redundant point (the one to replace next iteration).
  0 if undefined (single-side).
- `ibest`/`iworst` — by `|f - aim|`.
"""
@inline function _three_point_classify(a::AbstractVector{<:Real},
                                        f::AbstractVector{<:Real},
                                        aim::Float64;
                                        default_ibest::Int = 1)
    # `default_ibest` controls the tie-break for `ibest`: C++ initial
    # classifier (lines 303-322) uses `ibest = 2 (0-indexed) = 3 (1-based)`
    # with `ecarmn = |flsb[2]-aim|`; the L500 classifier (lines 412-443)
    # uses `ibest = 0 = 1 (1-based)` with `ecarmn = |aim-flsb[0]|`.
    ileft = 0; iright = 0; iout = 0
    ibest = default_ibest
    iworst = 1
    noless = 0
    ecarmn = abs(f[default_ibest] - aim)
    ecarmx = 0.0
    @inbounds for i in 1:3
        ecart = abs(f[i] - aim)
        if ecart < ecarmn
            ecarmn = ecart
            ibest = i
        end
        if ecart > ecarmx
            ecarmx = ecart
            iworst = i
        end
        if f[i] > aim
            # right side: C++ MnFunctionCross.cxx:426-434
            if iright == 0
                iright = i
            elseif f[i] > f[iright]
                # new is farther above aim than current iright → new is redundant
                iout = i
            else
                # new is closer to aim than current iright → swap
                iout = iright
                iright = i
            end
        else
            # left side (f <= aim): C++ MnFunctionCross.cxx:435-442
            if ileft == 0
                ileft = i
            elseif f[i] < f[ileft]
                # new is farther below aim → new is redundant
                iout = i
            else
                # new is closer to aim than current ileft → swap
                iout = ileft
                ileft = i
            end
            if f[i] < aim
                noless += 1
            end
        end
    end
    return ibest, iworst, ileft, iright, iout, noless, ecarmn, ecarmx
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared cross-search core — Phase 1.x A3/A4.
#
# Implements the C++ MnFunctionCross.cxx:117-507 algorithm with the
# `_probe` closure providing the inner-MIGRAD evaluation at a given
# α. Both `function_cross` (single-param) and `function_cross_multi`
# (multi-param) reduce to this core after their probe-closure
# construction.
#
# The closure signature is:
#     _probe(aopt::Float64, max_budget::Integer) -> (FunctionMinimum, nfcn_inc)
#
# Returns an `MnCross` describing the crossing search outcome.
# ─────────────────────────────────────────────────────────────────────────────

function _cross_core(_probe::F, fmin_val::Float64, up::Float64,
                     state_fallback::MinimumState;
                     tlr::Float64 = 0.1,
                     maxcalls::Integer = 1000,
                     prec::MachinePrecision = MachinePrecision(),
                     up_scale::Float64 = 1.0,
                     print_level::Integer = 0) where {F<:Function}
    # P5: `up_scale` (= sigma² for the MnMinos `sigma=k` API) scales the
    # effective ErrorDef so the crossing aim becomes `fmin + up · sigma²`.
    # Mirrors iminuit's `_TemporaryUp(self._fcn, factor=sigma²)` wrapper
    # around MnMinos: all subsequent references to `up` inside the search
    # see `up · sigma²`. Tolerances (`tlf`, `f[1]`-seed) scale together so
    # the algorithm's relative convergence behavior is preserved.
    up_eff = up * up_scale
    aim = fmin_val + up_eff

    # gap M1: header. The C++ analog is `print.Info(...)` calls at
    # MnFunctionCross.cxx:108-122. Print `up_eff` (post-sigma-scale)
    # so the trace reflects what the algorithm actually uses.
    # Outer-guarded so the @sprintf only runs at level ≥ 1 (this is
    # called per MINOS direction; avoiding even one alloc per direction
    # keeps level=0 a clean no-op).
    if print_level >= 1
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("start: fmin=%.10g  up=%.4g  aim=%.10g  tlr=%.4g  maxcalls=%d",
                              fmin_val, up_eff, aim, tlr, maxcalls))
    end

    # **Crossing convergence tolerances are HARDCODED 0.01** per C++
    # MnFunctionCross.cxx:38-40 (the user-supplied `tlr` is repurposed
    # only as the inner-MIGRAD tolerance via 0.5·tlr). Without this
    # override the crossing test would be 10× looser than C++ at the
    # default user `tlr = 0.1` (Opus review #5 BLOCKING #1).
    tlf = 0.01 * up_eff    # crossing function-value tolerance
    tla_base = 0.01        # crossing α-tolerance (scaled per iter)

    maxitr = 15
    nfcn = 0
    ipt = 1               # we treat the cached fmin as alsb[1] = 0, flsb[1] = fmin
    a = Vector{Float64}(undef, 3)
    f = Vector{Float64}(undef, 3)
    a[1] = 0.0
    # f[1] follows C++ line 141: max(min0.Fval(), aminsv + 0.1*up). Since
    # we cache fmin (≡ min0.Fval()), this collapses to fmin + 0.1*up.
    # We skip the α=0 MIGRAD entirely (perf win; documented deviation).
    f[1] = fmin_val + 0.1 * up_eff

    # ── Quadratic seed for α (C++ line 142) ──────────────────────────────
    aopt_seed = sqrt(up_eff / (f[1] - fmin_val)) - 1.0
    # Convergence at α=0 (extremely rare; only fires if user tlr ≥ 1)
    if abs(f[1] - aim) < tlf
        return MnCross(state_fallback, aopt_seed, nfcn; valid=true)
    end
    aopt_seed = clamp(aopt_seed, -0.5, 1.0)

    # ── Probe 1: α = aopt_seed (C++ "min1" / our seed-1 MIGRAD) ────────
    min1, nf1 = _probe(aopt_seed, maxcalls - nfcn)
    nfcn += nf1
    # gap M1: outer-guarded — without this the @sprintf fires per probe
    # at level 0. MINOS triggers `function_cross` 2× per parameter
    # which translates to 4-20+ probes for a typical fit.
    if print_level >= 2
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s",
                              ipt + 1, aopt_seed, fval(min1), min1.is_valid))
    end
    fval(min1) < fmin_val - tlf &&
        return MnCross(min1.state, NaN, nfcn; valid=false, new_min=true)
    min1.reached_call_limit &&
        return MnCross(min1.state, NaN, nfcn; valid=false, fcn_limit=true)
    min1.is_valid || return MnCross(state_fallback, NaN, nfcn; valid=false)
    ipt += 1
    a[2] = aopt_seed
    f[2] = fval(min1)
    dfda = (f[2] - f[1]) / (a[2] - a[1])
    last_min = min1
    aopt = aopt_seed

    # L300 inner-step counter — local to each L300 entry (NOT cumulative
    # ipt). Opus review IMPORTANT #4: C++ uses `it = 0..maxlk` local on
    # each `goto L300`, so step size resets to 0.2 on each redo entry.
    l300_step_count = 0

    @label l300_extend
    # ── L300: while dfda < 0, extend outward ─────────────────────────────
    while dfda < 0.0 && ipt < maxitr
        a[1] = a[2]; f[1] = f[2]
        # C++ line 199: aopt = alsb[0] + 0.2 * (it + 1). `it` starts at 0
        # on each L300 re-entry.
        l300_step_count += 1
        aopt = a[1] + 0.2 * l300_step_count
        m, nf = _probe(aopt, maxcalls - nfcn)
        nfcn += nf
        if print_level >= 2
            _trace_info(print_level, "MnFunctionCross",
                        @sprintf("L300 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s",
                                  ipt + 1, aopt, fval(m), m.is_valid))
        end
        fval(m) < fmin_val - tlf &&
            return MnCross(m.state, NaN, nfcn; valid=false, new_min=true)
        m.reached_call_limit &&
            return MnCross(m.state, NaN, nfcn; valid=false, fcn_limit=true)
        m.is_valid || return MnCross(state_fallback, NaN, nfcn; valid=false)
        ipt += 1
        a[2] = aopt; f[2] = fval(m)
        dfda = (f[2] - f[1]) / (a[2] - a[1])
        last_min = m
        dfda > 0.0 && break
    end
    if dfda <= 0.0
        return MnCross(state_fallback, NaN, nfcn; valid=false)
    end

    @label l460_extrapolate
    # ── L460: linear extrapolation to seed point 3 ───────────────────────
    aopt = a[2] + (aim - f[2]) / dfda
    fdist = min(abs(aim - f[1]), abs(aim - f[2]))
    adist = min(abs(aopt - a[1]), abs(aopt - a[2]))
    tla_loop = abs(aopt) > 1.0 ? tla_base * abs(aopt) : tla_base
    if adist < tla_loop && fdist < tlf
        return MnCross(last_min.state, aopt, nfcn; valid=true)
    end
    if ipt >= maxitr
        return MnCross(state_fallback, NaN, nfcn; valid=false)
    end
    bmin = min(a[1], a[2]) - 1.0
    bmax = max(a[1], a[2]) + 1.0
    aopt = clamp(aopt, bmin, bmax)

    m, nf = _probe(aopt, maxcalls - nfcn)
    nfcn += nf
    if print_level >= 2
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("L460 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s",
                              ipt + 1, aopt, fval(m), m.is_valid))
    end
    fval(m) < fmin_val - tlf &&
        return MnCross(m.state, NaN, nfcn; valid=false, new_min=true)
    m.reached_call_limit &&
        return MnCross(m.state, NaN, nfcn; valid=false, fcn_limit=true)
    m.is_valid || return MnCross(state_fallback, NaN, nfcn; valid=false)
    ipt += 1
    a[3] = aopt; f[3] = fval(m)
    last_min = m

    # ── 3-point classifier + dispatch (C++ lines 303-351) ────────────────
    # Initial classifier biases `ibest` toward the THIRD point on ties
    # (C++ initial: `ibest = 2; ecarmn = |flsb[2]-aim|`). This matters
    # for the all-equal degenerate path (Opus review IMPORTANT #5).
    ibest, iworst, ileft, iright, iout, noless, ecarmn, ecarmx =
        _three_point_classify(a, f, aim; default_ibest = 3)

    # Dispatch on noless (C++ lines 327-351):
    if noless == 1 || noless == 2
        @goto l500_enter
    elseif noless == 0 && ibest != 3
        # All three above aim; third probe not closest → invalid
        return MnCross(state_fallback, NaN, nfcn; valid=false)
    elseif noless == 3 && ibest != 3
        # All three below aim; slope went negative again. Move 3rd
        # point into [2] slot and re-extend outward (re-enter L300).
        a[2] = a[3]; f[2] = f[3]
        dfda = (f[2] - f[1]) / (a[2] - a[1])
        l300_step_count = 0   # reset C++ `it` counter on goto L300
        @goto l300_extend
    else
        # ELSE branch (C++ lines 343-351): the "new straight line thru
        # first two points". Replace iworst with the 3rd probe, recompute
        # dfda from the kept-two-points, re-enter L460. Covers
        # noless ∈ {0, 3} with ibest == 3 (the third probe is best).
        # Opus review BLOCKING #2 — without this branch the algorithm
        # falls into L500 with all 3 points one-sided and returns invalid.
        a[iworst] = a[3]
        f[iworst] = f[3]
        dfda = (f[2] - f[1]) / (a[2] - a[1])
        @goto l460_extrapolate
    end

    @label l500_enter
    # ── L500 loop: parabolic root-find with 3-point window ───────────────
    while ipt < maxitr
        A, B, C = _parabola_fit3(a, f)
        sol = _parabola_solve_for_aim(A, B, C, aim, prec)
        sol === nothing &&
            return MnCross(state_fallback, NaN, nfcn; valid=false)  # curvature wrong sign
        aopt_new, slope = sol

        # Convergence at ibest (C++ line 404)
        tla_l500 = abs(aopt_new) > 1.0 ? tla_base * abs(aopt_new) : tla_base
        if abs(aopt_new - a[ibest]) < tla_l500 && abs(f[ibest] - aim) < tlf
            return MnCross(last_min.state, aopt_new, nfcn; valid=true)
        end

        # Re-classify (L500 inner-loop classifier: C++ lines 412-443 use
        # `ibest = 0, ecarmn = |aim - flsb[0]|` — i.e., default ibest=1
        # in 1-based)
        ibest, _, ileft, iright, iout, _, ecarmn, ecarmx =
            _three_point_classify(a, f, aim; default_ibest = 1)

        # Defensive: if either anchor missing or iout undefined (e.g. all
        # three on one side of aim), give up. This should be rare after
        # the noless-dispatch above filters out all-same-side cases.
        if ileft == 0 || iright == 0 || iout == 0
            return MnCross(state_fallback, NaN, nfcn; valid=false)
        end

        # "Avoid keeping a bad point next time" — C++ line 449
        if ecarmx > 10.0 * abs(f[iout] - aim)
            aopt_new = 0.5 * (aopt_new + 0.5 * (a[iright] + a[ileft]))
        end

        # Acceptable window (C++ lines 452-465)
        smalla = 0.1 * tla_l500
        if abs(slope) > 0.0 && slope * smalla > tlf
            smalla = tlf / slope
        end
        aleft  = a[ileft]  + smalla
        aright = a[iright] - smalla
        aopt_new = clamp(aopt_new, aleft, aright)
        if aleft > aright
            aopt_new = 0.5 * (aleft + aright)
        end

        # Probe at new aopt (C++ lines 481-487)
        m, nf = _probe(aopt_new, maxcalls - nfcn)
        nfcn += nf
        if print_level >= 2
            _trace_info(print_level, "MnFunctionCross",
                        @sprintf("L500 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s",
                                  ipt + 1, aopt_new, fval(m), m.is_valid))
        end
        fval(m) < fmin_val - tlf &&
            return MnCross(m.state, NaN, nfcn; valid=false, new_min=true)
        m.reached_call_limit &&
            return MnCross(m.state, NaN, nfcn; valid=false, fcn_limit=true)
        m.is_valid || return MnCross(state_fallback, NaN, nfcn; valid=false)

        ipt += 1
        # Replace iout with new point (C++ lines 500-502)
        a[iout] = aopt_new; f[iout] = fval(m)
        ibest = iout
        aopt = aopt_new
        last_min = m
    end

    if print_level >= 1
        _trace_warn(print_level, "MnFunctionCross",
                    @sprintf("did not converge in %d iters", maxitr))
    end
    return MnCross(state_fallback, NaN, nfcn; valid=false)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: wrap a user FCN with one parameter fixed at a value.
# Returns a new (n-1)-dim CostFunction.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _fix_one_param(cf::CostFunction, i::Int, v::Float64, n::Int) -> CostFunction

Build an (n-1)-dim `CostFunction` from `cf` (an n-dim FCN) by fixing
the i-th argument to `v`. The returned CostFunction's call counter is
fresh; counts accrued in it must be added back to the outer counter.

Implementation: closure captures `cf.f`, `i`, `v`. Each call assembles
a temporary n-vector by splicing. Phase 1 first cut accepts the per-
call alloc; Phase 1.x can ship a workspace-passing variant.
"""
function _fix_one_param(cf::CostFunction, i::Integer, v::Float64, n::Integer)
    f = cf.f
    up = cf.up
    i_ = Int(i)
    n_ = Int(n)
    # Per-thread splice-buffer pool (Phase G).
    #
    # The closure body writes the free slots from `y` and the fixed slot
    # from `v` on every call. Loop structure identical to Phase A V3 — we
    # *only* hoist the alloc + scale it to per-thread, not change the
    # gather pattern, to keep LLVM's branch + memory layout decisions
    # unchanged.
    #
    # Single-threaded Julia: `Threads.maxthreadid() == 1`, so `full_bufs`
    # has length 1 and `Threads.threadid()` always returns 1 →
    # ZERO behavioral / memory / perf change vs Phase A V3 default.
    #
    # Multi-threaded Julia (`julia -t N`): allocate one buffer per
    # possible threadid (`maxthreadid()` ≥ N, with one extra for the
    # interactive thread on Julia 1.10+). When the inner-gradient
    # `Threads.@threads :static for i in 1:n` calls `cf(xw)` from
    # parallel tasks, each task indexes a distinct `full_bufs[tid]`
    # → no race. Memory cost = N × n × 8 bytes (tiny: 9 × 10 × 8 = 720
    # bytes for the typical M3 + 10D fit).
    #
    # User FCN `f` is the caller's responsibility for thread safety —
    # if `f` has hidden mutable state (cache, RNG, file I/O) the user
    # must guard it (documented in `Minuit(..., threaded_gradient=true)`
    # docstring).
    nbuf = max(1, Threads.maxthreadid())
    full_bufs = [Vector{Float64}(undef, n_) for _ in 1:nbuf]
    wrapped = let full_bufs = full_bufs, i_ = i_, n_ = n_, v = v, f = f
        function (y::AbstractVector{<:Real})
            # `` deliberately omitted on the tid index — bounds-check cost (~1 ns) is negligible vs the FCN call (≥100 ns), and the check protects against silent memory corruption if Julia's threadpool model ever expands at runtime. The body of f(full_buf) is still `` where it matters.
            full_buf = full_bufs[Threads.threadid()]
            @inbounds for k in 1:(i_ - 1)
                full_buf[k] = y[k]
            end
            @inbounds full_buf[i_] = v
            @inbounds for k in (i_ + 1):n_
                full_buf[k] = y[k - 1]
            end
            return f(full_buf)
        end
    end
    return CostFunction(wrapped, up)
end

"""
    _fix_one_param(cf::CostFunctionWithGradient, i, v, n) -> CostFunctionWithGradient

Phase F overload: when the user FCN carries an analytical gradient
`cf.g`, the fixed-parameter wrapper must splice BOTH the function and
its gradient — otherwise the inner cross-search loses the AD path and
silently falls back to numerical-gradient via the `::CostFunction`
overload's plain `CostFunction` wrapping.

The wrapped FCN is identical to the numerical-gradient overload: splice
`v` into slot `i`, pass the (n-1)-vector `y` through. The wrapped
GRADIENT delegates to `cf.g(full)` on the same spliced full-length
vector and returns the (n-1)-vector with slot `i` removed (the
gradient component w.r.t. the fixed parameter is discarded — inner
MIGRAD doesn't see it). Both wrappers use the same lifted `full_buf`
+ `out_buf` strategy as Phase A V3 to keep per-call alloc at zero.

Thread-safety contract is identical to the numerical-gradient
overload: `full_buf` and `out_buf` are closure-captured and shared
across calls within the wrapper's lifetime. Safe under single-
threaded MnFunctionCross / MINOS / MnContours.
"""
function _fix_one_param(cf::CostFunctionWithGradient, i::Integer, v::Float64, n::Integer)
    f = cf.f
    g = cf.g
    up = cf.up
    # Counters are FRESH (not shared with outer cf) — symmetric with the
    # numerical `_fix_one_param(::CostFunction, ...)` overload above.
    # `inner_min.nfcn` and `ContoursError.nfcn` carry the inner delta if
    # callers need to introspect.
    i_ = Int(i)
    n_ = Int(n)
    # Per-thread `full_buf` AND per-thread `out_buf` (the n-1 gradient
    # splice scratch). Same Phase G rationale as the numerical-gradient
    # overload above — single-threaded Julia gets 1 buffer each (zero
    # overhead vs Phase F); multi-threaded gets one per `threadid()`.
    nbuf = max(1, Threads.maxthreadid())
    full_bufs = [Vector{Float64}(undef, n_) for _ in 1:nbuf]
    out_bufs  = [Vector{Float64}(undef, n_ - 1) for _ in 1:nbuf]
    f_wrapped = let full_bufs = full_bufs, i_ = i_, n_ = n_, v = v, f = f
        function (y::AbstractVector{<:Real})
            # `` deliberately omitted on the tid index — bounds-check cost (~1 ns) is negligible vs the FCN call (≥100 ns), and the check protects against silent memory corruption if Julia's threadpool model ever expands at runtime. The body of f(full_buf) is still `` where it matters.
            full_buf = full_bufs[Threads.threadid()]
            @inbounds for k in 1:(i_ - 1)
                full_buf[k] = y[k]
            end
            @inbounds full_buf[i_] = v
            @inbounds for k in (i_ + 1):n_
                full_buf[k] = y[k - 1]
            end
            return f(full_buf)
        end
    end
    g_wrapped = let full_bufs = full_bufs, out_bufs = out_bufs,
                     i_ = i_, n_ = n_, v = v, g = g
        function (y::AbstractVector{<:Real})
            tid = Threads.threadid()
            full_buf = full_bufs[tid]
            out_buf  = out_bufs[tid]
            @inbounds for k in 1:(i_ - 1)
                full_buf[k] = y[k]
            end
            @inbounds full_buf[i_] = v
            @inbounds for k in (i_ + 1):n_
                full_buf[k] = y[k - 1]
            end
            grad_full = g(full_buf)
            # Splice out slot i_: copy the n-1 free-coord components into
            # the pre-allocated out_buf. Avoids a per-call alloc when
            # `g` returns a fresh Vector (the common ForwardDiff case).
            @inbounds for k in 1:(i_ - 1)
                out_buf[k] = Float64(grad_full[k])
            end
            @inbounds for k in (i_ + 1):n_
                out_buf[k - 1] = Float64(grad_full[k])
            end
            return out_buf
        end
    end
    return CostFunctionWithGradient(f_wrapped, g_wrapped, up)
end

# ─────────────────────────────────────────────────────────────────────────────
# Multi-param fix helper (Phase 1.x — for contour_exact / general
# MnFunctionCross calls with npar > 1).
# ─────────────────────────────────────────────────────────────────────────────

"""
    _fix_multi_params(cf::CostFunctionWithGradient, par_idxs, v, n)
        -> CostFunctionWithGradient

Phase F overload (multi-param fix variant of `_fix_one_param` above).
Splices both `cf.f` and `cf.g` so the inner cross-search keeps the
analytical/AD gradient path.
"""
function _fix_multi_params(
    cf::CostFunctionWithGradient,
    par_idxs::AbstractVector{<:Integer},
    v::AbstractVector{<:Real},
    n::Integer,
)
    length(par_idxs) == length(v) ||
        throw(DimensionMismatch("par_idxs / v length mismatch"))
    f = cf.f
    g = cf.g
    up = cf.up
    # Counters are fresh — same rationale as `_fix_one_param` above.
    n_ = Int(n)
    is_fixed = falses(n_)
    fixed_value = zeros(Float64, n_)
    @inbounds for (idx, k) in enumerate(par_idxs)
        kk = Int(k)
        1 <= kk <= n_ || throw(ArgumentError("par_idx $kk out of bounds for n=$n_"))
        is_fixed[kk] = true
        fixed_value[kk] = Float64(v[idx])
    end
    n_free = n_ - count(is_fixed)
    # Per-thread buffer pools — Phase G threading support.
    nbuf = max(1, Threads.maxthreadid())
    full_bufs = [Vector{Float64}(undef, n_) for _ in 1:nbuf]
    out_bufs  = [Vector{Float64}(undef, n_free) for _ in 1:nbuf]
    f_wrapped = let full_bufs = full_bufs, is_fixed = is_fixed, fixed_value = fixed_value, n_ = n_, f = f
        function (y::AbstractVector{<:Real})
            # `` deliberately omitted on the tid index — bounds-check cost (~1 ns) is negligible vs the FCN call (≥100 ns), and the check protects against silent memory corruption if Julia's threadpool model ever expands at runtime. The body of f(full_buf) is still `` where it matters.
            full_buf = full_bufs[Threads.threadid()]
            j = 1
            @inbounds for k in 1:n_
                if is_fixed[k]
                    full_buf[k] = fixed_value[k]
                else
                    full_buf[k] = y[j]
                    j += 1
                end
            end
            return f(full_buf)
        end
    end
    g_wrapped = let full_bufs = full_bufs, out_bufs = out_bufs,
                     is_fixed = is_fixed, fixed_value = fixed_value,
                     n_ = n_, g = g
        function (y::AbstractVector{<:Real})
            tid = Threads.threadid()
            full_buf = full_bufs[tid]
            out_buf  = out_bufs[tid]
            j = 1
            @inbounds for k in 1:n_
                if is_fixed[k]
                    full_buf[k] = fixed_value[k]
                else
                    full_buf[k] = y[j]
                    j += 1
                end
            end
            grad_full = g(full_buf)
            j = 1
            @inbounds for k in 1:n_
                if !is_fixed[k]
                    out_buf[j] = Float64(grad_full[k])
                    j += 1
                end
            end
            return out_buf
        end
    end
    return CostFunctionWithGradient(f_wrapped, g_wrapped, up)
end

function _fix_multi_params(
    cf::CostFunction,
    par_idxs::AbstractVector{<:Integer},
    v::AbstractVector{<:Real},
    n::Integer,
)
    length(par_idxs) == length(v) ||
        throw(DimensionMismatch("par_idxs / v length mismatch"))
    f = cf.f
    up = cf.up
    n_ = Int(n)
    is_fixed = falses(n_)
    fixed_value = zeros(Float64, n_)
    @inbounds for (idx, k) in enumerate(par_idxs)
        kk = Int(k)
        1 <= kk <= n_ || throw(ArgumentError("par_idx $kk out of bounds for n=$n_"))
        is_fixed[kk] = true
        fixed_value[kk] = Float64(v[idx])
    end
    # Per-thread splice-buffer pool — same rationale as `_fix_one_param`
    # above. Single-threaded: 1 buffer, zero overhead. Multi-threaded:
    # safe under inner-gradient parallel calls.
    nbuf = max(1, Threads.maxthreadid())
    full_bufs = [Vector{Float64}(undef, n_) for _ in 1:nbuf]
    wrapped = let full_bufs = full_bufs, is_fixed = is_fixed, fixed_value = fixed_value, n_ = n_, f = f
        function (y::AbstractVector{<:Real})
            # `` deliberately omitted on the tid index — bounds-check cost (~1 ns) is negligible vs the FCN call (≥100 ns), and the check protects against silent memory corruption if Julia's threadpool model ever expands at runtime. The body of f(full_buf) is still `` where it matters.
            full_buf = full_bufs[Threads.threadid()]
            j = 1
            @inbounds for k in 1:n_
                if is_fixed[k]
                    full_buf[k] = fixed_value[k]
                else
                    full_buf[k] = y[j]
                    j += 1
                end
            end
            return f(full_buf)
        end
    end
    return CostFunction(wrapped, up)
end

function _migrad_with_multi_fixed(
    cf::AbstractCostFunction,
    state::MinimumState,
    par_idxs::AbstractVector{<:Integer},
    v::AbstractVector{<:Real};
    tol::Float64,
    maxcalls::Integer,
    prec::MachinePrecision,
    strategy::Strategy = Strategy(0),
    warm_state::Union{Nothing,MinimumState} = nothing,
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    print_level::Integer = 0,
)
    n = length(state.parameters)
    is_fixed = falses(n)
    @inbounds for k in par_idxs
        is_fixed[Int(k)] = true
    end
    n_free_inner = n - count(is_fixed)
    if n_free_inner == 0
        # All parameters fixed — no inner MIGRAD needed, just evaluate FCN
        # at the fully-fixed point. This is the typical 2D-contour case
        # (n=2, npar=2).
        full = Vector{Float64}(undef, n)
        @inbounds for (idx, k) in enumerate(par_idxs)
            full[Int(k)] = Float64(v[idx])
        end
        @inbounds for k in 1:n
            if !is_fixed[k]
                full[k] = state.parameters.x[k]
            end
        end
        f_val = Float64(cf.f(full))
        # Build a degenerate FunctionMinimum representing this evaluation
        fake_par = MinimumParameters(full, f_val)
        fake_err = MinimumError(Symmetric(Matrix{Float64}(undef, 0, 0), :U), MnHesseValid)
        fake_grad = FunctionGradient(0)
        fake_state = MinimumState(fake_par, fake_err, fake_grad, 0.0, 1)
        fake_min = FunctionMinimum(fake_state, fake_state, cf.up;
                                    is_valid = true)
        return fake_min, 1
    end

    cf_fixed = _fix_multi_params(cf, par_idxs, v, n)
    inner_strategy = Strategy(max(0, strategy.level - 1))

    # WARM-START PATH: when `warm_state` is supplied (the previous
    # parabolic-fit probe's converged inner state, in the same
    # (n - npar)-dim free-coord space as the NEW cf_fixed), skip
    # `seed_state` entirely. `warm_restart_state` re-evaluates the new
    # cf_fixed at the warm position (1 FCN call), refines the gradient
    # using the prev gradient's step sizes (Numerical2P converges in
    # ~1 cycle), and KEEPS the prev inv_hessian. Then `_migrad_loop`'s
    # DFP iterations start from the warm Hessian — typically converges
    # in 2-3 iters instead of 5-10. Mirrors C++ MnFunctionCross.cxx:
    # 106-216, where a single MnMigrad instance reuses MnUserParameterState
    # across the 3-15 parabolic iterations.
    #
    # Falls back to cold path (full seed_state) when warm_restart_state
    # returns nothing: dim mismatch, invalid prev state, or any
    # negative g2 in the refined gradient (caller's seed_state path
    # handles those via initial_gradient + negative_g2_line_search).
    if warm_state !== nothing && length(warm_state) == n_free_inner
        seed_warm = warm_restart_state(warm_state, cf_fixed;
                                        strategy = inner_strategy, prec = prec)
        if seed_warm !== nothing
            inner_min = migrad(cf_fixed, seed_warm;
                                tol = tol, maxfcn = Int(maxcalls),
                                strategy = inner_strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient,
                                print_level = print_level)
            return inner_min, ncalls(cf_fixed)
        end
    end

    # COLD PATH: build initial point + errors from the OUTER minimum's
    # converged x + sqrt(2·up·V[k,k]) per-coord errors.
    y0 = Vector{Float64}(undef, n_free_inner)
    errs = Vector{Float64}(undef, n_free_inner)
    V = state.error.inv_hessian
    scale = 2.0 * cf.up
    x_min = state.parameters.x
    j = 1
    @inbounds for k in 1:n
        is_fixed[k] && continue
        y0[j] = x_min[k]
        errs[j] = sqrt(max(scale * V[k, k], prec.eps2))
        j += 1
    end

    inner_min = migrad(cf_fixed, y0, errs;
                        tol = tol, maxfcn = Int(maxcalls),
                        strategy = inner_strategy, prec = prec,
                        scratch = scratch,
                        threaded_gradient = threaded_gradient,
                        print_level = print_level)
    return inner_min, ncalls(cf_fixed)
end

function function_cross_multi(
    fmin::FunctionMinimum,
    cf::AbstractCostFunction,
    par_idxs::AbstractVector{<:Integer},
    pmid::AbstractVector{<:Real},
    pdir::AbstractVector{<:Real};
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    sigma::Real = 1.0,
    print_level::Integer = 0,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    state = fmin.state
    n = length(state.parameters)
    npar = length(par_idxs)
    length(pmid) == npar ||
        throw(DimensionMismatch("pmid length $(length(pmid)) != par_idxs length $npar"))
    length(pdir) == npar ||
        throw(DimensionMismatch("pdir length $(length(pdir)) != par_idxs length $npar"))
    n >= npar ||
        throw(ArgumentError("function_cross_multi needs n >= npar (got n=$n, npar=$npar)"))

    fmin_val = state.parameters.fval
    up = cf.up
    pmid_f = Float64[Float64(pmid[i]) for i in 1:npar]
    pdir_f = Float64[Float64(pdir[i]) for i in 1:npar]

    # Probe closure: builds the multi-fix vector for a given α and runs
    # the inner MIGRAD. THREADS THE WARM STATE forward across probes
    # (see _migrad_with_multi_fixed for the C++ MnMigrad single-instance
    # rationale).
    #
    # Phase D — also pin a single MigradScratch across all probes of
    # this cross-search. The inner_dim (n - npar) is constant within
    # one function_cross_multi call, so one scratch instance amortizes
    # ~15 vector + 3 matrix allocations across the 3-15 parabolic-fit
    # iterations. When the caller supplies a `scratch` (e.g., from a
    # contour_exact driver that's pooling across multiple cross-searches
    # at the SAME inner_dim), we reuse THAT; otherwise we lazily
    # construct one when the first probe needs it (kept in
    # scratch_holder so the inner-dim==0 degenerate path doesn't
    # allocate at all).
    warm_state_ref = Ref{Union{Nothing,MinimumState}}(nothing)
    scratch_holder = Ref{Union{Nothing,MigradScratch}}(scratch)
    let pmid_f = pmid_f, pdir_f = pdir_f, npar = npar,
        warm_state_ref = warm_state_ref,
        scratch_holder = scratch_holder, n = n
        probe = function (aopt::Float64, budget::Integer)
            v_probe = Vector{Float64}(undef, npar)
            @inbounds for i in 1:npar
                v_probe[i] = pmid_f[i] + aopt * pdir_f[i]
            end
            # Lazy: allocate scratch on first non-degenerate probe;
            # subsequent probes reuse. The all-fixed degenerate path
            # (npar == n) inside _migrad_with_multi_fixed short-circuits
            # before touching the scratch, so allocating here is wasted
            # in that corner case — but harmless and tiny.
            n_free_inner = n - npar
            if n_free_inner >= 1
                _get_scratch!(scratch_holder, n_free_inner)
            end
            inner_min, nf = _migrad_with_multi_fixed(
                cf, state, par_idxs, v_probe;
                tol = 0.5 * tlr, maxcalls = budget,
                prec = prec, strategy = strategy,
                warm_state = warm_state_ref[],
                scratch = scratch_holder[],
                threaded_gradient = threaded_gradient,
                print_level = print_level)
            # On successful inner-MIGRAD, update the warm state for the
            # next probe. On failure keep the previous valid warm state
            # (or `nothing` for the cold first probe).
            if inner_min.is_valid
                warm_state_ref[] = inner_min.state
            end
            return inner_min, nf
        end
        return _cross_core(probe, fmin_val, up, state;
                            tlr = Float64(tlr),
                            maxcalls = maxcalls, prec = prec,
                            up_scale = Float64(sigma)^2,
                            print_level = print_level)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run inner MIGRAD with parameter i fixed at value v.
# Returns (inner_min, total_inner_nfcn).
# ─────────────────────────────────────────────────────────────────────────────

function _migrad_with_fixed(
    cf::AbstractCostFunction, state::MinimumState, i::Integer, v::Float64;
    tol::Float64, maxcalls::Integer, prec::MachinePrecision,
    strategy::Strategy = Strategy(0),
    warm_state::Union{Nothing,MinimumState} = nothing,
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    print_level::Integer = 0,
)
    n = length(state.parameters)
    cf_fixed = _fix_one_param(cf, i, v, n)
    # Thread strategy (parallel-review #4 A5 — previously silently
    # defaulted to Strategy(0) regardless of outer arg). Inner uses
    # the outer level minus 1 per C++ MnFunctionCross.cxx:106.
    inner_strategy = Strategy(max(0, strategy.level - 1))

    # WARM-START PATH: see the matching block in _migrad_with_multi_fixed
    # above for the full rationale (MnFunctionCross single-MnMigrad-
    # instance pattern). The MINOS / single-param-cross path threads
    # the prev probe's converged (n-1)-dim inner state, which is the
    # same free-coord space as the new cf_fixed (only `v` changes).
    if warm_state !== nothing && length(warm_state) == n - 1
        seed_warm = warm_restart_state(warm_state, cf_fixed;
                                        strategy = inner_strategy, prec = prec)
        if seed_warm !== nothing
            inner_min = migrad(cf_fixed, seed_warm;
                                tol = tol, maxfcn = Int(maxcalls),
                                strategy = inner_strategy, prec = prec,
                                scratch = scratch,
                                threaded_gradient = threaded_gradient,
                                print_level = print_level)
            return inner_min, ncalls(cf_fixed)
        end
    end

    # COLD PATH: seed from the outer-minimum x (parameter i removed),
    # with per-coord errors derived from the outer inv_hessian.
    # C++ MnUserParameterState constructs free-parameter errors as
    # sqrt(2·up·V[i,i]) (reference/Minuit2_cpp/src/MnUserParameterState.cxx
    # :151-154).
    x_min = state.parameters.x
    y0 = Vector{Float64}(undef, n - 1)
    @inbounds for k in 1:(i - 1)
        y0[k] = x_min[k]
    end
    @inbounds for k in (i + 1):n
        y0[k - 1] = x_min[k]
    end
    errs = Vector{Float64}(undef, n - 1)
    V = state.error.inv_hessian
    scale = 2.0 * cf.up
    @inbounds for k in 1:(i - 1)
        errs[k] = sqrt(max(scale * V[k, k], prec.eps2))
    end
    @inbounds for k in (i + 1):n
        errs[k - 1] = sqrt(max(scale * V[k, k], prec.eps2))
    end

    inner_min = migrad(cf_fixed, y0, errs;
                        tol = tol, maxfcn = Int(maxcalls),
                        strategy = inner_strategy, prec = prec,
                        scratch = scratch,
                        threaded_gradient = threaded_gradient,
                        print_level = print_level)
    return inner_min, ncalls(cf_fixed)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main: function_cross — find the alpha such that min_{x_{-i}}(f) = fmin + up.
# ─────────────────────────────────────────────────────────────────────────────

"""
    function_cross(fmin, cf, par_idx, dir; tlr=0.1, maxcalls=1000,
                   strategy=Strategy(0), prec=MachinePrecision()) -> MnCross

Find the step multiplier α along parameter `par_idx` such that the
constrained-minimum (other params re-optimized) satisfies
`f - fmin = up`. Used by MINOS (asymmetric errors) and contours.

# Arguments

- `fmin::FunctionMinimum` — the converged MIGRAD result.
- `cf::CostFunction` — the user FCN (must match the one used for fmin).
- `par_idx::Integer` — 1-based parameter index to scan along.
- `dir::Real` — sign of the scan direction (+1.0 for upper error, -1.0
  for lower). Combined with the 1-sigma step from `state.error`.

# Keyword arguments

- `tlr::Real=0.1` — tolerance. Internal tolerances `tlf = tlr·up`
  and `tla = tlr` mirror C++ MnFunctionCross.cxx:42-44.
- `maxcalls::Integer=1000` — call budget across all inner MIGRADs.
- `strategy::Strategy=Strategy(0)` — passed to inner MIGRADs.
- `prec::MachinePrecision`.
- `sigma::Real=1.0` — confidence level in σ-units. The crossing aim
  becomes `fmin + up · sigma²` (mirrors iminuit's `minos(cl=)` scaling
  of `MnFunctionCross.aim`). At sigma=1 the behavior is C++-identical
  to a single MnFunctionCross call; at sigma=k the returned `aopt`
  converges to ≈ k (in the parabolic approximation), so the caller's
  `aopt · σ_1` product is the k-σ error.

# Returns

[`MnCross`](@ref). Check `.valid`, `.new_min`, `.fcn_limit` to interpret.

# Known limitations

- `par_limit` is never raised. Bounded MINOS works correctly (via the
  internal-coord CF wrap in `Minuit.minos!` and `migrad_bounded.jl`),
  but the boundary-saturation flag isn't surfaced. Equivalent C++ flag:
  `MnCross::CrossParLimit()`.
- Inner MIGRAD strategy is `max(0, strategy.level - 1)`, matching C++
  `MnFunctionCross.cxx:106`.
"""
function function_cross(
    fmin::FunctionMinimum,
    cf::AbstractCostFunction,
    par_idx::Integer,
    dir::Real;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    sigma::Real = 1.0,
    print_level::Integer = 0,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    state = fmin.state
    n = length(state.parameters)
    1 <= par_idx <= n ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n"))
    n > 1 ||
        throw(ArgumentError("function_cross requires n > 1 (cannot fix the only parameter)"))

    x_min = state.parameters.x
    fmin_val = state.parameters.fval
    up = cf.up

    # 1-sigma external step along par_idx (Phase 1 first cut: assume
    # no bounds → internal == external; sigma = sqrt(2·up·V[i,i])).
    sigma_i = sqrt(max(2.0 * up * state.error.inv_hessian[par_idx, par_idx],
                        prec.eps2))
    step = Float64(dir) * sigma_i
    x_pivot = x_min[par_idx]

    # Probe closure: runs inner MIGRAD at α along par_idx. Threads the
    # warm STATE across probes, AND pins one MigradScratch (inner_dim
    # = n - 1, constant within this call). See function_cross_multi
    # above for the full Phase D rationale.
    warm_state_ref = Ref{Union{Nothing,MinimumState}}(nothing)
    scratch_holder = Ref{Union{Nothing,MigradScratch}}(scratch)
    let x_pivot = x_pivot, step = step,
        warm_state_ref = warm_state_ref,
        scratch_holder = scratch_holder, n = n
        probe = function (aopt::Float64, budget::Integer)
            v = x_pivot + aopt * step
            _get_scratch!(scratch_holder, n - 1)
            inner_min, nf = _migrad_with_fixed(cf, state, par_idx, v;
                                tol = 0.5 * tlr, maxcalls = budget,
                                prec = prec, strategy = strategy,
                                warm_state = warm_state_ref[],
                                scratch = scratch_holder[],
                                threaded_gradient = threaded_gradient,
                                print_level = print_level)
            if inner_min.is_valid
                warm_state_ref[] = inner_min.state
            end
            return inner_min, nf
        end
        return _cross_core(probe, fmin_val, up, state;
                            tlr = Float64(tlr),
                            maxcalls = maxcalls, prec = prec,
                            up_scale = Float64(sigma)^2,
                            print_level = print_level)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Bound-aware external-coord MINOS — mirrors C++ MnMinos.cxx:119-131
# architecture. This is the proper bounded-parameter MINOS path; it
# replaces the previous int-coord search + post-conversion approach
# that triggered the Jacobian-sign / sign-cross issues round-1 and
# round-2 reviewers caught.
#
# Architecture (vs the int-coord function_cross above):
#   - alpha-search operates on EXTERNAL coords. The "step" passed to
#     `_cross_core` is the external direction (truncated against any
#     bound BEFORE the search starts).
#   - Inner MIGRAD at each probe uses the bounded migrad API (with the
#     scanning parameter FIXED at the trial external value). Bounds on
#     other free parameters are respected.
#   - Sign convention is automatic: for `dir = +1` (upper search) the
#     ext step is positive → positive aopt → positive ext error; for
#     `dir = -1` (lower search) the ext step is negative → positive aopt
#     → negative ext error. No Jacobian-swap or sign-cross detection
#     needed.
#
# C++ reference: reference/Minuit2_cpp/src/MnMinos.cxx:119-131 truncates
# the trial value against the parameter limit BEFORE constructing
# `xmid` / `xdir`; the alpha search inside MnFunctionCross is then in
# the (truncated) external direction.
# ─────────────────────────────────────────────────────────────────────────────

"""
    function_cross_external(bfm, cf, par_idx, dir; tlr=0.1, maxcalls=1000,
                             strategy=Strategy(0), prec=MachinePrecision()) -> MnCross

Bound-aware MINOS one-sided search. `bfm::BoundedFunctionMinimum` is
the converged bounded fit; `cf::CostFunction` is the USER FCN (takes
external coords); `par_idx::Integer` is the 1-based external parameter
index; `dir::Real` is +1 (upper search) or -1 (lower search).

The 1σ external step is truncated against `par.lower` / `par.upper`
before the search begins. The returned `MnCross.aopt` is the multiplier
on the (possibly truncated) step; `aopt * step_ext = ext_error`.

Sets `par_limit = true` when the 1σ step is fully truncated by the
bound (no extrapolation possible).
"""
function function_cross_external(
    bfm,                # ::BoundedFunctionMinimum — typed at use site
    cf::AbstractCostFunction,
    par_idx::Integer,
    dir::Real;
    tlr::Real = 0.1,
    maxcalls::Integer = 1000,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
    threaded_gradient::Bool = false,
    sigma::Real = 1.0,
    print_level::Integer = 0,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    params = bfm.params
    n_total = n_pars(params)
    1 <= par_idx <= n_total ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n_total"))
    par = params.pars[par_idx]
    is_fixed(par) &&
        throw(ArgumentError("Cannot run MINOS on fixed parameter $par_idx"))
    n_free(params) > 1 ||
        throw(ArgumentError("function_cross_external requires n_free > 1"))

    ext_min = bfm.ext_values[par_idx]
    ext_err = bfm.ext_errors[par_idx]
    direction = Float64(dir)
    abs(direction) ≈ 1.0 ||
        throw(ArgumentError("dir must be ±1, got $direction"))

    # Sanity: if Hesse error is zero or non-finite (e.g. HESSE failed),
    # fall back to the user's step from the Parameters seed.
    if !isfinite(ext_err) || ext_err <= 0
        ext_err = abs(par.error)
        ext_err > 0 ||
            throw(ArgumentError("Cannot run MINOS: parameter $par_idx has zero error"))
    end

    # ── Truncate trial step against the parameter bound (C++ MnMinos
    #    :119-131). Use side-specific predicates only — `has_limits` is
    #    `has_lower_limit OR has_upper_limit`, so testing it here would
    #    incorrectly trigger the wrong-side clamp on one-sided
    #    parameters (par.lower=NaN → max(x, NaN)=NaN).
    val_trial = ext_min + direction * ext_err
    if direction > 0 && has_upper_limit(par)
        val_trial = min(val_trial, par.upper)
    end
    if direction < 0 && has_lower_limit(par)
        val_trial = max(val_trial, par.lower)
    end
    step_ext = val_trial - ext_min
    # Saturated against the bound. Threshold is 0.1% of the nominal
    # 1σ step (not machine epsilon): MIGRAD on bounded params often
    # converges to within numerical-stability-roundoff of the bound
    # (e.g., 2e-9 close to a bound at 10), not bit-exact. A step
    # that's < 0.1% of the natural error scale is physically
    # saturated — no useful extrapolation. This matches iminuit's
    # behavior, which treats "within ~ulps_of_bound × scale" as
    # at-limit.
    #
    # M4: attach `bfm.ext_values` as the `ext_state` — the converged
    # outer-MIGRAD ext vector IS the "state at the bound" the user
    # wants to publish on the at-limit side (codex review nb-O2).
    if abs(step_ext) <= 1e-3 * ext_err
        return MnCross(bfm.internal.state, 0.0, 0;
                        valid = false, par_limit = true,
                        ext_state = copy(bfm.ext_values))
    end

    fmin_val = fval(bfm)
    up = cf.up
    inner_strategy = Strategy(max(0, strategy.level - 1))

    # aulim-style detection: the maximum alpha that keeps the trial
    # value inside the bound. Mirrors C++ MnFunctionCross.cxx:64-104
    # — when aopt exceeds aulim, the search hit the bound; this is a
    # `par_limit` event, not a `fcn_limit` event. Tracked via Ref
    # captured in the probe closure (Julia idiom for "mutable state
    # observable across a closure call").
    aulim = if step_ext > 0 && has_upper_limit(par)
        (par.upper - ext_min) / step_ext
    elseif step_ext < 0 && has_lower_limit(par)
        (par.lower - ext_min) / step_ext
    else
        Inf
    end
    limset = Ref(false)
    # M4: capture the inner-bounded-MIGRAD's converged EXTERNAL parameter
    # vector at the last successful probe so the caller can publish a
    # full ext snapshot via `MinosError.{upper,lower}_state`. `_cross_core`
    # returns the converged `last_min.state` (internal coords); we hold
    # the ext slice in a closure-captured Ref that the probe overwrites
    # on each `inner_bfm.is_valid` call. After `_cross_core` returns
    # valid, `last_ext_state[]` holds the converged ext snapshot.
    last_ext_state = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    # Build probe closure. At each alpha, set par to `ext_min + aopt *
    # step_ext` (clamped against the bound if aopt > aulim), copy ALL
    # other free params from the CONVERGED minimum (NOT the user seed
    # — codex round-3 catch: starting from the seed loses the
    # MIGRAD-converged information and can land the inner search in a
    # different basin for hard FCNs), build Parameters with par_idx
    # FIXED at the trial value, run bounded migrad on the inner problem.
    let par_idx = Int(par_idx), step_ext = step_ext,
        ext_min = ext_min, par = par, params = params,
        aulim = aulim, limset = limset,
        last_ext_state = last_ext_state,
        ext_values = bfm.ext_values,
        inner_strategy = inner_strategy
        probe = function (aopt::Float64, budget::Integer)
            # aulim check: if alpha overshoots the bound, clamp and
            # mark limset. Round-3 BLOCKING fix.
            clamped_aopt = aopt
            if aopt > aulim
                clamped_aopt = aulim
                limset[] = true
            end
            ext_val = ext_min + clamped_aopt * step_ext
            # Defensive re-clamp: rounding might leave ext_val just past
            # the bound by 1 ulp.
            if has_upper_limit(par)
                ext_val = min(ext_val, par.upper)
            end
            if has_lower_limit(par)
                ext_val = max(ext_val, par.lower)
            end
            inner_pars = MinuitParameter[]
            sizehint!(inner_pars, length(params.pars))
            for (i, p) in enumerate(params.pars)
                # Start each non-scanned param from the CONVERGED ext
                # value (bfm.ext_values[i]), not the user's original
                # seed (p.value). Without this, the inner MIGRAD
                # restarts from the user's initial guess every probe.
                converged_v = ext_values[i]
                lo = isnan(p.lower) ? NaN : p.lower
                hi = isnan(p.upper) ? NaN : p.upper
                if i == par_idx
                    push!(inner_pars, MinuitParameter(p.name, ext_val,
                                                       p.error;
                                                       lower = lo, upper = hi,
                                                       fixed = true))
                else
                    push!(inner_pars, MinuitParameter(p.name, converged_v,
                                                       p.error;
                                                       lower = lo, upper = hi,
                                                       fixed = p.fixed))
                end
            end
            inner_params = Parameters(inner_pars, prec)
            inner_bfm = migrad(cf, inner_params;
                                tol = 0.5 * tlr, maxfcn = Int(budget),
                                strategy = inner_strategy, prec = prec,
                                threaded_gradient = threaded_gradient,
                                print_level = print_level)
            # M4: snapshot the converged ext values so the caller can
            # publish them on `MinosError.{upper,lower}_state`. Only
            # update when the inner MIGRAD reached a valid minimum —
            # invalid probes' ext vectors are not physically meaningful.
            if inner_bfm.internal.is_valid
                last_ext_state[] = copy(inner_bfm.ext_values)
            end
            # nfcn correctly = the inner bounded migrad's call count
            # (NOT ncalls(cf), which is the OUTER cf and never gets
            # incremented because bounded migrad wraps cf into
            # cf_internal with its own counter). Round-3 BLOCKING fix.
            return inner_bfm.internal, nfcn(inner_bfm)
        end
        result = _cross_core(probe, fmin_val, up, bfm.internal.state;
                              tlr = Float64(tlr),
                              maxcalls = maxcalls, prec = prec,
                              up_scale = Float64(sigma)^2,
                              print_level = print_level)
        # Round-4 codex BLOCKING fix: the probe clamps ext_val when
        # aopt > aulim, but `_cross_core` still records the UNCLAMPED
        # aopt. If the inner-fval at the clamped bound happens to be
        # close enough to aim, _cross_core can return valid=true with
        # aopt > aulim — the published ext error would then exceed
        # the physical distance to the bound (e.g., user with bound
        # at +0.997σ sees an asymmetric error of +1.39σ). This is
        # a silently wrong physics result.
        #
        # Mirroring C++ MnFunctionCross.cxx:494-496 (CrossParLimit
        # raised when limset && Fval < aim): if the returned aopt
        # exceeds aulim, the search effectively converged against the
        # constant-f region past the bound. Snap aopt to aulim, mark
        # par_limit, invalidate the side (the crossing wasn't found
        # before hitting the bound).
        if result.valid && aulim < Inf && result.aopt > aulim
            result = MnCross(result.state, aulim, result.nfcn;
                              valid = false, par_limit = true,
                              new_min = result.new_min,
                              fcn_limit = result.fcn_limit,
                              ext_state = last_ext_state[])
        end
        # Round-3 partial-truncation fix: search exited invalid AND
        # we hit the bound during the walk → relabel as par_limit.
        if !result.valid && limset[] && !result.par_limit
            result = MnCross(result.state, result.aopt, result.nfcn;
                              valid = false, par_limit = true,
                              new_min = result.new_min,
                              fcn_limit = result.fcn_limit,
                              ext_state = last_ext_state[])
        end
        # M4: attach the captured ext snapshot for the SUCCESS path
        # too — the inner-MIGRAD's converged ext values at the crossing.
        # `_cross_core` reconstructs MnCross internally without ext_state,
        # so we rebuild here ONLY when the result is valid and we have a
        # captured snapshot (we leave failure-mode results alone to keep
        # ext_state==nothing as their semantic; bounded `par_limit` got
        # the snapshot via the wraps above when relevant).
        if result.valid && last_ext_state[] !== nothing &&
           result.ext_state === nothing
            result = MnCross(result.state, result.aopt, result.nfcn;
                              valid = result.valid,
                              new_min = result.new_min,
                              fcn_limit = result.fcn_limit,
                              par_limit = result.par_limit,
                              ext_state = last_ext_state[])
        end
        return result
    end
end
