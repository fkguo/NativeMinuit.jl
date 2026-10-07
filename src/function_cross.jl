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
# iterations, with α measured from the HESSE ±1σ point as in C++:
#
#   1. Inner MIGRAD at the ±1σ point (α = 0); if its value is already
#      within `tlf = 0.01·up` of the aim, return the quadratic-model
#      estimate `√(up/(f - fmin)) - 1` (exact for a parabolic profile).
#   2. Otherwise probe there, extend outward while the slope is negative,
#      extrapolate linearly, then iterate a parabola through the three
#      most recent points until either
#      (a) the predicted α is within `tla = 0.01` of the best probed α
#          AND that probe's value is within `tlf` of the aim → converged,
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
  invalid. From [`function_cross`](@ref) / [`function_cross_external`](@ref)
  it multiplies the HESSE ±1σ step **from the minimum** (so
  `aopt · σ` is the MINOS error, ≈ 1 for a parabolic profile); from
  [`function_cross_multi`](@ref) it multiplies `pdir` from `pmid`, which
  is the C++ `MnCross::Value()` convention.
- `nfcn::Int` — cumulative FCN calls made by `function_cross`.
- `valid::Bool` — `true` if a crossing was found within tolerance.
- `new_min::Bool` — `true` if a lower minimum was discovered during the
  scan (Phase 1+ should restart MIGRAD here).
- `fcn_limit::Bool` — `true` if the call budget was exhausted.
- `par_limit::Bool` — `true` when a probe converged below the aim at the
  parameter bound (C++ `CrossParLimit`). C++ marks that result *valid*
  with the limit flag raised; here it is `valid = false, par_limit =
  true`, and the user-facing [`MinosError`](@ref) lifts it to a valid
  at-limit side.
- `ext_state::Union{Nothing,Vector{Float64}}` — the inner-bounded-MIGRAD's
  converged EXTERNAL parameter vector at the crossing (bounded path
  only; always `nothing` for unbounded `function_cross`, and `nothing`
  for invalid results). Used by [`MinosError`](@ref) M4 snapshot fields.
"""
struct MnCross{S<:MinimumState,E<:Union{Nothing,AbstractVector{Float64}}}
    state::S
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
    #
    # Typed on the container, not pinned to `Vector{Float64}`: the snapshot is
    # a copy of the bounded fit's `ext_values`, which keeps the user's
    # coordinate container (and its axis labels) when they supplied one.
    ext_state::E
end

MnCross(state::MinimumState, aopt::Real, nfcn::Integer; valid=true,
         new_min=false, fcn_limit=false, par_limit=false,
         ext_state::Union{Nothing,AbstractVector{Float64}} = nothing) =
    MnCross(state, Float64(aopt), Int(nfcn), valid, new_min,
            fcn_limit, par_limit, ext_state)

# Same result with `aopt` shifted by `delta`. The single-parameter MINOS
# wrappers use it to turn the C++ `MnCross::Value()` (measured from the
# HESSE ±1σ point) into the multiplier of that step from the minimum.
_shift_aopt(cr::MnCross, delta::Real) =
    MnCross(cr.state, cr.aopt + Float64(delta), cr.nfcn;
            valid = cr.valid, new_min = cr.new_min, fcn_limit = cr.fcn_limit,
            par_limit = cr.par_limit, ext_state = cr.ext_state)

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
# Shared cross-search core — a line-by-line port of C++
# `MnFunctionCross::operator()` (reference/Minuit2_cpp/src/MnFunctionCross.cxx
# :25-508).
#
# The `_probe` closure runs the inner MIGRAD with the scanned parameter(s)
# held at `pmid + α·pdir` and returns `(FunctionMinimum, nfcn_increment)`.
# As in C++, α is measured FROM `pmid` (for MINOS: the HESSE ±1σ point,
# truncated against a bound), so α = 0 is the first probe and α = −1 is
# the minimum. The returned `MnCross.aopt` is the C++ `MnCross::Value()`;
# the single-parameter wrappers (`function_cross`, `function_cross_external`)
# publish `1 + aopt` so that `aopt · step` stays the error from the
# minimum, while `function_cross_multi` (contours) passes it through.
#
# History: through v0.7.3 this core skipped the α = 0 inner MIGRAD and
# substituted the fictitious point (α=0 at the minimum, f = fmin + 0.1·up)
# into the slope and parabola fits, and it had no quadratic early exit.
# That biased every crossing by up to the 0.01·up crossing tolerance
# (|ΔF/up| ≈ 1e-2 at the reported MINOS end points, where C++ Minuit2
# reaches 1e-4), and changed the number of inner minimisations per side.
# See CHANGELOG [Unreleased].
# ─────────────────────────────────────────────────────────────────────────────

# Classify the result of an inner MIGRAD probe exactly as the C++ does
# after every `migrad(maxcalls, mgr_tlr)` call (MnFunctionCross.cxx:125-136
# and the identical blocks after each later probe). Returns `nothing`
# when the search may continue, otherwise the `MnCross` to return.
@inline function _cross_probe_verdict(m::FunctionMinimum, fmin_val::Float64,
                                      tlf::Float64, aim::Float64,
                                      limset::Bool, nfcn::Int,
                                      state_fallback::MinimumState)
    fval(m) < fmin_val - tlf &&
        return MnCross(m.state, NaN, nfcn; valid = false, new_min = true)
    m.reached_call_limit &&
        return MnCross(m.state, NaN, nfcn; valid = false, fcn_limit = true)
    m.is_valid || return MnCross(state_fallback, NaN, nfcn; valid = false)
    # C++ `MnCross(state, nfcn, CrossParLimit())` is *valid* with the
    # limit flag raised; NativeMinuit keeps `valid = false, par_limit =
    # true` at this level (the user-facing `MinosError` lifts it).
    (limset && fval(m) < aim) &&
        return MnCross(m.state, NaN, nfcn; valid = false, par_limit = true)
    return nothing
end

function _cross_core(_probe::F, fmin_val::Float64, up::Float64,
                     state_fallback::MinimumState;
                     tlr::Float64 = 0.1,
                     maxcalls::Integer = 1000,
                     prec::MachinePrecision = MachinePrecision(),
                     up_scale::Float64 = 1.0,
                     print_level::Integer = 0,
                     aulim::Float64 = 100.0) where {F<:Function}
    # P5: `up_scale` (= sigma² for the MnMinos `sigma=k` API) scales the
    # effective ErrorDef so the crossing aim becomes `fmin + up · sigma²`.
    # Mirrors iminuit's `_TemporaryErrordef(self._fcn, factor)` wrapper
    # around MnMinos: everything the C++ reads through `fFCN.Up()` sees
    # `up · sigma²` (the aim, `tlf`, the `0.1·up` floor), while the HESSE
    # step handed in through `pdir` stays at the unscaled `up`.
    up_eff = up * up_scale
    aim = fmin_val + up_eff

    if print_level >= 1
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("start: fmin=%.10g  up=%.4g  aim=%.10g  tlr=%.4g  maxcalls=%d  aulim=%.4g",
                              fmin_val, up_eff, aim, tlr, maxcalls, aulim))
    end

    # C++ lines 38-47. The caller's `tlr` is used ONLY as the inner-MIGRAD
    # tolerance (`mgr_tlr = 0.5·tlr`, applied inside the probe closures);
    # the crossing tolerances are hard-coded at 0.01: converged when F is
    # within `tlf = 0.01·up` of the aim AND the next α prediction is within
    # `tla = 0.01` (scaled by |α| beyond 1) of the best probed α.
    tlr_c = 0.01
    tlf = tlr_c * up_eff
    tla = tlr_c
    maxitr = 15          # 6.24.0 (the pinned reference); ROOT ≥ 6.30 uses 30
    ipt = 0
    aopt = 0.0
    limset = false
    nfcn = 0
    alsb = Vector{Float64}(undef, 3)
    flsb = Vector{Float64}(undef, 3)

    # C++ line 103: already (within tla) at the limit before the first probe.
    if aulim < aopt + tla
        limset = true
    end

    # ── Probe at α = 0, i.e. at `pmid` (C++ `min0`, lines 119-136) ─────────
    min0, nf = _probe(0.0, maxcalls)
    nfcn += nf
    if print_level >= 2
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s  nfcn=%d  innerup=%.4g",
                              ipt + 1, 0.0, fval(min0), min0.is_valid, nf, min0.up))
    end
    v = _cross_probe_verdict(min0, fmin_val, tlf, aim, limset, nfcn, state_fallback)
    v === nothing || return v

    ipt += 1
    alsb[1] = 0.0
    # C++ line 141: floor the first value at fmin + 0.1·up so the quadratic
    # model below cannot divide by ~0 when pmid sits in the flat bottom.
    flsb[1] = max(fval(min0), fmin_val + 0.1 * up_eff)
    # C++ line 142: quadratic model through the minimum, `F − fmin ∝ (1+α)²`
    # (exact for a parabolic profile when pdir is the HESSE σ step).
    aopt = sqrt(up_eff / (flsb[1] - fmin_val)) - 1.0
    if abs(flsb[1] - aim) < tlf
        return MnCross(min0.state, aopt, nfcn; valid = true)
    end
    aopt > 1.0 && (aopt = 1.0)
    aopt < -0.5 && (aopt = -0.5)
    limset = false
    if aopt > aulim
        aopt = aulim
        limset = true
    end

    # ── Probe 2 (C++ `min1`, lines 164-186) ──────────────────────────────
    min1, nf = _probe(aopt, maxcalls)
    nfcn += nf
    if print_level >= 2
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s  nfcn=%d  innerup=%.4g",
                              ipt + 1, aopt, fval(min1), min1.is_valid, nf, min1.up))
    end
    v = _cross_probe_verdict(min1, fmin_val, tlf, aim, limset, nfcn, state_fallback)
    v === nothing || return v

    ipt += 1
    alsb[2] = aopt
    flsb[2] = fval(min1)
    dfda = (flsb[2] - flsb[1]) / (alsb[2] - alsb[1])
    last_min = min1

    @label L300
    # ── L300 (C++ lines 188-242): slope of the wrong sign — step outward
    #    by 0.2·it (it restarts at 1 on every re-entry) until dfda > 0.
    if dfda < 0.0
        maxlk = maxitr - ipt
        for it in 1:maxlk
            alsb[1] = alsb[2]
            flsb[1] = flsb[2]
            aopt = alsb[1] + 0.2 * it
            limset = false
            if aopt > aulim
                aopt = aulim
                limset = true
            end
            min1, nf = _probe(aopt, maxcalls)
            nfcn += nf
            if print_level >= 2
                _trace_info(print_level, "MnFunctionCross",
                            @sprintf("L300 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s  nfcn=%d  innerup=%.4g",
                                      ipt + 1, aopt, fval(min1), min1.is_valid, nf, min1.up))
            end
            v = _cross_probe_verdict(min1, fmin_val, tlf, aim, limset, nfcn, state_fallback)
            v === nothing || return v
            ipt += 1
            alsb[2] = aopt
            flsb[2] = fval(min1)
            dfda = (flsb[2] - flsb[1]) / (alsb[2] - alsb[1])
            last_min = min1
            dfda > 0.0 && break
        end
        if ipt > maxitr
            return MnCross(state_fallback, NaN, nfcn; valid = false)
        end
    end

    @label L460
    # ── L460 (C++ lines 244-299): two points with positive slope — linear
    #    extrapolation to the aim, with the convergence test on it.
    aopt = alsb[2] + (aim - flsb[2]) / dfda
    fdist = min(abs(aim - flsb[1]), abs(aim - flsb[2]))
    adist = min(abs(aopt - alsb[1]), abs(aopt - alsb[2]))
    tla = tlr_c
    if abs(aopt) > 1.0
        tla = tlr_c * abs(aopt)
    end
    if adist < tla && fdist < tlf
        return MnCross(last_min.state, aopt, nfcn; valid = true)
    end
    if ipt > maxitr
        return MnCross(state_fallback, NaN, nfcn; valid = false)
    end
    bmin = min(alsb[1], alsb[2]) - 1.0
    aopt < bmin && (aopt = bmin)
    bmax = max(alsb[1], alsb[2]) + 1.0
    aopt > bmax && (aopt = bmax)
    limset = false
    if aopt > aulim
        aopt = aulim
        limset = true
    end

    min2, nf = _probe(aopt, maxcalls)
    nfcn += nf
    if print_level >= 2
        _trace_info(print_level, "MnFunctionCross",
                    @sprintf("L460 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s  nfcn=%d  innerup=%.4g",
                              ipt + 1, aopt, fval(min2), min2.is_valid, nf, min2.up))
    end
    v = _cross_probe_verdict(min2, fmin_val, tlf, aim, limset, nfcn, state_fallback)
    v === nothing || return v

    ipt += 1
    alsb[3] = aopt
    flsb[3] = fval(min2)
    last_min = min2

    # ── Three points: how many below the aim? (C++ lines 301-351) ────────
    # The initial classifier seeds `ibest` with the THIRD point.
    ibest, iworst, _, _, _, noless, _, _ =
        _three_point_classify(alsb, flsb, aim; default_ibest = 3)
    if noless == 1 || noless == 2
        @goto L500
    elseif noless == 0 && ibest != 3
        # all three above the aim and the newest is not the closest
        return MnCross(state_fallback, NaN, nfcn; valid = false)
    elseif noless == 3 && ibest != 3
        # all three below and the slope went negative again — re-extend
        alsb[2] = alsb[3]
        flsb[2] = flsb[3]
        @goto L300
    end
    # otherwise: new straight line through the two best points
    flsb[iworst] = flsb[3]
    alsb[iworst] = alsb[3]
    dfda = (flsb[2] - flsb[1]) / (alsb[2] - alsb[1])
    @goto L460

    @label L500
    # ── L500 (C++ lines 353-507): parabola through the three points,
    #    take the root with positive slope, keep a point on each side.
    while true
        A, B, C = _parabola_fit3(alsb, flsb)
        sol = _parabola_solve_for_aim(A, B, C, aim, prec)
        sol === nothing &&
            return MnCross(state_fallback, NaN, nfcn; valid = false)  # determ < eps
        aopt, slope = sol

        tla = tlr_c
        if abs(aopt) > 1.0
            tla = tlr_c * abs(aopt)
        end
        if abs(aopt - alsb[ibest]) < tla && abs(flsb[ibest] - aim) < tlf
            return MnCross(last_min.state, aopt, nfcn; valid = true)
        end

        # ileft / iright / iout / ibest (C++ lines 412-443; `ibest` seeded
        # with the FIRST point here).
        ibest, _, ileft, iright, iout, _, _, ecarmx =
            _three_point_classify(alsb, flsb, aim; default_ibest = 1)
        # With one point on each side of the aim all three are defined;
        # bail defensively rather than index a sentinel.
        if ileft == 0 || iright == 0 || iout == 0
            return MnCross(state_fallback, NaN, nfcn; valid = false)
        end

        # avoid keeping a bad point next time around (C++ line 449)
        if ecarmx > 10.0 * abs(flsb[iout] - aim)
            aopt = 0.5 * (aopt + 0.5 * (alsb[iright] + alsb[ileft]))
        end

        # acceptable window between the left and right anchors (C++ 452-465)
        smalla = 0.1 * tla
        if slope * smalla > tlf
            smalla = tlf / slope
        end
        aleft = alsb[ileft] + smalla
        aright = alsb[iright] - smalla
        aopt < aleft && (aopt = aleft)
        aopt > aright && (aopt = aright)
        aleft > aright && (aopt = 0.5 * (aleft + aright))

        limset = false
        if aopt > aulim
            aopt = aulim
            limset = true
        end

        min2, nf = _probe(aopt, maxcalls)
        nfcn += nf
        if print_level >= 2
            _trace_info(print_level, "MnFunctionCross",
                        @sprintf("L500 probe ipt=%d  aopt=%.6g  f=%.10g  valid=%s  nfcn=%d  innerup=%.4g",
                                  ipt + 1, aopt, fval(min2), min2.is_valid, nf, min2.up))
        end
        v = _cross_probe_verdict(min2, fmin_val, tlf, aim, limset, nfcn, state_fallback)
        v === nothing || return v

        ipt += 1
        # replace the redundant point with the new one, which is now `ibest`
        alsb[iout] = aopt
        flsb[iout] = fval(min2)
        ibest = iout
        last_min = min2
        ipt < maxitr || break
    end

    if print_level >= 1
        _trace_warn(print_level, "MnFunctionCross",
                    @sprintf("did not converge in %d iters", maxitr))
    end
    return MnCross(state_fallback, NaN, nfcn; valid = false)
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
function _fix_one_param(cf::CostFunction, i::Integer, v::Float64, n::Integer,
                         template::AbstractVector{Float64} =
                             Vector{Float64}(undef, Int(n));
                         up::Real = cf.up)
    f = cf.f
    up = Float64(up)
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
    full_bufs = [similar(template, Float64) for _ in 1:nbuf]
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
function _fix_one_param(cf::CostFunctionWithGradient, i::Integer, v::Float64,
                         n::Integer,
                         template::AbstractVector{Float64} =
                             Vector{Float64}(undef, Int(n));
                         up::Real = cf.up)
    f = cf.f
    g = cf.g
    up = Float64(up)
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
    full_bufs = [similar(template, Float64) for _ in 1:nbuf]
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
    # `check_gradient=false`: the user gradient is validated once at the
    # top-level fit's seed; re-running the CheckGradient discrepancy check
    # on every MINOS/contour cross-search re-seed (this fixed-param probe)
    # would be redundant and cost extra FCN calls. The probe gradient is
    # derived from the same already-validated user gradient.
    return CostFunctionWithGradient(f_wrapped, g_wrapped, up; check_gradient = false)
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
    template::AbstractVector{Float64} = Vector{Float64}(undef, Int(n));
    up::Real = cf.up,
)
    length(par_idxs) == length(v) ||
        throw(DimensionMismatch("par_idxs / v length mismatch"))
    f = cf.f
    g = cf.g
    up = Float64(up)
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
    full_bufs = [similar(template, Float64) for _ in 1:nbuf]
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
    # `check_gradient=false`: the user gradient is validated once at the
    # top-level fit's seed; re-running the CheckGradient discrepancy check
    # on every MINOS/contour cross-search re-seed (this fixed-param probe)
    # would be redundant and cost extra FCN calls. The probe gradient is
    # derived from the same already-validated user gradient.
    return CostFunctionWithGradient(f_wrapped, g_wrapped, up; check_gradient = false)
end

function _fix_multi_params(
    cf::CostFunction,
    par_idxs::AbstractVector{<:Integer},
    v::AbstractVector{<:Real},
    n::Integer,
    template::AbstractVector{Float64} = Vector{Float64}(undef, Int(n));
    up::Real = cf.up,
)
    length(par_idxs) == length(v) ||
        throw(DimensionMismatch("par_idxs / v length mismatch"))
    f = cf.f
    up = Float64(up)
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
    full_bufs = [similar(template, Float64) for _ in 1:nbuf]
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
    # Conditional covariance of the free parameters with `par_idxs` fixed,
    # when the caller already holds it (a contour driver fixes the same two
    # parameters at every point); computed here otherwise.
    prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    # ErrorDef of the inner minimisation. A `sigma = k` cross search runs
    # under iminuit's temporary errordef `up·k²`, which C++ reads everywhere
    # inside the inner MIGRAD (EDM goal, numerical-gradient and HESSE steps);
    # the seed errors below still use the outer `cf.up` (C++ takes them from
    # the minimum's user state, built at the original errordef).
    up_inner::Real = cf.up,
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
        # Container-preserving, because it is handed to `cf.f` below. The
        # degenerate state built from it is re-flattened first — the warm-state
        # slot is typed `DenseMinimumState` and every reduced-coordinate probe
        # must keep that invariant.
        full = similar(state.parameters.x, Float64)
        @inbounds for (idx, k) in enumerate(par_idxs)
            full[Int(k)] = Float64(v[idx])
        end
        @inbounds for k in 1:n
            if !is_fixed[k]
                full[k] = state.parameters.x[k]
            end
        end
        f_val = Float64(cf.f(full))
        # Build a degenerate FunctionMinimum representing this evaluation.
        # Flattened: this state is stored in a `DenseMinimumState` warm slot.
        full_dense = Vector{Float64}(undef, n)
        @inbounds copyto!(full_dense, full)
        fake_par = MinimumParameters(full_dense, f_val)
        fake_err = MinimumError(Symmetric(Matrix{Float64}(undef, 0, 0), :U), MnHesseValid)
        fake_grad = FunctionGradient(0)
        fake_state = MinimumState(fake_par, fake_err, fake_grad, 0.0, 1)
        fake_min = FunctionMinimum(fake_state, fake_state, cf.up;
                                    is_valid = true)
        return fake_min, 1
    end

    # Template = the outer state's own coordinate vector, so the spliced
    # full-length buffer handed to the user's FCN keeps its container on the
    # LOW-LEVEL path too (the high-level path wraps `cf` and is unaffected).
    cf_fixed = _fix_multi_params(cf, par_idxs, v, n, state.parameters.x; up = up_inner)
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

    # C++ MnContours.cxx:125-131 (`upar.Fix(px); upar.Fix(py)`): the inner
    # MIGRAD of every ray search is seeded with the outer covariance
    # squeezed at all the fixed parameters (conditional covariance).
    inner_prior_cov = prior_cov === nothing ?
                      _conditional_prior_cov(state.error, par_idxs; prec = prec) :
                      prior_cov
    inner_min = migrad(cf_fixed, y0, errs;
                        tol = tol, maxfcn = Int(maxcalls),
                        strategy = inner_strategy, prec = prec,
                        scratch = scratch,
                        threaded_gradient = threaded_gradient,
                        print_level = print_level,
                        prior_cov = inner_prior_cov)
    return inner_min, ncalls(cf_fixed)
end

# The same objective at another ErrorDef (iminuit's temporary errordef for a
# `sigma = k` MINOS / contour run). Fresh FCN counter — every consumer wraps
# or counts per run — while `ngrad` and the non-finite tally stay shared.
_with_errordef(cf::CostFunction, up::Real) =
    Float64(up) == cf.up ? cf : CostFunction(cf.f, Float64(up))
_with_errordef(cf::CostFunctionWithGradient, up::Real) =
    Float64(up) == cf.up ? cf :
    CostFunctionWithGradient(cf.f, cf.g, Float64(up), Ref(0), cf.ngrad, cf.n_nonfinite;
                             check_gradient = cf.check_gradient)

"""
    _conditional_prior_cov(err, idxs; prec) -> Union{Nothing,Matrix{Float64}}

Covariance of the remaining parameters once those in `idxs` are held
fixed: the outer inverse Hessian is inverted, the rows and columns in
`idxs` removed, and the result inverted back — C++ `MnCovarianceSqueeze`,
which `MnUserParameterState::Fix` applies for every parameter a cross
search fixes, so that the inner MIGRAD seed (`MnSeedGenerator`,
`HasCovariance()` branch) starts from the conditional covariance.
Returns `nothing` when the outer error matrix is unavailable or a
squeeze cannot invert, in which case the caller falls back to the
diagonal seed.
"""
function _conditional_prior_cov(err::MinimumError, idxs;
                                prec::MachinePrecision = MachinePrecision())
    is_available(err) || return nothing
    e = err
    for i in sort!(Int[idxs...]; rev = true)
        e = squeeze_error(e, i; prec = prec)
        e.status == MnInvertFailed && return nothing
    end
    # Materialise the full symmetric matrix (the `Symmetric` wrapper only
    # carries the upper triangle, which `seed_state`'s symmetry check rejects).
    return Matrix(e.inv_hessian)
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
    prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
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
    # Typed on `DenseMinimumState`, NOT `typeof(state)`: the probes below run
    # over a reduced coordinate space seeded from a fresh `Vector{Float64}`,
    # so their states are dense even when the outer fit uses a structured
    # container. Pinning this slot to the outer type makes the very first
    # `warm_state_ref[] = inner_min.state` unstorable.
    warm_state_ref = Ref{Union{Nothing,DenseMinimumState}}(nothing)
    scratch_dense = scratch === nothing ? nothing :
                    scratch::MigradScratch{Vector{Float64}}
    scratch_holder = Ref{Union{Nothing,MigradScratch{Vector{Float64}}}}(scratch_dense)
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
                # `0.5·tlr` is C++ `mgr_tlr`. Under iminuit's
                # `_TemporaryErrordef` every inner MIGRAD runs at the
                # σ²-scaled errordef (EDM goal, gradient and HESSE steps),
                # so the inner cost function carries `up·σ²`.
                tol = 0.5 * tlr, maxcalls = budget,
                up_inner = up * Float64(sigma)^2,
                prec = prec, strategy = strategy,
                warm_state = warm_state_ref[],
                scratch = scratch_holder[],
                threaded_gradient = threaded_gradient,
                print_level = print_level,
                prior_cov = prior_cov)
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
    other_param_seed::Union{Nothing,AbstractVector{<:Real}} = nothing,
    # ErrorDef of the inner minimisation — see `_migrad_with_multi_fixed`.
    up_inner::Real = cf.up,
)
    n = length(state.parameters)
    # Template = the outer state's coordinate vector — see the matching
    # comment in `_migrad_with_multi_fixed`.
    cf_fixed = _fix_one_param(cf, i, v, n, state.parameters.x; up = up_inner)
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

    # COLD PATH: seed the inner MIGRAD from a length-(n-1) starting
    # vector. Three-way priority (codex round-1 MEDIUM — "or cold-
    # fallback from the last warm position"; this is the NativeMinuit-local
    # interpretation, NOT a literal C++ reproduction):
    #
    #   1. If `warm_state` is supplied (probe 2..N path where
    #      `warm_restart_state` failed — negative g2, edm refinement
    #      diverged, etc.): use the LAST VALID converged x. This is
    #      NativeMinuit's analog to C++'s single-MnMigrad-instance
    #      pattern: when NativeMinuit's pre-MIGRAD g2 hygiene check
    #      bails, we salvage the previous probe's converged position
    #      (rebuild g2/hessian from scratch, keep x) rather than
    #      re-applying the α=1 pre-shift OR snapping back to x_min.
    #      Note: this is NOT literal C++ behavior — C++ MnFunctionCross
    #      aborts the cross-search on `!min1.IsValid()` (line 225-226)
    #      and does not retry. NativeMinuit's `warm_restart_state` is a
    #      finer-grained hygiene check (g2 sign + edm) than C++'s
    #      MnMigrad-level IsValid; cold-restarting from the prior x
    #      with fresh g2 is a reasonable middle ground that empirically
    #      preserves convergence on side-basin-prone profiles.
    #
    #   2. Elif `other_param_seed` is supplied (probe 1, MnMinos
    #      pre-shift from `minos.jl`): use it.
    #
    #   3. Else (probe 1, no pre-shift caller): outer-minimum x with
    #      parameter `i` removed. Historical Phase-1 behavior.
    #
    # The pre-shift matters when the χ² profile along par_idx is
    # strongly non-convex and the unshifted (other params at outer
    # min) starting point sits on a steep wall: gradient descent
    # from there can land the inner MIGRAD in a side basin, biasing
    # subsequent warm-started probes and silently invalidating the
    # MINOS crossing search. Mirrors C++ MnMinos.cxx:143-165 which
    # SetValue's the other params to their 1σ-correlated guess
    # before constructing MnFunctionCross.
    x_min = state.parameters.x
    y0 = Vector{Float64}(undef, n - 1)
    if warm_state !== nothing && length(warm_state) == n - 1
        # Priority 1: cold-fallback from last warm position.
        warm_x = warm_state.parameters.x
        @inbounds for k in 1:(n - 1)
            y0[k] = warm_x[k]
        end
    elseif other_param_seed !== nothing
        # Priority 2: probe 1 with MnMinos pre-shift.
        length(other_param_seed) == n - 1 ||
            throw(DimensionMismatch(
                "other_param_seed length $(length(other_param_seed)) != n-1 = $(n-1)"))
        @inbounds for k in 1:(n - 1)
            y0[k] = Float64(other_param_seed[k])
        end
    else
        # Priority 3: probe 1 unshifted.
        @inbounds for k in 1:(i - 1)
            y0[k] = x_min[k]
        end
        @inbounds for k in (i + 1):n
            y0[k - 1] = x_min[k]
        end
    end
    # Per-coord errors derived from the outer inv_hessian. C++
    # MnUserParameterState constructs free-parameter errors as
    # sqrt(2·up·V[i,i]) (reference/Minuit2_cpp/src/MnUserParameterState.cxx
    # :151-154).
    errs = Vector{Float64}(undef, n - 1)
    V = state.error.inv_hessian
    scale = 2.0 * cf.up
    @inbounds for k in 1:(i - 1)
        errs[k] = sqrt(max(scale * V[k, k], prec.eps2))
    end
    @inbounds for k in (i + 1):n
        errs[k - 1] = sqrt(max(scale * V[k, k], prec.eps2))
    end

    # Inner MIGRAD seed covariance (`prior_cov`, taken with `dcovar = 0`
    # like C++ MnSeedGenerator's `st.HasCovariance()` branch). C++
    # MnMinos.cxx:118/167 copies the outer `UserState()` and `Fix(par)`s
    # it, which squeezes the covariance through MnCovarianceSqueeze
    # (invert V, drop row/col `i`, invert back): the CONDITIONAL
    # covariance of the other parameters with `i` held fixed. Through
    # v0.7.3 the marginal minor `V[~i, ~i]` was passed instead; it
    # over-estimates the conditional covariance by the correlation with
    # `i`, so the inner DFP's first Newton step overshot on correlated
    # fits. A cold fallback from the last warm position carries the
    # previous probe's covariance, as the C++ single-MnMigrad instance
    # does (`fState = min.UserState()`).
    inner_prior_cov = if warm_state !== nothing && length(warm_state) == n - 1 &&
                         is_available(warm_state.error)
        Matrix(warm_state.error.inv_hessian)
    else
        _conditional_prior_cov(state.error, (Int(i),); prec = prec)
    end

    inner_min = migrad(cf_fixed, y0, errs;
                        tol = tol, maxfcn = Int(maxcalls),
                        strategy = inner_strategy, prec = prec,
                        scratch = scratch,
                        threaded_gradient = threaded_gradient,
                        print_level = print_level,
                        prior_cov = inner_prior_cov)
    return inner_min, ncalls(cf_fixed)
end

"""
    _extract_minor_cov(V, i, n) -> Matrix{Float64}

Extract the (n-1)×(n-1) minor of `V` by removing row+column `i`. Helper
for MnMinos inner-MIGRAD prior covariance seeding. Returns a `Matrix`
(not a view) so the caller can pass it to `prior_cov=` without aliasing
the outer Hessian.
"""
function _extract_minor_cov(V::AbstractMatrix, i::Int, n::Int)
    n >= 2 || throw(ArgumentError("_extract_minor_cov needs n ≥ 2"))
    1 <= i <= n || throw(ArgumentError("i $i out of bounds for n=$n"))
    M = Matrix{Float64}(undef, n - 1, n - 1)
    @inbounds for col in 1:n
        col == i && continue
        cc = col < i ? col : col - 1
        for row in 1:n
            row == i && continue
            rr = row < i ? row : row - 1
            M[rr, cc] = V[row, col]
        end
    end
    return M
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

- `tlr::Real=0.1` — tolerance of the inner MIGRADs (`0.5·tlr`, C++
  MnFunctionCross.cxx:38). The crossing tolerances themselves are fixed
  at `tlf = 0.01·up` on the function value and `tla = 0.01` on α (C++
  lines 40-46), and the returned α is the model prediction (quadratic,
  linear or parabolic) at convergence, not the last probed α.
- `maxcalls::Integer=1000` — call budget of EACH inner MIGRAD (C++ passes
  the full `maxcalls` to every `migrad(maxcalls, mgr_tlr)` of the search;
  the search itself is capped at 15 probes).
- `strategy::Strategy=Strategy(0)` — passed to inner MIGRADs.
- `prec::MachinePrecision`.
- `sigma::Real=1.0` — confidence level in σ-units. The crossing aim
  becomes `fmin + up · sigma²` (mirrors iminuit's `minos(cl=)` scaling
  of `MnFunctionCross.aim`). At sigma=1 the behavior is C++-identical
  to a single MnFunctionCross call; at sigma=k the returned `aopt`
  converges to ≈ k (in the parabolic approximation), so the caller's
  `aopt · σ_1` product is the k-σ error.
- `other_param_seed::Union{Nothing,AbstractVector{<:Real}}=nothing` —
  optional length-(n-1) starting vector for the inner MIGRAD on the
  free parameters (par_idx removed). When provided, used as the COLD-
  path seed of probe 1; consumed (i.e. cleared to `nothing` so later
  probes get warm-state continuation, not the original α=1 seed).
  Mirrors C++ MnMinos's pre-shift (MnMinos.cxx:143-165) — without it,
  the inner MIGRAD on strongly-correlated, non-convex profiles can
  descend into side basins. Caller (`minos.jl`) supplies the pre-
  shifted seed; other callers (e.g. contours) leave it `nothing`.

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
    other_param_seed::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    sigma > 0 ||
        throw(ArgumentError("sigma must be positive, got $sigma"))
    state = fmin.state
    n = length(state.parameters)
    1 <= par_idx <= n ||
        throw(ArgumentError("par_idx $par_idx out of bounds for n=$n"))
    n > 1 ||
        throw(ArgumentError("function_cross requires n > 1 (cannot fix the only parameter)"))
    other_param_seed === nothing || length(other_param_seed) == n - 1 ||
        throw(DimensionMismatch(
            "other_param_seed length $(length(other_param_seed)) != n-1 = $(n-1)"))

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
    #
    # `other_param_seed` is passed unchanged to every probe; it is
    # consumed only in `_migrad_with_fixed`'s COLD path AND only when
    # `warm_state` is `nothing` (i.e. probe 1). On probe 2..N, even
    # if warm-restart fails, the cold-fallback uses `warm_state.x`
    # (codex C++-faithful interpretation) — never the original α=1
    # seed, so subsequent probes don't reset to the linear-tangent
    # prediction once the inner MIGRAD has found the actual valley.
    # Dense by construction — see the matching comment in
    # `function_cross_multi` above.
    warm_state_ref = Ref{Union{Nothing,DenseMinimumState}}(nothing)
    scratch_dense = scratch === nothing ? nothing :
                    scratch::MigradScratch{Vector{Float64}}
    scratch_holder = Ref{Union{Nothing,MigradScratch{Vector{Float64}}}}(scratch_dense)
    let x_pivot = x_pivot, step = step,
        warm_state_ref = warm_state_ref,
        scratch_holder = scratch_holder, n = n,
        other_param_seed = other_param_seed
        probe = function (aopt::Float64, budget::Integer)
            # C++ α-convention: α = 0 is the HESSE ±1σ point `x_pivot + step`
            # (MnMinos.cxx:118-131 → `xmid = val`, `xdir = err`).
            v = x_pivot + step * (1.0 + aopt)
            _get_scratch!(scratch_holder, n - 1)
            # σ²-scaled errordef: see the matching comment in `function_cross_multi`.
            inner_min, nf = _migrad_with_fixed(cf, state, par_idx, v;
                                tol = 0.5 * tlr, maxcalls = budget,
                                up_inner = up * Float64(sigma)^2,
                                prec = prec, strategy = strategy,
                                warm_state = warm_state_ref[],
                                scratch = scratch_holder[],
                                threaded_gradient = threaded_gradient,
                                print_level = print_level,
                                other_param_seed = other_param_seed)
            if inner_min.is_valid
                warm_state_ref[] = inner_min.state
            end
            return inner_min, nf
        end
        cr = _cross_core(probe, fmin_val, up, state;
                          tlr = Float64(tlr),
                          maxcalls = maxcalls, prec = prec,
                          up_scale = Float64(sigma)^2,
                          print_level = print_level)
        # Publish the multiplier of the ±1σ step FROM THE MINIMUM
        # (C++ `MinosError::Upper() = err · (1 + Value())`).
        return cr.valid ? _shift_aopt(cr, 1.0) : cr
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

# Keyword arguments

- `other_param_seed_ext::Union{Nothing,AbstractVector{<:Real}}=nothing`
  — optional length-`n_total` (i.e., full-external-vector) starting
  point for the inner MIGRAD's NON-scanned parameters on probe 1.
  When provided, used as the cold seed of probe 1 only and then
  consumed; subsequent probes seed each non-scanned slot from
  `bfm.ext_values[i]` (the OUTER fit's converged value). Mirrors
  C++ MnMinos.cxx:143-165 pre-shift, with the additional Int2ext +
  EXT bound clamp required to keep doubly-bounded "other" parameters
  inside their valid range. Caller (`_minos_external_via_function_cross`
  in src/minuit.jl) computes the seed.
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
    other_param_seed_ext::Union{Nothing,AbstractVector{<:Real}} = nothing,
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
    other_param_seed_ext === nothing || length(other_param_seed_ext) == n_total ||
        throw(DimensionMismatch(
            "other_param_seed_ext length $(length(other_param_seed_ext)) != n_total = $n_total"))

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
    # Inner minimisations run at the σ²-scaled errordef (iminuit's temporary
    # errordef): see the matching comment in `function_cross_multi`.
    cf_inner = _with_errordef(cf, up * Float64(sigma)^2)

    # `aulim`: the largest α (measured from `val_trial`, the C++ `pmid`)
    # that keeps the scanned parameter inside its bound — C++
    # MnFunctionCross.cxx:64-104, including the default of 100 when no
    # bound lies in the search direction. `_cross_core` clamps every
    # proposed α to it and raises `par_limit` when the search saturates
    # there below the aim. `limset` records that the walk touched the
    # bound at all, so a failed search can be relabelled below.
    aulim = if step_ext > 0 && has_upper_limit(par)
        min(100.0, (par.upper - val_trial) / step_ext)
    elseif step_ext < 0 && has_lower_limit(par)
        min(100.0, (par.lower - val_trial) / step_ext)
    else
        100.0
    end
    limset = Ref(false)
    # Inner-MIGRAD seed covariance, as C++ carries it in the single
    # MnMigrad instance of MnFunctionCross: probe 1 starts from the outer
    # fit's INTERNAL covariance squeezed at the scanned parameter
    # (`MnUserParameterState::Fix` → MnCovarianceSqueeze: the conditional
    # covariance of the others with it held fixed); later probes start
    # from the previous probe's converged internal covariance
    # (`MnApplication::operator()` → `fState = min.UserState()`).
    ind_int_scan = params.int_of_ext[par_idx]
    prior_cov_ref = Ref{Union{Nothing,Matrix{Float64}}}(
        _conditional_prior_cov(bfm.internal.state.error, (ind_int_scan,);
                               prec = prec))
    # M4: capture the inner-bounded-MIGRAD's converged EXTERNAL parameter
    # vector at the last successful probe so the caller can publish a
    # full ext snapshot via `MinosError.{upper,lower}_state`. `_cross_core`
    # returns the converged `last_min.state` (internal coords); we hold
    # the ext slice in a closure-captured Ref that the probe overwrites
    # on each `inner_bfm.is_valid` call. After `_cross_core` returns
    # valid, `last_ext_state[]` holds the converged ext snapshot.
    # Typed on the fit's own external container so a structured snapshot is
    # storable (and reaches `MinosError.{upper,lower}_state` with its axes).
    last_ext_state = Ref{Union{Nothing,typeof(bfm.ext_values)}}(nothing)

    # Consume-once pre-shift Ref. Mirrors function_cross's
    # `pre_seed_ref` pattern (codex consume-once finding). C++ MnMinos
    # sets the pre-shifted other-param values into `upar` ONCE before
    # constructing MnFunctionCross; subsequent SetValue calls only
    # touch the scanned param, so the OTHER params start at whatever
    # MnMigrad left them on the previous probe.
    #
    # NativeMinuit closes this in two stages:
    #
    # (a) Probe 1: consume the caller-supplied `other_param_seed_ext`
    #     (the MnMinos pre-shift). Held in `pre_seed_ext_ref`, cleared
    #     after probe 1.
    #
    # (b) Probes 2..N: seed each non-scanned slot from the PREVIOUS
    #     probe's converged ext values, NOT from `bfm.ext_values[i]`
    #     (the OUTER fit's converged value, which loses the side-
    #     basin-avoidance that probe 1's pre-shift just earned us).
    #     `prev_probe_ext_ref` holds the last successfully converged
    #     inner ext vector; updated on every `inner_bfm.is_valid` probe
    #     and consumed by the inner_params builder. Mirrors C++ MnMigrad
    #     single-instance "OTHER params left where the previous probe
    #     converged them" semantics. Without this, the side-basin
    #     pathology re-emerges on probes 2..N even though probe 1 was
    #     well-seeded.
    #
    # Note: this still rebuilds the INTERNAL Hessian estimate from
    # scratch on each probe (DFP starts from diagonal `errs`). Full
    # C++ parity would also thread the previous probe's converged
    # internal covariance forward as `prior_cov` for the next
    # bounded-migrad call — that requires extending the bounded
    # `migrad` API to accept a warm internal state, tracked as a
    # follow-up (see GAP_AUDIT).
    pre_seed_ext_ref = Ref{Union{Nothing,Vector{Float64}}}(
        other_param_seed_ext === nothing ? nothing :
            Float64[Float64(other_param_seed_ext[i]) for i in 1:n_total])
    prev_probe_ext_ref = Ref{Union{Nothing,Vector{Float64}}}(nothing)

    # Build probe closure. At each alpha, set par to `ext_min + aopt *
    # step_ext` (clamped against the bound if aopt > aulim), copy ALL
    # other free params from the CONVERGED minimum (NOT the user seed
    # — codex round-3 catch: starting from the seed loses the
    # MIGRAD-converged information and can land the inner search in a
    # different basin for hard FCNs), build Parameters with par_idx
    # FIXED at the trial value, run bounded migrad on the inner problem.
    # §3.3 named hoist (issue #45, stage C2): read the `params.pars`
    # property once, outside the probe closure, instead of twice per
    # probe call. `all_pars` is the derived read-only records view over
    # `params`, which is never mutated during the cross (each probe
    # builds a fresh `inner_params`), so every element read through the
    # hoisted view is identical to a fresh `params.pars` read.
    let par_idx = Int(par_idx), step_ext = step_ext, val_trial = val_trial,
        ext_min = ext_min, par = par, params = params,
        prior_cov_ref = prior_cov_ref, cf_inner = cf_inner,
        all_pars = params.pars,
        aulim = aulim, limset = limset,
        last_ext_state = last_ext_state,
        ext_values = bfm.ext_values,
        inner_strategy = inner_strategy,
        pre_seed_ext_ref = pre_seed_ext_ref,
        prev_probe_ext_ref = prev_probe_ext_ref
        probe = function (aopt::Float64, budget::Integer)
            # `_cross_core` already clamps α to `aulim` (C++ `limset`);
            # record that the walk touched the bound so a failed search
            # can be relabelled `par_limit` below.
            clamped_aopt = aopt
            if aopt >= aulim
                clamped_aopt = aulim
                limset[] = true
            end
            # C++ α-convention: α = 0 is `val_trial` (the truncated ±1σ point).
            ext_val = val_trial + clamped_aopt * step_ext
            # Defensive re-clamp: rounding might leave ext_val just past
            # the bound by 1 ulp.
            if has_upper_limit(par)
                ext_val = min(ext_val, par.upper)
            end
            if has_lower_limit(par)
                ext_val = max(ext_val, par.lower)
            end
            # Three-tier seed priority (C++ MnMigrad single-instance
            # parity, see comment block above `pre_seed_ext_ref`):
            #
            #   1. Previous probe's converged ext (probes 2+): the
            #      C++-faithful "OTHER params left where MnMigrad left
            #      them" — preserves probe 1's basin choice forward.
            #
            #   2. `pre_seed_ext_ref` (probe 1): the MnMinos pre-shift
            #      caller-supplied seed. Consumed after first use.
            #
            #   3. `bfm.ext_values[i]` (probe 1, no caller seed): the
            #      OUTER fit's converged value. Historical behavior.
            prev_ext = prev_probe_ext_ref[]
            pre_shift = pre_seed_ext_ref[]
            pre_seed_ext_ref[] = nothing
            inner_pars = MinuitParameter[]
            sizehint!(inner_pars, length(all_pars))
            for (i, p) in enumerate(all_pars)
                converged_v = if i == par_idx
                    ext_values[i]   # scanned slot — overridden below
                elseif prev_ext !== nothing
                    prev_ext[i]     # priority 1
                elseif pre_shift !== nothing
                    pre_shift[i]    # priority 2
                else
                    ext_values[i]   # priority 3
                end
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
            # Seed from `params` so the inner bounded MIGRAD keeps the outer
            # fit's coordinate container: its external workspaces are built
            # with `similar(values)`, and they are what the user's objective
            # and gradient are called with.
            inner_params = Parameters(inner_pars, params)
            inner_bfm = migrad(cf_inner, inner_params;
                                tol = 0.5 * tlr, maxfcn = Int(budget),
                                strategy = inner_strategy, prec = prec,
                                threaded_gradient = threaded_gradient,
                                print_level = print_level,
                                prior_cov = prior_cov_ref[])
            # Capture this probe's converged ext values and internal
            # covariance for the NEXT probe's seed (priority-1 above).
            # Only update on valid probes — an invalid converged x would
            # propagate the bad state.
            if inner_bfm.internal.is_valid
                prev_probe_ext_ref[] = copy(inner_bfm.ext_values)
                err_in = inner_bfm.internal.state.error
                prior_cov_ref[] = is_available(err_in) ?
                                  Matrix(err_in.inv_hessian) : nothing
            end
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
                              print_level = print_level,
                              aulim = aulim)
        # Publish the multiplier of the (truncated) ±1σ step FROM THE
        # MINIMUM (C++ `MinosError::Upper() = err · (1 + Value())`). The
        # core never returns a valid α beyond `aulim` — it clamps the
        # proposal and reports `par_limit` when the search saturates at
        # the bound below the aim (C++ MnFunctionCross.cxx:494-496).
        if result.valid
            result = _shift_aopt(result, 1.0)
        end
        # A search that touched the bound but then failed for another
        # reason (call limit, new minimum, invalid inner MIGRAD, no bracket)
        # keeps that failure: C++ MnFunctionCross returns the call-limit /
        # new-minimum / invalid result first and raises CrossParLimit only
        # for a probe that converged below the aim at the bound
        # (`_cross_probe_verdict`), so `limset` alone never makes a side
        # "at limit" here.
        # M4: attach the captured ext snapshot — the inner-MIGRAD's
        # converged ext values at the crossing, or at the bound for an
        # at-limit side. `_cross_core` builds its MnCross without
        # ext_state, so we rebuild here when a snapshot was captured
        # (other failure modes keep `ext_state == nothing`).
        if (result.valid || result.par_limit) && last_ext_state[] !== nothing &&
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
