# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# transform.jl — parameter bound transformations.
#
# Mirrors three C++ files (one per bound kind):
#   reference/Minuit2_cpp/src/SinParameterTransformation.cxx
#   reference/Minuit2_cpp/src/SqrtUpParameterTransformation.cxx
#   reference/Minuit2_cpp/src/SqrtLowParameterTransformation.cxx
#
# C++ Minuit2 maps user-visible (external, bounded) parameter values to
# internal (unbounded) values via three transformations chosen per
# parameter:
#
#   - **Both bounds** [L, U]:  Sin family. ext = L + 0.5·(U-L)·(sin(int)+1).
#   - **Upper only**  (-∞, U]: SqrtUp.    ext = U + 1 - √(int² + 1).
#   - **Lower only**  [L, ∞):  SqrtLow.   ext = L - 1 + √(int² + 1).
#   - **No bounds**            identity:  ext = int.
#
# Three operations per kind:
#   - `int2ext`:     internal → external (the optimizer's view → user's view).
#   - `ext2int`:     external → internal (user value → optimizer seed).
#   - `dint2ext`:    derivative d(ext)/d(int) — used by the chain rule for
#                    gradient and covariance transformation.
#
# **Sign-aware derivatives** (review #2 B1): `sqrtup_dint2ext` is **negative**
# for `int > 0`; this matters when transforming off-diagonal covariance
# entries between upper-only and lower-only parameters.
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================================
# Sin transformation — both bounds (lower < upper)
# ============================================================================

"""
    sin_int2ext(v, lower, upper) -> Float64

Internal-to-external for a parameter bounded in `[lower, upper]`.
Mirrors `SinParameterTransformation::Int2ext`
(`reference/Minuit2_cpp/src/SinParameterTransformation.cxx:19-23`).

`ext = lower + 0.5·(upper - lower)·(sin(v) + 1)`
"""
@inline sin_int2ext(v::Float64, lower::Float64, upper::Float64) =
    lower + 0.5 * (upper - lower) * (sin(v) + 1.0)

"""
    sin_ext2int(value, lower, upper, prec=MachinePrecision()) -> Float64

External-to-internal for a parameter bounded in `[lower, upper]`.
Mirrors `SinParameterTransformation::Ext2int`
(`reference/Minuit2_cpp/src/SinParameterTransformation.cxx:25-52`).

Clamps to `±π/2 − 8·√eps2` near the boundaries to prevent `asin(±1)`
returning the singular endpoint that the derivative blows up at.
"""
function sin_ext2int(
    value::Float64, lower::Float64, upper::Float64,
    prec::MachinePrecision = MachinePrecision(),
)
    piby2 = 2.0 * atan(1.0)           # = π/2
    distnn = 8.0 * sqrt(prec.eps2)
    vlimhi = piby2 - distnn
    vlimlo = -piby2 + distnn

    yy = 2.0 * (value - lower) / (upper - lower) - 1.0
    yy2 = yy * yy
    if yy2 > (1.0 - prec.eps2)
        return yy < 0 ? vlimlo : vlimhi
    end
    return asin(yy)
end

"""
    sin_dint2ext(v, lower, upper) -> Float64

`d(ext)/d(int) = 0.5·(upper - lower)·cos(v)`. Mirrors
`SinParameterTransformation::DInt2Ext`
(`reference/Minuit2_cpp/src/SinParameterTransformation.cxx:54-58`).
"""
@inline sin_dint2ext(v::Float64, lower::Float64, upper::Float64) =
    0.5 * (upper - lower) * cos(v)

# ============================================================================
# SqrtUp transformation — upper bound only (parameter in (-∞, upper])
# ============================================================================

"""
    sqrtup_int2ext(v, upper) -> Float64

`ext = upper + 1 - √(v² + 1)`. Mirrors
`SqrtUpParameterTransformation::Int2ext`
(`reference/Minuit2_cpp/src/SqrtUpParameterTransformation.cxx:22-27`).
"""
@inline sqrtup_int2ext(v::Float64, upper::Float64) =
    upper + 1.0 - sqrt(v * v + 1.0)

"""
    sqrtup_ext2int(value, upper, prec=MachinePrecision()) -> Float64

External-to-internal for a parameter with only an upper bound.
Mirrors `SqrtUpParameterTransformation::Ext2int`
(`reference/Minuit2_cpp/src/SqrtUpParameterTransformation.cxx:29-38`).
`prec` is accepted but unused (C++ signature parity).

Returns 0 when `(upper - value + 1)² < 1` — i.e. when `value` is too
close to `upper` to invert meaningfully.
"""
function sqrtup_ext2int(value::Float64, upper::Float64,
                         ::MachinePrecision = MachinePrecision())
    yy = upper - value + 1.0
    yy2 = yy * yy
    return yy2 < 1.0 ? 0.0 : sqrt(yy2 - 1.0)
end

"""
    sqrtup_dint2ext(v, upper) -> Float64

`d(ext)/d(int) = -v / √(v² + 1)`. **Note the negative sign** — for
`v > 0`, increasing the internal coordinate decreases the external
value (toward more negative numbers). Mirrors
`SqrtUpParameterTransformation::DInt2Ext`
(`reference/Minuit2_cpp/src/SqrtUpParameterTransformation.cxx:40-45`).

Critical for off-diagonal covariance entries: the chain rule applies
`dint2ext[i] * dint2ext[j]` to `cov[i,j]`, so a sign flip on one
factor flips the sign of the external covariance entry.
"""
@inline sqrtup_dint2ext(v::Float64, ::Float64) =
    -v / sqrt(v * v + 1.0)

# ============================================================================
# SqrtLow transformation — lower bound only (parameter in [lower, ∞))
# ============================================================================

"""
    sqrtlow_int2ext(v, lower) -> Float64

`ext = lower - 1 + √(v² + 1)`. Mirrors
`SqrtLowParameterTransformation::Int2ext`
(`reference/Minuit2_cpp/src/SqrtLowParameterTransformation.cxx:22-26`).
"""
@inline sqrtlow_int2ext(v::Float64, lower::Float64) =
    lower - 1.0 + sqrt(v * v + 1.0)

"""
    sqrtlow_ext2int(value, lower, prec=MachinePrecision()) -> Float64

External-to-internal for a parameter with only a lower bound.
Mirrors `SqrtLowParameterTransformation::Ext2int`
(`reference/Minuit2_cpp/src/SqrtLowParameterTransformation.cxx:29-38`).
`prec` accepted but unused.
"""
function sqrtlow_ext2int(value::Float64, lower::Float64,
                          ::MachinePrecision = MachinePrecision())
    yy = value - lower + 1.0
    yy2 = yy * yy
    return yy2 < 1.0 ? 0.0 : sqrt(yy2 - 1.0)
end

"""
    sqrtlow_dint2ext(v, lower) -> Float64

`d(ext)/d(int) = +v / √(v² + 1)`. **Positive sign** (cf.
[`sqrtup_dint2ext`](@ref) which is negative). Mirrors
`SqrtLowParameterTransformation::DInt2Ext`
(`reference/Minuit2_cpp/src/SqrtLowParameterTransformation.cxx:40-45`).
"""
@inline sqrtlow_dint2ext(v::Float64, ::Float64) =
    v / sqrt(v * v + 1.0)

# ============================================================================
# Bound-kind dispatch
# ============================================================================

"""
    @enum BoundKind NoBounds BothBounds UpperOnly LowerOnly

Classifies a parameter by which bounds are active. Used by
`Transformation` (in `parameters.jl`) to select the appropriate
int2ext/ext2int/dint2ext functions per parameter.
"""
@enum BoundKind begin
    NoBounds   = 0
    BothBounds = 1
    UpperOnly  = 2
    LowerOnly  = 3
end

"""
    bound_kind(lower, upper) -> BoundKind

Classify a parameter's bound configuration. `NaN` is the "absent
bound" sentinel.
"""
function bound_kind(lower::Float64, upper::Float64)
    has_lo = !isnan(lower)
    has_hi = !isnan(upper)
    if has_lo && has_hi
        lower < upper ||
            throw(ArgumentError("bound_kind: lower ($lower) must be < upper ($upper)"))
        return BothBounds
    elseif has_hi
        return UpperOnly
    elseif has_lo
        return LowerOnly
    else
        return NoBounds
    end
end

"""
    int2ext(kind, v, lower, upper) -> Float64

Dispatch to the correct internal-to-external transformation by bound
kind. For `NoBounds`, returns `v` unchanged.
"""
@inline function int2ext(kind::BoundKind, v::Float64,
                          lower::Float64, upper::Float64)
    if kind == NoBounds
        return v
    elseif kind == BothBounds
        return sin_int2ext(v, lower, upper)
    elseif kind == UpperOnly
        return sqrtup_int2ext(v, upper)
    else  # LowerOnly
        return sqrtlow_int2ext(v, lower)
    end
end

"""
    ext2int(kind, ext, lower, upper, prec=MachinePrecision()) -> Float64

Dispatch to the correct external-to-internal transformation by bound
kind.
"""
@inline function ext2int(kind::BoundKind, ext::Float64,
                          lower::Float64, upper::Float64,
                          prec::MachinePrecision = MachinePrecision())
    if kind == NoBounds
        return ext
    elseif kind == BothBounds
        return sin_ext2int(ext, lower, upper, prec)
    elseif kind == UpperOnly
        return sqrtup_ext2int(ext, upper, prec)
    else  # LowerOnly
        return sqrtlow_ext2int(ext, lower, prec)
    end
end

"""
    int2ext_error(kind, val, err, lower, upper) -> Float64

External (asymmetric-averaged) parameter error from the internal error
`err`. **`err` is the errordef-SCALED internal 1σ error**, i.e.
`err = sqrt(cov(i,i)) = sqrt(2·up·V_int[i,i])` — NOT the raw
`sqrt(V_int[i,i])`. The C++ comment is `err = sigma Value ==
std::sqrt(cov(i,i))` and the caller passes
`std::sqrt(2.*up*Error().InvHessian()(i,i))`
(`reference/Minuit2_cpp/src/MnUserParameterState.cxx:142`). Mirrors C++
`MnUserTransformation::Int2extError`
(`reference/Minuit2_cpp/src/MnUserTransformation.cxx:115-141`).

For unbounded parameters: returns `err` unchanged.

For bounded parameters: computes the symmetric average of the
two-sided perturbations through `int2ext`:

    ui = int2ext(val)
    du1 = int2ext(val + err) - ui
    du2 = int2ext(val - err) - ui
    return 0.5 · (|du1| + |du2|)

with a special clamp for double-bounded when `err > 1` (the
sin-transform saturates, so |du1| is replaced by the full range).

Phase 1.x D5 (codex parallel-review #4) — closes the near-bound
error mis-reporting gap. The Jacobian-diagonal alternative
`sqrt(V_ext[i,i]) = D · sqrt(V_int[i,i])` underscores near bounds
because D = d(ext)/d(int) shrinks toward zero; the two-sided
formula captures the actual nonlinear remapping.
"""
function int2ext_error(
    kind::BoundKind, val::Float64, err::Float64,
    lower::Float64, upper::Float64,
)
    kind == NoBounds && return err

    ui = int2ext(kind, val, lower, upper)
    du1 = int2ext(kind, val + err, lower, upper) - ui
    du2 = int2ext(kind, val - err, lower, upper) - ui

    # Double-bounded saturation clamp (C++ MnUserTransformation.cxx:132-133)
    if kind == BothBounds && err > 1.0
        du1 = upper - lower
    end

    return 0.5 * (abs(du1) + abs(du2))
end

"""
    dint2ext(kind, v, lower, upper) -> Float64

Dispatch to `d(ext)/d(int)`. For `NoBounds`, returns `1.0`.
"""
@inline function dint2ext(kind::BoundKind, v::Float64,
                           lower::Float64, upper::Float64)
    if kind == NoBounds
        return 1.0
    elseif kind == BothBounds
        return sin_dint2ext(v, lower, upper)
    elseif kind == UpperOnly
        return sqrtup_dint2ext(v, upper)
    else  # LowerOnly
        return sqrtlow_dint2ext(v, lower)
    end
end
