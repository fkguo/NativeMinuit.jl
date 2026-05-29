# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# posdef.jl — MnPosDef equivalent.
#
# Mirrors reference/Minuit2_cpp/src/MnPosDef.cxx:23-104.
#
# Forces the inverse-Hessian (covariance) matrix to be positive-definite
# by (a) adding to the diagonal if min(diag) ≤ 0, then (b) computing
# eigenvalues of the **normalized correlation** form (1/√diag scaling
# applied first) and adding to the diagonal if min eigenvalue is too
# small relative to max eigenvalue.
#
# Called from MnSeedGenerator + VariableMetricBuilder when an updated
# inverse Hessian goes non-pos-def, common at the start of MIGRAD or
# after a problematic DFP update.
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_posdef!(err::MinimumError, prec=MachinePrecision()) -> MinimumError

In-place: enforce positive-definiteness of `err.inv_hessian`. Mirrors
`MnPosDef::operator()(const MinimumError&, ...)` from
`reference/Minuit2_cpp/src/MnPosDef.cxx:30-104`.

Returns a **new** `MinimumError` (not the same instance) — matches C++
semantic which constructs a new BasicMinimumError. The original `err`
is untouched. Status of the returned object:

- `MnHesseValid` — if `err` passed the eigenvalue gate (C++ MnPosDef.cxx:85-86:
  `return MinimumError(err, e.Dcovar())`, the (matrix, dcovar) ctor that forces
  valid+pos-def). The incoming `err.dcovar` is preserved; the incoming `status`
  is NOT — a stale `MnMadePosDef` must not survive the gate (audit §11b).
- `MnMadePosDef` — if any diagonal perturbation was applied (C++ line 103) or
  the n=1 clamp fired (line 39). `dcovar` is forced to `1.0` to match the C++
  `BasicMinimumError` tag constructor; this drives MIGRAD's
  `edm_corrected = edm·(1+3·dcovar)` after a pos-def event (audit §11a).

# Algorithm (per `MnPosDef.cxx:30-104`)

1. **n=1 fast path**: if matrix is 1×1, just clamp to 1.0 if below `eps`.
2. **Diagonal floor**: find `dgmin = min(diag(err))`. If `dgmin ≤ 0`,
   add `dg = 0.5 + epspdf − dgmin` to all diagonal entries (`epspdf =
   max(1e-6, eps2)`). Any post-add still-negative diagonal is forced to 1.
3. **Normalized correlation matrix**: build `p[i,j] = err[i,j] /
   √(err[i,i]·err[j,j])`. Diagonals of `p` are all 1 by construction.
4. **Eigenvalue check**: compute eigenvalues of `p`. If
   `λ_min > epspdf · max(|λ_max|, 1)`, the matrix is pos-def-enough;
   return.
5. **Add `padd = 0.001·λ_max − λ_min`** to all diagonals of err, where
   λ_max here is `max(|λ_max|, 1)`. Mark `MnMadePosDef`.

# Phase 0

Used by `MnSeedGenerator` (via Phase-1 wiring) and by
`VariableMetricBuilder` Strategy ≥ 1 (Phase 1). For Phase 0
Strategy = 0 the seed doesn't always call this, but the algorithm
must exist when seed.jl lands.
"""
function make_posdef(err::MinimumError, prec::MachinePrecision = MachinePrecision())
    M_in = parent(err.inv_hessian)
    n = LinearAlgebra.checksquare(M_in)

    eps = prec.eps
    eps2 = prec.eps2

    # n=1 fast path (MnPosDef.cxx:37-43)
    if n == 1
        if M_in[1, 1] < eps
            new_M = fill(0.0, 1, 1)
            new_M[1, 1] = 1.0
            # C++ MnPosDef.cxx:39 `MinimumError(err, MnMadePosDef())` — tag ctor
            # forces dcovar = 1.0 (BasicMinimumError.h), not the incoming dcovar.
            return MinimumError(Symmetric(new_M, :U), 1.0, MnMadePosDef, true)
        elseif M_in[1, 1] > eps
            return err  # already valid (C++ MnPosDef.cxx:41)
        end
        # M_in[1,1] == eps exactly: C++ takes NEITHER n==1 early return
        # (MnPosDef.cxx:37/41 use strict `<` / `>`), so it falls through to the
        # eigenvalue gate, which forces valid+pos-def. Fall through here too
        # (the general path below handles n=1 and returns MnHesseValid).
    end

    # Work on a fresh copy — C++ takes a copy of err matrix at line 36
    err_M = copy(M_in)

    # Diagonal floor (MnPosDef.cxx:46-65)
    epspdf = max(1.0e-6, eps2)
    dgmin = err_M[1, 1]
    for i in 1:n
        if err_M[i, i] < dgmin
            dgmin = err_M[i, i]
        end
    end

    dg = 0.0
    if dgmin <= 0
        dg = 0.5 + epspdf - dgmin
    end

    # Build the normalized correlation matrix p and apply diagonal add
    # (MnPosDef.cxx:67-77). Both p and err_M are upper-triangle authoritative.
    p_M = zeros(n, n)
    s = zeros(n)
    for i in 1:n
        # Add dg to diagonal; floor negative to 1
        err_M[i, i] += dg
        if err_M[i, i] < 0
            err_M[i, i] = 1.0
        end
        s[i] = 1.0 / sqrt(err_M[i, i])
        # p[i,j] = err[i,j] * s[i] * s[j] for j ≤ i, mirrored to upper.
        # We write the upper triangle of p:
        for j in 1:i
            # In our :U convention, the authoritative entry of err for (i,j) when
            # j ≤ i is err_M[j, i] (the upper-triangle storage), and
            # err[i,j] == err[j,i] mathematically. So p[j,i] = err[j,i]·s[i]·s[j].
            p_M[j, i] = err_M[j, i] * s[i] * s[j]
        end
    end
    p_sym = Symmetric(p_M, :U)

    # Eigenvalue gate (MnPosDef.cxx:80-86)
    eval_ = sym_eigvals(p_sym)
    pmin = eval_[1]                 # eigvals returns sorted ascending
    pmax = eval_[end]
    pmax = max(abs(pmax), 1.0)      # MnPosDef.cxx:84
    if pmin > epspdf * pmax
        # Pos-def enough. C++ MnPosDef.cxx:85-86 returns `MinimumError(err,
        # e.Dcovar())` — the (matrix, dcovar) ctor, which forces valid+pos-def
        # (fValid=fPosDef=true, fMadePosDef=false) while KEEPING the incoming
        # dcovar. Preserving err.status here (v1) could carry a stale
        # MnMadePosDef across the gdel>0→edm<0 re-invocation within a single
        # MIGRAD iteration (audit §11b) — force MnHesseValid instead.
        return MinimumError(Symmetric(err_M, :U), err.dcovar, MnHesseValid, true)
    end

    # Force pos-def: add (0.001·pmax − pmin) · diag (MnPosDef.cxx:88-91)
    padd = 0.001 * pmax - pmin
    for i in 1:n
        err_M[i, i] *= (1.0 + padd)
    end

    # C++ MnPosDef.cxx:103 returns `MinimumError(err, MnMadePosDef())` — the tag
    # ctor forces dcovar = 1.0 (BasicMinimumError.h), inflating MIGRAD's
    # edm·(1+3·dcovar) correction after the pos-def event (audit §11a). v1
    # passed the incoming err.dcovar, under-inflating the correction.
    return MinimumError(Symmetric(err_M, :U), 1.0, MnMadePosDef, true)
end

"""
    make_posdef(state::MinimumState, prec=MachinePrecision()) -> MinimumState

Apply `make_posdef` to the error component of a `MinimumState`,
returning a new state with the same parameters/gradient/edm/nfcn but
a possibly-corrected error matrix. Mirrors
`MnPosDef::operator()(const MinimumState&, ...)` at
`reference/Minuit2_cpp/src/MnPosDef.cxx:23-28`.
"""
function make_posdef(state::MinimumState, prec::MachinePrecision = MachinePrecision())
    new_err = make_posdef(state.error, prec)
    return MinimumState(state.parameters, new_err, state.gradient,
                        state.edm, state.nfcn)
end

"""
    is_posdef_enough(err::MinimumError, prec=MachinePrecision()) -> Bool

Quick pre-check: returns `true` if `err.inv_hessian` is already
positive-definite to MIGRAD's tolerance (i.e. `make_posdef` would
return it essentially unchanged). Useful for skipping the eigenvalue
work when the matrix is obviously fine.
"""
function is_posdef_enough(err::MinimumError, prec::MachinePrecision = MachinePrecision())
    M = parent(err.inv_hessian)
    n = size(M, 1)
    eps2 = prec.eps2
    epspdf = max(1.0e-6, eps2)

    # Quick diagonal check
    for i in 1:n
        if M[i, i] <= 0
            return false
        end
    end

    # Eigenvalue check on normalized correlation
    s = [1.0 / sqrt(M[i, i]) for i in 1:n]
    p_M = zeros(n, n)
    for i in 1:n, j in 1:i
        p_M[j, i] = M[j, i] * s[i] * s[j]
    end
    eval_ = sym_eigvals(Symmetric(p_M, :U))
    pmax = max(abs(eval_[end]), 1.0)
    return eval_[1] > epspdf * pmax
end
