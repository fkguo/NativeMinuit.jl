# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# linalg.jl — JuMinuit's thin linear-algebra layer.
#
# JuMinuit Phase 0 uses dense `Symmetric{Float64,Matrix{Float64}}` for the
# inverse Hessian (DR-004). Storage convention: **`:U` upper-triangle
# authoritative** — matches Julia's default `Symmetric(M)` and BLAS's
# default `'U'` in `syr!`. C++ Minuit2 uses lower-triangular packed
# (`LASymMatrix`) but we abandon both packed BLAS and ABObj expression
# templates in favor of dense + `LinearAlgebra.BLAS.*` (ROADMAP §2.2).
#
# This file collects the kernels MIGRAD's hot path needs as a named,
# documented surface — so a future packed-storage variant can be
# swapped in at a single point if benchmarks ever demand it.
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra: BLAS, LAPACK

"""
    SYMMETRIC_UPLO

The triangle convention used throughout JuMinuit: `:U` (upper-triangle
authoritative). Matches the default `Symmetric(M)` and the default
`'U'` in `BLAS.syr!`. Document once; use everywhere.
"""
const SYMMETRIC_UPLO = :U

# ─────────────────────────────────────────────────────────────────────────────
# In-place symmetric rank-1 update (DSYR)
# ─────────────────────────────────────────────────────────────────────────────

"""
    sym_rank1_update!(S, α, x) -> S

In-place symmetric rank-1 update: `S ← S + α·x·xᵀ`.

Wraps `BLAS.syr!` on the underlying `parent(S)` matrix so callers don't
need to remember the upper/lower convention. Replaces the C++
`Outer_product(x) * α` (an ABObj expression template — see
`reference/Minuit2_cpp/src/LaOuterProduct.cxx:55–58`) with one DSYR
BLAS call.

Used by the DFP Hessian update in Phase 0 day 13–18:

```julia
sym_rank1_update!(vUpd,  1/delgam, dx)   # rank-2 base, term 1
sym_rank1_update!(vUpd, -1/gvg,    vg)   # rank-2 base, term 2
if delgam > gvg
    @. dx_minus_vg = dx/delgam - vg/gvg
    sym_rank1_update!(vUpd, gvg, dx_minus_vg)  # rank-1 additive
end
```

Zero allocation when `S` is dense and `x` is a concrete `Vector{Float64}`.
"""
@inline function sym_rank1_update!(
    S::Symmetric{Float64,Matrix{Float64}},
    α::Real,
    x::AbstractVector{Float64},
)
    BLAS.syr!(S.uplo, Float64(α), x, parent(S))
    return S
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place symmetric matrix-vector (DSYMV)
# ─────────────────────────────────────────────────────────────────────────────

"""
    sym_mul!(y, S, x, α=1.0, β=0.0) -> y

In-place symmetric matrix-vector product: `y ← α·S·x + β·y`.

Thin alias for `LinearAlgebra.mul!(y, S, x, α, β)` — Julia dispatches
this to BLAS DSYMV automatically for `Symmetric{Float64,Matrix{Float64}}`.

Named alias so MIGRAD code reads clearly and the implementation can be
swapped at one point if a packed variant ever wins on benchmarks.

Used for:
- `step = -V · g` in `VariableMetricBuilder.cxx:243`: `sym_mul!(step, V, g, -1.0, 0.0)`
- `vg = V · dg` in `DavidonErrorUpdator.cxx:58`: `sym_mul!(vg, V, dg)`
"""
@inline function sym_mul!(
    y::AbstractVector{Float64},
    S::Symmetric{Float64,Matrix{Float64}},
    x::AbstractVector{Float64},
    α::Real = 1.0,
    β::Real = 0.0,
)
    mul!(y, S, x, Float64(α), Float64(β))
    return y
end

# ─────────────────────────────────────────────────────────────────────────────
# Symmetric inverse via LAPACK Bunch–Kaufman
# ─────────────────────────────────────────────────────────────────────────────

"""
    sym_invert!(S; throw_on_fail=true) -> S

In-place symmetric matrix inversion via LAPACK Bunch–Kaufman
(`sytrf!` + `sytri!`). After the call, `S` holds `inv(S_original)`.

Replaces the C++ `mnvert.cxx` Gauss-Jordan path. For n ≥ ~8 this is
both faster and numerically more accurate; for n ≤ 4 the speeds are
comparable.

Used at:
- Seed time: trivial diagonal (initial step sizes → initial inverse Hessian).
- `MnPosDef`: after eigenvalue perturbation.
- Final HESSE (Phase 1): when we have the second derivatives and need
  the covariance.

# Failure modes

- Singular matrix → `LinearAlgebra.SingularException` (or `(S, :singular)`
  with `throw_on_fail=false`).
- Other LAPACK errors propagate.

Returns `S` (mutated to hold the inverse).
"""
function sym_invert!(
    S::Symmetric{Float64,Matrix{Float64}};
    throw_on_fail::Bool = true,
)
    M = parent(S)
    LinearAlgebra.checksquare(M)
    uplo_char = S.uplo isa AbstractChar ? S.uplo : Char(S.uplo)

    # Bunch-Kaufman in-place factorization.
    _, ipiv, info = LAPACK.sytrf!(uplo_char, M)
    if info > 0
        throw_on_fail && throw(LinearAlgebra.SingularException(info))
        return S  # caller can inspect parent(S) and treat as invalid
    end

    # In-place inversion using the factorization.
    LAPACK.sytri!(uplo_char, M, ipiv)
    return S
end

# ─────────────────────────────────────────────────────────────────────────────
# Symmetric eigenvalues (for MnPosDef)
# ─────────────────────────────────────────────────────────────────────────────

"""
    sym_eigvals(S) -> Vector{Float64}

Eigenvalues of a symmetric matrix, sorted ascending. Wraps
`LinearAlgebra.eigvals(S)` (LAPACK `syevr!` under the hood).

Used inside `MnPosDef` (`reference/Minuit2_cpp/src/MnPosDef.cxx:80`)
to detect non-positive-definite Hessians and apply the
add-to-diagonal trick. Reminder (per ROADMAP §2.2 + DR): the matrix
fed there is the **normalized correlation** form (`1/sqrt(diag)`
scaling), not the raw error matrix.
"""
sym_eigvals(S::Symmetric{Float64,Matrix{Float64}}) = eigvals(S)
