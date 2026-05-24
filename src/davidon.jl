# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# davidon.jl — DFP (Davidon-Fletcher-Powell) inverse Hessian update.
#
# Mirrors reference/Minuit2_cpp/src/DavidonErrorUpdator.cxx:24-73.
#
# **CRITICAL NUMERICAL SEMANTIC** (parallel reviewer flag, blocking #1):
# the C++ formula at DavidonErrorUpdator.cxx:60-65 computes the rank-2
# DFP base **unconditionally**, then **adds** a rank-1 correction
# *on top* when `delgam > gvg`. It is **not** a branched if/else
# choosing rank-2 or rank-1 — that misreading silently diverges from
# C++ after one update. The comment in the C++ source (`// use rank 1
# formula`) is misleading; the mathematics is a BFGS-style rank-3 hybrid.
#
# FLOP order in this Julia port matches C++ lines 58-72 line-for-line so
# the 1e-12 numerical-equivalence trace audit (Risk #1 mitigation (c))
# remains achievable.
# ─────────────────────────────────────────────────────────────────────────────

"""
    davidon_update!(V, dx, dg, prev_dcovar, vg_work, vUpd_work) -> (new_dcovar, status)

In-place DFP update of the inverse Hessian `V` (a
`Symmetric{Float64,Matrix{Float64}}`) given the parameter change `dx`
and gradient change `dg` from one MIGRAD iteration to the next, plus
the previous iteration's `Dcovar` (a quality estimator C++ uses to
gate the inner `MnHesse` call when Strategy ≥ 1).

Mirrors C++ `DavidonErrorUpdator::Update` at
`reference/Minuit2_cpp/src/DavidonErrorUpdator.cxx:24-73`.

# Mathematical formula

Let `v₀ = V` (input), `δx = dx`, `δg = dg`,
`δ = δx·δg`, `γ = δg' · V · δg`. Then:

```
vUpd  = (δx ⊗ δx) / δ  −  (V·δg ⊗ V·δg) / γ            [rank-2 base]
if δ > γ:
    vUpd += γ · ((δx/δ − V·δg/γ) ⊗ (δx/δ − V·δg/γ))    [rank-1 additive]
sum_upd = ∑ vUpd        [before adding V]
V += vUpd               [in place]
new_dcov = ½ (prev_dcov + sum_upd / ∑V)
```

The summation `∑M` is over the authoritative triangle plus diagonal
(matches C++ `sum_of_elements(LASymMatrix)`); see `sum_sym`.

# Arguments

- `V::Symmetric{Float64,Matrix{Float64}}` — input/output: the inverse
  Hessian. Mutated in-place to hold `V_new`.
- `dx::AbstractVector{Float64}` — parameter change `x_new − x_old`.
- `dg::AbstractVector{Float64}` — gradient change `g_new − g_old`.
- `prev_dcovar::Real` — previous iteration's `Dcovar`.
- `vg_work::AbstractVector{Float64}` — preallocated workspace of
  length `n`. Used to hold `V·dg`, then later (during the rank-1
  branch) overwritten with `dx/δ − vg/γ`.
- `vUpd_work::Symmetric{Float64,Matrix{Float64}}` — preallocated
  workspace symmetric matrix of size `n×n`, same `uplo` as `V`.

# Returns

A tuple `(new_dcovar::Float64, status::Symbol)`:
- `(_, :updated)` — successful update; `V` and `vUpd_work` have been
  written.
- `(prev_dcovar, :unchanged_delgam_zero)` — `δ = 0`; per C++ line 43-46,
  cannot update; `V` is untouched.
- `(prev_dcovar, :unchanged_gvg_nonpositive)` — `γ ≤ 0`; per C++ line
  52-56, cannot update; `V` is untouched.

A negative `δ` (first derivatives increasing along the search line) is
warned but not blocking — matches C++ behavior at lines 48-50.
"""
function davidon_update!(
    V::Symmetric{Float64,Matrix{Float64}},
    dx::AbstractVector{Float64},
    dg::AbstractVector{Float64},
    prev_dcovar::Real,
    vg_work::AbstractVector{Float64},
    vUpd_work::Symmetric{Float64,Matrix{Float64}},
)
    n = LinearAlgebra.checksquare(parent(V))
    length(dx) == n ||
        throw(DimensionMismatch("dx length $(length(dx)) != V size $n"))
    length(dg) == n ||
        throw(DimensionMismatch("dg length $(length(dg)) != V size $n"))
    length(vg_work) == n ||
        throw(DimensionMismatch("vg_work length $(length(vg_work)) != V size $n"))
    LinearAlgebra.checksquare(parent(vUpd_work)) == n ||
        throw(DimensionMismatch("vUpd_work size != V size"))
    V.uplo == vUpd_work.uplo ||
        throw(ArgumentError("V and vUpd_work must share triangle convention"))

    prev_dc_f = Float64(prev_dcovar)

    # ── Step 1: scalar δ = dx · dg ────────────────────────────────────
    delgam = dot(dx, dg)
    if delgam == 0
        @warn "DFP update: delgam = 0; cannot update (matrix unchanged)"
        return (prev_dc_f, :unchanged_delgam_zero)
    end
    if delgam < 0
        @warn "DFP update: delgam < 0 — first derivatives increasing along search line"
    end

    # ── Step 2: vg = V·dg, γ = dg·vg ─────────────────────────────────
    sym_mul!(vg_work, V, dg)
    gvg = dot(dg, vg_work)
    if gvg <= 0
        @warn "DFP update: gvg ≤ 0; cannot update (matrix unchanged)"
        return (prev_dc_f, :unchanged_gvg_nonpositive)
    end

    # ── Step 3: rank-2 base in vUpd_work ─────────────────────────────
    # vUpd = (dx ⊗ dx)/δ − (vg ⊗ vg)/γ
    fill!(parent(vUpd_work), 0.0)        # zero authoritative + non-auth
    sym_rank1_update!(vUpd_work,  1.0 / delgam, dx)
    sym_rank1_update!(vUpd_work, -1.0 / gvg,    vg_work)

    # ── Step 4: optional rank-1 additive correction ──────────────────
    # When δ > γ: vUpd += γ · ((dx/δ − vg/γ) ⊗ (dx/δ − vg/γ))
    # Reuse vg_work as scratch — it's no longer needed for itself.
    if delgam > gvg
        @inbounds for i in 1:n
            vg_work[i] = dx[i] / delgam - vg_work[i] / gvg
        end
        sym_rank1_update!(vUpd_work, gvg, vg_work)
    end

    # ── Step 5: sum_upd from update-only matrix (BEFORE += V) ─────────
    sum_upd = sum_sym(vUpd_work)

    # ── Step 6: V ← V + vUpd_work (in place, authoritative triangle) ─
    add_sym!(V, vUpd_work)

    # ── Step 7: new dcov ──────────────────────────────────────────────
    new_dcov = 0.5 * (prev_dc_f + sum_upd / sum_sym(V))

    return (new_dcov, :updated)
end

"""
    davidon_update(error, dx, dg) -> MinimumError

Allocating convenience wrapper. Constructs a fresh `MinimumError`
holding the updated inverse Hessian and new `Dcovar`. The previous
`error.status` is propagated unchanged.

Allocates: one workspace vector + one workspace symmetric matrix.
"""
function davidon_update(
    err::MinimumError,
    dx::AbstractVector{Float64},
    dg::AbstractVector{Float64},
)
    n = size(err)[1]
    # Deep-copy the inverse Hessian so the input is preserved.
    V_new = Symmetric(copy(parent(err.inv_hessian)), err.inv_hessian.uplo == 'U' ? :U : :L)
    vg_work = Vector{Float64}(undef, n)
    vUpd_work = Symmetric(zeros(n, n), V_new.uplo == 'U' ? :U : :L)
    new_dcov, status = davidon_update!(V_new, dx, dg, err.dcovar, vg_work, vUpd_work)
    # `status` is informational — return the new MinimumError regardless.
    return MinimumError(V_new, new_dcov, err.status, true)
end
