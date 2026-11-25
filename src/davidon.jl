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
- `vUpd_work` — a preallocated symmetric matrix of size `n×n` with the
  same `uplo` as `V`. For a fit-owned upper-triangle `V`, a vector of length
  `n*(n+1)÷2` can hold the original triangle while `V` holds the update.
  In that case `V` and the packed buffer must be private and disjoint from
  `dx`, `dg`, and `vg_work`.

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
    vUpd_work::Union{Symmetric{Float64,Matrix{Float64}},Vector{Float64}},
)
    n = LinearAlgebra.checksquare(parent(V))
    length(dx) == n ||
        throw(DimensionMismatch("dx length $(length(dx)) != V size $n"))
    length(dg) == n ||
        throw(DimensionMismatch("dg length $(length(dg)) != V size $n"))
    length(vg_work) == n ||
        throw(DimensionMismatch("vg_work length $(length(vg_work)) != V size $n"))
    _check_davidon_workspace(V, vUpd_work, n)

    prev_dc_f = Float64(prev_dcovar)

    # ── Step 1: scalar δ = dx · dg ────────────────────────────────────
    delgam = dot(dx, dg)
    if delgam == 0
        @warn "DFP update: delgam = 0; cannot update (matrix unchanged)"
        return (prev_dc_f, :unchanged_delgam_zero)
    end
    if delgam < 0
        # @debug (not @warn) per parallel-review A6: this fires on a
        # normal mid-fit condition (first derivatives increasing along
        # search line). C++ uses MnPrint::Warn which is suppressed by
        # default; Julia @warn prints by default and allocates ~160 B
        # per fire — too noisy + allocates in the hot path.
        @debug "DFP update: delgam < 0 — first derivatives increasing along search line"
    end

    # ── Step 2: vg = V·dg, γ = dg·vg ─────────────────────────────────
    sym_mul!(vg_work, V, dg)
    gvg = dot(dg, vg_work)
    if gvg <= 0
        @warn "DFP update: gvg ≤ 0; cannot update (matrix unchanged)"
        return (prev_dc_f, :unchanged_gvg_nonpositive)
    end

    # ── Step 3: rank-2 base in the cleared triangle ─────────────────
    # vUpd = (dx ⊗ dx)/δ − (vg ⊗ vg)/γ
    delta = _davidon_workspace!(V, vUpd_work, n)
    sym_rank1_update!(delta,  1.0 / delgam, dx)
    sym_rank1_update!(delta, -1.0 / gvg,    vg_work)

    # ── Step 4: optional rank-1 additive correction ──────────────────
    # When δ > γ: vUpd += γ · ((dx/δ − vg/γ) ⊗ (dx/δ − vg/γ))
    # Reuse vg_work as scratch — it's no longer needed for itself.
    if delgam > gvg
        @inbounds for i in 1:n
            vg_work[i] = dx[i] / delgam - vg_work[i] / gvg
        end
        sym_rank1_update!(delta, gvg, vg_work)
    end

    sum_upd, sum_new = _davidon_add_and_sum!(V, vUpd_work, n)
    new_dcov = 0.5 * (prev_dc_f + sum_upd / sum_new)

    return (new_dcov, :updated)
end

function _check_davidon_workspace(V, work::Symmetric, n)
    LinearAlgebra.checksquare(parent(work)) == n ||
        throw(DimensionMismatch("vUpd_work size != V size"))
    V.uplo == work.uplo ||
        throw(ArgumentError("V and vUpd_work must share triangle convention"))
    return nothing
end

function _check_davidon_workspace(V, work::Vector{Float64}, n)
    length(work) == n * (n + 1) ÷ 2 ||
        throw(DimensionMismatch("vUpd_work length != packed triangle size"))
    V.uplo == 'U' ||
        throw(ArgumentError("packed Davidon workspace requires the upper triangle"))
    return nothing
end

function _davidon_workspace!(V, work::Symmetric, n)
    fill!(parent(work), 0.0)
    return work
end

function _davidon_workspace!(V, work::Vector{Float64}, n)
    # Save the original triangle and use V as the destination for the same
    # BLAS.syr! calls. BLAS.spr! can round differently on some backends.
    matrix = parent(V)
    k = 0
    @inbounds for j in 1:n, i in 1:j
        k += 1
        work[k] = matrix[i, j]
        matrix[i, j] = 0.0
    end
    return V
end

# The caller checks the sizes and triangle convention. Keep the order of each
# sum_sym reduction and compute the matrix entry with the original addition.
function _davidon_add_and_sum!(V::Symmetric{Float64,Matrix{Float64}},
                               update::Symmetric{Float64,Matrix{Float64}}, n::Int)
    old = parent(V)
    delta = parent(update)
    if Base.mightalias(old, delta)
        sum_upd = sum_sym(update)
        add_sym!(V, update)
        return sum_upd, sum_sym(V)
    end
    sum_upd = 0.0
    sum_new = 0.0
    upper = V.uplo == 'U'
    @inbounds for j in 1:n
        for i in (upper ? (1:j) : (j:n))
            entry = delta[i, j]
            value = old[i, j] + entry
            old[i, j] = value
            sum_upd += abs(entry)
            sum_new += abs(value)
        end
    end
    return sum_upd, sum_new
end

function _davidon_add_and_sum!(V::Symmetric{Float64,Matrix{Float64}},
                               original::Vector{Float64}, n::Int)
    matrix = parent(V)
    sum_upd = 0.0
    sum_new = 0.0
    k = 0
    @inbounds for j in 1:n, i in 1:j
        k += 1
        entry = matrix[i, j]
        value = original[k] + entry
        matrix[i, j] = value
        sum_upd += abs(entry)
        sum_new += abs(value)
    end
    return sum_upd, sum_new
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
