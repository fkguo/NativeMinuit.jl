# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# covariance_squeeze.jl — MnCovarianceSqueeze.
#
# Mirrors reference/Minuit2_cpp/src/MnCovarianceSqueeze.cxx.
#
# Removes a row+column from a symmetric matrix. Three flavors:
#  1. Pure matrix:        H_squeezed = H[setdiff, setdiff]
#  2. MinimumError:       invert → squeeze H → invert back, with fall-back
#                         diagonal if either inversion fails.
#  3. User covariance:    invert covariance → squeeze H → invert back, with
#                         the same fall-back. Used when a parameter is
#                         fixed-then-released or vice versa.
# ─────────────────────────────────────────────────────────────────────────────

"""
    squeeze_symmetric(M::Symmetric, n::Integer) -> Symmetric

Return a new symmetric matrix with row + column `n` removed. Index is
1-based (Julia). Mirrors
`reference/Minuit2_cpp/src/MnCovarianceSqueeze.cxx:89-109` (the
3-argument overload) up to base-index conversion.
"""
function squeeze_symmetric(M::Symmetric{Float64,Matrix{Float64}}, n::Integer)
    p = parent(M)
    nrow = LinearAlgebra.checksquare(p)
    1 <= n <= nrow ||
        throw(ArgumentError("squeeze index $n out of bounds for $(nrow)x$(nrow) matrix"))
    nrow > 1 ||
        throw(ArgumentError("cannot squeeze a 1x1 matrix"))

    out_n = nrow - 1
    out = zeros(Float64, out_n, out_n)
    @inbounds for i in 1:nrow
        i == n && continue
        oi = i < n ? i : i - 1
        for k in i:nrow
            k == n && continue
            ok = k < n ? k : k - 1
            out[oi, ok] = M[i, k]
        end
    end
    return Symmetric(out, :U)
end

"""
    squeeze_error(err::MinimumError, n::Integer; prec=MachinePrecision())
        -> MinimumError

Remove parameter `n` from a `MinimumError`. The error matrix (inverse
Hessian `V`) is inverted to get the Hessian `H`, `H` is squeezed, then
re-inverted to the new error matrix.

Two inversion-failure fallbacks, each matching C++
`MnCovarianceSqueeze::operator()(const MinimumError&, unsigned int)`
(`reference/Minuit2_cpp/src/MnCovarianceSqueeze.cxx:65-87`):

- **First inversion (`V → H`) fails.** In C++ this inversion lives
  *inside* `err.Hessian()` (`BasicMinimumError::Hessian()`,
  `BasicMinimumError.cxx:20-35`), which on failure returns the diagonal
  Hessian `diag(1/V[i,i])`. `MnCovarianceSqueeze` then squeezes that
  diagonal and re-inverts it — a diagonal is trivially invertible — so
  the overall result is the **valid** `MinimumError(squeezed,
  err.Dcovar())`, *not* a failure. We short-circuit to the
  algebraically-identical diagonal `diag(V[i,i])` (the `1/(1/V[i,i])`
  round-trip) and tag it **valid** carrying `err.dcovar`, matching C++
  (audit §12 — this path was previously mis-tagged `MnInvertFailed`).
- **Second inversion (squeezed `H → V`) fails.** Returns a diagonal
  `MinimumError` of `1/H_squeezed[i,i]` tagged `MnInvertFailed`
  (`MnCovarianceSqueeze.cxx:76-84`).

The valid-path `MinimumError`'s `dcovar` is preserved from the input
(matches C++ line 86: `MinimumError(squeezed, err.Dcovar())`).
"""
function squeeze_error(err::MinimumError, n::Integer;
                        prec::MachinePrecision = MachinePrecision())
    # Step 1: invert the inverse-Hessian (V) to get the Hessian (H)
    H = copy(parent(err.inv_hessian))
    Hsym = Symmetric(H, :U)
    inv_ok_1 = true
    try
        sym_invert!(Hsym)
    catch
        inv_ok_1 = false
    end

    if !inv_ok_1
        # First inversion (V → H) failed. C++ does NOT bail to a failure
        # status here: this inversion lives inside `err.Hessian()`
        # (BasicMinimumError.cxx:20-35), which on failure returns the
        # diagonal Hessian diag(1/V[i,i]). MnCovarianceSqueeze then
        # squeezes that diagonal and re-inverts it (a diagonal inverts
        # cleanly), yielding the VALID `MinimumError(squeezed, err.Dcovar())`
        # whose diagonal is diag(1/(1/V[i,i])) = diag(V[i,i]). We produce
        # the same diagonal directly and tag it VALID carrying `err.dcovar`
        # — matching C++ (audit §12; this was previously mis-tagged
        # MnInvertFailed).
        n_in = LinearAlgebra.checksquare(parent(err.inv_hessian))
        diag = zeros(Float64, n_in - 1, n_in - 1)
        oi = 0
        for i in 1:n_in
            i == n && continue
            oi += 1
            diag[oi, oi] = err.inv_hessian[i, i]  # = 1/(1/V[i,i]) round-trip
        end
        return MinimumError(Symmetric(diag, :U), err.dcovar)
    end

    # Step 2: squeeze the Hessian
    Hs = squeeze_symmetric(Hsym, n)

    # Step 3: squeeze_symmetric allocated Hs, so its storage can hold
    # the inverse. The full Hessian still carries the original diagonal.
    inv_ok_2 = true
    try
        sym_invert!(Hs)
    catch
        inv_ok_2 = false
    end

    if !inv_ok_2
        # A failed factorization may already have overwritten Hs's diagonal.
        # Read the same pre-inversion entries from the untouched full Hessian.
        n_out = LinearAlgebra.checksquare(parent(Hs))
        diag = zeros(Float64, n_out, n_out)
        for i in 1:n_out
            source = i < n ? i : i + 1
            diag[i, i] = 1.0 / Hsym[source, source]
        end
        return MinimumError(Symmetric(diag, :U), MnInvertFailed)
    end

    # Preserve dcovar from the input error (C++ line 86).
    return MinimumError(Hs, err.dcovar)
end
