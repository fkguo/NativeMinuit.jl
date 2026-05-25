# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# CovStatus — covariance-matrix status enum (DR-006).
# Mirrors the tag-typed status set in
# reference/Minuit2_cpp/inc/Minuit2/MinimumError.h:29–32 plus the implicit
# "valid" case. Inferable, printable, dispatchable on.
# ─────────────────────────────────────────────────────────────────────────────

"""
    CovStatus

Status of a `MinimumError` covariance matrix. Mirrors the C++ Minuit2 tag
types (`MnHesseFailed`, `MnInvertFailed`, `MnMadePosDef`, `MnNotPosDef`)
plus the implicit "valid" case.

Values:
- `MnHesseValid` — Hesse calculation succeeded; matrix is accurate.
- `MnHesseFailed` — Hesse failed during refinement.
- `MnMadePosDef` — Matrix was forced positive-definite via `MnPosDef`.
- `MnInvertFailed` — Matrix inversion failed.
- `MnNotPosDef` — Matrix is not positive-definite.

`MnHesseValid` is the default for successful MIGRAD-only convergence.
"""
@enum CovStatus begin
    MnHesseValid    = 0
    MnHesseFailed   = 1
    MnMadePosDef    = 2
    MnInvertFailed  = 3
    MnNotPosDef     = 4
end

# ─────────────────────────────────────────────────────────────────────────────
# MinimumParameters — current point + step + function value.
# Mirrors reference/Minuit2_cpp/inc/Minuit2/BasicMinimumParameters.h.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MinimumParameters

Mirror of `MinimumParameters` from
`reference/Minuit2_cpp/inc/Minuit2/BasicMinimumParameters.h`.

Holds the current parameter vector, the per-parameter step size vector,
the function value, and validity/step-size flags. **Immutable wrapper**
— the `Vector{Float64}` fields are heap-allocated arrays shared by
reference across iterations (cf. ROADMAP §2.2 "MinimumState as an
immutable wrapper").

# Fields

- `x::Vector{Float64}` — current parameter values (internal coordinates
  in Phase 1+ with bounds; raw user values in Phase 0).
- `dirin::Vector{Float64}` — initial parameter step sizes (`x1 − x0`).
- `fval::Float64` — function value at `x`.
- `valid::Bool` — `false` if constructed without parameters (an
  invalid sentinel n-dimensional point); `true` otherwise.
- `has_step_size::Bool` — `true` only when `dirin` was supplied
  explicitly (matches C++ `fHasStep`).

# Constructors

- `MinimumParameters(n, fval = 0.0)` — invalid n-dim sentinel
  (matches `BasicMinimumParameters(n, fval)`).
- `MinimumParameters(x, fval)` — valid, no step size info.
- `MinimumParameters(x, dirin, fval)` — fully valid.
"""
struct MinimumParameters
    x::Vector{Float64}
    dirin::Vector{Float64}
    fval::Float64
    valid::Bool
    has_step_size::Bool
end

function MinimumParameters(n::Integer, fval::Real = 0.0)
    MinimumParameters(zeros(Float64, n), zeros(Float64, n), Float64(fval), false, false)
end

function MinimumParameters(x::Vector{Float64}, fval::Real)
    MinimumParameters(x, zeros(Float64, length(x)), Float64(fval), true, false)
end

function MinimumParameters(x::Vector{Float64}, dirin::Vector{Float64}, fval::Real)
    length(dirin) == length(x) ||
        throw(DimensionMismatch("dirin length $(length(dirin)) != x length $(length(x))"))
    MinimumParameters(x, dirin, Float64(fval), true, true)
end

Base.length(p::MinimumParameters) = length(p.x)
is_valid(p::MinimumParameters) = p.valid
has_step_size(p::MinimumParameters) = p.has_step_size

# ─────────────────────────────────────────────────────────────────────────────
# FunctionGradient — gradient + second-derivative estimate + step sizes.
# Mirrors reference/Minuit2_cpp/inc/Minuit2/BasicFunctionGradient.h.
# ─────────────────────────────────────────────────────────────────────────────

"""
    FunctionGradient

Mirror of `FunctionGradient` from
`reference/Minuit2_cpp/inc/Minuit2/BasicFunctionGradient.h`.

# Fields

- `grad::Vector{Float64}` — first derivatives at the current point.
- `g2::Vector{Float64}` — diagonal second derivatives (per-parameter).
- `gstep::Vector{Float64}` — step size used for the central-difference
  numerical gradient at each parameter (used by `Numerical2P` to adapt
  step from iteration to iteration).
- `analytical::Bool` — `true` if `grad` came from a user-supplied
  analytical gradient (Phase 1+); `false` for the Phase-0 numerical-
  gradient default.
- `valid::Bool` — `false` if this is an invalid sentinel (default
  for the n-dimensional zero constructor).
"""
struct FunctionGradient
    grad::Vector{Float64}
    g2::Vector{Float64}
    gstep::Vector{Float64}
    analytical::Bool
    valid::Bool
end

function FunctionGradient(n::Integer)
    z = zeros(Float64, n)
    FunctionGradient(z, copy(z), copy(z), false, false)
end

function FunctionGradient(grad::Vector{Float64}, g2::Vector{Float64},
                          gstep::Vector{Float64}; analytical::Bool = false)
    n = length(grad)
    (length(g2) == n && length(gstep) == n) ||
        throw(DimensionMismatch("FunctionGradient field lengths must agree: " *
                                "grad=$n, g2=$(length(g2)), gstep=$(length(gstep))"))
    FunctionGradient(grad, g2, gstep, analytical, true)
end

Base.length(g::FunctionGradient) = length(g.grad)
is_valid(g::FunctionGradient) = g.valid
is_analytical(g::FunctionGradient) = g.analytical

# ─────────────────────────────────────────────────────────────────────────────
# MinimumError — inverse Hessian + status flags.
# Mirrors reference/Minuit2_cpp/inc/Minuit2/MinimumError.h +
# BasicMinimumError.h. Includes 7 derived boolean predicates matching C++.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MinimumError

Mirror of `MinimumError` from
`reference/Minuit2_cpp/inc/Minuit2/MinimumError.h`.

Holds the **inverse Hessian** (used for parameter step `−V·g` and for
covariance updates via `DavidonErrorUpdator`), the `Dcovar` quality
indicator, and the status enum.

# Fields

- `inv_hessian::Symmetric{Float64,Matrix{Float64}}` — inverse Hessian
  (the covariance matrix is `2·inv_hessian` for χ²-like FCNs with
  `up=1`). Stored with `:U` (upper-triangle authoritative) per the
  Julia/BLAS convention; see ROADMAP §2.2 "Symmetric/syr! caveat".
- `dcovar::Float64` — relative change in covariance (DFP convergence
  estimator); 0 when fully converged, > 0 during MIGRAD iterations.
- `status::CovStatus` — see [`CovStatus`](@ref).

# Constructors

- `MinimumError(n)` — invalid n×n sentinel.
- `MinimumError(M, dcovar)` — valid Hesse error.
- `MinimumError(M, status)` — explicit failure status.
"""
struct MinimumError
    inv_hessian::Symmetric{Float64,Matrix{Float64}}
    dcovar::Float64
    status::CovStatus
    available::Bool   # mirrors fAvailable in BasicMinimumError
end

function MinimumError(n::Integer)
    MinimumError(Symmetric(zeros(Float64, n, n), :U), 0.0, MnHesseValid, false)
end

function MinimumError(M::AbstractMatrix{<:Real}, dcovar::Real)
    n = LinearAlgebra.checksquare(M)
    sm = M isa Symmetric ? convert(Symmetric{Float64,Matrix{Float64}}, M) :
                           Symmetric(Matrix{Float64}(M), :U)
    MinimumError(sm, Float64(dcovar), MnHesseValid, true)
end

function MinimumError(M::AbstractMatrix{<:Real}, status::CovStatus)
    n = LinearAlgebra.checksquare(M)
    sm = M isa Symmetric ? convert(Symmetric{Float64,Matrix{Float64}}, M) :
                           Symmetric(Matrix{Float64}(M), :U)
    # C++ BasicMinimumError tag constructors all set fDCovar = 1.0 (see
    # reference/Minuit2_cpp/inc/Minuit2/BasicMinimumError.h:55-75). This
    # propagates into the EDM correction `edm *= (1 + 3·dcov)` which makes
    # MIGRAD iterate harder after a MnPosDef event. v1 of state.jl
    # incorrectly set dcov=0.0 here (codex/Opus parallel-review #2 A1).
    MinimumError(sm, 1.0, status, true)
end

# Derived predicates — mirror C++ BasicMinimumError accessors line-for-line
# (reference/Minuit2_cpp/inc/Minuit2/BasicMinimumError.h:55-75 tag ctors
# determine the per-status (fValid, fPosDef) pairs; review #2 A1).
#
# Status   → fValid  fPosDef  fMadePosDef  fHesseFailed  fInvertFailed
# Valid    → true    true     false        false         false
# Hesse-F  → false   false    false        true          false
# MadePD   → true    false    true         false         false
# InvertF  → false   true     false        false         true
# NotPD    → false   false    false        false         false
is_valid(e::MinimumError) =
    e.available && (e.status == MnHesseValid || e.status == MnMadePosDef)
is_accurate(e::MinimumError) = e.dcovar < 0.1
is_pos_def(e::MinimumError) =
    e.available && (e.status == MnHesseValid || e.status == MnInvertFailed)
is_made_pos_def(e::MinimumError) = e.status == MnMadePosDef
hesse_failed(e::MinimumError) = e.status == MnHesseFailed
invert_failed(e::MinimumError) = e.status == MnInvertFailed
is_available(e::MinimumError) = e.available

Base.size(e::MinimumError) = size(e.inv_hessian)
Base.size(e::MinimumError, d::Integer) = size(e.inv_hessian, d)

# ─────────────────────────────────────────────────────────────────────────────
# MinimumState — composite snapshot at the end of one MIGRAD iteration.
# Mirrors reference/Minuit2_cpp/inc/Minuit2/MinimumState.h.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MinimumState

Mirror of `MinimumState` from
`reference/Minuit2_cpp/inc/Minuit2/MinimumState.h`.

Composite snapshot of one MIGRAD iteration: parameter values, error
matrix, gradient, expected distance to minimum, and cumulative FCN
call count.

Immutable wrapper; its `MinimumParameters`/`MinimumError`/
`FunctionGradient` fields are themselves immutable wrappers over
heap-allocated arrays, so rebuilding `MinimumState` each iteration
costs ~5 pointer copies — never bulk data copies (ROADMAP §2.2).

# Fields

- `parameters::MinimumParameters`
- `error::MinimumError`
- `gradient::FunctionGradient`
- `edm::Float64` — Expected Distance to Minimum (`0.5·gᵀ·V·g`).
- `nfcn::Int` — cumulative FCN call count up to this state.

# Constructors

- `MinimumState(n)` — invalid n-dim sentinel.
- `MinimumState(fval, edm, nfcn)` — only scalar fields (Simplex/Scan).
- `MinimumState(params, edm, nfcn)` — params, no gradient.
- `MinimumState(params, err, grad, edm, nfcn)` — full MIGRAD state.
"""
struct MinimumState
    parameters::MinimumParameters
    error::MinimumError
    gradient::FunctionGradient
    edm::Float64
    nfcn::Int
end

function MinimumState(n::Integer)
    MinimumState(MinimumParameters(n), MinimumError(n),
                 FunctionGradient(n), 0.0, 0)
end

function MinimumState(fval::Real, edm::Real, nfcn::Integer)
    p = MinimumParameters(Float64[], Float64[], Float64(fval), false, false)
    MinimumState(p, MinimumError(0), FunctionGradient(0), Float64(edm), Int(nfcn))
end

function MinimumState(parameters::MinimumParameters, edm::Real, nfcn::Integer)
    n = length(parameters)
    MinimumState(parameters, MinimumError(n), FunctionGradient(n),
                 Float64(edm), Int(nfcn))
end

function MinimumState(parameters::MinimumParameters, error::MinimumError,
                      gradient::FunctionGradient, edm::Real, nfcn::Integer)
    MinimumState(parameters, error, gradient, Float64(edm), Int(nfcn))
end

# Derived predicates — mirror C++ MinimumState accessors.
Base.length(s::MinimumState) = length(s.parameters)
fval(s::MinimumState) = s.parameters.fval
edm(s::MinimumState) = s.edm
nfcn(s::MinimumState) = s.nfcn
is_valid(s::MinimumState) =
    is_valid(s.parameters) && is_valid(s.error) && is_valid(s.gradient)
has_parameters(s::MinimumState) = is_valid(s.parameters)
has_covariance(s::MinimumState) = is_available(s.error)
