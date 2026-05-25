# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# result.jl — FunctionMinimum (Phase 0 form).
#
# Mirrors reference/Minuit2_cpp/inc/Minuit2/FunctionMinimum.h.
#
# Phase 0 stores only the seed + final state (storage_level=0; ROADMAP
# DR-009). Phase 1 will add an optional state-history vector when
# storage_level == 1 is enabled.
# ─────────────────────────────────────────────────────────────────────────────

"""
    FunctionMinimum

The result of a MIGRAD minimization. Phase 0 contract:

- `state` — final `MinimumState` (parameters, error matrix, gradient,
  EDM, NFcn).
- `seed` — the initial state from `MnSeedGenerator`, kept for
  reference (e.g. to query starting parameters or initial EDM).
- `up` — ErrorDef (1.0 for χ², 0.5 for NLL); mirrors the user's
  `CostFunction.up`.

Status flags (mirror C++ `FunctionMinimum`):

- `is_valid` — overall convergence success.
- `reached_call_limit` — `nfcn ≥ maxfcn` was hit.
- `above_max_edm` — final EDM is more than 10× the requested tolerance.
- `hesse_failed` — Hesse refinement (Phase 1) failed.
- `made_pos_def` — MnPosDef perturbed the error matrix.
"""
struct FunctionMinimum
    state::MinimumState
    seed::MinimumState
    up::Float64
    is_valid::Bool
    reached_call_limit::Bool
    above_max_edm::Bool
    hesse_failed::Bool
    made_pos_def::Bool
end

function FunctionMinimum(state::MinimumState, seed::MinimumState, up::Real;
                          is_valid::Bool = true,
                          reached_call_limit::Bool = false,
                          above_max_edm::Bool = false,
                          hesse_failed::Bool = false,
                          made_pos_def::Bool = false)
    FunctionMinimum(state, seed, Float64(up),
                    is_valid, reached_call_limit, above_max_edm,
                    hesse_failed, made_pos_def)
end

# ─────────────────────────────────────────────────────────────────────────────
# Accessors mirroring C++ FunctionMinimum API
# ─────────────────────────────────────────────────────────────────────────────

fval(m::FunctionMinimum) = m.state.parameters.fval
edm(m::FunctionMinimum) = m.state.edm
nfcn(m::FunctionMinimum) = m.state.nfcn
parameters(m::FunctionMinimum) = m.state.parameters
errors(m::FunctionMinimum) = m.state.error
gradient(m::FunctionMinimum) = m.state.gradient
# Overload Base.values rather than introducing a clashing JuMinuit-local
# `values` symbol. Phase 3 will add `m.values` property access via
# getproperty for iminuit copy-paste compatibility.
Base.values(m::FunctionMinimum) = m.state.parameters.x
is_valid(m::FunctionMinimum) = m.is_valid && is_valid(m.state)
has_covariance(m::FunctionMinimum) = has_covariance(m.state)
reached_call_limit(m::FunctionMinimum) = m.reached_call_limit
above_max_edm(m::FunctionMinimum) = m.above_max_edm
hesse_failed(m::FunctionMinimum) = m.hesse_failed
made_pos_def(m::FunctionMinimum) = m.made_pos_def

"""
    covariance(m::FunctionMinimum) -> Symmetric{Float64,Matrix{Float64}}

The covariance matrix `2·up·V` where `V = inv(H)` is the inverse Hessian
and `up = ErrorDef`. For χ² fits with `up=1`, this is `2·V`.

Returns `nothing` if the minimum doesn't have a valid covariance
(`!has_covariance(m)`).

The matrix is constructed lazily — each call builds a fresh matrix.
"""
function covariance(m::FunctionMinimum)
    has_covariance(m) || return nothing
    M = parent(m.state.error.inv_hessian)
    n = size(M, 1)
    cov = Matrix{Float64}(undef, n, n)
    factor = 2.0 * m.up
    @inbounds for j in 1:n, i in 1:n
        # Read symmetrically via Symmetric view, then scale
        cov[i, j] = factor * m.state.error.inv_hessian[i, j]
    end
    return Symmetric(cov, :U)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing (matches iminuit/IMinuit.jl conventions roughly)
# ─────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", m::FunctionMinimum)
    println(io, "JuMinuit FunctionMinimum")
    println(io, "  valid:              ", m.is_valid)
    println(io, "  fval:               ", m.state.parameters.fval)
    println(io, "  edm:                ", m.state.edm)
    println(io, "  nfcn:               ", m.state.nfcn)
    println(io, "  reached_call_limit: ", m.reached_call_limit)
    println(io, "  above_max_edm:      ", m.above_max_edm)
    println(io, "  has_covariance:     ", has_covariance(m))
    println(io, "  parameters:")
    x = m.state.parameters.x
    n = length(x)
    if has_covariance(m)
        for i in 1:n
            sig = sqrt(2 * m.up * m.state.error.inv_hessian[i, i])
            println(io, "    [", i, "] ", x[i], " ± ", sig)
        end
    else
        for i in 1:n
            println(io, "    [", i, "] ", x[i])
        end
    end
end

Base.show(io::IO, m::FunctionMinimum) =
    print(io, "FunctionMinimum(fval=", m.state.parameters.fval,
              ", edm=", m.state.edm,
              ", nfcn=", m.state.nfcn,
              ", valid=", m.is_valid, ")")
