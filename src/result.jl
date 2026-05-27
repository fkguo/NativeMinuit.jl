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

M6 (GAP_AUDIT) — per-iteration history:

- `states::Vector{MinimumState}` — per-iteration snapshots when MIGRAD
  was invoked with `storage_level >= 1`. Empty (`MinimumState[]`) by
  default (`storage_level = 0`) — keeps the zero-alloc gate happy.
- `storage_level::Int` — `0` (default) for no history, `1` for
  per-iteration snapshots. Mirrors C++
  `BasicFunctionMinimum.h:109,165`.
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
    # M6: per-iteration MIGRAD history, populated when `_migrad_loop`
    # was invoked with `storage_level >= 1`. Each entry is a
    # deep-copied snapshot of `s0` at the end of one DFP iteration
    # (snapshot needed because the loop's ping-pong buffers would
    # otherwise mutate the entries on subsequent iterations). The seed
    # is NOT prepended here — callers can read `seed` separately.
    states::Vector{MinimumState}
    storage_level::Int
end

function FunctionMinimum(state::MinimumState, seed::MinimumState, up::Real;
                          is_valid::Bool = true,
                          reached_call_limit::Bool = false,
                          above_max_edm::Bool = false,
                          hesse_failed::Bool = false,
                          made_pos_def::Bool = false,
                          states::Vector{MinimumState} = MinimumState[],
                          storage_level::Integer = 0)
    FunctionMinimum(state, seed, Float64(up),
                    is_valid, reached_call_limit, above_max_edm,
                    hesse_failed, made_pos_def,
                    states, Int(storage_level))
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
    # iminuit-style two-column box (Phase 3 parity). The table is 71 chars
    # wide (35 + 1 separator + 35) — matches iminuit's default repr layout.
    _show_minimum_box(io, m)
    println(io)
    _show_parameter_box(io, m)
end

# Internal: 71-char two-column box with title and status flags
function _show_minimum_box(io::IO, m::FunctionMinimum)
    fval = m.state.parameters.fval
    edm = m.state.edm
    nfcn = m.state.nfcn

    # Top border + title
    println(io, "┌", "─"^71, "┐")
    println(io, "│", _center("Migrad", 71), "│")
    println(io, "├", "─"^35, "┬", "─"^35, "┤")

    # Row 1: FCN | Nfcn
    lhs = " FCN = $(_fmt_num(fval))"
    rhs = _center("Nfcn = $nfcn", 35)
    println(io, "│", _ljust(lhs, 35), "│", rhs, "│")

    # Row 2: EDM | (blank or strategy hint)
    edm_str = " EDM = $(_fmt_num(edm)) (Goal: $(_fmt_num(2e-3 * m.up)))"
    println(io, "│", _ljust(edm_str, 35), "│", " "^35, "│")

    # Status row 1
    println(io, "├", "─"^35, "┼", "─"^35, "┤")
    valid_str = m.is_valid ? "Valid Minimum" : "INVALID Minimum"
    edm_status = m.above_max_edm ? "EDM ABOVE threshold (x 10)" :
                                     "Below EDM threshold (goal x 10)"
    println(io, "│", _center(valid_str, 35), "│", _center(edm_status, 35), "│")

    # Status row 2: parameter limits & call limit
    println(io, "├", "─"^35, "┼", "─"^35, "┤")
    # FunctionMinimum doesn't track parameter limits (bounded variant has);
    # report a placeholder. BoundedFunctionMinimum overrides show separately.
    lim_str = "No parameters at limit"
    nfcn_str = m.reached_call_limit ? "AT call limit" : "Below call limit"
    println(io, "│", _center(lim_str, 35), "│", _center(nfcn_str, 35), "│")

    # Status row 3: hesse + covariance
    println(io, "├", "─"^35, "┼", "─"^35, "┤")
    hesse_str = m.hesse_failed ? "Hesse FAILED" : "Hesse OK"
    cov_str = has_covariance(m) ?
        (m.made_pos_def ? "Covariance forced pos-def" : "Covariance accurate") :
        "No covariance"
    println(io, "│", _center(hesse_str, 35), "│", _center(cov_str, 35), "│")

    println(io, "└", "─"^35, "┴", "─"^35, "┘")
end

# Internal: parameter table
function _show_parameter_box(io::IO, m::FunctionMinimum)
    x = m.state.parameters.x
    n = length(x)
    # Column widths: idx(3) name(8) value(15) hesse(15) total = 41 + 3 separators = 44
    println(io, "┌", "─"^3, "┬", "─"^8, "┬", "─"^15, "┬", "─"^15, "┐")
    println(io, "│", _center("", 3), "│", _center("Name", 8), "│",
                 _center("Value", 15), "│", _center("Hesse Err", 15), "│")
    println(io, "├", "─"^3, "┼", "─"^8, "┼", "─"^15, "┼", "─"^15, "┤")
    has_cov = has_covariance(m)
    for i in 1:n
        idx = _ljust(" $i", 3)
        name = _ljust(" x$(i-1)", 8)
        val_s = _center(_fmt_num(x[i]), 15)
        err_s = has_cov ?
            _center(_fmt_num(sqrt(max(2 * m.up * m.state.error.inv_hessian[i, i], 0.0))), 15) :
            _center("—", 15)
        println(io, "│", idx, "│", name, "│", val_s, "│", err_s, "│")
    end
    println(io, "└", "─"^3, "┴", "─"^8, "┴", "─"^15, "┴", "─"^15, "┘")
end

Base.show(io::IO, m::FunctionMinimum) =
    print(io, "FunctionMinimum(fval=", m.state.parameters.fval,
              ", edm=", m.state.edm,
              ", nfcn=", m.state.nfcn,
              ", valid=", m.is_valid, ")")

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers (iminuit-style boxes use these — see `_show_minimum_box`,
# `_show_parameter_box`, and the bounded/MinosError variants in their own
# files). String widths use byte counts not visual width — we use only
# ASCII fillers internally so width math is safe.
# ─────────────────────────────────────────────────────────────────────────────

@inline function _ljust(s::AbstractString, w::Int)
    n = length(s)
    n >= w ? s : s * " "^(w - n)
end

@inline function _center(s::AbstractString, w::Int)
    n = length(s)
    n >= w && return s
    lpad = (w - n) ÷ 2
    rpad = w - n - lpad
    return " "^lpad * s * " "^rpad
end

# Number formatter: 3-significant-digit scientific for tiny/huge, fixed for moderate
function _fmt_num(x::Real)
    isnan(x) && return "NaN"
    isinf(x) && return string(x)
    ax = abs(x)
    if ax == 0.0
        return "0"
    elseif ax < 1e-3 || ax >= 1e5
        # scientific
        return @sprintf("%.3e", x)
    else
        return @sprintf("%.4g", x)
    end
end
