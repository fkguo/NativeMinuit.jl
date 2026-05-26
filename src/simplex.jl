# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# simplex.jl — Nelder-Mead simplex (MnSimplex) port.
#
# Mirrors reference/Minuit2_cpp/src/SimplexBuilder.cxx and
# reference/Minuit2_cpp/src/SimplexParameters.cxx.
#
# Gradient-free minimizer. Useful when:
# - the FCN is non-smooth / discontinuous (where MIGRAD's gradient
#   estimation is fragile)
# - the FCN evaluation is cheap (simplex needs O(n²) evaluations per
#   iteration vs MIGRAD's O(2n) for the gradient, but each Simplex
#   evaluation can be 5-50× faster than a numerical-gradient step)
# - as a robust fallback when MIGRAD gets stuck or diverges
#
# Returns a `FunctionMinimum` so MINOS / contour / HESSE can run on
# the result.
# ─────────────────────────────────────────────────────────────────────────────

# Internal simplex container — mirrors C++ SimplexParameters.
#
# Stores n+1 vertices `(fval, x)` plus tracking indices `jh, jl`
# (highest / lowest fval). EDM = f(jh) - f(jl); Dirin per-dim is
# the range of x values across all vertices.
mutable struct SimplexParameters
    pts::Vector{Tuple{Float64,Vector{Float64}}}
    jh::Int      # 1-based index of highest fval
    jl::Int      # 1-based index of lowest fval
end

Base.getindex(sp::SimplexParameters, i::Integer) = sp.pts[i]
edm(sp::SimplexParameters) = sp.pts[sp.jh][1] - sp.pts[sp.jl][1]

function update!(sp::SimplexParameters, y::Float64, p::AbstractVector{Float64})
    # Replace the high vertex with (y, p), then recompute jh / jl.
    sp.pts[sp.jh] = (y, copy(p))
    if y < sp.pts[sp.jl][1]
        sp.jl = sp.jh
    end
    new_jh = 1
    @inbounds for i in 2:length(sp.pts)
        if sp.pts[i][1] > sp.pts[new_jh][1]
            new_jh = i
        end
    end
    sp.jh = new_jh
    return sp
end

function dirin(sp::SimplexParameters)
    # Per-dim range across all simplex vertices. Mirrors
    # SimplexParameters::Dirin() in SimplexParameters.cxx:33-49.
    n = length(sp.pts[1][2])
    d = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        pbig = sp.pts[1][2][i]
        plit = pbig
        for j in 2:length(sp.pts)
            v = sp.pts[j][2][i]
            v < plit && (plit = v)
            v > pbig && (pbig = v)
        end
        d[i] = pbig - plit
    end
    return d
end

# ─────────────────────────────────────────────────────────────────────────────
# Main simplex algorithm
# ─────────────────────────────────────────────────────────────────────────────

"""
    simplex(cf::CostFunction, x0, errs;
            maxfcn=nothing, minedm=nothing, prec=MachinePrecision()) -> FunctionMinimum

Nelder-Mead simplex minimizer. Gradient-free; calls `cf(x)` only.
Mirrors `SimplexBuilder::Minimum` in
`reference/Minuit2_cpp/src/SimplexBuilder.cxx`.

# Arguments

- `cf::CostFunction` — user FCN wrapper. `cf.up` is the ErrorDef
  (1.0 for χ², 0.5 for NLL) used to scale the final per-parameter
  errors.
- `x0::AbstractVector{<:Real}` — initial parameter values.
- `errs::AbstractVector{<:Real}` — initial 1σ step sizes. The simplex
  initial edge length is `10·errs[i]` per dim (matches the C++ default
  `step = 10 · seed.Gradient().Gstep()`).

# Keyword arguments

- `maxfcn::Union{Integer,Nothing}=nothing` — FCN call budget; defaults
  to `200 + 100·n + 5·n²` (same as MIGRAD).
- `minedm::Union{Real,Nothing}=nothing` — convergence tolerance on
  `edm = f(jh) - f(jl)`. Defaults to `1e-5 · up` (matches C++
  `SimplexBuilder` literal `0.1·tol·up·1e-3` with the canonical
  `tol = 0.1`). The 2004 CERN MINUIT user guide literal reading is
  `tol·up` (≈ 10⁴× looser); JuMinuit follows the C++ Minuit2 default
  instead, consistent with iminuit / IMinuit.jl behavior. Pass
  `minedm = tol·cf.up` explicitly if you want the strict manual
  semantics.
- `prec::MachinePrecision` — floating-point precision (rarely tuned).

# Returns

`FunctionMinimum` with `state.parameters.x` = best vertex, errors
derived from the simplex extent (no inverse Hessian — covariance is
`nothing`). Run a follow-up `hesse(cf, state)` if you need a covariance.

# Notes

- This is a direct port of the C++ Nelder-Mead recipe (reflect /
  expand / contract / shrink). The "rho" adaptive step is the L162-176
  block — a parabolic fit through `(y_h, y_star, y_stst)` choosing the
  next probe location.
- Without a gradient, EDM is not the variable-metric estimator (`gᵀV g`)
  but the simpler `f_high - f_low` over the simplex.
"""
function simplex(
    cf::CostFunction,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real};
    maxfcn::Union{Integer,Nothing} = nothing,
    minedm::Union{Real,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(x0)
    length(errs) == n ||
        throw(DimensionMismatch("simplex: errs length $(length(errs)) != x0 length $n"))
    n > 0 || throw(ArgumentError("simplex needs at least one parameter"))

    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)
    # C++ default: 0.1 * tol * up * 1e-3 with tol=0.1 → 1e-5·up.
    minedm_eff = minedm === nothing ? 1.0e-5 * cf.up : Float64(minedm)

    # Nelder-Mead coefficients — exact match to C++ defaults
    α = 1.0
    β = 0.5
    γ = 2.0
    ρmin = 4.0
    ρmax = 8.0
    ρ1 = 1.0 + α
    ρ2 = 1.0 + α * γ   # David Sachs (FNAL) change vs original (ρ2 = ρ1 + αγ)

    # ── Initial simplex ───────────────────────────────────────────────
    x = collect(Float64, x0)
    step = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        step[i] = 10.0 * abs(Float64(errs[i]))
    end

    # Seed (vertex 1) is the initial point.
    f_seed = cf(x)
    pts = Vector{Tuple{Float64,Vector{Float64}}}(undef, n + 1)
    pts[1] = (f_seed, copy(x))

    jl = 1
    jh = 1
    amin = f_seed
    aming = f_seed

    @inbounds for i in 1:n
        # C++ line 56: step[i] = max(step[i], 8·eps²·(|x_i| + eps²)).
        dmin = 8.0 * prec.eps2 * (abs(x[i]) + prec.eps2)
        step[i] < dmin && (step[i] = dmin)
        x[i] += step[i]
        tmp = cf(x)
        if tmp < amin
            amin = tmp
            jl = i + 1
        end
        if tmp > aming
            aming = tmp
            jh = i + 1
        end
        pts[i + 1] = (tmp, copy(x))
        x[i] -= step[i]
    end
    sp = SimplexParameters(pts, jh, jl)

    # ── Iterate ───────────────────────────────────────────────────────
    pbar = Vector{Float64}(undef, n)
    pstar = Vector{Float64}(undef, n)
    pstst = Vector{Float64}(undef, n)
    prho = Vector{Float64}(undef, n)
    wg = 1.0 / Float64(n)
    edm_prev = edm(sp)
    n_iter = 0

    fcn_limit = false
    above_max_edm = false

    while true
        if !(edm(sp) > minedm_eff || edm_prev > minedm_eff)
            break
        end
        if ncalls(cf) >= maxfcn_eff
            fcn_limit = true
            break
        end

        jl = sp.jl
        jh = sp.jh
        amin = sp.pts[jl][1]
        edm_prev = edm(sp)

        # pbar = centroid of all vertices EXCEPT jh
        fill!(pbar, 0.0)
        @inbounds for i in 1:(n + 1)
            i == jh && continue
            for k in 1:n
                pbar[k] += wg * sp.pts[i][2][k]
            end
        end

        # Reflection: pstar = (1+α)·pbar - α·x_jh
        @inbounds for k in 1:n
            pstar[k] = (1.0 + α) * pbar[k] - α * sp.pts[jh][2][k]
        end
        ystar = cf(pstar)

        if ystar > amin
            # Reflection didn't improve the best.
            if ystar < sp.pts[jh][1]
                update!(sp, ystar, pstar)
                if jh != sp.jh
                    n_iter += 1
                    continue
                end
            end
            # Contraction: pstst = β·x_jh + (1-β)·pbar
            @inbounds for k in 1:n
                pstst[k] = β * sp.pts[jh][2][k] + (1.0 - β) * pbar[k]
            end
            ystst = cf(pstst)
            if ystst > sp.pts[jh][1]
                break  # contraction failed — simplex collapsed
            end
            update!(sp, ystst, pstst)
            n_iter += 1
            continue
        end

        # Reflection improved best — try expansion.
        @inbounds for k in 1:n
            pstst[k] = γ * pstar[k] + (1.0 - γ) * pbar[k]
        end
        ystst = cf(pstst)

        # Adaptive step ρ from quadratic fit through y_jh, y_star, y_stst
        y1 = (ystar - sp.pts[jh][1]) * ρ2
        y2 = (ystst - sp.pts[jh][1]) * ρ1
        ρ = 0.5 * (ρ2 * y1 - ρ1 * y2) / (y1 - y2)
        if ρ < ρmin
            if ystst < sp.pts[jl][1]
                update!(sp, ystst, pstst)
            else
                update!(sp, ystar, pstar)
            end
            n_iter += 1
            continue
        end
        ρ > ρmax && (ρ = ρmax)

        @inbounds for k in 1:n
            prho[k] = ρ * pbar[k] + (1.0 - ρ) * sp.pts[jh][2][k]
        end
        yrho = cf(prho)

        if yrho < sp.pts[jl][1] && yrho < ystst
            update!(sp, yrho, prho)
            n_iter += 1
            continue
        end
        if ystst < sp.pts[jl][1]
            update!(sp, ystst, pstst)
            n_iter += 1
            continue
        end
        if yrho > sp.pts[jl][1]
            if ystst < sp.pts[jl][1]
                update!(sp, ystst, pstst)
            else
                update!(sp, ystar, pstar)
            end
            n_iter += 1
            continue
        end
        if ystar > sp.pts[jh][1]
            @inbounds for k in 1:n
                pstst[k] = β * sp.pts[jh][2][k] + (1.0 - β) * pbar[k]
            end
            ystst = cf(pstst)
            if ystst > sp.pts[jh][1]
                break  # contraction failed
            end
            update!(sp, ystst, pstst)
        end
        n_iter += 1
    end

    # ── Final centroid + scaled errors ───────────────────────────────
    # NB: `above_max_edm` is evaluated AFTER the post-loop centroid swap
    # (line below) — matches `SimplexBuilder.cxx:217` where the EDM check
    # uses the final-state simplex. v1 snapshotted it BEFORE the swap,
    # which could mismatch when the final centroid lowers `jl` enough
    # to flip the edm-vs-minedm threshold (review IMPORTANT #6).
    jl = sp.jl
    jh = sp.jh
    amin = sp.pts[jl][1]

    fill!(pbar, 0.0)
    @inbounds for i in 1:(n + 1)
        i == jh && continue
        for k in 1:n
            pbar[k] += wg * sp.pts[i][2][k]
        end
    end
    ybar = cf(pbar)
    if ybar < amin
        update!(sp, ybar, pbar)
        # If pbar is the new low, ybar/pbar become the result; else
        # we fall through to use jl (the existing low).
    else
        copyto!(pbar, sp.pts[jl][2])
        ybar = sp.pts[jl][1]
    end

    final_dirin = dirin(sp)
    edm_final = edm(sp)
    above_max_edm = (edm_final > minedm_eff) && !fcn_limit
    # Per-param errors ≈ dirin · √(up/edm). The dirin is the simplex
    # extent; scaling by √(up/edm) projects onto the local-curvature
    # estimate. Mirrors SimplexBuilder.cxx:200-201.
    if edm_final > 0.0
        scale = sqrt(cf.up / edm_final)
        @inbounds for i in 1:n
            final_dirin[i] *= scale
        end
    end

    # ── Build FunctionMinimum from the final state ───────────────────
    par_state = MinimumParameters(copy(pbar), final_dirin, ybar)
    # Simplex does NOT compute an inverse Hessian. Mark the
    # MinimumError as `available = false` so downstream `has_covariance`
    # returns false and `_internal_to_external_results` / `m.matrix` /
    # `eigenvalues(m)` / `global_cc(m)` all correctly return `nothing`.
    # An identity placeholder would leak through as a fake covariance —
    # review BLOCKING #1.
    err_state = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U),
                              1.0, MnHesseFailed, false)
    grad_state = FunctionGradient(zeros(n), zeros(n), zeros(n))
    state = MinimumState(par_state, err_state, grad_state, edm_final,
                          ncalls(cf))

    seed_state_obj = state   # simplex doesn't track a separate seed
    is_valid = !fcn_limit && !above_max_edm
    # `hesse_failed = false`: simplex never RAN Hesse, so calling it
    # "failed" is misleading. The `available = false` flag on
    # `state.error` is the authoritative "no covariance" indicator.
    # Review IMPORTANT #4.
    return FunctionMinimum(state, seed_state_obj, cf.up;
                            is_valid = is_valid,
                            reached_call_limit = fcn_limit,
                            above_max_edm = above_max_edm,
                            hesse_failed = false,
                            made_pos_def = false)
end

# Bare-function convenience overload — wraps `f` in a CostFunction
# (mirrors `migrad(f, x0, errs; up=1.0)` ergonomics).
function simplex(
    f::F,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real};
    up::Real = 1.0,
    kwargs...,
) where {F}
    cf = CostFunction(f, up)
    return simplex(cf, x0, errs; kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bound-aware simplex — mirrors `migrad(cf, params)` shape.
# ─────────────────────────────────────────────────────────────────────────────

"""
    simplex(cf::CostFunction, params::Parameters; ...) -> BoundedFunctionMinimum

Bound-aware Nelder-Mead. User FCN receives external coordinates; the
inner simplex runs in internal (transformed) coordinates so bounds
and fixed parameters are respected.

Like its unbounded sibling, this overload **does not** produce a
covariance matrix (simplex has no inverse Hessian). `m.matrix`,
`eigenvalues(m)`, `global_cc(m)` all return `nothing` after a
simplex-only fit — run [`hesse`](@ref) on top if you need a real cov.
External per-parameter errors are derived from the simplex dirin
(simplex extent × √(up/edm)) via the int→ext Jacobian.
"""
function simplex(
    cf::CostFunction,
    params::Parameters;
    maxfcn::Union{Integer,Nothing} = nothing,
    minedm::Union{Real,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
)
    n_active = n_free(params)
    n_active > 0 ||
        throw(ArgumentError("simplex needs at least one free parameter"))

    int_vals = initial_int_values(params)
    int_errs = initial_int_errors(params)
    cf_internal = _wrap_fcn_internal_to_external(cf, params)

    fmin_int = simplex(cf_internal, int_vals, int_errs;
                        maxfcn = maxfcn, minedm = minedm, prec = prec)

    # Build ext_values + ext_errors directly via the int→ext Jacobian.
    # We do NOT call `_internal_to_external_results` because:
    #   (a) the simplex's MinimumError has `available=false`, so the
    #       covariance branch would skip and ext_errors would fall back
    #       to `par.error` (the user's initial step) — useless;
    #   (b) we WANT the dirin-derived errors here (real information
    #       extracted by the simplex), just not via a fake inverse
    #       Hessian. Review BLOCKING #1.
    n_total = n_pars(params)
    ext_values     = Vector{Float64}(undef, n_total)
    ext_errors_vec = zeros(Float64, n_total)
    int_x      = fmin_int.state.parameters.x
    int_dirin  = fmin_int.state.parameters.dirin
    @inbounds for ext_idx in 1:n_total
        par = params.pars[ext_idx]
        int_idx = params.int_of_ext[ext_idx]
        if int_idx == 0
            ext_values[ext_idx]     = par.value
            ext_errors_vec[ext_idx] = 0.0
        else
            ext_values[ext_idx] = int_to_ext_value(params, int_idx,
                                                    int_x[int_idx])
            # For bounded parameters: use the same C++ two-sided
            # Int2extError formula as `_internal_to_external_results`
            # so simplex and MIGRAD report consistent ext_errors near
            # the boundary (review IMPORTANT #1 round-2). For
            # unbounded parameters the formula collapses to
            # `|jac| * int_dirin` (jac=1) — identical to the
            # first-order Taylor approximation.
            if has_limits(par) || has_upper_limit(par) || has_lower_limit(par)
                kind = bound_kind(par.lower, par.upper)
                ext_errors_vec[ext_idx] = int2ext_error(
                    kind, int_x[int_idx], int_dirin[int_idx],
                    par.lower, par.upper)
            else
                ext_errors_vec[ext_idx] = int_dirin[int_idx]
            end
        end
    end

    return BoundedFunctionMinimum(
        fmin_int, params, ext_values, ext_errors_vec, nothing,
        cf_internal,
    )
end
