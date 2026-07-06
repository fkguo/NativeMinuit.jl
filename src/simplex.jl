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
  initial edge length is `10·Gstep[i]` per dim with the seed
  `Gstep[i] = max(gsmin, 0.1·|errs[i]|)`, i.e. an effective edge ≈ `|errs[i]|`
  (matches the C++ default `step = 10 · seed.Gradient().Gstep()`;
  `SimplexBuilder.cxx:38` + `InitialGradientCalculator.cxx:64`).

# Keyword arguments

- `maxfcn::Union{Integer,Nothing}=nothing` — FCN call budget; defaults
  to `200 + 100·n + 5·n²` (same as MIGRAD).
- `minedm::Union{Real,Nothing}=nothing` — convergence tolerance on
  `edm = f(jh) - f(jl)`. Defaults to `0.1 · up`, the C++/iminuit Simplex
  EDM goal: `ModularFunctionMinimizer::Minimize` scales
  `effective_toler = toler·Up()` with the canonical `toler = 0.1` and
  passes it to ALL builders (`ModularFunctionMinimizer.cxx:175`). The
  extra `×0.002` of `VariableMetricBuilder.cxx:66` is MIGRAD-only —
  Simplex does not apply it. Pass a smaller `minedm` explicitly for a
  tighter (non-C++-default) stopping rule.
- `prec::MachinePrecision` — floating-point precision (rarely tuned).
- `warn_nonfinite::Bool = true` — emit the single end-of-run warning
  when the FCN returned non-finite values AND the run did not end
  valid (P6, same policy as [`migrad`](@ref)). Inner probes (the
  `use_simplex` multistart inside `migrad!`) pass `false`.

# Returns

`FunctionMinimum` with `state.parameters.x` = best vertex, errors
derived from the simplex extent (no inverse Hessian — covariance is
`nothing`). Run a follow-up `hesse(cf, state)` if you need a covariance.

A non-finite final fval is never valid (P6): the result comes back
`is_valid = false` with the explicit `nonfinite_fval = true` reason, and
`n_nonfinite_calls` counts the FCN evaluations that returned NaN/±Inf
during this run. iminuit parity: `m.simplex()` on an all-NaN FCN reports
`fval=nan, valid=False`.

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
    warn_nonfinite::Bool = true,
)
    n = length(x0)
    length(errs) == n ||
        throw(DimensionMismatch("simplex: errs length $(length(errs)) != x0 length $n"))
    n > 0 || throw(ArgumentError("simplex needs at least one parameter"))

    # P6: baseline BEFORE the seed evaluation so a non-finite f(x0) at
    # call #1 is included in the run's non-finite tally (mirrors
    # `nonfinite_baseline` in `_migrad_loop`).
    nf_base = nonfinite_calls(cf)
    # Budget is RUN-LOCAL: C++ constructs a fresh `MnFcn` (the call
    # counter) per `MnApplication::operator()`, so `maxfcn` always means
    # "calls in THIS application". A reused `CostFunction` must not
    # inherit earlier runs' calls into this run's budget — snapshot the
    # baseline and compare deltas everywhere (pre-builder gate, loop
    # guard, post-`ybar` verdict, and the nfcn stored in the states).
    ncall_base = ncalls(cf)

    maxfcn_eff = _effective_maxfcn(maxfcn, n)
    # C++ Simplex EDM goal = effective_toler = toler·Up() with the canonical
    # toler = 0.1: ModularFunctionMinimizer::Minimize scales the tolerance by
    # Up() for ALL builders (ModularFunctionMinimizer.cxx:175). The extra
    # ×0.002 in VariableMetricBuilder.cxx:66 is MIGRAD-only — Simplex does NOT
    # apply it. ⇒ minedm = 0.1·up (audit §5). v1's 1e-5·up was ~10⁴× too tight,
    # making Simplex over-iterate and report `above_max_edm` far too readily.
    # C++ also floors effective_toler to eps2 (ModularFunctionMinimizer.cxx:178-179);
    # unreachable for any sane Up() (0.1·up ≫ eps2) but kept here for exactness.
    minedm_eff = minedm === nothing ? max(0.1 * cf.up, prec.eps2) : Float64(minedm)

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
        # C++ SimplexBuilder.cxx:38 — initial edge = 10·Gstep, where Gstep is
        # the seed InitialGradientCalculator value gstep = max(gsmin, 0.1·dirin)
        # (InitialGradientCalculator.cxx:64) with dirin ≈ |errs[i]| in the
        # no-limits case and gsmin = 8·eps2·(|x|+eps2). The effective edge is
        # therefore ≈ |errs[i]|, NOT 10·|errs[i]| (audit §5 — v1 was 10× too
        # large, which inflated the starting simplex and the call count).
        gsmin = 8.0 * prec.eps2 * (abs(x[i]) + prec.eps2)
        gstep = max(gsmin, 0.1 * abs(Float64(errs[i])))
        step[i] = 10.0 * gstep
    end

    # Seed (vertex 1) is the initial point.
    f_seed = cf(x)
    pts = Vector{Tuple{Float64,Vector{Float64}}}(undef, n + 1)
    pts[1] = (f_seed, copy(x))

    # C++ `ModularFunctionMinimizer::Minimize` (ModularFunctionMinimizer
    # .cxx:78-85) bails out BETWEEN seed generation and the builder when
    # the seed evaluation already exhausted the budget (`NumOfCalls() >=
    # maxfcn` ⇒ MnReachedCallLimit): iminuit's `m.simplex(ncall=1)`
    # returns after exactly the one seed call and never builds the
    # initial simplex. Observables mirrored from the C++ simplex seed
    # state: fval = f(x0); per-param dirin = the input steps; edm = n·up
    # — the InitialGradientCalculator seed has g2 = 2·up/dirin² and
    # V = diag(1/g2), so its EDM estimate Σ g2·dirin²/2 collapses to
    # n·up identically (iminuit 2.31.3 empirical: edm=2.0 at n=2, up=1).
    if ncalls(cf) - ncall_base >= maxfcn_eff
        dirin_bail = Float64[abs(Float64(errs[i])) for i in 1:n]
        par_bail = MinimumParameters(copy(x), dirin_bail, f_seed)
        err_bail = MinimumError(Symmetric(Matrix{Float64}(I, n, n), :U),
                                 1.0, MnHesseFailed, false)
        grad_bail = FunctionGradient(zeros(n), zeros(n), zeros(n))
        st_bail = MinimumState(par_bail, err_bail, grad_bail,
                                n * cf.up, ncalls(cf) - ncall_base)
        fm_bail = FunctionMinimum(st_bail, st_bail, cf.up;
                                   is_valid = false,
                                   reached_call_limit = true,
                                   above_max_edm = false,
                                   hesse_failed = false,
                                   made_pos_def = false,
                                   nonfinite_fval = !isfinite(f_seed),
                                   n_nonfinite_calls = nonfinite_calls(cf) - nf_base)
        warn_nonfinite && _warn_nonfinite_fcn(fm_bail; minimizer = "SIMPLEX")
        return fm_bail
    end

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

    # C++ `SimplexBuilder.cxx:99/196` is a do-while: the FIRST
    # Nelder-Mead round runs unconditionally, and `(Edm() > minedm ||
    # edmPrev > minedm) && NumOfCalls() < maxfcn` is evaluated only at
    # the BOTTOM of each round (a C++ `continue` jumps to that bottom
    # check, exactly like `continue` here jumps back to this top guard —
    # so exempting round 1 makes the two forms identical). Without the
    # exemption, a seed whose INITIAL simplex already satisfies
    # edm ≤ minedm (warm start) — or whose edm is NaN (all-non-finite
    # FCN; IEEE `>` is false) — skips the body entirely and drifts from
    # C++/iminuit by the 2-3 FCN calls of that mandatory first round
    # (iminuit 2.31.3: warm quadratic nfcn=6 vs 4, all-NaN nfcn=7 vs 4).
    first_round = true
    while true
        if !first_round
            if !(edm(sp) > minedm_eff || edm_prev > minedm_eff)
                break
            end
            if ncalls(cf) - ncall_base >= maxfcn_eff
                fcn_limit = true
                break
            end
        end
        first_round = false

        jl = sp.jl
        jh = sp.jh
        amin = sp.pts[jl][1]
        edm_prev = edm(sp)
        # C++ `niterations++` fires at the top of every ATTEMPTED round
        # (SimplexBuilder.cxx:118), not only on update paths.
        n_iter += 1

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
            continue
        end
        ρ > ρmax && (ρ = ρmax)

        @inbounds for k in 1:n
            prho[k] = ρ * pbar[k] + (1.0 - ρ) * sp.pts[jh][2][k]
        end
        yrho = cf(prho)

        if yrho < sp.pts[jl][1] && yrho < ystst
            update!(sp, yrho, prho)
            continue
        end
        if ystst < sp.pts[jl][1]
            update!(sp, ystst, pstst)
            continue
        end
        if yrho > sp.pts[jl][1]
            if ystst < sp.pts[jl][1]
                update!(sp, ystst, pstst)
            else
                update!(sp, ystar, pstar)
            end
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
    end

    # ── Final centroid + scaled errors ───────────────────────────────
    # NB: `above_max_edm` is evaluated AFTER the post-loop centroid swap
    # (line below) — matches `SimplexBuilder.cxx:235` where the EDM check
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
    # C++ `SimplexBuilder.cxx:229-237` decides the verdict AFTER the
    # final-centroid evaluation, budget first: `NumOfCalls() > maxfcn`
    # (strict — the `ybar` call above counts) ⇒ MnReachedCallLimit even
    # when the loop exited edm-converged; only otherwise can
    # `Edm() > minedm` ⇒ MnAboveMaxEdm. Recompute the flag here from the
    # post-`ybar` call count instead of trusting the in-loop exit reason
    # (the in-loop `fcn_limit` remains as loop-exit control only).
    fcn_limit = ncalls(cf) - ncall_base > maxfcn_eff
    above_max_edm = (edm_final > minedm_eff) && !fcn_limit
    # Per-param errors ≈ dirin · √(up/edm). The dirin is the simplex
    # extent; scaling by √(up/edm) projects onto the local-curvature
    # estimate. C++ `SimplexBuilder.cxx:218` applies it UNCONDITIONALLY:
    # edm = 0 (e.g. constant FCN) ⇒ errors +Inf; edm = NaN (all-non-
    # finite FCN) ⇒ errors NaN — both verified against iminuit 2.31.3
    # (errors [inf, inf] / [nan, nan] respectively). A `edm > 0` guard
    # here silently produced finite seed-scale errors on exactly those
    # degenerate paths. Julia-only wrinkle: `sqrt` THROWS on negative
    # arguments where C++ returns a quiet NaN, so map a negative ratio
    # (pathological Up or edm) to NaN explicitly.
    ratio = cf.up / edm_final
    scale = ratio < 0.0 ? NaN : sqrt(ratio)
    @inbounds for i in 1:n
        final_dirin[i] *= scale
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
                          ncalls(cf) - ncall_base)

    seed_state_obj = state   # simplex doesn't track a separate seed
    # P6: a non-finite incumbent fval can NEVER be a valid minimum — and
    # no C++-mirrored flag catches it here: with an all-NaN FCN `edm(sp)`
    # is NaN, so both loop guards (`NaN > minedm` = false) break
    # immediately AND `above_max_edm = (NaN > minedm) && …` stays false,
    # so the verdict came out `is_valid = true` with `fval = NaN` (same
    # hole as `_migrad_loop`'s verdict, handoff F7). iminuit 2.31
    # observable for the same FCN under `m.simplex()`: fval=nan,
    # valid=False (surfaced there via is_above_max_edm; NativeMinuit reports
    # the explicit `nonfinite_fval` reason, exactly like MIGRAD).
    nonfinite_final = !isfinite(ybar)
    is_valid = !fcn_limit && !above_max_edm && !nonfinite_final
    # `hesse_failed = false`: simplex never RAN Hesse, so calling it
    # "failed" is misleading. The `available = false` flag on
    # `state.error` is the authoritative "no covariance" indicator.
    # Review IMPORTANT #4.
    fm = FunctionMinimum(state, seed_state_obj, cf.up;
                          is_valid = is_valid,
                          reached_call_limit = fcn_limit,
                          above_max_edm = above_max_edm,
                          hesse_failed = false,
                          made_pos_def = false,
                          nonfinite_fval = nonfinite_final,
                          n_nonfinite_calls = nonfinite_calls(cf) - nf_base)
    warn_nonfinite && _warn_nonfinite_fcn(fm; minimizer = "SIMPLEX")
    return fm
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
    warn_nonfinite::Bool = true,
)
    n_active = n_free(params)
    n_active > 0 ||
        throw(ArgumentError("simplex needs at least one free parameter"))

    int_vals = initial_int_values(params)
    int_errs = initial_int_errors(params)
    cf_internal = _wrap_fcn_internal_to_external(cf, params)

    # P6: `warn_nonfinite` is forwarded — the inner low-level simplex owns
    # the single end-of-run warning (same split as bounded `migrad`);
    # `migrad!`'s `use_simplex` multistart probe passes `false`.
    fmin_int = simplex(cf_internal, int_vals, int_errs;
                        maxfcn = maxfcn, minedm = minedm, prec = prec,
                        warn_nonfinite = warn_nonfinite)

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
