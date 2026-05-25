# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# migrad.jl — the MIGRAD loop.
#
# Mirrors reference/Minuit2_cpp/src/VariableMetricBuilder.cxx.
# C++ has two `Minimum(...)` methods:
#   - Outer (lines 54-203) — handles maxfcn=80% retry trick and Strategy
#     ≥ 1 MnHesse refinement.
#   - Inner (lines 205-375) — the actual iteration loop.
# Phase 0 lock (DR-008) is Strategy(0): the outer's retry-with-Hesse
# branch is dead code, so the Julia version collapses outer and inner
# into a single loop. Phase 1 will split.
#
# Algorithm sketch per iteration:
#   1. step = -V·g                      (sym_mul!)
#   2. gdel = step·g; if gdel > 0, MnPosDef + recompute
#   3. line search along step           (line_search)
#   4. accept new point if improving
#   5. compute new gradient             (numerical_gradient!)
#   6. EDM with OLD error               (estimate_edm)
#   7. MnPosDef if EDM < 0
#   8. DFP update                        (davidon_update!)
#   9. build new state; corrected edm = edm·(1 + 3·dcovar)
#  10. converge when edm ≤ tol·0.002 (C++ VariableMetricBuilder.cxx:66)
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad(cf, x0, errs;
           strategy=Strategy(0),
           tol=0.1,
           maxfcn=200 + 100·n + 5·n²,
           prec=MachinePrecision()) -> FunctionMinimum

Native-Julia MIGRAD minimization of the user FCN `cf` starting from
`x0` with per-parameter step sizes `errs`. Returns a
[`FunctionMinimum`](@ref).

Phase 0 — see ROADMAP §3:
- Unconstrained (no bounds, no fixed parameters).
- Numerical gradient only (no analytical FCN gradient yet).
- Strategy(0) only (Strategy ≥ 1 needs Phase 1's `MnHesse`).
- Default `maxfcn` matches C++ `MnApplication.cxx:43`.
- Convergence: stop when `edm ≤ tol · 0.002` (`VariableMetricBuilder.cxx:66`),
  or when `nfcn ≥ maxfcn`.

# Arguments

- `cf::CostFunction` — user FCN wrapper (auto-counts calls).
- `x0::AbstractVector{<:Real}` — initial parameter values.
- `errs::AbstractVector{<:Real}` — initial step sizes (≥ 0; algorithm
  uses `|werr|` defensively).

# Keyword arguments

- `strategy::Strategy=Strategy(0)` — Phase 0 only supports level 0.
- `tol::Real=0.1` — convergence tolerance on EDM (after `*0.002` factor).
- `maxfcn::Integer=200+100·n+5·n²` — call-count limit.
- `prec::MachinePrecision=MachinePrecision()` — floating-point precision.

# Returns

`FunctionMinimum` with:
- `is_valid=true` if MIGRAD converged within tol AND nfcn.
- `reached_call_limit=true` if `nfcn ≥ maxfcn` before convergence.
- `above_max_edm=true` if final EDM > 10·(tol·0.002).
- `made_pos_def=true` if MnPosDef perturbed the matrix at any point.
"""
function migrad(
    cf::CostFunction,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real};
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(x0)
    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)

    seed = seed_state(cf, x0, errs, strategy, prec)
    return _migrad_loop(seed, cf, strategy, Float64(tol), maxfcn_eff, prec)
end

"""
    _migrad_loop(seed, cf, strategy, tol, maxfcn, prec) -> FunctionMinimum

The MIGRAD iteration loop proper. Public via `migrad(...)`; exposed for
re-entry tests and benchmarks.
"""
function _migrad_loop(
    seed::MinimumState,
    cf::CostFunction,
    strategy::Strategy,
    tol::Real,
    maxfcn::Integer,
    prec::MachinePrecision,
)
    n = length(seed)

    # Tolerance multiplier matches C++ VariableMetricBuilder.cxx:66 exactly.
    edmval = Float64(tol) * 0.002

    if n == 0
        return FunctionMinimum(seed, seed, cf.up; is_valid = false)
    end
    if !is_valid(seed)
        return FunctionMinimum(seed, seed, cf.up; is_valid = false)
    end
    if seed.edm < 0
        return FunctionMinimum(seed, seed, cf.up; is_valid = false)
    end

    s0 = seed
    # Initial-state EDM correction (C++ line 229)
    edm_corrected = s0.edm * (1.0 + 3.0 * s0.error.dcovar)

    # Per-iteration scratch (reused across the loop)
    step      = zeros(Float64, n)
    ls_work   = Vector{Float64}(undef, n)
    grad_work = Vector{Float64}(undef, n)
    vg_work   = Vector{Float64}(undef, n)
    vUpd_work = Symmetric(zeros(Float64, n, n), :U)

    made_pos_def_flag = false

    while edm_corrected > edmval && ncalls(cf) < maxfcn
        # ── Step 1: step = -V·g
        sym_mul!(step, s0.error.inv_hessian, s0.gradient.grad, -1.0, 0.0)

        # ── Check zero gradient (C++ line 247-250)
        if dot(s0.gradient.grad, s0.gradient.grad) <= 0
            break
        end

        gdel = dot(step, s0.gradient.grad)

        # ── Step 2: if gdel > 0, matrix not pos-def — try MnPosDef
        if gdel > 0
            s0 = make_posdef(s0, prec)
            made_pos_def_flag = true
            sym_mul!(step, s0.error.inv_hessian, s0.gradient.grad, -1.0, 0.0)
            gdel = dot(step, s0.gradient.grad)
            if gdel > 0
                break  # still bad — bail
            end
        end

        # ── Step 3: line search
        pp = line_search(cf, s0.parameters, step, gdel, prec; work_x = ls_work)

        # ── Step 4: no-improvement check (C++ line 278)
        if abs(pp.y - s0.parameters.fval) <= abs(s0.parameters.fval) * prec.eps
            # Accept latest fval but keep error/gradient unchanged
            s0 = MinimumState(s0.parameters, s0.error, s0.gradient,
                              s0.edm, ncalls(cf))
            break
        end

        # ── Step 5: accept new point, compute new gradient
        new_x = Vector{Float64}(undef, n)
        @inbounds @. new_x = s0.parameters.x + pp.x * step
        new_par = MinimumParameters(new_x, pp.y)
        new_grad = FunctionGradient(zeros(Float64, n), zeros(Float64, n), zeros(Float64, n))
        numerical_gradient!(new_grad, grad_work, new_par, s0.gradient,
                             cf, strategy, prec)

        # ── Step 6: EDM using OLD error matrix (C++ line 300)
        new_edm = estimate_edm(new_grad, s0.error)

        if isnan(new_edm)
            break
        end

        # ── Step 7: if EDM < 0, try MnPosDef on s0's error
        if new_edm < 0
            s0 = make_posdef(s0, prec)
            made_pos_def_flag = true
            new_edm = estimate_edm(new_grad, s0.error)
            if new_edm < 0
                break
            end
        end

        # ── Step 8: DFP update
        dx = Vector{Float64}(undef, n)
        dg = Vector{Float64}(undef, n)
        @inbounds @. dx = new_par.x - s0.parameters.x
        @inbounds @. dg = new_grad.grad - s0.gradient.grad

        new_V = Symmetric(copy(parent(s0.error.inv_hessian)), :U)
        new_dcov, _ = davidon_update!(new_V, dx, dg, s0.error.dcovar,
                                       vg_work, vUpd_work)
        new_err = MinimumError(new_V, new_dcov, s0.error.status, true)

        # ── Step 9: build new state, correct edm
        s0 = MinimumState(new_par, new_err, new_grad, new_edm, ncalls(cf))
        edm_corrected = new_edm * (1.0 + 3.0 * new_err.dcovar)
    end

    # ── Determine final status
    final = s0
    reached_limit = ncalls(cf) >= maxfcn && edm_corrected > edmval
    above_max = edm_corrected > 10 * edmval

    is_valid_final = !reached_limit && !above_max && is_valid(final)

    return FunctionMinimum(
        final, seed, cf.up;
        is_valid = is_valid_final,
        reached_call_limit = reached_limit,
        above_max_edm = above_max,
        hesse_failed = false,  # Phase 1
        made_pos_def = made_pos_def_flag,
    )
end
