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

# Thread safety

All internal scratch buffers (ping-pong state buffers, line-search
scratch, etc.) are stack-local to each `migrad` call, so multiple
threads running `migrad` concurrently on **different** `CostFunction`
objects are safe.

`CostFunction` carries a mutable `Base.RefValue{Int}` call counter,
so sharing one `CostFunction` across threads causes a benign race
on the counter (no memory corruption, but `nfcn` will be undercounted).
Best practice: one `CostFunction` per thread (cheap — the wrapped
function is reused).
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
    migrad(f, x0, errs; up=1.0, strategy=Strategy(0), tol=0.1, maxfcn=..., prec=...)

Convenience overload that wraps a bare callable `f` into a
[`CostFunction`](@ref) before dispatching to the main `migrad`. The
parametric `CostFunction{typeof(f)}` ensures closure specialization at
the call site (parallel-review #2 F4 + ROADMAP §3.4 Criterion 4).

`up=1.0` for χ² fits, `up=0.5` for negative log-likelihood.
"""
function migrad(
    f::F,
    x0::AbstractVector{<:Real},
    errs::AbstractVector{<:Real};
    up::Real = 1.0,
    kwargs...,
) where {F}
    cf = CostFunction(f, up)
    return migrad(cf, x0, errs; kwargs...)
end

"""
    _migrad_loop(seed, cf, strategy, tol, maxfcn, prec) -> FunctionMinimum

The MIGRAD iteration loop proper. Public via `migrad(...)`; exposed for
re-entry tests and benchmarks.
"""
function _migrad_loop(
    seed::MinimumState,
    cf,                          # accepts CostFunction OR CostFunctionWithGradient;
    strategy::Strategy,          # method dispatch on numerical_gradient! etc.
    tol::Real,                   # picks the right per-FCN-type implementation.
    maxfcn::Integer,
    prec::MachinePrecision,
)
    n = length(seed)

    # Tolerance handling matches C++ exactly:
    # - ModularFunctionMinimizer.cxx:175 scales by Up()
    #   then floors at MnMachinePrecision().Eps2().
    # - VariableMetricBuilder.cxx:66 multiplies by 0.002.
    edmval = Float64(tol) * Float64(cf.up)
    if edmval < prec.eps2
        edmval = prec.eps2
    end
    edmval *= 0.002

    # Pre-loop call-limit check (matches C++ ModularFunctionMinimizer.cxx:182-187:
    # "Stop before iterating - call limit already exceeded").
    if ncalls(cf) >= maxfcn
        return FunctionMinimum(
            seed, seed, cf.up;
            is_valid = false,
            reached_call_limit = true,
        )
    end

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

    # ─────────────────────────────────────────────────────────────────────
    # Per-FIT scratch — allocated ONCE, shared across all iterations.
    #
    # • `step`, `ls_work`, `grad_work`, `vg_work`, `vUpd_work` are pure
    #   in-iteration scratch (overwritten each use; no aliasing risk).
    # • `dx_buf`, `dg_buf` are DFP-update temporaries (read once per iter,
    #   consumed by davidon_update!).
    # • State buffers (x/grad/g2/gstep/V) need **ping-pong** because s0
    #   wraps one set while the next iteration computes the new state into
    #   the OTHER set — without ping-pong we'd overwrite s0's data
    #   mid-iteration. Two sets (A, B) alternate via `use_a` flag.
    #
    # Net per-iteration allocations after this refactor: zero vectors,
    # zero matrices — only four ~48-byte immutable struct wrappers
    # (MinimumParameters/FunctionGradient/MinimumError/MinimumState).
    # ─────────────────────────────────────────────────────────────────────
    # `step` is fully written by `sym_mul!(step, V, g, -1.0, 0.0)` (β=0
    # → step is not read) before each use, so `undef` is safe.
    # `vUpd_work` stays `zeros` because `davidon_update!` reads its
    # initial state in some branches.
    step      = Vector{Float64}(undef, n)
    ls_work   = Vector{Float64}(undef, n)
    grad_work = Vector{Float64}(undef, n)
    vg_work   = Vector{Float64}(undef, n)
    vUpd_work = Symmetric(zeros(Float64, n, n), :U)
    dx_buf    = Vector{Float64}(undef, n)
    dg_buf    = Vector{Float64}(undef, n)

    # Ping-pong state buffer sets (A and B).
    #
    # V_a / V_b allocated `undef` — the lower triangle starts as garbage.
    # First use: `copyto!(parent(nV_buf), parent(s0.error.inv_hessian))`
    # copies the FULL parent matrix from the seed (which is `zeros(n,n)`
    # in `seed.jl:88`), so the lower triangle becomes 0 on first use.
    # Subsequent ops (davidon_update! via BLAS `syr!`, make_posdef via
    # full `copy`) preserve the zero lower triangle. This invariant
    # means we don't need `zeros(n,n)` here — saves one n²·8-byte fill
    # per fit.
    x_a   = Vector{Float64}(undef, n);  x_b   = Vector{Float64}(undef, n)
    g_a   = Vector{Float64}(undef, n);  g_b   = Vector{Float64}(undef, n)
    g2_a  = Vector{Float64}(undef, n);  g2_b  = Vector{Float64}(undef, n)
    gs_a  = Vector{Float64}(undef, n);  gs_b  = Vector{Float64}(undef, n)
    V_a   = Symmetric(Matrix{Float64}(undef, n, n), :U)
    V_b   = Symmetric(Matrix{Float64}(undef, n, n), :U)
    use_a = true   # next iter writes to set A; s0 ends up wrapping A

    made_pos_def_flag = false
    hessian_computed = false
    hesse_failed_flag = false

    # ─────────────────────────────────────────────────────────────────────
    # Outer do-while loop (C++ VariableMetricBuilder.cxx:111-185).
    #
    # First pass uses `maxfcn_eff = maxfcn`. After the first pass, the
    # budget grows to `int(maxfcn * 1.3)` so the optional Hesse-after-MIGRAD
    # refinement + re-iteration can finish (C++ line 182-184).
    #
    # Strategy ≥ 1 triggers MnHesse on the converged inner state:
    #   - Strategy 2: ALWAYS
    #   - Strategy 1: when Dcovar > 0.05 (DFP approximation is loose)
    # If the post-Hesse edm > edmval (and above machine accuracy), the
    # outer loop re-iterates the inner DFP loop with the refined state.
    # ─────────────────────────────────────────────────────────────────────
    maxfcn_eff = Int(maxfcn)
    ipass = 0
    iterate = true

    while iterate
        iterate = false

        # ── Inner DFP variable-metric loop ────────────────────────────
        while edm_corrected > edmval && ncalls(cf) < maxfcn_eff
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

            # ── Step 5: accept new point, compute new gradient.
            # Select target buffers via ping-pong (s0 currently wraps the
            # *other* set; we write to `use_a` set here).
            nx_buf  = use_a ? x_a  : x_b
            ng_buf  = use_a ? g_a  : g_b
            ng2_buf = use_a ? g2_a : g2_b
            ngs_buf = use_a ? gs_a : gs_b
            nV_buf  = use_a ? V_a  : V_b

            @inbounds @. nx_buf = s0.parameters.x + pp.x * step
            new_par = MinimumParameters(nx_buf, pp.y)
            # numerical_gradient! does copyto!(out.*, prev.*) internally,
            # so we don't need to zero-init ng_*. The ping-pong guarantees
            # `out` (= new_grad) and `prev` (= s0.gradient) reference
            # disjoint buffer sets, so the copyto! is well-defined.
            new_grad = FunctionGradient(ng_buf, ng2_buf, ngs_buf)
            numerical_gradient!(new_grad, grad_work, new_par, s0.gradient,
                                 cf, strategy, prec)

            # ── Step 6: EDM using OLD error matrix (C++ line 300).
            # `estimate_edm!` reuses vg_work; the allocating `estimate_edm`
            # would let BLAS build an internal temporary each call.
            new_edm = estimate_edm!(vg_work, new_grad, s0.error)

            if isnan(new_edm)
                break
            end

            # ── Step 7: if EDM < 0, try MnPosDef on s0's error
            if new_edm < 0
                s0 = make_posdef(s0, prec)
                made_pos_def_flag = true
                new_edm = estimate_edm!(vg_work, new_grad, s0.error)
                if new_edm < 0
                    break
                end
            end

            # ── Step 8: DFP update (reuses pre-allocated dx_buf, dg_buf, nV_buf)
            @inbounds @. dx_buf = new_par.x - s0.parameters.x
            @inbounds @. dg_buf = new_grad.grad - s0.gradient.grad

            # Copy s0's V into nV_buf's storage, then mutate in place.
            # `parent(...)` strips the Symmetric wrapper to the underlying
            # Matrix; we only need to copy the upper triangle but copyto!
            # on the full storage is faster (no branches) and the lower
            # half is unused by Symmetric(:U) semantics anyway.
            copyto!(parent(nV_buf), parent(s0.error.inv_hessian))
            new_dcov, _ = davidon_update!(nV_buf, dx_buf, dg_buf, s0.error.dcovar,
                                           vg_work, vUpd_work)
            # Reset status to MnHesseValid (matches C++ DavidonErrorUpdator.cxx:67-72
            # which constructs MinimumError(vUpd, dcov) — the regular dcov ctor,
            # not the tag ctor, so status implicitly clears to valid). Without this
            # reset, a transient MnMadePosDef from earlier sticks indefinitely.
            new_err = MinimumError(nV_buf, new_dcov, MnHesseValid, true)

            # ── Step 9: build new state, correct edm
            s0 = MinimumState(new_par, new_err, new_grad, new_edm, ncalls(cf))
            edm_corrected = new_edm * (1.0 + 3.0 * new_err.dcovar)

            use_a = !use_a   # flip so next iter writes to the OTHER set
        end

        # ── Strategy ≥ 1 inner-Hesse refinement (C++ lines 138-173) ─────
        # Bail out before Hesse if we already hit call limit, otherwise
        # Hesse will fail/be wasted.
        if ncalls(cf) >= maxfcn_eff
            break
        end

        if strategy.level == 2 ||
           (strategy.level == 1 && s0.error.dcovar > 0.05)
            # Compute remaining budget for Hesse. C++ passes maxfcn (the
            # full original budget) but we have ncalls already; let Hesse
            # use the leftover up to maxfcn_eff. Hesse internally floors
            # at its own default budget.
            budget_left = maxfcn_eff - ncalls(cf)
            s_hesse = hesse(cf, s0, strategy;
                             prec = prec, maxcalls = max(budget_left, 1))
            hessian_computed = true
            if !is_valid(s_hesse)
                hesse_failed_flag = true
                # Keep s0 as-is (C++ comment line 152: "Invalid Hessian - exit")
                break
            end
            s0 = s_hesse
            new_edm_h = s0.edm

            # Re-iterate the outer loop if Hesse moved edm above tolerance
            # AND above machine accuracy (C++ lines 160-168)
            if new_edm_h > edmval
                machine_limit = abs(prec.eps2 * s0.parameters.fval)
                if new_edm_h >= machine_limit
                    iterate = true
                end
            end
            edm_corrected = new_edm_h * (1.0 + 3.0 * s0.error.dcovar)
        end

        # Second-pass budget bump (C++ lines 182-184)
        if ipass == 0
            maxfcn_eff = floor(Int, maxfcn * 1.3)
        end
        ipass += 1
    end

    # ── Determine final status
    final = s0
    # C++ VariableMetricBuilder.cxx:350 marks reached-call-limit UNCONDITIONALLY
    # when nfcn ≥ maxfcn — even if EDM happens to be at convergence. Drop the
    # v1 AND-gate with edm_corrected > edmval (parallel-review #2 E5).
    # Use the EFFECTIVE budget (post-bump) for the check, not the original.
    reached_limit = ncalls(cf) >= maxfcn_eff
    above_max = edm_corrected > 10 * edmval

    is_valid_final = !reached_limit && !above_max && !hesse_failed_flag && is_valid(final)

    return FunctionMinimum(
        final, seed, cf.up;
        is_valid = is_valid_final,
        reached_call_limit = reached_limit,
        above_max_edm = above_max,
        hesse_failed = hesse_failed_flag,
        made_pos_def = made_pos_def_flag,
    )
end
