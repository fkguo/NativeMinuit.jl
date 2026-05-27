# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# gradient.jl — Initial + Numerical2P gradient calculators.
#
# Mirrors reference/Minuit2_cpp/src/InitialGradientCalculator.cxx and
# reference/Minuit2_cpp/src/Numerical2PGradientCalculator.cxx.
#
# Bounded fits go through `migrad_bounded.jl`, which wraps the user
# FCN to take internal coords; the int↔ext transformation is in
# `transform.jl`. The C++ `if HasLimits; step ≤ 0.5` clamps at
# InitialGradientCalculator.cxx:66-69 and Numerical2PGradientCalculator
# .cxx:136-139 are not implemented (they'd require threading
# `has_limits[]` through `seed_state`); the inline "NOTE — known
# limitation" comments below document why they're dormant in practice.
#
# Per ROADMAP §2.1, the Numerical2P gradient is the dominant cost in
# MIGRAD for cheap FCNs: 2·n FCN calls per cycle × up to Ncycle cycles
# per gradient call × however many gradient calls per minimization.
# This file is therefore on the critical performance path.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Initial gradient — rough estimate from parameter step sizes.
# Mirrors InitialGradientCalculator.cxx:25-79.
# ─────────────────────────────────────────────────────────────────────────────

"""
    initial_gradient!(out, par, errs, up, prec=MachinePrecision()) -> out

In-place: compute the initial rough gradient estimate using parameter
step sizes (a.k.a. parameter "errors" / Error()). Writes into
`out.grad`, `out.g2`, `out.gstep`.

For the no-bounds case (Phase 0), per `InitialGradientCalculator.cxx:60-72`:

- `gsmin = 8 · eps2 · (|x_i| + eps2)`
- `dirin = max(werr_i, gsmin)`  (no-limit simplification: `vplu = werr`,
  `vmin = -werr`, so `0.5(|vplu|+|vmin|) = werr`)
- `g2 = 2 · ErrorDef / dirin²`
- `gstep = max(gsmin, 0.1 · dirin)`
- `grd = g2 · dirin`

# Arguments
- `out::FunctionGradient` — must be a valid `FunctionGradient` of size
  `length(par)`; its three vectors are overwritten in place.
- `par::MinimumParameters` — current parameter point (reads `par.x`).
- `errs::AbstractVector{Float64}` — per-parameter step size estimates
  (the "Error" values supplied with the user's `x0`).
- `up::Real` — error definition (`cf.up`).
- `prec::MachinePrecision` — defaults to `MachinePrecision()`.

# Phase 0 / Phase 1 boundary

The `HasLimits()` branches at C++ lines 47-58 and 66-69 (handling
`UpperLimit`, `LowerLimit`, `gstep` clamp to 0.5) are **not ported in
Phase 0**. They reappear in Phase 1's `transform.jl`.
"""
function initial_gradient!(
    out::FunctionGradient,
    par::MinimumParameters,
    errs::AbstractVector{Float64},
    up::Real,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    length(out) == n ||
        throw(DimensionMismatch("FunctionGradient length $(length(out)) != par length $n"))
    length(errs) == n ||
        throw(DimensionMismatch("errs length $(length(errs)) != par length $n"))

    eps2 = prec.eps2
    up_f = Float64(up)

    @inbounds for i in 1:n
        var = par.x[i]
        werr = errs[i]
        gsmin = 8.0 * eps2 * (abs(var) + eps2)
        # No-limits simplification of C++ `0.5 * (|vplu| + |vmin|)`:
        # in the no-limits case vplu = werr, vmin = -werr, so
        # 0.5·(|vplu|+|vmin|) = |werr|. The `abs` matters when a
        # user supplies negative step sizes — without it Julia would
        # silently differ from C++ (parallel-review B1).
        dirin = max(abs(werr), gsmin)
        # Safety: never zero (prevents NaN in g2 = 2·up/dirin²).
        # NOTE — known limitation: C++ InitialGradientCalculator.cxx:66-69
        # clamps `gstep > 0.5` for parameters with limits, to keep the
        # finite-difference step inside the sin/sqrt transform's locally
        # linear region. JuMinuit doesn't thread `has_limits[]` through
        # `seed_state`, so the clamp isn't applied. Dormant for typical
        # use because `gstep = 0.1·dirin` with `dirin ≤ 5` keeps gstep
        # under the threshold; would matter if a user supplied huge
        # initial step sizes (`errs ≫ 5`) on a bounded parameter.
        g2 = 2.0 * up_f / (dirin * dirin)
        gstep = max(gsmin, 0.1 * dirin)
        grd = g2 * dirin
        out.grad[i] = grd
        out.g2[i] = g2
        out.gstep[i] = gstep
    end
    return out
end

"""
    initial_gradient(par, errs, cf, prec=MachinePrecision()) -> FunctionGradient

Allocating convenience: returns a fresh `FunctionGradient` with the
initial rough estimate. Allocates three `Vector{Float64}` of length `n`.
"""
function initial_gradient(
    par::MinimumParameters,
    errs::AbstractVector{Float64},
    cf::CostFunction,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    out = FunctionGradient(zeros(n), zeros(n), zeros(n))
    initial_gradient!(out, par, errs, cf.up, prec)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Numerical2P gradient — two-point central-difference refinement.
# Mirrors Numerical2PGradientCalculator.cxx:34-230.
# ─────────────────────────────────────────────────────────────────────────────

"""
    numerical_gradient!(out, x_work, par, prev, cf, strategy, prec) -> out

In-place two-point central-difference gradient calculation with
per-coordinate iterative step refinement. Mirrors
`Numerical2PGradientCalculator.cxx:34-230`.

# Algorithm (per parameter `i`, up to `ncycle` cycles):

1. Compute optimal central-difference step `optstp = √(dfmin / (|g2|+epspri))`.
2. Clamp step against `0.1·gstep`, `10·gstep`, and `8·eps2·|x_i|`.
3. If `|step − step_prev| / step < step_tol`, break (step converged).
4. Save new step into `gstep[i]`; compute `fs1 = f(x + step·e_i)`,
   `fs2 = f(x − step·e_i)`.
5. Update `grad[i] = (fs1 − fs2) / (2·step)`, `g2[i] = (fs1 + fs2 − 2·fcnmin) / step²`.
6. If `|grdb4 − grad[i]| / (|grad[i]| + dfmin/step) < grad_tol`, break.

# Arguments

- `out::FunctionGradient` — written in place. Caller must initialize it
  with previous (or initial) `grad`, `g2`, `gstep` — these are *also*
  used as input. (To avoid surprises, the convenience wrapper
  `numerical_gradient(...)` does this for you.)
- `x_work::Vector{Float64}` — preallocated workspace of length `n`.
  Reset to `par.x` at entry; mutated and restored at every coordinate
  perturbation; finally still equals `par.x` (modulo Float64 roundoff
  in step add/subtract) on exit.
- `par::MinimumParameters` — current point + `fval` (= `fcnmin`).
- `prev::FunctionGradient` — previous iteration's gradient. Source of
  the initial `grad/g2/gstep` for the refinement (copied into `out`
  before refinement starts).
- `cf::CostFunction` — the user FCN; `cf.up` is `ErrorDef`.
- `strategy::Strategy` — supplies `grad_ncycles`, `grad_step_tolerance`,
  `grad_tolerance`.
- `prec::MachinePrecision`.

# Bounded parameters

The C++ `HasLimits()` clamp at lines 136-139 (`step > 0.5 → step = 0.5`
when the parameter has limits) is intentionally not implemented in
JuMinuit. See the inline "NOTE — known limitation" comment in the
body for the rationale. Dormant for typical fits because `step` is
bounded by `10·gstep` and our `gstep ≈ 0.1·dirin` keeps it under the
0.5 threshold for reasonable user-supplied step sizes.

# Performance

Zero-allocation in the inner cycle (no per-iteration vector creation;
the user FCN may allocate at its discretion — that's measured separately
by `bench_long_fit.jl`). The FCN call counter (`cf.nfcn[]`) advances
2× per cycle per coordinate, capped at `2·n·ncycle` for a full call.
"""
function numerical_gradient!(
    out::FunctionGradient,
    x_work::AbstractVector{Float64},
    par::MinimumParameters,
    prev::FunctionGradient,
    cf::CostFunction,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision();
    threaded::Bool = false,
)
    n = length(par)
    length(out) == n ||
        throw(DimensionMismatch("out length $(length(out)) != par length $n"))
    length(x_work) == n ||
        throw(DimensionMismatch("x_work length $(length(x_work)) != par length $n"))
    length(prev) == n ||
        throw(DimensionMismatch("prev length $(length(prev)) != par length $n"))

    # Seed the working gradient from the previous iteration's values.
    copyto!(out.grad, prev.grad)
    copyto!(out.g2, prev.g2)
    copyto!(out.gstep, prev.gstep)

    # Working parameter vector — modified at column i during cycle j,
    # restored before moving to the next coordinate.
    copyto!(x_work, par.x)
    fcnmin = par.fval

    eps2 = prec.eps2
    eps = prec.eps
    up = cf.up
    dfmin = 8.0 * eps2 * (abs(fcnmin) + up)
    vrysml = 8.0 * eps * eps
    ncycle = strategy.grad_ncycles
    step_tol = strategy.grad_step_tolerance
    grad_tol = strategy.grad_tolerance

    if threaded && Threads.nthreads() > 1
        # Phase 2.2 threaded gradient — mirrors the OpenMP block at
        # reference/Minuit2_cpp/src/Numerical2PGradientCalculator.cxx:116-127.
        # Each parameter index `i` writes to disjoint slots of
        # out.grad/g2/gstep; no cross-thread reduction needed. Each
        # thread holds a private copy of x_work so the per-coord
        # `x_work[i] = xtf ± step` mutations don't race.
        # Use Threads.maxthreadid() because Julia 1.12 may dispatch
        # tasks to thread IDs > Threads.nthreads() (interactive +
        # foreign-task scenarios).
        n_buffers = max(Threads.maxthreadid(), Threads.nthreads())
        x_work_perthread = [copy(x_work) for _ in 1:n_buffers]
        # Phase G — `:static` scheduling. Codex review: under non-`:static`
        # schedules Julia DOES NOT guarantee `Threads.threadid()` stays
        # constant within a single iteration body (task may migrate
        # between yields), which would break the per-thread buffer
        # indexing in cf_fixed's `full_bufs[threadid()]`. `:static` IS
        # guaranteed to keep threadid stable within an iter under the
        # implicit no-yield contract our numeric FCN bodies satisfy.
        # Trade-off: M3 P/E-core asymmetry means `:static`'s round-robin
        # lets fast cores wait on slow cores → measured ~1.25× speedup
        # (vs `:dynamic` ~1.39× but UNSAFE under thread-migration). The
        # 0.14× gap is acceptable cost for correctness.
        Threads.@threads :static for i in 1:n
            tid = Threads.threadid()
            xw = x_work_perthread[tid]
            xtf = xw[i]
            epspri = eps2 + abs(out.grad[i] * eps2)
            stepb4 = 0.0
            @inbounds for _ in 1:ncycle
                optstp = sqrt(dfmin / (abs(out.g2[i]) + epspri))
                step = max(optstp, abs(0.1 * out.gstep[i]))
                stpmax = 10.0 * abs(out.gstep[i])
                if step > stpmax; step = stpmax; end
                stpmin = max(vrysml, 8.0 * abs(eps2 * xw[i]))
                if step < stpmin; step = stpmin; end
                if abs((step - stepb4) / step) < step_tol
                    break
                end
                out.gstep[i] = step
                stepb4 = step

                xw[i] = xtf + step
                fs1 = cf(xw)
                xw[i] = xtf - step
                fs2 = cf(xw)
                xw[i] = xtf

                grdb4 = out.grad[i]
                out.grad[i] = 0.5 * (fs1 - fs2) / step
                out.g2[i] = (fs1 + fs2 - 2.0 * fcnmin) / (step * step)

                if abs(grdb4 - out.grad[i]) / (abs(out.grad[i]) + dfmin / step) < grad_tol
                    break
                end
            end
        end
        return out
    end

    @inbounds for i in 1:n
        xtf = x_work[i]
        epspri = eps2 + abs(out.grad[i] * eps2)
        stepb4 = 0.0
        for _ in 1:ncycle
            optstp = sqrt(dfmin / (abs(out.g2[i]) + epspri))
            step = max(optstp, abs(0.1 * out.gstep[i]))
            # NOTE — known limitation: C++ Numerical2PGradientCalculator
            # .cxx:136-139 clamps `if HasLimits(i) && step > 0.5; step = 0.5`
            # for the same reason as `initial_gradient!` above (sin/sqrt
            # transform linearity floor). Not implemented because we
            # don't thread `has_limits[]` through to this function;
            # dormant for current tests because `step ≤ stpmax = 10·gstep`
            # and typical gstep is well below 0.05.
            stpmax = 10.0 * abs(out.gstep[i])
            if step > stpmax
                step = stpmax
            end
            stpmin = max(vrysml, 8.0 * abs(eps2 * x_work[i]))
            if step < stpmin
                step = stpmin
            end
            if abs((step - stepb4) / step) < step_tol
                break
            end
            out.gstep[i] = step
            stepb4 = step

            x_work[i] = xtf + step
            fs1 = cf(x_work)
            x_work[i] = xtf - step
            fs2 = cf(x_work)
            x_work[i] = xtf

            grdb4 = out.grad[i]
            out.grad[i] = 0.5 * (fs1 - fs2) / step
            out.g2[i] = (fs1 + fs2 - 2.0 * fcnmin) / (step * step)

            if abs(grdb4 - out.grad[i]) / (abs(out.grad[i]) + dfmin / step) < grad_tol
                break
            end
        end
    end

    return out
end

"""
    numerical_gradient(par, prev, cf, strategy, prec) -> FunctionGradient

Allocating convenience wrapper around [`numerical_gradient!`](@ref).
Allocates `out` (three vectors of length `n`) plus a one-shot
`x_work` workspace. Returns the refined gradient.
"""
function numerical_gradient(
    par::MinimumParameters,
    prev::FunctionGradient,
    cf::CostFunction,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    out = FunctionGradient(zeros(n), zeros(n), zeros(n))
    x_work = similar(par.x)
    numerical_gradient!(out, x_work, par, prev, cf, strategy, prec)
    return out
end

"""
    numerical_gradient(par, cf, strategy, prec) -> FunctionGradient

First-call convenience: computes the initial gradient internally (using
the seed step sizes from `par.dirin`), then refines via the two-point
calculator. Equivalent to the C++ overload at
`Numerical2PGradientCalculator.cxx:34-42`.
"""
function numerical_gradient(
    par::MinimumParameters,
    cf::CostFunction,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    has_step_size(par) ||
        throw(ArgumentError("numerical_gradient(par, cf, ...) without explicit prev " *
                            "gradient requires par to carry step sizes (par.dirin); " *
                            "construct via MinimumParameters(x, dirin, fval)."))
    init = initial_gradient(par, par.dirin, cf, prec)
    return numerical_gradient(par, init, cf, strategy, prec)
end
