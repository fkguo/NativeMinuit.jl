# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# MigradScratch — per-fit scratch pool.
#
# Phase D (perf): when MnFunctionCross / MnContours runs N sequential
# inner MIGRADs at the same dimension (one per parabolic-fit probe), each
# `_migrad_loop` call allocates ~15 vectors + 3 symmetric matrices for
# its in-iteration scratch — that's ~160 probes × ~17 allocs ≈ 2700
# allocations per contour just for scratch. By pooling the scratch into
# a single `MigradScratch{n}` and passing it to `_migrad_loop` via the
# `scratch` kwarg, the contour driver allocates ONE scratch per inner
# dimension (typically n-1 for MINOS + axis points, n-2 for ray points)
# and reuses it across all probes.
#
# Buffers are stored in a mutable struct so the outer driver can pin
# the same reference across probe iterations. All `Vector{Float64}` and
# `Symmetric{Matrix{Float64}}` storage is allocated `undef` — the
# `_migrad_loop` body writes every buffer before any read (sym_mul! with
# β=0 zeros step; davidon_update!'s own fill! zeros vUpd_work; etc.), so
# no zero-fill is needed at construction.
# ─────────────────────────────────────────────────────────────────────────────

"""
    MigradScratch(n)

Pre-allocated scratch buffers for one or more `_migrad_loop` invocations
at dimension `n`. Reuse across calls saves ~15 vector + 3 symmetric
matrix allocations per call — significant when MnContours / MINOS runs
hundreds of inner MIGRADs at fixed inner dimension.

Mutable so callers can pin one instance across multiple probes via the
`Ref`/holder idiom. All fields are exposed for internal use; this
struct is not part of the public API (no export).

# Fields

- `n::Int` — buffer length / matrix order.
- `step, ls_work, grad_work, vg_work, dx_buf, dg_buf` — per-iteration
  scratch vectors (length `n`).
- `vUpd_work` — DFP update scratch (Symmetric{Matrix{Float64}}, n×n).
- `x_a, g_a, g2_a, gs_a, V_a` and `_b` ping-pong sets — alternating
  MinimumState buffer storage; one set wraps `s0`, the other receives
  the next iteration's values.

# Usage pattern (in `function_cross` / `function_cross_multi`)

```julia
scratch_holder = Ref{Union{Nothing,MigradScratch}}(nothing)
for each_probe in probes
    n_inner = ...                           # may differ across probes
    scratch = _get_scratch!(scratch_holder, n_inner)
    inner_min = migrad(cf_fixed, seed; scratch = scratch, ...)
end
```

`_get_scratch!` (helper below) lazily constructs or replaces the
scratch when the dimension changes; identical dimensions reuse.

# Aliasing / thread-safety

Same contract as `_fix_one_param` / `_fix_multi_params`: the scratch
struct's buffers are NOT re-entrant. Pass distinct scratches across
threads if you ever run parallel `_migrad_loop` calls.
"""
mutable struct MigradScratch
    n::Int
    step::Vector{Float64}
    ls_work::Vector{Float64}
    grad_work::Vector{Float64}
    vg_work::Vector{Float64}
    vUpd_work::Symmetric{Float64,Matrix{Float64}}
    dx_buf::Vector{Float64}
    dg_buf::Vector{Float64}
    x_a::Vector{Float64};  x_b::Vector{Float64}
    g_a::Vector{Float64};  g_b::Vector{Float64}
    g2_a::Vector{Float64}; g2_b::Vector{Float64}
    gs_a::Vector{Float64}; gs_b::Vector{Float64}
    V_a::Symmetric{Float64,Matrix{Float64}}
    V_b::Symmetric{Float64,Matrix{Float64}}
end

function MigradScratch(n::Integer)
    n_ = Int(n)
    n_ >= 1 ||
        throw(ArgumentError("MigradScratch n must be ≥ 1, got $n_"))
    MigradScratch(
        n_,
        Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_),
        Symmetric(Matrix{Float64}(undef, n_, n_), :U),
        Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_), Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_), Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_), Vector{Float64}(undef, n_),
        Vector{Float64}(undef, n_), Vector{Float64}(undef, n_),
        Symmetric(Matrix{Float64}(undef, n_, n_), :U),
        Symmetric(Matrix{Float64}(undef, n_, n_), :U),
    )
end

"""
    _get_scratch!(holder, n) -> MigradScratch

Lazy/replace helper: if `holder[]` is `nothing` or has wrong dimension,
construct a fresh `MigradScratch(n)` and assign back; otherwise return
the existing one. Used by the cross-search drivers to reuse scratch
across probes of the same inner dimension while still handling
dimension changes (MINOS n-1 → axis n-1 → ray n-2).
"""
@inline function _get_scratch!(
    holder::Base.RefValue{Union{Nothing,MigradScratch}},
    n::Integer,
)
    s = holder[]
    if s === nothing || s.n != Int(n)
        holder[] = MigradScratch(n)
    end
    return holder[]::MigradScratch
end

# ─────────────────────────────────────────────────────────────────────────────
# Phase H — thread-safety verification.
#
# The single biggest pitfall when enabling `threaded_gradient=true` is a
# user FCN with hidden mutable state (module-level scratch buffer, RNG,
# cache). Concurrent gradient calls race on that state, producing
# corrupted gradients; MIGRAD then slides to a different local minimum
# than the sequential run, SILENTLY giving a wrong answer.
#
# The IAM 2π form-factor fit (BenchmarkExamples/IAM_2Pformfactor/)
# exhibits this: `St4_00!` mutates `const c_00_4 = zeros(ComplexF64, 3, 3)`.
# Single-thread converges to χ²≈614; threaded "converges" to χ²≈987.
#
# `_verify_thread_safety` runs sequential + threaded numerical gradient
# at the seed point and asserts the results match within FP tolerance.
# If they differ, throws `ThreadSafetyError` with the max difference
# location and a pointer to the README mitigations.
# ─────────────────────────────────────────────────────────────────────────────

"""
    ThreadSafetyError(message)

Raised by `_migrad_loop` when `threaded_gradient=true` is used with
`verify_threading=true` and the user FCN's threaded gradient does not
match its sequential counterpart at the seed point (Phase H safety
check). Indicates the FCN has hidden mutable state that races under
parallel evaluation.
"""
struct ThreadSafetyError <: Exception
    message::String
end
Base.showerror(io::IO, e::ThreadSafetyError) = print(io, "ThreadSafetyError: ", e.message)

"""
    _verify_thread_safety(cf, seed::MinimumState, strategy, prec; tol=1e-8)

Run the same numerical gradient at `seed.parameters` two ways:
sequentially (`threaded=false`) and in parallel (`threaded=true`). If
the maximum element-wise relative difference exceeds `tol`, throw
`ThreadSafetyError` with a detailed diagnostic.

Cost: 2 × (one gradient evaluation) ≈ 4·n·grad_ncycles FCN calls
(~negligible for expensive FCNs; ~ms for cheap ones).

Internal — drivers call this via `verify_threading=true` kwarg.
"""
function _verify_thread_safety(cf, seed::MinimumState, strategy::Strategy,
                                 prec::MachinePrecision; tol::Float64 = 1e-8)
    n = length(seed)
    n >= 1 || return nothing

    # Seed buffers for `numerical_gradient!` from the seed's existing
    # gradient (so step refinement starts at the same point in both runs).
    grad_seq = FunctionGradient(zeros(n), zeros(n), zeros(n))
    grad_par = FunctionGradient(zeros(n), zeros(n), zeros(n))

    x_work_seq = similar(seed.parameters.x)
    x_work_par = similar(seed.parameters.x)

    # Sequential gradient (ground truth)
    numerical_gradient!(grad_seq, x_work_seq, seed.parameters, seed.gradient,
                          cf, strategy, prec; threaded = false)
    # Threaded gradient (potentially corrupted by user-FCN race)
    numerical_gradient!(grad_par, x_work_par, seed.parameters, seed.gradient,
                          cf, strategy, prec; threaded = true)

    # Element-wise relative difference
    max_rel_diff = 0.0
    bad_idx = 0
    @inbounds for i in 1:n
        diff = abs(grad_seq.grad[i] - grad_par.grad[i])
        scale = max(abs(grad_seq.grad[i]), abs(grad_par.grad[i]), 1e-12)
        rel = diff / scale
        if rel > max_rel_diff
            max_rel_diff = rel
            bad_idx = i
        end
    end

    if max_rel_diff > tol
        throw(ThreadSafetyError("""
            Threaded numerical gradient disagrees with sequential gradient
            at the seed point (Phase H verification).

              max relative difference = $(round(max_rel_diff; sigdigits=3))
              at parameter index $bad_idx of $n
              sequential[$bad_idx] = $(grad_seq.grad[bad_idx])
              threaded[$bad_idx]   = $(grad_par.grad[bad_idx])
              tolerance            = $tol

            ROOT CAUSE — your user FCN is not thread-safe.

            Most common HEP-fit pattern that violates this:
              const T_BUF = zeros(ComplexF64, 3, 3)     # ← module-level
              function chi2(par)
                  fill_T_matrix!(T_BUF, par)            # ← multiple threads race
                  return loss_from(T_BUF)
              end

            With `threaded_gradient=true`, n parallel calls to chi2 all
            mutate T_BUF simultaneously → MIGRAD gets corrupted gradients
            → silently converges to the WRONG local minimum.

            Mitigations (see README "THREAD-SAFETY CONTRACT" section):
              1. Move scratch into local scope (allocate per call).
              2. Use per-thread storage:
                   const T_POOL = [zeros(ComplexF64, 3, 3)
                                   for _ in 1:Threads.maxthreadid()]
                   function chi2(par)
                       T = T_POOL[Threads.threadid()]
                       fill_T_matrix!(T, par); loss_from(T)
                   end
              3. Disable threading: `threaded_gradient = false`.

            To bypass this check after verifying thread-safety some other
            way (NOT recommended unless you're absolutely sure):
              migrad(..., threaded_gradient=true, verify_threading=false)
            """))
    end
    return nothing
end

"""
    is_thread_safe(cf::AbstractCostFunction, x0::AbstractVector;
                    errs=fill(0.1, length(x0)), tol=1e-8,
                    strategy=Strategy(0), prec=MachinePrecision()) -> Bool

Standalone helper: test whether a user FCN gives the same numerical
gradient sequentially vs. threaded at `x0`. Returns `true` if safe,
`false` otherwise. Does NOT throw — for the throw version, see the
auto-check triggered by `migrad(..., threaded_gradient=true,
verify_threading=true)` (the default for high-level callers).

Use this to probe a new FCN before committing to `threaded_gradient=true`:

```julia
using JuMinuit
cf = CostFunction(my_chi2)
if Threads.nthreads() > 1 && JuMinuit.is_thread_safe(cf, x0)
    m = Minuit(my_chi2, x0; threaded_gradient=true)
else
    m = Minuit(my_chi2, x0)
end
```

Cost: equivalent to 2 gradient evaluations (≈ 4·n·grad_ncycles FCN
calls).
"""
function is_thread_safe(
    cf::AbstractCostFunction,
    x0::AbstractVector{<:Real};
    errs::AbstractVector{<:Real} = fill(0.1, length(x0)),
    tol::Float64 = 1e-8,
    strategy::Strategy = Strategy(0),
    prec::MachinePrecision = MachinePrecision(),
)
    if Threads.nthreads() == 1
        # Single-threaded Julia — threading is a no-op, "thread-safe" by definition
        return true
    end
    seed = seed_state(cf, x0, errs, strategy, prec)
    try
        _verify_thread_safety(cf, seed, strategy, prec; tol = tol)
        return true
    catch e
        e isa ThreadSafetyError || rethrow()
        return false
    end
end

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
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    verify_threading::Bool = threaded_gradient,
    prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    storage_level::Integer = 0,
)
    n = length(x0)
    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)

    # M5: optional `prior_cov` overrides the diagonal-from-g2 seed
    # inv_hessian (and sets `dcovar = 0`). Mirrors C++
    # MnSeedGenerator.cxx:63-67 `state.HasCovariance()` branch.
    seed = seed_state(cf, x0, errs, strategy, prec; prior_cov = prior_cov)
    return _migrad_loop(seed, cf, strategy, Float64(tol), maxfcn_eff, prec;
                          scratch = scratch,
                          threaded_gradient = threaded_gradient,
                          verify_threading = verify_threading,
                          storage_level = storage_level)
end

"""
    migrad(cf::CostFunction, seed::MinimumState;
           strategy=Strategy(0), tol=0.1, maxfcn=..., prec=MachinePrecision())
        -> FunctionMinimum

Low-level MIGRAD entry point that takes a pre-built `MinimumState` seed.
Skips the ~1 + 4·n FCN-call `seed_state` bootstrap. Intended for callers
that already have a warm gradient + inv_hessian (e.g., the parabolic-fit
probe chain in `function_cross_multi` — each probe reuses the previous
probe's converged state via [`warm_restart_state`](@ref)).

The seed must satisfy `is_valid(seed)`. Caller is responsible for having
evaluated the FCN at `seed.parameters.x` (so the call counter agrees
with `seed.nfcn`); `warm_restart_state` handles this automatically.

Mirrors the C++ `MnApplication::operator()(unsigned int maxfcn, double
tolerance)` overload that takes an already-constructed
`MinimumSeed`/`MinimumState` (vs. the user-x0/errs overload which calls
`MnSeedGenerator`).
"""
function migrad(
    cf::CostFunction,
    seed::MinimumState;
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    verify_threading::Bool = false,
    storage_level::Integer = 0,
)
    n = length(seed)
    maxfcn_eff = maxfcn === nothing ? (200 + 100 * n + 5 * n^2) : Int(maxfcn)
    # verify_threading default false here: warm-restart `migrad(cf, seed)`
    # is called from inner cross-search probes where outer migrad has
    # already verified. The 2-evaluation cost would multiply by probe
    # count if defaulted true.
    return _migrad_loop(seed, cf, strategy, Float64(tol), maxfcn_eff, prec;
                          scratch = scratch,
                          threaded_gradient = threaded_gradient,
                          verify_threading = verify_threading,
                          storage_level = storage_level)
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
    _migrad_loop(seed, cf, strategy, tol, maxfcn, prec;
                 scratch=nothing) -> FunctionMinimum

The MIGRAD iteration loop proper. Public via `migrad(...)`; exposed for
re-entry tests and benchmarks.

The optional `scratch::MigradScratch` argument lets a caller pool the
per-fit scratch buffers across multiple `_migrad_loop` invocations of
the same dimension. When `scratch === nothing` (default) the loop
allocates its own buffers — bit-for-bit identical to the pre-Phase-D
behavior. When supplied, scratch dimension MUST match `length(seed)`
or the call falls back to local allocation (defensive — should not
happen if the caller uses [`_get_scratch!`](@ref) correctly).
"""
function _migrad_loop(
    seed::MinimumState,
    cf,                          # accepts CostFunction OR CostFunctionWithGradient;
    strategy::Strategy,          # method dispatch on numerical_gradient! etc.
    tol::Real,                   # picks the right per-FCN-type implementation.
    maxfcn::Integer,
    prec::MachinePrecision;
    scratch::Union{Nothing,MigradScratch} = nothing,
    threaded_gradient::Bool = false,
    verify_threading::Bool = false,
    storage_level::Integer = 0,
)
    n = length(seed)
    # M6: per-iteration history. Empty when `storage_level == 0`
    # (default) — the snapshot pass below is skipped entirely so
    # there's no allocation overhead in the zero-alloc gate path.
    history = MinimumState[]

    # Phase H — thread-safety verification. When `threaded_gradient=true`
    # AND `verify_threading=true`, run one sequential + one threaded
    # gradient evaluation at the seed point and compare. If they differ
    # beyond FP-roundoff tolerance, the user FCN has hidden mutable state
    # that races under threading (common pattern: module-level scratch
    # buffer mutated by FCN body). Throw a `ThreadSafetyError` rather
    # than silently converge to a wrong minimum.
    #
    # The high-level drivers (`Minuit(..., threaded_gradient=true)`,
    # bare `migrad(cf, x0, errs; threaded_gradient=true)`) default
    # `verify_threading=true` for safety. Inner cross-search probes
    # (`_migrad_with_fixed`, `_migrad_with_multi_fixed`) skip via
    # `verify_threading=false` — the outer call already verified the
    # FCN, and JuMinuit's `cf_fixed` splice infrastructure (Phase G.1
    # per-thread `full_buf`) does not introduce thread-unsafety on
    # top of a thread-safe user FCN.
    if threaded_gradient && verify_threading && n >= 1 && is_valid(seed)
        _verify_thread_safety(cf, seed, strategy, prec)
    end

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
            states = history, storage_level = Int(storage_level),
        )
    end

    if n == 0
        return FunctionMinimum(seed, seed, cf.up; is_valid = false,
                                states = history,
                                storage_level = Int(storage_level))
    end
    if !is_valid(seed)
        return FunctionMinimum(seed, seed, cf.up; is_valid = false,
                                states = history,
                                storage_level = Int(storage_level))
    end
    if seed.edm < 0
        return FunctionMinimum(seed, seed, cf.up; is_valid = false,
                                states = history,
                                storage_level = Int(storage_level))
    end

    s0 = seed
    # Initial-state EDM correction (C++ line 229)
    edm_corrected = s0.edm * (1.0 + 3.0 * s0.error.dcovar)

    # ─────────────────────────────────────────────────────────────────────
    # Per-FIT scratch — bound either to a caller-supplied pool
    # (`scratch::MigradScratch`, reused across probes inside MnContours /
    # MINOS) or to a freshly-allocated local pool (when caller passed
    # `scratch=nothing` — backward-compatible path).
    #
    # IMPORTANT ALIASING CONTRACT (Phase D codex review):
    #   The returned `FunctionMinimum.state` wraps the scratch's buffers
    #   directly. If a caller pools the SAME scratch across multiple
    #   `_migrad_loop` invocations and then RETAINS the returned state
    #   from an earlier call past the start of a later call, the
    #   retained state's parameters / gradient / inv_hessian are MUTATED
    #   by the later iteration. Current `minos` / `contour_exact` are
    #   safe because they consume `aopt` / `validity` / scalar y-axis
    #   values immediately and never retain inner states across pooled
    #   probes (minos.jl:144, contours.jl:148, contours.jl:219). Treat
    #   `scratch` as INTERNAL only.
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
    # FAIL FAST on size mismatch — silent fallback would hide caller bugs.
    # The drivers (`function_cross[_multi]`, `contour_exact`) ALWAYS
    # preflight via `_get_scratch!`, so this throw can only fire from
    # external/direct callers wiring the wrong scratch dimension.
    #
    # Note on `undef` initialization: `step` is fully written by
    # `sym_mul!(step, V, g, -1.0, 0.0)` (β=0 → step is not read) before
    # each use. `vUpd_work` is `fill!(parent(vUpd_work), 0.0)`-zeroed
    # by `davidon_update!` itself (davidon.jl:123). Ping-pong V_a / V_b
    # have their lower triangles overwritten via `copyto!(parent(...), ...)`
    # before any read; subsequent in-place DFP / make_posdef ops preserve
    # the upper-triangle authoritative semantics. So `undef` everywhere
    # is safe and saves one n² zero-fill per buffer at construction.
    # ─────────────────────────────────────────────────────────────────────
    if scratch !== nothing && scratch.n != n
        throw(DimensionMismatch(
            "MigradScratch.n=$(scratch.n) ≠ seed dim $n; the driver " *
            "should call `_get_scratch!(holder, n)` before `_migrad_loop`."))
    end
    s_eff = scratch === nothing ? MigradScratch(n) : scratch
    step      = s_eff.step
    ls_work   = s_eff.ls_work
    grad_work = s_eff.grad_work
    vg_work   = s_eff.vg_work
    vUpd_work = s_eff.vUpd_work
    dx_buf    = s_eff.dx_buf
    dg_buf    = s_eff.dg_buf
    x_a   = s_eff.x_a;  x_b   = s_eff.x_b
    g_a   = s_eff.g_a;  g_b   = s_eff.g_b
    g2_a  = s_eff.g2_a; g2_b  = s_eff.g2_b
    gs_a  = s_eff.gs_a; gs_b  = s_eff.gs_b
    V_a   = s_eff.V_a
    V_b   = s_eff.V_b
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
                    break  # still bad — terminate this iteration
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
                                 cf, strategy, prec;
                                 threaded = threaded_gradient)

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
            #
            # Self-copy guard (Phase D): when `scratch` is reused across
            # `MnFunctionCross` probes AND the previous probe's final
            # write landed on the same ping-pong slot the new iter just
            # selected, `s0.error.inv_hessian` (wrapping prev's final
            # V_a/V_b) can ALIAS `nV_buf`. `copyto!(M, M)` on plain
            # `Matrix{Float64}` is a well-defined no-op today, but
            # treating the storage identity check as the invariant
            # protects against any future move to view-based / sparse
            # inv_hessian storage where `copyto!` overlap is UB.
            # (Phase A reviewer's NICE-TO-HAVE; cost = one pointer
            # comparison per inner iter, ~ns.)
            dst_M = parent(nV_buf)
            src_M = parent(s0.error.inv_hessian)
            if dst_M !== src_M
                copyto!(dst_M, src_M)
            end
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

            # M6: snapshot per-iteration state when storage_level >= 1.
            # Deep-copies all internal storage because the ping-pong
            # buffers will be overwritten next iter, mutating any
            # shallow snapshot we'd push here. The copy is paid only
            # by callers who explicitly opted in via storage_level=1.
            if storage_level >= 1
                push!(history, _snapshot_state(s0))
            end

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

            # M6: snapshot the post-Hesse refined state too — it's a
            # genuine algorithmic step (not just a DFP iter) that the
            # caller doing convergence-plot work would want to see.
            if storage_level >= 1
                push!(history, _snapshot_state(s0))
            end

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
        states = history,
        storage_level = Int(storage_level),
    )
end

# M6: deep-copy a MinimumState so it can be safely retained past the
# next ping-pong iteration (`_migrad_loop` overwrites the V_a/V_b /
# x_a/x_b / g_a/g_b buffers each iter, mutating any shallow reference
# in-place). Only called when `storage_level >= 1` so the alloc cost
# is paid only by callers who explicitly opted in.
function _snapshot_state(s::MinimumState)
    par_old = s.parameters
    par = par_old.has_step_size ?
           MinimumParameters(copy(par_old.x), copy(par_old.dirin), par_old.fval) :
           MinimumParameters(copy(par_old.x), par_old.fval)
    g_old = s.gradient
    grad = FunctionGradient(copy(g_old.grad), copy(g_old.g2), copy(g_old.gstep);
                             analytical = g_old.analytical)
    err_old = s.error
    # `parent(...)` strips the Symmetric wrapper; `copy` then makes a
    # fresh Matrix. Wrap as :U (the convention used throughout).
    inv_h = Symmetric(copy(parent(err_old.inv_hessian)), :U)
    err = MinimumError(inv_h, err_old.dcovar, err_old.status, err_old.available)
    return MinimumState(par, err, grad, s.edm, s.nfcn)
end
