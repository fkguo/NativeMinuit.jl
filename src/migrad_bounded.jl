# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# migrad_bounded.jl — bound-aware MIGRAD wrapper.
#
# Mirrors the C++ Minuit2 user-facing flow at
# reference/Minuit2_cpp/src/MnUserFcn.cxx (int→ext transform at the FCN
# call boundary) + MnUserParameterState round-tripping.
#
# Design:
#  - Public API: `migrad(cf, params::Parameters; kwargs...) -> BoundedFunctionMinimum`
#  - User FCN `cf.f` operates on EXTERNAL parameter values (the bounded
#    physical values they care about).
#  - The internal MIGRAD optimizer operates on UNBOUNDED INTERNAL values
#    via sin/sqrt transforms (transform.jl).
#  - A wrapper FCN converts internal→external before each user FCN call.
#  - The resulting MinimumState lives in internal coordinates; we
#    transform back to external for the user-visible result.
#
# Phase 1 first cut scope:
#  - Fixed parameters: respected via the `Parameters.int_of_ext`/`ext_of_int`
#    maps. The internal optimizer sees only the FREE parameters.
#  - Bounds: handled via int↔ext transforms.
#  - Covariance back-conversion: V_ext[i,j] = V_int[i,j] · dint2ext_i · dint2ext_j
#    (Jacobian product per parameter).
# ─────────────────────────────────────────────────────────────────────────────

"""
    BoundedFunctionMinimum

Result of bounded `migrad(cf, params)`. Wraps the internal
`FunctionMinimum` and exposes user-visible external values.

# Fields

- `internal::FunctionMinimum` — the MIGRAD result in internal coords.
- `params::Parameters` — the parameter metadata (bounds, fixed flags).
- `ext_values::Vector{Float64}` — final external parameter values
  (length = total `n_pars`, including fixed parameters at their
  initial values).
- `ext_errors::Vector{Float64}` — final external 1σ errors via
  Jacobian chain rule (length = `n_pars`; fixed params have 0).
- `ext_covariance::Union{Nothing,Matrix{Float64}}` — full
  `n_pars × n_pars` external covariance matrix (or `nothing` if
  inner MIGRAD did not produce a covariance).
- `internal_cf::CostFunction` — the int-coord-wrapped FCN used by the
  internal MIGRAD. Required for follow-up calls (MINOS, contour) that
  operate on `internal` and must consume internal coordinates — using
  the user's external `cf` there would leak coordinate frames
  (parallel-review #4 A7/B4 blocking).
"""
struct BoundedFunctionMinimum
    internal::FunctionMinimum
    params::Parameters
    ext_values::Vector{Float64}
    ext_errors::Vector{Float64}
    ext_covariance::Union{Nothing,Matrix{Float64}}
    # Phase F: was concretely `CostFunction`. Now `AbstractCostFunction`
    # so the AD gradient survives MIGRAD → MINOS / contour transitions
    # at the high-level Minuit / Minuit.minos! / Minuit.draw_mncontour
    # path (codex Phase F review: the low-level direct calls were already
    # AD-correct; the user-facing wrapper was silently dropping the
    # gradient through the plain `CostFunction` wrap at migrad_bounded.jl
    # construction).
    internal_cf::AbstractCostFunction
end

# Accessors mirroring iminuit-style
fval(m::BoundedFunctionMinimum) = m.internal.state.parameters.fval
edm(m::BoundedFunctionMinimum) = edm(m.internal)
nfcn(m::BoundedFunctionMinimum) = nfcn(m.internal)
is_valid(m::BoundedFunctionMinimum) = m.internal.is_valid
Base.values(m::BoundedFunctionMinimum) = m.ext_values

"""
    ext_errors(m::BoundedFunctionMinimum) -> Vector{Float64}

External-coordinate 1σ errors via Jacobian chain rule.
"""
ext_errors(m::BoundedFunctionMinimum) = m.ext_errors

"""
    ext_covariance(m::BoundedFunctionMinimum) -> Union{Nothing,Matrix{Float64}}

Full external-coordinate covariance matrix, dimension n_total × n_total
with zero rows/columns for fixed parameters. Returns `nothing` if no
covariance was produced.
"""
ext_covariance(m::BoundedFunctionMinimum) = m.ext_covariance

"""
    free_covariance(m::BoundedFunctionMinimum) -> Union{Nothing,Matrix{Float64}}

The free-parameter sub-block of the external covariance, dimension
n_free × n_free. Matches the C++ `MnUserParameterState::Covariance()`
shape (which omits fixed parameters entirely). Use this when
interoperating with C++ output or when fixed-parameter zero rows are
unwanted.

Parallel-review #4 D4 — the design choice was to expose the full
n_total × n_total matrix as the default for indexing convenience;
this accessor provides the C++-shape alternative on demand.
"""
function free_covariance(m::BoundedFunctionMinimum)
    cov = m.ext_covariance
    cov === nothing && return nothing
    free_idx = [params_i for params_i in 1:n_pars(m.params)
                if !is_fixed(m.params.pars[params_i])]
    return cov[free_idx, free_idx]
end

# Accessor parity with FunctionMinimum (parallel-review #4 E3).
errors(m::BoundedFunctionMinimum) = m.ext_errors
covariance(m::BoundedFunctionMinimum) = m.ext_covariance

# ─────────────────────────────────────────────────────────────────────────────
# Helper: wrap a user FCN to take internal coords and call user FCN with ext.
# ─────────────────────────────────────────────────────────────────────────────

function _wrap_fcn_internal_to_external(cf::CostFunction, params::Parameters)
    f = cf.f
    up = cf.up
    p_ref = params
    # Per-thread reusable ext buffer so the int→ext transform on the
    # per-FCN-call hot path allocates nothing after warm-up. Count with
    # `maxthreadid()` and index with `threadid()` — the canonical in-repo
    # idiom (function_cross.jl:525-530, _fix_one_param). The threaded
    # numerical gradient drives this closure from a `@threads :static`
    # loop where `threadid()` is stable within an iteration, so distinct
    # threads touch distinct buffers (no race). Single-threaded Julia →
    # one buffer, zero overhead.
    nbuf = max(1, Threads.maxthreadid())
    ext_bufs = [Vector{Float64}(undef, n_pars(p_ref)) for _ in 1:nbuf]
    # Skip the `collect(Float64, int_vec)` allocation; int_to_ext_vector!
    # accepts any AbstractVector<:Real (parallel-review #4 D6).
    wrapped = let ext_bufs = ext_bufs, p_ref = p_ref, f = f
        function (int_vec::AbstractVector{<:Real})
            ext_full = int_to_ext_vector!(ext_bufs[Threads.threadid()], p_ref, int_vec)
            return f(ext_full)
        end
    end
    return CostFunction(wrapped, up)
end

# Bounded path with user-supplied gradient. Wraps both f and g into
# internal coords. Chain rule on the gradient is component-wise because
# the int↔ext transform per Minuit2 has no cross-parameter coupling:
#
#   g_int[i] = (∂ext_i / ∂int_i) · g_ext[ext_of_int[i]]
#            = dint2ext_value(params, i, int_val) · g_ext[ext_of_int[i]]
#
# Fixed parameters never appear in the int vector; their gradient
# components from g_ext are dropped here.
function _wrap_fcn_internal_to_external(cf::CostFunctionWithGradient,
                                          params::Parameters)
    f = cf.f
    g = cf.g
    up = cf.up
    p_ref = params
    n_active = n_free(params)
    # Per-thread reusable ext buffers (see the ::CostFunction overload
    # above for the threadid/maxthreadid rationale). wrapped_f and
    # wrapped_g get independent pools so a function eval and a gradient
    # eval never alias the same scratch. The g_int result stays freshly
    # allocated — only the int→ext transform is buffered here (scope:
    # this perf change is the ext-vector reuse alone).
    nbuf = max(1, Threads.maxthreadid())
    ext_bufs_f = [Vector{Float64}(undef, n_pars(p_ref)) for _ in 1:nbuf]
    ext_bufs_g = [Vector{Float64}(undef, n_pars(p_ref)) for _ in 1:nbuf]
    wrapped_f = let ext_bufs_f = ext_bufs_f, p_ref = p_ref, f = f
        function (int_vec::AbstractVector{<:Real})
            ext_full = int_to_ext_vector!(ext_bufs_f[Threads.threadid()], p_ref, int_vec)
            return f(ext_full)
        end
    end
    wrapped_g = let ext_bufs_g = ext_bufs_g, p_ref = p_ref, g = g, n_active = n_active
        function (int_vec::AbstractVector{<:Real})
            ext_full = int_to_ext_vector!(ext_bufs_g[Threads.threadid()], p_ref, int_vec)
            g_ext = g(ext_full)
            g_int = Vector{Float64}(undef, n_active)
            @inbounds for int_idx in 1:n_active
                ext_idx = p_ref.ext_of_int[int_idx]
                d = dint2ext_value(p_ref, int_idx, Float64(int_vec[int_idx]))
                g_int[int_idx] = d * Float64(g_ext[ext_idx])
            end
            return g_int
        end
    end
    # Share the user-facing CFwG's nfcn + ngrad Refs so call counters
    # surfaced via `m.nfcn` / `m.ngrad` reflect ALL calls (the wrap
    # closure is what the inner MIGRAD actually drives). `check_gradient`
    # must be forwarded too — this wrap is on the path EVERY `migrad!(m;
    # grad=…)` takes, so dropping it would silently re-enable the
    # CheckGradient seed check and defeat the `check_gradient=false` opt-out.
    return CostFunctionWithGradient(wrapped_f, wrapped_g, up,
                                     cf.nfcn, cf.ngrad;
                                     check_gradient = cf.check_gradient)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-internal(free)-parameter bound flags for the MnHesse step
# clamp (MnHesse.cxx:160-167, 194-195). Entry `i` is `true` iff the i-th
# FREE parameter (internal index) maps to an externally-bounded parameter
# — the JuMinuit analogue of C++ `trafo.Parameter(i).HasLimits()`. The
# bounded `migrad(cf, params)` / `hesse(m::Minuit)` paths pass the result
# as `has_limits=` so the diagonal probe step `d` is clamped at 0.5 in
# INTERNAL (transformed) coordinates — the frame the C++ clamp targets.
# Length matches the internal optimizer's parameter vector (`n_free`).
# ─────────────────────────────────────────────────────────────────────────────
function _has_limits_internal(params::Parameters)
    n_active = n_free(params)
    out = Vector{Bool}(undef, n_active)
    @inbounds for int_idx in 1:n_active
        p = params.pars[params.ext_of_int[int_idx]]
        out[int_idx] = has_lower_limit(p) || has_upper_limit(p)
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build (ext_values, ext_errors, ext_covariance) from an internal-coord
# FunctionMinimum + the Parameters describing the bound structure.
#
# Shared by `migrad(cf, params)` (Phase 1.x) and `hesse(m::Minuit)` (Phase 3).
# Both need the same int→ext transform machinery: Jacobian chain rule on the
# covariance, two-sided `Int2extError` for the bounded diagonals.
# ─────────────────────────────────────────────────────────────────────────────

function _internal_to_external_results(
    fmin_int::FunctionMinimum,
    params::Parameters,
    up::Float64,
)
    n_total  = n_pars(params)
    n_active = n_free(params)
    int_x    = fmin_int.state.parameters.x

    ext_values     = Vector{Float64}(undef, n_total)
    ext_errors_vec = zeros(Float64, n_total)
    ext_cov_mat    = nothing

    # Build the full external parameter vector (fixed params keep
    # their initial values; free params come from the internal state).
    @inbounds for ext_idx in 1:n_total
        par = params.pars[ext_idx]
        int_idx = params.int_of_ext[ext_idx]
        if int_idx == 0
            ext_values[ext_idx] = par.value
        else
            ext_values[ext_idx] = int_to_ext_value(params, int_idx, int_x[int_idx])
        end
    end

    if has_covariance(fmin_int)
        # Read symmetrically via the Symmetric{:U} wrapper, NOT through
        # `parent(...)` (parallel-review #4 D3 blocking — using `parent`
        # gives only the upper-triangle storage; lower-triangle reads
        # return uninitialized zeros, producing an asymmetric external
        # covariance matrix).
        V_int = fmin_int.state.error.inv_hessian  # Symmetric{:U} view

        # Jacobian d(ext)/d(int) per free parameter
        dint2ext_diag = Vector{Float64}(undef, n_active)
        @inbounds for int_idx in 1:n_active
            dint2ext_diag[int_idx] = dint2ext_value(params, int_idx, int_x[int_idx])
        end

        # External covariance for the FREE parameters: C_ext = D · C_int · D
        # where D = diag(dint2ext) and C_int = 2·up·V_int (per C++; see
        # result.jl::covariance).
        c_int_scale = 2.0 * up
        cov_free = zeros(Float64, n_active, n_active)
        @inbounds for i in 1:n_active, j in 1:n_active
            cov_free[i, j] = c_int_scale * V_int[i, j] *
                              dint2ext_diag[i] * dint2ext_diag[j]
        end

        # Set diagonal external errors. For unbounded parameters the
        # Jacobian-diagonal sqrt(cov_free[i,i]) is exact. For bounded
        # parameters near the boundary the Jacobian shrinks (the sin
        # transform's derivative goes to 0 at the limit) — use the C++
        # `Int2extError` two-sided formula instead (parallel-review #4 D5).
        @inbounds for int_idx in 1:n_active
            ext_idx = params.ext_of_int[int_idx]
            par = params.pars[ext_idx]
            if has_limits(par) || has_upper_limit(par) || has_lower_limit(par)
                kind = bound_kind(par.lower, par.upper)
                int_err = sqrt(max(V_int[int_idx, int_idx], 0.0))
                ext_errors_vec[ext_idx] = int2ext_error(
                    kind, int_x[int_idx], int_err, par.lower, par.upper)
            else
                ext_errors_vec[ext_idx] = sqrt(max(cov_free[int_idx, int_idx], 0.0))
            end
        end

        # Promote to n_total × n_total covariance (fixed params get 0 row/col)
        cov_full = zeros(Float64, n_total, n_total)
        @inbounds for i in 1:n_active, j in 1:n_active
            ei = params.ext_of_int[i]
            ej = params.ext_of_int[j]
            cov_full[ei, ej] = cov_free[i, j]
        end
        ext_cov_mat = cov_full
    else
        # No covariance available — fall back to user's initial errors
        @inbounds for ext_idx in 1:n_total
            par = params.pars[ext_idx]
            ext_errors_vec[ext_idx] = is_fixed(par) ? 0.0 : par.error
        end
    end

    return ext_values, ext_errors_vec, ext_cov_mat
end

# ─────────────────────────────────────────────────────────────────────────────
# Main: bound-aware migrad
# ─────────────────────────────────────────────────────────────────────────────

"""
    migrad(cf::CostFunction, params::Parameters;
           strategy=Strategy(0), tol=0.1, maxfcn=nothing,
           prec=MachinePrecision()) -> BoundedFunctionMinimum

Bound- and fixed-parameter-aware MIGRAD. User FCN receives external
parameter values; internal MIGRAD sees only the free parameters in
internal (unbounded) coordinates.

Phase 1 first cut. Mirrors the C++ Minuit2 user flow but with the
Parameters/Transformation wired in.

# Arguments

- `cf::CostFunction` — wraps the user FCN on EXTERNAL parameter values.
- `params::Parameters` — parameter metadata: values, errors, bounds,
  fixed flags, names.

# Keyword arguments

- `strategy::Strategy=Strategy(0)` — Strategy level; 0/1/2 all supported
  (the `Minuit` front end defaults to `Strategy(1)`).
- `tol::Real=0.1` — convergence tolerance.
- `maxfcn::Union{Integer,Nothing}=nothing` — defaults to the standard
  `200 + 100·n + 5·n²` where n is the number of free parameters.
- `prec::MachinePrecision`.

# Returns

[`BoundedFunctionMinimum`](@ref) with `.ext_values` and `.ext_errors`
in EXTERNAL coordinates.
"""
function migrad(
    cf::CostFunction,
    params::Parameters;
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
    threaded_gradient::Bool = false,
    verify_threading::Bool = threaded_gradient,
    prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    print_level::Integer = 0,
)
    n_total = n_pars(params)
    n_active = n_free(params)
    n_active > 0 ||
        throw(ArgumentError("migrad needs at least one free parameter"))

    # Initial internal values + errors via the C++-faithful Taylor/
    # two-sided transformations (parameters.jl initial_int_*).
    int_vals = initial_int_values(params)
    int_errs = initial_int_errors(params)

    # Wrap the user FCN to accept internal coords
    cf_internal = _wrap_fcn_internal_to_external(cf, params)

    # Run internal MIGRAD. `prior_cov` (when supplied) is already in
    # INTERNAL coordinates — callers (e.g. `migrad!` retry loop) extract
    # it from a prior `bfm.internal.state.error.inv_hessian`, which the
    # internal MIGRAD produced. No further coordinate transform needed.
    fmin_int = migrad(cf_internal, int_vals, int_errs;
                       strategy = strategy, tol = tol, maxfcn = maxfcn,
                       prec = prec,
                       threaded_gradient = threaded_gradient,
                       verify_threading = verify_threading,
                       prior_cov = prior_cov,
                       has_limits = _has_limits_internal(params),
                       print_level = print_level)

    # ── Convert internal results back to external ────────────────
    ext_values, ext_errors_vec, ext_cov_mat =
        _internal_to_external_results(fmin_int, params, cf.up)

    return BoundedFunctionMinimum(
        fmin_int, params, ext_values, ext_errors_vec, ext_cov_mat,
        cf_internal,
    )
end

"""
    migrad(cf::CostFunctionWithGradient, params::Parameters; ...) ->
        BoundedFunctionMinimum

AD-aware bounded MIGRAD. The user FCN + gradient operate on EXTERNAL
parameters; this overload threads the int↔ext chain rule into the
gradient before handing both to the unbounded `migrad(cf::CFwG, ...)`
path. The 5-10× FCN-call savings of analytical gradients carry through
to bounded fits.

Phase F: previously this stored a *plain* `CostFunction` view of the
internal FCN in `BoundedFunctionMinimum.internal_cf`, which dropped the
analytical gradient on follow-up MINOS / contour calls — codex review
caught the regression. The internal CF is now stored as the FULL
`CostFunctionWithGradient` so the AD path survives MIGRAD → MINOS /
contour transitions (the downstream `function_cross[_multi]` / `minos`
/ `contour_exact` signatures already accept `AbstractCostFunction`).
"""
function migrad(
    cf::CostFunctionWithGradient,
    params::Parameters;
    strategy::Strategy = Strategy(0),
    tol::Real = 0.1,
    maxfcn::Union{Integer,Nothing} = nothing,
    prec::MachinePrecision = MachinePrecision(),
    threaded_gradient::Bool = false,
    verify_threading::Bool = false,
    prior_cov::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    print_level::Integer = 0,
)
    n_total = n_pars(params)
    n_active = n_free(params)
    n_active > 0 ||
        throw(ArgumentError("migrad needs at least one free parameter"))

    int_vals = initial_int_values(params)
    int_errs = initial_int_errors(params)

    cf_internal_grad = _wrap_fcn_internal_to_external(cf, params)

    # Internal MIGRAD dispatches to the CFwG path → uses analytical gradient.
    # threaded_gradient + verify_threading are no-ops for AD path but
    # accepted for API symmetry. `prior_cov` is in INTERNAL coordinates
    # (see plain-CF overload comment above).
    fmin_int = migrad(cf_internal_grad, int_vals, int_errs;
                       strategy = strategy, tol = tol, maxfcn = maxfcn,
                       prec = prec,
                       threaded_gradient = threaded_gradient,
                       verify_threading = verify_threading,
                       prior_cov = prior_cov,
                       has_limits = _has_limits_internal(params),
                       print_level = print_level)

    ext_values, ext_errors_vec, ext_cov_mat =
        _internal_to_external_results(fmin_int, params, cf.up)

    return BoundedFunctionMinimum(
        fmin_int, params, ext_values, ext_errors_vec, ext_cov_mat,
        cf_internal_grad,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", m::BoundedFunctionMinimum)
    println(io, "JuMinuit BoundedFunctionMinimum")
    println(io, "  valid:   ", is_valid(m))
    println(io, "  fval:    ", fval(m))
    println(io, "  edm:     ", edm(m))
    println(io, "  nfcn:    ", nfcn(m))
    println(io, "  parameters (external):")
    for ext_idx in 1:n_pars(m.params)
        par = m.params.pars[ext_idx]
        val = m.ext_values[ext_idx]
        err = m.ext_errors[ext_idx]
        fixed_tag = is_fixed(par) ? "  [FIXED]" : ""
        bounds = if has_limits(par)
            "  [$(par.lower), $(par.upper)]"
        elseif has_upper_limit(par)
            "  (-∞, $(par.upper)]"
        elseif has_lower_limit(par)
            "  [$(par.lower), ∞)"
        else
            ""
        end
        if is_fixed(par)
            println(io, "    [", ext_idx, "] ", par.name, " = ", val,
                    fixed_tag, bounds)
        else
            println(io, "    [", ext_idx, "] ", par.name, " = ", val,
                    " ± ", err, bounds)
        end
    end
end

Base.show(io::IO, m::BoundedFunctionMinimum) =
    print(io, "BoundedFunctionMinimum(fval=", fval(m),
              ", valid=", is_valid(m), ", n=", n_pars(m.params), ")")
