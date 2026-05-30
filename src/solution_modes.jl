# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# solution_modes.jl — multi-modal solution detection (BEYOND iminuit).
#
# Problem. When a Δχ² acceptance region is sampled (see the MC-Δχ² sampling
# layer, `get_contours_samples`), the accepted parameter-vector set is NOT
# always one connected blob. Widen the sampling range and it can break into a
# main cluster PLUS one or more separated regions (possibly only a few points).
# Each separated region is a DISTINCT solution: its χ² is within Δχ² of the
# global best (statistically acceptable) but its parameters — and therefore its
# PHYSICS — are different. Such modes must be reported and treated
# independently, never merged into a single error bar. iminuit/C++ Minuit2 have
# no auto-detection of this; this file adds it.
#
#   Physics motivation: the X(3872) `J/ψρ + DD̄*` line-shape fit
#   (arXiv:2404.12003, "Dip versus peak") admits several physically distinct
#   local minima (different scattering-length combinations) with comparable χ².
#   A single MINOS/HESSE error bar around the global best hides them.
#
# Method. Cluster the accepted samples into disconnected modes, then optionally
# re-fit (MIGRAD) from each cluster to recover that mode's true local minimum
# and its own errors.
#
# ── CRITICAL: the distance metric must be WHITENED (error-normalized) ──
# Fit parameters span wildly different scales (LECs ~1e-3 next to couplings
# ~1). A naive Euclidean distance is dominated by the largest-scale parameter,
# so two modes separated only in a tiny-scale parameter look identical (their
# Euclidean separation is swamped by the within-mode spread of the big-scale
# parameter) — clustering then WRONGLY merges them. We therefore cluster in
# WHITENED coordinates:
#
#   • whiten=:cov     full Mahalanobis  z = L⁻¹·x  (Σ_free = L·Lᵀ, Cholesky).
#                     Decorrelates AND rescales using the fit covariance — the
#                     statistically correct metric. Pairwise distance is then
#                     d(i,j)² = (xᵢ−xⱼ)ᵀ Σ⁻¹ (xᵢ−xⱼ), i.e. separation in σ.
#   • whiten=:errors  per-parameter scale only, z_k = x_k / σ_k. Cheaper, no
#                     matrix inversion, ignores correlations. Robust fallback.
#
# Both make distances dimensionless "number of σ", so a fixed `threshold`
# (default 1σ) is physically meaningful and scale-invariant.
#
# Clustering backends.
#   • method=:components (default, no dependency): single-linkage connected
#     components in whitened space — union points whose pairwise whitened
#     distance is ≤ `threshold`, then take connected components. `min_size`
#     separates real sparse modes from stray noise points.
#   • method=:dbscan (optional): density-based, handles arbitrary shapes +
#     outliers. Provided by `ext/JuMinuitClusteringExt.jl` (weakdep
#     `Clustering.jl`); errors with a helpful message if Clustering isn't
#     loaded.
#
# Complexity. The built-in connected-components clusterer is O(N²·d) in the
# number of samples N (all-pairs). That is fine for the hundreds-to-few-
# thousand samples a Δχ² scan produces; for very large N prefer
# `method=:dbscan` (Clustering.jl uses a spatial tree → ~O(N·log N)).
# ─────────────────────────────────────────────────────────────────────────────

"""
    SolutionMode

One distinct solution found by [`find_solution_modes`](@ref): a connected
cluster of Δχ²-accepted samples that is separated (in whitened parameter
space) from the other clusters.

# Fields

- `index::Int` — rank, 1 = main solution (lowest χ²), 2, 3, … by ascending χ².
- `representative::Vector{Float64}` — the minimum-χ² sample in the cluster
  (full external parameter vector, length = total parameter count).
- `fval::Float64` — χ² (cost) at `representative`.
- `delta_fval::Float64` — `fval` minus the global-best χ². `≥ 0` for every
  mode when the global best is the fit minimum; the main mode is `≈ 0`.
  Interpret against the fit's `errordef` (`up`): a mode with
  `delta_fval ≲ up` is statistically comparable to the main solution.
- `param_ranges::Vector{Tuple{Float64,Float64}}` — per-parameter `(min, max)`
  over the cluster's samples (full external coordinates).
- `n_points::Int` — number of samples in the cluster.
- `fraction::Float64` — `n_points` / total accepted samples.
- `member_indices::Vector{Int}` — row indices (into the `samples` matrix) of
  this cluster's members.

Re-fit fields, populated only when `find_solution_modes(...; refine=true)`:

- `refined::Bool` — whether a per-mode MIGRAD re-fit was run and succeeded.
- `refined_values::Vector{Float64}` — re-fit parameter values (empty if not
  refined).
- `refined_errors::Vector{Float64}` — re-fit errors (empty if not refined).
- `refined_fval::Float64` — re-fit χ² (`NaN` if not refined).
- `refined_valid::Bool` — whether the re-fit MIGRAD validated.
- `refined_nfcn::Int` — FCN calls used by the re-fit.
- `new_min::Bool` — **`true` if this mode's re-fit reached a DEEPER minimum
  than the global best** — i.e. the main fit missed the better basin. Flagged
  prominently in the report; connects to the IAM cold-start convergence gap
  (a separated cluster can be the basin the global fit failed to find).
"""
struct SolutionMode
    index::Int
    representative::Vector{Float64}
    fval::Float64
    delta_fval::Float64
    param_ranges::Vector{Tuple{Float64,Float64}}
    n_points::Int
    fraction::Float64
    member_indices::Vector{Int}
    # ── refine=true outputs ──
    refined::Bool
    refined_values::Vector{Float64}
    refined_errors::Vector{Float64}
    refined_fval::Float64
    refined_valid::Bool
    refined_nfcn::Int
    new_min::Bool
end

"""
    SolutionModes <: AbstractVector{SolutionMode}

Result of [`find_solution_modes`](@ref): an indexable, iterable vector of
[`SolutionMode`](@ref)s (so `modes[1]`, `length(modes)`, `for s in modes`
all work) plus the metadata needed for the summary report. Pretty-prints a
"Found K distinct solutions within Δχ²" table; index it to get the individual
modes.

# Fields (besides the modes themselves)

- `global_best::Float64` — reference χ² the `delta_fval`s are measured from.
- `up::Float64` — the fit's `errordef` (1.0 χ², 0.5 NLL) — the natural Δχ²
  yardstick.
- `whiten::Symbol` — the metric actually used (`:cov` or `:errors`; may differ
  from the request if `:cov` fell back to `:errors`).
- `method::Symbol` — `:components` or `:dbscan`.
- `threshold::Float64` — whitened connection radius (σ units).
- `n_noise::Int` — samples dropped as noise (clusters smaller than `min_size`).
- `n_samples::Int` — total accepted samples clustered.
- `param_names::Vector{String}` — parameter names (for the report).
"""
struct SolutionModes <: AbstractVector{SolutionMode}
    modes::Vector{SolutionMode}
    global_best::Float64
    up::Float64
    whiten::Symbol
    method::Symbol
    threshold::Float64
    n_noise::Int
    n_samples::Int
    param_names::Vector{String}
end

Base.size(s::SolutionModes) = size(s.modes)
Base.getindex(s::SolutionModes, i::Int) = s.modes[i]
Base.IndexStyle(::Type{SolutionModes}) = IndexLinear()

# ─────────────────────────────────────────────────────────────────────────────
# Whitening
# ─────────────────────────────────────────────────────────────────────────────

# Build the whitened sample matrix `Z` (d × N, columns = points) from the
# free-parameter sub-samples `Xfree` (N × d). For :cov, `scale` is the
# d×d free covariance Σ; for :errors it's the length-d error vector σ.
# Returns Z. May throw PosDefException for :cov (caller handles fallback).
function _whiten_samples(Xfree::AbstractMatrix{Float64}, scale, whiten::Symbol)
    N, d = size(Xfree)
    if whiten === :cov
        Σ = scale
        # Σ = L·Lᵀ ; z = L⁻¹·x ⇒ ‖zᵢ−zⱼ‖² = (xᵢ−xⱼ)ᵀ Σ⁻¹ (xᵢ−xⱼ) (Mahalanobis).
        C = cholesky(Symmetric(Matrix(Σ)))
        Linv = inv(LowerTriangular(C.L))          # d×d lower-triangular inverse
        return Linv * permutedims(Xfree)          # (d×d)·(d×N) = d×N
    elseif whiten === :errors
        σ = scale
        Z = Matrix{Float64}(undef, d, N)
        @inbounds for k in 1:d
            sk = σ[k]
            inv_sk = (sk > 0 && isfinite(sk)) ? 1.0 / sk : 0.0  # degenerate dim → 0 contribution
            for i in 1:N
                Z[k, i] = Xfree[i, k] * inv_sk
            end
        end
        return Z
    else
        throw(ArgumentError("whiten must be :cov or :errors, got :$whiten"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Built-in single-linkage connected-components clusterer (whitened space)
# ─────────────────────────────────────────────────────────────────────────────

# Union-find with path compression.
@inline function _uf_find!(parent::Vector{Int}, x::Int)
    root = x
    @inbounds while parent[root] != root
        root = parent[root]
    end
    @inbounds while parent[x] != root        # path compression
        parent[x], x = root, parent[x]
    end
    return root
end

"""
    _connected_components(Z, threshold) -> (labels, nraw)

Single-linkage connected components on the whitened sample matrix `Z`
(`d × N`, columns = points): union any two points whose Euclidean distance in
whitened space is `≤ threshold`, then label connected components `1:nraw`.
`labels[i]` is point `i`'s raw component id. Internal; exposed for tests
(e.g. to demonstrate that NAIVE/unwhitened coordinates merge distinct modes).
"""
function _connected_components(Z::AbstractMatrix{<:Real}, threshold::Real)
    d, N = size(Z)
    parent = collect(1:N)
    thr2 = Float64(threshold)^2
    @inbounds for i in 1:(N - 1)
        ri = _uf_find!(parent, i)
        for j in (i + 1):N
            # squared whitened distance, with early exit once it exceeds thr².
            s = 0.0
            for k in 1:d
                δ = Float64(Z[k, i]) - Float64(Z[k, j])
                s += δ * δ
                s > thr2 && break
            end
            if s <= thr2
                rj = _uf_find!(parent, j)
                if rj != ri
                    parent[rj] = ri          # attach j's root under i's root
                end
            end
        end
    end
    # Compact roots into dense ids 1:nraw.
    labels = Vector{Int}(undef, N)
    rootmap = Dict{Int,Int}()
    nraw = 0
    @inbounds for i in 1:N
        r = _uf_find!(parent, i)
        id = get(rootmap, r, 0)
        if id == 0
            nraw += 1
            id = nraw
            rootmap[r] = id
        end
        labels[i] = id
    end
    return labels, nraw
end

# Demote raw components with < min_size members to noise (label 0) and
# re-pack the survivors into dense ids 1:k. Returns (labels, k, n_noise).
function _apply_min_size(labels::Vector{Int}, nraw::Int, min_size::Int)
    counts = zeros(Int, nraw)
    @inbounds for l in labels
        counts[l] += 1
    end
    remap = zeros(Int, nraw)
    k = 0
    for raw in 1:nraw
        if counts[raw] >= min_size
            k += 1
            remap[raw] = k
        end
    end
    out = Vector{Int}(undef, length(labels))
    n_noise = 0
    @inbounds for i in eachindex(labels)
        new = remap[labels[i]]
        out[i] = new
        new == 0 && (n_noise += 1)
    end
    return out, k, n_noise
end

# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN backend stub — concrete method added by ext/JuMinuitClusteringExt.jl
# when `using Clustering` is loaded. Calling :dbscan without it raises a
# helpful error (see find_solution_modes). Signature:
#   _dbscan_labels(Z::Matrix{Float64}, radius, min_neighbors, min_cluster_size)
#     -> labels::Vector{Int}   (0 = noise)
# ─────────────────────────────────────────────────────────────────────────────
function _dbscan_labels end

# ─────────────────────────────────────────────────────────────────────────────
# χ² evaluation at the accepted samples
# ─────────────────────────────────────────────────────────────────────────────

# Evaluate the user FCN at every sample (full external vectors). Calls the
# raw `m.fcn.f` (not the counting call operator) so the fit's nfcn is not
# polluted. Parallelized across samples when `parallel` (same Phase-G/H
# thread-safety contract as threaded gradients).
function _eval_fvals(m::Minuit, Xfull::Matrix{Float64}, parallel::Bool)
    N = size(Xfull, 1)
    f = m.fcn.f
    out = Vector{Float64}(undef, N)
    if parallel
        Threads.@threads :static for i in 1:N
            @inbounds out[i] = Float64(f(@view Xfull[i, :]))
        end
    else
        @inbounds for i in 1:N
            out[i] = Float64(f(@view Xfull[i, :]))
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-mode re-fit (refine=true)
# ─────────────────────────────────────────────────────────────────────────────

# Re-run MIGRAD from a cluster's representative point, preserving the parent
# fit's cost function, gradient, limits, fixed-parameter structure, errordef,
# strategy and tolerance. Threading is forced OFF on the inner fit so that, when
# the modes are refined in parallel, we don't oversubscribe threads (the
# parallelism is ACROSS modes). Returns a NamedTuple of re-fit results; any
# failure is caught and reported as `valid=false` rather than aborting the
# whole call.
function _refine_mode(m::Minuit, rep_full::Vector{Float64},
                       global_best::Float64, iterate::Int)
    empty_vec = Float64[]
    try
        grad = m.cfwg === nothing ? nothing : m.cfwg.g
        mm = grad === nothing ?
            Minuit(m.fcn.f, m; threaded_gradient = false, verify_threading = false) :
            Minuit(m.fcn.f, m; grad = grad,
                   threaded_gradient = false, verify_threading = false)
        # Start from the cluster representative, but PIN fixed parameters at the
        # fit's value: a fixed parameter must not be re-anchored to whatever the
        # sample happened to carry in that column (defensive against full-width
        # input whose fixed column is not held constant).
        start = copy(rep_full)
        @inbounds for j in eachindex(start)
            if is_fixed(m.params.pars[j])
                start[j] = m.params.pars[j].value
            end
        end
        mm.values = start
        migrad!(mm; iterate = iterate)
        valid = mm.valid
        rv = Float64[mm.values[i] for i in 1:mm.ndim]
        re = Float64[mm.errors[i] for i in 1:mm.ndim]
        rf = mm.fval
        nf = mm.nfcn
        # Deeper basin: a valid re-fit strictly below the global best (with a
        # small absolute+relative tolerance so numerical jitter at the same
        # minimum is NOT flagged).
        tol = 1e-7 * max(1.0, abs(global_best))
        new_min = valid && isfinite(rf) && (rf < global_best - tol)
        return (refined = true, values = rv, errors = re, fval = rf,
                valid = valid, nfcn = nf, new_min = new_min)
    catch err
        @warn "find_solution_modes: per-mode re-fit failed; reporting cluster without refinement" exception = err
        return (refined = false, values = empty_vec, errors = empty_vec,
                fval = NaN, valid = false, nfcn = 0, new_min = false)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

"""
    find_solution_modes(samples, m::Minuit; whiten=:cov, method=:components,
                        threshold=1.0, min_size=1, min_neighbors=1,
                        fvals=nothing, refine=false, refine_iterate=5,
                        parallel=nothing) -> SolutionModes

Cluster a set of Δχ²-accepted parameter samples into DISTINCT solution modes —
a "beyond iminuit" capability for multi-modal posteriors. See the file header
and `docs/src/error_analysis.md` for the full rationale.

# Arguments

- `samples::AbstractMatrix` — accepted samples, **one parameter vector per
  row**. Width must be either the total parameter count (`m.ndim`, full
  external vectors — what the MC-Δχ² sampler `get_contours_samples` produces)
  or the free-parameter count (`m.npar`; fixed parameters are filled from the
  fit and the clustering ignores them either way).
- `m::Minuit` — the converged fit the samples were drawn around. Supplies the
  whitening metric (covariance / errors), the cost function, and — for
  `refine` — the full re-fit configuration.

# Keyword arguments

- `whiten::Symbol=:cov` — distance metric. `:cov` = full Mahalanobis using the
  fit's free-parameter covariance (decorrelating + rescaling; the correct
  metric); `:errors` = per-parameter `x/σ` scaling (no correlations, cheaper,
  robust). **Naive unwhitened Euclidean is intentionally not offered** — it is
  dominated by the largest-scale parameter and merges modes that differ only in
  a small-scale parameter. `:cov` falls back to `:errors` (with a warning) if
  the covariance is unavailable or not positive-definite.
- `method::Symbol=:components` — `:components` (built-in single-linkage
  connected components, no dependency) or `:dbscan` (density-based; requires
  `using Clustering`).
- `threshold::Real=1.0` — connection radius in WHITENED units (σ). Two samples
  closer than this (Mahalanobis/scaled distance) are linked. Smaller = stricter
  (won't bridge distinct modes, but may split a sparsely-sampled one); larger
  risks single-linkage chaining across a gap. For `:dbscan` this is the ε
  radius.
- `min_size::Integer=1` — clusters with fewer members are dropped as noise.
  Default `1` keeps every separated region (the accepted set is already χ²-
  filtered, so even a few-point region is a candidate solution); raise it to
  suppress scatter.
- `min_neighbors::Integer=1` — `:dbscan` core-point density (ignored by
  `:components`).
- `fvals::Union{Nothing,AbstractVector}=nothing` — precomputed χ² per sample
  (e.g. the values the sampler already kept). If `nothing`, the FCN is
  evaluated at each sample.
- `refine::Bool=false` — re-run MIGRAD from each cluster's representative to
  recover that mode's true local minimum + its own errors. Flags `new_min` if a
  mode is DEEPER than the global best (the main fit missed the better basin).
- `refine_iterate::Integer=5` — `iterate` passed to each re-fit `migrad!`.
- `parallel::Union{Bool,Nothing}=nothing` — parallelize FCN evaluation and
  per-mode re-fits across threads. `nothing` ⇒ auto-on when
  `m.threaded_gradient` is set and `Threads.nthreads() > 1` (the same FCN
  thread-safety contract as Phase G/H threaded gradients). Set `true`/`false`
  to force.

# Returns

A [`SolutionModes`](@ref) (an `AbstractVector{SolutionMode}`), sorted with the
main solution (lowest χ²) first. Pretty-prints a summary report.

# Example

```julia
m = Minuit(chi2, x0; names=pnames); migrad!(m)
samples = get_contours_samples(m; ...)          # MC-Δχ² accepted set (rows = vectors)
modes = find_solution_modes(samples, m)          # cluster into distinct solutions
length(modes) > 1 && @warn "multi-modal: \$(length(modes)) distinct solutions"
modes_refined = find_solution_modes(samples, m; refine=true)   # + per-mode re-fit
```
"""
function find_solution_modes(samples::AbstractMatrix, m::Minuit;
        whiten::Symbol = :cov,
        method::Symbol = :components,
        threshold::Real = 1.0,
        min_size::Integer = 1,
        min_neighbors::Integer = 1,
        fvals::Union{Nothing,AbstractVector} = nothing,
        refine::Bool = false,
        refine_iterate::Integer = 5,
        parallel::Union{Bool,Nothing} = nothing,
    )
    ndim = m.ndim
    npar = m.npar
    free_idx = [i for i in 1:ndim if !is_fixed(m.params.pars[i])]
    nfree = length(free_idx)
    nfree >= 1 ||
        throw(ArgumentError("find_solution_modes: fit has no free parameters to cluster on"))

    # ── Resolve sample orientation → full external matrix Xfull (N × ndim) ──
    N, ncol = size(samples)
    N >= 1 ||
        throw(ArgumentError("find_solution_modes: `samples` has no rows"))
    Xfull = if ncol == ndim
        Matrix{Float64}(samples)
    elseif ncol == nfree && nfree != ndim
        # free-only vectors → splice in the fixed-parameter values from the fit
        fixed_vals = Float64[p.value for p in m.params.pars]
        Xf = Matrix{Float64}(undef, N, ndim)
        @inbounds for i in 1:N
            for j in 1:ndim
                Xf[i, j] = fixed_vals[j]
            end
            for (k, fi) in enumerate(free_idx)
                Xf[i, fi] = Float64(samples[i, k])
            end
        end
        Xf
    else
        throw(ArgumentError(
            "find_solution_modes: each `samples` row must be a parameter vector of " *
            "length ndim=$ndim (full) or npar=$nfree (free-only), got width $ncol. " *
            "Note: rows = samples, columns = parameters."))
    end

    threshold > 0 || throw(ArgumentError("threshold must be > 0, got $threshold"))
    min_size >= 1 || throw(ArgumentError("min_size must be ≥ 1, got $min_size"))
    !(refine && refine_iterate < 1) ||
        throw(ArgumentError("refine_iterate must be ≥ 1, got $refine_iterate"))

    Xfree_mat = Matrix{Float64}(Xfull[:, free_idx])

    # ── Whitening scale from the fit; :cov → :errors fallback if needed ──
    whiten === :cov || whiten === :errors ||
        throw(ArgumentError("whiten must be :cov or :errors, got :$whiten"))
    eff_whiten = whiten
    errs_full = Float64[m.errors[i] for i in 1:ndim]
    σ_free = errs_full[free_idx]
    Σ = whiten === :cov ? matrix(m; skip_fixed = true) : nothing   # compute once
    if whiten === :cov && Σ === nothing
        @warn "find_solution_modes: whiten=:cov requested but the fit has no " *
              "covariance (run migrad! first?). Falling back to whiten=:errors."
        eff_whiten = :errors
    end

    Z = if eff_whiten === :cov
        try
            _whiten_samples(Xfree_mat, Σ, :cov)
        catch err
            err isa LinearAlgebra.PosDefException || rethrow()
            @warn "find_solution_modes: free covariance is not positive-definite; " *
                  "falling back to whiten=:errors."
            eff_whiten = :errors
            _whiten_samples(Xfree_mat, σ_free, :errors)
        end
    else
        _whiten_samples(Xfree_mat, σ_free, :errors)
    end

    # ── Cluster in whitened space ──
    do_parallel = parallel === nothing ?
        (_use_threads(m) && Threads.nthreads() > 1) : parallel
    labels, n_noise = if method === :components
        raw, nraw = _connected_components(Z, threshold)
        l, _, nn = _apply_min_size(raw, nraw, Int(min_size))
        l, nn
    elseif method === :dbscan
        if isempty(methods(_dbscan_labels))
            throw(ArgumentError(
                "find_solution_modes: method=:dbscan requires the Clustering.jl " *
                "extension — run `using Clustering` (alongside `using JuMinuit`) " *
                "first, or use method=:components (the built-in, no-dependency " *
                "clusterer)."))
        end
        l = _dbscan_labels(Matrix{Float64}(Z), Float64(threshold),
                           Int(min_neighbors), Int(min_size))
        length(l) == N ||
            error("_dbscan_labels returned $(length(l)) labels for $N samples")
        l, count(==(0), l)
    else
        throw(ArgumentError("method must be :components or :dbscan, got :$method"))
    end
    K = isempty(labels) ? 0 : maximum(labels)

    # ── χ² at every sample (for representatives + Δχ²) ──
    fv = if fvals === nothing
        _eval_fvals(m, Xfull, do_parallel)
    else
        length(fvals) == N ||
            throw(ArgumentError("fvals length $(length(fvals)) ≠ number of samples $N"))
        Float64.(fvals)
    end

    # Global-best reference: the fit minimum if available, else the lowest
    # sample χ². Used as the Δχ² zero-point.
    global_best = isnan(m.fval) ? (isempty(fv) ? NaN : minimum(fv)) : m.fval

    # ── Build a preliminary record per surviving cluster ──
    members = [Int[] for _ in 1:K]
    @inbounds for i in 1:N
        l = labels[i]
        l == 0 && continue
        push!(members[l], i)
    end

    prelim = Vector{NamedTuple}(undef, K)
    for c in 1:K
        idxs = members[c]
        # representative = minimum-χ² sample in the cluster
        best_local = idxs[1]
        for i in idxs
            fv[i] < fv[best_local] && (best_local = i)
        end
        rep = Float64[Xfull[best_local, j] for j in 1:ndim]
        ranges = Vector{Tuple{Float64,Float64}}(undef, ndim)
        @inbounds for j in 1:ndim
            lo = Xfull[idxs[1], j]; hi = lo
            for i in idxs
                v = Xfull[i, j]
                v < lo && (lo = v)
                v > hi && (hi = v)
            end
            ranges[j] = (lo, hi)
        end
        prelim[c] = (rep = rep, fval = fv[best_local], ranges = ranges,
                     n = length(idxs), members = idxs)
    end

    # Sort clusters by χ² ascending (main solution first).
    order = sortperm([p.fval for p in prelim])

    # ── Optional per-mode re-fit (parallel-aware) ──
    refine_results = Vector{Any}(undef, K)
    if refine && K > 0
        if do_parallel && K > 1
            Threads.@threads :static for oi in 1:K
                c = order[oi]
                refine_results[c] = _refine_mode(m, prelim[c].rep,
                                                  global_best, Int(refine_iterate))
            end
        else
            for c in 1:K
                refine_results[c] = _refine_mode(m, prelim[c].rep,
                                                  global_best, Int(refine_iterate))
            end
        end
    end

    # ── Assemble sorted, indexed SolutionModes ──
    modes = Vector{SolutionMode}(undef, K)
    for (rank, c) in enumerate(order)
        p = prelim[c]
        delta = p.fval - global_best
        if refine && K > 0
            r = refine_results[c]
            modes[rank] = SolutionMode(rank, p.rep, p.fval, delta, p.ranges,
                                       p.n, p.n / N, p.members,
                                       r.refined, r.values, r.errors, r.fval,
                                       r.valid, r.nfcn, r.new_min)
        else
            modes[rank] = SolutionMode(rank, p.rep, p.fval, delta, p.ranges,
                                       p.n, p.n / N, p.members,
                                       false, Float64[], Float64[], NaN,
                                       false, 0, false)
        end
    end

    names = [p.name for p in m.params.pars]
    return SolutionModes(modes, global_best, m.up, eff_whiten, method,
                         Float64(threshold), n_noise, N, names)
end

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

# Compact one-line vector preview (avoid dumping 50-parameter vectors).
function _preview_vec(v::AbstractVector{<:Real}; maxn::Int = 6)
    n = length(v)
    if n <= maxn
        return "[" * join((@sprintf("%.4g", x) for x in v), ", ") * "]"
    else
        head = join((@sprintf("%.4g", v[i]) for i in 1:maxn), ", ")
        return "[" * head * ", … ($(n - maxn) more)]"
    end
end

function Base.show(io::IO, s::SolutionMode)
    tag = s.index == 1 ? "main" : "mode $(s.index)"
    if s.index == 1
        @printf(io, "SolutionMode[%s]: %d pts (%.1f%%) χ²=%.6g rep=%s",
                tag, s.n_points, 100 * s.fraction, s.fval, _preview_vec(s.representative))
    else
        @printf(io, "SolutionMode[%s]: %d pts (%.1f%%) χ²=%.6g Δχ²=%.4g rep=%s",
                tag, s.n_points, 100 * s.fraction, s.fval, s.delta_fval,
                _preview_vec(s.representative))
    end
    if s.refined
        flag = s.new_min ? " ⚠DEEPER-MIN" : ""
        @printf(io, " | refit χ²=%.6g%s%s", s.refined_fval,
                s.refined_valid ? "" : " (invalid)", flag)
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::SolutionModes)
    K = length(s.modes)
    whiten_desc = s.whiten === :cov ? ":cov (Mahalanobis)" : ":errors (per-σ scale)"
    println(io, "SolutionModes: $K distinct solution(s) from $(s.n_samples) accepted sample(s)")
    @printf(io, "  metric: whiten=%s  method=:%s  threshold=%.3g σ  errordef(up)=%.3g\n",
            whiten_desc, s.method, s.threshold, s.up)
    if K == 0
        println(io, "  (no clusters survived min_size; all samples classified as noise)")
        return
    end
    any_refined = any(x -> x.refined, s.modes)
    any_newmin = any(x -> x.new_min, s.modes)
    for sm in s.modes
        tag = sm.index == 1 ? "main  " : @sprintf("mode %d", sm.index)
        if sm.index == 1
            @printf(io, "  [%d] %s: %5d pts (%5.1f%%)  χ²=%-12.6g            rep=%s\n",
                    sm.index, tag, sm.n_points, 100 * sm.fraction, sm.fval,
                    _preview_vec(sm.representative))
        else
            @printf(io, "  [%d] %s: %5d pts (%5.1f%%)  χ²=%-12.6g Δχ²=%-9.4g rep=%s\n",
                    sm.index, tag, sm.n_points, 100 * sm.fraction, sm.fval,
                    sm.delta_fval, _preview_vec(sm.representative))
        end
        if sm.refined
            valid = sm.refined_valid ? "valid" : "INVALID"
            flag = sm.new_min ? "  ⚠ DEEPER than global best" : ""
            @printf(io, "        ↳ re-fit: χ²=%-12.6g (%s, %d fcn)%s\n",
                    sm.refined_fval, valid, sm.refined_nfcn, flag)
        end
    end
    if s.n_noise > 0
        @printf(io, "  (%d sample(s) classified as noise; raise min_size to suppress, lower to expose sparse modes)\n", s.n_noise)
    end
    if K > 1
        println(io, "  ⚠ separated modes have comparable χ² but DIFFERENT physics — treat")
        println(io, "    them independently; do NOT merge into a single error bar.")
    end
    if any_newmin
        println(io, "  ⚠⚠ a refined mode reached a DEEPER minimum than the global best —")
        println(io, "     the main fit likely missed the better basin (see `new_min`).")
    elseif any_refined && K > 1
        println(io, "  (refined modes are distinct local minima; compare χ² above.)")
    end
end
