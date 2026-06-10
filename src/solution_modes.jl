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
# ── CRITICAL: the distance metric must be WHITENED (scale-normalized) ──
# Fit parameters span wildly different scales (LECs ~1e-3 next to couplings
# ~1). A naive Euclidean distance is dominated by the largest-scale parameter,
# so two modes separated only in a tiny-scale parameter look identical (their
# Euclidean separation is swamped by the within-mode spread of the big-scale
# parameter) — clustering then WRONGLY merges them. We therefore cluster in
# WHITENED coordinates. WHICH scale to whiten by is the second critical
# choice (the 2026-06 f1(1420) field stress test, where the fit-local metric
# scored a real two-solution cloud as "0 modes"):
#
#   • whiten=:sample  per-coordinate ROBUST CLOUD scale, σ_k = 1.4826·MAD of
#                     the sample column (MAD = median absolute deviation;
#                     1.4826 makes it a consistent σ for normal data).
#                     Scale-free and fit-independent: it measures the cloud
#                     with the cloud's own yardstick, so it works on
#                     MULTI-BASIN clouds whose spread is many fit-σ wide — the
#                     regime this function exists for. Robust against the
#                     between-cluster variance inflation that breaks a full
#                     sample covariance.
#   • whiten=:auto    (default) picks the metric by the cloud/fit scale ratio:
#                     :sample when the cloud is WIDER than the fit's local
#                     scale (max_k σ_cloud_k/σ_fit_k > 4 — the multi-basin
#                     regime where the fit-local metrics isolate every point),
#                     otherwise the fit metric :cov (with its trust-fallback
#                     chain) — so single-basin Δχ² clouds sampled at the fit
#                     scale keep the statistically tightest metric. Falls back
#                     to :errors when neither the cloud nor the covariance
#                     provides a usable scale.
#   • whiten=:cov     full Mahalanobis  z = L⁻¹·x  (Σ_free = L·Lᵀ, Cholesky)
#                     using the FIT's covariance. Decorrelates AND rescales —
#                     the statistically tightest metric WHEN the cloud is a
#                     single basin sampled at the fit's own scale. On a cloud
#                     whose spread ≫ the local fit σ (any cross-basin scan)
#                     every point looks isolated → 0 modes; prefer :sample
#                     there. Falls back to :errors (with a warning) if the
#                     covariance is missing, not positive-definite, or comes
#                     from an invalid / forced-posdef fit (`!m.accurate`).
#   • whiten=:errors  per-parameter FIT scale, z_k = x_k / σ_fit_k. Cheaper
#                     than :cov, no correlations; the robust fallback. Same
#                     local-metric caveat as :cov on multi-basin clouds.
#
# All make distances dimensionless "number of σ" (fit-σ for :cov/:errors,
# cloud-σ for :sample), so a fixed `threshold` (default 1σ) is physically
# meaningful and scale-invariant.
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
#
# FCN cost. Unless told otherwise, `find_solution_modes` evaluates the user
# FCN at EVERY sample (to pick min-χ² representatives and report Δχ²). On
# expensive FCNs pass the χ² values you already have (`fvals = <vector>`), or
# use `fvals = :lazy` (evaluate only the K cluster medoids) / `fvals = :none`
# (no evaluation at all). See the docstring.
# ─────────────────────────────────────────────────────────────────────────────

"""
    SolutionMode

One distinct solution found by [`find_solution_modes`](@ref): a connected
cluster of Δχ²-accepted samples that is separated (in whitened parameter
space) from the other clusters.

# Fields

- `index::Int` — rank, 1 = main solution (lowest χ²), 2, 3, … by ascending χ².
  (With `fvals = :none` the per-sample χ² is unknown: modes are ranked by
  refined χ² when `refine=true`, else by cluster population, largest first.)
- `representative::Vector{Float64}` — the minimum-χ² sample in the cluster
  (full external parameter vector, length = total parameter count). With
  `fvals = :none` or `:lazy` the representative is instead the cluster's
  whitened-space **medoid** (most central member).
- `fval::Float64` — χ² (cost) at `representative` (`NaN` with `fvals = :none`).
- `delta_fval::Float64` — `fval` minus the global-best χ². `≥ 0` for every
  mode when the global best is the fit minimum; the main mode is `≈ 0`.
  Interpret against the fit's `errordef` (`up`): a mode with
  `delta_fval ≲ up` is statistically comparable to the main solution.
  (`NaN` with `fvals = :none`.)
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
- `refined_walltime::Float64` — wall-clock seconds the re-fit took (`NaN` if
  `refine=false`).
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
    refined_walltime::Float64
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
- `whiten::Symbol` — the metric actually used (`:sample`, `:cov` or `:errors`;
  may differ from the request after a fallback, and is never `:auto` —
  `:auto` resolves to `:sample` or `:errors`).
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
# d×d free covariance Σ; for :errors it's a length-d scale vector σ (fit
# errors for whiten=:errors, robust cloud scales for whiten=:sample — the
# :sample metric reuses this per-coordinate branch with its own scales).
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

# whiten=:auto switches from the fit-local metric (:cov chain) to the cloud
# metric (:sample) when the cloud is wider than the fit's own scale in ANY
# coordinate by more than this factor. This is a WIDTH heuristic — "could this
# fit's Δχ² region have produced a cloud this wide?" — not a basin detector.
# Calibration: a single-basin Δχ² scan out to ~4σ has σ_cloud/σ_fit ≈
# 0.74·4 ≈ 3 (uniform MAD), safely below. A ~50/50 two-basin cloud separated
# by ≥ 8σ has MAD ≈ half-separation ⇒ ratio ≥ ~6, safely above. An UNBALANCED
# two-basin cloud has MAD ≈ the majority basin's own width (the median sits
# inside it), so the gate fires only when some coordinate's scatter is itself
# ≫ the fit σ — a fit-scale unbalanced cloud stays on :cov, which resolves it
# (its basins are many fit-σ apart by construction). Borderline structures
# resolve under either metric.
const _AUTO_WIDE_FACTOR = 4.0

# Robust per-coordinate CLOUD scale for whiten=:sample / :auto:
# σ_k = 1.4826·MAD(Xfree[:, k]) (median absolute deviation, scaled to a
# consistent σ estimator for normal data). Coordinates whose MAD is zero /
# non-finite (e.g. a constant column) carry no cloud scale; they are returned
# in `degenerate` and their σ slot is filled with the FIT σ for that
# coordinate instead (the caller warns — the metric must never collapse
# silently). If the fit σ is itself unusable the slot stays degenerate and
# `_whiten_samples` drops the coordinate (0 contribution), again with a
# caller-side warning.
function _sample_mad_scales(Xfree::AbstractMatrix{Float64}, σ_fit::Vector{Float64})
    N, d = size(Xfree)
    σ = Vector{Float64}(undef, d)
    degenerate = Int[]
    buf = Vector{Float64}(undef, N)
    for k in 1:d
        @inbounds for i in 1:N
            buf[i] = Xfree[i, k]
        end
        med = median!(buf)                        # reorders buf — refilled next pass
        @inbounds for i in 1:N
            buf[i] = abs(Xfree[i, k] - med)
        end
        s = 1.4826 * median!(buf)
        if s > 0 && isfinite(s)
            σ[k] = s
        else
            push!(degenerate, k)
            σ[k] = σ_fit[k]
        end
    end
    return σ, degenerate
end

# Warn about fit-σ coordinates that cannot contribute to the :errors metric —
# previously they were dropped silently (0 contribution), which can collapse
# the metric without any trace.
function _warn_dead_fit_sigma(σ_free::Vector{Float64}, names_free::Vector{String})
    dead = [k for k in eachindex(σ_free) if !(σ_free[k] > 0 && isfinite(σ_free[k]))]
    isempty(dead) && return nothing
    @warn "find_solution_modes: fit σ is zero / non-finite for parameter(s) " *
          join(names_free[dead], ", ") *
          " — these coordinate(s) contribute NOTHING to the whitened distance " *
          "(separations along them are invisible to the clustering). Fix the " *
          "fit errors or use whiten=:sample."
    return nothing
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
# Zero-modes / high-noise diagnostics (a K=0 outcome must be actionable)
# ─────────────────────────────────────────────────────────────────────────────

# Nearest-neighbour whitened-distance statistics, computed exactly for
# N ≤ 2048 and on a deterministic stride-subsample above that (a diagnostic,
# not a result — O(n²·d) without the clusterer's early exit).
function _nn_whitened_stats(Z::AbstractMatrix{Float64})
    d, N = size(Z)
    idx = N <= 2048 ? collect(1:N) : unique(round.(Int, range(1, N; length = 2048)))
    n = length(idx)
    nn = fill(Inf, n)
    @inbounds for a in 1:(n - 1)
        ia = idx[a]
        for b in (a + 1):n
            ib = idx[b]
            s = 0.0
            for k in 1:d
                δ = Z[k, ia] - Z[k, ib]
                s += δ * δ
            end
            dist = sqrt(s)
            dist < nn[a] && (nn[a] = dist)
            dist < nn[b] && (nn[b] = dist)
        end
    end
    sort!(nn)
    med = nn[cld(n, 2)]
    q90 = nn[clamp(ceil(Int, 0.9 * n), 1, n)]
    return med, q90
end

# Emitted when clustering finds nothing (K=0) or bins most samples as noise:
# report the cloud's nearest-neighbour scale against the threshold and suggest
# the concrete fix (cloud-scaled metric, or a threshold that would connect
# typical neighbours) instead of leaving the user to debug blind.
function _warn_unclustered(Z::AbstractMatrix{Float64}, threshold::Float64,
                            eff_whiten::Symbol, K::Int, n_noise::Int, N::Int)
    med, q90 = _nn_whitened_stats(Z)
    head = K == 0 ?
        "0 modes — every sample is isolated / noise at threshold=$(threshold)" :
        "$(n_noise)/$(N) samples ($(round(Int, 100 * n_noise / N))%) classified as noise"
    thr_sugg = round(2 * med; sigdigits = 2)
    sugg = if eff_whiten === :cov || eff_whiten === :errors
        "The fit-local metric (whiten=:$(eff_whiten)) is likely too tight for this " *
        "cloud — a cloud whose spread is many fit-σ wide (e.g. a multi-basin scan) " *
        "makes every point look isolated. Try whiten=:sample (robust cloud " *
        "scale), or raise threshold to ≈ $(thr_sugg)."
    else
        "Try raising threshold to ≈ $(thr_sugg) (single-linkage must bridge the " *
        "typical nearest-neighbour gap), lowering min_size, or method=:dbscan."
    end
    @warn "find_solution_modes: $head. Median nearest-neighbour whitened distance " *
          "≈ $(round(med; sigdigits = 3)) σ (90% of points ≤ $(round(q90; sigdigits = 3)) σ) " *
          "vs threshold=$(threshold). $sugg"
    return nothing
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

# Whitened-space medoid of a cluster: the member minimizing the summed
# Euclidean distance to all other members — the FCN-free representative used
# by `fvals = :none` / `:lazy`. O(n²·d) per cluster.
function _whitened_medoid(Z::AbstractMatrix{Float64}, idxs::Vector{Int})
    length(idxs) == 1 && return idxs[1]
    d = size(Z, 1)
    best, best_cost = idxs[1], Inf
    @inbounds for a in idxs
        s = 0.0
        for b in idxs
            b == a && continue
            acc = 0.0
            for k in 1:d
                δ = Z[k, a] - Z[k, b]
                acc += δ * δ
            end
            s += sqrt(acc)
            s >= best_cost && break              # cannot beat the incumbent
        end
        if s < best_cost
            best, best_cost = a, s
        end
    end
    return best
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-mode re-fit (refine=true)
# ─────────────────────────────────────────────────────────────────────────────

# Re-run MIGRAD from a cluster's representative point, preserving the parent
# fit's cost function, gradient, limits, fixed-parameter structure, errordef,
# strategy and tolerance (strategy / tol / per-attempt FCN budget each
# overridable via the refine_* kwargs of find_solution_modes). Threading is
# forced OFF on the inner fit so that, when the modes are refined in parallel,
# we don't oversubscribe threads (the parallelism is ACROSS modes). Returns a
# NamedTuple of re-fit results; any failure is caught and reported as
# `valid=false` rather than aborting the whole call.
function _refine_mode(m::Minuit, rep_full::Vector{Float64},
                       global_best::Float64, iterate::Int;
                       maxfcn::Union{Nothing,Integer} = nothing,
                       strategy::Union{Nothing,Strategy,Integer} = nothing,
                       tol::Union{Nothing,Real} = nothing)
    empty_vec = Float64[]
    try
        grad = m.cfwg === nothing ? nothing : m.cfwg.g
        # Preserve the parent's check_gradient choice — otherwise the constructor
        # default (true) re-enables the seed-time CheckGradient check on every
        # refine fit, emitting spurious warnings when the user set it false.
        mm = grad === nothing ?
            Minuit(m.fcn.f, m; threaded_gradient = false, verify_threading = false) :
            Minuit(m.fcn.f, m; grad = grad, check_gradient = m.cfwg.check_gradient,
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
        migrad!(mm; iterate = iterate,
                strategy = strategy === nothing ? mm.strategy : strategy,
                tol = tol === nothing ? mm.tol : tol,
                maxfcn = maxfcn)
        valid = mm.valid
        rv = Float64[mm.values[i] for i in 1:mm.ndim]
        re = Float64[mm.errors[i] for i in 1:mm.ndim]
        rf = mm.fval
        nf = mm.nfcn
        # Deeper basin: a valid re-fit strictly below the global best (with a
        # small absolute+relative tolerance so numerical jitter at the same
        # minimum is NOT flagged).
        tolnm = 1e-7 * max(1.0, abs(global_best))
        new_min = valid && isfinite(rf) && (rf < global_best - tolnm)
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
    find_solution_modes(samples, m::Minuit; whiten=:auto, method=:components,
                        threshold=1.0, min_size=1, min_neighbors=1,
                        fvals=nothing, refine=false, refine_iterate=5,
                        refine_maxfcn=nothing, refine_strategy=nothing,
                        refine_tol=nothing, refine_callback=nothing,
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
  fallback whitening metric (covariance / errors), the cost function, and —
  for `refine` — the full re-fit configuration.

# FCN cost (read this for expensive cost functions)

By default (`fvals = nothing`) the FCN is evaluated at **every sample** to
pick min-χ² representatives and report `delta_fval` — `N` full FCN calls
before any output (skipped automatically when clustering finds no modes).
On expensive FCNs either pass the per-sample χ² you already have
(`fvals = <vector>`; `get_contours_samples` keeps them), or choose

- `fvals = :lazy` — evaluate ONLY the K cluster representatives
  (whitened-space medoids): K FCN calls, full report (χ², `delta_fval`,
  χ²-sorted modes), but the representative is the most central member, not
  the lowest-χ² one.
- `fvals = :none` — **zero** FCN calls: representatives are medoids,
  `fval`/`delta_fval` are `NaN`, and modes are sorted by refined χ² when
  `refine=true`, else by population (largest first).

# Keyword arguments

- `whiten::Symbol=:auto` — distance metric (this choice decides whether the
  tool works at all on its target inputs):
  - `:sample` — per-coordinate **robust cloud scale**, `σ_k = 1.4826·MAD` of
    the sample column. Fit-independent: it measures the cloud with its own
    yardstick, which is what a MULTI-BASIN cloud needs — basin separations
    stay several whitened units wide while intra-basin distances stay ≪ 1.
    A coordinate with zero cloud spread (constant column) falls back to the
    fit σ for that coordinate, with a warning. (Note for high-dimensional,
    sparsely-sampled clouds: with ≳ 8 free parameters and only a few hundred
    samples, nearest-neighbour distances inside a single cluster approach
    1 cloud-σ, so a sparse cluster can fall apart at `threshold=1` — the
    zero-modes diagnostic below reports the cloud's NN scale and the
    threshold that would reconnect it.)
  - `:auto` (default) — picks the metric from the cloud/fit width ratio:
    `:sample` when the cloud is wider than the fit's local scale in some
    coordinate (`max_k σ_cloud_k/σ_fit_k > 4` — the multi-basin / cross-basin
    regime where fit-local metrics isolate every point), otherwise the `:cov`
    chain below (a single-basin Δχ² cloud sampled at the fit's own scale
    keeps the statistically tightest metric). Falls back to `:errors` when
    neither the cloud nor the covariance provides a usable scale.
  - `:cov` — full Mahalanobis using the fit's free-parameter covariance
    (decorrelating + rescaling). The statistically tightest metric **when the
    samples live at the fit's own scale** (a single-basin Δχ² cloud from this
    exact fit). On a cloud spread over many fit-σ (any cross-basin scan,
    scatter not generated from this fit's errors) every point looks isolated
    → 0 modes; use `:sample` there. Falls back to `:errors` (with a warning)
    if the covariance is unavailable, not positive-definite, or **not
    trustworthy** (`!m.accurate` — invalid fit or forced-posdef Hessian).
  - `:errors` — per-parameter fit scale, `z_k = x_k / σ_k`. No correlations,
    no matrix inversion; the robust fallback. Same local-metric caveat as
    `:cov`. **Naive unwhitened Euclidean is intentionally not offered** — it
    is dominated by the largest-scale parameter and merges modes that differ
    only in a small-scale parameter.
- `method::Symbol=:components` — `:components` (built-in single-linkage
  connected components, no dependency) or `:dbscan` (density-based; requires
  `using Clustering`).
- `threshold::Real=1.0` — connection radius in WHITENED units: fit-σ for
  `:cov`/`:errors`, cloud-σ for `:sample`. Two samples closer than this are
  linked; single-linkage then chains links into components. Smaller = stricter
  (won't bridge distinct modes, but may split a sparsely-sampled one); larger
  risks chaining across a gap. For `:dbscan` this is the ε radius.
- `min_size::Integer=1` — clusters with fewer members are dropped as noise.
  Default `1` keeps every separated region (the accepted set is already χ²-
  filtered, so even a few-point region is a candidate solution); raise it to
  suppress scatter.
- `min_neighbors::Integer=1` — `:dbscan` core-point density (ignored by
  `:components`).
- `fvals::Union{Nothing,Symbol,AbstractVector}=nothing` — per-sample χ²
  policy: a vector of precomputed values, `nothing` (evaluate the FCN at all
  samples), `:lazy` (K representative evaluations) or `:none` (no FCN calls).
  See *FCN cost* above.
- `refine::Bool=false` — re-run MIGRAD from each cluster's representative to
  recover that mode's true local minimum + its own errors. Flags `new_min` if a
  mode is DEEPER than the global best (the main fit missed the better basin).
- `refine_iterate::Integer=5` — `iterate` passed to each re-fit `migrad!`
  (max MIGRAD attempts per mode).
- `refine_maxfcn::Union{Nothing,Integer}=nothing` — FCN call cap **per MIGRAD
  attempt** of a re-fit. Reaching the cap also stops the retry loop, so a
  budgeted mode costs ≈ `1.3 × refine_maxfcn` calls in practice (an attempt
  may extend its own budget by ×1.3 — the C++ second-pass bump — and the cap
  is checked once per MIGRAD iteration, so it can overshoot by one
  iteration's evaluations); `refine_maxfcn × refine_iterate` is a hard upper
  bound. `nothing` = Minuit's default budget `200 + 100·n + 5·n²` per
  attempt — effectively uncapped; SET THIS on expensive FCNs.
- `refine_strategy::Union{Nothing,Strategy,Integer}=nothing` — MIGRAD strategy
  for the re-fits (`nothing` inherits the parent fit's; `0` = fastest, for
  triage on slow FCNs).
- `refine_tol::Union{Nothing,Real}=nothing` — EDM tolerance for the re-fits
  (`nothing` inherits the parent fit's).
- `refine_callback=nothing` — a function called once per **finished** re-fit
  (in completion order), for checkpointing long runs on slow FCNs: a killed
  job then loses at most the mode in flight. Receives a NamedTuple
  `(k, K, representative, n_points, member_indices, refined, refined_values,
  refined_errors, refined_fval, refined_valid, refined_nfcn, new_min,
  walltime)` where `k` is the completion count (1…K). May be invoked from
  worker threads when `parallel`, but invocations are serialized (a lock), so
  writing to a file/IO from it is safe; exceptions it throws are caught and
  warned, never aborting the run.
- `parallel::Union{Bool,Nothing}=nothing` — parallelize FCN evaluation and
  per-mode re-fits across threads. `nothing` ⇒ auto-on when
  `m.threaded_gradient` is set and `Threads.nthreads() > 1` (the same FCN
  thread-safety contract as Phase G/H threaded gradients). Set `true`/`false`
  to force.

# Returns

A [`SolutionModes`](@ref) (an `AbstractVector{SolutionMode}`), sorted with the
main solution first (lowest χ²; see `fvals = :none` above for the no-χ²
ordering). Pretty-prints a summary report.

When clustering finds **no modes** (or bins more than half the samples as
noise), a diagnostic warning reports the cloud's median nearest-neighbour
whitened distance against `threshold` and suggests the concrete fix (a
cloud-scaled metric or a larger threshold) — a 0-mode result on a sane cloud
almost always means the metric, not the cloud.

# Example

```julia
m = Minuit(chi2, x0; names=pnames); migrad!(m)
samples = get_contours_samples(m; ...)          # MC-Δχ² accepted set (rows = vectors)
modes = find_solution_modes(samples, m)          # cluster into distinct solutions
length(modes) > 1 && @warn "multi-modal: \$(length(modes)) distinct solutions"
modes_refined = find_solution_modes(samples, m; refine=true)   # + per-mode re-fit

# Expensive FCN (seconds per call): zero-eval triage, then budgeted refine
# with per-mode checkpointing.
modes = find_solution_modes(samples, m; fvals = :none)
modes = find_solution_modes(samples, m; fvals = :lazy, refine = true,
                            refine_maxfcn = 500, refine_strategy = 0,
                            refine_callback = r -> println("mode \$(r.k)/\$(r.K): ",
                                                           r.refined_fval))
```
"""
function find_solution_modes(samples::AbstractMatrix, m::Minuit;
        whiten::Symbol = :auto,
        method::Symbol = :components,
        threshold::Real = 1.0,
        min_size::Integer = 1,
        min_neighbors::Integer = 1,
        fvals::Union{Nothing,Symbol,AbstractVector} = nothing,
        refine::Bool = false,
        refine_iterate::Integer = 5,
        refine_maxfcn::Union{Nothing,Integer} = nothing,
        refine_strategy::Union{Nothing,Strategy,Integer} = nothing,
        refine_tol::Union{Nothing,Real} = nothing,
        refine_callback = nothing,
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
    refine_maxfcn === nothing || refine_maxfcn >= 1 ||
        throw(ArgumentError("refine_maxfcn must be ≥ 1, got $refine_maxfcn"))
    if fvals isa Symbol && !(fvals === :none || fvals === :lazy)
        throw(ArgumentError("fvals must be a per-sample χ² vector, nothing, " *
                            ":none or :lazy, got :$fvals"))
    end

    Xfree_mat = Matrix{Float64}(Xfull[:, free_idx])

    # ── Resolve the whitening metric: cloud-scale :sample / :auto, or the
    #    fit-scale :cov / :errors with the trust-fallback chain ──
    whiten === :auto || whiten === :sample || whiten === :cov || whiten === :errors ||
        throw(ArgumentError("whiten must be :auto, :sample, :cov or :errors, got :$whiten"))
    eff_whiten = whiten
    errs_full = Float64[m.errors[i] for i in 1:ndim]
    σ_free = errs_full[free_idx]
    names_free = String[m.params.pars[i].name for i in free_idx]
    scale = nothing                       # σ vector (:errors/:sample) or Σ (:cov)
    want_sample = whiten === :sample
    if whiten === :auto || whiten === :sample
        σ_smp, degen = _sample_mad_scales(Xfree_mat, σ_free)
        if whiten === :auto
            # :auto gate — cloud metric only in the regime where the fit-local
            # metrics fail: the cloud is wider than the fit's own scale (in
            # some coordinate) by more than _AUTO_WIDE_FACTOR. A cloud sampled
            # at the fit scale (single-basin Δχ² region) keeps :cov.
            if length(degen) == nfree
                eff_whiten = :cov          # no cloud scale at all → fit-metric chain
            else
                ratios = Float64[σ_smp[k] / σ_free[k] for k in 1:nfree
                                 if !(k in degen) && σ_free[k] > 0 && isfinite(σ_free[k])]
                # No usable fit σ to compare against → trust the cloud scale.
                want_sample = isempty(ratios) ||
                              maximum(ratios) > _AUTO_WIDE_FACTOR
                eff_whiten = want_sample ? :sample : :cov
            end
        end
        if want_sample
            if length(degen) == nfree
                # The cloud defines NO scale in any coordinate (single sample,
                # or all rows identical) — the sample metric is undefined.
                @warn "find_solution_modes: the sample cloud has zero spread in " *
                      "every coordinate (MAD = 0) — whiten=:sample is undefined " *
                      "here. Falling back to whiten=:errors (fit σ)."
                eff_whiten = :errors
            else
                if !isempty(degen)
                    dead = [k for k in degen if !(σ_free[k] > 0 && isfinite(σ_free[k]))]
                    msg = "find_solution_modes: zero cloud spread (MAD = 0) in " *
                          "coordinate(s) " * join(names_free[degen], ", ") *
                          " — using the fit σ for " *
                          (length(degen) == 1 ? "that coordinate" : "those coordinates") * "."
                    if !isempty(dead)
                        msg *= " The fit σ is ALSO degenerate for " *
                               join(names_free[dead], ", ") *
                               " — excluded from the distance metric entirely."
                    end
                    @warn msg
                end
                eff_whiten = :sample
                scale = σ_smp
            end
        end
    end
    if eff_whiten === :cov                # explicit :cov, or :auto → fit metric
        req = whiten === :auto ? "whiten=:auto resolved to the fit metric (:cov)" :
                                 "whiten=:cov requested"
        Σ = matrix(m; skip_fixed = true)
        if Σ === nothing
            @warn "find_solution_modes: $req but the fit has no " *
                  "covariance (run migrad! first?). Falling back to whiten=:errors."
            eff_whiten = :errors
        elseif !m.accurate
            # A forced-posdef / invalid-fit covariance is not a metric, it is a
            # patched-up placeholder — using it silently is how a multi-basin
            # cloud quietly becomes "0 modes" (2026-06 field stress test, F2).
            @warn "find_solution_modes: $req but the fit's " *
                  "covariance is NOT trustworthy (valid=$(m.valid), " *
                  "accurate=$(m.accurate) — invalid fit or forced-posdef " *
                  "Hessian). Falling back to whiten=:errors; consider " *
                  "whiten=:sample, which uses the sample cloud's own robust " *
                  "scale instead of this fit's."
            eff_whiten = :errors
        else
            scale = Σ
        end
    end
    if eff_whiten === :errors
        _warn_dead_fit_sigma(σ_free, names_free)
        scale = σ_free
    end

    Z = if eff_whiten === :cov
        try
            _whiten_samples(Xfree_mat, scale, :cov)
        catch err
            err isa LinearAlgebra.PosDefException || rethrow()
            @warn "find_solution_modes: free covariance is not positive-definite; " *
                  "falling back to whiten=:errors."
            eff_whiten = :errors
            _warn_dead_fit_sigma(σ_free, names_free)
            _whiten_samples(Xfree_mat, σ_free, :errors)
        end
    else
        # :sample and :errors share the per-coordinate scaling kernel.
        _whiten_samples(Xfree_mat, scale, :errors)
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

    # An empty / mostly-noise outcome must carry its own diagnosis (the most
    # common cause is a metric mismatch, which the NN statistics expose).
    if N >= 2 && (K == 0 || n_noise > N ÷ 2)
        _warn_unclustered(Z, Float64(threshold), eff_whiten, K, n_noise, N)
    end

    # ── χ² at the samples — only what the fvals policy needs ──
    # `nothing` = evaluate all N (skipped when there is no mode to report);
    # vector = trust the caller; :lazy / :none = defer to medoids / skip.
    fv = Float64[]
    if fvals isa AbstractVector
        length(fvals) == N ||
            throw(ArgumentError("fvals length $(length(fvals)) ≠ number of samples $N"))
        fv = Float64.(fvals)
    elseif fvals === nothing && K > 0
        fv = _eval_fvals(m, Xfull, do_parallel)
    end
    have_fv = !isempty(fv)

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
        # Representative: minimum-χ² member when per-sample χ² is available,
        # else the whitened-space medoid (most central member).
        best_local = if have_fv
            bl = idxs[1]
            for i in idxs
                fv[i] < fv[bl] && (bl = i)
            end
            bl
        else
            _whitened_medoid(Z, idxs)
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
        prelim[c] = (rep = rep, fval = have_fv ? fv[best_local] : NaN,
                     ranges = ranges, n = length(idxs), members = idxs)
    end

    # :lazy — evaluate the FCN at the K representatives only.
    if fvals === :lazy && K > 0
        Xreps = Matrix{Float64}(undef, K, ndim)
        @inbounds for c in 1:K, j in 1:ndim
            Xreps[c, j] = prelim[c].rep[j]
        end
        repf = _eval_fvals(m, Xreps, do_parallel && K > 1)
        prelim = NamedTuple[merge(prelim[c], (fval = repf[c],)) for c in 1:K]
    end

    # Global-best reference: the fit minimum if available, else the lowest
    # χ² we did evaluate. Used as the Δχ² zero-point.
    global_best = if !isnan(m.fval)
        m.fval
    elseif have_fv
        minimum(fv)
    elseif fvals === :lazy && K > 0
        minimum(p.fval for p in prelim)
    else
        NaN
    end

    # ── Optional per-mode re-fit (parallel-aware, budgeted, checkpointable) ──
    refine_results = Vector{Any}(undef, K)
    if refine && K > 0
        cb_lock = ReentrantLock()
        cb_count = Ref(0)
        run_one = function (c::Int)
            t0 = time()
            r = _refine_mode(m, prelim[c].rep, global_best, Int(refine_iterate);
                             maxfcn = refine_maxfcn, strategy = refine_strategy,
                             tol = refine_tol)
            r = merge(r, (walltime = time() - t0,))
            refine_results[c] = r
            if refine_callback !== nothing
                # Serialized across modes: the callback may write to a file /
                # IO (its whole point is checkpointing slow runs).
                lock(cb_lock) do
                    cb_count[] += 1
                    payload = (k = cb_count[], K = K,
                               representative = prelim[c].rep,
                               n_points = prelim[c].n,
                               member_indices = prelim[c].members,
                               refined = r.refined,
                               refined_values = r.values,
                               refined_errors = r.errors,
                               refined_fval = r.fval,
                               refined_valid = r.valid,
                               refined_nfcn = r.nfcn,
                               new_min = r.new_min,
                               walltime = r.walltime)
                    try
                        refine_callback(payload)
                    catch err
                        @warn "find_solution_modes: refine_callback threw; continuing" exception = err
                    end
                end
            end
            return nothing
        end
        if do_parallel && K > 1
            Threads.@threads :static for c in 1:K
                run_one(c)
            end
        else
            for c in 1:K
                run_one(c)
            end
        end
    end

    # ── Rank the clusters ──
    # χ² ascending when we have it; with fvals=:none fall back to refined χ²
    # (when refine ran) or population, largest first.
    order = if fvals === :none
        if refine && K > 0
            sortperm(Float64[(r = refine_results[c];
                              isfinite(r.fval) ? r.fval : Inf) for c in 1:K])
        else
            sortperm(Int[-prelim[c].n for c in 1:K])
        end
    else
        sortperm(Float64[prelim[c].fval for c in 1:K])
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
                                       r.valid, r.nfcn, r.new_min, r.walltime)
        else
            modes[rank] = SolutionMode(rank, p.rep, p.fval, delta, p.ranges,
                                       p.n, p.n / N, p.members,
                                       false, Float64[], Float64[], NaN,
                                       false, 0, false, NaN)
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

# χ² display that tolerates the fvals=:none case (no evaluation → NaN).
_fmt_chi2(x::Float64) = isnan(x) ? "n/a" : @sprintf("%.6g", x)

function Base.show(io::IO, s::SolutionMode)
    tag = s.index == 1 ? "main" : "mode $(s.index)"
    @printf(io, "SolutionMode[%s]: %d pts (%.1f%%) χ²=%s",
            tag, s.n_points, 100 * s.fraction, _fmt_chi2(s.fval))
    if s.index != 1 && !isnan(s.delta_fval)
        @printf(io, " Δχ²=%.4g", s.delta_fval)
    end
    @printf(io, " rep=%s", _preview_vec(s.representative))
    if s.refined
        flag = s.new_min ? " ⚠DEEPER-MIN" : ""
        @printf(io, " | refit χ²=%.6g%s%s", s.refined_fval,
                s.refined_valid ? "" : " (invalid)", flag)
    end
end

function Base.show(io::IO, ::MIME"text/plain", s::SolutionModes)
    K = length(s.modes)
    whiten_desc = s.whiten === :cov ? ":cov (Mahalanobis)" :
                  s.whiten === :sample ? ":sample (robust cloud scale)" :
                  ":errors (per-σ scale)"
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
        if sm.index != 1 && !isnan(sm.delta_fval)
            @printf(io, "  [%d] %s: %5d pts (%5.1f%%)  χ²=%-12s Δχ²=%-9.4g rep=%s\n",
                    sm.index, tag, sm.n_points, 100 * sm.fraction,
                    _fmt_chi2(sm.fval), sm.delta_fval,
                    _preview_vec(sm.representative))
        else
            @printf(io, "  [%d] %s: %5d pts (%5.1f%%)  χ²=%-12s            rep=%s\n",
                    sm.index, tag, sm.n_points, 100 * sm.fraction,
                    _fmt_chi2(sm.fval), _preview_vec(sm.representative))
        end
        if sm.refined
            valid = sm.refined_valid ? "valid" : "INVALID"
            flag = sm.new_min ? "  ⚠ DEEPER than global best" : ""
            wall = isnan(sm.refined_walltime) ? "" :
                   @sprintf(", %.3gs", sm.refined_walltime)
            @printf(io, "        ↳ re-fit: χ²=%-12.6g (%s, %d fcn%s)%s\n",
                    sm.refined_fval, valid, sm.refined_nfcn, wall, flag)
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
