# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# JuMinuitClusteringExt — DBSCAN backend for multi-modal solution detection.
#
# Activated automatically when the user has `using Clustering` loaded
# alongside `using JuMinuit`. Provides the concrete method for the
# `_dbscan_labels` stub declared in `src/solution_modes.jl`, enabling
# `find_solution_modes(samples, m; method=:dbscan)`.
#
# Why a package extension: Clustering.jl pulls in Distances, NearestNeighbors,
# StatsBase, and more — a non-trivial transitive load. The built-in
# single-linkage connected-components clusterer (`method=:components`) covers
# the common case with ZERO dependencies; DBSCAN is the optional, density-based
# upgrade for arbitrary cluster shapes + explicit outlier handling. Julia 1.9+
# `[weakdeps]` + `[extensions]` is the standard idiom for this "optional
# algorithm backend" pattern — mirrors `JuMinuitForwardDiffExt`.
#
# Whitening note: the points handed to DBSCAN here are ALREADY whitened
# (Mahalanobis or per-σ) by `find_solution_modes`, so the radius is in σ units
# and the plain Euclidean metric Clustering uses is the correct distance — see
# the `src/solution_modes.jl` header for why the whitening is mandatory.
# ─────────────────────────────────────────────────────────────────────────────

module JuMinuitClusteringExt

using JuMinuit
using Clustering

"""
    JuMinuit._dbscan_labels(Z, radius, min_neighbors, min_cluster_size) -> Vector{Int}

DBSCAN backend (Clustering.jl) for [`JuMinuit.find_solution_modes`](@ref) with
`method=:dbscan`.

`Z` is the WHITENED sample matrix (`d × N`, columns = points) produced by
`find_solution_modes`; because it is already whitened, `radius` is a distance in
σ units and the default Euclidean metric is the statistically correct one.
`min_neighbors` is the DBSCAN core-point density (a point needs this many
neighbors within `radius` to seed a cluster); `min_cluster_size` drops clusters
smaller than this as noise.

Returns a length-`N` label vector, `0 = noise`, surviving clusters labelled
`1, 2, …` (re-packed dense). `find_solution_modes` then orders them by χ².
"""
function JuMinuit._dbscan_labels(Z::Matrix{Float64}, radius::Float64,
                                  min_neighbors::Int, min_cluster_size::Int)
    # Clustering.dbscan expects points as columns (d × N) — Z already is.
    res = Clustering.dbscan(Z, radius;
                            min_neighbors = min_neighbors,
                            min_cluster_size = min_cluster_size)
    raw = res.assignments        # length N, 0 = noise, else 1-based cluster id

    # Re-pack surviving cluster ids into a dense 1:k range (Clustering may
    # already do this, but be defensive — `find_solution_modes` assumes dense
    # labels with `maximum(labels) == #clusters`).
    remap = Dict{Int,Int}()
    out = Vector{Int}(undef, length(raw))
    k = 0
    @inbounds for i in eachindex(raw)
        a = raw[i]
        if a == 0
            out[i] = 0
        else
            id = get(remap, a, 0)
            if id == 0
                k += 1
                id = k
                remap[a] = id
            end
            out[i] = id
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Precompile the DBSCAN backend on a tiny already-whitened matrix (two obvious
# clusters), so the first `find_solution_modes(...; method=:dbscan)` after
# `using Clustering` skips cold-compiling Clustering.dbscan + the label re-pack.
# Cheap (a 2×6 matrix). try/catch-wrapped so a hiccup never breaks precompile.
# ─────────────────────────────────────────────────────────────────────────────
using PrecompileTools

PrecompileTools.@setup_workload begin
    _Z = [0.0 0.1 0.2 5.0 5.1 5.2;
          0.0 0.1 0.0 5.0 4.9 5.1]
    PrecompileTools.@compile_workload begin
        try
            JuMinuit._dbscan_labels(_Z, 1.0, 1, 1)
        catch
            # Don't fail precompile on transient issues
        end
    end
end

end # module JuMinuitClusteringExt
