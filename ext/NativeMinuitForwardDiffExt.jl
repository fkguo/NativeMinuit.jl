# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# NativeMinuitForwardDiffExt — ForwardDiff-backed AD gradient factory.
#
# Activated automatically when the user has `using ForwardDiff` loaded
# alongside `using NativeMinuit`. Provides the concrete method for the
# `CostFunctionAD` factory stub declared in `src/ad_gradient.jl`.
#
# Why a package extension: ForwardDiff is a 30+ MB transitive load (it
# pulls in StaticArrays, DiffRules, SpecialFunctions, etc.). Making it
# a hard dependency would inflate every NativeMinuit install. Julia 1.9+
# `[weakdeps]` + `[extensions]` is the standard idiom for "optional
# convenience layer".
#
# Beyond C++ Minuit2: see the `CostFunctionAD` docstring in
# `src/ad_gradient.jl` for the design rationale. C++'s virtual-function
# constraint on `MnFcn::operator()` blocks the AD-promoting-the-input-
# type trick that powers `ForwardDiff.gradient(f, x)`; Julia's generic
# function dispatch lifts that constraint at zero user cost.
# ─────────────────────────────────────────────────────────────────────────────

module NativeMinuitForwardDiffExt

using NativeMinuit
using ForwardDiff

mutable struct _ForwardDiffWorkspace
    config::Union{Nothing,ForwardDiff.GradientConfig}
    busy::Bool
end

struct _ForwardDiffGradient{F,C} <: Function
    objective::F
    chunk::C
    owner::Task
    workspace::_ForwardDiffWorkspace
end

function (g::_ForwardDiffGradient{F,Nothing})(x) where {F}
    return ForwardDiff.gradient(g.objective, x)
end

function (g::_ForwardDiffGradient)(x)
    cfg = ForwardDiff.GradientConfig(g.objective, x, g.chunk)
    return ForwardDiff.gradient(g.objective, x, cfg)
end

function NativeMinuit._fit_gradient(g::_ForwardDiffGradient, x::Vector{Float64})
    workspace = g.workspace
    # Only the creating task accesses the mutable cache. A recursive call
    # from a yielding or reentrant objective gets fresh ForwardDiff buffers.
    if current_task() !== g.owner || workspace.busy
        return g(x)
    end
    chunk = g.chunk === nothing ? ForwardDiff.Chunk(x) : g.chunk
    return _cached_gradient(g, x, chunk)
end

function _cached_gradient(g::_ForwardDiffGradient{F}, x::Vector{Float64},
                          chunk::ForwardDiff.Chunk{N}) where {F,N}
    workspace = g.workspace
    workspace.busy = true
    try
        # For a dense Float64 input the tag and dual workspace type follow
        # from the objective and chunk. Refine the cached type before the call.
        Tag = ForwardDiff.Tag{F,Float64}
        DenseConfig = ForwardDiff.GradientConfig{Tag,Float64,N,Vector{ForwardDiff.Dual{Tag,Float64,N}}}
        cfg = workspace.config
        if cfg isa DenseConfig && length(cfg.duals) == length(x)
            return ForwardDiff.gradient(g.objective, x, cfg)
        end
        fresh = ForwardDiff.GradientConfig(g.objective, x, chunk)
        workspace.config = fresh
        return ForwardDiff.gradient(g.objective, x, fresh)
    finally
        workspace.busy = false
    end
end

"""
    NativeMinuit.CostFunctionAD(f, up=1.0; chunk_size=nothing)

ForwardDiff-backed factory. Builds a `CostFunctionWithGradient` whose
gradient is `x -> ForwardDiff.gradient(f, x)` (with optional chunk-size
tuning).

See the stub docstring in `src/ad_gradient.jl` for full semantics and
the "Beyond C++ Minuit2" design rationale.

# Chunking

`chunk_size` controls ForwardDiff's vectorized partials packing. For
`n ≤ 12`, the default (`nothing` → ForwardDiff picks `n`) usually wins.
For `n ≳ 12`, a chunk of 4-6 reduces memory pressure and can be faster.

# FCN-genericity reminder

The user's `f` MUST be generic over element type. Common pitfalls:

- `function f(x::Vector{Float64}) ... end` — type-restrict blocks Dual.
  Fix: `function f(x) ... end`.
- `c::Complex{Float64} = ...` — type-locks the intermediate.
  Fix: `c = complex(...)` or `c = ... + im * ...`.
- Pre-allocated `Vector{Float64}` scratch INSIDE f — Dual cannot store.
  Fix: `scratch = similar(x, eltype(x))` or fresh-allocate per call.

If your FCN can't be made generic (mutates Float64 buffers, calls C
libraries, etc.), use plain `CostFunction(f, up)` + `threaded_gradient=true`
on `julia -t N` instead. See README "Beyond C++ Minuit2" section.
"""
function NativeMinuit.CostFunctionAD(f, up::Real = 1.0;
                                   chunk_size::Union{Integer,Nothing} = nothing,
                                   check_gradient::Bool = true)
    chunk = chunk_size === nothing ? nothing : ForwardDiff.Chunk{Int(chunk_size)}()
    g = _ForwardDiffGradient(f, chunk, current_task(), _ForwardDiffWorkspace(nothing, false))
    return NativeMinuit.CostFunctionWithGradient(f, g, Float64(up);
                                             check_gradient = check_gradient)
end

# Execute the same bounded operations during precompilation and focused tests.
# Exceptions propagate so a failed operation cannot masquerade as completion.
using PrecompileTools

function _precompile_workload()
    f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    cf = NativeMinuit.CostFunctionAD(f, 1.0)
    minimum = NativeMinuit.migrad(cf, [0.0, 0.0], [0.1, 0.1])
    g = x -> ForwardDiff.gradient(f, x)
    m = NativeMinuit.Minuit(f, [0.0, 0.0]; grad=g)
    NativeMinuit.migrad!(m)
    NativeMinuit.minos!(m, 1)
    return cf, minimum, m
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        _precompile_workload()
    end
end

end # module NativeMinuitForwardDiffExt
