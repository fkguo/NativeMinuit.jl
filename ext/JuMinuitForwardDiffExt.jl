# SPDX-License-Identifier: LGPL-2.1-or-later
#
# ─────────────────────────────────────────────────────────────────────────────
# JuMinuitForwardDiffExt — ForwardDiff-backed AD gradient factory.
#
# Activated automatically when the user has `using ForwardDiff` loaded
# alongside `using JuMinuit`. Provides the concrete method for the
# `CostFunctionAD` factory stub declared in `src/ad_gradient.jl`.
#
# Why a package extension: ForwardDiff is a 30+ MB transitive load (it
# pulls in StaticArrays, DiffRules, SpecialFunctions, etc.). Making it
# a hard dependency would inflate every JuMinuit install. Julia 1.9+
# `[weakdeps]` + `[extensions]` is the standard idiom for "optional
# convenience layer".
#
# Beyond C++ Minuit2: see the `CostFunctionAD` docstring in
# `src/ad_gradient.jl` for the design rationale. C++'s virtual-function
# constraint on `MnFcn::operator()` blocks the AD-promoting-the-input-
# type trick that powers `ForwardDiff.gradient(f, x)`; Julia's generic
# function dispatch lifts that constraint at zero user cost.
# ─────────────────────────────────────────────────────────────────────────────

module JuMinuitForwardDiffExt

using JuMinuit
using ForwardDiff

"""
    JuMinuit.CostFunctionAD(f, up=1.0; chunk_size=nothing)

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
function JuMinuit.CostFunctionAD(f, up::Real = 1.0;
                                   chunk_size::Union{Integer,Nothing} = nothing)
    g = if chunk_size === nothing
        x -> ForwardDiff.gradient(f, x)
    else
        # Build a per-call GradientConfig with the chunk size pinned.
        # The config can in principle be cached across calls, but FCN
        # call-site x-types may vary (Float64 / Dual chains), and
        # mismatched cached configs throw. Fresh config is safer; the
        # alloc is small (a few hundred bytes).
        n_chunk = Int(chunk_size)
        x -> begin
            cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{n_chunk}())
            ForwardDiff.gradient(f, x, cfg)
        end
    end
    return JuMinuit.CostFunctionWithGradient(f, g, Float64(up))
end

end # module JuMinuitForwardDiffExt
