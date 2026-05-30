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
                                   chunk_size::Union{Integer,Nothing} = nothing,
                                   check_gradient::Bool = true)
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
    return JuMinuit.CostFunctionWithGradient(f, g, Float64(up);
                                             check_gradient = check_gradient)
end

# ─────────────────────────────────────────────────────────────────────────────
# Precompile the AD gradient path end-to-end. With NO workload here the whole
# ForwardDiff-backed flow (gradient factory, the analytical-gradient MIGRAD
# branch, the CheckGradient seed validation, MINOS) cold-compiles on the user's
# first AD fit. Runs when the extension is precompiled (Julia ≥1.10) — i.e. the
# moment `using ForwardDiff` is loaded alongside JuMinuit. try/catch-wrapped so
# a workload hiccup never breaks the extension's precompilation.
# ─────────────────────────────────────────────────────────────────────────────
using PrecompileTools

PrecompileTools.@setup_workload begin
    _wl_f = x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2
    PrecompileTools.@compile_workload begin
        try
            # CostFunctionAD factory (this extension's own method) + MIGRAD.
            _cf = JuMinuit.CostFunctionAD(_wl_f, 1.0)
            JuMinuit.migrad(_cf, [0.0, 0.0], [0.1, 0.1])
            # iminuit-style high-level entry: Minuit(f, x0; grad=AD) → MIGRAD/MINOS.
            _g = x -> ForwardDiff.gradient(_wl_f, x)
            _m = JuMinuit.Minuit(_wl_f, [0.0, 0.0]; grad = _g)
            JuMinuit.migrad!(_m)
            JuMinuit.minos!(_m, 1)
        catch
            # Don't fail precompile on transient issues
        end
    end
end

end # module JuMinuitForwardDiffExt
