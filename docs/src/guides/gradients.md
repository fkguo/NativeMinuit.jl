# Gradients: AD & threading

MIGRAD, MINOS, and contour searches use gradients of the cost function.
The default is a central-difference numerical gradient. C++ Minuit2 also
accepts user-supplied analytical gradients through
[`FCNGradientBase`](https://github.com/GooFit/Minuit2/blob/57dc936a2b74d0b4dda1254c3dd63e7c61a97c84/inc/Minuit2/FCNGradientBase.h).
NativeMinuit accepts `grad=` callbacks and provides two Julia integrations:

- **ForwardDiff gradients** for objectives generic over their parameter element
  type, including complex intermediates supported by ForwardDiff. AD evaluates
  derivatives with dual numbers; it avoids finite-difference step selection.
- **Threaded numerical gradients** for objectives whose simultaneous calls keep
  mutable state private. Start Julia with `julia -t N` to enable parallel work.

These choices also apply to MINOS and contour searches. For a cheap objective,
threading overhead can outweigh its benefit. AD costs depend on the objective
and ForwardDiff chunk size. Compare the actual supported workload when choosing
a gradient; there is no universal crossover in parameter count or call time.
Threading the per-coordinate numerical loop is a no-op when `grad=` is supplied.

## AD gradients via ForwardDiff

Install the optional package with `Pkg.add("ForwardDiff")`, then load
`using NativeMinuit, ForwardDiff`. Loading both activates the extension
that supplies `CostFunctionAD`; installing ForwardDiff alone does not activate it.
You can pass a gradient callback directly or use the gradient factory.

A complete example:

```jldoctest
julia> using NativeMinuit, ForwardDiff

julia> f(x) = sum(abs2, x .- [1.0, 2.0]);

julia> m = Minuit(f, zeros(2); grad = x -> ForwardDiff.gradient(f, x)); migrad!(m);

julia> m.valid && isapprox(collect(m.values), [1.0, 2.0]; atol=1e-4, rtol=0)
true
```

**Pass `grad=` to [`Minuit`](@ref)** for a model and data you have defined:

```julia
using NativeMinuit, ForwardDiff          # extension auto-activates

function chi2(par)
    mass, coupling, width = par
    χ² = 0.0
    for (sᵢ, yᵢ) in data
        amp   = coupling / (sᵢ - mass^2 - im * mass * width)   # complex BW
        model = abs2(amp)
        χ²   += (model - yᵢ)^2
    end
    return χ²
end

m = Minuit(chi2, x0; error = errs, grad = x -> ForwardDiff.gradient(chi2, x))
migrad!(m)                           # AD propagates through MINOS / contours too
minos!(m)
```

**Or build a [`CostFunctionAD`](@ref)** — a cost object that carries its own AD
gradient, handy when you want to reuse it or hand it straight to `migrad`:

```julia
using NativeMinuit, ForwardDiff

cf  = CostFunctionAD(chi2, 1.0)      # up = 1 (χ²); 0.5 for an NLL
fmin = migrad(cf, x0, errs)
```

`CostFunctionAD(f, up)` is exactly `CostFunctionWithGradient(f, x ->
ForwardDiff.gradient(f, x), up)` — see [`CostFunctionWithGradient`](@ref). For
`n ≳ 12` you can pass `chunk_size = 4` (or so) to trade a little speed for lower
memory pressure; the default lets ForwardDiff pick.

> **Note — Calling `CostFunctionAD` without ForwardDiff**
>
> `CostFunctionAD` is only a stub until `using ForwardDiff` is loaded — calling
> it beforehand raises an informative error. ForwardDiff is a weak (optional)
> dependency, so a plain `using NativeMinuit` install stays lightweight.

### The genericity requirement

ForwardDiff differentiates by evaluating your FCN on `ForwardDiff.Dual` numbers
instead of `Float64`. For that to work your FCN must be **generic over its
element type** — it must not assume the input (or any intermediate) is a
concrete `Float64` / `Complex{Float64}`. The three common pitfalls:

| Anti-pattern | Why it breaks | Fix |
|---|---|---|
| `function f(x::Vector{Float64})` | the signature rejects a `Vector{Dual}` | `function f(x)` (no type annotation) |
| `c::Complex{Float64} = …` | type-locks an intermediate | `c = complex(…)` or `c = re + im*im_part` |
| `buf = zeros(Float64, k)` scratch *inside* `f` | a `Dual` can't be stored in it | `buf = similar(x, eltype(x))`, or allocate fresh per call |

A correctly generic FCN needs **no special code** for the complex intermediates
common in HEP amplitude fits — ForwardDiff propagates straight through
`Complex{Dual}`. Write the physics naturally (using `complex(...)` / `im`, never
`Complex{Float64}` literals) and AD just works.

If a FCN genuinely cannot be made generic, do not fight it — use the threaded
numerical gradient below instead.

### The seed-time gradient check

By default NativeMinuit validates the AD (or hand-written) gradient against a
numerical 2-point estimate **once**, at the seed point, and **warns** — never
crashes — on disagreement beyond tolerance (the C++ Minuit2 `CheckGradient`
diagnostic). A warning means the supplied derivative disagrees with the numerical estimate
at that point. Check the derivative, the objective’s smoothness, and the
finite-difference scale; the comparison does not identify the cause by itself. Pass `check_gradient = false` to skip it once
you trust the gradient.

## Threaded numerical gradient

The numerical gradient evaluates each parameter's central difference
independently — a natural parallel loop. Start Julia with `julia -t N` and flip
the `threaded_gradient` switch; the per-coordinate loop then runs across threads,
and the setting propagates through MINOS and the contour scans.

```julia
m = Minuit(my_chi2, x0; error = errs, threaded_gradient = true)
migrad!(m)
```

`threaded_gradient` is a **3-way switch**:

| value | behaviour |
|---|---|
| `false` *(default)* | serial numerical gradient without a parallel safety probe. |
| `true` | force the threaded gradient. With verification enabled, compare serial and threaded gradients at the entry seed; a detected mismatch raises [`ThreadSafetyError`](@ref). |
| `:auto` | probe at the seed until a result is obtained, then reuse that result (memoized on the fit). If the probe passes, thread; otherwise warn and fall back to serial. Exceptions other than `ThreadSafetyError` propagate. Neither mode proves safety at every input or task interleaving. |

The modes differ in their response to a detected mismatch: `true` throws,
whereas `:auto` warns and uses serial gradients. Both require an objective
that respects the ownership contract below; a seed-point probe is a diagnostic.
A few edge behaviours of `:auto` worth knowing:

- On single-thread Julia (`julia -t 1`) it is silently serial — no probe, no
  warning (threading is impossible, so there is nothing to check).
- It is a no-op for AD (`grad=`) fits — the gradient is one call, so there is no
  per-coordinate loop to parallelize and no probe is run.
- A completed probe result is cached on the fit and reused by later MINOS /
  contour evaluations. If an exception interrupts the probe, no result is cached.

Parallel evaluation can help when objective calls are expensive enough to
amortize scheduling overhead. Measure the actual objective and parameter count;
the default is serial.

### The thread-safety contract

> **Your FCN must not share mutable state across threads.** Module-level scratch
> buffers, a shared RNG, file I/O — anything that two simultaneous calls can step
> on — makes the threaded gradient race.

The classic HEP anti-pattern is a `const` scratch matrix mutated inside the FCN:

```julia
const T_BUF = zeros(ComplexF64, 3, 3)     # ← shared module-level scratch
function chi2(par)
    fill_T_matrix!(T_BUF, par)            # ← parallel calls all write T_BUF
    return loss_from(T_BUF)
end
```

Under `threaded_gradient=true` the `N` parallel calls all clobber `T_BUF` at
once, MIGRAD receives corrupted gradients, and it **silently converges to the
wrong minimum** — a single-threaded χ² of 614 has been observed "converging" to
987 once threaded. This is the failure mode the verification step exists to
catch (worked failure case:
[`BenchmarkExamples/IAM_2Pformfactor/`](https://github.com/fkguo/NativeMinuit.jl/tree/main/BenchmarkExamples)).

NativeMinuit's internal gradient and transform buffers belong to the work unit
or invocation using them. Your FCN must likewise avoid sharing mutable scratch. Two safety nets back it up:

- **Automatic verification.** With `threaded_gradient=true` (and the default
  `verify_threading=true`), the entry check runs the gradient both
  sequentially and threaded at the seed and compares them; a mismatch raises
  [`ThreadSafetyError`](@ref) with a diagnostic pointing at the likely cause.
- **A standalone probe.** [`is_thread_safe(cf, x0)`](@ref) returns `true`/`false`
  on a completed probe, converting `ThreadSafetyError` to `false`; other
  exceptions propagate. Use it to gate the decision:

```julia
using NativeMinuit
cf = CostFunction(my_chi2)
m  = if Threads.nthreads() > 1 && is_thread_safe(cf, x0)
    Minuit(my_chi2, x0; error = errs, threaded_gradient = true)
else
    Minuit(my_chi2, x0; error = errs)
end
```

(`:auto` runs this same probe internally and acts on it for you.)

### Fixing a thread-unsafe FCN: private scratch

The fix is to stop sharing the scratch. Two ways:

**Allocate per call** — simplest, always correct under any schedule, and stays
AD-generic:

```julia
function chi2(par)
    T = Matrix{Complex{eltype(par)}}(undef, 3, 3)   # fresh per call; element type from `par`
    fill_T_matrix!(T, par)
    return loss_from(T)
end
```

Measure the allocation cost for your objective. (Note the element type is derived from `eltype(par)`,
not hard-coded to `ComplexF64`: under AD `par` carries `ForwardDiff.Dual`
numbers, so a `ComplexF64` buffer could not hold the resulting `Complex{Dual}`
values and would break differentiation. This stays AD-generic.)

For reusable scratch, give each task or work unit exclusive ownership for the
entire objective call. Indexing a shared pool by `Threads.threadid()` is unsafe:
an objective can yield, another task can reuse the same slot, and the original
task can migrate. NativeMinuit uses composable dynamic scheduling; threaded
fits and mode analysis can run from a spawned worker or an outer threaded loop.
Use separate `CostFunction`/`Minuit` objects for concurrent fits, since their
counters and fit state are mutable.

Confirm the objective with `is_thread_safe(cf, x0)` or the default verification
in `threaded_gradient=true`. This seed-point probe is a diagnostic, not a proof
of safety for every possible input or task interleaving.

## See also

- AD gradient implementation:
  [`src/ad_gradient.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/ad_gradient.jl)
  and the ForwardDiff extension
  [`ext/NativeMinuitForwardDiffExt.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/ext/NativeMinuitForwardDiffExt.jl)
- Numerical + threaded gradient:
  [`src/gradient.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/gradient.jl);
  the thread-safety verification lives in
  [`src/migrad.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/src/migrad.jl)
- [Cost functions](../cost_functions.md) — all built-in cost types support AD
  when their model, PDF or CDF is generic over the parameter element type.
  Threaded numerical evaluation additionally requires thread-safe callbacks.
