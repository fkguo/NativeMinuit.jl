# Gradients: AD & threading

MIGRAD, MINOS, and the contour scans all need the gradient of the cost function.
C++ Minuit2 has exactly one way to get it: a two-point central-difference
**numerical** gradient (`2·n` FCN calls per refinement cycle). JuMinuit keeps
that as the default, and — because the FCN is an ordinary generic Julia function,
not a `double`-locked virtual call — adds **two** options that C++ cannot offer:

1. **Automatic differentiation** via the ForwardDiff package extension — one
   exact gradient call replaces the `2·n` finite-difference evaluations.
2. **A threaded numerical gradient** — the same finite differences, but with the
   per-coordinate loop spread across `julia -t N` threads.

Both route through the full MIGRAD → MINOS → contour chain unchanged. This page
is the map: which one, when, and the contract each imposes on your FCN.

## Which one, when

The default (serial numerical) is the right choice far more often than not:
threading and AD both have fixed overhead that only pays off when the FCN itself
is expensive. The decision is driven by **per-FCN cost** and **parameter count
`n`**, not by taste.

| Per-FCN cost | `n ≤ 5` | `5 < n ≤ 30` | `n > 30` |
|---|---|---|---|
| `< ~500 ns` | numerical | numerical | numerical |
| `~1–50 μs` | AD | AD | threaded or AD |
| `≥ ~50 μs` | AD | AD or threaded | **threaded** |

Rules of thumb:

- A **cheap** FCN (sub-microsecond): stay serial numerical. The threading and
  AD machinery cost more than they save.
- An **expensive** FCN that is **generic on element type**: prefer **AD**. One
  `cf.g(x)` call versus `2·n·ncycles` finite-difference evaluations is typically
  a 2–10× reduction in FCN work, and the gradient is exact (no step-size noise).
- An expensive FCN that **cannot be made generic** (mutates `Float64` buffers,
  hard-codes types, calls a C library): use the **threaded numerical** gradient
  on `julia -t N`. It needs nothing from the FCN except thread-safety.

The two options are not mutually exclusive in spirit but are in mechanism: the
AD path makes a single gradient call, so threading the per-coordinate loop is a
no-op for it (passing `threaded_gradient=true` to an AD fit is silently ignored).

## AD gradients via ForwardDiff

Loading ForwardDiff alongside JuMinuit auto-activates the extension that backs
the gradient factory. You then have two equivalent ways to wire AD in.

**Pass `grad=` to [`Minuit`](@ref)** — the iminuit-style entry point:

```julia
using JuMinuit, ForwardDiff          # extension auto-activates

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
using JuMinuit, ForwardDiff

cf  = CostFunctionAD(chi2, 1.0)      # up = 1 (χ²); 0.5 for an NLL
fmin = migrad(cf, x0, errs)
```

`CostFunctionAD(f, up)` is exactly `CostFunctionWithGradient(f, x ->
ForwardDiff.gradient(f, x), up)` — see [`CostFunctionWithGradient`](@ref). For
`n ≳ 12` you can pass `chunk_size = 4` (or so) to trade a little speed for lower
memory pressure; the default lets ForwardDiff pick.

!!! note "Calling `CostFunctionAD` without ForwardDiff"
    `CostFunctionAD` is only a stub until `using ForwardDiff` is loaded — calling
    it beforehand raises an informative error. ForwardDiff is a weak (optional)
    dependency, so a plain `using JuMinuit` install stays lightweight.

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

By default JuMinuit validates the AD (or hand-written) gradient against a
numerical 2-point estimate **once**, at the seed point, and **warns** — never
crashes — on disagreement beyond tolerance (the C++ Minuit2 `CheckGradient`
diagnostic). A warning here almost always means the FCN is not actually generic
(so the "AD gradient" silently fell back to something wrong) or the gradient
function returns the wrong thing. Pass `check_gradient = false` to skip it once
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
| `false` *(default)* | serial gradient — always safe, zero overhead. |
| `true` | force the threaded gradient. On the first gradient call it auto-verifies thread-safety and raises [`ThreadSafetyError`](@ref) if the FCN races (the thread-safety contract below). |
| `:auto` | probe thread-safety **once** at the seed (memoized on the fit). If the probe passes, thread; otherwise emit a single `@warn` and fall back to serial. Never throws. Best-effort single-point probe — catches the common shared-buffer race but not one that only appears away from the seed (use `true` for the strict per-call check). |

The two safe-by-construction modes differ in what happens to an *unsafe* FCN:
`true` **refuses** it (throws), `:auto` **demotes** it (warns + serial). Use
`true` when you expect the FCN to be safe and want a hard failure if it is not;
use `:auto` for "thread it if you can, silently, but never give me a wrong
answer." A few edge behaviours of `:auto` worth knowing:

- On single-thread Julia (`julia -t 1`) it is silently serial — no probe, no
  warning (threading is impossible, so there is nothing to check).
- It is a no-op for AD (`grad=`) fits — the gradient is one call, so there is no
  per-coordinate loop to parallelize and no probe is run.
- The probe runs **at most once** per fit and the result is cached, so it never
  re-runs on later MINOS / contour evaluations.

The win scales with FCN cost and `n` (see the table above): an expensive FCN at
high `n` benefits most, a sub-microsecond FCN not at all. That is exactly why the
default is `false`.

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
[`BenchmarkExamples/IAM_2Pformfactor/`](https://github.com/fkguo/JuMinuit.jl/tree/main/BenchmarkExamples)).

JuMinuit's *own* internal buffers are all per-thread; the contract is entirely on
your FCN. Two safety nets back it up:

- **Automatic verification.** With `threaded_gradient=true` (and the default
  `verify_threading=true`), the first gradient call runs the gradient both
  sequentially and threaded at the seed and compares them; a mismatch raises
  [`ThreadSafetyError`](@ref) with a diagnostic pointing at the likely cause.
- **A standalone probe.** [`is_thread_safe(cf, x0)`](@ref) returns `true`/`false`
  without throwing — useful to gate the decision yourself:

```julia
using JuMinuit
cf = CostFunction(my_chi2)
m  = if Threads.nthreads() > 1 && is_thread_safe(cf, x0)
    Minuit(my_chi2, x0; error = errs, threaded_gradient = true)
else
    Minuit(my_chi2, x0; error = errs)
end
```

(`:auto` runs this same probe internally and acts on it for you.)

### Fixing a thread-unsafe FCN: per-thread buffers

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

For a millisecond-scale FCN — exactly the regime where you would thread — that
allocation is negligible. (Note the element type is derived from `eltype(par)`,
not hard-coded to `ComplexF64`: under AD `par` carries `ForwardDiff.Dual`
numbers, so a `ComplexF64` buffer could not hold the resulting `Complex{Dual}`
values and would break differentiation. This stays AD-generic.)

**One buffer per thread** — if you must avoid the allocation, give each thread
its own slot and index by `Threads.threadid()`:

```julia
const T_POOL = [zeros(ComplexF64, 3, 3) for _ in 1:Threads.maxthreadid()]
function chi2(par)
    T = T_POOL[Threads.threadid()]        # this thread's private buffer
    fill_T_matrix!(T, par)
    return loss_from(T)
end
```

Two details make this sound: JuMinuit threads the gradient with `Threads.@threads
:static`, which **pins each loop iteration to a fixed thread**, so `threadid()`
is stable within a call (under the `:dynamic` / `@spawn` schedules it would not
be — do not use this pattern there). And the pool is sized with
`Threads.maxthreadid()`, not `nthreads()`, because Julia can hand out thread ids
beyond `nthreads()` in interactive sessions.

Either way, confirm the fix with `is_thread_safe(cf, x0)` (or just let
`threaded_gradient=true` auto-verify on the first call).

## See also

- AD gradient implementation:
  [`src/ad_gradient.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/ad_gradient.jl)
  and the ForwardDiff extension
  [`ext/JuMinuitForwardDiffExt.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/ext/JuMinuitForwardDiffExt.jl)
- Numerical + threaded gradient:
  [`src/gradient.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/gradient.jl);
  the thread-safety verification lives in
  [`src/migrad.jl`](https://github.com/fkguo/JuMinuit.jl/blob/main/src/migrad.jl)
- [Cost functions](../cost_functions.md) — AD works with generic FCNs and the
  `LeastSquares` / `UnbinnedNLL` / `ExtendedUnbinnedNLL` costs; the binned costs
  (`BinnedNLL` / `ExtendedBinnedNLL`) push their CDF values through `Float64`
  buffers and are **not** currently AD-generic. The threaded numerical gradient
  works on all of them.
