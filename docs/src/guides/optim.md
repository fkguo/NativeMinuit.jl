# Alternative minimizers (Optim.jl)

MIGRAD is NativeMinuit's workhorse, but it is not the only minimizer you can point at
a [`Minuit`](@ref). [`optim`](@ref)`(m)` is the Julia-native analogue of iminuit's
`Minuit.scipy()` escape hatch: it minimises the FCN with any optimizer from
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) — LBFGS, BFGS, Nelder-Mead,
Newton, … — starting from `m`'s current values, then **writes the optimum back
into `m`** so you can carry on with NativeMinuit's [`hesse!`](@ref) / [`minos!`](@ref)
exactly as after a [`migrad!`](@ref).

It is a package extension (like the Plots / DataFrames / ForwardDiff ones): Optim
pulls in a sizeable transitive stack, so it is not a hard dependency. Activate the
bridge with `using Optim`; without it, `optim(m)` raises a helpful "load Optim"
message rather than a bare `MethodError`.

```julia
using NativeMinuit, Optim

m = Minuit(fcn, x0)
optim(m; method = :lbfgs)   # minimise with Optim's LBFGS (or:  m |> optim)
hesse!(m)                   # covariance / symmetric errors, à la iminuit
```

## When to reach for it

MIGRAD converges quickly and gives you a covariance for free, so it stays the
default. Use [`optim`](@ref) when:

- **You want to cross-check a MIGRAD minimum.** Re-minimising with a structurally
  different algorithm (e.g. derivative-free Nelder-Mead, or a trust-region-flavoured
  method) and landing on the same point is a cheap, reassuring sanity check that
  MIGRAD did not stop short.
- **MIGRAD struggles on a hard landscape.** On stiff / ill-conditioned problems a
  different optimizer sometimes makes progress where the DFP update stalls. This is
  exactly the role iminuit's `m.scipy()` plays.
- **You specifically want a particular optimizer** — say LBFGS with your own
  analytical gradient, or Newton's method on a smooth problem.

For a robust, gradient-free fallback that stays inside pure NativeMinuit (no Optim
dependency), [`simplex`](@ref)`(m)` runs NativeMinuit's own Nelder-Mead.

## Choosing the method

Pass `method = …` a name (case / dash / underscore insensitive) from this table:

| `method`                                   | Optim optimizer        | Order          |
|:-------------------------------------------|:-----------------------|:---------------|
| `:lbfgs`, `"L-BFGS-B"`                      | `LBFGS()`              | first          |
| `:bfgs`                                     | `BFGS()`               | first          |
| `:conjugategradient`, `:cg`                 | `ConjugateGradient()`  | first          |
| `:gradientdescent`                          | `GradientDescent()`    | first          |
| `:neldermead`, `:simplex`                   | `NelderMead()`         | derivative-free|
| `:newton`                                   | `Newton()`             | second         |

The default is `:lbfgs`. For full control over the optimizer object itself — its
line search, history length, and so on — use the [`minimize_with`](@ref) alias and
hand it a constructed Optim optimizer, bypassing the name table entirely:

```julia
using NativeMinuit, Optim

minimize_with(m, LBFGS())                  # an Optim optimizer object
minimize_with(m, NelderMead())
minimize_with(m; method = :bfgs, tol = 1e-10)   # by name, identical to optim(m; …)
```

[`minimize_with`](@ref) and [`optim`](@ref) are the same bridge under two names;
`optim` mirrors iminuit's `m.scipy`, `minimize_with` reads more clearly when you
pass an optimizer object.

### Gradients, bounds, and fixed parameters

- **Fixed parameters** are held out of the optimisation and restored afterwards;
  their values are untouched. If *every* parameter is fixed, [`optim`](@ref)
  throws (nothing to minimise).
- **Box limits** are honoured through Optim's `Fminbox`. `Fminbox` requires a
  **first-order** inner optimizer, so derivative-free (`:neldermead`) and
  second-order (`:newton`) methods **cannot** be combined with limits — use a
  first-order method (`:lbfgs` / `:bfgs` / `:conjugategradient` /
  `:gradientdescent`) for bounded fits, or remove the limits. A clear error tells
  you which case you hit.
- **Analytical gradients.** When the [`Minuit`](@ref) was built with `grad = …`,
  that gradient is passed through to first-order optimizers automatically. For
  derivative-free and Newton methods Optim builds the derivatives it needs from the
  objective itself.

## Tuning the optimizer

| Keyword            | Maps to (Optim)   | Meaning                                       |
|:-------------------|:------------------|:----------------------------------------------|
| `method`           | —                 | optimizer selector (table above); default `:lbfgs` |
| `ncall` / `maxcall`| `f_calls_limit`   | function-evaluation budget                    |
| `tol`              | `g_tol`           | gradient-norm convergence tolerance           |
| `options`          | `Optim.Options`   | full options object; overrides the three above |

```julia
optim(m; method = :bfgs, maxcall = 10_000, tol = 1e-9)

# Full control — pass an Optim.Options directly:
using Optim
optim(m; method = :lbfgs, options = Optim.Options(g_tol = 1e-12, iterations = 5_000))
```

!!! note "Bounded (Fminbox) fits"
    For bounded fits `ncall` / `maxcall` / `tol` configure Fminbox's *inner*
    optimizer (per outer iteration), not the global call budget or the outer stop
    criterion. For hard control of the outer Fminbox loop pass a full
    `options = Optim.Options(outer_iterations = …, outer_g_abstol = …)`.

## How the result maps back

[`optim`](@ref) / [`minimize_with`](@ref) **return `m`** and update it in place,
just like [`migrad!`](@ref):

- `m.values` / `m.fval` hold the converged point and FCN value.
- `m.valid` reflects whether Optim reported convergence.
- the FCN-evaluation count Optim spent is surfaced as the minimum's `nfcn`.
- any previously cached MINOS errors are cleared (they are stale at the new point).

What it does **not** do by itself is produce a covariance — neither does iminuit's
`m.scipy`. The optimum is seeded back the same way [`migrad`](@ref) constructs its
minimum (a diagonal seed at the converged point), so the natural next step is to
refine the errors with NativeMinuit's own machinery:

```julia
using NativeMinuit, Optim

m = Minuit(cost, x0)
optim(m; method = :lbfgs)   # alternative minimizer finds the minimum
hesse!(m)                   # full covariance + symmetric errors
minos!(m)                   # asymmetric errors, if you want them
```

This is the Julia-native counterpart of iminuit's scipy-then-`hesse()` flow: a
different optimizer locates the minimum, and you still get NativeMinuit's full
HESSE / MINOS error analysis afterwards (see
[Error analysis](../error_analysis.md)).

## See also

- [`simplex`](@ref) — NativeMinuit's built-in gradient-free Nelder-Mead, no Optim needed.
- [`migrad!`](@ref) — the default minimizer.
- [`hesse!`](@ref) / [`minos!`](@ref) — error analysis to run after the fit.
- Implementation: [`ext/NativeMinuitOptimExt.jl`](https://github.com/fkguo/NativeMinuit.jl/blob/main/ext/NativeMinuitOptimExt.jl).
