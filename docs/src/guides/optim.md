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
bridge with `Pkg.add("Optim")` followed by `using Optim`; without it, `optim(m)` raises a helpful "load Optim"
message rather than a bare `MethodError`.

```julia
using NativeMinuit, Optim

m = Minuit(fcn, x0)
optim(m; method = :lbfgs)   # minimise with Optim's LBFGS (or:  m |> optim)
hesse!(m)                   # covariance / symmetric errors, à la iminuit
```

A complete quadratic example:

```jldoctest
julia> using NativeMinuit, Optim

julia> m = Minuit(x -> sum(abs2, x .- [1.0, 2.0]), zeros(2)); optim(m; method=:lbfgs); hesse!(m);

julia> m.valid && m.fval < 1e-10 && isapprox(collect(m.values), [1.0, 2.0]; atol=1e-4, rtol=0)
true
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

Choose one of the symbols or strings in the table. The bridge lowercases names
and maps spaces and dashes to underscores.

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
| `options`          | `Optim.Options`   | full options object; overrides `ncall`, `maxcall` and `tol` |

```julia
optim(m; method = :bfgs, maxcall = 10_000, tol = 1e-9)

# Full control — pass an Optim.Options directly:
using Optim
optim(m; method = :lbfgs, options = Optim.Options(g_tol = 1e-12, iterations = 5_000))
```

> **Note — Bounded (Fminbox) fits**
>
> For bounded fits `ncall` / `maxcall` / `tol` configure Fminbox's *inner*
> optimizer (per outer iteration), not the global call budget or the outer stop
> criterion. For hard control of the outer Fminbox loop pass a full
> `options = Optim.Options(outer_iterations = …, outer_g_abstol = …)`.

## How the result maps back

[`optim`](@ref) / [`minimize_with`](@ref) **return `m`** and update it in place,
just like [`migrad!`](@ref):

- `m.values` / `m.fval` report Optim's final point and objective value.
- `m.valid` reflects whether Optim reported convergence.
- The reported `nfcn` is the larger of Optim's objective-call count and
  NativeMinuit's seed call count.
- The bridge clears cached MINOS errors and initializes the covariance from
  NativeMinuit's seed at the returned point.

After a valid fit, run HESSE to estimate the full covariance and MINOS for
asymmetric errors.

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
