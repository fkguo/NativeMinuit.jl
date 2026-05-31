# Cost functions

For common fits you do not have to write the χ² or negative-log-likelihood by
hand. JuMinuit ships a small, Julia-native family of **cost-function objects**
that carry their own data, model, and `errordef`, so MINOS/HESSE error scaling
is automatic and several costs can be summed into a joint fit.

This mirrors the role of `iminuit.cost`, but is a Julia type family (not a
transliteration), and interoperates with IMinuit.jl's [`chisq`](@ref) / `Data`
helpers.

## The family

| Cost | Use for | `errordef` |
|---|---|---|
| [`LeastSquares`](@ref) | χ² of `y ± σ` vs a model curve | `1.0` |
| [`UnbinnedNLL`](@ref) | unbinned max-likelihood from a normalized pdf | `0.5` |
| [`ExtendedUnbinnedNLL`](@ref) | unbinned, with the total yield as a parameter | `0.5` |
| [`BinnedNLL`](@ref) | histogram fit from a cumulative model | `0.5` |
| [`ExtendedBinnedNLL`](@ref) | histogram fit, yield as a parameter | `0.5` |
| [`CostSum`](@ref) | a joint fit of several costs (`+`) | mixed (rescaled) |

Every cost is a callable `cost(params) -> Float64`, and `errordef(cost)` returns
its `up` (1 for a χ² cost, 0.5 for a `−lnL` cost). The model/pdf is generic on
its parameter vector, so AD (ForwardDiff) and threading work unchanged.

## Least squares

```julia
using JuMinuit

model(x, p) = p[1] * x + p[2]          # a straight line
x  = [1.0, 2.0, 3.0, 4.0, 5.0]
y  = [2.1, 3.9, 6.2, 7.8, 10.1]
σy = fill(0.2, 5)

cost = LeastSquares(x, y, σy, model; name = [:a, :b])
m = Minuit(cost, [1.0, 0.0])           # up = 1 and the data count are read off the cost
migrad!(m)
m.values        # ≈ [1.99, 0.05]
```

`Minuit(cost, x0)` extracts `errordef` and the data count from the cost
automatically (like a `model_fit`). You can also build the cost from an
IMinuit.jl `Data`:

```julia
cost = LeastSquares(Data(x, y, σy), model; name = [:a, :b])
```

Pass `mask = <BitVector>` to fit a subset of points without copying the data.

## Likelihood costs

```julia
# Unbinned: pdf(x, p) must be normalized over the observed range
gpdf(x, p) = exp(-0.5 * ((x - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
nll = UnbinnedNLL(samples, gpdf; name = [:μ, :σ])

# Pass log=true if your function already returns log(pdf):
nll = UnbinnedNLL(samples, (x, p) -> logpdf(x, p); log = true)

# Extended: density need not integrate to 1; `integral(p)` gives the expected total
ext = ExtendedUnbinnedNLL(samples, density, integral; name = [:λ, :N])

# Binned: cdf(edge, p) is the cumulative of the model; bins come from `xe`
bn  = BinnedNLL(counts, edges, cdf; name = [:λ])
ebn = ExtendedBinnedNLL(counts, edges, scaled_cdf; name = [:λ, :N])
```

All likelihood costs use `errordef = 0.5`, so MINOS and HESSE return the correct
`−2Δln L = 1` (1σ) interval without any manual scaling.

## Composing a joint fit — `CostSum`

Add costs with `+` to fit them simultaneously. Parameters are **unified by
name**, so a shared parameter is genuinely shared across datasets:

```julia
# Two datasets sharing the slope `a` but with their own intercepts:
cA = LeastSquares(xA, yA, σA, model; name = [:a, :b])
cB = LeastSquares(xB, yB, σB, model; name = [:a, :c])
joint = cA + cB                 # parameters: a, b, c
m = Minuit(joint, [1.0, 0.0, 0.0]); migrad!(m)
```

`CostSum` evaluates `Σₖ costₖ(sub) / errordef(costₖ)`, i.e. each component is
rescaled to a common `−2lnL`-equivalent before summing. This makes a mixed
least-squares + likelihood fit statistically consistent (a `LeastSquares`
contributes `χ²`, an NLL contributes `2·(−lnL)`), and the combined object
reports `errordef = 1`.

## Relation to IMinuit.jl `chisq`

If you already use IMinuit.jl's `chisq(model, data, par)` / `Data`, those remain
available and unchanged. `LeastSquares` is the object-oriented equivalent: it
shares the same χ² kernel, so a `LeastSquares` fit and the corresponding
`chisq`-based `model_fit` give bit-identical results. Use whichever style you
prefer — `chisq` for a quick functional call, the cost objects when you want
composition (`CostSum`), automatic `errordef`, or the resampling helpers
([`bootstrap`](@ref) / [`jackknife`](@ref) accept cost objects directly).
