# Named parameter containers

Install `ComponentArrays` with `Pkg.add("ComponentArrays")` for the examples below.
NativeMinuit does not install it automatically.

A fit with more than a handful of parameters is easier to read — and much
harder to mis-index — when the parameters carry names inside the objective
itself:

```@example named_parameters
using ComponentArrays, NativeMinuit, LinearAlgebra

# A three-parameter line-plus-offset fit, written against names rather than
# positions. Runnable as-is.
xdata = collect(0.0:0.5:5.0)
ydata = 2.0 .* xdata .+ 1.0 .+ 0.3 .* sin.(3.0 .* xdata)
σ     = fill(0.3, length(xdata))

start = ComponentArray(slope = 0.0, intercept = 0.0, curvature = 0.0)

function chi2(p)
    model = p.slope .* xdata .+ p.intercept .+ p.curvature .* xdata.^2
    sum(((model .- ydata) ./ σ).^2)
end

m = Minuit(chi2, start; error = [0.1, 0.1, 0.01])
migrad!(m)
minos!(m)

m.fmin.ext_values.slope     # fitted slope, by name
m.fmin.ext_errors.slope     # its parabolic error
@assert m.valid # hide
X = hcat(xdata, ones(length(xdata)), xdata.^2) # hide
reference_values = X \ ydata # hide
reference_covariance = inv(X' * X) * σ[1]^2 # hide
@assert isapprox(collect(m.values), reference_values; atol=1e-4, rtol=0) # hide
@assert isapprox(m.covariance, reference_covariance; atol=1e-6, rtol=1e-6) # hide
@assert isapprox(m.fmin.ext_values.slope, args(m).slope; atol=0, rtol=0) # hide
(; slope=m.fmin.ext_values.slope, error=m.fmin.ext_errors.slope)
```

NativeMinuit has no dependency on ComponentArrays and no extension for it.
The support is structural: `x0` is used as the **allocation template** for
every coordinate-shaped workspace, so any `AbstractVector{Float64}` whose
`similar`, `copy`, broadcast and mutation preserve its structure is carried
along for free. `ComponentVector` is the common case; a `LabelledArrays`
`LVector`, or your own array type, works on the same terms.

## What is guaranteed

Your container is used for **every value that crosses back into your code**:

- the objective `f(θ)` and a user-supplied gradient `g(θ)`, on every path —
  MIGRAD, HESSE, MINOS, the contour family, `profile` / `mnprofile`, `scan`,
  SIMPLEX, `extremize` / `profile_band`, `find_solution_modes`,
  `get_contours_samples`, the `Optim` bridge, MCMC, all three posterior
  samplers including NUTS, and the `threaded_gradient = :auto` safety probe;
- a user prior in `bayesian` / `posterior_sample`;
- the model in `bootstrap` / `jackknife` and the curve in `@plt_best`;
- the derived-quantity function in `quantiles` and `quantile_band`, and the
  rows from `ens[i]` / iterating a `LikelihoodEnsemble`;
- the low-level kernel entry points — `migrad(cf, x0, errs)`,
  `simplex(cf, x0, errs)`, and `minos` / `contour_exact` driven directly off a
  `CostFunction`;
- the reported results: `m.fmin.ext_values`, `m.fmin.ext_errors`,
  `copy(m.values)`, `copy(m.errors)`, `args(m)`, the MINOS crossing snapshots
  `m.merrors[name].upper_state` / `.lower_state`, and the endpoint vectors
  `plo` / `phi` on both `extremize` and `profile_band`.

This holds with bounds, with fixed parameters, and with an analytic gradient.

Your `x0` itself is never mutated — everything is allocated fresh from it.
Ensemble rows are copies too, so a model that mutates its argument cannot
corrupt the stored samples.

## Internal coordinates are labelled honestly

MIGRAD does not always work in your coordinates. Two things change the space:

- a **bound** makes the stored number an arcsin- or sqrt-transformed
  coordinate, so it is not the physical parameter value;
- a **fixed** parameter is dropped, so the vector is shorter and slot `i` no
  longer corresponds to external parameter `i`.

In either case the internal workspace is a plain `Vector{Float64}`. It would
be worse than useless to label a transformed coordinate `mass`: reading
`.mass` off it would hand you a number that looks physical and is not.

```@example named_parameters
errs = [0.1, 0.1, 0.01]
mb = Minuit(chi2, start; error = errs, limits = [(0.0, 4.0), nothing, nothing], tol = 1e-6)
migrad!(mb)
hesse!(mb)

mb.fmin.internal.state.parameters.x    # Vector{Float64} — transformed coords
mb.fmin.ext_values.slope               # the physical fitted slope
@assert mb.valid && 0 <= mb.fmin.ext_values.slope <= 4 # hide
@assert isapprox(collect(mb.values), reference_values; atol=1e-4, rtol=0) # hide
@assert isapprox(mb.covariance, reference_covariance; atol=1e-6, rtol=1e-6) # hide
@assert mb.fmin.internal.state.parameters.x isa Vector{Float64} # hide
(; slope=mb.fmin.ext_values.slope)
```

Only when the internal and external spaces coincide — every parameter free
**and** unbounded — does the internal state keep the container, because there
each slot still denotes the parameter its label names.

The rule to remember: **read results from `ext_values` / `m.values`, never
from the internal state.** That was already true before named containers; the
labelling just makes it visible.

## Limits

- `load_ensemble` returns rows as plain vectors. A JSON round-trip cannot
  carry a container, so a reloaded ensemble has no template to rebuild onto.
- `names` still default to `x0`, `x1`, … — the component names are not (yet)
  used as parameter names, so `m.values["x0"]` and `m.merrors["x0"]` keep the
  positional names. Pass `name = ["mass", "width", "coupling"]` if you want
  them to agree.
- Anything you build yourself from `collect(m.values)` is, of course, a plain
  vector. Use `copy(m.values)` when you want the container.
