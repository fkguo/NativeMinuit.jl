# Building the manual

Use Julia 1.11 or later. From the repository root, resolve the documentation
environment against the local checkout and build:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build checks that `NativeMinuit` was loaded from this checkout. The docs
environment declares the optional integrations used in executable examples.
Doctests compare semantic output; complete `@example` blocks generate displayed
results. Missing internal API documentation can warn, but unresolved references,
missing `@docs` bindings, failed examples and failed doctests stop the build.

Local builds omit repository source links, so a local clone need not have a
hosted Git remote. CI resolves those links from its hosted checkout.
Open `docs/build/index.html` after a local build. Deployment is separate and
occurs only through the documentation workflow.
