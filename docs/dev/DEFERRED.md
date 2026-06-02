# Deferred — known but intentionally postponed

Companion to `ROADMAP.md` § 9. This file is the "memory field" — anything
the team decided not to do now but might do later. Each item lists *why*
it's deferred and *what would change that*.

A `[REVISIT]` tag in code (`# REVISIT: see DEFERRED.md#item-name`) should
point back here whenever a Phase-0/Phase-1 implementation makes a choice
that this file documents.

## Algorithms & code paths

### Fumili minimizer (10 files in C++, ~3K LOC)

`FumiliBuilder`, `FumiliMinimizer`, `FumiliErrorUpdator`,
`FumiliGradientCalculator`, `FumiliChi2FCN`, `FumiliMaximumLikelihoodFCN`,
`FumiliStandardChi2FCN`, `FumiliStandardMaximumLikelihoodFCN`,
`FumiliFCNBase`, `FumiliFCNAdapter`, `MnFumiliMinimize.h/.cxx`.

- **Why deferred**: specialized for χ² / max-likelihood with Jacobian-style
  updates. The ROOT-recommended minimizer for histogram fits, but not
  the default for general fitting. iminuit/IMinuit.jl users have no
  built-in dependency on it.
- **Revisit when**: a concrete user requests Fumili in JuMinuit.jl
  (likely a HEP analysis with many histogram bins where the Fumili
  speedup matters), or when Phase 3 IMinuit.jl-compat scripts in the
  wild use Fumili and break.

### Multi-node (cluster) gradient parallelism — `Distributed.jl`, NOT MPI

C++ analog: `MPIProcess.h/.cxx`, `MPI_SYNCH_PROC` guards in
`Numerical2PGradientCalculator.cxx:102–214` and `MnHesse.cxx:240` — C++
splits the per-coordinate numerical gradient + HESSE probes across MPI
ranks.

- **Why deferred**: no cluster in use yet. Building cluster parallelism
  with no multi-node hardware to validate against is premature — it
  should be designed + benchmarked against a real cluster + a real
  expensive fit when the need is concrete.
- **DECISION (locked) — use `Distributed.jl`, not MPI.jl**, when this is
  built. Reasoning:
  - The parallelised work is the per-iteration gradient/HESSE: an
    embarrassingly-parallel map over parameter indices with a small
    gather (the gradient vector) each MIGRAD iteration.
  - This parallelism only pays off for an **expensive FCN** (cheap-FCN
    communication overhead dominates — exactly why C++ gates it behind
    `#ifdef`). In that expensive regime, FCN **compute dominates** and
    the per-iteration sync latency is amortised, so MPI's low-latency
    collective advantage does **not** materialise. Minuit's design range
    (n ≤ ~50) keeps the per-gradient message modest, reinforcing this.
  - `Distributed.jl` is **stdlib (zero external dependency)** → lowest
    adoption friction for the open-source community (no system-MPI build
    headache); composes with `SlurmClusterManager.jl` for HEP Slurm
    clusters; is GC-aware and testable by spinning workers on one machine
    in CI. MPI.jl wins only in a narrow "very-many-params + moderate FCN
    + latency-bound" window or when slotting into an existing MPI
    workflow — neither is JuMinuit's target.
  - Build on the **Phase G threaded-gradient abstraction** (the
    parallel-gradient executor already exists for threads): add a
    pluggable `:distributed` executor rather than a parallel code path
    from scratch.
- **Revisit when**: a real >1-node cluster fit (expensive FCN, many
  params) is in hand to design + benchmark against. Single-machine fits
  are already covered by Phase G threads — MPI/Distributed give nothing
  there.

### BFGS Hessian updator

`BFGSErrorUpdator.h/.cxx`, `VariableMetricMinimizer(BFGSType)`.

- **Why deferred**: Davidon (DFP) is Minuit's default and what every
  user tunes against. BFGS is a strict alternative, not an improvement.
- **Revisit when**: Phase 2 opens. Likely a ~200-LOC port; trivial
  given DFP is already in Phase 0.

### CombinedMinimizer / ScanMinimizer

`CombinedMinimizer.h/.cxx`, `CombinedMinimumBuilder.cxx`,
`ScanMinimizer.h`, `ScanBuilder.cxx`.

- **Why deferred**: CombinedMinimizer = MIGRAD + SIMPLEX fallback. Julia
  expresses this as `migrad(...) || simplex(...)` composition rather
  than as a new minimizer type. Cleaner.
- **Revisit when**: a user wants the C++-style `MnApplication`-derived
  combined-minimizer struct (Phase 3 facade).

### Cubic and Brent line searches

`MnLineSearch.cxx:321–820` under `#ifdef USE_OTHER_LS`.

- **Why deferred**: disabled by default in the C++ build; default
  parabolic search is what every Minuit2 user runs.
- **Revisit when**: a user proves a specific fit benefits from non-
  parabolic search. Unlikely in HEP.

## Numerical / storage

### Packed lower-triangular symmetric storage (LASymMatrix)

C++ uses `Vector{Float64}` of size `n(n+1)/2` for symmetric matrices,
with packed-BLAS routines DSPMV/DSPR.

- **Why deferred (as default)**: dense `Matrix{Float64}` + `Symmetric`
  view wins for the typical Minuit2 `n ≤ 50` (Decision Q3 locked in
  ROADMAP v2). Both representations fit in L1; OpenBLAS DSYMV
  vectorizes better than DSPMV.
- **What we ship**: `linalg.jl` uses dense `Symmetric` throughout and
  **abandons** packed BLAS (and the ABObj layer) — see the file header. The
  layout is isolated behind the documented surface so a packed variant *could*
  be swapped in later if it ever wins on a benchmark (n ≥ 200); none is shipped.
- **Revisit when**: a Phase 0 benchmark shows packed beating dense by
  more than 10% on any blocking scenario.

### StaticArrays (SVector/SMatrix) for small n

Phase 0 benchmark corpus includes n=2, 4, 10, 40; the n=2/4 cases
*may* benefit from StaticArrays specialization.

- **Why deferred**: adds a runtime dependency before evidence justifies
  it. Violates the ROADMAP's "measure first" rule.
- **Decision time**: Phase 0 day 26–28, after end-to-end MIGRAD
  benchmarks are in. Three scenarios:
  - n=2 win > 30%: ship `MigradStatic{N}` variant in Phase 1.
  - 10–30% win: ship in Phase 2.x as a package extension.
  - < 10% or worse: drop, keep `Vector` for all n.

## Tooling / ecosystem

### ROOT serialization compatibility

`G__DICTIONARY` macros in `FunctionMinimum.h:18–20`, `LinkDef.h`,
all of `inc/Math` and `inc/Fit`.

- **Why deferred**: only relevant inside a running ROOT process.
  Julia/iminuit users have no ROOT runtime.
- **Revisit when**: never, unless we ship a `ROOT.jl` integration
  package — and even then, only the serialization shim, not the full
  type hierarchy.

### MnPlot ASCII text plotting

`MnPlot.cxx`, `mntplot.cxx`, `mnbins.cxx`.

- **Why deferred**: Julia users primarily plot via Plots.jl/Makie.jl.
- **What we provide instead**: a `mn_plot_text` helper (plot_text.jl) for
  terminal-only users — methods take a `ContoursError` or a raw
  `Vector{Tuple{Real,Real}}` of contour points (e.g.
  `mn_plot_text(contour(fmin, cf, 1, 2))`) and render an ASCII contour glance.

### ParametricFunction integration

`ParametricFunction.h/.cxx`.

- **Why deferred**: ROOT IFunction wrapper; Julia users pass plain
  callables directly.

### Hand-rolled BLAS (mn*.cxx)

`mndaxpy`, `mndscal`, `mnddot`, `mndspmv`, `mndspr`, `mndasum`,
`mnlsame`, `mnxerbla`.

- **Why deferred (permanently)**: these are f2c-translated BLAS routines.
  Julia's `LinearAlgebra.BLAS.*` calls the same OpenBLAS/MKL routines
  the C++ ultimately calls; the f2c layer is pure historical baggage.

### ABObj expression templates

`ABObj.h`, `ABSum.h`, `ABProd.h`, `ABTypes.h`, `LaSum.h`, `LaProd.h`,
`LaOuterProduct.h`, `LaInverse.h`, `VectorOuterProduct.h`,
`MatrixInverse.h`.

- **Why deferred (permanently)**: the expression-template machinery is
  a 2003-era C++ workaround for avoiding intermediate temporaries in
  compound arithmetic. Julia's `mul!`/`syr!`/`axpy!` into preallocated
  buffers solves the same problem more transparently.

### StackAllocator

`StackAllocator.h`.

- **Why deferred (permanently)**: pool allocator to bypass `malloc`.
  Workspace preallocation in `MigradWorkspace` is the Julia analog —
  same effect, GC-aware, no custom allocator needed.

### MnRefCountedPointer / MnReferenceCounter

`MnRefCountedPointer.h`, `MnReferenceCounter.h`.

- **Why deferred (permanently)**: pre-C++11 reference counting.
  Superseded by `std::shared_ptr` in the C++ code itself; irrelevant
  in Julia.

## Compatibility shims

### MnTinyMain / FORTRAN-Minuit shims

`MnTiny.h/.cxx`, `TMinuit2TraceObject.cxx` (parts).

- **Why deferred (permanently)**: compatibility with the original
  Fortran Minuit interface from the 1970s.

## Out-of-scope but worth noting

- Custom minimizer plugins (the `Math::Minimizer` plugin system in
  ROOT). Not relevant for a standalone library.
- ROOT-specific result formats (`ROOT::Fit::FitResult`). Julia users
  serialize to JSON/JLD2 instead — see Phase 2.5.

---

## Process: how to revisit something here

1. Open an issue titled `[revisit-deferred] <topic>` in the JuMinuit.jl
   repo.
2. Link to the section above.
3. State the new evidence/demand that justifies revisiting (a user
   request, a benchmark result, an upstream change).
4. The reply lands either as a Phase 2.x / Phase 3 milestone or stays
   here with an updated "Revisit when" line.
