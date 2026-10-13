# Repeated-fit reference

`cpp_retry_witness.cxx` evaluates the public `migrad!` retry policy using
Minuit2's `MnMigrad` and `MnSimplex`. Minuit2 has no `iterate` keyword: the
witness implements the policy explicitly, with an initial Strategy(1) fit,
Strategy(2) retries, optional Simplex, finite-positive resume-error fallback,
validity-first selection and the same revisit predicate. Multistart, bounded
coordinates, fixed parameters and analytical gradients are outside this witness.
The implementation uses [the pinned upstream source](../docs/UPSTREAM.md).

```sh
cmake -S tools -B tools/retry-build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/retry-build --target cpp_retry_witness
./tools/retry-build/cpp_retry_witness > retry_reference.txt 2> retry_trace.txt
```

Compare the generated table with `test/reference_data/retry_reference.txt`
before updating it. Run the Julia comparison with:

```sh
julia --project=. --check-bounds=yes -e 'using NativeMinuit, Test; include("test/test_retry_cpp_oracle.jl")'
```

Rows contain case name, iteration budget, Simplex flag, validity, call-limit
flag, MIGRAD pass count, actual objective call count, selected pass number,
two coordinates, objective, reported EDM, seed-only EDM representation flag,
valid-error flag, and covariance entries (1,1), (1,2), (2,2).
The final `selector` row reports selection in either argument order, candidate
validity flags and the opposing objective order. It compares a valid quadratic
minimum with a lower-cost invalid singular-quadratic result, so cost-only
selection gives the wrong answer.

The stderr trace preserves each MIGRAD result's reported call count and seed
EDM separately. The fixture's total counts come from an independent objective
counter; a failed error-matrix check can spend calls absent from a retained
minimum's reported count. The pinned C++ seed-only `FunctionMinimum` constructor
puts the objective into its EDM field. That representation is flagged explicitly
and its EDM is not compared to Julia's mathematical seed estimate. Covariance
comparison applies only when the reference error matrix is valid. Coordinate,
objective and shared-state EDM/covariance budgets are fixed in
`test/test_retry_cpp_oracle.jl`; invalid status never constitutes convergence or
a global-minimum claim. A failing comparison must remain a failure until its
input, orchestration or numerical discrepancy is understood.
