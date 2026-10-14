# Matched MIGRAD measurements

The [recorded measurements and environment](results/readme_075.json) include all five process-pair medians, numerical comparisons and source hashes.

The five standard objectives are compared using numerical-gradient MIGRAD,
Strategy(0), tolerance 0.1, at most 10000 objective calls, and storage level 0, without full iteration-state snapshots. The C++ reference is Minuit2 6.24.0 at
[`57dc936a`](https://github.com/GooFit/Minuit2/tree/57dc936a2b74d0b4dda1254c3dd63e7c61a97c84).
The recorded Julia numerical source is NativeMinuit 0.7.5 at
[`68de553c`](https://github.com/fkguo/NativeMinuit.jl/tree/68de553cbccf1dd4f83949cea7b9e6d83df982fb).
The collector files are identified separately by SHA-256 in the measurement
record. Historical C++ timings used different Gaussian data and included data
generation, so they cannot be compared with these measurements.

The measured operation creates a fresh counted objective and solver state,
runs core MIGRAD, and materializes an owned parameter vector, parameter-error
vector, full covariance matrix, objective value, EDM, call counts and validity
and termination flags. The C++ entry point is
`VariableMetricMinimizer::Minimize`; Julia uses `migrad`. Both compute the same
required terminal outputs for these unbounded problems. The C++ `MnMigrad`
application additionally computes global correlation coefficients; those are
outside this workload. Immutable Gaussian data are prepared from the same
[fixture](fixtures/migrad_inputs.json) before timing. C++ receives a decimal
transport whose values are checked to round-trip to exactly the same Float64
values; its hash is recorded. JSON serialization is outside timing.

Each language runs in five fresh processes, alternating Julia/C++ and C++/Julia
within process pairs. Each process uses five warmups and 50 measured samples
per objective, with one fresh fit per sample. Julia performs four explicit
warmups and BenchmarkTools performs the fifth. Julia uses one thread and one
actual BLAS thread; child processes receive `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`. BenchmarkTools records allocation and GC information
with `gctrial=true`, `gcsample=false`, and zero overhead correction. C++ uses
`steady_clock` around each operation. C++ object destruction and Julia garbage
collection follow their language runtimes; cold loading and compilation are
outside the comparison.

Both implementations must return valid, finite results with a valid covariance
and positive finite covariance diagonals. Quadratic and Gaussian parameter
comparisons use relative tolerance 1e-10 and absolute tolerance 1e-12; objective
values use 1e-10 and 1e-18. EDM, errors and covariance use 1e-6 and 1e-12.
All differences, results and call counts are retained. Rosenbrock also retains
the pre-existing diagnostic checks in [the oracle tests](../test/test_cpp_oracle.jl),
but passing those looser checks does not admit an equivalent-computation timing
ratio. Failed strict comparisons remain visible.

The reported ratio is Julia time divided by C++ time. The central value is the
median of five paired median ratios; its range is the smallest to largest paired
ratio observed. These summarize variation between interleaved processes on one
machine; they are not population confidence intervals or broad speed guarantees.
CPU scheduling, thermal state and unrelated background work are not controlled.

First establish the pinned reference checkout as described in [Upstream C++ reference](../docs/UPSTREAM.md#re-establishing-the-reference). To repeat the measurement with the pinned sources and the recorded benchmark dependencies:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
cmake -S benchmark/cpp -B benchmark/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark/cpp/build --target cpp_readme_bench
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 benchmark/compare_readme.py --source . \
  --cpp-binary benchmark/cpp/build/cpp_readme_bench \
  --protocol benchmark/readme_protocol.json \
  --out benchmark/results/new-comparison
```

Validate a completed record with:

```sh
julia --project=benchmark benchmark/check_readme_evidence.jl benchmark/results/new-comparison
```

The output directory must be new. Each completed process record contains raw
sample times and its result. The comparison also retains raw deltas, per-pair
medians, source/input/binary hashes and failures or missing rows. A nonzero exit
status or a failed numerical row cannot be promoted to a performance claim.
For an exact environment reproduction, use the dependency versions recorded in
the published measurement manifest; a newly resolved environment is a new
measurement.
