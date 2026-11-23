# Matched MIGRAD measurements

## NativeMinuit 0.8.0 on macOS

The [0.8.0 measurement record](results/readme_080_macos.json) gives the
tested commits, process order, timing-block medians and terminal results.
Each measured process uses two seconds of continuous warmup and six timing
blocks of at least 100 ms per objective. The process statistic is the median
of its block medians. The README gives the median and observed range of the
ratios calculated within each Julia/C++ process pair.

In the two-parameter Gaussian case, both objectives compute the log of the
width and its square once per likelihood evaluation. The per-observation
formula and accumulation order are identical to the earlier objective.

The quadratic and Gaussian cases use the strict numerical comparison.
Rosenbrock comparisons use the same stopping settings and require both
results to be valid and pass the recorded diagnostic tolerances. Final
objective values and call counts are reported for these two cases. A row
marked **Variable** has a largest-to-smallest spread above 10% in the
paired ratios or in the block medians of at least one implementation.

To reproduce the macOS run, use the workload commit and the collector commit
from the measurement record in a separate checkout. Copy the fixed-duration
collectors and their protocol into that checkout:

```sh
git worktree add --detach ../nativeminuit-benchmark bac2ad6c92bd99c650fabf14e41973d069cc43a0
git archive 39e7dd5dd99dfed795e86db658bc827f9dea2452 \
  benchmark/check_readme_evidence.jl benchmark/compare_readme.py \
  benchmark/cpp/cpp_readme_bench.cxx benchmark/readme_julia_bench.jl \
  benchmark/readme_protocol.json | tar -x -C ../nativeminuit-benchmark
cd ../nativeminuit-benchmark
git diff 942b733 bac2ad6 -- benchmark/cpp/cpp_readme_bench.cxx | git apply
```

Set `julia_commit` in the copied protocol to the workload commit from the
measurement record. Establish the pinned C++ reference and use the build and
collection commands below. For a second session, reverse the two languages
in each process pair. The collectors retain all completed observations.
Their historical strict verdicts remain available alongside the endpoint
comparison used in the README.

## Archived 0.7.5 measurements

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
