# JET diagnostics

The narrow optimization diagnostic checks the raw-function and `CostFunction`
MIGRAD entrypoints for dispatch reports from NativeMinuit. An intentionally
untyped callable is a negative control: JET must report its runtime dispatch.
These diagnostics supplement numerical tests and do not measure performance.

Use Julia 1.11.7 and the separate environment, which pins JET 0.9.20:

```sh
julia --startup-file=no --project=scripts/jet -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --startup-file=no --project=scripts/jet scripts/jet/run.jl
```

Run both commands from the repository root. The environment is separate from
ordinary package tests and does not add JET to the mandatory test target.
[JET's compatibility documentation](https://github.com/aviatesk/JET.jl#julia-compatibility-and-versioning)
identifies its 0.9 series for Julia 1.11; later series target different compiler
versions.

The runner writes `scripts/jet/status.toml` (or the path specified by
`NATIVEMINUIT_JET_STATUS`). Exit 0 means all three checks executed and passed;
exit 1 means an executed check failed; exit 2 means the pinned runtime or tool
was unavailable. CI runs this entrypoint as a required job, retains its status,
and treats both nonzero exit codes as failures. Optional JET checks in the usual
test suite can be unavailable; that does not establish diagnostic acceptance.

The precompile workload has 34 separately callable units. The focused regression
executes every unit without suppressing errors. It validates execution coverage;
no improvement in first-execution time is asserted without timing measurements.

## Package-quality checks

Run the ordinary test suite with standard compilation and cache settings:

```sh
julia --startup-file=no --project -e 'using Pkg; Pkg.test(; julia_args=["--threads=4", "--check-bounds=yes"])'
```

Aqua's persistent-task check observes whether its precompilation process exits
after a package-load marker. The marker can precede compiler-output completion
when `--compiled-modules=existing --compile=min -O0` are inherited by the child
process. A failed process-exit check alone does not identify a leaking task.
Retain the original failure and inspect child arguments and stacks before
attributing it to package code; keep the check's deadline unchanged.
