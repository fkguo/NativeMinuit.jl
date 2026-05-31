# Regenerating C++ reference data

The JSON files under `test/reference_data/` are produced by
`tools/cpp_trace_harness.cxx` running C++ Minuit2 MIGRAD on a fixed
corpus of benchmark FCNs. JuMinuit's test suite loads these as the
1e-10 numerical-equivalence oracle (the cross-platform parameter-value tolerance).

## When to regenerate

- Upstream Minuit2 bump — pin moves from
  the current `57dc936` (v6.24.0) to a newer commit.
- A benchmark case is added or modified in `tools/cpp_trace_harness.cxx`.
- A platform/BLAS upgrade that exceeds the documented tolerance
  hierarchy (see the tolerance hierarchy below).

**Do not regenerate** for cosmetic reasons. The committed JSON is the
contract; gratuitous regen invalidates the audit trail.

## How to regenerate

```bash
tools/regen_reference.sh
```

The script:

1. Configures CMake against `reference/Minuit2_cpp` (which must be at
   the pinned commit).
2. Builds Minuit2 standalone as a static library + `cpp_trace_harness`.
3. Runs the harness, writing JSON to `test/reference_data/*.json`.
4. Records the machine fingerprint to `test/reference_data/_machine.txt`
   (CPU, compiler, CMake version, Minuit2 SHA, date).

## Pre-flight checklist

Run through this *before* `git add test/reference_data/*.json`:

- [ ] `reference/Minuit2_cpp` is at the pinned commit
      (`git -C reference/Minuit2_cpp rev-parse HEAD`
      = `57dc936a2b74d0b4dda1254c3dd63e7c61a97c84`)
- [ ] Working tree is clean (no uncommitted Julia changes that could
      confound interpretation of test failures after regen)
- [ ] Document in the commit message:
      - What changed in the C++ side that motivated this regen
      - Which JSON files moved by more than 1 ULP and why
      - The machine the regen was performed on
- [ ] Re-run JuMinuit's full test suite; expect either no change
      (if benchmarks didn't move) or a documented set of tolerance
      bumps in `test/test_migrad_*.jl`.

## Tolerance hierarchy

- **1e-10**: final parameter values, cross-platform.
- **1e-6**: `inv_hessian` element-wise, same-platform.
- **1e-3**: iteration-by-iteration trace divergence after iteration 5,
  any platform.

Reference data is bit-exact to the machine + BLAS combination recorded
in `_machine.txt`. Cross-platform comparison uses the loosest
tolerance in the hierarchy above.

## Adding a new benchmark case

1. Add an `FCNBase` subclass to `cpp_trace_harness.cxx`.
2. Add a `run_case(...)` invocation in `main()`.
3. Run the regen script.
4. Add the matching Julia test in `test/test_migrad_*.jl` that loads
   the JSON and asserts equivalence within the tolerance hierarchy.

## Format

Each JSON is a single object with these keys:

| Key | Type | Meaning |
|---|---|---|
| `name` | string | Case name (matches filename stem) |
| `_meta` | object | Source commit, version, strategy_level, generator |
| `x0` | float[] | Initial parameter values |
| `errs0` | float[] | Initial parameter step sizes |
| `fval` | float | Final function value |
| `edm` | float | Estimated Distance to Minimum |
| `nfcn` | int | FCN call count |
| `is_valid` | bool | C++ `FunctionMinimum::IsValid()` |
| `has_covariance` | bool | C++ `FunctionMinimum::HasCovariance()` |
| `has_pos_def_cov` | bool | `HasPosDefCovar()` |
| `hesse_failed` | bool | `HesseFailed()` |
| `made_pos_def` | bool | `HasMadePosDefCovar()` |
| `reached_call_limit` | bool | `HasReachedCallLimit()` |
| `params` | float[] | Final parameter values |
| `errors` | float[] | Final parameter errors (√diag(2·V)) |
| `covariance_upper` | float[] or null | External covariance matrix, upper triangle row-major; `null` if `!has_covariance` |

Float64 values are printed at 17 significant digits to round-trip
IEEE 754 doubles. `NaN`, `Inf`, `-Inf` are encoded as JSON strings
since JSON has no native support.
