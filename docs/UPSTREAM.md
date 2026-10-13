# Upstream C++ Minuit2 reference

NativeMinuit.jl is a port of the C++ Minuit2 library. The canonical reference
checkout is mirrored locally at `reference/Minuit2_cpp/` (gitignored, see
`.gitignore`) and pinned to:

- **Source**: [GooFit/Minuit2](https://github.com/GooFit/Minuit2) (standalone export of
  ROOT's `math/minuit2`)
- **Commit**: `57dc936a2b74d0b4dda1254c3dd63e7c61a97c84`
- **Tag**: `6.24.0`

## Re-establishing the reference

```bash
mkdir -p reference && cd reference
git clone https://github.com/GooFit/Minuit2.git Minuit2_cpp
cd Minuit2_cpp && git checkout 57dc936a2b74d0b4dda1254c3dd63e7c61a97c84
```

## Upgrade policy

The port pins to a specific upstream commit for two reasons:

1. **Reproducible benchmarks.** The benchmark suite compares Julia wall time to
   the same C++ binary. Bumping upstream invalidates historical benchmarks.
2. **Stable golden data.** Tests use C++-produced reference data (gradient,
   Hessian, MIGRAD iterations) as oracle. New upstream behavior would require
   regenerating goldens.

Upstream bumps happen at major Julia version boundaries and require:

1. A note recording pre-/post-bump benchmark deltas (see `BenchmarkExamples/RESULTS.md`).
2. Regeneration of `test/golden/` artifacts.
3. CI re-baseline.

## Seed-only result conventions

The pinned reference's [seed-only constructor](https://github.com/GooFit/Minuit2/blob/57dc936a2b74d0b4dda1254c3dd63e7c61a97c84/inc/Minuit2/BasicFunctionMinimum.h#L43-L48)
puts the seed objective value into the result's EDM field. The
[MIGRAD early returns](https://github.com/GooFit/Minuit2/blob/57dc936a2b74d0b4dda1254c3dd63e7c61a97c84/src/VariableMetricBuilder.cxx#L75-L91)
can expose this result before minimization iterations. NativeMinuit retains
the EDM stored in its seed state instead. The retry reference tests flag this
representation difference explicitly; they do not compare those EDM fields
as if they had identical meanings, or infer convergence from an invalid result.

This difference is not classified as a confirmed upstream algorithm defect.
The constructor's design intent and any downstream impact require separate
evidence. In particular, a reference-code difference alone is insufficient:
the earlier proposed removal of the HESSE second diagonal clamp was withdrawn
after the [warm-start analysis](dev/DAVIDON_CXX_AUDIT.md) established its role.
