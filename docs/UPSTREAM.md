# Upstream C++ Minuit2 reference

JuMinuit.jl is a port of the C++ Minuit2 library. The canonical reference
checkout is mirrored locally at `reference/Minuit2_cpp/` (gitignored, see
`.gitignore`) and pinned to:

- **Source**: https://github.com/GooFit/Minuit2 (standalone export of
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
