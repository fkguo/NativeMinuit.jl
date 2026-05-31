# Benchmark results — JuMinuit.jl vs IMinuit.jl on real physics fits

This file captures the most recent comparison runs of the two example
fits across all available execution schemes. Re-run with:

```bash
julia -t 8 --project=scripts BenchmarkExamples/X3872_dip/bench_full.jl
julia -t 8 --project=scripts BenchmarkExamples/IAM_2Pformfactor/bench_full.jl
```

Each script reports stage-by-stage median wall-time (3 rounds + warmup)
and cross-checks every stage (minimum, MINOS errors, mncontour
centroid). The tables below are the **2026-05-31 re-run** on commit
`41226f5` (the post-perf-merge v0.3.0 release tip) — macOS / Julia 1.12 /
`julia -t 8` / `BLAS.set_num_threads(1)`. See the commit history for older runs.

## Scheme legend

| label       | description                                              |
|-------------|----------------------------------------------------------|
| `jm_num`    | JuMinuit numerical gradient, sequential                  |
| `jm_ad`     | JuMinuit AD (ForwardDiff) — package extension            |
| `jm_th_num` | JuMinuit threaded numerical                              |
| `jm_th_ad`  | JuMinuit threaded AD                                     |
| `iminuit`   | Python `iminuit` via PyCall (IMinuit.jl)                 |

## X(3872) dip fit — 3 params, FCN ~ 38 μs/call, 4 data points

Published analysis: V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
"How does the X(3872) show up in e⁺e⁻ collisions: Dip versus peak",
*Phys. Rev. D* **109** (2024) 11, L111501,
[arXiv:2404.12003](https://arxiv.org/abs/2404.12003),
[INSPIRE 2778938](https://inspirehep.net/literature/2778938).

| scheme       | migrad+hesse | minos (3 params) | mncontour (20 pts) |
|--------------|-------------:|-----------------:|-------------------:|
| `jm_ad`      |  **4.7 ms**  |  **72.8 ms**     |  26.9 ms           |
| `jm_th_ad`   |  5.1 ms      |  71.2 ms         |  27.6 ms           |
| `jm_th_num`  |  5.6 ms      | 135.3 ms         |  80.5 ms           |
| `jm_num`     |  6.2 ms      | 131.2 ms         |  76.6 ms           |
| `iminuit`    |  7.4 ms      | 154.7 ms         |  49.0 ms           |

All schemes converge to the same minimum (`fval = 0.0174`, the published
global minimum). The AD path differs by `Δx ≈ 0.015` on the flat
degenerate valley — statistically negligible (a C++-faithful seed-`g2`
detail; see [`../docs/dev/AD_OFFSET_X3872.md`](../docs/dev/AD_OFFSET_X3872.md)).

**Headlines**

- JuMinuit **AD vs iminuit**: **1.6× faster** on migrad+HESSE (4.7 vs 7.4 ms)
  and **2.1× faster** on MINOS (72.8 vs 154.7 ms).
- JuMinuit **numerical vs iminuit**: ~1.2× faster on both (6.2 vs 7.4 ms;
  131.2 vs 154.7 ms).
- AD is ~1.3× faster than JuMinuit-numerical on migrad, ~1.8× on MINOS.
- The 3-parameter problem is too small for threading to help
  (`jm_th_*` ≈ sequential).

**MNCONTOUR caveat.** The fit overfits 3 parameters on 4 points
(`χ²_min = 0.0174`), so the 1σ region collapses to near machine precision.
**Both** libraries terminate early ("MnContours unable to find first two
points") on every parameter pair; the wall-times are time-to-early-exit,
not a successful contour. Not a meaningful comparison for this fit.

(MINOS for `par[2]` now returns `(-0.0043, +0.0043)` on both backends —
an earlier JuMinuit `(0, 0)` edge case in `function_cross` for this tight
well has been resolved.)

## IAM 2π form-factor — 9 LECs, FCN ~ 9 ms/call, 85 data points

| scheme       | migrad+hesse | fval        | minos (par 1) | mncontour (8 pts) |
|--------------|-------------:|------------:|--------------:|------------------:|
| `jm_num`     | **17.78 s**  | **404.15**  | 20.61 s       | 27.57 s           |
| `iminuit`    | 18.52 s      | 409.89      | REFUSED       | REFUSED           |
| `jm_ad`      | FAILED       | —           | —             | —                 |
| `jm_th_*`    | SKIPPED (Phase H rejects) | — | —          | —                 |

**Headlines**

- JuMinuit MIGRAD reaches a **deeper minimum than iminuit** on this stiff
  9-LEC fit — `fval = 404.15` vs `409.89` — at an essentially equal
  wall-time (17.78 vs 18.52 s; ~1.04×). **Both** report `is_valid = false`: the IAM
  landscape is pathologically ill-conditioned and neither library's
  cold-start fully converges (an even deeper ~325 basin is reachable only
  with a more aggressive retry / `Strategy(2)`). This is a hard, honest
  draw where JuMinuit edges ahead — **not** a clean speed showcase.
  *(Historical note: a stale pre-0.3.0 run showed JuMinuit at the shallower
  `613.5`; the Strategy(1)-default fix closed that gap.)*
- iminuit hard-refuses MINOS / MNCONTOUR on an invalid `fmin`
  (`RuntimeError("Function minimum is not valid")`); JuMinuit runs both to
  completion. On this degenerate well its results are themselves marginal
  (MINOS `≈ (-1.41, 1.41)`, an empty contour) — neither library produces a
  trustworthy uncertainty here.
- **Phase-H pre-flight catches the IAM thread-unsafety in milliseconds**:
  `is_thread_safe(chi2_iam, paras0) == false` because `St4_00!` mutates a
  module-level `const c_00_4` buffer, so all `jm_th_*` schemes are refused
  before any work — the silent-wrong-answer guard demonstrated on a real fit.
- The **AD path fails** here: the IAM source (`src/amplitudes.jl`, …) carries
  `Float64` annotations that block ForwardDiff `Dual` propagation — a
  limitation of the IAM code, not of JuMinuit.

## Methodology

- **Wall-time**: median of 3 rounds (X3872: 5) + 1 warmup; `GC.gc()` and a
  short sleep between rounds.
- **Cross-checks**: every stage compares all paths against `jm_num`
  (the conservative reference); mismatches are flagged, not aborted.
- **Phase H**: when `Threads.nthreads() > 1`, the bench probes
  `is_thread_safe(cf, x0)` and refuses racey FCNs upfront.
- **FCN cost**: measured with `BenchmarkTools.@benchmark`, reported in each
  script's header.
