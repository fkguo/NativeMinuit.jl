# Benchmark results — NativeMinuit.jl vs IMinuit.jl on real physics fits

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
| `jm_num`    | NativeMinuit numerical gradient, sequential                  |
| `jm_ad`     | NativeMinuit AD (ForwardDiff) — package extension            |
| `jm_th_num` | NativeMinuit threaded numerical                              |
| `jm_th_ad`  | NativeMinuit threaded AD                                     |
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

- NativeMinuit **AD vs iminuit**: **1.6× faster** on migrad+HESSE (4.7 vs 7.4 ms)
  and **2.1× faster** on MINOS (72.8 vs 154.7 ms).
- NativeMinuit **numerical vs iminuit**: ~1.2× faster on both (6.2 vs 7.4 ms;
  131.2 vs 154.7 ms).
- AD is ~1.3× faster than NativeMinuit-numerical on migrad, ~1.8× on MINOS.
- The 3-parameter problem is too small for threading to help
  (`jm_th_*` ≈ sequential).

**MNCONTOUR caveat.** The fit overfits 3 parameters on 4 points
(`χ²_min = 0.0174`), so the 1σ region collapses to near machine precision.
**Both** libraries terminate early ("MnContours unable to find first two
points") on every parameter pair; the wall-times are time-to-early-exit,
not a successful contour. Not a meaningful comparison for this fit.

(MINOS for `par[2]` now returns `(-0.0043, +0.0043)` on both backends —
an earlier NativeMinuit `(0, 0)` edge case in `function_cross` for this tight
well has been resolved.)

## IAM ππ phase-shift fit — paper-faithful 7 free LECs (L6 fixed)

> Free LECs **L1, L2, L3, L4, L5, L7, L8** (7); **L6 fixed** = 0.07×10⁻³ — in
> ππ/Kπ/KK̄ only `2L6+L8` enters, so the paper
> ([arXiv:2011.00921](https://arxiv.org/abs/2011.00921)) fixes L6; the πη
> normalization `c` is **not** a ππ-fit parameter and is dropped (it was the
> vestigial flat 9th param). FCN ~9 ms/call, ππ I=0/1/2 phase shifts. This is a
> deliberately **ill-conditioned stress-test**: the IAM unitarization is nonlinear
> with resonance poles, so the χ² surface is multi-basin (`gvg ≤ 0` DFP warnings =
> non-positive curvature) — *not* a clean fval/speed showcase. (The paper's actual
> fit — global multi-channel data + error inflation — is well-behaved; this strips
> it to ππ-only with a generic seed.)

**Cold-start MIGRAD `fval`** (same seed; lower = deeper; reproduce with
[`iam_strategy_sweep.jl`](IAM_2Pformfactor/iam_strategy_sweep.jl); iminuit 2.18.0):

| `Strategy`    | NativeMinuit (1-shot / +retry) | iminuit (1-shot / +retry) |
|---------------|---------------------------:|--------------------------:|
| 0             | 358.78 ✓ / 358.78 ✓        | 350.09 ✗ / 350.09 ✗       |
| 1 *(default)* | 502.24 ✗ / **360.10 ✓**    | 456.53 ✗ / 456.53 ✗       |
| 2             | 337.66 ✗ / 337.66 ✗        | 376.76 ✗ / 376.76 ✗       |

✓/✗ = MIGRAD valid/invalid. Default-config wall-time (S=1 + retry, migrad+hesse,
min of 3): **NativeMinuit 16.2 s → 360.10 (valid)**; iminuit 10.7 s → 456.53 (invalid —
it gives up earlier). `jm_ad` still FAILS (IAM source non-generic) and `jm_th_*` is
SKIPPED (Phase-H rejects the thread-unsafe FCN).

**Headlines**

- **NativeMinuit and iminuit are the *same* optimizer numerically.** Seeded near any
  minimum (locally well-conditioned), they converge to the same point to **~10⁻⁹**
  ([`iam_localmin_check.jl`](IAM_2Pformfactor/iam_localmin_check.jl)); a +0.5σ nudge
  drops *both* into a deeper shared basin at **≈322**. So the cold-start splits in
  the table above are **not** a fidelity difference — on this multi-basin surface a
  far-enough start lets ULP-level Julia-vs-C++ arithmetic decide which basin (a
  butterfly effect). The C++ oracle tests and the M2 fit show the same: identical
  minima whenever the surface is well-conditioned.
- **On a cold start NativeMinuit is the more robust here**: it reaches a *valid* minimum
  at S=0 and at its default S=1 (360.10), while iminuit fails to validate at *any*
  strategy and — being invalid — hard-refuses MINOS/MNCONTOUR. That is path-luck on
  a chaotic surface, not a systematic edge (in the over-parameterized 9-free variant
  the luck ran the other way). On this degenerate well neither library's
  uncertainties are trustworthy regardless. The Strategy-2 *stall* seen with the old
  9-free setup is gone — that was the vestigial flat parameter; see
  [`docs/dev/IAM_CONVERGENCE_GAP.md`](../docs/dev/IAM_CONVERGENCE_GAP.md).
- **Phase-H pre-flight catches the IAM thread-unsafety in milliseconds**:
  `is_thread_safe(chi2_iam, paras0) == false` because `St4_00!` mutates a
  module-level `const c_00_4` buffer, so all `jm_th_*` schemes are refused
  before any work — the silent-wrong-answer guard demonstrated on a real fit.
- The **AD path fails** here: the IAM source (`src/amplitudes.jl`, …) carries
  `Float64` annotations that block ForwardDiff `Dual` propagation — a
  limitation of the IAM code, not of NativeMinuit.

## Methodology

- **Wall-time**: median of 3 rounds (X3872: 5) + 1 warmup; `GC.gc()` and a
  short sleep between rounds.
- **Cross-checks**: every stage compares all paths against `jm_num`
  (the conservative reference); mismatches are flagged, not aborted.
- **Phase H**: when `Threads.nthreads() > 1`, the bench probes
  `is_thread_safe(cf, x0)` and refuses racey FCNs upfront.
- **FCN cost**: measured with `BenchmarkTools.@benchmark`, reported in each
  script's header.
