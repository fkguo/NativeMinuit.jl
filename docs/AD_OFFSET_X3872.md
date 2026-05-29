# AD vs numerical MIGRAD offset on the X(3872) dip fit

**Status:** Investigation complete — **verdict (a): expected behavior, no
algorithm fix warranted.** Documentation-only changes applied.

**Date:** 2026-05-29 · **Scope:** `BenchmarkExamples/X3872_dip/bench_full.jl`

---

## TL;DR

On the X(3872) dip fit the ForwardDiff (AD) MIGRAD path lands a deterministic
`Δx = 0.0149` away from the numerical-gradient path, at a marginally deeper
`fval` (0.0173444 vs 0.0174225). The cause is **not** a bug:

1. The numerical and AD **seeds differ in `g2`** (the per-parameter diagonal
   2nd-derivative estimate). The numerical seed refines `g2` by finite
   differences; the AD seed uses the cheap `2·up/dirin²` estimate from the
   initial step sizes. This produces seed inverse-Hessian diagonals that
   differ by **6×–90×** and a seed EDM that differs by **60×**.
2. This seed-`g2` asymmetry is **faithful to C++ Minuit2** — the C++
   `MnSeedGenerator` analytical-gradient overload does exactly the same thing
   (`MnSeedGenerator.cxx:119-122`).
3. The X(3872) χ² minimum is a **flat, degenerate valley** (MINOS cannot close
   most 1σ crossings). On such a surface MIGRAD's `edm < goal` stop is
   satisfied at trajectory-dependent points, so two seeds → two slightly
   different valley-floor stopping points.
4. The offset is **0.5–0.7 % of the 1σ parabolic error** in every parameter —
   statistically negligible. Both points are valid minima.

The AD gradient *value* is correct — it agrees with an independent
central-difference reference to ~1e-8 and is chunk-size-invariant. The
divergence is entirely a seed-*curvature* (`g2`) effect amplified by the
degenerate valley, not a gradient-value effect.

---

## Symptom (reproducible, deterministic)

```
jm_num (numerical):  x=[2.547796, 0.00273288, -6.615084]  fval=0.01742248
jm_ad  (ForwardDiff): x=[2.538232, 0.00275672, -6.630024]  fval=0.01734440  ← deeper
cross-check: jm_ad vs jm_num → Δx=0.0149407  Δfval=7.807e-05
iminuit (numerical):  matches jm_num to Δx≈8.9e-6
```

Δx is bit-stable across runs. Both `jm_ad` and `jm_th_ad` (threaded AD) land at
the same point → it is an AD-path property, not a threading artifact.

## Reproduce

```
rm -rf ~/.julia/compiled/v1.12/JuMinuit*/      # avoid stale-cache traps
julia -t 8 --project=scripts scratch/ad_offset_probe.jl   # seeds + per-iter trace
julia -t 8 --project=scripts scratch/ad_offset_sigma.jl   # offset vs 1σ + MINOS
```

---

## Root cause, localized

### Step 1 — the seeds already differ (iteration 0)

Both paths start at the same `x = [3, 1e-4, -4]`, same `fval = 20.40313`, and
near-identical gradients. They differ in `g2` and `gstep` — but it is the `g2`
difference (and hence the seed inverse-Hessian diagonal `V_diag = 1/g2` and the
seed EDM) that drives the first search direction. (`gstep` feeds only the
numerical gradient's per-coordinate step selection, which the AD path never
exercises, so the `gstep` difference is inert here.)

| quantity | numerical seed | AD seed | ratio AD/num |
|---|---|---|---|
| `grad`   | `[-39.5546, -13852.640, 0.548335]` | `[-39.5546, -13852.755, 0.548337]` | ~1 (Δ ≲ 1e-5 rel; worst is `r` at 8e-6) |
| `g2`     | `[30.519, 1.7792e6, 0.33042]` | `[200, 20000, 2]` | `[6.55, 0.0112, 6.05]` |
| `gstep`  | `[4.09e-4, 1e-5, 3.93e-3]` | `[0.01, 1e-3, 0.1]` | (inert — AD path uses no step selection) |
| `V_diag` | `[0.032767, 5.621e-7, 3.0264]` | `[0.005, 5e-5, 0.5]` | `[0.153, 88.9, 0.165]` |
| `edm`    | `80.02` | `4801.5` | `60×` |

The two seed gradients agree to ≲1e-5 relative (the `r` component differs by
8e-6, just the FD truncation in JuMinuit's adaptive numerical seed). That the
*gradient value itself* is correct — no sign or scaling bug in the AD path — is
established separately by the gradient-value check (hypothesis 3 below), where
the AD gradient agrees with an independent central-difference reference to ~1e-8
and is bit-identical across chunk sizes. Neither figure is large enough to
explain a 0.0149 offset; the 89× `V_diag` difference on `r` is.

The AD `g2 = [200, 20000, 2]` is exactly `2·up/dirin²` with `up = 1` and
`dirin = ERRS = [0.1, 0.01, 1.0]` — i.e. the rough `InitialGradientCalculator`
estimate, never refined. The numerical `g2` is the finite-difference curvature.

**Why:**

- *Numerical seed* (`src/seed.jl:124-128`): computes the rough initial gradient
  with `initial_gradient!`, then **refines** `grad`/`g2`/`gstep` in place with
  `numerical_gradient!` (two-point central difference). So `g2` is the FD
  curvature: `g2[i] = (f(x+h)+f(x−h)−2f(x))/h²` (`src/gradient.jl:312`).
- *AD seed* (`src/ad_gradient.jl:262` → `analytical_gradient`, lines 212-225):
  computes the rough initial gradient with `initial_gradient`, then
  `analytical_gradient!` (lines 184-208) overwrites only `grad` with the exact
  AD gradient and **propagates `g2`/`gstep` unchanged** from the rough seed.
  So `g2[i] = 2·up/dirin²` (`src/gradient.jl:94`), never refined.

Both then build the diagonal seed inverse-Hessian `V_diag[i] = 1/g2[i]`
(`src/seed.jl:138-139` and `src/ad_gradient.jl:270-272`).

### Step 2 — the first DFP step diverges (iteration 1)

The first MIGRAD search direction is `−V·g` (`src/migrad.jl:723-724`); the
accepted displacement is that direction scaled by the line search
(`src/migrad.jl:778`). The gradients are equal, so the direction difference is
driven entirely by the 6×–90× difference in `V_diag`. The most extreme is
parameter `r` (index 2), where the AD `V_diag` is **89× larger** → the AD path
takes a far larger first step in `r`:

```
seed : x=[3, 1e-4, -4]                      (both identical)
num it1: x=[3.0648, 4.893e-4, -4.0830]      Δx_from_AD = 0.0823
AD  it1: x=[3.0005, 1.832e-3, -4.0007]      ← r jumps ~18×, p0/a22 barely move
```

Per-iteration max-|Δx| between the two trajectories (from `ad_offset_probe.jl`):

```
iter   Δx(max abs)   Δfval
0      0.0           0.0          ← identical seed position
1      8.23e-02      6.45e+00     ← diverges here, from the seed-V difference
2      3.38e-01      7.76e-01
...    (both wander across the flat valley floor on different paths)
13     1.49e-02      7.81e-05     ← final
```

The divergence originates at the **iteration 0 → 1 transition** and is a pure
seed-curvature effect — confirming hypothesis 2 (seed `g2` inconsistency) as
the *mechanism*, not gradient precision during iteration.

### Step 3 — the flat valley converts "different trajectory" into "different minimum"

Both paths converge validly:

```
numerical: 11 iters, final edm = 7.37e-5
AD       : 13 iters, final edm = 2.45e-5
edm goal = tol·up·0.002 = 0.1·1·0.002 = 2e-4   (identical for both paths, src/migrad.jl:530-534)
```

Both final EDMs are below the same goal `2e-4`. On a well-conditioned fit the
two trajectories would re-converge to the same unique minimum (DFP corrects the
inverse-Hessian and a sharp minimum pins a single point). The X(3872) valley is
**not** well-conditioned: it is nearly flat along the coupling-degeneracy
direction, so "EDM below goal" is satisfied over an extended region of the
valley floor. The two trajectories enter that region at different places and
stop at different — but equally valid — points.

### How negligible is Δx? (offset vs 1σ)

From `ad_offset_sigma.jl` (parabolic HESSE errors, `σ = √(2·up·V_ii)`):

| par | x_num | x_ad | σ (1σ) | Δx | **Δx/σ** |
|---|---|---|---|---|---|
| p0  | 2.547796 | 2.538232 | 1.833 | 0.00957 | **0.0052** |
| r   | 0.0027329 | 0.0027567 | 0.004390 | 2.38e-5 | **0.0054** |
| a22 | −6.615084 | −6.630024 | 2.181 | 0.01494 | **0.0069** |

The offset is **~0.5–0.7 % of one standard deviation** in every parameter.
Because MINOS cannot close most of the 1σ contours on this fit (below), this `σ`
is the **parabolic-HESSE scale**, used here as an order-of-magnitude
negligibility check rather than a rigorous confidence interval — the true
intervals are wider still, making the offset even more negligible. MINOS
confirms the degeneracy directly — most 1σ crossings do not even exist:

```
p0 : lower invalid,  upper = +1.038
r  : lower invalid,  upper invalid   (fell back to parabolic ±0.00439)
a22: lower invalid,  upper = +1.836
```

A 0.0149 shift on a parameter whose 1σ band is ±2.18 (and whose lower MINOS
crossing runs off to infinity) is physically meaningless.

---

## Hypotheses evaluated

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Degenerate valley + EDM-stopping | **Confirmed (amplifier)** | Both edm < goal at different valley-floor points; MINOS can't close 1σ |
| 2 | Seed `g2` num/AD inconsistency | **Confirmed (mechanism)** | Seed `g2` differs 6×–90×; divergence starts at iter 0→1 |
| 3 | ForwardDiff chunk-size / Dual precision | **Ruled out** | chunk 1/2/3 give bit-identical gradients; AD vs FD grad agree to ~1e-8 |
| 4 | Convergence-tolerance asymmetry | **Ruled out** | `edmval = tol·up·0.002 = 2e-4` identical for both paths (`src/migrad.jl:530-534`) |

The full picture is **hypothesis 2 (mechanism) amplified by hypothesis 1
(degenerate valley)**.

---

## Is the seed-`g2` asymmetry a bug? No — it is C++ Minuit2-faithful

This is the crux of the verdict. C++ Minuit2 has the *same* asymmetry between
its numerical and analytical seed paths:

**Numerical seed** — `reference/Minuit2_cpp/src/MnSeedGenerator.cxx:60`:
```cpp
FunctionGradient dgrad = gc(pa);   // gc = Numerical2PGradientCalculator → refines g2 by FD
```

**Analytical seed** — `reference/Minuit2_cpp/src/MnSeedGenerator.cxx:119-122`:
```cpp
InitialGradientCalculator igc(fcn, st.Trafo(), stra);
FunctionGradient tmp = igc(pa);    // rough g2/gstep from step sizes
FunctionGradient grd = gc(pa);     // AnalyticalGradientCalculator → grad only (g2/gstep zeroed)
FunctionGradient dgrad(grd.Grad(), tmp.G2(), tmp.Gstep());  // exact grad + ROUGH g2/gstep
```

`AnalyticalGradientCalculator::operator()` returns `FunctionGradient(v)` with
only the gradient filled (`AnalyticalGradientCalculator.cxx:21-48`) — so C++
*must* borrow `g2`/`gstep` from the rough `InitialGradientCalculator`, exactly
as JuMinuit's `analytical_gradient!` propagates the rough seed's `g2`/`gstep`.

So both libraries seed:
- numerical path → **FD-refined** `g2`,
- analytical path → **rough `2·up/dirin²`** `g2`.

`iminuit` matches `jm_num` here *only because iminuit is also numerical* in this
benchmark. An iminuit fit driven by an analytical gradient would use the same
rough-`g2` analytical seed and would be expected to show the same kind of
offset. The design rationale is sound: the whole point of supplying an
analytical gradient is to *avoid* the `2n` FCN calls a numerical `g2` costs;
refining `g2` numerically on the analytical path would defeat that.

---

## Verdict (a): expected, no algorithm fix

The offset is:

- **deterministic** — same seed difference every run;
- **negligible** — 0.5–0.7 % of 1σ, on a fit MINOS calls degenerate;
- **C++-faithful** — the seed-`g2` asymmetry mirrors `MnSeedGenerator.cxx`;
- **two valid minima** — both satisfy the identical `edm < 2e-4` goal.

There is no unique "correct" minimum on a flat valley, so forcing the AD path to
match the numerical `x` (e.g. by snapping, or by numerically refining the AD
seed `g2`) would either break C++ faithfulness or merely paper over a
non-problem. **No change to the MIGRAD algorithm or the seed is recommended.**

### What *was* changed (documentation only)

- This document.
- A clarifying comment + a printed note in
  `BenchmarkExamples/X3872_dip/bench_full.jl` at the migrad cross-check, so the
  `⚠ MISMATCH` line for `jm_ad` is understood as expected on this degenerate
  fit rather than a regression. (The pass/fail logic is unchanged — a real
  mismatch on a *well-constrained* fit must still be flagged.)
- A "Known caveat" subsection in `README.md` next to the existing
  `Strategy(2)` cold-seed caveat.

### Optional future enhancement (out of scope, Beyond-C++)

ForwardDiff could supply the **exact diagonal Hessian** as the AD seed `g2`
(via `ForwardDiff.hessian` diagonal or nested duals) instead of the rough
`2·up/dirin²`. That would give the AD path a *better-than-numerical* seed (exact
curvature vs FD estimate) and would likely shrink — though not eliminate — the
offset, because the exact `g2` still differs from the FD `g2`. This is a genuine
"Beyond C++ Minuit2" feature, not a bug fix, and is deliberately left for a
future phase. It must not be implemented as a way to "make the numbers match."

---

## Files

- `scratch/ad_offset_probe.jl` — seed comparison, gradient/chunk checks,
  per-iteration trajectory + divergence table, high-level reproduction.
- `scratch/ad_offset_sigma.jl` — offset vs 1σ (parabolic) and MINOS intervals.
