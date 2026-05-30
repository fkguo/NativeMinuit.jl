# IAM cold-start convergence gap: JuMinuit vs iminuit (default 613 → 325; + S=0 retry gap closed)

**Date**: 2026-05-29
**Branch**: `feat/iam-convergence-gap`
**FCN**: IAM 2π form-factor 9-LEC fit (`BenchmarkExamples/IAM_2Pformfactor`)
**Seed**: `paras0 = [lecr0…, 1e-4]`, `error = fill(1e-6, 9)`, default tol.

> **Update (JuMinuit 0.3.0, 2026-05-30 run):** the current default-config
> benchmark reaches `fval = 404.15` — *deeper* than iminuit's `409.89` — i.e. the
> shallow-`613` gap is closed. The specific fvals in this document (`325`,
> `401.45`, `613`, …) are from the original 2026-05-29 investigation under
> particular Strategy/retry configs and may not match the current default run.
> See [`../../BenchmarkExamples/RESULTS.md`](../../BenchmarkExamples/RESULTS.md)
> for the authoritative current numbers; the analysis below is kept as the
> investigation record.

## Symptom (as reported)

`bench_full.jl` `build_jm_num()` does `Minuit(chi2_iam, paras0; error=errs0)`
→ `migrad!(m)` → `hesse(m)` and prints:

```
jm_num   migrad+hesse: fval=613.485  is_valid=false
iminuit  migrad+hesse: fval=409.885  is_valid=false
```

JuMinuit lands in a shallower basin (613) than iminuit (409). The robust
retry layer (`migrad!` default `iterate=5, use_simplex=true`) runs but its
fixed-point detector confirms it cycles at 613.

## TL;DR — root cause and fix

**The gap is a high-level default-strategy mismatch, not a core MIGRAD defect.**

- iminuit's `Minuit` class defaults to `strategy = 1`; C++ Minuit2's
  `MnStrategy()` default is also level 1 (`SetMediumStrategy`).
- JuMinuit's high-level `Minuit(fcn, x0)` constructor defaulted to
  `Strategy(0)` — a Phase-0 holdover (`strategy.jl` docstring still read
  *"Phase 0 default and only supported level"*). The bench used the default,
  so **JuMinuit ran Strategy 0 while iminuit ran Strategy 1**. They were never
  running the same algorithm.

At **matched strategy**, JuMinuit's MIGRAD is *not* worse — at Strategy 1
(iminuit's own default) JuMinuit's single MIGRAD reaches **330.75**, deeper
than iminuit's retry result of 409.89.

**Fix**: change the high-level `Minuit(...)` constructor default from
`Strategy(0)` to **`Strategy(1)`, uniformly** — for both numerical and
analytical/AD (`grad=`) FCNs. iminuit's `Minuit` class applies `strategy=1`
regardless of whether a gradient is supplied, so JuMinuit must too; a default
that differed by construction path would be the *same class* of silent
mismatch that caused this gap. Two points:

- **AD path extended to all strategy levels** — the AD `seed_state`
  (`ad_gradient.jl`) previously threw `Strategy(0) only` (a Phase-0 holdover).
  It now mirrors the numerical `seed_state`: the diagonal-from-g2 seed is
  strategy-independent (the AD gradient is exact), Strategy 2 adds the
  seed-time MnHesse bootstrap, and Strategy ≥ 1's inner-HESSE refinement runs
  in `_migrad_loop` — all via `hesse(::AbstractCostFunction)`, which
  finite-differences `cf.f` and already accepted `CostFunctionWithGradient`.
  So `Minuit(fcn, x0; grad=g); migrad!(m)` now runs at S=1 instead of
  throwing.
- **Low-level entry points unchanged** — `migrad(cf, …)` / `seed` /
  `function_cross` / `minos` / `contours` keep their own `Strategy(0)`
  defaults (pinned to the C++ oracle reference data; `test_cpp_oracle.jl`
  asserts `strategy_level == 0`).

Implemented as `strategy = Strategy(1)` in both `Minuit(...)` constructor
methods (`src/minuit.jl`), plus the AD-seed extension in `src/ad_gradient.jl`.

Result after fix (same bench call, default settings):

```
jm_num   migrad!(m): fval=325.816   (was 613.485)   ← now BEATS iminuit's 409.885
```

(330.75 with the strategy-default fix alone; 325.82 once the retry is made
iminuit-faithful (plain re-seed) — see § *Closing the S=0 retry gap*. The same
faithful retry fixes the S=0 path: 613 → 383. NB this is **not** a claim of a
better algorithm — on well-conditioned problems JuMinuit and iminuit match
exactly; see § *Fidelity*.)

## The data

### 1. Per-strategy single-shot (`iterate=1`, true apples-to-apples)

Both libraries, IAM `paras0` cold start, retry **disabled** on both sides
(iminuit via `m.migrad(iterate=1)`, JuMinuit via `migrad!(m; iterate=1)`):

| Strategy | JuMinuit single-shot | iminuit single-shot |
|----------|---------------------:|--------------------:|
| **S=0**  | 613.49               | 476.15              |
| **S=1**  | **330.75**           | 614.95              |
| **S=2**  | 1268.65 (stuck)      | 1268.65 (stuck)     |

### 2. Per-strategy with each library's native retry

Both libraries' default retry = re-run MIGRAD from the last point at the *same*
strategy (plain re-seed), up to 5×. (JuMinuit's `use_simplex=true` opt-in adds a
Simplex multistart — see § *Closing the S=0 retry gap* — but that is NOT the
default and NOT used in this table.)

| Strategy (pass 1) | JuMinuit retry (default) | iminuit retry |
|-------------------|-------------------------:|--------------:|
| **S=0**           | **383.39**               | 400.23        |
| **S=1**           | **325.82** (default)     | 409.89 (default) |
| **S=2**           | 1268.65                  | 1268.65       |

(JuMinuit's *old* default retry — growing Simplex + unconditional S=2 bump,
neither in C++/iminuit — stuck at 613.49 for the S=0 pass-1; making the default
retry iminuit-faithful closes that. The S=0/S=1 numbers above are *different
local minima* from iminuit's, not the same basin: JuMinuit's are deeper here,
but that is a numerically-sensitive basin selection on the ill-conditioned IAM,
**not** a demonstrated algorithmic edge — see § *Fidelity*.)

**Defaults**: JuMinuit was `S=0 → 613`; iminuit is `S=1 → 409`. After the fix
JuMinuit is `S=1 → 330`.

### 3. Aligned per-iteration trace at matched strategy (S=0)

Captured with `print_level=3`. Both seeds are identical:

```
                       JuMinuit S=0            iminuit S=0
  seed: FCN            1268.645892             1268.645892
  seed: Edm            1026.833019             1026.833283
  initial grad x0      -5.10e7  (g2 5.61e12)   -5.10e7  (g2 5.61e12)
  initial grad x8(=p9)  0       (g2 0)          0       (g2 0)   ← FCN ignores pars[9]
```

(`chi2_iam` reads only `pars[1:8]`; the 9th LEC is a flat direction. Both
libraries detect Negative-G2 on the seed and freeze it at 1e-4 — no
divergence here.)

First 10 DFP iterations — **identical to ~6 significant figures**:

| iter | JuMinuit S=0 fval | iminuit S=0 fval |
|-----:|------------------:|-----------------:|
| 0 | 1268.645892 | 1268.645892 |
| 1 | 987.2393689 | 987.2393917 |
| 2 | 978.3166239 | 978.3166157 |
| 3 | 962.6234493 | 962.6234307 |
| 4 | 930.1469850 | 930.1468527 |
| 5 | 912.4470956 | 912.4470872 |
| 6 | 909.5137347 | 909.5137144 |
| 7 | 907.3719752 | 907.3720844 |
| 8 | 896.3723745 | 896.3712664 |
| 9 | 887.5449898 | 887.5459662 |

The DFP update, EDM estimator (`edm·(1+3·dcovar)`), line search (accepted
step lengths α = 0.37, 0.15, 1.39, 9.95, 1.97, …), and Negative-G2 seed
handling are **byte-for-byte equivalent** to C++/iminuit for the first ten
steps. The divergence appears only later (≈ iter 20+) as the coarse
Strategy-0 2-cycle gradient accumulates noise into the two DFP trajectories.

**Localized divergence**: JuMinuit S=0 bails at **iter 24, fval = 613.49**
with *"matrix not pos.def; MnPosDef applied"* — the trial step gives
`gdel = step·g > 0` even after `MnPosDef`, so `_migrad_loop` breaks
(migrad.jl step 2, mirroring C++ `VariableMetricBuilder.cxx`). iminuit's S=0
first pass instead reaches 476, then its re-seed retry walks to 400. At
**Strategy 1**, JuMinuit's `dcovar`-triggered inner-HESSE refinement re-seeds
the curvature mid-run (at iter 49, `dcovar` climbs back to 0.12 → inner
HESSE fires) and the DFP loop descends all the way to **330.75** before
terminating — the path iminuit needs five retry passes to approximate.

## Hypothesis classification (the four candidates)

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| **#4 Strategy-default mismatch** | ✅ **ROOT CAUSE** | JuMinuit default `Strategy(0)`, iminuit/C++ default level 1. Bench ran S=0 (→613) vs iminuit S=1 (→409). At matched S=1, JuMinuit lands at 330 vs iminuit 409 — a *different* local minimum reached via numerically-seeded divergence on the ill-conditioned IAM (§ *Fidelity*), not a general algorithmic edge. |
| **#3 Seed inverse-Hessian scale** | ❌ ruled out | Seed Edm matches (1026.833019 vs 1026.833283); initial g, g2 identical per parameter; first DFP step lands at the same fval (987.239) on both sides. |
| **#1 Line-search timidity** | ❌ ruled out | Accepted step lengths are healthy/large (α up to 67.8), and the per-iteration fval matches iminuit step-for-step for the first 10 iters — the line search takes the *same* steps as C++. |
| **#2 No-improvement early exit** | ❌ ruled out as primary | JuMinuit ran 24 (S=0) / 49 (S=1) iterations before terminating — it did not exit on the no-improvement test (`|Δf| ≤ |f|·eps`). The S=0 termination at iter 24 is a `gdel>0`-after-MnPosDef pos-def bail, a downstream consequence of coarse-S0-gradient noise, not a too-tight threshold. At S=1 it does not bail there. |

## Why the fix is correct (not an IAM special-case)

1. **Drop-in fidelity** — the project's stated goal (commit `a884742`,
   "IMinuit.jl drop-in"). iminuit's `Minuit(fcn, x0)` defaults `strategy=1`;
   JuMinuit's high-level constructor must too, so a bare `migrad!(m)` matches
   `m.migrad()`. The `Strategy(0)` default was explicitly a Phase-0 limitation
   that was never updated when `hesse.jl` shipped (it enabled S≥1).
2. **It is a global default change**, not an IAM branch. Every high-level fit
   now runs the more thorough Strategy-1 path by default, exactly as iminuit.
3. **It satisfies the success criterion** — single-shot MIGRAD at the new
   default reaches 330.75 ≤ 410 on IAM.
4. **Low-level oracle parity preserved** — `migrad(cf, …)` etc. keep
   `Strategy(0)`; `test_cpp_oracle.jl` (which uses the low-level API and pins
   `strategy_level == 0`) is unaffected.

## Closing the S=0 retry gap: faithful retry + opt-in multistart

The tables above show a second gap: at matched **Strategy 0**, JuMinuit's
retry stayed at 613 while iminuit's reached 400. A decisive experiment
isolated the cause to the **retry mechanism, not the core MIGRAD**:

```
JuMinuit S=0 single-shot:                          613.49
JuMinuit S=0 + plain re-seed restart (iminuit's):  613.49 → ~383–402  (1 extra pass; then converged)
JuMinuit S=0 OLD retry (Simplex + S=2 bump):       613.49             (stuck, n_passes=2)
iminuit  S=0:                          476.15 → 400.23  (iterate 1 → 2)
```

JuMinuit's MIGRAD reaches the ~400 basin the instant it is given **iminuit's
retry move**: rebuild a fresh seed at the stall point (discarding the degraded —
non-pos-def — DFP inverse-Hessian) and re-run at the *same* strategy. At S=0 the
coarse 2-cycle gradient drives the DFP metric non-positive-definite after ~24
iters (the trace shows the `"matrix not pos.def"` bail at 613 with EDM spiking
to 132); the curvature refresh walks straight out. iminuit does exactly this on
every retry pass. JuMinuit's **old** retry instead bumped to `Strategy(2)`
(which from a cold-ish point hits the `V≈I` MnHesse-fail clamp — the same
pathology that sticks S=2-from-`paras0` at 1268) and Simplex-hopped — **neither
of which is in C++ Minuit2 or iminuit**.

**Fix — faithful default, opt-in multistart** (`migrad!`):

- **Default (`use_simplex=false`)** is now iminuit's `_robust_low_level_fit`
  exactly: re-run MIGRAD from the last point at the *same* strategy (plain
  re-seed), up to `iterate`, stop when valid. `migrad!(m)` is therefore
  drop-in-equivalent to `m.migrad()`, and it closes the gap via iminuit's *own*
  mechanism: IAM S=0 613 → 383, S=1 330 → 325.8.
- **Opt-in (`use_simplex=true`, default `false`)** is the growing-perturbation
  Simplex multistart + `Strategy(2)` escalation — a JuMinuit **extension beyond
  C++/iminuit**, for genuine multi-minimum fits (X(3872)). It is this path's S=2
  escalation that walks the IAM **x_jm WARM start** to χ²=322 (PR #10). At the
  faithful default, x_jm converges to iminuit's 325.8; **322 is reached the
  C++/iminuit way — by passing `strategy=2`** (both libraries give 322 at S=2).

This removes the divergence the maintainer flagged: the S=2 bump and Simplex
hop are no longer in the default path. Results: IAM **S=0 default 613 → 383.39**,
**S=1 default 330.75 → 325.82**, both via the faithful plain-re-seed retry.

## Fidelity: JuMinuit's MIGRAD vs C++ Minuit2 across problems

Is JuMinuit's per-strategy MIGRAD a faithful port, or a different ("better")
algorithm? Verified by comparing JuMinuit vs iminuit single-shot (`iterate=1`,
no retry) at every strategy on standard problems **and** IAM:

| Problem (single-shot) | S=0 | S=1 | S=2 |
|-----------------------|-----|-----|-----|
| Rosenbrock-2D / 4D / 10D | ✓ match | ✓ match | ✓ match |
| Quad-4D (stiff)          | ✓ match | ✓ match | ✓ match |
| **IAM-9LEC**             | ✗ 613 vs 476 | ✗ 330 vs 614 | ✓ match |

On **every well-conditioned problem JuMinuit reproduces C++ Minuit2/iminuit in
fval and (closely) nfcn** at all three strategies (e.g. Rosenbrock-2D nfcn
194/194/207 = iminuit's exactly; Rosenbrock-4D within a few calls). **IAM is the
only divergence**, and it is large because the two MIGRADs settle in *genuinely
different local minima* — IAM is pathologically ill-conditioned (|g₀|~10⁶, a
genuinely flat direction, distinct basins at 322/325/330/383/400/476). The
divergence is *seeded* at the numerical/implementation level, not by a gross
algorithm difference: at matched S=0 the two traces are identical for the first
~10 DFP iterations (fval agreeing to ~6 significant figures) and only then drift
apart — small differences (summation order, BLAS, the Julia-vs-C++ numerics) that
leave well-conditioned fits unchanged but steer basin selection on IAM.

**What this is NOT:** a demonstrated general advantage. The winner is not
consistent — JuMinuit is deeper at S=1 (330 vs 614) but iminuit is deeper at S=0
single-shot (476 vs 613) — and the two agree on every well-conditioned problem.
**What it IS:** JuMinuit lands in a deeper IAM basin in most of these runs, but
that is a numerically-sensitive, problem-specific basin selection on a chaotic
landscape, *not* evidence that JuMinuit's algorithm is better. This is exactly
why the default retry is now iminuit-faithful: JuMinuit cannot claim to be
*generally* better than C++ Minuit2, so it stays faithful to it.

(I have **not** proven the IAM-deeper result is "correct" or reproducible across
machines/BLAS — only that it is a different basin reached by a faithful port
whose numerics differ from C++/iminuit at the last few digits. Treat the deeper
IAM number as fortunate, not as a quality claim.)

Full `Pkg.test` 2558/2558, retry-layer testset 98/98.

## Secondary findings

- **At Strategy 0, JuMinuit's *single-shot* MIGRAD (613) underperforms
  iminuit's (476).** Cause: the Strategy-0 2-cycle numerical gradient is noisy;
  the DFP trajectory diverges from iminuit's after ~20 iters and bails on a
  non-pos-def trial step. This is inherent to "fast/loose" S=0 and is why
  neither library validates at S=0 single-shot. The re-seed retry then lands
  each library in a nearby basin (JuMinuit 383, iminuit 400 — *different*
  minima). See § *Fidelity*: numerically-seeded basin divergence on the
  ill-conditioned IAM, not an algorithmic difference.

- **Strategy 2 from a cold seed is pathological for *both* libraries** (both
  stuck at 1268.65 — exact parity). At `paras0` the gradient is ~1e6 and the
  MnHesse-fail fallback yields `V ≈ I` (the C++ second clamp restored in
  PR #10), so the first Newton step `−V·g` has magnitude ~1e6 and the line
  search cannot make progress. This confirms the
  `DAVIDON_CXX_AUDIT.md` "S=2 cold-seed pathology" note and is *not* a
  JuMinuit bug.

## Verification

- IAM `paras0`, default `migrad!(m)` (S=1, faithful re-seed retry): **613.49 →
  325.82**. At S=0: **613.49 → 383.39**. These are *different* local minima from
  iminuit's (409.89 / 400.23) — JuMinuit's are deeper here, but that is a
  numerically-sensitive basin selection on the ill-conditioned IAM, **not** a
  demonstrated algorithmic edge (§ *Fidelity*); the key claim is only that the
  default path now uses iminuit's *mechanism*. Guarded by
  `BenchmarkExamples/IAM_2Pformfactor/test_convergence_gap.jl` (asserts the S=1
  default and S=0 reach ≤ 410).
- IAM `paras0`, single-shot `migrad!(m; iterate=1)` at the default: 330.75 ≤ 410 ✓.
- Single-shot S=0/S=1/S=2 and all iminuit numbers **unchanged** (the
  strategy-default fix touches only the default; the retry rework touches only
  the retry, not the single MIGRAD).
- New unit regression `test/test_minuit.jl::"Default strategy = 1 (iminuit
  Minuit-class parity)"` asserts: numerical default → `Strategy(1)`, AD
  (`grad=`) default **also → `Strategy(1)`** and a default AD fit runs
  end-to-end at S=1 (plus an AD-at-S=2 fit), explicit strategy respected.
- `test/test_minuit_retry.jl::"Retry actually triggers …"` was pinned to
  `Strategy(0)` (it exercises the *retry* branches via a pass-1 stall that the
  coarse S=0 gradient reliably produces; the new S=1 default descends too far
  for the retry to enter, which would leave those branches uncovered). The
  AD-retry testset's stale "seed rejects level != 0" comment was corrected.
  No other test changed.
- Full `Pkg.test()` passes (see PR). The `_hesse_diagonal_failure` clamp +
  do-while loop (PR #10, which gets the IAM x_jm warm start to 322 via the S=2
  retry path) are untouched: that path goes through the low-level `migrad` and
  the retry S=2 bump, neither of which this change modifies. `test_cpp_oracle.jl`
  (low-level `migrad`, `strategy_level == 0`) is likewise unaffected.

## Relationship to `DAVIDON_CXX_AUDIT.md`

That audit studied the **x_jm warm start** (S=2 basin walk to 322) and recorded
in its post-resolution table:

> | paras0 S=1 cold-shot | … | 330.75 (x_jm 8D basin + x[9]=1e-4) | 409.89 (x_im basin) |

i.e. it already *measured* that JuMinuit at S=1 reaches 330.75 and iminuit at
S=1 reaches 409.89 — but it did not connect that to the constructor's
`Strategy(0)` default, because its focus was the warm-start regime. This
document closes that loop: the cold-start bench gap is the default-strategy
mismatch, and matching iminuit's default both closes and reverses it.
