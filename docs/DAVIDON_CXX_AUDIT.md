# DFP iter-1 EDM divergence audit (JuMinuit vs iminuit, IAM x_jm)

**Date**: 2026-05-28
**Branch**: `feat/davidon-cxx-audit`

## Symptom (from `iminuit` and JuMinuit traces at S=2 from x_jm)

```
iminuit (C++ Minuit2 v6):                JuMinuit (post PR #6):
  iter 0: fval=325.8015  edm=7.7e-6        iter 0: fval=325.8015  edm=7.7e-6
  iter 1: fval=325.8015  edm=2.2e6  ← !    iter 1: fval=325.8015  edm=1.9e-5
  iter 2..7: gradual descent                (loop exits: edm < edmval)
  iter 8: fval=322.5859 ← reaches deep basin (Strategy=2 inner HESSE fires,
  ...                                         then fails → "hesse-failed")
  Final: fval = 322.59 (deep basin)         Final: fval = 325.80 (stuck)
```

iter-1 EDM jumps by **11 orders of magnitude** between the two libraries. The
formula `edm = 0.5 · g_new' · V_old · g_new` is identical in
[src/edm.jl](src/edm.jl) and `reference/Minuit2_cpp/src/VariableMetricEDMEstimator.cxx`.
The DavidonErrorUpdator formulas in [src/davidon.jl](src/davidon.jl) and
`reference/Minuit2_cpp/src/DavidonErrorUpdator.cxx` are likewise identical (verified
line-by-line). So the divergence must be in either V_old (the seed V at iter 0
entry) or g_new (the gradient at the line-search exit point).

## Root cause

`reference/Minuit2_cpp/src/MnHesse.cxx:177-180` — the C++ `MnHesse`-failure
fall-back when one of the diagonal 2nd derivatives stays zero (g2[j] = 0 happens
at x_jm for parameter 9):

```cpp
for (unsigned int j = 0; j < n; j++) {
    double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
    vhmat(j, j) = tmp < prec.Eps2() ? 1. : tmp;   // ← SECOND CLAMP
}
```

Per-coordinate behavior at x_jm (typical |g2| ≈ 1e10, prec.Eps2() = 2.98e-8):

| g2[j] | tmp = 1/g2 | tmp < eps2? | C++ vhmat[j,j] |
|---|---|---|---|
| 1e10 | 1e-10 | **yes** | **1.0** ← clamped! |
| 0 | (skipped) | — | 1.0 |
| 1e8 | 1e-8 | yes (1e-8 < 3e-8) | 1.0 |
| 1e4 | 1e-4 | no | 1e-4 (genuine 1/g2) |

So C++ MnHesse-fail at x_jm produces **V ≈ I** (most diagonals clamped to 1).
The Newton step `−V·g ≈ −g` then has magnitude ≈ ‖g‖ ≈ 485 — orders of magnitude
larger than the natural parameter scale. The line search backtracks 11 times
(visible in iminuit's MnLineSearch trace) before landing at a useful slam, but
the net effect is that x moves substantially each DFP iteration. Across 8 DFP
iterations, the algorithm walks from x_jm into the χ²=322 basin.

[src/hesse.jl:438-449](src/hesse.jl:438) — JuMinuit's `_hesse_diagonal_failure`
**explicitly removes that second clamp** (PR #6):

```julia
v = abs(g2[j]) > prec.eps2 ? 1.0 / g2[j] : 1.0
M[j, j] = (isfinite(v) && v != 0.0) ? v : 1.0   # NO `< eps2 → 1` clamp
```

JuMinuit therefore produces `V ≈ diag(1/g2) ≈ diag(1e-10)` at x_jm. The Newton
step `−V·g ≈ 1e-8` per coord is essentially zero. The line search finds tiny
α, fval barely changes, edm drops, the inner DFP loop exits without exploring,
and MIGRAD terminates back at 325.8.

## The tension

PR #6's design doc (hesse.jl:405–422) argues that C++'s second clamp is a bug
**for the `paras0` regime** (initial seed with |g| ≈ 1e6, |g2| ≈ 1e16): with
V=I, the Newton step `−g` has magnitude 1e6, the line search blows up, and
MIGRAD bails. PR #6 fixed `paras0` by removing the clamp, but the same change
**broke** x_jm.

The two failure regimes need **opposite** V scales:

| Regime | ‖g‖ | Desired V | Newton step ‖−V·g‖ |
|---|---|---|---|
| x_jm warm start (near local min) | ≈ 500 | I (so step ≈ ‖g‖ ≈ 500) | crosses basin |
| paras0 cold start (huge gradient) | ≈ 1e6 | diag(1/g2) ≈ 1e-10·I (so step ≈ 1e-4) | no blowup |

The C++ second clamp ALWAYS picks V≈I. PR #6 ALWAYS picks V≈diag(1/g2). Neither
is uniformly right.

## Options to fix x_jm without re-breaking paras0

1. **Revert PR #6's clamp removal**. Restores x_jm → 322 (matches iminuit).
   Re-breaks paras0 → line-search blowup → MIGRAD bails. Net change:
   correctness now matches C++ Minuit2 exactly, but paras0 needs the retry
   layer (Simplex hop) to recover — which PR #4 added.

2. **Per-coord adaptive clamp** based on |g[j]|. If |g[j]| is small, V[j,j]
   should be O(1) (so step is meaningful). If |g[j]| is large, V[j,j] should be
   O(1/|g|) (so step stays bounded). For example:
   `V[j,j] = min(1.0, max(1/g2[j], step_max / |g[j]|))`. Not textbook; would
   need empirical tuning.

3. **Two-stage fallback**: try V = diag(1/g2) first; if MIGRAD bails with
   "no improvement", retry with V = I. This is the spirit of PR #4's retry
   layer but with finer granularity inside `_hesse_diagonal_failure`.

4. **Leave PR #6 as-is and rely on the retry+Simplex layer to find 322 from
   x_jm**. Empirically tested (this audit): JuMinuit DFP + retry from x_jm
   stays at 325.8 (Simplex hop doesn't perturb enough to escape). Would need
   to make Simplex hops larger / multi-scale.

## Recommended next step

**Option 1** — revert PR #6. This makes JuMinuit's `_hesse_diagonal_failure`
match C++ Minuit2 exactly. Verify both:
- IAM x_jm warm start: reaches χ²=322.59 (matches iminuit)
- IAM paras0 cold start + retry layer (PR #4 #8): doesn't regress catastrophically

If paras0 regresses, decide whether the retry layer + a less-aggressive HESSE
trigger is enough, or whether option 2 (per-coord adaptive clamp) is needed.

## Other findings during the audit

- `negative_g2_line_search` ([src/negative_g2.jl](src/negative_g2.jl)) fires at
  x_jm because g2[9]=0 triggers `has_negative_g2`. The line search's per-coord
  step refinement uses `step_size = 0` for all 9 coords (per the iminuit
  trace), so it effectively doesn't move x — both libraries match here.

- The `while edm_corrected > edmval` check-first form of JuMinuit's inner DFP
  loop (currently on main) skips iteration entirely when the warm-start edm
  is below threshold. C++ uses do-while (`while (cond) … } while (cond);`)
  semantics. The stashed do-while fix is necessary but **not sufficient** to
  reach χ²=322; the V=I clamp issue dominates. Apply both for full parity.

- The SR1+TR backend on `feat/sr1-trust-region` cannot fix this — trust region
  by design bounds step magnitude, while the basin escape from x_jm requires
  the OPPOSITE (a giant step). Documented in
  `docs/SR1_TR_DESIGN.md` §2.6. That branch is shelved (option B from the
  earlier session triage: keep as reference, do not merge); the work served as
  the motivation that led to this audit.

## Resolution

Adopted **option 1 + do-while + status-gated entry shortcut** (per-iter cost
hybrid) across three commits on this branch:

1. **`fix(hesse): restore C++ MnHesse.cxx:177-180 second clamp (revert PR #6)`**
   — `_hesse_diagonal_failure` now mirrors C++ exactly. At IAM x_jm warm start,
   the MnHesse-fail fallback produces `V ≈ I` (was `V ≈ diag(1/g2) ≈ 1e-10·I`
   under PR #6).

2. **`feat(migrad): do-while inner loop + status-gated entry shortcut`** —
   `_migrad_loop`'s inner DFP loop converted from check-first
   (`while edm > edmval`) to do-while (`while true … break` with termination
   check at the bottom), mirroring C++ `VariableMetricBuilder.cxx:237/341`
   semantics exactly.

   Added a status-gated entry shortcut at the top of the loop: when
   `(edm_corrected <= edmval) && (status == MnHesseValid)`, skip the body
   immediately. This is a documented JuMinuit-only divergence from C++ that
   preserves the warm-restart `function_cross_multi` / MINOS / contour_exact
   contract (no-op when seed is already converged and V is trustworthy).
   The placeholder-V cases (status `MnHesseFailed` / `MnMadePosDef` /
   `MnInvertFailed`) bypass the shortcut — which is exactly where do-while
   semantics matter for correctness (the IAM x_jm basin walk).

3. The companion test `test/test_hesse.jl::"Regression: _hesse_diagonal_failure
   preserves 1/g2 for huge g2"` was inverted to assert `V[1,1] == 1.0` (the
   restored clamp) rather than `V[1,1] < 1e-5`.

### Empirical verdict (post-resolution)

| Config | Pre-resolution | Post-resolution | iminuit reference |
|---|---|---|---|
| x_jm S=2 single-shot | 325.80 (wrong basin) | **322.5966** (correct basin) | 322.5859 |
| x_jm S=2 + retry | 325.80, nfcn=64 | **322.5966**, nfcn=123 | n/a |
| x_jm S=0/1 single-shot | 325.80, nfcn=37 | 325.80, nfcn=37 (warm-restart no-op preserved by status gate) | 325.80 |
| paras0 S=2 cold-shot | 325.80 (via retry+Simplex) | **1268.65 stuck** (matches iminuit) | 1268.65 stuck |
| paras0 S=1 cold-shot | 325.80 (different basin) | 330.75 (x_jm 8D basin + x[9]=1e-4) | 409.89 (x_im basin) |

paras0+S=2 trap: real regression vs PR #6, **but exact parity with iminuit
S=2**. Mitigated by the standard HEP workflow `S=0/1 from cold → polish at
S=2 after entering a basin`. See README "S=2 cold-seed pathology" note.

### Branches involved

- `feat/sr1-trust-region` — the SR1+Trust-Region backend exploration that
  led to identifying this issue. Shelved (not merged) per the session triage;
  no longer needed once `feat/davidon-cxx-audit` lands.
- `feat/davidon-cxx-audit` — this branch. Contains the audit doc, the three
  resolution commits, and updated regression test.
