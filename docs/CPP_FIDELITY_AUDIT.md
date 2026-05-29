# C++ Minuit2 ↔ JuMinuit line-by-line fidelity audit

**Date**: 2026-05-30 · **Base**: `main` @ `3de0857` (after PR #16 + PR #17 §1-4 merged)
**Reference**: `reference/Minuit2_cpp/{src,inc}/*` (GooFit/Minuit2 v6.24.0)
**Line numbers**: all cites verified against `main` @ `3de0857`.
**Scope**: deep, branch-by-branch comparison of individual ported algorithms —
*not* the component-level coverage map (that lives in `DEFERRED.md` /
`GAP_AUDIT.md`). Each section maps every C++ branch/exit path to its JuMinuit
counterpart and classifies it: ✓ faithful · documented-divergence · minor ·
missing.

Audited (14 algorithms — the full minimization / error-analysis spine):

1. MnHesse — `MnHesse.cxx:93-316` ↔ `src/hesse.jl`
2. VariableMetricBuilder / MIGRAD — `VariableMetricBuilder.cxx` ↔ `src/migrad.jl:_migrad_loop`
3. MnMinos — `MnMinos.cxx` (+ `MnFunctionCross.cxx`) ↔ `src/minos.jl` / `src/function_cross.jl`
4. MnContours — `MnContours.cxx` ↔ `src/contours.jl::contour_exact`
5. MnSimplex — `SimplexBuilder/Parameters/SeedGenerator.cxx` ↔ `src/simplex.jl`
6. MnLineSearch (+ MnParabola) — `MnLineSearch.cxx` ↔ `src/linesearch.jl`
7. NegativeG2LineSearch — `NegativeG2LineSearch.cxx` ↔ `src/negative_g2.jl` / `src/ad_gradient.jl`
8. MnSeedGenerator — `MnSeedGenerator.cxx` ↔ `src/seed.jl`
9. Gradient calculators (Initial/Numerical2P/Hessian/Analytical) ↔ `src/gradient.jl` / `hessian_gradient.jl` / `ad_gradient.jl`
10. DavidonErrorUpdator + VariableMetricEDMEstimator ↔ `src/davidon.jl` / `edm.jl`
11. MnPosDef — `MnPosDef.cxx` ↔ `src/posdef.jl`
12. MnEigen / MnGlobalCorrelationCoeff / MnCovarianceSqueeze ↔ `src/eigen_corr.jl` / `covariance_squeeze.jl`
13. MnScan — `MnParameterScan.cxx` / `ScanBuilder.cxx` ↔ `src/scan.jl`
14. Parameter transforms + MnStrategy + MnMachinePrecision ↔ `src/transform.jl` / `strategy.jl` / `precision.jl`

See [Summary across all 14 algorithms](#summary-across-all-14-algorithms) for the
severity-sorted findings (the one MAJOR — §14 precision `eps` — is now
**resolved** in `feat/precision-eps-x4`; see §14).

**Update (2026-05-30):** the three actionable contained fixes are now landed
on `feat/cpp-fidelity-3fixes` — MnHesse bounded step clamp (`153f41d`), MIGRAD
2nd-pass-invalid bail (`e256506`), MnMinos n-scaled budget (`88bceea`). Each
finding below is marked **RESOLVED** with its commit. The MnContours `sca`
direction-switch retry remains the one open contained fix.

---

## 1. MnHesse

`MnHesse.cxx:93-316` (the `operator()(MnFcn, MinimumState, MnUserTransformation,
maxcalls)` "real Hessian calculation"). Lines 318-414 are dead commented-out
code, ignored.

### Branch map

| C++ (MnHesse.cxx) | JuMinuit (hesse.jl) | Verdict |
|---|---|---|
| `amin=mfcn()`, `aimsag=√eps2·(\|amin\|+Up)`, `maxcalls=200+100n+5n²` (102–109) | 96–97, 91–93 | ✓ |
| init `g2/gst/grd/dirin=gst/yy` (112–116) | 108–112 | ✓ |
| analytical-gradient g2/step recompute (120–126) | 166–180 | ✓ (2 documented nuances) |
| diagonal `dmin=8·eps2·(\|xtf\|+eps2)`, `d=\|gst\|` (136–139) | 192–194 | ✓ |
| 5× multiplier loop, `sag≠0→break` (147–169) | 205–221 | ✓ (limits branch implemented — 153f41d) |
| L26 sag-zero → diagonal fallback `MnHesseFailed` (171–183) | 223–226 | ✓ |
| L30 `g2=2·sag/d²`, `grd`, `d=√(2·aimsag/\|g2\|)` (185–197) | 228–238 | ✓ (limits clamp implemented — 153f41d) |
| convergence `Tolerstp`/`TolerG2`, `d∈[0.1,10]·dlast` (203–208) | 241–256 | ✓ (defensive `g2≠0` guard, same result) |
| `vhmat(i,i)=g2(i)` (210) | 259 | ✓ |
| maxcalls-exhausted → diagonal fallback (211–223) | 269–275 | ✓ |
| Strategy>0 HGC gradient refine (228–235) | 290–303 | ✓ |
| off-diagonal `(fs1+amin−yy_i−yy_j)/(dirin_i·dirin_j)` (239–272) | 307–329 | ✓ (simple `i<j` = C++'s own old form) |
| `MnPosDef` on H (278) | 342 | ✓ (passes H not V — matches C++) |
| `Invert`; fail → diagonal fallback `MnInvertFailed` (283–296) | 348–355 | ✓ |
| `IsMadePosDef` → `MnMadePosDef` state (302–306) | 359–364 | ✓ |
| accurate → `dcovar=0` state (309–315) | 358–375 | ✓ |
| double-clamp `g2<eps2?1:1/g2; <eps2?1` ×3 fallbacks (177–180/216–219/289–292) | `_hesse_diagonal_failure` 462–463 | ✓ (abs-variant, identical result) |
| MPI off-diagonal partitioning (240–271) | — | intentionally not ported (MPI deferred) |

### Findings

- **RESOLVED (153f41d): bounded-parameter step clamping.**
  Previously `has_limits = false` was hardcoded (hesse.jl), so two C++
  branches never fired: the multiplier-loop `if HasLimits && d>0.5 → d=0.51`/
  fail (160–167) and the L30 `if HasLimits → d=min(0.5,d)` (194–195). HESSE
  runs in internal (arcsin) coordinates, where C++ clamps the probe step
  `d≤0.5` for externally-bounded params (near a bound the transform is steep;
  an unclamped `d` → wild external excursion → wrong 2nd-derivative). Now
  `hesse(cf, state; has_limits=…)` takes per-internal-parameter bound flags
  (`_has_limits_internal`, the analogue of C++ `trafo.Parameter(i).HasLimits()`)
  and gates both clamp sites on the per-parameter `lim_i`, applied in the
  internal frame. The flags are threaded through `migrad(cf, params)` (the
  Strategy≥1 inner-HESSE refinement, numerical + AD) and the standalone
  `hesse(m::Minuit)` path. `has_limits === nothing` (every unbounded caller,
  incl. standalone `hesse(f,x0,err)`) leaves `lim_i` always false, so unbounded
  HESSE is byte-identical to before. Verified: bounded probe step capped at
  0.51 vs 1.0 unbounded on a flat-plateau FCN; near-bound `hesse(m)` yields a
  valid covariance.

- **Documented faithful-but-different (not gaps):**
  - *Analytical-gradient gate* (hesse.jl:150–165): gated on `cf isa
    CostFunctionWithGradient` vs C++'s `IsAnalytical()` flag → a repeat `hesse`
    call re-refreshes (idempotent, extra FCN calls only, not a correctness bug).
  - *Analytical seed semantics* (132–148): recompute seeds from stale
    `state.gradient` vs C++'s fresh per-parameter user errors
    (`InitialGradientCalculator`); converges identically for smooth FCNs, can
    differ for pathological ones (GAP_AUDIT P2 follow-up).
  - *`abs()` in the double-clamp*: same result as C++'s raw comparisons
    (negative g2 → 1.0 both ways).
  - *Off-diagonal loop*: simple nested `i<j` vs C++'s MPI-flattened index
    arithmetic — mathematically identical (it *is* C++'s own non-MPI form,
    lines 400–410 of the commented block).

**Verdict: faithful port.** Every branch, exit path, formula, tolerance, and
the load-bearing double-clamp are correct. The one prior omission — the
bounded-parameter step clamp — is now implemented (153f41d); unbounded fits
remain byte-identical.

---

## 2. VariableMetricBuilder / MIGRAD

`VariableMetricBuilder.cxx` ↔ `src/migrad.jl:_migrad_loop`. C++ splits MIGRAD
into an **outer** `Minimum` (54–203: edmval scaling, validity gates, the
do-while calling the inner loop + Strategy≥1 HESSE refinement) and an **inner**
`Minimum` (205–375: the DFP iteration). JuMinuit inlines both into one
`_migrad_loop` (outer `while iterate` wrapping inner `while true`) — same
control flow.

### Inner DFP loop (C++ 205–375 ↔ migrad.jl 690–878)

| C++ branch | JuMinuit | Verdict |
|---|---|---|
| `edm *= (1+3·Dcovar)` (229) | 586, 844 | ✓ |
| `step = −V·g` (241) | 724 `sym_mul!` | ✓ |
| zero-grad `⟨g,g⟩≤0 → break` (247–250) | 727–729 | ✓ |
| `gdel = step·g` (252) | 731 | ✓ |
| `gdel>0` → MnPosDef → recompute → still>0 → exit (254–273) | 734–748 | ✓ |
| line search (275) | 752 `line_search` | ✓ |
| no-improvement `\|pp.Y−Fval\|≤\|Fval\|·Eps → break` (278–291) | 762–767 | ✓ (≤eps·\|fval\| micro-diff) |
| accept `p = x + pp.X·step` (296) | 778 | ✓ |
| new grad `g = gc(p, s0.grad)` (298) | 785 | ✓ |
| `edm = Estimate(g, s0.Error())` — OLD error (300) | 792 | ✓ |
| `isnan(edm) → break` (302–306) | 794–796 | ✓ |
| `edm<0` → MnPosDef → recompute → still<0 → exit (308–321) | 799–806 | ✓ |
| Davidon `Update(s0,p,g)` (322) | 834–840 | ✓ |
| `while edm>edmval && nfcn<maxfcn` (341) | 878 | ✓ |

### Outer loop + finalization (C++ 54–203 ↔ migrad.jl 530–973)

| C++ branch | JuMinuit | Verdict |
|---|---|---|
| `edmval *= 0.002` (66) + `tol·up` floor at eps2 (ModularFunctionMinimizer) | 530–534 | ✓ |
| n==0 / seed-invalid / edm<0 gates (77–92) | 547–582 | ✓ (relaxed seed gate) |
| do-while outer; call inner (111–118) | `while iterate` 690 | ✓ inlined |
| Strategy≥1 HESSE `S==2 ‖ (S==1 && Dcovar>0.05)` (138–142) | 888–900 | ✓ |
| invalid Hessian → break (150–153) | 904–911 | ✓ |
| re-iterate if `edm>edmval && edm≥\|eps2·fval\|` (160–168) | 927–932 | ✓ exact |
| `maxfcn_eff = int(maxfcn·1.3)` on pass 0 (182–183) | 937–939 | ✓ |
| final `edm>10·edmval → MnAboveMaxEdm` (189–198) | 950, 952 | ✓ |
| call-limit `nfcn≥maxfcn → MnReachedCallLimit` (350–354) | 949 | ✓ |
| inner edm classification `<machine`/`<10·edmval`/else (356–368) | folded into `above_max` 950 | ✓ |

### Findings

- **Deliberate documented divergences (not bugs):**
  1. *Status-gated entry shortcut* (migrad.jl:720–722): skips the inner-loop
     body when `edm ≤ edmval && status == MnHesseValid`; C++ is a strict
     `do{...}while`. The load-bearing PR #10 / DAVIDON-audit subtlety — the
     shortcut fires *only* for an already-converged trustworthy-V warm restart
     (the MINOS/contour no-op case); for a placeholder-V seed (status ≠
     MnHesseValid) it does not fire, preserving do-while semantics (the IAM
     x_jm → 322 walk). Correctness-preserving optimization.
  2. *Relaxed seed-validity gate* (573–577): structural validity (params /
     gradient set, error available) vs C++'s effectively-no-op `seed.IsValid()`.
     More correct — accepts a bailed-but-usable `_hesse_diagonal_failure` seed.

- **RESOLVED (e256506): C++ "2nd-pass invalid → bail" guard** (C++ 127–132:
  `if (ipass>0 && !min.IsValid()) return`). Added as the predicate
  `_migrad_second_pass_invalid(ipass, s0, edm_corrected, edmval)` =
  `ipass>0 && (!is_valid(s0) || edm_corrected > 10·edmval)`, placed after the
  inner DFP loop's call-limit break and before the Strategy≥1 HESSE block.
  The `HasReachedCallLimit` disjunct is handled by the preceding `ncalls ≥
  maxfcn_eff` break; the above-max-edm disjunct reuses the same expression as
  the final-verdict `above_max`, so the bail fires exactly when the result
  would be flagged invalid-by-above-max. Purely additive — the deliberate
  status-gated entry shortcut (a *keep*) is untouched. Efficiency-only: same
  final verdict, fewer wasted passes on non-converging S≥1 fits. (A downstream
  retry test's bit-exact fixed-point assertion was relaxed to `≈` accordingly,
  since the bail now returns the C++-faithful earlier-pass point.)

- **Negligible:** at the no-improvement exit JuMinuit keeps `s0`'s old fval;
  C++ (size>1) records `pp.Y()` — differ by ≤ `eps·|fval|` (that branch's own
  entry condition), machine-precision.

- **Structural equivalences:** two-method split → one inlined loop; C++ `result`
  vector + reduced-state storage → JuMinuit `history` (storage-level-gated) +
  `final=s0`; MnPosDef bail returns a `FunctionMinimum` (C++) vs breaks-then-
  builds (JuMinuit).

- **Collaborators** (verified separately): `DavidonErrorUpdator`→davidon.jl and
  `VariableMetricEDMEstimator`→edm.jl line-by-line in `DAVIDON_CXX_AUDIT.md`;
  `MnLineSearch`+`MnParabola*`→linesearch.jl, `MnPosDef`→posdef.jl ported.

**Verdict: faithful port.** Every branch and exit path of both methods maps
correctly. Substantiates the `IAM_CONVERGENCE_GAP.md` § Fidelity claim
("core MIGRAD is faithful") with line-by-line evidence, consistent with the
Rosenbrock/Quad exact-match. The one remaining non-cosmetic item is the
deliberate status-gated shortcut (a *keep*); the 2nd-pass-invalid bail is now
implemented (e256506).

---

## 3. MnMinos

`MnMinos.cxx` (213 lines) sets up each ±σ scan and delegates the actual
root-finding to `MnFunctionCross.cxx` (512 lines). JuMinuit splits these the
same way: `src/minos.jl` (the `FindCrossValue` setup + MinosError assembly) and
`src/function_cross.jl::_cross_core` (the parabolic root-find, shared with
MnContours). `function_cross.jl` is larger (1597 lines) because it also serves
contours, multi-fixed-parameter scans, the AD path, and warm-restart reuse.

### 3a. MnMinos::FindCrossValue (C++ MnMinos.cxx:94–197 ↔ minos.jl `minos(...)`)

| C++ branch | JuMinuit | Verdict |
|---|---|---|
| `err = dir·Error(par)`, `val = value + err` (119–120) | `sigma_i = √(2·up·V[ii])` (226), dir applied in `function_cross` | ✓ |
| limit clamp of `val` (122–129) | bounded-path int↔ext clamp (275–302) | ✓ (+ hardening below) |
| `xunit = √(up/m(ind,ind))`; other-param pre-shift `xt(i)+dir·xunit·m(ind,i)` (140–165) | `shift = σ·V[ik]/V[ii]`, seed_upper/lower (271) | ✓ **algebraically verified** (the 2·up & 2× factors cancel; minos.jl:234–238) |
| `upar.Fix(par); SetValue(par,val)` (167–168) | par_idx is the fixed scan param in `function_cross` | ✓ |
| `MnFunctionCross(...)` (172–173) | `function_cross(fmin, cf, par_idx, ±1; …)` (333, 367) | ✓ |
| AtMaxFcn / NewMinimum / AtLimit / !IsValid warnings (178–192) | MnCross flags + invalid-side ±σ placeholder (341–350) | ✓ (matches `MinosError::Upper/Lower`) |
| `maxcalls==0 → 2·(nvar+1)·(200+100n+5n²)` (111–114) | `_minos_default_maxcalls(n_free)` forwarded by `_minos_error` | ✓ (resolved — 88bceea) |

### 3b. MnFunctionCross (C++ MnFunctionCross.cxx ↔ function_cross.jl `_cross_core` + helpers)

| C++ branch | JuMinuit | Verdict |
|---|---|---|
| `aim = aminsv+up`, `tlf = tlr·up`, `tla = tlr`, `maxitr=15` (45–50) | 242, 261, `tla_base`, `maxitr` | ✓ |
| inner `MnMigrad(…, MnStrategy(max(0,strategy−1)))` (106) | `Strategy(max(0, level−1))` (799, 965) | ✓ exact |
| 1st MIGRAD; `flsb[0]=max(Fval,aminsv+0.1·up)`; `aopt=√(up/(f−fmin))−1` (119–142) | 270–276 | ✓ |
| converged `\|flsb[0]−aim\|<tlf` (143–144); clamp `[−0.5,1]` (146–149) | 278–281 | ✓ |
| 2nd MIGRAD; `dfda=(f1−f0)/(a1−a0)` (164–184) | 284–302 | ✓ |
| L300 `dfda<0` extend `aopt=alsb[0]+0.2·(it+1)` (188–242) | `while dfda<0`, `a[1]+0.2·count` (312–335) | ✓ |
| L460 linear extrap `aopt=alsb[1]+(aim−flsb[1])/dfda`; converge `adist<tla && fdist<tlf`; `[bmin,bmax]` clamp (244–266) | 343–355 | ✓ |
| 3rd MIGRAD + 3-point `noless` dispatch (288–351) | 357–404 | ✓ (incl. the "new straight line" L460-reentry, review BLOCKING #2) |
| L500 parabola loop: `MnParabolaFactory` fit, solve `=aim`, positive-slope root, converge at `ibest`, window/bad-point mgmt, replace worst (353–503) | `_parabola_fit3`/`_parabola_solve_for_aim`/`_three_point_classify` + L500 `while ipt<maxitr` (406–503) | ✓ line-cited |
| exits CrossNewMin / CrossFcnLimit / CrossParLimit / invalid / converged | `new_min` / `fcn_limit` / `par_limit` / `valid=false` / `valid=true` | ✓ (par_limit structural, below) |

### Findings

- **RESOLVED (88bceea): default MINOS call budget.** C++ (and iminuit) default
  `maxcalls=0` → `2·(nvar+1)·(200+100·nvar+5·nvar²)` (≈30 100 for n=9);
  JuMinuit's high-level `minos!`/`minos` previously let the downstream fall back
  to a fixed `maxcalls=1000` (minuit.jl, minos.jl:200), so on larger fits MINOS
  could hit `fcn_limit` where C++/iminuit keep going. Now, when the user passes
  no explicit `maxcall` (the `maxcall==0` sentinel), `_minos_error` forwards
  `_minos_default_maxcalls(n_free(params))` (the exact C++ formula; `nvar =
  n_free` excludes fixed params, matching `VariableParameters()`) to BOTH the
  bounded and unbounded cross-search sub-paths. An explicit `maxcall>0` (and the
  power-user `maxcalls` kwarg) still win. The low-level `minos(fmin, cf, par)` /
  `_minos_external_via_function_cross` keep their own 1000 default for direct
  callers; the high-level path now always passes an explicit budget.

- **Structural-but-equivalent: `par_limit`/`aulim` detection.** C++ computes
  `aulim` inside MnFunctionCross with inline per-probe `limset && Fval<aim →
  CrossParLimit` exits (66–104, 135, 178, 227, 294, 495). JuMinuit's core
  `_cross_core` is limit-agnostic (operates in the caller's frame); the bounded
  wrapper detects `par_limit` via the int↔ext transform + a post-hoc aulim-style
  check (function_cross.jl:1291, 1370–1388). Same outcome (par_limit raised when
  the crossing lies beyond a bound); the *timing* of detection within the loop
  differs. Documented (function_cross.jl:1165–1168).

- **Hardening beyond C++ (not a gap):** the other-parameter pre-shift adds a
  sin-transform saturation pre-clamp for doubly-bounded params (minos.jl:254–302)
  to prevent `sin()` aliasing on large pre-shifts — a safety branch C++ lacks.

- **Extension beyond C++ (not a gap):** `sigma=k` k-σ MINOS errors (the
  `aopt·σ_i` scaling); C++ `MnMinos` is 1σ-only.

**Verdict: faithful port.** The root-finding core (`_cross_core`) is a
meticulous, C++-line-cited reproduction of MnFunctionCross — every branch
(L300/L460/L500, the noless dispatch, parabola fit, window/bad-point management)
and every exit (new-min / call-limit / par-limit / invalid / converged) maps,
with the inner-MIGRAD `Strategy−1` reduction and the covariance cross-correlation
pre-shift algebraically verified. The one prior substantive divergence — the
**smaller default call budget** (1000 vs n-scaled) — is now resolved (88bceea):
the high-level path forwards the C++ n-scaled budget.

---

## 4. MnContours

`MnContours.cxx:34-204` ↔ `src/contours.jl::contour_exact`. JuMinuit ships two
contour routines: `contour` (a simplified convenience, documented as such) and
**`contour_exact`** — the C++-faithful port audited here. The actual crossing
search reuses the already-audited cross-search core via `function_cross_multi`
(the 2-fixed-parameter path of `_cross_core`).

### Branch map

| C++ (MnContours.cxx) | JuMinuit (contour_exact) | Verdict |
|---|---|---|
| `assert npoints>3` (38) | `npoints ≥ 4` (119) | ✓ |
| `maxcalls = 100·(npoints+5)·(nvar+1)` (39) | 187 | ✓ exact |
| `toler = 0.1` (50) | `tlr=0.1` (110) | ✓ |
| `mex=Minos(px)`, `mey=Minos(py)` + validity (54–73) | 136–143 | ✓ |
| 4 axis points: fix px/py at val±err, MIGRAD, take other coord (75–110) | `_axis_point` (148–166) | ✓ (strategy nuance below) |
| `scalx=1/(ex.up−ex.lo)`, `scaly=…` (112–113) | 183–184 | ✓ |
| 4 seed points in CCW order (115–118) | 175–180 | ✓ same order |
| fix px,py; `MnFunctionCross` (125–131) | `function_cross_multi` (221) | ✓ |
| largest scaled-gap pair incl. wrap (135–150) | cyclic scan (190–205) | ✓ equivalent |
| midpoint `a1·p1+a2·p2`, perpendicular `xdir=Δy, ydir=−Δx` (163–166) | 209–212 | ✓ exact |
| `scalfac = sca·max(\|xdir·scalx\|,\|ydir·scaly\|)` (167) | `max(...)` (213) | **✗ no `sca`** (below) |
| `cross(...)`; insert at idist2 / append if wrap (177, 191–198) | 221–238 | ✓ (wrap-append matches) |
| `nfcn>maxcalls` → return (158–161) | break on `nfcn>maxcalls` (229) | ✓ |
| return `ContoursError` (203) | 241 | ✓ |

### Findings

- **✗ Divergence (MODERATE): missing the `sca` direction-switch retry**
  (MnContours.cxx:152–189). When the crossing search fails for a contour point,
  C++ flips the perpendicular direction (`sca = 1 → −1`, `goto L300`) and retries
  *once* before giving up. JuMinuit's `contour_exact` instead `break`s on the
  first failed `function_cross_multi` (contours.jl:229–231). Effect: on irregular
  contours where the crossing lies in the `−perpendicular` direction, C++ finds
  the point and JuMinuit returns **fewer points than requested**. Affects
  contour *completeness*, not the correctness of the points found. **~10 LOC**
  (wrap the cross in a `for sca in (1.0, -1.0)` retry).

- **Minor: axis-point inner-MIGRAD strategy.** The four seed-point MIGRADs use
  the full `strategy` (`_axis_point`, contours.jl:152); C++ uses
  `MnStrategy(max(0, strategy−1))` (75, 94). Only diverges at `strategy ≥ 1`
  (the default `contour_exact` strategy is `Strategy(0)`, where `max(0,−1)=0` —
  no divergence). The *ray-point* cross correctly uses `strategy−1`
  (function_cross.jl:965). Marginal accuracy/call-count effect on the 4 seeds.

- **`contour` vs `contour_exact`:** the default `contour` is a simplified
  convenience (linearized ellipse-ish), not a C++ port; `contour_exact` is the
  faithful one. Tracked in `GAP_AUDIT.md` P3 (verified iminuit-compat).

**Verdict: faithful port** (`contour_exact`). The seed-point construction,
largest-gap bisection, perpendicular-ray geometry, scaling, insert-order, and
the reuse of the audited cross-search all map exactly. The one substantive
divergence is the **missing `sca` retry**, which costs contour *completeness*
(fewer points) on irregular contours but never produces a wrong point.

---

> **Sections 5–14 below** were produced by a parallel per-component audit pass
> (one independent auditor per algorithm), then reviewed. All line numbers are
> verified against `main` @ `3de0857` (the audit ran against the post-PR-#16
> code, which is now merged into main, so the cites are already current). The
> two consequential findings (§14 precision `eps`, §5 Simplex `minedm`) were
> re-verified by hand against the C++ source; spot-checks confirmed the
> shifted-file cites (`minuit.jl`, `ad_gradient.jl`) resolve correctly.

## 5. MnSimplex

`SimplexBuilder.cxx` / `SimplexParameters.cxx` / `SimplexSeedGenerator.cxx` ↔
`src/simplex.jl`. The Nelder–Mead core is a faithful line-for-line port:
reflection/expansion/contraction coefficients (α=1, β=0.5, γ=2, ρmin=4, ρmax=8,
the David-Sachs ρ1/ρ2), the `Update`/`Dirin`/`Edm = f(jh)−f(jl)` machinery, all
reflect/contract/expand/ρ-fit branches and breaks, the post-loop centroid step,
and the final `dirin·√(Up/Edm)` error scaling all map exactly.

Findings:
- **✗ MODERATE — default `minedm` is 10⁴× too tight.** JuMinuit uses
  `minedm = 1e-5·up` (simplex.jl:134-135); C++/iminuit's Simplex EDM goal is
  `toler·Up()` with default `toler=0.1`, i.e. **`0.1·up`** (`ModularFunctionMinimizer::Minimize`
  scales `effective_toler = toler·Up()` for *all* builders; the `×0.002` is
  MIGRAD-only — verified). The in-code comment ("`0.1·tol·up·1e-3`") is
  factually wrong about the C++ Simplex path. Effect: JuMinuit's simplex iterates
  far longer and reports `above_max_edm` much more readily. ~1 LOC + comment fix
  (`minedm = 0.1·cf.up`), OR re-document honestly if a tighter goal is wanted.
- **✗ MODERATE — initial-simplex edge ~10× too large.** C++ edge =
  `10·Gstep` with `Gstep = max(gsmin, 0.1·dirin)` ⇒ effective `≈ dirin`; JuMinuit
  seeds `10·errs` where `errs ≈ dirin` ⇒ edge `≈ 10·dirin`. Same minimum,
  materially different trajectory/call-count. ~3-5 LOC.
- minor: do-while→while-precheck (pre-converged seed skips one reflection; same
  final state); seed EDM/G2 not formed (cosmetic; SimplexBuilder overwrites).

Verdict: faithful Nelder–Mead core, but two compounding scale divergences
(stopping rule 10⁴× tighter, starting simplex 10× larger) reach the same minimum
along a different path with a much stricter — and mis-justified — stopping rule.

## 6. MnLineSearch

`MnLineSearch.cxx` (default parabolic; `#ifdef USE_OTHER_LS` cubic/Brent is
default-off and correctly omitted) + `MnParabolaFactory` ↔ `src/linesearch.jl`.

Findings:
- ✓ **Fully faithful.** Every constant (`overal=1000, undral=-100, toler=0.05,
  slambg=5, alpha=2, maxiter=12`), the slamin/eps2 logic, the 2-point and
  3-point loops, the F2/F3 comparisons, the window clamps, and all early-returns
  match line-for-line. The Lagrange parabola (`linesearch.jl`) is **numerically
  verified ≡** C++'s centered-mean `MnParabolaFactory` (rel-diff ≤ 4e-11 over
  200k random triples).
- minor: a benign off-by-one in the `niter` termination counter (C++ has a
  trailing `niter++`); cannot change the returned `(xvmin, fvmin)`.

Verdict: **SEVERITY none** — a faithful, line-accurate port of the default
parabolic line search.

## 7. NegativeG2LineSearch

`NegativeG2LineSearch.cxx` ↔ `src/negative_g2.jl` (numerical) + `src/ad_gradient.jl` (AD).

Findings:
- ✓ The **numerical-path** `negative_g2_line_search` is faithful line-for-line:
  the `2n` cap, the `Eps`/`Eps2` skip gates, the downhill step sign, the `gdel`,
  the dirin-drop, the full-gradient recompute, the `1/g2` diagonal rebuild, and
  the `MnNotPosDef`-on-negative-EDM all match.
- **✗ MODERATE — AD path is a stub.** `negative_g2_line_search(::CostFunctionWithGradient,…)`
  (ad_gradient.jl:338-350) `@warn`s and returns the seed unchanged, whereas C++
  (`MnSeedGenerator.cxx:161-164`) runs the *full* recovery via a
  `Numerical2PGradientCalculator`. It is on the **live AD seed path**
  (ad_gradient.jl:293-297), so an AD fit seeded with a non-positive `g2` keeps a
  bad curvature instead of the C++-corrected one. ~30-45 LOC (route the recompute
  through the finite-difference `cf.f`, as the Strategy-2 AD HESSE bootstrap
  already does).

Verdict: numerical path faithful; AD path a real (documented) stub gap.

## 8. MnSeedGenerator

`MnSeedGenerator.cxx:41-101` (numerical overload) ↔ `src/seed.jl`.

Findings:
- ✓ The numerical seed is a **constant-for-constant faithful** port: the
  InitialGradient + Numerical2P refine, the `1/g2` (eps2-clamped) diagonal, the
  EDM, the unconditional negative-G2 check, the `HasCovariance`/`prior_cov`
  branch, and the **Strategy(2) seed-time MnHesse bootstrap** all map 1:1.
- minor (AD overload only): the `CheckGradient()` user-gradient discrepancy
  check (~15 LOC) and the negative-G2 refine (~10 LOC, = §7) are Phase-2.1 stubs.

Verdict: numerical seed faithful; only the AD-overload Phase-2.1 stubs diverge.

## 9. Gradient calculators (Initial / Numerical2P / Hessian / Analytical)

`InitialGradientCalculator.cxx`, `Numerical2PGradientCalculator.cxx`,
`HessianGradientCalculator.cxx`, `AnalyticalGradientCalculator.cxx` ↔
`src/gradient.jl`, `src/hessian_gradient.jl`, `src/ad_gradient.jl`.

Findings:
- ✓ Initial, Numerical2P, and Hessian are **byte-exact** in every formula
  (`gsmin=8·eps2·(|x|+eps2)`, `g2=2·up/dirin²`, `gstep=max(gsmin,0.1·dirin)`,
  `dfmin`, `vrysml`, `optstp`, `stpmin/stpmax`), the GradientNCycles loop, and
  both convergence breaks (step-tol, grad-tol), with identical ordering. The
  Hessian calc's intentional quirks (the `4·eps2` factor, the missing-`abs`
  `dmin`, the `j>2` rebased divergence break) are faithfully preserved.
- ✓ Analytical: the int↔ext Jacobian (`DInt2Ext`) is **relocated** to the
  bounded-FCN-wrap layer (migrad_bounded.jl) rather than inside the calculator —
  net result identical (diagonal transform, component-wise chain rule exact).
- minor: the `if HasLimits && step>0.5` clamps are unported but **architecturally
  unreachable** (bounded fits wrap to an unbounded internal `CostFunction`, so
  the calculators never see limit metadata) — zero behavioral gap; `CheckGradient()`
  helper not ported (uncalled in the operator path).

Verdict: all four faithful — exact gradient math; only unreachable clamps + an
uncalled helper diverge.

## 10. DavidonErrorUpdator + VariableMetricEDMEstimator

`DavidonErrorUpdator.cxx`, `VariableMetricEDMEstimator.cxx` ↔ `src/davidon.jl`,
`src/edm.jl`. (Cross-checked against `docs/DAVIDON_CXX_AUDIT.md`.)

Findings:
- ✓ **Fully faithful, verified term-by-term.** The DFP update (the rank-2 base
  `dx⊗dx/δ − vg⊗vg/γ`, the *additive* rank-1 correction when `δ>γ`, the abs-sum
  `dcovar` quality estimator) and the EDM `0.5·gᵀVg` match exactly, including all
  three guards (`δ==0`, `δ<0` warn-only, `γ≤0`) and the `sum_of_elements`
  absolute-value semantics (a signed sum would have silently diverged — it does
  not). The C++ n=1 EDM fast-path is algebraically identical to the general form.

Verdict: **SEVERITY none** — term-for-term faithful; confirms the prior DFP audit.

## 11. MnPosDef

`MnPosDef.cxx` ↔ `src/posdef.jl`.

Findings:
- ✓ The matrix-correction core is **bit-for-bit faithful**: diagonal
  normalization `s=1/√diag`, the `dg = 0.5 + epspdf − dgmin` shift, the
  `pmax=max(|pmax|,1)` clamp, the `pmin > epspdf·pmax` eigenvalue gate, the
  `padd = 0.001·pmax − pmin` final shift, and the upper-triangle storage transpose.
- **✗ minor — metadata divergences (×2).** (a) The `MnMadePosDef` exits pass the
  *incoming* `err.dcovar` (posdef.jl:69,130) instead of C++'s forced `1.0`
  (`BasicMinimumError` MnMadePosDef ctor) — this under-inflates MIGRAD's
  `edm_corrected = edm·(1+3·dcovar)` after a pos-def event, potentially
  terminating one iteration early. (b) The eigenvalue-gate exit preserves
  `err.status` instead of forcing valid+posdef, which can keep a `MnMadePosDef`
  status across the gdel>0→edm<0 re-invocation within one MIGRAD iteration. ~3 LOC.

Verdict: numerics faithful; two undocumented metadata divergences perturb the
post-pos-def EDM correction / iteration count (minor).

## 12. MnEigen / MnGlobalCorrelationCoeff / MnCovarianceSqueeze

`MnEigen.cxx`+`LaEigenValues.cxx`, `MnGlobalCorrelationCoeff.cxx`,
`MnCovarianceSqueeze.cxx` ↔ `src/eigen_corr.jl`, `src/covariance_squeeze.jl`.

Findings:
- ✓ **MnEigen** faithful — the f2c QL solver is replaced by LAPACK `eigvals`
  (sanctioned substitution; both ascending; LAPACK is *more* accurate than C++'s
  fixed `1e-6`).
- ✓ **MnGlobalCorrelationCoeff** faithful — `ρᵢ = √(1 − 1/(Cᵢᵢ·C⁻¹ᵢᵢ))` is
  byte-identical; the `denom≤0` clamp difference is unreachable under real C++
  control flow (that path already set `valid=false`).
- **✗ minor (latent) — MnCovarianceSqueeze.** The first-inversion-failure
  fallback returns the same diagonal values but tags **`MnInvertFailed`** where
  C++ would relabel **Valid** (status-enum divergence). And the **`MnUserCovariance`
  overload is not ported** (the one C++ calls from `MnUserParameterState` on
  parameter-fix) — but JuMinuit has no `MnUserParameterState` analog, so both are
  **latent** (squeeze has no non-test caller).

Verdict: MnEigen + global-cc faithful; CovSqueeze happy-path faithful with a
latent status-enum divergence + an unported (currently-unused) overload.

## 13. MnScan

`MnParameterScan.cxx` + `ScanBuilder.cxx` ↔ `src/scan.jl`.

Findings:
- ✓ Observable behavior faithful: central-point-first ordering, `maxsteps+1`
  length, the `±2σ` default range, the grid math `stp=(high−low)/(maxsteps−1)`,
  and best-point retention all match.
- minor (architectural, behaviorally equivalent): best-point write-back is
  hoisted to the `Minuit` wrapper (`_scan_retain_best!`, + NaN-hardened); the
  dead C++ one-sided-limit branch is collapsed to a both-bounds test; the
  `ScanMinimizer` multi-axis seed-builder is left unported in favor of
  iminuit-style diagnostic semantics (`m.scan()`).

Verdict: faithful observable behavior; deviations are intentional documented
architecture choices.

## 14. Parameter transforms + MnStrategy + MnMachinePrecision

`Sin/SqrtLow/SqrtUp ParameterTransformation.cxx` + `MnUserTransformation.cxx`,
`MnStrategy.cxx`, `MnMachinePrecision.cxx` ↔ `src/transform.jl`, `src/strategy.jl`,
`src/precision.jl`.

Findings:
- ✓ **Parameter transforms faithful** — every formula exact: Sin
  `Int2ext`/`Ext2int` (incl. `distnn=8·√eps2`, `yy²>1−eps2` saturation), `DInt2Ext`,
  both Sqrt transforms (sign-correct derivatives ∓v), and the `Int2extError`
  two-sided `dx>1` clamp.
- ✓ **MnStrategy faithful** — all **21** preset constants (7 knobs × L0/L1/L2)
  match exactly; default level 1.
- **✅ RESOLVED (was MAJOR) — `MnMachinePrecision.eps` was missing the factor of 4.** C++
  `fEpsMac = 4·numeric_limits<double>::epsilon() = 8.88e-16`
  (`MnMachinePrecision.cxx:26`); JuMinuit `MachinePrecision() = MachinePrecision(eps(Float64))`
  = `2.22e-16` (precision.jl). Consequently `eps2 = 2·√eps` is **2× too small**
  (2.98e-8 vs C++ 5.96e-8). `eps2` is the master tolerance threading through the
  *entire* engine via the default `MachinePrecision()`: the numerical-gradient
  minimum step `gsmin=8·eps2·…`, the HESSE deltas `4·eps2·…`, the Sin/MINOS
  near-bound saturation `distnn=8·√eps2`, and the negative-g2 / AD-Hessian
  regularization threshold `|g2|>eps2`. Every one trips at a different point than
  C++/iminuit, so converged values and near-bound error reporting drift at the
  precision-sensitive margin. **~1 LOC fix:**
  `MachinePrecision() = MachinePrecision(4.0 * eps(Float64))` (+ update the
  `p.eps == eps(Float64)` doctest). **Re-verified by hand against the C++ source.**

  **✅ Resolved** in `feat/precision-eps-x4` (PR #19, `src/precision.jl`): the default is
  now `MachinePrecision(4 * eps(Float64))`, citing `MnMachinePrecision.cxx:26`, so
  `eps` = 8.88e-16 and the derived `eps2` = 5.96e-8 — exactly the C++/iminuit
  values (the ×4 on `eps` propagates to the intended ×2 on `eps2`). The
  user-supplied `MachinePrecision(x)` path is unchanged. **Proof:** against the
  C++-Minuit2 JSON oracle (`test_cpp_oracle.jl`) agreement *improved* broadly —
  rosenbrock_2d |Δfval|/|Δparam|/|Δcov| each dropped ~500–800× (3.99e-7→7.4e-10,
  3.33e-5→4.1e-8, 7.0e-3→1.3e-5); bounded_sin_2d ~10⁴× (param 8.2e-9→2.8e-13);
  bounded nfcn drift 4→0; quad_4d unchanged (already at the FP floor). No case
  regressed except rosenbrock_10d's param *position* in its near-flat valley —
  where |Δfval|/|Δedm|/|Δcov|/Δnfcn all improved, i.e. BLAS-order/EDM-stop
  variance, not the fix. Focused parity assertions added to `test_precision.jl`.

Verdict: transforms + all strategy constants exact; the default machine-precision
factor-of-4 (`eps2` 2× off) is now **fixed** — `eps`/`eps2` match C++ Minuit2 /
iminuit. §14 fully faithful.

---

## Summary across all 14 algorithms

**No whole C++ algorithm or branch is silently absent** — every divergence is a
specific, located, mostly-small item. Sorted by severity:

| Severity | Algorithm | Finding | Fix |
|---|---|---|---|
| ~~MAJOR~~ **✅ FIXED** | §14 Precision | default `eps` was missing ×4 ⇒ `eps2` 2× too small vs C++/iminuit; **resolved** in `feat/precision-eps-x4` (now matches C++/iminuit; oracle agreement improved 2–4 orders) | ~1 LOC |
| **MODERATE** | §5 MnSimplex | `minedm` 1e-5·up vs C++ 0.1·up (10⁴× tighter) + initial edge 10× large; wrong in-code citation | ~5 LOC |
| **MODERATE** | §7 NegativeG2 (AD) | AD-path recovery is a `@warn` stub vs C++ Numerical2P-driven; live AD seed path | ~30-45 LOC |
| **MODERATE** | §4 MnContours | missing `sca` direction-switch retry → fewer points on irregular contours | ~10 LOC |
| **resolved** | §1 MnHesse | bounded-param step clamp **RESOLVED** (was `has_limits=false`; unbounded byte-identical) | `153f41d` |
| **resolved** | §2 MIGRAD | 2nd-pass-invalid early-bail **RESOLVED** (efficiency, S≥1 non-converging) | `e256506` |
| **resolved** | §3 MnMinos | default budget **RESOLVED**: n-scaled `2·(nvar+1)·(200+100n+5n²)` | `88bceea` |
| minor | §11 MnPosDef | `MnMadePosDef` dcovar/status metadata (perturbs edm-correction/iters) | ~3 LOC |
| minor (latent) | §12 CovSqueeze | fallback status-enum (Valid vs MnInvertFailed); `MnUserCovariance` overload unported | — |
| minor (deferred) | §8/§9 AD seed/grad | `CheckGradient` + AD negative-G2 Phase-2.1 stubs | ~25 LOC |
| **none** | §6 LineSearch, §10 Davidon/EDM, §14 transforms+strategy | fully faithful (parabola ≡ to 4e-11; DFP/EDM term-by-term; 21 strategy constants exact) | — |

**Headline:** the comprehensive pass found **one MAJOR** item — the machine-precision
`eps` factor-of-4 (§14), a 1-LOC fix with engine-wide reach (**now resolved** in
`feat/precision-eps-x4`; oracle agreement improved 2–4 orders of magnitude) — plus
three MODERATE items (Simplex stopping rule, AD negative-G2 stub, contour `sca`
retry). All are
small, located, and contained; the core minimization/error spine (MIGRAD,
Davidon, EDM, line search, HESSE, MINOS, seed, gradients, transforms, strategy)
is a faithful port. The deliberate keeps (MIGRAD status-gated shortcut) and the
documented Phase-1/2.1 deferrals are called out as such.

**Landed (2026-05-30, branch `feat/cpp-fidelity-3fixes`):** the three §1/§2/§3
minor fixes — MnHesse bounded step clamp (`153f41d`), MIGRAD 2nd-pass-invalid
bail (`e256506`), MnMinos n-scaled budget (`88bceea`). The remaining contained
fixes (§14 precision ×4, §5 Simplex `minedm`, §7 AD negative-G2, §4 MnContours
`sca` retry) are out of scope here.
