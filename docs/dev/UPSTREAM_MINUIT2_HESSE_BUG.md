# Upstream report draft — `MnHesse` `1/g2` clamp inverts physical meaning

> **⚠️ SUPERSEDED — DO NOT FILE AS WRITTEN. See [`DAVIDON_CXX_AUDIT.md`](DAVIDON_CXX_AUDIT.md).**
>
> This draft's central thesis — that C++ `MnHesse`'s **second** diagonal clamp
> (`vhmat(j,j) = tmp < eps2 ? 1 : tmp`) is an unambiguous bug to be removed —
> was **overturned** by JuMinuit's own later x_jm warm-start audit:
>
> - The clamp is **load-bearing**, not a localised mistake. It produces the
>   `V ≈ I` that lets a *warm* start take a large Newton step across basins
>   (iminuit reaches χ²=322.59 this way). This draft only examined the
>   *cold-start / FCN-flat* regime, where the same clamp instead causes a `V=I`
>   line-search blowup. The two regimes want **opposite** `V` scales, so the
>   clamp is a documented C++ **tradeoff**, not a clear bug.
> - **PR #6** (the prescription in § "Suggested upstream patch" — remove the
>   clamp, `std::fabs` form) was therefore **reverted** in JuMinuit. The current
>   `_hesse_diagonal_failure` (hesse.jl ~492) **restores the C++ second clamp**
>   and, unlike the patch below, uses a **raw** comparison (no `fabs`): a
>   negative `g2` falls back to `1.0` rather than keeping a sign-flipped negative
>   variance — see [`CPP_FIDELITY_AUDIT.md`](CPP_FIDELITY_AUDIT.md). The
>   cold-start blowup regime is handled by the retry + Simplex layer instead.
> - The specific fvals below (IAM `613.49 → 401.45`, X3872 `1.30 → 0.017`) are
>   **pre-revert** and no longer describe the shipped defaults; the current IAM
>   default reaches `404.15` (see [`IAM_CONVERGENCE_GAP.md`](IAM_CONVERGENCE_GAP.md)).
>
> Retained as an **investigation record**. The cold-start `V=I` blowup it
> reproduces is a real, reproducible C++ behaviour — it is simply *not* fixable
> by deleting the clamp (that breaks the warm-start basin walk). Whether a
> *revised* upstream note is still worth filing is an open call for the
> maintainer.

**Component**: ROOT / Minuit2 — `Math/Minuit2/src/MnHesse.cxx`
**Versions confirmed**: GooFit/Minuit2 v6.24.0 (commit `57dc936`), Python
`iminuit` ≥ 2 (which links the same C++ source via `pybind11`)
**Affected**: Strategy(1) and Strategy(2) MIGRAD on FCNs where any
parameter has an FCN-flat direction that drives the seed-time
`MnHesse::operator()` into its failure-mode return at line 178 or 217.

## Summary

The diagonal fallback inside `MnHesse::operator()` (run when the
finite-difference sag stays zero or when `maxcalls` is exhausted)
contains a `1/g2 < eps2` check that **inverts the physical meaning of
the diagonal element**:

```cpp
// src/MnHesse.cxx:178-179  (and again at :217-218)
for (unsigned int j = 0; j < n; j++) {
   double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
   vhmat(j, j) = tmp < prec.Eps2() ? 1. : tmp;   // ← bug
}
```

With double-precision `prec.Eps2() ≈ 1.49 × 10⁻⁸`, the second comparison
fires whenever `1/g2 < eps2`, i.e. `g2 > 1/eps2 ≈ 6.7 × 10⁷`, and
replaces `1/g2` with `1.0`.

* `1/g2 = 10⁻¹⁰` physically means: the parameter is **very well
  determined** (the FCN is steep along that direction; one σ is small).
* Setting `vhmat(j, j) = 1.0` tells the next MIGRAD step that the
  parameter is **poorly** determined (one σ is `O(1)`).
* The two meanings are opposites.

When any parameter is FCN-flat at the seed-time `_hesse_diagonal_failure`
exit, the diagonal `V` therefore comes out wrong. The resulting Newton
step `−V·g ≈ −g` with `|g| ~ 10⁶` is so large that line search backs
off to `slam ~ 10⁻³` while the FCN keeps rising, and the run terminates
with "No improvement in line search". MIGRAD reports `is_valid = false`
and the user is left at a much shallower local minimum than the true
one reachable from the same seed.

## Counter-example inside Minuit2 itself

`MnSeedGenerator` uses the *correct* form at `src/MnSeedGenerator.cxx:70`:

```cpp
mat(i, i) = (std::fabs(dgrad.G2()(i)) > prec.Eps2() ? 1. / dgrad.G2()(i) : 1.);
```

— guard against `1/0` only, accept any finite `1/g2`. The same convention
should hold inside the `MnHesse` failure block; the second comparison at
lines 179 and 218 is a localised inconsistency, not a deliberate design
choice.

## Reproduction

### Pure C++ (proposed for upstream report)

```cpp
// repro_mnhesse_diag.cxx — compile against ROOT's Minuit2:
//   g++ -O2 -std=c++17 repro_mnhesse_diag.cxx \
//       $(root-config --cflags --libs) -lMinuit2 -o repro_mnhesse_diag
#include <Minuit2/FCNBase.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/FunctionMinimum.h>
#include <cmath>
#include <iostream>

using namespace ROOT::Minuit2;

// 2-parameter FCN: p[0] is well-determined (steep quadratic);
// p[1] is FCN-flat (FCN does not depend on it). Constructing a seed
// where MnHesse falls into the failure-mode block reproduces the
// V = I error.
struct FlatChi2 : public FCNBase {
   double operator()(const std::vector<double>& p) const override {
      return 1e14 * (p[0] - 1.0) * (p[0] - 1.0);   // g2 ~ 2e14
   }
   double Up() const override { return 1.0; }
};

int main() {
   FlatChi2 fcn;
   MnUserParameters pars;
   pars.Add("p0", 0.0, 1e-6);
   pars.Add("p1", 0.0, 1e-6);
   MnStrategy strat(2);
   MnMigrad migrad(fcn, pars, strat);
   FunctionMinimum min = migrad();
   std::cout << "is_valid = " << min.IsValid() << "\n"
             << "fval     = " << min.Fval() << "\n"
             << "V.diag   = (" << min.Error().Matrix()(0,0) << ", "
                              << min.Error().Matrix()(1,1) << ")\n";
   // Expected with current C++: V.diag = (1.0, 1.0), is_valid = false.
   // Expected after fix:        V.diag = (5e-15, 1.0), is_valid = true.
   return 0;
}
```

### Through Python `iminuit`

```python
from iminuit import Minuit
def fcn(p0, p1):
    return 1e14 * (p0 - 1.0)**2
m = Minuit(fcn, p0=0.0, p1=0.0)
m.errors = [1e-6, 1e-6]
m.errordef = 1.0
m.strategy = 2
m.migrad()
print(repr(m.covariance))    # diag → (1.0, 1.0); should be (5e-15, 1.0)
```

### Physics fit (where this was first noticed)

The IAM 2π form-factor fit (9 LECs, last parameter FCN-flat by
construction) and the X(3872) `J/ψρ + DD̄*` dip fit (3 parameters, narrow
χ²-min) both reach this branch at Strategy(2). Concrete reproduction:

* IAM: `julia --project=scripts BenchmarkExamples/IAM_2Pformfactor/bench_full.jl` — with the bug, Strategy(2) returns `fval=613.49`; with the fix, returns `fval=401.45` (matching iminuit Strategy(0) at 400.23, same local minimum).
* X(3872): `julia --project=scripts BenchmarkExamples/X3872_dip/bench_full.jl` — with the bug, Strategy(2) returns `fval=1.30`; with the fix, returns `fval=0.017`, matching the published global minimum [Baru, Guo, Hanhart, Nefediev, *Phys. Rev. D* **109** (2024) L111501, [arXiv:2404.12003](https://arxiv.org/abs/2404.12003), [INSPIRE 2778938](https://inspirehep.net/literature/2778938)].

## Suggested upstream patch

Replace the buggy block at `src/MnHesse.cxx:178-179` and `:217-218` (both
identical) with the same convention `MnSeedGenerator.cxx:70` already
uses:

```cpp
for (unsigned int j = 0; j < n; j++) {
   double v = std::fabs(g2(j)) > prec.Eps2() ? 1. / g2(j) : 1.;
   vhmat(j, j) = (std::isfinite(v) && v != 0.) ? v : 1.;
}
```

The `isfinite && v != 0.` guard handles `g2 = ±Inf` (giving `1/g2 = ±0`,
which is structurally degenerate and should fall back to 1.) and `NaN`.
The `std::fabs` (versus the original strict `g2 < eps2`) keeps a
negative-`g2` value coming through with its sign — matching
`MnSeedGenerator`'s behaviour and letting `MnPosDef` handle the sign
flip downstream rather than silently inverting `1/(−g2)` to `1`.

## Cross-references

* JuMinuit.jl PR #6 originally applied this fix in the Julia port — **but it
  was later reverted** (the C++ second clamp was restored; see the SUPERSEDED
  banner above and `DAVIDON_CXX_AUDIT.md`):
  <https://github.com/fkguo/JuMinuit.jl/pull/6>
* JuMinuit.jl benchmark and physics-impact documentation:
  [`BenchmarkExamples/RESULTS.md`](../BenchmarkExamples/RESULTS.md)
* JuMinuit.jl gap audit listing this and other ports:
  [`GAP_AUDIT.md`](GAP_AUDIT.md)
