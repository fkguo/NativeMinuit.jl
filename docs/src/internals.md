# Internals

This page documents the implementation modules for contributors. End
users typically don't need any of this.

## Algorithm map

JuMinuit mirrors C++ Minuit2's module structure:

| C++ source                            | Julia source            | Phase |
|:--------------------------------------|:------------------------|:------|
| `VariableMetricBuilder.cxx`           | `migrad.jl`             | 0 + 1 |
| `MnLineSearch.cxx`                    | `linesearch.jl`         | 0     |
| `NegativeG2LineSearch.cxx`            | `negative_g2.jl`        | 0     |
| `MnPosDef.cxx`                        | `posdef.jl`             | 0     |
| `Numerical2PGradientCalculator.cxx`   | `gradient.jl`           | 0     |
| `DavidonErrorUpdator.cxx`             | `davidon.jl`            | 0     |
| `MinimumState.cxx` + headers          | `state.jl`              | 0     |
| `MnSeedGenerator.cxx`                 | `seed.jl`               | 0     |
| `MnHesse.cxx`                         | `hesse.jl`              | 1     |
| `MnMinos.cxx` + `MnFunctionCross.cxx` | `minos.jl` + `function_cross.jl` | 1 + 1.x |
| `MnContours.cxx`                      | `contours.jl`           | 1     |
| `MnUserTransformation.cxx`            | `transform.jl`          | 1     |
| `MnUserParameterState.cxx`            | `parameters.jl`         | 1     |
| `MnCovarianceSqueeze.cxx`             | `covariance_squeeze.jl` | 1     |

## Inner-MIGRAD loop

`_migrad_loop` in `src/migrad.jl` implements the C++
`VariableMetricBuilder::Minimum` do-while. Inner DFP iteration:

```
while EDM > tol && nfcn < maxfcn:
    step = -V·∇
    line-search → α*
    accept new point
    compute gradient
    compute EDM from old V
    DFP rank-2 update of V
```

Outer Strategy ≥ 1 wrapper (Phase 1):

```
do:
    inner DFP
    if Strategy == 2 OR (Strategy == 1 && Dcovar > 0.05):
        call hesse(state)
        if Hesse fails: break
        if new EDM > tol: iterate (re-enter inner DFP)
    if ipass == 0: bump maxfcn → maxfcn × 1.3
while iterate
```

## Bound transforms

Three transforms (mirroring C++):

| Bounds          | Transform | int2ext formula                          |
|:----------------|:----------|:------------------------------------------|
| `[L, U]`        | Sin       | `L + 0.5·(U-L)·(sin(v)+1)`                 |
| `(-∞, U]`       | SqrtUp    | `U + 1 - sqrt(v² + 1)`                     |
| `[L, ∞)`        | SqrtLow   | `L - 1 + sqrt(v² + 1)`                     |
| `(-∞, ∞)`       | identity  | `v`                                        |

**Sign-aware derivatives**: `sqrtup_dint2ext` is *negative* for v > 0
(increasing internal coord decreases external value). This matters for
off-diagonal covariance entries between upper-only and lower-only
parameters (the chain-rule factor flips sign).

## MnFunctionCross 3-point parabolic

The `_cross_core` in `function_cross.jl` ports C++
`MnFunctionCross.cxx:117-507` step-by-step:

1. Quadratic seed: `α₁ = sqrt(up / (f - fmin)) - 1` clamped to `[-0.5, 1]`.
2. Inner MIGRAD at `α₁`. Compute `dfda = (f₁ - f₀) / (α₁ - 0)`.
3. **L300**: while `dfda < 0`, extend `α` outward.
4. **L460**: linear extrapolation `α₂ = α₁ + (aim - f₁) / dfda`.
   Run inner MIGRAD at `α₂`, get 3rd point.
5. Classify (`noless` = count of points below aim, `ibest` = closest to aim):
   - `noless ∈ {1, 2}` → L500 (parabola fit).
   - `noless == 0 && ibest ≠ 3` → invalid.
   - `noless == 3 && ibest ≠ 3` → goto L300 (re-extend).
   - else (third point is best, all 3 on one side) → "new straight line":
     replace iworst with point 3, recompute dfda from 1+2, goto L460.
6. **L500**: parabola fit through 3 points; solve `A·α² + B·α + C = aim`;
   pick root with positive slope; window-clamp; re-probe; replace iout.
7. Convergence: `|α - α[ibest]| < tla AND |f[ibest] - aim| < tlf`.

Tolerances `tla = tlr = 0.01` and `tlf = 0.01·up` are hardcoded per C++
(`MnFunctionCross.cxx:40`); the user's `tlr` controls only the inner-MIGRAD
tolerance via `mgr_tlr = 0.5 · tlr`.
