# SPDX-License-Identifier: LGPL-2.1-or-later

"""
    Strategy(level::Integer)

Mirror of `MnStrategy` from
`reference/Minuit2_cpp/src/MnStrategy.cxx:33–70`.

Controls the cost/precision trade-off in gradient and Hessian
calculations via seven tunable knobs. Three preset levels:

| Field                      | C++ name                  | L0   | L1   | L2   |
|----------------------------|---------------------------|------|------|------|
| `grad_ncycles`             | `GradientNCycles`         | 2    | 3    | 5    |
| `grad_step_tolerance`      | `GradientStepTolerance`   | 0.5  | 0.3  | 0.1  |
| `grad_tolerance`           | `GradientTolerance`       | 0.1  | 0.05 | 0.02 |
| `hessian_ncycles`          | `HessianNCycles`          | 3    | 5    | 7    |
| `hessian_step_tolerance`   | `HessianStepTolerance`    | 0.5  | 0.3  | 0.1  |
| `hessian_g2_tolerance`     | `HessianG2Tolerance`      | 0.1  | 0.05 | 0.02 |
| `hessian_grad_ncycles`     | `HessianGradientNCycles`  | 1    | 2    | 6    |

- `Strategy(0)` — Low: fastest, lowest accuracy. **Phase 0 default and only
  supported level.**
- `Strategy(1)` — Medium: matches C++ Minuit2 default (Phase 1+).
- `Strategy(2)` — High: slowest, highest accuracy (Phase 1+).

The `level` field exposes the integer level (0/1/2) — `VariableMetricBuilder.cxx`
branches on `Strategy() >= 1` to invoke the inner `MnHesse` path. Phase 0
locks the entire library to level 0; Strategy ≥ 1 will be enabled when
`hesse.jl` ships in Phase 1 (see `docs/DESIGN.md` DR-008).

Like [`MachinePrecision`](@ref), this is an immutable, isbits, type-stable
struct — zero allocation when passed through the call chain.

# Examples

```julia
julia> s = Strategy(0);

julia> s.level, s.grad_ncycles, s.grad_step_tolerance
(0, 2, 0.5)

julia> Strategy(3)
ERROR: ArgumentError: ...
```
"""
struct Strategy
    level::Int                          # 0, 1, 2 — mirrors C++ fStrategy
    grad_ncycles::Int                   # GradientNCycles
    grad_step_tolerance::Float64        # GradientStepTolerance
    grad_tolerance::Float64             # GradientTolerance
    hessian_ncycles::Int                # HessianNCycles
    hessian_step_tolerance::Float64     # HessianStepTolerance
    hessian_g2_tolerance::Float64       # HessianG2Tolerance
    hessian_grad_ncycles::Int           # HessianGradientNCycles
end

function Strategy(level::Integer)
    if level == 0
        # SetLowStrategy — MnStrategy.cxx:33–44
        Strategy(0, 2, 0.5, 0.1, 3, 0.5, 0.1, 1)
    elseif level == 1
        # SetMediumStrategy — MnStrategy.cxx:46–57 (C++ Minuit2 default)
        Strategy(1, 3, 0.3, 0.05, 5, 0.3, 0.05, 2)
    elseif level == 2
        # SetHighStrategy — MnStrategy.cxx:59–70
        Strategy(2, 5, 0.1, 0.02, 7, 0.1, 0.02, 6)
    else
        throw(ArgumentError(
            "Strategy level must be 0, 1, or 2 (got $level). " *
            "Phase 0 also requires level == 0 — see docs/DESIGN.md DR-008."
        ))
    end
end
