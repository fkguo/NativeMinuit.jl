# SPDX-License-Identifier: LGPL-2.1-or-later

"""
    MachinePrecision(eps_value = eps(Float64))

Mirror of `MnMachinePrecision` from
`reference/Minuit2_cpp/inc/Minuit2/MnMachinePrecision.h`.

- `eps` — relative floating-point precision. Defaults to
  `eps(Float64) ≈ 2.22e-16`.
- `eps2` — `2·√eps`, the tolerance multiplier used by the numerical-
  gradient step-size algorithm.

A user can override the default to declare reduced precision when the
FCN value is itself computed at less than IEEE-754 nominal accuracy
(e.g. stochastic simulation FCNs).

This is an immutable, isbits, type-stable struct: stack-allocated when
used as a local, no GC pressure in hot loops.

# Examples

```julia
julia> p = MachinePrecision();

julia> p.eps == eps(Float64)
true

julia> p.eps2 ≈ 2 * sqrt(eps(Float64))
true

julia> p_noisy = MachinePrecision(1e-12);  # FCN noisy to 1e-12

julia> p_noisy.eps2 ≈ 2 * sqrt(1e-12)
true
```
"""
struct MachinePrecision
    eps::Float64
    eps2::Float64

    function MachinePrecision(eps_value::Real)
        e = Float64(eps_value)
        new(e, 2.0 * sqrt(e))
    end
end

MachinePrecision() = MachinePrecision(eps(Float64))
