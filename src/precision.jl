# SPDX-License-Identifier: LGPL-2.1-or-later

"""
    MachinePrecision(eps_value = 4 * eps(Float64))

Mirror of `MnMachinePrecision` from
`reference/Minuit2_cpp/inc/Minuit2/MnMachinePrecision.h`.

- `eps` — relative floating-point precision (C++ `fEpsMac`). Defaults to
  `4·eps(Float64) ≈ 8.88e-16`, matching C++ `MnMachinePrecision.cxx:26`
  (`fEpsMac = 4. * numeric_limits<double>::epsilon()`). The ×4 is the C++
  factor that absorbs the 2× between `numeric_limits::epsilon` and the
  DLAMCH-style epsilon Minuit2 was tuned against (see the C++ source note).
- `eps2` — `2·√eps` (C++ `fEpsMa2`), the tolerance multiplier used by the
  numerical-gradient step-size algorithm. With the default `eps` this is
  `2·√(4·eps(Float64)) = 4·√eps(Float64) ≈ 5.96e-8`.

A user can override the default to declare reduced precision when the
FCN value is itself computed at less than IEEE-754 nominal accuracy
(e.g. stochastic simulation FCNs).

This is an immutable, isbits, type-stable struct: stack-allocated when
used as a local, no GC pressure in hot loops.

# Examples

```julia
julia> p = MachinePrecision();

julia> p.eps == 4 * eps(Float64)
true

julia> p.eps2 ≈ 2 * sqrt(4 * eps(Float64))
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

# Default machine precision mirrors C++ MnMachinePrecision.cxx:26:
#   fEpsMac = 4. * std::numeric_limits<double>::epsilon();
# The ×4 is intentional and load-bearing: it makes the derived
# eps2 = 2·√eps match C++ Minuit2 / iminuit (≈5.96e-8, not 2.98e-8), so
# every eps2-gated step size and threshold — numerical-gradient steps
# (gradient.jl), HESSE probe deltas (hesse.jl), parameter-transform
# saturation (transform.jl), MINOS/function-cross clamps
# (function_cross.jl, minos.jl) and the negative-g2 gate (negative_g2.jl)
# — trips at the same point as the reference engine. See audit §14.
MachinePrecision() = MachinePrecision(4 * eps(Float64))
