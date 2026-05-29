# # JuMinuit.jl — interactive examples
#
# This notebook walks through the canonical usage patterns of
# [JuMinuit.jl](https://github.com/fkguo/JuMinuit.jl), the native-Julia
# port of C++ Minuit2. The API is intentionally close to
# [IMinuit.jl](https://github.com/fkguo/IMinuit.jl) /
# [iminuit](https://github.com/scikit-hep/iminuit) so existing fit code
# transfers with minimal change.

using JuMinuit

# ## 1. Quick start — unbounded fit
#
# A simple quadratic. The minimum is at `(1.0, 2.0)`, fval = 0.
# `migrad(m)` returns the same `m` after running MIGRAD in place.

m = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
            [0.0, 0.0];
            name   = ["a", "b"],
            error  = [0.1, 0.1])
migrad(m)

# Inspect results — iminuit-style property access:
m.values, m.errors, m.fval, m.is_valid

# Pretty-printed table (Unicode in terminal, HTML in this notebook):
m

# ## 2. Bounded parameters
#
# Same fit, but `a ∈ [-5, 5]`, `b ∈ [0, ∞)`. The sin / sqrt transforms
# are applied internally; you write external coords.

m_bnd = Minuit(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2,
               [0.0, 0.5];
               name   = ["a", "b"],
               error  = [0.1, 0.1],
               limits = [(-5.0, 5.0), (0.0, nothing)])
migrad(m_bnd)
m_bnd

# ## 3. MINOS — asymmetric errors
#
# `minos(m)` computes upper/lower asymmetric error bars by tracking the
# 1σ confidence contour through re-minimization. The result lives in
# `m.merrors[name]`.

minos(m)
m.merrors["a"], m.merrors["b"]

# Access the upper/lower offsets directly:
e = m.merrors["a"]
(e.upper, e.lower, e.upper_valid, e.lower_valid)

# ## 4. Saturated MINOS — at-limit semantics
#
# When MINOS hits a parameter bound, `e.upper_par_limit` is raised and
# the published `e.upper` equals the physical distance to the bound
# (matches iminuit's `m.merrors[name].upper`).

m_at = Minuit(x -> (x[1] - 12.0)^2 + (x[2] - 2.0)^2, [5.0, 0.0];
              name = ["x", "y"],
              limits = [(nothing, 10.0), nothing])
migrad(m_at); minos(m_at, 1)
m_at.merrors["x"]    # upper saturated at bound = 10.0

# ## 5. Contours — 2D confidence regions
#
# `contour(m, par1, par2)` returns the full set of (par1, par2) points
# tracing the 1σ contour. Plot directly via Plots.jl (RecipesBase
# integration is built in).

using Plots
ce = contour(m, 1, 2; npoints = 50)
plot(ce; xlabel = "a", ylabel = "b", title = "1σ contour")

# ## 6. Strategy
#
# `Strategy(1)` is the default for numerical FCNs (matching iminuit's
# `Minuit`-class default) and runs an inner-HESSE refinement for a tighter
# covariance. `Strategy(0)` is faster/looser — and the default when a
# `grad=` is supplied, since the AD seed implements level 0 only.
# `Strategy(2)` adds a seed-time MnHesse bootstrap.

m_s2 = Minuit(x -> sum(abs2, x .- [1.0, 2.0, 3.0]), [0.0, 0.0, 0.0])
migrad(m_s2; strategy = Strategy(2))
m_s2.values, m_s2.errors

# ## 7. Switching from IMinuit.jl
#
# JuMinuit aims to be a drop-in replacement. Most IMinuit.jl code runs
# unchanged — `migrad(m)` mutates `m`, `m.values`, `m.errors`,
# `m.fval`, `m.is_valid` all work, `args(m)` returns the value vector,
# `matrix(m)` returns the covariance.

args(m), matrix(m)
