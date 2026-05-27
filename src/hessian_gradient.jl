# SPDX-License-Identifier: LGPL-2.1-or-later

# ─────────────────────────────────────────────────────────────────────────────
# hessian_gradient.jl — gradient refinement between HESSE's diagonal and
# off-diagonal passes (Strategy ≥ 1).
#
# Mirrors reference/Minuit2_cpp/src/HessianGradientCalculator.cxx — the
# `DeltaGradient` member function (lines 70-154). Called from
# `src/hesse.jl` when `strategy.level > 0` (matches the C++ gate at
# `MnHesse.cxx:228`).
#
# Algorithm (per parameter i, up to `hessian_grad_ncycles` cycles):
#   - Start from `d = min(0.2·|gstep[i]|, sqrt(dfmin / (|g2[i]| + epspri)))`,
#     clamped against `dmin = 4·eps2·(x_i + eps2)`. The C++ HGC.cxx:100
#     formula is `xtf + eps2` WITHOUT the abs — when `xtf < 0`, `dmin`
#     becomes negative and the `if (d < dmin) d = dmin` clamp never
#     fires. The Julia port matches this quirk exactly (see body
#     comment); do not "fix" it without checking C++ parity tests.
#   - At each cycle j:
#       fs1 = f(x + d·e_i);  fs2 = f(x - d·e_i)
#       grdnew = (fs1 - fs2) / (2d)
#       dgmin  = eps · (|fs1| + |fs2|) / d
#       if grdnew == 0: break
#       change = |(grdold - grdnew) / grdnew|
#       if change > chgold and j > 2: break          (diverging; keep last)
#       chgold = change
#       grd[i] = grdnew;  gstep[i] = d                (commit refinement)
#       if change < 0.05: break                       (relative convergence)
#       if |grdold - grdnew| < dgmin: break           (absolute convergence)
#       if d < dmin: break                            (step floor hit)
#       d *= 0.2
#   - dgrd[i] = max(dgmin, |grdold - grdnew|)        (gradient uncertainty)
#
# `g2` is read-only here — HGC refines only `grd` and `gstep`. The
# Hessian (`g2`) is taken from the diagonal pass that already ran.
# ─────────────────────────────────────────────────────────────────────────────

"""
    hessian_gradient!(grd, gstep, dgrd, x_work, par, cf, g2,
                      strategy, prec=MachinePrecision()) -> (grd, gstep, dgrd)

In-place port of C++ `HessianGradientCalculator::DeltaGradient`
(`reference/Minuit2_cpp/src/HessianGradientCalculator.cxx:70-154`). Refines
`grd` and `gstep` via per-coordinate central-difference iteration on the
FCN; reports the per-parameter gradient uncertainty `dgrd`.

Called from [`hesse`](@ref) between the diagonal and off-diagonal Hessian
passes when `strategy.level > 0` — the C++ gate at `MnHesse.cxx:228`.

# Arguments
- `grd::AbstractVector{Float64}` — gradient (refined in place).
- `gstep::AbstractVector{Float64}` — step sizes (refined in place).
- `dgrd::AbstractVector{Float64}` — gradient uncertainty (written in
  place).
- `x_work::AbstractVector{Float64}` — working parameter buffer of length
  `n`; perturbed at coord `i` and restored at every cycle (matches C++
  `x(i) = xtf ± d; ...; x(i) = xtf;`).
- `par::MinimumParameters` — current point + fval. Read-only; `par.x` is
  copied into `x_work` at entry.
- `cf::AbstractCostFunction` — user FCN. Called `2 · ncycle · n` in the
  worst case (the algorithm typically breaks out earlier).
- `g2::AbstractVector{Float64}` — diagonal Hessian (read-only); used in
  `optstp = sqrt(dfmin / (|g2| + epspri))`.
- `strategy::Strategy` — supplies `hessian_grad_ncycles`.
- `prec::MachinePrecision` — supplies `eps` and `eps2`.

Returns `(grd, gstep, dgrd)` — the same buffers, now refined.

# Notes

- Zero-allocation in the inner loop (no per-iteration vector creation).
- For `cf isa CostFunctionWithGradient`, this still uses central-difference
  on the FCN (not the analytical gradient) — that mirrors the C++ behavior
  where HGC always recomputes the gradient numerically as part of the
  Hessian refinement.
- `g2` is **not** modified.
"""
function hessian_gradient!(
    grd::AbstractVector{Float64},
    gstep::AbstractVector{Float64},
    dgrd::AbstractVector{Float64},
    x_work::AbstractVector{Float64},
    par::MinimumParameters,
    cf::AbstractCostFunction,
    g2::AbstractVector{Float64},
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    length(grd)    == n || throw(DimensionMismatch("grd length $(length(grd)) != par length $n"))
    length(gstep)  == n || throw(DimensionMismatch("gstep length $(length(gstep)) != par length $n"))
    length(dgrd)   == n || throw(DimensionMismatch("dgrd length $(length(dgrd)) != par length $n"))
    length(x_work) == n || throw(DimensionMismatch("x_work length $(length(x_work)) != par length $n"))
    length(g2)     == n || throw(DimensionMismatch("g2 length $(length(g2)) != par length $n"))

    copyto!(x_work, par.x)
    fcnmin = par.fval
    eps2 = prec.eps2
    eps_  = prec.eps
    up_   = errordef(cf)
    # `4·eps2·(|fcnmin| + up)` — C++ HessianGradientCalculator.cxx:88. The
    # `8·eps2` constant in Numerical2PGradientCalculator is intentional
    # divergence — HGC uses a tighter floor since by the time it runs the
    # diagonal pass has already given us a good `g2`.
    dfmin = 4.0 * eps2 * (abs(fcnmin) + up_)
    ncycle = strategy.hessian_grad_ncycles

    @inbounds for i in 1:n
        xtf = x_work[i]
        # C++ HGC.cxx:100 uses `xtf + eps2` (no abs), but that gives a
        # negative floor when xtf < -eps2 — which then fails the
        # `d < dmin` comparison's intent. The C++ code happens to work
        # because immediately after, line 103 sets `d = 0.2·|gstep[i]|`
        # (always positive) and the subsequent `if (d < dmin) d = dmin;`
        # only fires when dmin > 0. Match the C++ formula exactly:
        dmin = 4.0 * eps2 * (xtf + eps2)
        epspri = eps2 + abs(grd[i] * eps2)
        optstp = sqrt(dfmin / (abs(g2[i]) + epspri))
        d = 0.2 * abs(gstep[i])
        if d > optstp
            d = optstp
        end
        if d < dmin
            d = dmin
        end

        chgold = 10000.0
        dgmin  = 0.0
        grdold = 0.0
        grdnew = 0.0

        for j in 1:ncycle
            x_work[i] = xtf + d
            fs1 = cf(x_work)
            x_work[i] = xtf - d
            fs2 = cf(x_work)
            x_work[i] = xtf

            grdold = grd[i]
            grdnew = (fs1 - fs2) / (2.0 * d)
            dgmin  = eps_ * (abs(fs1) + abs(fs2)) / d

            grdnew == 0 && break

            change = abs((grdold - grdnew) / grdnew)
            # C++ check is `change > chgold && j > 1` with C++ 0-indexed
            # j (so j > 1 means j = 2, 3, ... — the 3rd cycle onwards).
            # Julia 1-indexed j: j > 2 fires at j = 3, 4, ... (same 3rd
            # cycle onwards). Match.
            if change > chgold && j > 2
                break
            end
            chgold = change

            # Commit refinement BEFORE the convergence-break checks
            # (matches C++ HGC.cxx:131-133 — the break-on-convergence
            # cases keep the new value, only the divergence-break above
            # discards it).
            grd[i]   = grdnew
            gstep[i] = d

            change < 0.05 && break
            abs(grdold - grdnew) < dgmin && break
            d < dmin && break
            d *= 0.2
        end

        dgrd[i] = max(dgmin, abs(grdold - grdnew))
    end

    return (grd, gstep, dgrd)
end

"""
    hessian_gradient(par, grad_in, cf, strategy, prec=MachinePrecision()) ->
        (FunctionGradient, dgrd::Vector{Float64})

Allocating convenience wrapper around [`hessian_gradient!`](@ref).
Returns a fresh `FunctionGradient` (g2 unchanged from `grad_in`,
grd/gstep refined) plus the per-parameter uncertainty vector `dgrd`.

Mirrors the `DeltaGradient` overload at
`reference/Minuit2_cpp/src/HessianGradientCalculator.cxx:70-154` —
returns the same `(FunctionGradient, MnAlgebraicVector)` pair.
"""
function hessian_gradient(
    par::MinimumParameters,
    grad_in::FunctionGradient,
    cf::AbstractCostFunction,
    strategy::Strategy,
    prec::MachinePrecision = MachinePrecision(),
)
    n = length(par)
    length(grad_in) == n ||
        throw(DimensionMismatch("grad_in length $(length(grad_in)) != par length $n"))
    grd   = copy(grad_in.grad)
    gstep = copy(grad_in.gstep)
    dgrd  = zeros(Float64, n)
    x_work = similar(par.x)
    hessian_gradient!(grd, gstep, dgrd, x_work, par, cf, grad_in.g2, strategy, prec)
    return (FunctionGradient(grd, copy(grad_in.g2), gstep), dgrd)
end
