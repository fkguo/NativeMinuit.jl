# X(6200) — two-channel di-J/ψ fit (JuMinuit benchmark + Bayesian example)

A native-JuMinuit reproduction of the published coupled-channel analysis of the
**X(6200)**, the near-threshold state in the J/ψJ/ψ system, followed by Bayesian
propagation of its pole / scattering length / effective range / compositeness.

It does five things, each verified against the published numbers:

1. **Fits the real LHCb double-J/ψ spectrum** with JuMinuit (7 free parameters) —
   recovers the published best fit at χ²/dof = 28.708/29 = 0.99.
2. **Finds the X(6200) pole** by a search on the four Riemann sheets of the
   T-matrix (det(1 − G·V) = 0): a sheet-II pole at ≈ 6.203 GeV, just above the
   J/ψJ/ψ threshold (6.194 GeV), plus the broad sheet-III pole near 6.82 GeV.
3. **Reproduces the published error bars** by propagating the derived quantities
   over the paper's Δχ² parameter ensemble (the frequentist method it used).
4. **Propagates them with JuMinuit's Bayesian posterior sampling** (`posterior_sample` +
   `derived_interval`): classifies the near-threshold pole sheet-by-sheet — bound
   state (sheet I) vs virtual / resonance (sheet II) — to quantify the X(6200)'s
   two-fold interpretation, and shows why a near-unitary scattering length must be
   reported through `1/a`.
5. **Encodes a physical constraint as a prior** — causality (no spurious pole on
   the physical sheet) as `logprior = -Inf` on acausal couplings — and checks
   whether it tightens the posterior (for the X(6200) it does not bite).

## Files

| File | Purpose |
|---|---|
| `x6200_double_jpsi.jl` | The runnable example (fit → poles → frequentist band → posterior → physical-prior check) |
| `data_lhcb.csv` | Digitized LHCb double-J/ψ invariant-mass spectrum (mass, events, error) |
| `parametersets_2c.csv` | The published Δχ²≈1 parameter ensemble (322 sets) for the error bars |

## Run

```bash
julia --project=. BenchmarkExamples/X6200_double_jpsi/x6200_double_jpsi.jl
```

Needs only `JuMinuit` plus the standard library (`DelimitedFiles`, `Statistics`,
`LinearAlgebra`). Runs in a few seconds.

## Physics context

The LHCb collaboration reported a narrow X(6900) and **a broad structure just
above twice the J/ψ mass** in the di-J/ψ spectrum (arXiv:2006.16957). The
coupled-channel analysis of Dong *et al.* describes the data with a
unitarity-consistent T-matrix in two channels, **J/ψ J/ψ** and **ψ(2S) J/ψ**,
built from five energy-dependent contact couplings `(a1, a2, c, b1, b2)`. It
produces a pole on the second Riemann sheet right above the J/ψJ/ψ threshold —
the **X(6200)**, with quantum numbers Jᴾᶜ = 0⁺⁺ or 2⁺⁺ — a candidate
near-threshold (molecular) state.

The scattering length `a`, effective range `r`, and compositeness `X̄_A`
(molecular weight) follow from the near-threshold expansion

```
T(k)⁻¹ = -8π√s [ 1/a + ½ r k² - i k + O(k⁴) ].
```

## Results (reproduced)

| Quantity | Published (Dong *et al.*) | JuMinuit, this example |
|---|---|---|
| χ²/dof | 28.708 / 29 = 0.99 | **0.99** (same best fit) |
| X(6200) pole (sheet II) | ≈ 6.20 GeV near threshold | **6.203 + 0.012 i GeV** |
| scattering length a₀ | ≤ −0.49 or ≥ 0.48 fm | **≤ −0.49 / ≥ 0.48 fm** (frequentist) |
| effective range r₀ | −2.18 ⁺⁰·⁶⁶₋₀.₈₁ fm | **−2.18 +0.66 −0.81 fm** (frequentist) |
| compositeness X̄_A | 0.39 ⁺⁰·⁵⁸₋₀.₁₂ | **0.39 +0.58 −0.12** (frequentist) |

The Δχ²-ensemble (Part 3) propagation reproduces the published Table exactly. The
Bayesian posterior from the real data fit (Part 4) returns the credible-interval
analogue — *broad* here (e.g. r at 16/50/84 % ≈ −2.77 / −1.75 / −1.19 fm), because
the 36-bin spectrum weakly constrains the couplings; the centrals still match. See
*What this example also teaches* below.

## What this example also teaches

- **A near-unitary scattering length diverges.** `a` straddles ±∞ over the
  posterior (samples of both signs), so an equal-tailed credible interval on `a`
  is meaningless — the well-defined variable is `1/a`, which gives the paper's
  *disjoint* bound `|a| ≳ 0.48 fm`. Reparametrize a derived quantity that can
  diverge (report `1/a`, a bound, or an HPD region).
- **The sign of `1/a` is the bound-vs-virtual switch.** As `1/a` passes through
  zero (the unitary limit) the near-threshold pole crosses from sheet I (a *bound*
  state, `a<0`) to sheet II (a *virtual* state or *resonance*, `a>0`) — the paper's
  convention `1/a + ½rk² − ik = 0` gives `κ ≈ −1/a`. Part 4 classifies every
  posterior sample and reports `P(bound) ≈ 83 %`, `P(resonance) ≈ 15 %`,
  `P(virtual) ≈ 2 %` (binding `E_B ≈ 10–150 MeV` if bound) — a directly Bayesian
  answer to the X(6200)'s nature, with the best fit (a resonance) in the minority.
  This *is* the two-fold interpretation behind the paper's disjoint `a₀`.
- **Credible ≠ confidence.** Part 3 gives frequentist Δχ² (profile) ranges; Part 4
  gives Bayesian credible intervals. The best-fit centrals agree, but the credible
  intervals are *broad* — the 36-bin spectrum weakly constrains the five couplings
  (HESSE condition ~190) and the derived quantities are strongly nonlinear near
  unitarity, so the marginal posterior spreads well beyond the profile bars.
- **A physical constraint is a prior — but check whether it bites.** Part 5 encodes
  causality (no spurious pole on the physical sheet) as `logprior = -Inf` on acausal
  couplings. For the X(6200) it leaves the posterior essentially unchanged: nearly the
  whole posterior is already causal, so the constraint does not tighten the bars — it
  is **not** the reason the published bars are tight. The mechanism is general and
  powerful (a prior that genuinely cuts into the posterior — a hard support, a measured
  nuisance — does tighten it); it simply does not bite here.

## Sources and attribution

Published analysis (reproduced here):

> X.-K. Dong, V. Baru, F.-K. Guo, C. Hanhart, A. Nefediev,
> *Coupled-channel interpretation of the LHCb double-J/ψ spectrum and hints of a
> new state near the J/ψJ/ψ threshold*,
> Phys. Rev. Lett. **126** (2021) 132001, [arXiv:2009.07795](https://arxiv.org/abs/2009.07795),
> [INSPIRE](https://inspirehep.net/literature/1817481).
> Full original analysis (IMinuit.jl): <https://github.com/fkguo/double_jpsi_fit>.

Data:

> `data_lhcb.csv` is the digitized LHCb double-J/ψ invariant-mass spectrum
> (LHCb collaboration, *Observation of structure in the J/ψ-pair mass spectrum*,
> Sci. Bull. **65** (2020) 1983, [arXiv:2006.16957](https://arxiv.org/abs/2006.16957)),
> as used in arXiv:2009.07795. Both `data_lhcb.csv` and `parametersets_2c.csv`
> are vendored from <https://github.com/fkguo/double_jpsi_fit> with the author's
> permission.
