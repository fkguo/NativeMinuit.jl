# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 Feng-Kun Guo
#
# Derivative work of C++ Minuit2 (GooFit/Minuit2 @ v6.24.0); see LICENSE
# and docs/UPSTREAM.md.

"""
    JuMinuit

Native-Julia port of the C++ Minuit2 function-minimization library — the
algorithm at the heart of every HEP fit. A drop-in replacement for IMinuit.jl
(the Julia Minuit2 wrapper), with an iminuit-style API and C++-comparable
(often better) performance.

# What's included
- **MIGRAD** (variable-metric / DFP), **HESSE**, **MINOS** (asymmetric
  errors), **MnContours**, **Simplex** and **Scan** — ported from C++
  Minuit2 v6.24.0 with line-by-line fidelity.
- Bounds, fixed parameters and Strategy levels 0/1/2, using the same
  sin/√ transforms as C++; the user FCN always sees external coordinates.
- An iminuit-style `Minuit` front end (`m.values`, `m.errors`, `migrad!`,
  `minos!`, …) plus IMinuit.jl-compatible `Fit` / `ArrayFit`.
- A Julia-native cost-function family (`LeastSquares`, `UnbinnedNLL`,
  `BinnedNLL`, …) composable with `CostSum`.
- Error analysis beyond HESSE/MINOS: Monte-Carlo Δχ² regions, bootstrap,
  jackknife and multi-modal solution detection (see `docs/src/error_analysis.md`).
- AD-backed gradients (ForwardDiff extension), an opt-in threaded numerical
  gradient, and an `Optim.jl` alternative-minimizer bridge (`optim`).

The implementation mirrors `reference/Minuit2_cpp/` (pinned to GooFit/Minuit2
`57dc936`, v6.24.0); each src/ file maps to a C++ translation unit so audits
diff cleanly. Development history, the C++-fidelity audit, and the
deferred-feature list live in `docs/dev/`.

See the [manual](https://fkguo.github.io/JuMinuit.jl) for tutorials and the
full API.
"""
module JuMinuit

using LinearAlgebra
using Logging
using Printf
using Random
using Statistics

include("precision.jl")
include("strategy.jl")
include("trace.jl")
include("state.jl")
include("fcn.jl")
include("linalg.jl")
include("gradient.jl")
include("hessian_gradient.jl")
include("davidon.jl")
include("edm.jl")
include("posdef.jl")
include("linesearch.jl")
include("negative_g2.jl")
include("seed.jl")
include("result.jl")
include("migrad.jl")
include("ad_gradient.jl")
include("transform.jl")
include("parameters.jl")
include("hesse.jl")
include("covariance_squeeze.jl")
include("function_cross.jl")
include("minos.jl")
include("contours.jl")
include("migrad_bounded.jl")
include("simplex.jl")
include("scan.jl")
include("minuit.jl")
include("display.jl")
include("eigen_corr.jl")
include("solution_modes.jl")
include("iminuit_compat.jl")
include("cost_functions.jl")
include("resampling.jl")
include("error_sampling.jl")
include("serialize.jl")
include("plot_recipes.jl")
include("plot_text.jl")
include("precompile_workload.jl")

# Phase 0 public surface (will grow as files are added).
export MachinePrecision
export Strategy
export CovStatus
export MnHesseValid, MnHesseFailed, MnMadePosDef, MnInvertFailed, MnNotPosDef
export MinimumParameters, FunctionGradient, MinimumError, MinimumState
export CostFunction
export ncalls, reset_ncalls!, errordef
export is_valid, has_step_size, is_analytical, is_accurate, is_pos_def
export is_made_pos_def, hesse_failed, invert_failed, is_available
export has_parameters, has_covariance, fval, edm, nfcn
export initial_gradient, initial_gradient!
export numerical_gradient, numerical_gradient!
export estimate_edm, estimate_edm!
export make_posdef, is_posdef_enough
export ParabolaPoint, line_search
export has_negative_g2, negative_g2_line_search
export seed_state, warm_restart_state
export FunctionMinimum, migrad
export is_thread_safe, ThreadSafetyError
export parameters, errors, gradient, covariance
export reached_call_limit, above_max_edm
export MinuitParameter, Parameters
export has_lower_limit, has_upper_limit, has_limits, is_fixed
export n_pars, n_free, ext_index
export int_to_ext_value, ext_to_int_value, dint2ext_value
export int_to_ext_vector, ext_to_int_vector
export initial_int_values, initial_int_errors
export hesse, HesseResult
export squeeze_symmetric, squeeze_error
export MnCross, MinosError, minos, minos_lower, minos_upper
export ContoursError, contour, contour_exact, contour_parameter_sets
export function_cross_multi
# Sampling-based / contour error analysis (error_sampling.jl)
export delta_chisq, chisq_cl
export get_contours_samples, contour_df_samples
export BoundedFunctionMinimum, ext_errors, ext_covariance, free_covariance
export Minuit, migrad!, minos!
# IMinuit.jl drop-in fit-type names (AbstractFit supertype; Fit/ArrayFit
# are aliases of Minuit — see AbstractFit docstring for why not distinct types)
export AbstractFit, Fit, ArrayFit
# Jupyter-first rich output (display.jl)
export to_latex
# Per-parameter mutators (mirror C++ MnUserParameters; gap M3)
export fix!, release!, set_value!, set_error!, set_limits!, remove_limits!
# One-sided limit setters (mirror C++ MnUserParameters::SetUpper/LowerLimit)
export set_upper_limit!, set_lower_limit!
# IMinuit.jl-compatible helpers (NB: `reset` extends `Base.reset`,
# `matrix` is JuMinuit's own — IMinuit.jl's matrix returns the
# correlation/covariance matrix with the same signature).
export args, matrix, set_precision
export CostFunctionWithGradient, analytical_gradient, analytical_gradient!
export CostFunctionAD

# IMinuit.jl drop-in helpers (iminuit_compat.jl)
export Data, chisq, model_fit, @model_fit
export func_argnames
export chi2, poisson_chi2, multinominal_chi2
export @plt_data, @plt_data!, @plt_best, @plt_best!

# Julia-native cost type family (cost_functions.jl). `errordef` (the
# trait) is already exported above; these are the cost types + helpers.
export AbstractCost, LeastSquares, UnbinnedNLL, BinnedNLL
export ExtendedUnbinnedNLL, ExtendedBinnedNLL, CostSum
export parameter_names

# Data-resampling error analysis (resampling.jl)
export bootstrap, jackknife, BootstrapResult, JackknifeResult, correlation

# Algorithms ported from C++ Minuit2
export simplex, scan
export eigenvalues, global_cc

# Multi-modal solution detection (beyond iminuit) — cluster Δχ² samples
export SolutionMode, SolutionModes, find_solution_modes

# IMinuit.jl-compatible algorithm wrappers
export mncontour, profile, mnprofile
export draw_contour, draw_mncontour, draw_profile, draw_mnprofile, draw_mnmatrix
# Alternative-minimizer bridge (Optim.jl extension — `using Optim` to enable)
export optim, minimize_with

# Terminal / SSH / headless-CI ASCII renderer (plot_text.jl, gap M2)
export mn_plot_text

end # module JuMinuit
