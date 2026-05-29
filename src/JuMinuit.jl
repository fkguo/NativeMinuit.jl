# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2026 fkguo and JuMinuit.jl contributors
#
# Derivative work of C++ Minuit2 (GooFit/Minuit2 @ v6.24.0); see LICENSE
# and docs/UPSTREAM.md.

"""
    JuMinuit

Native-Julia port of the C++ Minuit2 function-minimization library, the
algorithm at the heart of every HEP fit. Targets drop-in replacement of
the iminuit/IMinuit.jl stack with C++-comparable performance.

Status: **Phase 0 (proof of concept)** — see [`ROADMAP.md`](../ROADMAP.md).
Phase 0 ships unconstrained MIGRAD with numerical gradient and
`Strategy(0)` only. Bounds, fixed parameters, MINOS, contours, and HESSE
land in Phase 1.

The implementation mirrors `reference/Minuit2_cpp/` (pinned to
`57dc936`, v6.24.0). Each src/ file maps 1-to-1 to a C++ translation
unit so audits diff cleanly. See `docs/PORTING.md` for the mapping.
"""
module JuMinuit

using LinearAlgebra
using Logging
using Printf

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
include("eigen_corr.jl")
include("iminuit_compat.jl")
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
export ContoursError, contour, contour_exact
export function_cross_multi
export BoundedFunctionMinimum, ext_errors, ext_covariance, free_covariance
export Minuit, migrad!, minos!
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

# Algorithms ported from C++ Minuit2
export simplex, scan
export eigenvalues, global_cc

# IMinuit.jl-compatible algorithm wrappers
export mncontour, profile, mnprofile
export draw_contour, draw_mncontour, draw_profile, draw_mnprofile, draw_mnmatrix
export scipy

# Terminal / SSH / headless-CI ASCII renderer (plot_text.jl, gap M2)
export mn_plot_text

end # module JuMinuit
