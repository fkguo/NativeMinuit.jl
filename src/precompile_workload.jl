# SPDX-License-Identifier: LGPL-2.1-or-later

# Independently callable compilation workloads. Each unit constructs its own
# inputs, so failure cannot skip another unit through shared setup state.
# Exceptions propagate during precompilation and in the focused regression test.
# These calls exercise compilation; they do not establish a TTFX improvement.
using PrecompileTools

_wl_quadratic(x) = (x[1] - 1.0)^2 + (x[2] - 2.0)^2
_wl_line(x, p) = p[1] * x + p[2]
_wl_pdf(x, p) = exp(-0.5 * ((x - p[1]) / p[2])^2) / (p[2] * sqrt(2π))
_wl_density(x, p) = p[3] * _wl_pdf(x, p)
_wl_integral(p) = p[3]
_wl_cdf(x, p) = 1.0 / (1.0 + exp(-(x - p[1]) / p[2]))
_wl_scaled_cdf(x, p) = p[3] * _wl_cdf(x, p)
_wl_data() = Data([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], ones(4))
_wl_cost() = CostFunction(_wl_quadratic, 1.0)
_wl_minimum() = migrad(_wl_cost(), [0.0, 0.0], [0.1, 0.1])
function _wl_minuit()
    m = Minuit(_wl_quadratic, [0.0, 0.0]; names = ["a", "b"], errors = [0.1, 0.1])
    migrad!(m)
    return m
end
function _wl_bounded()
    params = Parameters([
        MinuitParameter("a", 0.0, 0.1; lower = -3.0, upper = 3.0),
        MinuitParameter("b", 0.0, 0.1; fixed = true),
    ])
    return migrad(_wl_cost(), params)
end
_wl_ensemble() = mcmc_sample(_wl_minuit(); nsteps = 60, burn = 10, thin = 5,
                            seed = 1, warn = false)

function _precompile_migrad()
    return _wl_minimum()
end

function _precompile_minos()
    return minos(_wl_minimum(), _wl_cost(), 1)
end

function _precompile_contour()
    return contour_ellipse(_wl_minimum(), _wl_cost(), 1, 2; npoints = 6)
end

function _precompile_bounded()
    return _wl_bounded()
end

function _precompile_minuit()
    return _wl_minuit()
end

function _precompile_minuit_minos()
    return minos!(_wl_minuit(), 1)
end

function _precompile_hesse()
    return hesse(_wl_cost(), _wl_minimum().state, Strategy(0))
end

function _precompile_serialize_minimum()
    return to_dict(_wl_minimum())
end

function _precompile_serialize_bounded()
    return to_dict(_wl_bounded())
end

function _precompile_serialize_minos()
    return to_dict(minos(_wl_minimum(), _wl_cost(), 1))
end

function _precompile_least_squares()
    return migrad!(Minuit(LeastSquares([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], ones(4), _wl_line), [1.0, 0.0]))
end

function _precompile_least_squares_data()
    return migrad!(Minuit(LeastSquares(_wl_data(), _wl_line), [1.0, 0.0]))
end

function _precompile_unbinned()
    return migrad!(Minuit(UnbinnedNLL([-0.4, 0.1, 0.3, -0.2, 0.5], _wl_pdf), [0.0, 1.0]))
end

function _precompile_extended_unbinned()
    return migrad!(Minuit(ExtendedUnbinnedNLL([-0.4, 0.1, 0.3, -0.2, 0.5], _wl_density, _wl_integral), [0.0, 1.0, 5.0]))
end

function _precompile_binned()
    return migrad!(Minuit(BinnedNLL([2.0, 5.0, 3.0], [-1.0, -0.3, 0.3, 1.0], _wl_cdf), [0.0, 0.5]))
end

function _precompile_extended_binned()
    return migrad!(Minuit(ExtendedBinnedNLL([2.0, 5.0, 3.0], [-1.0, -0.3, 0.3, 1.0], _wl_scaled_cdf), [0.0, 0.5, 10.0]))
end

function _precompile_cost_sum()
    a = LeastSquares(_wl_data(), _wl_line; name = [:a, :b])
    b = LeastSquares(_wl_data(), _wl_line; name = [:a, :b])
    return migrad!(Minuit(a + b, [1.0, 0.0]))
end

function _precompile_mncontour()
    return mncontour(_wl_minuit(), 1, 2; numpoints = 6)
end

function _precompile_profile()
    return profile(_wl_minuit(), 1; bins = 5)
end

function _precompile_mnprofile()
    return mnprofile(_wl_minuit(), 1; bins = 4)
end

function _precompile_scan()
    return scan(_wl_minuit(), 1; maxsteps = 5)
end

function _precompile_simplex()
    return simplex(Minuit(_wl_quadratic, [0.0, 0.0]))
end

function _precompile_bootstrap()
    return bootstrap(_wl_line, _wl_data(), [1.0, 0.0]; nresample = 2, seed = 1)
end

function _precompile_jackknife()
    return jackknife(_wl_line, _wl_data(), [1.0, 0.0])
end

function _precompile_contour_samples()
    return get_contours_samples(_wl_minuit(); nsamples = 64, adaptive = false, seed = 1, warn = false)
end

function _precompile_mcmc()
    return _wl_ensemble()
end

function _precompile_quantiles()
    return quantiles(_wl_ensemble(), x -> x[1]; warn = false)
end

function _precompile_quantile_band()
    return quantile_band(_wl_ensemble(), (t, x) -> x[1] + x[2] * t, [0.0, 1.0]; warn = false)
end

function _precompile_curve_band()
    return quantile_band(_wl_ensemble(), x -> [x[1], x[1] + x[2]], [0.0, 1.0]; curve = true, warn = false)
end

function _precompile_serialize_ensemble()
    io = IOBuffer()
    save_ensemble(io, _wl_ensemble())
    return load_ensemble(IOBuffer(take!(io)))
end

function _precompile_solution_modes()
    samples = [0.98 1.99; 1.02 2.01; 1.00 2.00; 0.99 2.02; 1.01 1.98; 1.00 1.99]
    return find_solution_modes(samples, _wl_minuit())
end

function _precompile_recipe_contour()
    return RecipesBase.apply_recipe(Dict{Symbol,Any}(), _precompile_contour())
end

function _precompile_recipe_minos()
    return RecipesBase.apply_recipe(Dict{Symbol,Any}(), _precompile_minos())
end

function _precompile_recipe_minimum()
    return RecipesBase.apply_recipe(Dict{Symbol,Any}(), _wl_minimum())
end

const _PRECOMPILE_WORKLOADS = (
    :migrad => _precompile_migrad,
    :minos => _precompile_minos,
    :contour => _precompile_contour,
    :bounded => _precompile_bounded,
    :minuit => _precompile_minuit,
    :minuit_minos => _precompile_minuit_minos,
    :hesse => _precompile_hesse,
    :serialize_minimum => _precompile_serialize_minimum,
    :serialize_bounded => _precompile_serialize_bounded,
    :serialize_minos => _precompile_serialize_minos,
    :least_squares => _precompile_least_squares,
    :least_squares_data => _precompile_least_squares_data,
    :unbinned => _precompile_unbinned,
    :extended_unbinned => _precompile_extended_unbinned,
    :binned => _precompile_binned,
    :extended_binned => _precompile_extended_binned,
    :cost_sum => _precompile_cost_sum,
    :mncontour => _precompile_mncontour,
    :profile => _precompile_profile,
    :mnprofile => _precompile_mnprofile,
    :scan => _precompile_scan,
    :simplex => _precompile_simplex,
    :bootstrap => _precompile_bootstrap,
    :jackknife => _precompile_jackknife,
    :contour_samples => _precompile_contour_samples,
    :mcmc => _precompile_mcmc,
    :quantiles => _precompile_quantiles,
    :quantile_band => _precompile_quantile_band,
    :curve_band => _precompile_curve_band,
    :serialize_ensemble => _precompile_serialize_ensemble,
    :solution_modes => _precompile_solution_modes,
    :recipe_contour => _precompile_recipe_contour,
    :recipe_minos => _precompile_recipe_minos,
    :recipe_minimum => _precompile_recipe_minimum,
)

PrecompileTools.@compile_workload begin
    for (_, workload) in _PRECOMPILE_WORKLOADS
        workload()
    end
end
