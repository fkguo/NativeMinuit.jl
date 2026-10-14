# SPDX-License-Identifier: LGPL-2.1-or-later
# Generated with tools/cpp_gradient_witness.cxx against pinned Minuit2 v6.24.0.
# Budgets frozen before comparing: derivative 1e-8, curvature 1e-6 absolute;
# quadratic final params/fval 1e-10 relative, near-zero 1e-12/1e-18 absolute;
# EDM/covariance 1e-6 relative, 1e-12 absolute as the existing quadratic tier.
@testset "bounded gradients and near-boundary quadratic C++ oracle" begin
    cases = [("lower_large", 1, 3., 100., 1.), ("upper_large", 2, 1., 100., 2.),
             ("both_large", 3, 1., 100., 2.), ("near_lower", 1, 1e-7, 1e-5, 1.),
             ("near_upper", 2, 4-1e-7, 1e-5, 2.), ("near_both", 3, 1e-7, 1e-5, 2.),
             ("lower_small", 1, 3., .1, 1.), ("upper_small", 2, 1., .1, 2.),
             ("both_small", 3, 1., .1, 2.)]
    path = joinpath(@__DIR__, "reference_data", "bounded_gradient_reference.txt")
    refs = Dict(parts[1] => parse.(Float64, parts[2:end]) for parts in split.(readlines(path)))
    for (name, kind, start, error, target) in cases
        @testset "$name" begin
            params = Parameters([MinuitParameter("x", start, error;
                lower=kind==2 ? NaN : 0., upper=kind==1 ? NaN : 4.)])
            q(x) = (x[1]-target)^2
            cf = NativeMinuit._wrap_fcn_internal_to_external(CostFunction(q), params)
            x = NativeMinuit.initial_int_values(params)
            errs = NativeMinuit.initial_int_errors(params)
            par = MinimumParameters(x, errs, cf(x))
            ref = refs[name]
            init = NativeMinuit.initial_gradient(par, errs, cf; has_limits=[true])
            @test init.grad[1] ≈ ref[2] rtol=1e-12 atol=1e-14
            @test init.g2[1] ≈ ref[3] rtol=1e-12 atol=1e-14
            @test init.gstep[1] ≈ ref[4] rtol=1e-12 atol=1e-14
            for forced in (false, true), threaded in (false, true)
                prev = forced ? FunctionGradient([1.], [1e-14], [10.]) : init
                out = FunctionGradient(zeros(1), zeros(1), zeros(1))
                numerical_gradient!(out, similar(x), par, prev, cf, Strategy(0);
                                    threaded, has_limits=[true])
                expected = forced ? ref[8:10] : ref[5:7]
                @test out.grad[1] ≈ expected[1] atol=1e-8 rtol=0
                @test out.g2[1] ≈ expected[2] atol=1e-6 rtol=0
                @test out.gstep[1] ≈ expected[3] atol=1e-12 rtol=0
            end
            result = migrad(CostFunction(q), params; strategy=Strategy(0), tol=1e-12, maxfcn=1000)
            @test result.internal.is_valid == (ref[11] == 1)
            @test values(result)[1] ≈ ref[12] rtol=1e-10 atol=1e-12
            @test fval(result) ≈ ref[13] rtol=1e-10 atol=1e-18
            @test edm(result) ≈ ref[14] rtol=1e-6 atol=1e-12
            @test NativeMinuit.covariance(result)[1,1] ≈ ref[15] rtol=1e-6 atol=1e-12
        end
    end
end
