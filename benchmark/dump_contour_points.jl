# SPDX-License-Identifier: LGPL-2.1-or-later
# Diagnostic: dump full MNCONTOUR point coordinates so they can be diff'd
# against the C++ implementation (build via benchmark/cpp/dump_contour_points.cxx).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
using NativeMinuit
using Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)
Random.seed!(0xCAFE_F00D)

rosenbrock_2(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
quad_nd(x) = sum(abs2, x)
rosenbrock_nd(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2 for i in 1:length(x)-1)
function make_gauss_nll(n_events, mu0, sigma0)
    data = mu0 .+ sigma0 .* randn(n_events)
    par -> begin
        μ, σ = par[1], par[2]; σ<=0 && return 1e30
        s = 0.0
        for x in data; d=x-μ; s += log(σ)+0.5*d*d/(σ*σ); end
        s
    end
end
function make_gauss_nll_nd(n_pars, n_events)
    n_per = max(1, n_events ÷ n_pars)
    truths = randn(n_pars); data = [truths[i] .+ randn(n_per) for i in 1:n_pars]
    par -> begin
        s = 0.0
        for i in 1:n_pars; μ=par[i]; for x in data[i]; d=x-μ; s+=0.5*d*d; end; end
        s
    end
end

cases = [
    ("rosenbrock_2d",  rosenbrock_2, [-1.2,1.0], [0.1,0.1], 1.0),
    ("rosenbrock_10d", rosenbrock_nd, [(-1.2,1.0)[1+(i & 1)] for i in 0:9], fill(0.1,10), 1.0),
    ("quad_4d",        quad_nd, fill(1.0,4), fill(0.1,4), 1.0),
    ("gauss_ll_2_100", make_gauss_nll(100,2.0,1.0), [1.0,2.0], [0.1,0.1], 0.5),
    ("gauss_ll_10_1000", make_gauss_nll_nd(10,1000), zeros(10), fill(0.1,10), 0.5),
]

for (name, f, x0, errs, up) in cases
    cf = NativeMinuit.CostFunction(f, up)
    fmin = NativeMinuit.migrad(cf, x0, errs)
    ce = NativeMinuit.contour_exact(fmin, cf, 1, 2; npoints=30)
    println("== $name ==")
    println("  npts=$(length(ce.points))  nfcn=$(ce.nfcn)")
    @printf("  fmin_x=%.15g  fmin_y=%.15g\n",
            fmin.state.parameters.x[1], fmin.state.parameters.x[2])
    for (i, p) in enumerate(ce.points)
        @printf("  pt[%d] = (%.15g, %.15g)\n", i-1, p[1], p[2])
    end
end
