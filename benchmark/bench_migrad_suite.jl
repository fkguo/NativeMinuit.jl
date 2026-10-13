# SPDX-License-Identifier: LGPL-2.1-or-later
using NativeMinuit, BenchmarkTools, JSON3

const INPUT_PATH = joinpath(@__DIR__, "fixtures", "migrad_inputs.json")
const INPUTS = JSON3.read(read(INPUT_PATH, String), Dict{String,Any})
const SETTINGS = INPUTS["settings"]

rosenbrock_2(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
quad_nd(x) = sum(abs2, x)
function rosenbrock_nd(x)
    s = 0.0
    for i in 1:(length(x) - 1)
        s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return s
end
function gauss_nll(data)
    return function (p)
        mu, sigma = p
        sigma <= 0 && return 1e30
        s = 0.0
        for x in data
            d = x - mu
            s += log(sigma) + 0.5 * (d * d) / (sigma * sigma)
        end
        s
    end
end
function gauss_nll_nd(data)
    return function (p)
        s = 0.0
        for i in eachindex(data), x in data[i]
            d = x - p[i]
            s += 0.5 * d * d
        end
        s
    end
end

function build_cases()
    inputs = [
        (name="rosenbrock_2d", f=rosenbrock_2, x=[-1.2, 1.0], errors=fill(0.1, 2), up=1.0),
        (name="rosenbrock_10d", f=rosenbrock_nd, x=repeat([-1.2, 1.0], 5), errors=fill(0.1, 10), up=1.0),
        (name="quad_4d", f=quad_nd, x=ones(4), errors=fill(0.1, 4), up=1.0),
        (name="gauss_ll_2_100", f=gauss_nll(Float64.(INPUTS["gauss_2_100"])), x=[1.0, 2.0], errors=fill(0.1, 2), up=0.5),
        (name="gauss_ll_10_1000", f=gauss_nll_nd([Float64.(v) for v in INPUTS["gauss_10_1000"]]), x=zeros(10), errors=fill(0.1, 10), up=0.5),
    ]
    [merge(c, (name=c.name * (s == 0 ? "" : "_s1"), strategy=Int(s)))
     for c in inputs for s in SETTINGS["strategies"]]
end
const CASES = build_cases()

# Includes fresh CostFunction and all solver/result work; excludes immutable data
# preparation. The returned minimum includes its covariance and validity flags.
function run_fit(c)
    cf = CostFunction(c.f, c.up)
    migrad(cf, c.x, c.errors; strategy=Strategy(c.strategy),
           tol=SETTINGS["tol"], maxfcn=SETTINGS["maxfcn"],
           storage_level=SETTINGS["storage_level"])
end
const JULIA_PERF_BENCHMARKS = [c.name => @benchmarkable(run_fit($c), evals=1) for c in CASES]

function julia_perf_validate()
    outputs = Dict{String,Any}()
    for c in CASES
        result = run_fit(c)
        is_valid(result) || error("invalid benchmark minimum: $(c.name)")
        isfinite(fval(result)) || error("nonfinite benchmark minimum: $(c.name)")
        outputs[c.name] = Dict("result" => NativeMinuit.to_dict(result),
            "config" => Dict("strategy"=>c.strategy, "tol"=>SETTINGS["tol"],
                "maxfcn"=>SETTINGS["maxfcn"], "storage_level"=>SETTINGS["storage_level"],
                "x0"=>c.x, "errors"=>c.errors, "errordef"=>c.up),
            "operation" => "fresh CostFunction + full MIGRAD + materialized minimum")
    end
    outputs
end
