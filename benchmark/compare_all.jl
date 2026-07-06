# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Master comparison driver: Julia NativeMinuit vs C++ Minuit2 vs Python iminuit.
# Times MIGRAD + MINOS + MNCONTOUR across the 5 §3.3 benchmark FCNs.
#
# Usage:
#   julia --project=scripts benchmark/compare_all.jl
#
# Outputs:
#   - stdout: comparative table (Julia / C++ / iminuit ratios)
#   - benchmark/.julia-perf/runs/latest/compare_all.json (machine-readable)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "scripts"))
using JSON3
using BenchmarkTools
using Printf
using LinearAlgebra
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Fix BLAS threads = 1 on both sides for an apples-to-apples comparison
BLAS.set_num_threads(1)
# ─────────────────────────────────────────────────────────────────────────────

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const CPP_BENCH = joinpath(REPO_ROOT, "benchmark", "cpp", "build", "cpp_bench")
const PYTHON_BENCH = joinpath(REPO_ROOT, "benchmark", "python_iminuit_bench.py")

# Load NativeMinuit (development version under REPO_ROOT)
Pkg.develop(path=REPO_ROOT)
using NativeMinuit

# ─────────────────────────────────────────────────────────────────────────────
# FCN definitions — match bench_migrad_suite.jl + cpp_bench.cxx + python bench
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(0xCAFE_F00D)

rosenbrock_2(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

quad_nd(x) = sum(abs2, x)

function rosenbrock_nd(x)
    s = 0.0
    @inbounds for i in 1:(length(x) - 1)
        s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return s
end

function make_gauss_nll(n_events, true_mu, true_sigma)
    data = true_mu .+ true_sigma .* randn(n_events)
    return function(par)
        μ, σ = par[1], par[2]
        σ <= 0 && return 1e30
        s = 0.0
        @inbounds for x in data
            d = x - μ
            s += log(σ) + 0.5 * (d*d) / (σ*σ)
        end
        return s
    end
end

function make_gauss_nll_nd(n_pars, n_events)
    n_per = max(1, n_events ÷ n_pars)
    truths = randn(n_pars)
    data = [truths[i] .+ randn(n_per) for i in 1:n_pars]
    return function(par)
        s = 0.0
        @inbounds for i in 1:n_pars
            μ = par[i]
            for x in data[i]; d = x - μ; s += 0.5 * d * d; end
        end
        return s
    end
end

# Re-seed so each FCN factory produces the SAME data as the Python bench
function build_cases()
    Random.seed!(0xCAFE_F00D)
    return [
        ("rosenbrock_2d",       rosenbrock_2,
                                  [-1.2, 1.0], [0.1, 0.1], 1.0),
        ("rosenbrock_10d",      rosenbrock_nd,
                                  [(-1.2, 1.0)[1 + (i & 1)] for i in 0:9],
                                  fill(0.1, 10), 1.0),
        ("quad_4d",             quad_nd,
                                  [1.0, 1.0, 1.0, 1.0],
                                  [0.1, 0.1, 0.1, 0.1], 1.0),
        ("gauss_ll_2_100",      make_gauss_nll(100, 2.0, 1.0),
                                  [1.0, 2.0], [0.1, 0.1], 0.5),
        ("gauss_ll_10_1000",    make_gauss_nll_nd(10, 1000),
                                  zeros(10), fill(0.1, 10), 0.5),
    ]
end

# ─────────────────────────────────────────────────────────────────────────────
# Bench helpers
# ─────────────────────────────────────────────────────────────────────────────

function bench_migrad(name, f, x0, errs, up)
    setup = () -> NativeMinuit.CostFunction(f, up)
    b = @benchmark NativeMinuit.migrad(cf, $x0, $errs) setup=(cf = $setup()) samples=50 evals=1
    return median(b).time   # ns
end

function bench_minos(name, f, x0, errs, up)
    # Time MINOS only (not MIGRAD); rebuild fmin once per sample for fair
    # comparison with cpp_bench's `minos_bench` template.
    cf_for_setup = NativeMinuit.CostFunction(f, up)
    b = @benchmark NativeMinuit.minos(fmin, cf, 1) setup=(
        cf  = NativeMinuit.CostFunction($f, $up);
        fmin = NativeMinuit.migrad(cf, $x0, $errs)
    ) samples=20 evals=1
    return median(b).time
end

function bench_mncontour(name, f, x0, errs, up)
    # MNCONTOUR = full MnContours algorithm (contour_exact in NativeMinuit).
    # n_per_par = 30 points — matches python_iminuit_bench + cpp_bench.
    b = @benchmark NativeMinuit.contour_exact(fmin, cf, 1, 2; npoints=30) setup=(
        cf  = NativeMinuit.CostFunction($f, $up);
        fmin = NativeMinuit.migrad(cf, $x0, $errs)
    ) samples=10 evals=1
    return median(b).time
end

# ─────────────────────────────────────────────────────────────────────────────
# Run all three sides
# ─────────────────────────────────────────────────────────────────────────────

function run_julia()
    println(stderr, "→ Running Julia (NativeMinuit) benchmarks ...")
    out = Dict{String,Dict{String,Float64}}()
    for (name, f, x0, errs, up) in build_cases()
        print(stderr, "    $name ... ")
        m  = bench_migrad(name, f, x0, errs, up)
        mn = bench_minos(name, f, x0, errs, up)
        mc = bench_mncontour(name, f, x0, errs, up)
        out[name] = Dict("migrad" => m, "minos" => mn, "mncontour" => mc)
        @printf(stderr, "migrad=%6.0fμs  minos=%6.0fμs  mncontour=%7.0fμs\n",
                m/1000, mn/1000, mc/1000)
    end
    return out
end

function run_cpp()
    println(stderr, "→ Running C++ Minuit2 benchmarks ...")
    isfile(CPP_BENCH) ||
        error("cpp_bench not built: $CPP_BENCH. Build via " *
              "`cmake --build benchmark/cpp/build`.")
    data = JSON3.read(read(`$CPP_BENCH`, String))
    out = Dict{String,Dict{String,Float64}}()
    for b in data
        nm = String(b.name)
        # cpp_bench emits: <name>, <name>_s1, <name>_minos, <name>_mncontour
        if endswith(nm, "_minos")
            out_name = replace(nm, "_minos" => "")
            haskey(out, out_name) || (out[out_name] = Dict{String,Float64}())
            out[out_name]["minos"] = Float64(b.median_ns)
        elseif endswith(nm, "_mncontour")
            out_name = replace(nm, "_mncontour" => "")
            haskey(out, out_name) || (out[out_name] = Dict{String,Float64}())
            out[out_name]["mncontour"] = Float64(b.median_ns)
        elseif endswith(nm, "_s1")
            # skip Strategy(1) variants (Julia bench is Strategy(0))
            continue
        else
            haskey(out, nm) || (out[nm] = Dict{String,Float64}())
            out[nm]["migrad"] = Float64(b.median_ns)
        end
    end
    return out
end

function run_python()
    println(stderr, "→ Running Python iminuit benchmarks ...")
    isfile(PYTHON_BENCH) ||
        error("$PYTHON_BENCH not found")
    raw = read(`python3 $PYTHON_BENCH`, String)
    data = JSON3.read(raw)
    out = Dict{String,Dict{String,Float64}}()
    for b in data
        out[String(b.name)] = Dict(
            "migrad"    => Float64(b.migrad_ns),
            "minos"     => Float64(b.minos_ns),
            "mncontour" => Float64(b.mncontour_ns),
        )
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Print comparative table
# ─────────────────────────────────────────────────────────────────────────────

function print_table(julia_t, cpp_t, py_t)
    cases = ["rosenbrock_2d", "rosenbrock_10d", "quad_4d",
              "gauss_ll_2_100", "gauss_ll_10_1000"]
    ops   = ["migrad", "minos", "mncontour"]

    println()
    println("NativeMinuit vs C++ Minuit2 vs Python iminuit — median wall time per call")
    println("Strategy(0); BLAS.set_num_threads(1); Apple M3.")
    println()

    for op in ops
        println("── $op ─────────────────────────────────────────────────────────────────────")
        @printf("%-22s %12s %12s %12s   %10s %10s\n",
                "case", "Julia(μs)", "C++(μs)", "iminuit(μs)", "J/C++", "J/iminuit")
        println("─"^96)
        for c in cases
            j  = get(get(julia_t, c, Dict()), op, NaN) / 1000
            cp = get(get(cpp_t,   c, Dict()), op, NaN) / 1000
            py = get(get(py_t,    c, Dict()), op, NaN) / 1000
            rJC = isnan(j) || isnan(cp) ? NaN : j / cp
            rJP = isnan(j) || isnan(py) ? NaN : j / py
            @printf("%-22s %12.2f %12.2f %12.2f   %10.3f %10.3f\n",
                    c, j, cp, py, rJC, rJP)
        end
        println()
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    julia_t = run_julia()
    cpp_t   = run_cpp()
    py_t    = run_python()
    print_table(julia_t, cpp_t, py_t)

    # persist as JSON
    out_dir = joinpath(REPO_ROOT, "benchmark", ".julia-perf", "runs", "latest")
    isdir(out_dir) || mkpath(out_dir)
    out_path = joinpath(out_dir, "compare_all.json")
    open(out_path, "w") do io
        JSON3.write(io, Dict("julia" => julia_t,
                              "cpp" => cpp_t,
                              "python_iminuit" => py_t))
    end
    println(stderr, "→ Wrote $out_path")
    return 0
end

main()
