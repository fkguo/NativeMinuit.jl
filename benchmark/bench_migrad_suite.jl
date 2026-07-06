# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Phase 0 benchmark suite — drives the §3.4 evidence gate.
#
# Defines `JULIA_PERF_BENCHMARKS` per the julia-perf skill contract
# (.claude/skills/julia-perf/templates/bench-suite-template.jl).
# Each entry is a (name => zero-arg function) pair; `scripts/run_perf.jl`
# wraps with BenchmarkTools, captures medians + allocations, emits
# artifacts under `.julia-perf/runs/<timestamp>/`.
#
# The §3.3 ROADMAP corpus:
#
# | Name                       | n free | Status     |
# |----------------------------|--------|------------|
# | Rosenbrock-2               | 2      | blocking   |
# | Rosenbrock-10              | 10     | blocking   |
# | Quad4F                     | 4      | blocking   |
# | Gauss-LL-2 × 100           | 2      | blocking   |
# | Gauss-LL-10 × 1000         | 10     | blocking   |
# | Gauss-LL-40 × 1000         | 40     | diagnostic |
# | Cheap-FCN long fit         | 4      | diagnostic |
#
# Phase 0 first cut benchmarks only MIGRAD wall time (Criterion 2).
# The C++ comparison driver is `benchmark/compare_cpp.jl` (separate).

using NativeMinuit
using Random

# Lock RNG so the Gauss-LL benchmarks are reproducible across runs
Random.seed!(0xCAFE_F00D)

# ─────────────────────────────────────────────────────────────────────────────
# FCN definitions
# ─────────────────────────────────────────────────────────────────────────────

rosenbrock_2(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

quad_nd(x) = sum(abs2, x)

function rosenbrock_nd(x)
    s = 0.0
    @inbounds for i in 1:(length(x) - 1)
        s += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return s
end

# Gaussian negative log-likelihood: given data drawn from N(μ, σ²),
# fit μ and σ. f(μ, σ) = sum( log(σ) + (x_i − μ)² / (2σ²) ).
function make_gauss_nll(n_events::Integer, true_mu::Float64, true_sigma::Float64)
    data = true_mu .+ true_sigma .* randn(n_events)
    return function (par)
        μ, σ = par[1], par[2]
        if σ <= 0
            return 1e30  # Phase 0 has no bounds; soft-rejection
        end
        s = 0.0
        @inbounds for x in data
            d = x - μ
            s += log(σ) + 0.5 * (d * d) / (σ * σ)
        end
        return s
    end
end

# n-dim Gaussian product NLL: n parameters, all independent.
function make_gauss_nll_nd(n_pars::Integer, n_events::Integer)
    n_data_per = max(1, n_events ÷ n_pars)
    truths = randn(n_pars)
    data = [truths[i] .+ randn(n_data_per) for i in 1:n_pars]
    return function (par)
        s = 0.0
        @inbounds for i in 1:n_pars
            μ = par[i]
            for x in data[i]
                d = x - μ
                s += 0.5 * d * d
            end
        end
        return s
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark suite (julia-perf contract)
# ─────────────────────────────────────────────────────────────────────────────

const JULIA_PERF_BENCHMARKS = [
    "rosenbrock_2d" => (() -> begin
        cf = CostFunction(rosenbrock_2)
        migrad(cf, [-1.2, 1.0], [0.1, 0.1])
    end),

    "rosenbrock_10d" => (() -> begin
        cf = CostFunction(rosenbrock_nd)
        x0 = [(-1.2, 1.0)[1 + (i & 1)] for i in 0:9]
        errs = fill(0.1, 10)
        migrad(cf, x0, errs)
    end),

    "quad_4d" => (() -> begin
        cf = CostFunction(quad_nd)
        migrad(cf, [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1])
    end),

    "gauss_ll_2_100" => (() -> begin
        cf = CostFunction(make_gauss_nll(100, 2.0, 1.0), 0.5)
        migrad(cf, [1.0, 2.0], [0.1, 0.1])
    end),

    "gauss_ll_10_1000" => (() -> begin
        cf = CostFunction(make_gauss_nll_nd(10, 1000), 0.5)
        migrad(cf, zeros(10), fill(0.1, 10))
    end),

    # ── Strategy(1) variants — Phase 1 完成判据 #6 ─────────────────────
    # Strategy(1) is iminuit's default; the MIGRAD outer loop runs an
    # inner HESSE when Dcovar > 0.05. Adds ~20-40% nfcn overhead vs
    # Strategy(0) but produces a more accurate final covariance.

    "rosenbrock_2d_s1" => (() -> begin
        cf = CostFunction(rosenbrock_2)
        migrad(cf, [-1.2, 1.0], [0.1, 0.1]; strategy = Strategy(1))
    end),

    "rosenbrock_10d_s1" => (() -> begin
        cf = CostFunction(rosenbrock_nd)
        x0 = [(-1.2, 1.0)[1 + (i & 1)] for i in 0:9]
        migrad(cf, x0, fill(0.1, 10); strategy = Strategy(1))
    end),

    "quad_4d_s1" => (() -> begin
        cf = CostFunction(quad_nd)
        migrad(cf, [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1];
                strategy = Strategy(1))
    end),

    "gauss_ll_2_100_s1" => (() -> begin
        cf = CostFunction(make_gauss_nll(100, 2.0, 1.0), 0.5)
        migrad(cf, [1.0, 2.0], [0.1, 0.1]; strategy = Strategy(1))
    end),

    "gauss_ll_10_1000_s1" => (() -> begin
        cf = CostFunction(make_gauss_nll_nd(10, 1000), 0.5)
        migrad(cf, zeros(10), fill(0.1, 10); strategy = Strategy(1))
    end),
]
