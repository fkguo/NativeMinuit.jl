# SPDX-License-Identifier: LGPL-2.1-or-later
using Test
include(joinpath(@__DIR__, "..", "scripts", "run_perf.jl"))

@testset "Requested BLAS configuration is an acceptance condition" begin
    previous = BLAS.get_num_threads()
    try
        mktempdir() do directory
            suite = joinpath(directory, "suite.jl")
            write(suite, "using BenchmarkTools, LinearAlgebra\nJULIA_PERF_BENCHMARKS=[\"changes_threads\" => @benchmarkable(BLAS.set_num_threads(2), evals=1)]\n")
            config = joinpath(directory, "config.toml")
            cfg = Dict("target" => Dict("source_root"=>directory, "source_files"=>["suite.jl"]),
                       "blas"=>Dict("num_threads"=>1),
                       "benchmark"=>Dict("suite_file"=>"suite.jl", "samples"=>2, "seconds"=>0.01))
            open(io->TOML.print(io, cfg), config, "w")
            output = joinpath(directory, "output"); mkpath(output)
            @test worker_main(Dict("save_baseline"=>true), cfg, config, output) == 1
            @test !isfile(joinpath(output, "baseline.json"))
            summary = JSON3.read(read(joinpath(output, "summary.json"), String))
            @test summary.verdict == "FAIL"
            @test occursin("BLAS thread count differs", summary.hard_failures[1].detail)
            @test JSON3.read(read(joinpath(output,"case_status.json"),String)).changes_threads.status == "failed"
            write(suite, "using BenchmarkTools, LinearAlgebra, NativeMinuit\nx=ones(16)\nJULIA_PERF_BENCHMARKS=[\"dot\" => @benchmarkable(dot(\$x,\$x), evals=1)]\n")
            # Bind the developed package itself; transient test output is not
            # dependency source and must not enter the runtime fingerprint.
            cfg["target"] = Dict("module"=>"NativeMinuit",
                                 "source_root"=>normpath(joinpath(@__DIR__,"..")),
                                 "source_files"=>["src","Project.toml"])
            cfg["benchmark"]["timeout_seconds"] = 30
            open(io->TOML.print(io, cfg), config, "w")
            positive = joinpath(directory,"positive")
            @test main(["--config",config,"--out-dir",positive,"--save-baseline"]) == 0
            manifest = JSON3.read(read(joinpath(positive,"manifest.json"),String))
            @test manifest.runtime.blas_threads == 1
            @test manifest.runtime.blas_environment.OMP_NUM_THREADS == "1"
            @test JSON3.read(read(joinpath(positive,"summary.json"),String)).verdict == "BASELINE_CREATED"
        end
    finally
        BLAS.set_num_threads(previous)
    end
end
