# SPDX-License-Identifier: LGPL-2.1-or-later
using Test, NativeMinuit, JSON3
include(joinpath(@__DIR__, "..", "scripts", "run_perf.jl"))
include(joinpath(@__DIR__, "comparison_evidence.jl"))

@testset "loaded source and immutable input identity" begin
    cfg=TOML.parsefile(joinpath(@__DIR__, "perf-config.toml"))
    actual=loaded_source(cfg, joinpath(@__DIR__, "perf-config.toml"))
    @test actual["entry_relative_path"] == joinpath("src", "NativeMinuit.jl")
    mktempdir() do wrong
        bad=deepcopy(cfg); bad["target"]["source_root"]=wrong
        @test_throws ErrorException loaded_source(bad,joinpath(@__DIR__,"perf-config.toml"))
    end
    cases, suite=load_suite_cases(joinpath(@__DIR__,"bench_migrad_suite.jl"); with_module=true)
    a=suite_validation(suite,first.(cases)); b=suite_validation(suite,first.(cases))
    @test hash_json(a) == hash_json(b)
    @test length(a) == 10
    @test all(v["result"]["valid"] for v in values(a))
    @test_throws ErrorException suite_validation(suite,["missing"])
    initial=file_hashes(@__DIR__, ["fixtures/migrad_inputs.json"])
    @test !isempty(initial)
end

@testset "deadline terminates suite-owned descendants" begin
    mktempdir() do d
        pidfile=joinpath(d,"descendant.pid")
        # sleep inherits the worker process group. It is deliberately not
        # detached, as required by the runner's subprocess ownership contract.
        suite="using BenchmarkTools\np=run(`sleep 120`;wait=false)\nwrite(" * repr(pidfile) * ",string(getpid(p)))\nJULIA_PERF_BENCHMARKS=[\"slow\"=>@benchmarkable(sleep(120),evals=1)]\n"
        write(joinpath(d,"suite.jl"),suite)
        cfg=Dict("target"=>Dict("source_root"=>d,"source_files"=>["suite.jl"]),
                 "benchmark"=>Dict("suite_file"=>"suite.jl","samples"=>2,"seconds"=>0.01,"timeout_seconds"=>15))
        config=joinpath(d,"config.toml");open(io->TOML.print(io,cfg),config,"w")
        @test main(["--config",config,"--out-dir",joinpath(d,"output"),"--save-baseline"]) == 1
        @test isfile(pidfile)
        if isfile(pidfile)
            pid=parse(Int,read(pidfile,String))
            gone=timedwait(()->ccall(:kill,Cint,(Cint,Cint),pid,0)!=0,3.;pollint=.05)
            @test gone == :ok
            println("owned descendant PID ",pid," absent after deadline: ",gone==:ok)
        end
    end
end

@testset "historical comparisons never imply PASS" begin
    mktempdir() do d
        path=joinpath(d,"legacy.json")
        write_json(path,[Dict("name"=>"quad_4d", "median_ns"=>1., "valid"=>false),
                         Dict("name"=>"quad_4d", "median_ns"=>2.)])
        report=comparison_report(Dict("julia"=>path, "cpp"=>nothing, "python"=>nothing))
        @test report["verdict"] == "UNVERIFIED"
        @test report["exit_code"] == 2
        @test length(report["cases"]) == 5
        @test length(report["cases"]["quad_4d"]["sources"]["julia"]) == 2
        @test report["cases"]["quad_4d"]["sources"]["cpp"] === nothing
        @test report["sources"]["julia"]["raw"][1].valid == false
        empty=joinpath(d,"empty.json"); write(empty,"[]")
        @test comparison_report(Dict("julia"=>empty,"cpp"=>empty))["verdict"] == "UNVERIFIED"
        corrupt=joinpath(d,"corrupt.json");write(corrupt,"{oops")
        failure=comparison_report(Dict("julia"=>corrupt,"cpp"=>nothing))
        @test failure["verdict"] == "FAIL"
        @test failure["sources"]["julia"]["raw_text"] == "{oops"
    end
end

@testset "partial worker timeout retains all case statuses" begin
    mktempdir() do d
        write(joinpath(d,"suite.jl"), "using BenchmarkTools\nJULIA_PERF_BENCHMARKS=[\"a_completed\"=>@benchmarkable(zeros(16),evals=1),\"b_slow\"=>@benchmarkable(sleep(120),evals=1),\"c_not_started\"=>@benchmarkable(zeros(8),evals=1)]\n")
        cfg=Dict("target"=>Dict("source_root"=>d,"source_files"=>["suite.jl"]),
                 "benchmark"=>Dict("suite_file"=>"suite.jl","samples"=>2,"seconds"=>0.01,"timeout_seconds"=>30))
        config=joinpath(d,"config.toml");open(io->TOML.print(io,cfg),config,"w")
        output=joinpath(d,"output")
        @test main(["--config",config,"--out-dir",output,"--save-baseline"]) == 1
        @test !isfile(joinpath(output,"baseline.json"))
        rows=JSON3.read(read(joinpath(output,"benchmarks.json"),String))
        @test length(rows) == 1 && rows[1].name == "a_completed"
        statuses=JSON3.read(read(joinpath(output,"case_status.json"),String))
        @test statuses.a_completed.status == "completed"
        @test statuses.b_slow.status == "timed_out"
        @test statuses.c_not_started.status == "not_started"
    end
end
