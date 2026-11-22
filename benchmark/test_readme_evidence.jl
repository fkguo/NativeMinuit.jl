# SPDX-License-Identifier: LGPL-2.1-or-later
# Synthetic parser records exercise admission bookkeeping, not optimizer accuracy.
using Test, JSON3, Statistics
include(joinpath(@__DIR__,"check_readme_evidence.jl"))
module CollectorArgumentTests
include(joinpath(@__DIR__,"readme_julia_bench.jl"))
end

function write_record(path, record)
    open(path,"w") do io
        JSON3.write(io,record)
    end
end
function synthetic_result(n)
    covariance = [i == j ? 1.0 : 0.0 for i in 1:n for j in 1:n]
    Dict("params"=>zeros(n),"errors"=>ones(n),"covariance"=>covariance,
        "fval"=>1.0,"edm"=>1e-8,"nfcn"=>10,"actual_calls"=>10,
        "is_valid"=>true,"has_covariance"=>true,
        "reached_call_limit"=>false,"above_max_edm"=>false)
end
function synthetic_block(time, seconds)
    Dict("sample_times_ns"=>[time,time],"validated_outputs"=>2,
        "stable_outputs"=>true,"elapsed_seconds"=>seconds)
end
function synthetic_evidence(dir; blocks=2, warmup=2.0, block_seconds=0.2)
    protocol = loadjson(joinpath(@__DIR__,"readme_protocol.json"))
    protocol["rounds"] = 2
    protocol["order"] = ["julia,cpp","cpp,julia"]
    protocol["samples"] = 2
    timing = protocol["timing_reliability"]
    timing["blocks"] = blocks
    timing["fixed_warmup_seconds"] = warmup
    timing["block_min_seconds"] = block_seconds
    timing["initial_samples_retained"] = 2
    write_record(joinpath(dir,"protocol.json"),protocol)
    for name in ("source-hashes-before.json","source-hashes-after.json")
        write_record(joinpath(dir,name),Dict("synthetic-test-input"=>"not-source-provenance"))
    end
    for pair in 1:2, language in ("julia","cpp")
        time = language == "julia" ? 100.0 : 200.0
        rows = Any[]
        for name in protocol["cases"]
            push!(rows,Dict("name"=>name,"samples"=>2,"evals"=>1,
                "sample_times_ns"=>[time,time],"stable_outputs"=>true,
                "validated_outputs"=>3,"explicit_warmups"=>4,"benchmarktools_warmups"=>1,
                "continuous_warmup"=>synthetic_block(time,warmup),
                "timing_blocks"=>[synthetic_block(time,block_seconds) for _ in 1:blocks],
                "result"=>synthetic_result(README_CASE_DIMENSIONS[name])))
        end
        process = Dict("language"=>language,"warmups"=>5,"evals"=>1,"rows"=>rows,
            "julia_threads"=>1,"blas_threads_before"=>1,"blas_threads_after"=>1,
            "fixture_sha256"=>protocol["fixture_sha256"],
            "timing_reliability"=>Dict(key=>timing[key]
                for key in ("blocks","fixed_warmup_seconds","block_min_seconds")))
        write_record(joinpath(dir,"pair-$pair-$language.json"),process)
    end
    refresh_synthetic_comparisons(dir)
    protocol
end

# Fixture expectations use exact equality for the deliberately identical synthetic
# values. They do not invoke the checker's comparison or admission implementation.
function refresh_synthetic_comparisons(dir)
    protocol = loadjson(joinpath(dir,"protocol.json"))
    histories = Dict(name=>Dict("julia"=>Any[],"cpp"=>Any[]) for name in protocol["cases"])
    language_times = Dict(name=>Dict("julia"=>Float64[],"cpp"=>Float64[]) for name in protocol["cases"])
    ratios = Dict(name=>Float64[] for name in protocol["cases"])
    strict = Dict(name=>Bool[] for name in protocol["cases"])
    for pair in 1:protocol["rounds"]
        julia = loadjson(joinpath(dir,"pair-$pair-julia.json"))
        cpp = loadjson(joinpath(dir,"pair-$pair-cpp.json"))
        comparison = Dict{String,Any}()
        for (index,name) in enumerate(protocol["cases"])
            a,b = julia["rows"][index],cpp["rows"][index]
            medians = Dict{String,Float64}()
            for (language,row) in (("julia",a),("cpp",b))
                push!(histories[name][language],row["result"])
                times = [median(block["sample_times_ns"]) for block in row["timing_blocks"]]
                append!(language_times[name][language],times)
                medians[language] = median(times)
            end
            equal = a["result"] == b["result"]
            push!(strict[name],equal)
            push!(ratios[name],medians["julia"]/medians["cpp"])
            comparison[name] = Dict("julia_median_ns"=>medians["julia"],
                "cpp_median_ns"=>medians["cpp"],"strict_parity"=>equal,
                "diagnostic_pass"=>haskey(protocol["rosenbrock_diagnostic_tolerances"],name) ? equal : nothing)
        end
        write_record(joinpath(dir,"pair-$pair-comparison.json"),comparison)
    end
    cases = Dict{String,Any}()
    for name in protocol["cases"]
        spreads = Dict(lang=>maximum(times)/minimum(times)-1 for (lang,times) in language_times[name])
        ratio_spread = maximum(ratios[name])/minimum(ratios[name])-1
        stable = all(value <= protocol["timing_reliability"]["maximum_language_block_spread_fraction"]
            for value in values(spreads)) && ratio_spread <= protocol["timing_reliability"]["maximum_paired_ratio_spread_fraction"]
        identities = Dict(lang=>all(result == results[1] for result in results)
            for (lang,results) in histories[name])
        numeric = all(strict[name]) && all(values(identities))
        cases[name] = Dict("julia_identical_across_processes"=>identities["julia"],
            "cpp_identical_across_processes"=>identities["cpp"],
            "strict_numeric_admitted"=>numeric,"timing_stability_admitted"=>stable,
            "benchmark_admitted"=>numeric && stable,
            "language_block_spread_fraction"=>spreads,"paired_ratio_spread_fraction"=>ratio_spread)
    end
    write_record(joinpath(dir,"summary.json"),Dict("cases"=>cases,
        "all_cases_benchmark_admitted"=>all(value["benchmark_admitted"] for value in values(cases))))
end
function mutate_record(f,dir,name)
    path = joinpath(dir,name)
    record = loadjson(path)
    f(record)
    write_record(path,record)
end

@testset "Portable synthetic README evidence" begin
    mktempdir() do dir
        synthetic_evidence(dir)
        checked = check_evidence(dir)
        @test checked["validation_status"] == "PASS"
        @test checked["all_cases_benchmark_admitted"]
        @test length(checked["checks"]) == 10
        @test isfile(joinpath(dir,"julia-evidence-check.json"))
        # The original defect accepted the old 0.5-second/6-block configuration
        # under a changed protocol. This fixture intentionally uses 2 seconds,
        # 7 blocks and a 0.2-second minimum and requires their actual records.
        synthetic_evidence(dir;blocks=7)
        @test check_evidence(dir;write_output=false)["all_cases_benchmark_admitted"]
        mutate_record(dir,"pair-1-julia.json") do r
            r["rows"][1]["continuous_warmup"]["elapsed_seconds"] = 0.5
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
        synthetic_evidence(dir;blocks=7)
        mutate_record(dir,"pair-1-cpp.json") do r
            pop!(r["rows"][1]["timing_blocks"])
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
        synthetic_evidence(dir)
        mutate_record(dir,"pair-1-cpp.json") do r
            r["rows"][1]["timing_blocks"][1]["elapsed_seconds"] = 0.1
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
        synthetic_evidence(dir)
        mutate_record(dir,"pair-1-cpp.json") do r
            r["rows"][1]["timing_blocks"][1]["sample_times_ns"] = [400.0,400.0]
        end
        refresh_synthetic_comparisons(dir)
        checked = check_evidence(dir;write_output=false)
        @test !checked["all_cases_benchmark_admitted"]
        @test !checked["timing_checks"]["quad_4d"]["admitted"]
        mutate_record(dir,"summary.json") do r
            r["cases"]["quad_4d"]["timing_stability_admitted"] = true
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
        synthetic_evidence(dir)
        for language in ("julia","cpp")
            mutate_record(dir,"pair-2-$language.json") do r
                r["rows"][1]["result"]["params"][1] = 0.25
            end
        end
        refresh_synthetic_comparisons(dir)
        @test !check_evidence(dir;write_output=false)["all_cases_benchmark_admitted"]
        mutate_record(dir,"summary.json") do r
            r["cases"]["quad_4d"]["julia_identical_across_processes"] = true
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
        for (key,value) in (("blocks",true),("blocks",0),("blocks",1.5),
                            ("fixed_warmup_seconds",false),("fixed_warmup_seconds",0),
                            ("fixed_warmup_seconds",-1),("block_min_seconds",0),
                            ("maximum_language_block_spread_fraction",-0.1))
            protocol = synthetic_evidence(dir)
            protocol["timing_reliability"][key] = value
            @test_throws ArgumentError validate_evidence_protocol(protocol)
        end
        for key in ("fixed_warmup_seconds","block_min_seconds")
            protocol = synthetic_evidence(dir)
            protocol["timing_reliability"][key] = Inf
            @test_throws ArgumentError validate_evidence_protocol(protocol)
            protocol["timing_reliability"][key] = NaN
            @test_throws ArgumentError validate_evidence_protocol(protocol)
        end
        for (key,value) in (("strategy",1),("tol",0.2),("maxfcn",100),
                            ("storage_level",1),("warmups",6),("evals",true),
                            ("samples",false),("rounds",1.5))
            protocol = synthetic_evidence(dir)
            protocol[key] = value
            @test_throws ArgumentError validate_evidence_protocol(protocol)
        end
        protocol = synthetic_evidence(dir)
        reverse!(protocol["cases"])
        @test_throws ArgumentError validate_evidence_protocol(protocol)
        synthetic_evidence(dir)
        mutate_record(dir,"pair-1-julia.json") do r
            r["timing_reliability"]["blocks"] = 6
        end
        @test_throws ArgumentError check_evidence(dir;write_output=false)
    end
end

@testset "Collector configuration and strict error budgets" begin
    root = dirname(@__DIR__)
    parseargs = CollectorArgumentTests.collector_arguments
    args = ["unused.json","2",root,"2.0","7","0.2"]
    config = parseargs(args)
    @test config.samples == 2 && config.blocks == 7
    @test config.warmup_seconds == 2.0 && config.block_seconds == 0.2
    for (index,value) in ((2,"true"),(2,"0"),(4,"true"),(4,"NaN"),(4,"Inf"),
                          (4,"0"),(5,"false"),(5,"-1"),(6,"-0.1"),(6,"NaN"))
        invalid = copy(args)
        invalid[index] = value
        @test_throws ArgumentError parseargs(invalid)
    end
    @test_throws ArgumentError parseargs(args[1:3])
    @test approx_field(0.0,1e-12,Dict("rtol"=>1e-6,"atol"=>1e-12))
    @test !approx_field(0.0,1.1e-12,Dict("rtol"=>1e-6,"atol"=>1e-12))
    @test !approx_field([true],[1.0],Dict("rtol"=>0.0,"atol"=>0.0))
end
