# SPDX-License-Identifier: LGPL-2.1-or-later
# Run only via compare_readme.py, in the pinned benchmark environment.
using NativeMinuit, BenchmarkTools, JSON3, LinearAlgebra, SHA

function positive_count(raw, name)
    value = tryparse(Int, raw)
    value !== nothing && value > 0 || throw(ArgumentError("$name must be a positive integer"))
    value
end

function positive_seconds(raw, name)
    value = tryparse(Float64, raw)
    value !== nothing && isfinite(value) && value > 0 ||
        throw(ArgumentError("$name must be finite and positive"))
    value
end

function collector_arguments(args)
    length(args) == 6 || throw(ArgumentError(
        "usage: readme_julia_bench.jl OUTPUT_JSON SAMPLES SOURCE_ROOT WARMUP_SECONDS BLOCKS BLOCK_SECONDS"))
    (output=args[1], samples=positive_count(args[2], "samples"), source_root=realpath(args[3]),
     warmup_seconds=positive_seconds(args[4], "warmup seconds"),
     blocks=positive_count(args[5], "blocks"), block_seconds=positive_seconds(args[6], "block seconds"))
end

function materialized_fit(c)
    cf = CostFunction(c.f, c.up)
    m = migrad(cf, c.x, c.errors; strategy=Strategy(0), tol=0.1,
               maxfcn=10000, storage_level=0)
    cov = NativeMinuit.covariance(m)
    # Explicit row-major full matrix gives both collectors the same owned outputs.
    flat = cov === nothing ? Float64[] : [cov[i,j] for i in axes(cov,1) for j in axes(cov,2)]
    (fval=fval(m), edm=edm(m), nfcn=nfcn(m), actual_calls=NativeMinuit.ncalls(cf),
     is_valid=is_valid(m), has_covariance=has_covariance(m),
     reached_call_limit=reached_call_limit(m), above_max_edm=above_max_edm(m),
     params=copy(Base.values(m)), errors=cov === nothing ? Float64[] : sqrt.(diag(cov)), covariance=flat)
end

function timed_block(c, first, seconds)
    times=Float64[]; stable=true; validated=0; before=Base.gc_num(); start=time_ns()
    while (time_ns()-start)*1e-9 < seconds
        t=time_ns(); result=materialized_fit(c); push!(times,Float64(time_ns()-t))
        stable &= isequal(first,result) && result.nfcn==result.actual_calls
        validated += 1
    end
    elapsed=(time_ns()-start)*1e-9; diff=Base.GC_Diff(Base.gc_num(),before)
    Dict("sample_times_ns"=>times,"stable_outputs"=>stable,"validated_outputs"=>validated,
         "elapsed_seconds"=>elapsed,"gc_time_seconds"=>diff.total_time*1e-9,
         "allocated_bytes"=>diff.allocd)
end

function collect_case(c, samples, warmup_seconds, block_count, block_seconds)
    first = materialized_fit(c)
    for _ in 2:4
        materialized_fit(c)
    end
    received = typeof(first)[]
    slot = Ref(first)
    # Fresh solver state per eval; immutable inputs prepared before timing.
    b = @benchmarkable($slot[] = materialized_fit($c),
        teardown=(push!($received, $slot[])), evals=1,
        samples=samples, seconds=3600, gctrial=true, gcsample=false, overhead=0)
    trial = run(b)
    warmup=timed_block(c,first,warmup_seconds)
    blocks=[timed_block(c,first,block_seconds) for _ in 1:block_count]
    BLAS.get_num_threads() == 1 || error("BLAS thread count changed during timing")
    Dict("continuous_warmup"=>warmup,"timing_blocks"=>blocks,
         "name"=>c.name, "samples"=>length(trial.times), "evals"=>trial.params.evals,
         "sample_times_ns"=>trial.times, "sample_gc_times_ns"=>trial.gctimes,
         "memory_bytes"=>trial.memory, "allocations"=>trial.allocs,
         "stable_outputs"=>all(isequal(first),received),
         "validated_outputs"=>length(received), "explicit_warmups"=>4, "benchmarktools_warmups"=>1,
         "result"=>first, "different_outputs"=>filter(x->!isequal(first,x),received))
end

function main(args=ARGS)
    config = collector_arguments(args)
    output, samples, source_root = config.output, config.samples, config.source_root
    loaded_source = realpath(pathof(NativeMinuit))
    expected_source = realpath(joinpath(source_root, "src", "NativeMinuit.jl"))
    loaded_source == expected_source || error("loaded NativeMinuit source does not match SOURCE_ROOT")
    # Loading the fixture suite is delayed until configuration and source identity pass.
    Base.include(@__MODULE__, joinpath(source_root, "benchmark", "bench_migrad_suite.jl"))
    Threads.nthreads()==1 || error("one Julia thread required")
    ENV["OMP_NUM_THREADS"]=="1" || error("OMP_NUM_THREADS must be 1")
    ENV["OPENBLAS_NUM_THREADS"]=="1" || error("OPENBLAS_NUM_THREADS must be 1")
    BLAS.set_num_threads(1)
    # Force an elementary BLAS call before recording its effective setting.
    mul!(zeros(2,2),ones(2,2),ones(2,2))
    BLAS.get_num_threads()==1 || error("one actual BLAS thread required")
    names=["quad_4d","rosenbrock_2d","rosenbrock_10d","gauss_ll_2_100","gauss_ll_10_1000"]
    cases=[only(filter(c->c.name==name,CASES)) for name in names]
    rows=Dict{String,Any}[]
    report=Dict("schema_version"=>1,"language"=>"julia","julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(),"blas_threads_before"=>BLAS.get_num_threads(),
        "blas_config"=>sprint(show,BLAS.get_config()),"cpu"=>Sys.CPU_NAME,
        "fixture_sha256"=>bytes2hex(sha256(read(INPUT_PATH))),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(source_root,"benchmark","Manifest.toml")))),
        "package_path"=>loaded_source,"warmups"=>5,"evals"=>1,"rows"=>rows,
        "timing_reliability"=>Dict("fixed_warmup_seconds"=>config.warmup_seconds,
            "blocks"=>config.blocks,"block_min_seconds"=>config.block_seconds))
    for c in cases
        push!(rows,Base.invokelatest(collect_case,c,samples,config.warmup_seconds,config.blocks,config.block_seconds))
        report["blas_threads_after"]=BLAS.get_num_threads()
        # Every completed case has an atomic recoverable record.
        open(output*".tmp","w") do io
            JSON3.pretty(io,report);println(io)
        end
        mv(output*".tmp",output;force=true)
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
