# SPDX-License-Identifier: LGPL-2.1-or-later
# Run only via compare_readme.py, in the pinned benchmark environment.
using NativeMinuit, BenchmarkTools, JSON3, LinearAlgebra, SHA, Pkg
include(joinpath(@__DIR__, "bench_migrad_suite.jl"))

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

function collect_case(c, samples)
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
    BLAS.get_num_threads() == 1 || error("BLAS thread count changed during timing")
    Dict("name"=>c.name, "samples"=>length(trial.times), "evals"=>trial.params.evals,
         "sample_times_ns"=>trial.times, "sample_gc_times_ns"=>trial.gctimes,
         "memory_bytes"=>trial.memory, "allocations"=>trial.allocs,
         "stable_outputs"=>all(isequal(first),received),
         "validated_outputs"=>length(received), "explicit_warmups"=>4, "benchmarktools_warmups"=>1,
         "result"=>first, "different_outputs"=>filter(x->!isequal(first,x),received))
end

function main()
    length(ARGS)==2 || error("usage: readme_julia_bench.jl OUTPUT_JSON SAMPLES")
    output, raw_samples = ARGS
    samples=parse(Int,raw_samples);samples>0 || error("samples must be positive")
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
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"Manifest.toml")))),
        "package_path"=>pathof(NativeMinuit),"warmups"=>5,"evals"=>1,"rows"=>rows)
    for c in cases
        push!(rows,collect_case(c,samples))
        report["blas_threads_after"]=BLAS.get_num_threads()
        # Every completed case has an atomic recoverable record.
        open(output*".tmp","w") do io
            JSON3.pretty(io,report);println(io)
        end
        mv(output*".tmp",output;force=true)
    end
end
main()
