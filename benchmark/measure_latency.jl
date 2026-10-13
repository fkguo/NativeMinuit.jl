# SPDX-License-Identifier: LGPL-2.1-or-later
# Fresh-process import/first/second-call measurements. An existing precompile
# cache is required; cache construction and dependency installation are excluded.
using JSON3, SHA, Statistics, LinearAlgebra

function write_latency_json(path, data)
    Sys.isunix() || error("atomic JSON publication requires a POSIX host")
    temporary = path * ".writing"
    open(io -> JSON3.write(io, data), temporary, "w")
    # Same-directory POSIX replacement preserves the old record on failure;
    # unlike version-dependent force=true helpers, it never unlinks first.
    status = ccall(:rename, Cint, (Cstring, Cstring), temporary, path)
    Base.systemerror("atomic JSON rename", status != 0)
end

function latency_operation(kind)
    f(x) = (x[1]-1.0)^2 + (x[2]-2.0)^2
    if kind == "migrad"
        return () -> NativeMinuit.migrad(NativeMinuit.CostFunction(f), [0.,0.], [.1,.1]; maxfcn=10000)
    elseif kind == "minuit"
        return () -> begin
            m=NativeMinuit.Minuit(f,[0.,0.]; errors=[.1,.1])
            NativeMinuit.migrad!(m; maxfcn=10000, iterate=1)
            m.fmin
        end
    elseif kind == "bounded"
        return () -> begin
            p=NativeMinuit.Parameters([NativeMinuit.MinuitParameter("a",0.,.1;lower=-3.,upper=3.),
                                       NativeMinuit.MinuitParameter("b",0.,.1;fixed=true)])
            NativeMinuit.migrad(NativeMinuit.CostFunction(f),p;maxfcn=10000)
        end
    elseif kind == "least_squares"
        return () -> begin
            c=NativeMinuit.LeastSquares([0.,1.,2.,3.],[1.,3.,5.,7.],ones(4),(x,p)->p[1]*x+p[2])
            m=NativeMinuit.Minuit(c,[1.,0.];errors=[.1,.1])
            NativeMinuit.migrad!(m;maxfcn=10000,iterate=1)
            m.fmin
        end
    end
    error("unknown latency case: $kind")
end

function latency_child(kind,path)
    BLAS.set_num_threads(1)
    load_seconds=@elapsed @eval using NativeMinuit
    # Loading a package creates methods in a newer world than this caller.
    # Enter that world once, outside the measured operation calls.
    Base.invokelatest(latency_after_load,kind,path,load_seconds)
end

function latency_after_load(kind,path,load_seconds)
    BLAS.set_num_threads(1)
    root=realpath(joinpath(@__DIR__,".."))
    realpath(pathof(NativeMinuit)) == joinpath(root,"src","NativeMinuit.jl") || error("wrong NativeMinuit source")
    op=latency_operation(kind)
    first=@timed op()
    first_result=NativeMinuit.to_dict(first.value)
    second=@timed op()
    BLAS.get_num_threads() == 1 || error("latency BLAS thread count differs from requested 1")
    second_result=NativeMinuit.to_dict(second.value)
    isequal(first_result,second_result) || error("first/second result or counters differ")
    NativeMinuit.is_valid(first.value) || error("invalid latency result")
    record=Dict("case"=>kind,"pid"=>getpid(),"load_seconds"=>load_seconds,
      "first_seconds"=>first.time,"second_seconds"=>second.time,
      "first_bytes"=>first.bytes,"second_bytes"=>second.bytes,
      "result"=>first_result,"julia_version"=>string(VERSION),
      "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
      "omp_num_threads"=>get(ENV,"OMP_NUM_THREADS",nothing),
      "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
      "entry_sha256"=>bytes2hex(sha256(read(pathof(NativeMinuit)))),
      "loaded_entry"=>pathof(NativeMinuit))
    write_latency_json(path,record)
end

function latency_main(out)
    mkpath(out)
    isfile(joinpath(out,"latency.json")) && error("latency output exists")
    rows=Any[]
    for kind in ("migrad","minuit","bounded","least_squares"), rep in 1:5
        file=joinpath(out,"$kind-$rep.json")
        ispath(file) && error("sample output exists: $file")
        log=joinpath(out,"$kind-$rep.log")
        code=abspath(@__FILE__)
        command=`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(dirname(Base.active_project())) $code --child $kind $file`
        command=addenv(command,"OPENBLAS_NUM_THREADS"=>"1","OMP_NUM_THREADS"=>"1")
        started=time_ns()
        process=open(log,"w") do io
            run(pipeline(command;stdout=io,stderr=io);wait=false)
        end
        if timedwait(()->process_exited(process),120.;pollint=.05)==:timed_out
            kill(process,Base.SIGKILL);wait(process)
            error("latency child timed out: $kind/$rep; completed files retained")
        end
        wait(process); success(process) || error("latency child failed: $kind/$rep; inspect $log")
        row=JSON3.read(read(file,String),Dict{String,Any})
        row["child_total_seconds"]=(time_ns()-started)*1e-9
        row["repeat"]=rep
        prior=filter(r->r["case"]==kind,rows)
        isempty(prior) || isequal(prior[1]["result"],row["result"]) || error("fresh-process results differ")
        push!(rows,row)
        write_latency_json(joinpath(out,"latency.json"),Dict("status"=>"partial","rows"=>rows))
    end
    summary=Dict(kind=>Dict(key=>Dict("median"=>median([r[key] for r in rows if r["case"]==kind]),
              "minimum"=>minimum([r[key] for r in rows if r["case"]==kind]),
              "maximum"=>maximum([r[key] for r in rows if r["case"]==kind]))
              for key in ("load_seconds","first_seconds","second_seconds"))
              for kind in ("migrad","minuit","bounded","least_squares"))
    write_latency_json(joinpath(out,"latency.json"),Dict("status"=>"measured","claim_verified"=>false,
      "cache_policy"=>"fresh process, existing precompile cache; installation and cache rebuild excluded",
      "statistical_analysis"=>"descriptive median and full range only; no comparative speed claim",
      "rows"=>rows,"summary"=>summary))
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    if length(ARGS)==3 && ARGS[1]=="--child"
        latency_child(ARGS[2],ARGS[3])
    elseif length(ARGS)==1
        latency_main(abspath(ARGS[1]))
    else
        error("usage: julia --project=benchmark benchmark/measure_latency.jl OUTPUT_DIRECTORY")
    end
end
