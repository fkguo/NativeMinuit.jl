# SPDX-License-Identifier: LGPL-2.1-or-later
# Recheck a completed comparison using Julia's isapprox and positive diagonals.
using JSON3, SHA
loadjson(path)=JSON3.read(read(path,String),Dict{String,Any})
function approx_field(a,b,t)
    av=a isa AbstractVector ? a : [a]
    bv=b isa AbstractVector ? b : [b]
    length(av)==length(bv) && all(isapprox(x,y;rtol=t["rtol"],atol=t["atol"]) for (x,y) in zip(av,bv))
end
function main()
    length(ARGS)==1 || error("usage: check_readme_evidence.jl RESULT_DIRECTORY")
    dir=ARGS[1];p=loadjson(joinpath(dir,"protocol.json"));checks=Any[]
    before=loadjson(joinpath(dir,"source-hashes-before.json"));after=loadjson(joinpath(dir,"source-hashes-after.json"))
    before==after || error("source hashes changed")
    for pair in 1:p["rounds"]
        jj=loadjson(joinpath(dir,"pair-$pair-julia.json"));cc=loadjson(joinpath(dir,"pair-$pair-cpp.json"))
        comp=loadjson(joinpath(dir,"pair-$pair-comparison.json"))
        jj["julia_threads"]==jj["blas_threads_before"]==jj["blas_threads_after"]==1 || error("thread mismatch")
        jj["fixture_sha256"]==p["fixture_sha256"] || error("fixture mismatch")
        length(jj["rows"])==length(cc["rows"])==length(p["cases"]) || error("missing or extra rows")
        for name in p["cases"]
            a=only(filter(x->x["name"]==name,jj["rows"]));b=only(filter(x->x["name"]==name,cc["rows"]))
            for row in (a,b)
                r=row["result"];n=length(r["params"])
                row["samples"]==length(row["sample_times_ns"])==p["samples"] || error("sample mismatch")
                row["evals"]==1 && row["stable_outputs"] || error("evals/output instability")
                r["is_valid"] && r["has_covariance"] || error("invalid minimum/covariance")
                r["nfcn"]==r["actual_calls"] || error("callback count mismatch")
                all(isfinite,r["covariance"]) || error("nonfinite covariance")
                diag=[r["covariance"][(i-1)*n+i] for i in 1:n]
                all(x->isfinite(x)&&x>0,diag) || error("nonpositive covariance diagonal")
                all(isapprox.(r["errors"],sqrt.(diag);rtol=2eps(Float64),atol=0)) || error("error/covariance mismatch")
            end
            a["explicit_warmups"]+a["benchmarktools_warmups"]==jj["warmups"]==cc["warmups"]==5 || error("warmup mismatch")
            a["validated_outputs"]==p["samples"]+1 || error("missing sampled output check")
            strict=all(approx_field(a["result"][k],b["result"][k],t) for (k,t) in p["strict_tolerances"])
            strict==comp[name]["strict_parity"] || error("strict comparator disagreement")
            diagnostic=nothing
            if haskey(p["rosenbrock_diagnostic_tolerances"],name)
                tol=p["rosenbrock_diagnostic_tolerances"][name]
                diagnostic=all(approx_field(a["result"][k],b["result"][k],t) for (k,t) in tol if k!="nfcn_slack") && abs(a["result"]["nfcn"]-b["result"]["nfcn"])<=tol["nfcn_slack"]
                diagnostic==comp[name]["diagnostic_pass"] || error("diagnostic comparator disagreement")
            end
            push!(checks,Dict("pair"=>pair,"name"=>name,"strict_parity"=>strict,"diagnostic_pass"=>diagnostic,"positive_diagonals_and_errors"=>true))
        end
    end
    output=Dict("schema_version"=>1,"validation_status"=>"PASS","checks"=>checks,"checker_sha256"=>bytes2hex(sha256(read(@__FILE__))),"scope"=>"Cross-check the recorded comparison with Julia isapprox; validate covariance diagonals, error materialization, samples, warmups and threads. This does not establish independent numerical-method or model-family validation.")
    path=joinpath(dir,"julia-evidence-check.json");open(path*".tmp","w") do io;JSON3.pretty(io,output);println(io);end;mv(path*".tmp",path;force=true)
end
main()
