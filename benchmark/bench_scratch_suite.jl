# SPDX-License-Identifier: LGPL-2.1-or-later
using NativeMinuit, BenchmarkTools

function objective(kind)
    kind == :cheap && return x -> sum(abs2,x)
    data=collect(range(-1.,1.;length=1000))
    return x -> begin
        s=0.0
        for p in x, d in data
            s+=(p-d)^2
        end
        s/length(data)
    end
end
function gradient_fixture(n,kind)
    cf=CostFunction(objective(kind))
    x=fill(0.5,n)
    par=MinimumParameters(x,fill(.1,n),cf(x))
    prev=initial_gradient(par,par.dirin,cf)
    out=FunctionGradient(zeros(n),zeros(n),zeros(n))
    reset_ncalls!(cf)
    (cf=cf,par=par,prev=prev,out=out,work=zeros(n))
end
function gradient_call(s,threaded)
    numerical_gradient!(s.out,s.work,s.par,s.prev,s.cf,Strategy(0);threaded)
end
function gradient_result(s,threaded)
    gradient_call(s,threaded)
    Dict("grad"=>copy(s.out.grad),"g2"=>copy(s.out.g2),"gstep"=>copy(s.out.gstep),
         "calls"=>ncalls(s.cf),"nonfinite"=>nonfinite_calls(s.cf))
end
function snapshot_fit(n,kind,threaded,scratch)
    result=migrad(CostFunction(objective(kind)),fill(.5,n),fill(.1,n);
                  maxfcn=10000,threaded_gradient=threaded,verify_threading=threaded,scratch)
    # The consumer keeps an owned materialized snapshot across workspace reuse.
    NativeMinuit.to_dict(result)
end
const JULIA_PERF_BENCHMARKS=Pair{String,Any}[]
for n in (2,10,40), kind in (:cheap,:data), threaded in (false,true)
    name="gradient_$(n)_$(kind)_$(threaded)"
    push!(JULIA_PERF_BENCHMARKS,name=>@benchmarkable(gradient_call(s,$threaded),
        setup=(s=gradient_fixture($n,$kind)),evals=1))
end
# Allocation/reuse comparison delivers the same owned snapshot in both cases.
for reuse in (false,true)
    scratch=reuse ? NativeMinuit.MigradScratch(10) : nothing
    name=reuse ? "snapshot_fit_reused" : "snapshot_fit_fresh"
    push!(JULIA_PERF_BENCHMARKS,name=>@benchmarkable(snapshot_fit(10,:cheap,false,$scratch),evals=1))
end

function julia_perf_validate()
    outputs=Dict{String,Any}()
    for n in (2,10,40), kind in (:cheap,:data)
        serial=gradient_result(gradient_fixture(n,kind),false)
        threaded=gradient_result(gradient_fixture(n,kind),true)
        isequal(serial,threaded) || error("serial/threaded outputs or counters differ for $n/$kind")
        outputs["gradient_$(n)_$(kind)_false"]=serial
        outputs["gradient_$(n)_$(kind)_true"]=threaded
    end
    s=NativeMinuit.MigradScratch(10)
    fresh=snapshot_fit(10,:cheap,false,nothing)
    reused=snapshot_fit(10,:cheap,false,s)
    fresh["valid"] && reused["valid"] || error("invalid scratch benchmark minimum")
    isfinite(fresh["fval"]) && isfinite(reused["fval"]) || error("nonfinite scratch benchmark minimum")
    isequal(fresh,reused) || error("fresh/reused scratch outputs differ")
    saved=deepcopy(reused)
    snapshot_fit(10,:data,false,s)
    isequal(saved,reused) || error("retained snapshot changed on reuse")
    again=snapshot_fit(10,:cheap,false,s)
    isequal(fresh,again) || error("scratch failed to reset")
    outputs["snapshot_fit_fresh"]=fresh
    outputs["snapshot_fit_reused"]=reused
    outputs
end
