# SPDX-License-Identifier: LGPL-2.1-or-later
# Frozen budgets: coordinates rtol=1e-10/atol=1e-12; fval
# rtol=1e-10/atol=1e-18; shared EDM/covariance rtol=1e-6/atol=1e-12.
# Status, pass count and independently observed objective calls are exact.
@testset "public retry policy with pinned C++ Minuit2 building blocks" begin
    rugged(x) = .3*(x[1]^2+x[2]^2) +
        1e-3*(abs(sin(1e3*x[1]))+abs(sin(1e3*x[2]))) -
        8exp(-((x[1]-.8)^2+(x[2]-.8)^2)/.3)
    smooth(x) = (x[1]-1)^2+(x[2]-2)^2+10
    singular(x) = x[1]^2
    rows = split.(readlines(joinpath(@__DIR__, "reference_data", "retry_reference.txt")))
    selector = only(filter(r -> r[1] == "selector", rows))
    refs = Dict((r[1],parse(Int,r[2]),parse(Int,r[3])) => parse.(Float64,r[4:end])
                for r in rows if r[1] != "selector")
    results = Dict{Tuple{String,Int,Int},Any}()
    for (name, it, us) in sort!(collect(keys(refs)))
        @testset "$name iterate=$it simplex=$us" begin
            f, x, e = name == "rugged" ? (rugged,[.8,.8],[.05,.05]) :
                name == "selector_invalid" ? (singular,[.1,.1],[.05,.05]) :
                (smooth,[0.,0.],[.1,.1])
            count = Ref(0)
            m = Minuit(x -> (count[] += 1; f(x)), x;
                       errors=e, threaded_gradient=false)
            migrad!(m; iterate=it, use_simplex=(us==1), tol=1e-6, maxfcn=4000)
            results[(name,it,us)] = m.fmin
            ref = refs[(name,it,us)]
            if get(ENV, "NATIVEMINUIT_RETRY_TRACE", "0") == "1"
                println("retry_raw ", name, " iterate=", it, " simplex=", us,
                        " valid=", m.valid, " limit=", m.fmin.internal.reached_call_limit,
                        " passes=", m.n_passes, " calls=", m.nfcn,
                        " values=", collect(m.values), " fval=", m.fval,
                        " edm=", m.edm, " covariance=", NativeMinuit.covariance(m.fmin))
                flush(stdout)
            end
            @test m.valid == (ref[1] == 1)
            @test m.fmin.internal.reached_call_limit == (ref[2] == 1)
            @test m.n_passes == ref[3]
            @test m.nfcn == ref[4]
            @test m.nfcn == count[]
            @test m.values[1] ≈ ref[6] rtol=1e-10 atol=1e-12
            @test m.values[2] ≈ ref[7] rtol=1e-10 atol=1e-12
            @test m.fval ≈ ref[8] rtol=1e-10 atol=1e-18
            @test f(collect(m.values)) ≈ m.fval rtol=0 atol=1e-12
            if ref[10] == 1
                # Pinned C++ seed-only constructor stores fval in EDM. Keep
                # the representation difference explicit, never enlarge a budget.
                @test !m.valid
                @test ref[9] == ref[8]
                @test m.edm != m.fval
            else
                @test m.edm ≈ ref[9] rtol=1e-6 atol=1e-12
            end
            if ref[11] == 1
                cov = NativeMinuit.covariance(m.fmin)
                @test cov[1,1] ≈ ref[12] rtol=1e-6 atol=1e-12
                @test cov[1,2] ≈ ref[13] rtol=1e-6 atol=1e-12
                @test cov[2,2] ≈ ref[14] rtol=1e-6 atol=1e-12
            end
        end
    end
    # Actual smooth minima with opposing cost/validity test the selection rule;
    # cost-only selection fails this control in either argument order.
    good = results[("selector_valid",1,0)]
    bad = results[("selector_invalid",1,0)]
    @test parse.(Int,selector[2:end]) == [1,0,1,0,1]
    @test NativeMinuit.is_valid(good)
    @test !NativeMinuit.is_valid(bad)
    @test NativeMinuit.fval(good) > NativeMinuit.fval(bad)
    @test NativeMinuit._retry_select_better(good,bad) === good
    @test NativeMinuit._retry_select_better(bad,good) === good
end
