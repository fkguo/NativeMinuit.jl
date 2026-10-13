# SPDX-License-Identifier: LGPL-2.1-or-later
using Test, JSON3
include(joinpath(@__DIR__, "..", "scripts", "run_perf.jl"))
include(joinpath(@__DIR__, "measure_latency.jl"))

@testset "same-directory atomic JSON replacement" begin
    @test Sys.isunix()
    @test Threads.nthreads() >= 2
    for writer in (write_json, write_latency_json)
        @testset "$(nameof(writer))" begin
            mktempdir() do directory
                path=joinpath(directory,"record.json")
                writer(path,Dict("generation"=>0,"rows"=>Int[]))
                ready=Threads.Atomic{Bool}(false)
                done=Threads.Atomic{Bool}(false)
                failures=String[]
                reader=Threads.@spawn begin
                    count=0
                    ready[]=true
                    while !done[]
                        try
                            data=JSON3.read(read(path,String))
                            data.generation == length(data.rows) || error("partial cumulative record")
                            collect(data.rows) == collect(1:data.generation) || error("inconsistent cumulative record")
                            count+=1
                        catch err
                            push!(failures,sprint(showerror,err))
                        end
                    end
                    count
                end
                while !ready[]
                    yield()
                end
                try
                    for generation in 1:200
                        writer(path,Dict("generation"=>generation,"rows"=>collect(1:generation)))
                        yield()
                    end
                finally
                    done[]=true
                end
                reads=fetch(reader)
                @test reads > 0
                @test isempty(failures)
                @test JSON3.read(read(path,String)).generation == 200
                @test !ispath(path * ".writing")

                previous=read(path)
                mkdir(path * ".writing")
                @test_throws SystemError writer(path,Dict("generation"=>201,"rows"=>collect(1:201)))
                @test read(path) == previous
                @test JSON3.read(read(path,String)).generation == 200
                rm(path * ".writing")

                # A failed rename must leave the destination intact and retain
                # the complete temporary record; there is no copy/remove fallback.
                occupied=joinpath(directory,"occupied")
                mkdir(occupied);write(joinpath(occupied,"keep"),"unchanged")
                @test_throws SystemError writer(occupied,Dict("complete"=>true))
                @test read(joinpath(occupied,"keep"),String) == "unchanged"
                @test JSON3.read(read(occupied * ".writing",String)).complete == true
                println(nameof(writer),": ",reads," concurrent complete reads; 200 cumulative replacements")
            end
        end
    end
end
