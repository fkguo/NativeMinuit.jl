# SPDX-License-Identifier: LGPL-2.1-or-later
# Exit codes: 0 executed/pass; 1 executed/failed; 2 unavailable.
using TOML

function main()
    status_path = get(ENV, "NATIVEMINUIT_JET_STATUS", joinpath(@__DIR__, "status.toml"))
    result = Dict{String,Any}("status" => "unavailable", "executed" => false,
                              "julia_version" => string(VERSION), "checks" => 0)
    code = 2
    try
        VERSION == v"1.11.7" || error("Use the pinned Julia 1.11.7 diagnostic runtime")
        @eval using NativeMinuit, JET, Test
        result["jet_version"] = string(Base.pkgversion(JET))
        Base.pkgversion(JET) == v"0.9.20" || error("Use the pinned JET 0.9.20 environment")
        # Probe compiler integration before marking diagnostics executed.
        probe = Base.invokelatest(JET.report_opt, sin, (Float64,))
        Base.invokelatest(JET.get_reports, probe)
        # Execute the shared narrow tests; failed negative controls also fail.
        result["executed"] = true
        result["status"] = "failed"
        code = 1
        tests = include(joinpath(@__DIR__, "..", "..", "test", "test_jet_optanalysis.jl"))
        counts = Base.invokelatest(Test.get_test_counts, tests)
        result["checks"] = counts.passes + counts.cumulative_passes
        result["checks"] == 3 || error("Expected exactly three executed checks")
        result["passed"] = result["checks"]
        result["failed"] = 0
        result["errored"] = 0
        result["skipped"] = 0
        result["status"] = "pass"
        code = 0
    catch err
        underlying = err
        while underlying isa LoadError
            underlying = underlying.error
        end
        if isdefined(Main, :Test) && underlying isa Test.TestSetException
            result["checks"] = underlying.pass + underlying.fail + underlying.error + underlying.broken
            result["passed"] = underlying.pass
            result["failed"] = underlying.fail
            result["errored"] = underlying.error
            result["skipped"] = underlying.broken
        end
        result["reason"] = sprint(showerror, err)
        showerror(stderr, err, catch_backtrace())
        println(stderr)
    finally
        mkpath(dirname(abspath(status_path)))
        open(status_path, "w") do io
            TOML.print(io, result)
        end
        println("JET diagnostic status: ", result["status"], " (executed=", result["executed"], ")")
    end
    return code
end

exit(main())
