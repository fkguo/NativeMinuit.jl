# SPDX-License-Identifier: LGPL-2.1-or-later

using JSON

@testset "serialize.jl — to_dict / minimum_summary_from_dict (Phase 2.5)" begin

    @testset "BoundedFunctionMinimum round-trip" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        params = Parameters([
            MinuitParameter("x", 0.0, 0.1; lower = -5.0, upper = 5.0),
            MinuitParameter("y", 0.0, 0.1),
        ])
        m = migrad(cf, params)
        @test is_valid(m)

        d = NativeMinuit.to_dict(m)
        @test d["type"] == "BoundedFunctionMinimum"
        @test d["valid"] == true
        @test d["nfcn"] > 0
        @test length(d["ext_values"]) == 2
        @test length(d["ext_errors"]) == 2
        @test d["param_names"] == ["x", "y"]
        @test d["param_lower"][1] == -5.0
        @test d["param_lower"][2] === nothing
        @test d["param_upper"][1] == 5.0

        # JSON round-trip
        json_str = JSON.json(d)
        parsed = JSON.parse(json_str)
        summary = NativeMinuit.minimum_summary_from_dict(parsed)
        @test summary.fval ≈ d["fval"]
        @test summary.nfcn == d["nfcn"]
        @test summary.values ≈ d["ext_values"]
        @test summary.type == "BoundedFunctionMinimum"
    end

    @testset "FunctionMinimum round-trip" begin
        cf = CostFunction(x -> sum(abs2, x))
        m = migrad(cf, [1.0, 2.0], [0.1, 0.1])
        @test m.is_valid

        d = NativeMinuit.to_dict(m)
        @test d["type"] == "FunctionMinimum"
        @test d["valid"] == true
        @test length(d["values"]) == 2

        json_str = JSON.json(d)
        parsed = JSON.parse(json_str)
        summary = NativeMinuit.minimum_summary_from_dict(parsed)
        @test summary.fval ≈ d["fval"]
        @test length(summary.values) == 2
    end

    @testset "MinosError round-trip" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        e = minos(fmin, cf, 1)
        @test NativeMinuit.is_valid(e)

        d = NativeMinuit.to_dict(e)
        @test d["type"] == "MinosError"
        @test d["par_idx"] == 1
        @test d["upper"] ≈ e.upper
        @test d["lower"] ≈ e.lower

        # Round-trip through JSON
        json_str = JSON.json(d)
        parsed = JSON.parse(json_str)
        e2 = NativeMinuit.minos_error_from_dict(parsed)
        @test e2.par_idx == e.par_idx
        @test e2.upper ≈ e.upper
        @test e2.lower ≈ e.lower
        @test NativeMinuit.is_valid(e2) == NativeMinuit.is_valid(e)
    end

    @testset "ContoursError round-trip" begin
        cf = CostFunction(x -> (x[1] - 1.0)^2 + (x[2] - 2.0)^2)
        fmin = migrad(cf, [0.0, 0.0], [0.1, 0.1])
        c = contour_ellipse(fmin, cf, 1, 2; npoints = 8)
        @test c.valid

        d = NativeMinuit.to_dict(c)
        @test d["type"] == "ContoursError"
        @test d["par_x"] == 1
        @test d["par_y"] == 2
        @test length(d["points"]) == 8
        @test d["minos_x"]["type"] == "MinosError"

        # JSON round-trip
        json_str = JSON.json(d)
        parsed = JSON.parse(json_str)
        @test parsed["par_x"] == 1
        @test length(parsed["points"]) == 8
    end
end
