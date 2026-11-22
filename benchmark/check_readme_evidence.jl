# SPDX-License-Identifier: LGPL-2.1-or-later
# Independently recheck recorded numerical comparisons and timing admission.
using JSON3, SHA, Statistics

const README_CASE_DIMENSIONS = Dict("quad_4d"=>4, "rosenbrock_2d"=>2,
    "rosenbrock_10d"=>10, "gauss_ll_2_100"=>2, "gauss_ll_10_1000"=>10)
const README_CASE_ORDER = ["quad_4d", "rosenbrock_2d", "rosenbrock_10d",
    "gauss_ll_2_100", "gauss_ll_10_1000"]
loadjson(path) = JSON3.read(read(path, String), Dict{String,Any})
finite_number(x) = x isa Real && !(x isa Bool) && isfinite(x)
positive_integer(x) = x isa Integer && !(x isa Bool) && x > 0
require_record(condition, message) = condition || throw(ArgumentError(message))

function validate_evidence_protocol(p)
    require_record(p["cases"] == README_CASE_ORDER, "unsupported cases or order")
    for (key, expected) in (("strategy",0), ("tol",0.1), ("maxfcn",10000),
                            ("storage_level",0), ("evals",1), ("warmups",5))
        value = p[key]
        require_record(finite_number(value) && value == expected, "unsupported $key")
    end
    for key in ("rounds", "samples")
        require_record(positive_integer(p[key]), "$key must be a positive integer")
    end
    require_record(length(p["order"]) == p["rounds"] &&
        all(x in ("julia,cpp", "cpp,julia") for x in p["order"]), "invalid process order")
    for key in ("julia", "blas", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        require_record(positive_integer(p["threads"][key]) && p["threads"][key] == 1,
            "unsupported thread contract")
    end
    timing = p["timing_reliability"]
    for key in ("fixed_warmup_seconds", "block_min_seconds")
        require_record(finite_number(timing[key]) && timing[key] > 0,
            "$key must be finite and positive")
    end
    require_record(positive_integer(timing["blocks"]), "blocks must be a positive integer")
    for key in ("maximum_language_block_spread_fraction", "maximum_paired_ratio_spread_fraction")
        require_record(finite_number(timing[key]) && timing[key] >= 0,
            "$key must be finite and nonnegative")
    end
    require_record(timing["all_blocks_retained"] === true &&
        positive_integer(timing["initial_samples_retained"]) &&
        timing["initial_samples_retained"] == p["samples"], "sample retention mismatch")
    require_record(Set(keys(p["strict_tolerances"])) ==
        Set(("params", "fval", "edm", "errors", "covariance")), "missing strict tolerance")
    for tolerance in values(p["strict_tolerances"])
        validate_tolerance(tolerance)
    end
    for (name, tolerances) in p["rosenbrock_diagnostic_tolerances"]
        require_record(name in ("rosenbrock_2d", "rosenbrock_10d"), "unsupported diagnostic case")
        for (key, tolerance) in tolerances
            if key == "nfcn_slack"
                require_record(tolerance isa Integer && !(tolerance isa Bool) && tolerance >= 0,
                    "invalid callback slack")
            else
                require_record(key in ("params", "fval", "edm", "errors", "covariance"),
                    "unsupported diagnostic field")
                validate_tolerance(tolerance)
            end
        end
    end
    timing
end

function validate_tolerance(t)
    for key in ("rtol", "atol")
        require_record(finite_number(t[key]) && t[key] >= 0, "invalid numerical tolerance")
    end
end

# Scalar comparisons reproduce the frozen max(atol, rtol*max(abs(a),abs(b)))
# budget; no vector norm, summed error budget, or diagnostic tolerance replaces it.
function approx_field(a, b, tolerance)
    av = a isa AbstractVector ? a : [a]
    bv = b isa AbstractVector ? b : [b]
    length(av) == length(bv) && !isempty(av) &&
        all(finite_number(x) && finite_number(y) &&
            isapprox(x,y; rtol=tolerance["rtol"], atol=tolerance["atol"])
            for (x,y) in zip(av,bv))
end

function validate_timing_block(block, seconds)
    ts = block["sample_times_ns"]
    require_record(ts isa AbstractVector && !isempty(ts) &&
        all(x -> finite_number(x) && x > 0, ts), "invalid block times")
    require_record(positive_integer(block["validated_outputs"]) &&
        length(ts) == block["validated_outputs"], "missing block output checks")
    require_record(block["stable_outputs"] === true &&
        finite_number(block["elapsed_seconds"]) && block["elapsed_seconds"] >= seconds,
        "block timing/output failure")
end

function validate_result(r, n)
    for (key, dimension) in (("params",n), ("errors",n), ("covariance",n*n))
        require_record(r[key] isa AbstractVector && length(r[key]) == dimension &&
            all(finite_number,r[key]), "invalid $key shape or value")
    end
    require_record(finite_number(r["fval"]) && finite_number(r["edm"]), "invalid objective or EDM")
    require_record(r["is_valid"] === true && r["has_covariance"] === true &&
        r["above_max_edm"] === false && r["reached_call_limit"] === false,
        "invalid minimum/covariance/termination")
    require_record(positive_integer(r["nfcn"]) && positive_integer(r["actual_calls"]) &&
        r["nfcn"] == r["actual_calls"], "callback count mismatch")
    diagonal = [r["covariance"][(i-1)*n+i] for i in 1:n]
    require_record(all(x -> x > 0, diagonal), "nonpositive covariance diagonal")
    require_record(all(isapprox.(r["errors"],sqrt.(diagonal);rtol=2eps(Float64),atol=0)),
        "error/covariance mismatch")
end

function check_evidence(dir; summary_path=joinpath(dir,"summary.json"), write_output=true)
    p = loadjson(joinpath(dir,"protocol.json"))
    timing = validate_evidence_protocol(p)
    before = loadjson(joinpath(dir,"source-hashes-before.json"))
    after = loadjson(joinpath(dir,"source-hashes-after.json"))
    require_record(!isempty(before) && before == after, "source hashes absent or changed")
    summary = loadjson(summary_path)
    require_record(Set(keys(summary["cases"])) == Set(p["cases"]), "summary cases mismatch")
    checks = Any[]
    blocks = Dict(name=>Dict("julia"=>Float64[],"cpp"=>Float64[]) for name in p["cases"])
    ratios = Dict(name=>Float64[] for name in p["cases"])
    process_results = Dict(name=>Dict("julia"=>Any[],"cpp"=>Any[]) for name in p["cases"])
    for pair in 1:p["rounds"]
        jj = loadjson(joinpath(dir,"pair-$pair-julia.json"))
        cc = loadjson(joinpath(dir,"pair-$pair-cpp.json"))
        comp = loadjson(joinpath(dir,"pair-$pair-comparison.json"))
        require_record(jj["language"] == "julia" && cc["language"] == "cpp", "language mismatch")
        for key in ("julia_threads", "blas_threads_before", "blas_threads_after")
            require_record(positive_integer(jj[key]) && jj[key] == 1, "thread mismatch")
        end
        require_record(jj["fixture_sha256"] == p["fixture_sha256"], "fixture mismatch")
        for process in (jj,cc)
            require_record(process["warmups"] == p["warmups"] && process["evals"] == 1,
                "warmup/eval mismatch")
            for key in ("fixed_warmup_seconds", "blocks", "block_min_seconds")
                require_record(finite_number(process["timing_reliability"][key]) &&
                    process["timing_reliability"][key] == timing[key], "requested timing mismatch")
            end
            require_record(length(process["rows"]) == length(p["cases"]) &&
                Set(row["name"] for row in process["rows"]) == Set(p["cases"]), "missing or duplicate rows")
        end
        for name in p["cases"]
            a = only(filter(x -> x["name"] == name,jj["rows"]))
            b = only(filter(x -> x["name"] == name,cc["rows"]))
            for (language,row) in (("julia",a),("cpp",b))
                r = row["result"]
                validate_result(r,README_CASE_DIMENSIONS[name])
                push!(process_results[name][language],r)
                require_record(positive_integer(row["samples"]) &&
                    row["samples"] == length(row["sample_times_ns"]) == p["samples"], "sample mismatch")
                require_record(all(x -> finite_number(x) && x > 0,row["sample_times_ns"]), "invalid sample times")
                require_record(row["evals"] == 1 && row["stable_outputs"] === true, "eval/output instability")
                require_record(length(row["timing_blocks"]) == timing["blocks"], "missing timing block")
                validate_timing_block(row["continuous_warmup"],timing["fixed_warmup_seconds"])
                foreach(block -> validate_timing_block(block,timing["block_min_seconds"]),row["timing_blocks"])
                append!(blocks[name][language],[median(v["sample_times_ns"]) for v in row["timing_blocks"]])
            end
            require_record(a["explicit_warmups"] == 4 && a["benchmarktools_warmups"] == 1,
                "Julia warmup mismatch")
            require_record(a["validated_outputs"] == p["samples"]+1, "missing sampled output check")
            jm = median([median(v["sample_times_ns"]) for v in a["timing_blocks"]])
            cm = median([median(v["sample_times_ns"]) for v in b["timing_blocks"]])
            require_record(jm == comp[name]["julia_median_ns"] && cm == comp[name]["cpp_median_ns"],
                "block median disagreement")
            push!(ratios[name],jm/cm)
            strict = all(approx_field(a["result"][k],b["result"][k],t) for (k,t) in p["strict_tolerances"])
            require_record(strict === comp[name]["strict_parity"], "strict comparator disagreement")
            diagnostic = nothing
            if haskey(p["rosenbrock_diagnostic_tolerances"],name)
                tolerance = p["rosenbrock_diagnostic_tolerances"][name]
                diagnostic = all(approx_field(a["result"][k],b["result"][k],t)
                    for (k,t) in tolerance if k != "nfcn_slack") &&
                    abs(a["result"]["nfcn"]-b["result"]["nfcn"]) <= tolerance["nfcn_slack"]
                require_record(diagnostic === comp[name]["diagnostic_pass"], "diagnostic comparator disagreement")
            end
            push!(checks,Dict("pair"=>pair,"name"=>name,"strict_parity"=>strict,
                "diagnostic_pass"=>diagnostic,"positive_diagonals_and_errors"=>true))
        end
    end
    timing_checks = Dict{String,Any}()
    admissions = Bool[]
    for name in p["cases"]
        language_spread = Dict(lang=>maximum(v)/minimum(v)-1 for (lang,v) in blocks[name])
        ratio_spread = maximum(ratios[name])/minimum(ratios[name])-1
        passed = all(x <= timing["maximum_language_block_spread_fraction"] for x in values(language_spread)) &&
            ratio_spread <= timing["maximum_paired_ratio_spread_fraction"]
        record = summary["cases"][name]
        identities = Dict(lang=>all(isequal(first(results)),results) for (lang,results) in process_results[name])
        require_record(all(record[lang*"_identical_across_processes"] === identities[lang]
            for lang in ("julia","cpp")), "process output identity disagreement")
        numeric = all(x["strict_parity"] for x in checks if x["name"] == name) && all(values(identities))
        require_record(numeric === record["strict_numeric_admitted"], "numeric admission disagreement")
        admitted = numeric && passed
        require_record(admitted === record["benchmark_admitted"], "overall admission disagreement")
        require_record(passed === record["timing_stability_admitted"], "timing admission disagreement")
        require_record(all(isapprox(language_spread[lang],record["language_block_spread_fraction"][lang];
            rtol=2eps(),atol=0) for lang in ("julia","cpp")), "spread disagreement")
        require_record(isapprox(ratio_spread,record["paired_ratio_spread_fraction"];rtol=2eps(),atol=0),
            "ratio spread disagreement")
        push!(admissions,admitted)
        timing_checks[name] = Dict("language_spread"=>language_spread,"ratio_spread"=>ratio_spread,"admitted"=>passed)
    end
    require_record(all(admissions) === summary["all_cases_benchmark_admitted"], "overall admission flag disagreement")
    output = Dict("all_cases_benchmark_admitted"=>all(admissions),
        "scope"=>"Record consistency only; within-process stability uses collector checks. Admission may fail.",
        "timing_checks"=>timing_checks,"schema_version"=>1,"validation_status"=>"PASS",
        "checks"=>checks,"checker_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    if write_output
        path = joinpath(dir,"julia-evidence-check.json")
        open(path*".tmp","w") do io
            JSON3.pretty(io,output);println(io)
        end
        mv(path*".tmp",path;force=true)
    end
    output
end

function main(args=ARGS)
    length(args) in (1,2) || error("usage: check_readme_evidence.jl RESULT_DIRECTORY [SUMMARY_JSON]")
    check_evidence(args[1];summary_path=length(args)==2 ? args[2] : joinpath(args[1],"summary.json"))
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
