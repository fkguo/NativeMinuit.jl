# SPDX-License-Identifier: LGPL-2.1-or-later
# Historical cross-language measurements have incompatible input and timing
# contracts. Preserve every supplied row; none is a current performance gate.
using JSON3
const LEGACY_CASES = ["rosenbrock_2d", "rosenbrock_10d", "quad_4d", "gauss_ll_2_100", "gauss_ll_10_1000"]

function read_comparison_record(path)
    path === nothing && return Dict("status"=>"missing", "path"=>nothing, "raw"=>nothing)
    isfile(path) || return Dict("status"=>"missing", "path"=>path, "raw"=>nothing)
    raw = read(path, String)
    try
        value = JSON3.read(raw)
        return Dict("status"=>"unverified", "path"=>abspath(path), "raw"=>value)
    catch err
        return Dict("status"=>"invalid", "path"=>abspath(path), "raw_text"=>raw,
                    "error"=>sprint(showerror, err), "raw"=>nothing)
    end
end

function comparison_report(paths)
    sources = Dict(name=>read_comparison_record(path) for (name,path) in paths)
    rows = Dict{String,Any}(name=>Dict("status"=>"UNVERIFIED", "sources"=>Dict{String,Any}()) for name in LEGACY_CASES)
    for (language, record) in sources
        raw = record["raw"]
        # Both legacy arrays and current Julia baseline rows are retained.
        entries = raw isa AbstractVector ? raw :
                  raw isa AbstractDict && haskey(raw, :rows) ? raw[:rows] : Any[]
        if entries isa AbstractVector
            for entry in entries
                if haskey(entry, :name)
                    name = String(entry[:name])
                    haskey(rows, name) || (rows[name]=Dict("status"=>"UNVERIFIED", "sources"=>Dict{String,Any}()))
                    bucket = get!(rows[name]["sources"], language, Any[])
                    push!(bucket, entry) # preserve duplicates and failed numeric rows
                end
            end
        end
    end
    for row in values(rows), language in keys(paths)
        haskey(row["sources"], language) || (row["sources"][language] = nothing)
    end
    invalid = any(r["status"] == "invalid" for r in values(sources))
    Dict("verdict"=>invalid ? "FAIL" : "UNVERIFIED", "exit_code"=>invalid ? 1 : 2,
         "claim_verified"=>false, "sources"=>sources, "cases"=>rows,
         "reason"=>"No matched cross-language fixture, operation, result-validity and environment contract exists. Historical medians cannot establish a current ratio or PASS.")
end

function comparison_main(args; languages=["julia","cpp"])
    paths = Dict{String,Union{Nothing,String}}(name=>nothing for name in languages)
    # Default only reads existing evidence. It never rebuilds/runs a different
    # external executable against historical timings or overwrites old records.
    historical = joinpath(@__DIR__, ".julia-perf", "runs", "latest", "benchmarks.json")
    isfile(historical) && (paths["julia"] = historical)
    i=1
    while i <= length(args)
        option=args[i]
        startswith(option,"--") && endswith(option,"-record") || error("expected --<language>-record <JSON file>")
        language=option[3:end-7]
        haskey(paths,language) || error("unsupported record language: $language")
        i < length(args) || error("missing record path")
        paths[language]=args[i+1]; i+=2
    end
    report=comparison_report(paths)
    JSON3.write(stdout, report); println()
    report["exit_code"]
end
