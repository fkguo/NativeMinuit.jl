# SPDX-License-Identifier: LGPL-2.1-or-later
#!/usr/bin/env julia
using Dates, SHA, TOML, Statistics, LinearAlgebra
using BenchmarkTools, JSON3

const USAGE = """
run_perf.jl --config <path> [--out-dir <path>] [--save-baseline]
  --mode standalone|ecosystem|auto
  --artifact-root <path> --tag <tag> [--agent-id <id>]
  --save-baseline writes baseline.json inside the output directory.
Exit: 0 declared thresholds pass OR baseline created; 1 failure;
      2 comparison unverified; 3 configuration error.
Baseline creation is not a performance PASS. Claim verification is external.
"""

function nested(cfg, keys, default=nothing)
    cur = cfg
    for key in keys
        cur isa AbstractDict && haskey(cur, key) || return default
        cur = cur[key]
    end
    return cur
end
sha256_file(path) = bytes2hex(sha256(read(path)))
canonical(x::AbstractDict) = [[string(key), canonical(x[key])] for key in sort!(collect(keys(x)); by=string)]
canonical(x::AbstractVector) = canonical.(x)
canonical(x) = x
hash_json(x) = bytes2hex(sha256(JSON3.write(canonical(x))))
utcnow() = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
resolve_path(base, value) = value === nothing || isempty(strip(string(value))) ? nothing : abspath(base, string(value))

function parse_args(args)
    opts = Dict{String,Any}("save_baseline" => false, "help" => false)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--save-baseline", "--help")
            opts[replace(arg[3:end], '-' => '_')] = true
        elseif arg in ("--config", "--out-dir", "--mode", "--artifact-root", "--tag", "--agent-id")
            i < length(args) || error("Missing value for $arg")
            i += 1
            opts[replace(arg[3:end], '-' => '_')] = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    opts["help"] || haskey(opts, "config") || error("--config is required")
    return opts
end

function positive_number(value, label; integer=false)
    value isa Real && !(value isa Bool) && isfinite(value) && value > 0 || error("$label must be positive and finite")
    integer && !(value isa Integer) && error("$label must be an integer")
    return integer ? Int(value) : Float64(value)
end

function protocol_config(cfg)
    policy = nested(cfg, ["benchmark", "evals", "policy"], "fixed")
    policy in ("fixed", "suite") || error("benchmark.evals.policy must be fixed or suite")
    return Dict{String,Any}(
        "samples" => positive_number(nested(cfg, ["benchmark", "samples"], 100), "samples"; integer=true),
        "seconds" => positive_number(nested(cfg, ["benchmark", "seconds"], 30), "seconds"),
        "evals_policy" => policy,
        "fixed_evals" => positive_number(nested(cfg, ["benchmark", "evals", "fixed_value"], 1), "fixed_value"; integer=true),
        "gctrial" => true, "gcsample" => false, "overhead_ns" => 0.0,
        "estimator" => "median", "engine" => "BenchmarkTools.Trial")
end

function regression_config(cfg)
    limits = get(cfg, "regression", Dict{String,Any}())
    limits isa AbstractDict || error("regression must be a table")
    for (key, value) in limits
        key in ("max_time_ratio", "max_memory_ratio", "max_memory_bytes", "max_allocations") || error("Unknown regression threshold: $key")
        value isa Real && !(value isa Bool) && isfinite(value) || error("regression.$key must be numeric and finite")
        if key in ("max_time_ratio", "max_memory_ratio")
            value > 0 || error("regression.$key must be positive")
        else
            value isa Integer && value >= 0 || error("regression.$key must be a nonnegative integer")
        end
    end
    return Dict{String,Any}(limits)
end

function materialize_output_dir(raw_dir)
    resolved = abspath(raw_dir)
    if basename(resolved) == "latest"
        ispath(resolved) && !islink(resolved) && error("latest exists as a real path; refusing to replace it")
        mkpath(dirname(resolved))
        run_dir = mktempdir(dirname(resolved); prefix=Dates.format(now(UTC), "yyyymmddTHHMMSS") * "-", cleanup=false)
        islink(resolved) && rm(resolved)
        symlink(basename(run_dir), resolved)
        return run_dir, resolved
    end
    mkpath(resolved)
    for filename in ("manifest.json", "benchmarks.json", "summary.json", "baseline.json")
        ispath(joinpath(resolved, filename)) && error("output contains $filename; choose a fresh run directory")
    end
    return resolved, nothing
end

function write_json(path, obj)
    Sys.isunix() || error("atomic JSON publication requires a POSIX host")
    tmp = path * ".writing"
    open(tmp, "w") do io
        JSON3.write(io, obj)
        write(io, '\n')
    end
    # Both paths share a directory/filesystem. POSIX rename replaces atomically
    # without first removing the old record and has no copy fallback.
    status = ccall(:rename, Cint, (Cstring, Cstring), tmp, path)
    Base.systemerror("atomic JSON rename", status != 0)
end

function normalize_benchmark_cases(raw)
    items = raw isa AbstractDict ? collect(raw) : raw
    items isa AbstractVector || error("suite must return a vector or dictionary of named cases")
    isempty(items) && error("suite must contain at least one benchmark")
    cases = Pair{String,Any}[]
    names = Set{String}()
    for item in items
        item isa Pair || error("each case must be name => Benchmark or name => zero-argument function")
        name = strip(string(first(item)))
        isempty(name) && error("benchmark name must not be empty")
        name in names && error("duplicate benchmark name: $name")
        workload = last(item)
        workload isa BenchmarkTools.Benchmark || workload isa Function || error("invalid workload: $name")
        push!(names, name)
        push!(cases, name => workload)
    end
    sort!(cases; by=first)
    return cases
end

function load_suite_cases(path; with_module=false)
    isfile(path) || error("suite file not found: $path")
    suite = Module(gensym(:JuliaPerfSuite))
    # Module(...) does not install bindings emitted by `module ... end`.
    Core.eval(suite, :(include(path::AbstractString) = Base.include($suite, path)))
    Core.eval(suite, :(include(mapexpr::Function, path::AbstractString) = Base.include(mapexpr, $suite, path)))
    Core.eval(suite, :(eval(expression) = Core.eval($suite, expression)))
    Base.include(suite, path)
    raw = if Base.invokelatest(isdefined, suite, :JULIA_PERF_BENCHMARKS)
        Base.invokelatest(getfield, suite, :JULIA_PERF_BENCHMARKS)
    elseif Base.invokelatest(isdefined, suite, :julia_perf_benchmarks)
        Base.invokelatest(Base.invokelatest(getfield, suite, :julia_perf_benchmarks))
    else
        error("suite must define JULIA_PERF_BENCHMARKS or julia_perf_benchmarks()")
    end
    cases = normalize_benchmark_cases(raw)
    return with_module ? (cases, suite) : cases
end


# Package suites can provide deterministic result checks outside the timed region.
function suite_validation(suite, required_names)
    isdefined(suite, :julia_perf_validate) || return nothing
    values = Base.invokelatest(getfield(suite, :julia_perf_validate))
    values isa AbstractDict || error("suite validation must return a named result dictionary")
    Set(string.(keys(values))) == Set(required_names) || error("suite validation case coverage differs from benchmarks")
    return values
end

function loaded_source(cfg, config_path)
    name = nested(cfg, ["target", "module"])
    name === nothing && return nothing
    source_root = resolve_path(dirname(config_path), nested(cfg, ["target", "source_root"], "."))
    matches = [mod for (id, mod) in Base.loaded_modules if id.name == name]
    length(matches) == 1 || error("target module $name must be loaded exactly once by the suite")
    actual = pathof(only(matches))
    expected = joinpath(source_root, "src", "$name.jl")
    actual !== nothing && isfile(expected) && realpath(actual) == realpath(expected) ||
        error("loaded $name source does not match target.source_root")
    return Dict("module" => name, "entry_sha256" => sha256_file(actual),
                "entry_relative_path" => relpath(realpath(actual), realpath(source_root)))
end

function case_evals(workload, protocol)
    if protocol["evals_policy"] == "suite"
        workload isa BenchmarkTools.Benchmark || error("suite evals policy requires Benchmark objects")
        workload.params.evals_set || error("Benchmark must explicitly declare evals")
        return positive_number(workload.params.evals, "case evals"; integer=true)
    end
    return protocol["fixed_evals"]
end

function run_benchmark_case(name, workload, protocol)
    evals = case_evals(workload, protocol)
    bench = workload isa BenchmarkTools.Benchmark ? workload : @benchmarkable $workload() evals=evals
    trial = BenchmarkTools.run(bench; samples=protocol["samples"], seconds=protocol["seconds"],
        evals=evals, gctrial=true, gcsample=false, overhead=0.0)
    isempty(trial.times) && error("benchmark produced no samples: $name")
    return Dict{String,Any}("name" => name, "current_median_ns" => median(trial.times),
        "minimum_ns" => minimum(trial.times), "allocations" => trial.allocs,
        "memory_bytes" => trial.memory, "samples" => length(trial.times),
        "evals" => evals, "sample_times_ns" => trial.times, "sample_gc_times_ns" => trial.gctimes)
end

function file_hashes(base, paths)
    paths isa AbstractVector || error("input_files/source_files must be arrays")
    entries = Pair{String,String}[]
    for relative in paths
        path = resolve_path(base, relative)
        path === nothing && error("empty path in file list")
        if isdir(path)
            for (directory, subdirs, files) in walkdir(path)
                filter!(x -> x != ".git", subdirs)
                for filename in files
                    file = joinpath(directory, filename)
                    push!(entries, relpath(file, base) => sha256_file(file))
                end
            end
        elseif isfile(path)
            push!(entries, relpath(path, base) => sha256_file(path))
        else
            error("fingerprinted path does not exist: $path")
        end
    end
    sort!(entries; by=first)
    return Dict(entries)
end

function git_evidence(root)
    git = Cmd([get(ENV, "JULIA_PERF_GIT", "git")])
    try
        revision = strip(read(pipeline(`$git -C $root rev-parse HEAD`; stderr=devnull), String))
        dirty = read(pipeline(`$git -C $root status --porcelain --untracked-files=all`; stderr=devnull), String)
        diff = read(pipeline(`$git -C $root diff HEAD --binary`; stderr=devnull), String)
        return Dict("revision" => revision, "dirty" => !isempty(dirty),
            "status_sha256" => bytes2hex(sha256(dirty)), "diff_sha256" => bytes2hex(sha256(diff)))
    catch
        return Dict("revision" => nothing, "dirty" => nothing)
    end
end

function runtime_options()
    opts = Base.JLOptions()
    fields = (:compile_enabled, :opt_level, :opt_level_min, :debug_level,
        :check_bounds, :can_inline, :use_compiled_modules, :use_pkgimages,
        :use_sysimage_native_code, :code_coverage, :malloc_log, :fast_math,
        :polly, :strip_metadata, :strip_ir, :permalloc_pkgimg)
    result = Dict{String,Any}(string(field) => getfield(opts, field)
        for field in fields if hasproperty(opts, field))
    result["cpu_target"] = opts.cpu_target == C_NULL ? nothing : unsafe_string(opts.cpu_target)
    return result
end

function runtime_fingerprint(source_root=nothing; project=Base.active_project())
    opts = Base.JLOptions()
    image_pointer = opts.image_file
    image_path = image_pointer == C_NULL ? nothing : unsafe_string(image_pointer)
    # Ask the actual Julia loader: this honors explicit manifest paths,
    # version-specific names/precedence, and shared workspace-root manifests.
    isdefined(Base, :project_file_manifest_path) || error("This Julia runtime has no read-only loader manifest selector; comparison unavailable")
    manifest_path = project === nothing ? nothing : Base.project_file_manifest_path(project)
    deps = Dict{String,Any}()
    records = manifest_path === nothing ? Dict() : get(TOML.parsefile(manifest_path), "deps", Dict())
    for (name, entries) in records
        dep_records = Any[]
        for info in entries
            source_hash = nothing
            candidate = false
            if haskey(info, "path")
                source = abspath(dirname(manifest_path), info["path"])
                candidate = source_root !== nothing && realpath(source) == realpath(source_root)
                if !candidate
                    # Extensions, build scripts, generated/native libraries,
                    # and other loaded files can live outside src. Conservatively
                    # fingerprint the entire path dependency except .git.
                    source_hash = hash_json(file_hashes(source, ["."]))
                end
            end
            push!(dep_records, Dict("uuid" => get(info, "uuid", nothing), "version" => get(info, "version", nothing),
                "tree_hash" => get(info, "git-tree-sha1", nothing), "candidate_dependency" => candidate, "path_source_sha256" => source_hash))
        end
        deps[name] = dep_records
    end
    return Dict{String,Any}(
        "julia_version" => string(VERSION), "julia_binary_sha256" => sha256_file(joinpath(Sys.BINDIR, Base.julia_exename())),
        "julia_sysimage_sha256" => image_path === nothing || !isfile(image_path) ? nothing : sha256_file(image_path),
        "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "threadpools" => Dict("default" => Threads.nthreads(:default), "interactive" => Threads.nthreads(:interactive)),
        "gc_threads" => Dict("actual_total" => Threads.ngcthreads(), "requested_mark" => opts.nmarkthreads, "requested_sweep" => opts.nsweepthreads),
        "blas_config" => sprint(show, BLAS.get_config()), "os" => string(Sys.KERNEL),
        "blas_environment" => Dict(key=>get(ENV,key,nothing) for key in
            ("OPENBLAS_NUM_THREADS","GOTO_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")),
        "arch" => string(Sys.ARCH), "cpu" => Sys.CPU_NAME, "cpu_threads" => Sys.CPU_THREADS,
        "machine" => gethostname(), "word_size" => Sys.WORD_SIZE,
        "project_sha256" => project === nothing ? nothing : sha256_file(project),
        "manifest_toml_sha256" => manifest_path === nothing || !isfile(manifest_path) ? nothing : sha256_file(manifest_path),
        "dependencies" => deps, "benchmarktools_version" => string(pkgversion(BenchmarkTools)),
        "julia_cpu_target" => get(ENV, "JULIA_CPU_TARGET", ""),
        "julia_options" => runtime_options())
end

function make_manifest(cfg, config_path, suite_path, cases, protocol, out_dir; results_sha256=hash_json(nothing))
    base = dirname(config_path)
    source_root = resolve_path(base, nested(cfg, ["target", "source_root"], "."))
    source_files = nested(cfg, ["target", "source_files"], [p for p in ("src", "Project.toml", "Manifest.toml") if ispath(joinpath(source_root, p))])
    return Dict{String,Any}(
        "schema_version" => 2, "tool" => "julia-perf", "timestamp_utc" => utcnow(),
        "config_sha256" => sha256_file(config_path), "suite_sha256" => sha256_file(suite_path),
        "runner_sha256" => sha256_file(@__FILE__), "source_root" => source_root,
        "source_files" => file_hashes(source_root, source_files), "git" => git_evidence(source_root),
        "runtime" => runtime_fingerprint(source_root), "protocol" => protocol,
        "loaded_source" => loaded_source(cfg, config_path), "results_sha256" => results_sha256,
        "case_evals" => Dict(name => case_evals(workload, protocol) for (name, workload) in cases),
        "input_files" => file_hashes(base, nested(cfg, ["benchmark", "input_files"], String[])),
        "output_dir" => out_dir,
        "limitations" => ["Type-inference, JET, profiling and implementation-identity checks are not automated.",
            "Statistical significance and uncertainty inference are not implemented; ratios are descriptive estimates.",
            "Input/source files are caller-declared; external data/environment effects require their own records.",
            "Deadline begins after runner imports/configuration; child startup, loading, compilation, warmup and samples are bounded.",
            "POSIX worker process groups are cleaned on timeout and exit; suites must not detach or daemonize descendants. Other platforms fail before starting the suite.",
            "Power state, background load and thermal conditions are not measured."])
end

comparison_signature(manifest) = Dict(k => manifest[k] for k in ("suite_sha256", "runner_sha256", "runtime", "protocol", "case_evals", "input_files", "results_sha256"))

valid_sha256(value) = value isa AbstractString && occursin(r"^[0-9a-f]{64}$", value)

function validate_hash_map(value, label; nonempty=false)
    value isa AbstractDict || error("baseline $label must be a file-hash object")
    nonempty && isempty(value) && error("baseline $label must contain source evidence")
    for (path, digest) in value
        path isa AbstractString && !isempty(strip(path)) && valid_sha256(digest) || error("invalid baseline $label entry")
    end
end

function validate_baseline_manifest(manifest)
    manifest isa AbstractDict || error("baseline manifest is missing")
    required = ("schema_version", "tool", "timestamp_utc", "config_sha256", "suite_sha256",
        "runner_sha256", "source_root", "source_files", "git", "runtime", "protocol",
        "case_evals", "input_files", "output_dir", "limitations", "loaded_source", "results_sha256")
    all(key -> haskey(manifest, key), required) || error("baseline manifest is missing provenance fields")
    version = manifest["schema_version"]
    version isa Integer && !(version isa Bool) && version == 2 || error("invalid baseline manifest schema")
    manifest["tool"] == "julia-perf" || error("invalid baseline manifest tool")
    stamp = manifest["timestamp_utc"]
    stamp isa AbstractString && tryparse(DateTime, stamp, dateformat"yyyy-mm-ddTHH:MM:SSZ") !== nothing || error("invalid baseline timestamp")
    for key in ("config_sha256", "suite_sha256", "runner_sha256", "results_sha256")
        valid_sha256(manifest[key]) || error("invalid baseline $key")
    end
    for key in ("source_root", "output_dir")
        manifest[key] isa AbstractString && !isempty(strip(manifest[key])) || error("invalid baseline $key")
    end
    validate_hash_map(manifest["source_files"], "source_files"; nonempty=true)
    validate_hash_map(manifest["input_files"], "input_files")
    git = manifest["git"]
    git isa AbstractDict && haskey(git, "revision") && haskey(git, "dirty") || error("invalid baseline git evidence")
    if git["revision"] === nothing
        git["dirty"] === nothing || error("invalid unavailable baseline git evidence")
    else
        git["revision"] isa AbstractString && occursin(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", git["revision"]) && git["dirty"] isa Bool || error("invalid baseline git revision/status")
        all(key -> haskey(git, key) && valid_sha256(git[key]), ("status_sha256", "diff_sha256")) || error("baseline git evidence lacks status/diff hashes")
    end
    for key in ("runtime", "protocol", "case_evals")
        manifest[key] isa AbstractDict && !isempty(manifest[key]) || error("invalid baseline $key")
    end
    for (name, evals) in manifest["case_evals"]
        name isa AbstractString && !isempty(strip(name)) || error("invalid baseline case name")
        positive_number(evals, "baseline case evals"; integer=true)
    end
    limits = manifest["limitations"]
    limits isa AbstractVector && !isempty(limits) && all(x -> x isa AbstractString && !isempty(strip(x)), limits) || error("invalid baseline limitations")
    return manifest
end

function validate_loaded_source(baseline, current)
    expected = current["loaded_source"]
    expected === nothing && return
    recorded = baseline["loaded_source"]
    recorded isa AbstractDict || error("baseline lacks loaded target source evidence")
    get(recorded, "module", nothing) == expected["module"] || error("baseline target module differs")
    get(recorded, "entry_relative_path", nothing) == expected["entry_relative_path"] || error("baseline target entry differs")
    valid_sha256(get(recorded, "entry_sha256", nothing)) || error("baseline target entry hash is missing")
end

function validate_rows(rows)
    rows isa AbstractVector && !isempty(rows) || error("baseline rows must be a nonempty array")
    indexed = Dict{String,Any}()
    for row in rows
        name = get(row, "name", nothing)
        name isa AbstractString && !isempty(strip(name)) || error("baseline row name is invalid")
        haskey(indexed, name) && error("duplicate baseline row: $name")
        time = get(row, "current_median_ns", nothing)
        time isa Real && !(time isa Bool) && isfinite(time) && time > 0 || error("invalid baseline median: $name")
        for key in ("allocations", "memory_bytes")
            value = get(row, key, nothing)
            value isa Integer && !(value isa Bool) && value >= 0 || error("invalid baseline $key: $name")
        end
        raw = get(row, "sample_times_ns", nothing)
        raw isa AbstractVector && !isempty(raw) && all(t -> t isa Real && !(t isa Bool) && isfinite(t) && t > 0, raw) || error("invalid baseline raw samples: $name")
        get(row, "samples", nothing) == length(raw) || error("baseline sample count mismatch: $name")
        time == median(raw) || error("baseline median differs from raw samples: $name")
        indexed[name] = row
    end
    return indexed
end

function load_baseline(path)
    path !== nothing && isfile(path) || error("baseline file is missing")
    data = JSON3.read(read(path, String), Dict{String,Any})
    version = get(data, "schema_version", nothing)
    version isa Integer && !(version isa Bool) && version == 2 && get(data, "kind", nothing) == "julia-perf-baseline" || error("baseline lacks the required provenance schema")
    manifest = get(data, "manifest", nothing)
    validate_baseline_manifest(manifest)
    rows = validate_rows(get(data, "rows", nothing))
    evals = manifest["case_evals"]
    Set(keys(rows)) == Set(keys(evals)) || error("baseline row names differ from its protocol")
    for (name, row) in rows
        row_evals = get(row, "evals", nothing)
        row_evals isa Integer && !(row_evals isa Bool) && row_evals > 0 && row_evals == evals[name] || error("baseline evals differ from its protocol: $name")
    end
    return manifest, rows
end

function compare_rows!(rows, baseline_rows, limits)
    Set(row["name"] for row in rows) == Set(keys(baseline_rows)) || error("baseline and current benchmark names differ")
    failures = Dict{String,Any}[]
    for row in rows
        base = baseline_rows[row["name"]]
        row["baseline_median_ns"] = base["current_median_ns"]
        row["ratio"] = row["current_median_ns"] / base["current_median_ns"]
        row["allocations_delta"] = row["allocations"] - base["allocations"]
        row["memory_bytes_delta"] = row["memory_bytes"] - base["memory_bytes"]
        memory_ratio = base["memory_bytes"] == 0 ? (row["memory_bytes"] == 0 ? 1.0 : Inf) : row["memory_bytes"] / base["memory_bytes"]
        row["memory_ratio"] = isfinite(memory_ratio) ? memory_ratio : nothing
        for (key, metric) in (("max_time_ratio", row["ratio"]), ("max_memory_ratio", memory_ratio),
            ("max_memory_bytes", row["memory_bytes"]), ("max_allocations", row["allocations"]))
            if haskey(limits, key) && metric > limits[key]
                push!(failures, Dict("check_id" => "regression-threshold", "detail" => "$(row["name"]): $key exceeded", "threshold" => limits[key], "observed" => isfinite(metric) ? metric : "infinity"))
            end
        end
    end
    return failures
end

function write_summary(out_dir, verdict, code, failures; notes=String[])
    summary = Dict("verdict" => verdict, "exit_code" => code, "thresholds_passed" => verdict == "PASS", "claim_verified" => false,
        "hard_failures" => failures, "notes" => notes, "timestamp_utc" => utcnow(),
        "verification_scope" => "declared descriptive regression thresholds only",
        "statistical_analysis" => "not_performed", "equivalence_check" => "external_required", "type_diagnostics" => "not_checked")
    write_json(joinpath(out_dir, "summary.json"), summary)
    println("wrote artifacts to: ", out_dir, "\nverdict: ", verdict)
    return code
end

function check_blas_threads(requested)
    requested === nothing && return
    actual = BLAS.get_num_threads()
    actual == requested || error("BLAS thread count differs: requested $requested, actual $actual")
end

function worker_main(opts, cfg, config_path, out_dir)
    rows = Dict{String,Any}[]
    failures = Dict{String,Any}[]
    case_status = Dict{String,Any}()
    try
        protocol = protocol_config(cfg)
        limits = regression_config(cfg)
        suite_path = resolve_path(dirname(config_path), nested(cfg, ["benchmark", "suite_file"]))
        suite_path === nothing && error("benchmark.suite_file is required")
        blas_threads = nested(cfg, ["blas", "num_threads"])
        blas_threads === nothing || BLAS.set_num_threads(positive_number(blas_threads, "blas.num_threads"; integer=true))
        cases, suite = load_suite_cases(suite_path; with_module=true)
        # Package initialization may load a BLAS backend after the first call.
        blas_threads === nothing || BLAS.set_num_threads(positive_number(blas_threads, "blas.num_threads"; integer=true))
        for (name, _) in cases
            case_status[name] = Dict("status" => "not_started")
        end
        write_json(joinpath(out_dir, "case_status.json"), case_status)
        outputs = suite_validation(suite, first.(cases))
        # Preparation can initialize another BLAS backend. Configure the fully
        # prepared workload before capturing its identity or timing any case.
        blas_threads === nothing || BLAS.set_num_threads(positive_number(blas_threads, "blas.num_threads"; integer=true))
        check_blas_threads(blas_threads)
        manifest = make_manifest(cfg, config_path, suite_path, cases, protocol, out_dir;
                                 results_sha256=hash_json(outputs))
        write_json(joinpath(out_dir, "manifest.json"), manifest)
        outputs === nothing || write_json(joinpath(out_dir, "results.json"), outputs)
        baseline = nothing
        if !opts["save_baseline"]
            baseline_path = resolve_path(dirname(config_path), nested(cfg, ["benchmark", "baseline_file"]))
            baseline = load_baseline(baseline_path)
            validate_loaded_source(first(baseline), manifest)
            comparison_signature(manifest) == comparison_signature(first(baseline)) || error("baseline is incomparable: suite, inputs, protocol, runtime, hardware or dependencies differ")
            Set(first.(cases)) == Set(keys(last(baseline))) || error("baseline and current benchmark names differ")
        end
        for (name, workload) in cases
            check_blas_threads(blas_threads)
            case_status[name] = Dict("status" => "running")
            write_json(joinpath(out_dir, "case_status.json"), case_status)
            push!(rows, Base.invokelatest(run_benchmark_case, name, workload, protocol))
            write_json(joinpath(out_dir, "benchmarks.json"), rows)
            check_blas_threads(blas_threads)
            case_status[name] = Dict("status" => "completed")
            write_json(joinpath(out_dir, "case_status.json"), case_status)
        end
        outputs_after = suite_validation(suite, first.(cases))
        check_blas_threads(blas_threads)
        hash_json(outputs) == hash_json(outputs_after) || error("suite outputs or counters changed during measurement")
        after = make_manifest(cfg, config_path, suite_path, cases, protocol, out_dir; results_sha256=hash_json(outputs_after))
        for key in ("source_files", "suite_sha256", "config_sha256", "runtime", "input_files", "loaded_source")
            manifest[key] == after[key] || error("fingerprinted $key changed during benchmarking")
        end
        if opts["save_baseline"]
            write_json(joinpath(out_dir, "baseline.json"), Dict("schema_version" => 2, "kind" => "julia-perf-baseline", "manifest" => manifest, "rows" => rows))
            return write_summary(out_dir, "BASELINE_CREATED", 0, failures; notes=["Baseline creation does not verify performance, correctness or implementation identity."])
        end
        append!(failures, compare_rows!(rows, last(baseline), limits))
        write_json(joinpath(out_dir, "benchmarks.json"), rows)
        isempty(failures) || return write_summary(out_dir, "FAIL", 1, failures)
        isempty(limits) && return write_summary(out_dir, "UNVERIFIED", 2, failures; notes=["Comparable measurements collected, but no regression thresholds were declared."])
        manifest["runtime"]["manifest_toml_sha256"] === nothing && return write_summary(out_dir, "UNVERIFIED", 2, failures; notes=["No resolved Manifest.toml was available."])
        return write_summary(out_dir, "PASS", 0, failures; notes=["PASS covers declared thresholds only; statistical and equivalence evidence is external."])
    catch err
        println(stderr, "ERROR: ", sprint(showerror, err))
        push!(failures, Dict("check_id" => "benchmark-or-baseline-invalid", "detail" => sprint(showerror, err)))
        for (name, status) in case_status
            if status["status"] == "running"
                case_status[name] = Dict("status" => "failed", "error" => sprint(showerror, err))
            end
        end
        write_json(joinpath(out_dir, "case_status.json"), case_status)
        write_json(joinpath(out_dir, "benchmarks.json"), rows)
        return write_summary(out_dir, "FAIL", 1, failures)
    end
end

function owned_worker(command)
    # libuv's detached flag creates a new session/process group on POSIX.
    # Descendants inherit it unless they explicitly daemonize/detach, which
    # suites using this deadline must not do. No global process search is used.
    Sys.isunix() || error("owned process-group deadlines require a POSIX host; suite was not started")
    run(Cmd(command; detach=true); wait=false)
end

function terminate_owned_group(child, group)
    # Signal only the process group created for this invocation, including any
    # ordinary suite subprocesses. ESRCH means that it has already disappeared.
    result = ccall(:kill, Cint, (Cint, Cint), -group, Base.SIGKILL)
    result == 0 || Libc.errno() == Libc.ESRCH || error("could not terminate owned worker process group")
    wait(child)
end

function main(args)
    try
        opts = parse_args(args)
        opts["help"] && (println(USAGE); return 0)
        config_path = abspath(opts["config"])
        cfg = TOML.parsefile(config_path)
        protocol_config(cfg)
        regression_config(cfg)
        deadline = positive_number(nested(cfg, ["benchmark", "timeout_seconds"], 600), "timeout_seconds")
        base = dirname(config_path)
        mode = get(opts, "mode", nested(cfg, ["mode", "mode"], "auto"))
        mode in ("standalone", "ecosystem", "auto") || error("invalid mode")
        artifact_root = resolve_path(base, get(opts, "artifact_root", nested(cfg, ["integration", "artifact_root"])))
        tag = get(opts, "tag", nested(cfg, ["integration", "tag"], ""))
        mode == "auto" && (mode = artifact_root !== nothing && !isempty(tag) ? "ecosystem" : "standalone")
        mode == "ecosystem" && (artifact_root === nothing || isempty(tag)) && error("ecosystem mode requires artifact root and tag")
        agent = get(opts, "agent_id", nested(cfg, ["integration", "agent_id"], ""))
        for value in (tag, agent)
            isempty(value) || (basename(value) == value && value ∉ (".", "..")) || error("tag and agent-id must be single path components")
        end
        output = get(opts, "out_dir", mode == "ecosystem" ? joinpath(artifact_root, tag, "julia-perf", agent) : nested(cfg, ["output", "out_dir"], ".julia-perf/runs/latest"))
        out_dir, _ = materialize_output_dir(resolve_path(base, output))
        child_args = [config_path, out_dir, opts["save_baseline"] ? "true" : "false"]
        project = Base.active_project()
        project_arg = project === nothing ? String[] : ["--project=$(dirname(project))"]
        # The private child entry is fixed code; arbitrary paths arrive only as
        # ARGS. Public parse_args never accepts a supervisor-bypass flag.
        child_code = "include(" * repr(abspath(@__FILE__)) * "); cfg=TOML.parsefile(ARGS[1]); exit(worker_main(Dict(\"save_baseline\"=>ARGS[3]==\"true\"), cfg, ARGS[1], ARGS[2]))"
        # julia_cmd preserves compiler options, but does not propagate -t.
        default_threads = Threads.nthreads(:default)
        interactive_threads = Threads.nthreads(:interactive)
        # Julia 1.11 rejects an explicit zero interactive pool. Its single-count
        # syntax already preserves that runtime's zero interactive default.
        thread_pools = VERSION < v"1.12" && interactive_threads == 0 ?
                       string(default_threads) : "$(default_threads),$(interactive_threads)"
        options = Base.JLOptions()
        # Zero is the runtime's automatic-GC sentinel, not a valid CLI count.
        gc_args = options.nmarkthreads > 0 ? ["--gcthreads=$(options.nmarkthreads),$(options.nsweepthreads)"] : String[]
        command = `$(Base.julia_cmd()) $project_arg --threads=$thread_pools $gc_args --startup-file=no -e $child_code $child_args`
        blas_threads = nested(cfg,["blas","num_threads"])
        if blas_threads !== nothing
            requested = string(positive_number(blas_threads,"blas.num_threads";integer=true))
            # OpenMP OpenBLAS builds can reapply omp_get_max_threads during
            # operations. Set both limits before this owned worker initializes.
            command = addenv(command,"OPENBLAS_NUM_THREADS"=>requested,"OMP_NUM_THREADS"=>requested)
        end
        child = owned_worker(command)
        group = getpid(child)
        done = timedwait(() -> process_exited(child), deadline; pollint=0.05)
        if done == :timed_out
            terminate_owned_group(child, group)
            failures = [Dict("check_id" => "timeout", "detail" => "worker exceeded $(deadline) seconds; complete case artifacts retained")]
            statuses = joinpath(out_dir, "case_status.json")
            if isfile(statuses)
                data = JSON3.read(read(statuses, String), Dict{String,Any})
                for (name, status) in data
                    status["status"] == "running" && (data[name] = Dict("status" => "timed_out"))
                end
                write_json(statuses, data)
            end
            baseline = joinpath(out_dir, "baseline.json")
            isfile(baseline) && rm(baseline)
            return write_summary(out_dir, "FAIL", 1, failures)
        end
        terminate_owned_group(child, group)
        if !isfile(joinpath(out_dir, "summary.json"))
            return write_summary(out_dir, "FAIL", 1, [Dict("check_id" => "worker-failed", "detail" => "worker exited without a summary")])
        end
        return child.exitcode
    catch err
        println(stderr, "ERROR: ", sprint(showerror, err))
        return 3
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main(ARGS))
end
