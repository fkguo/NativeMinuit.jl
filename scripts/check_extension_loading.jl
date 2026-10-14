# SPDX-License-Identifier: LGPL-2.1-or-later
# Resolve isolated consumer environments and exercise fresh-process loading.
using Pkg, TOML

const package_root = normpath(joinpath(@__DIR__, ".."))
const package_project = TOML.parsefile(joinpath(package_root, "Project.toml"))

mktempdir() do workspace
    core = joinpath(workspace, "core")
    full = joinpath(workspace, "full")
    mkpath(core)
    mkpath(full)
    deps = merge(package_project["deps"],
                 Dict(name => package_project["extras"][name]
                      for name in package_project["targets"]["test"]))
    compat = Dict(name => value for (name, value) in package_project["compat"]
                  if name == "julia" || haskey(deps, name))
    open(joinpath(full, "Project.toml"), "w") do io
        TOML.print(io, Dict("deps" => deps, "compat" => compat))
    end
    for consumer in (core, full)
        Pkg.activate(consumer)
        Pkg.develop(path=package_root)
        Pkg.instantiate()
    end
    command = `$(Base.julia_cmd()) --startup-file=no --project=$full $(joinpath(package_root, "test", "test_extension_loading.jl")) $core`
    run(addenv(command, "JULIA_LOAD_PATH" => "@:@stdlib"))
end
