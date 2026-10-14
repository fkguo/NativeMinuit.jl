# SPDX-License-Identifier: LGPL-2.1-or-later
# Run in an environment containing NativeMinuit and its extension triggers,
# passing a separate core-only consumer project as the first argument.
using Test
length(ARGS) == 1 || error("Pass the core-only consumer project directory")
const _loading_core_project = abspath(ARGS[1])
const _loading_full_project = dirname(Base.active_project())
const _loading_extensions = [:NativeMinuitForwardDiffExt, :NativeMinuitDataFramesExt,
    :NativeMinuitClusteringExt, :NativeMinuitOptimExt, :NativeMinuitPlotsExt,
    :NativeMinuitAdvancedHMCExt]
const _loading_artifacts = get(ENV, "NATIVE_MINUIT_LOADING_LOGS", mktempdir())
mkpath(_loading_artifacts)

function _loading_case(name, project, code)
    @testset "$name" begin
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1 --check-bounds=yes --project=$project -e $code`
        cmd = addenv(cmd, "JULIA_LOAD_PATH" => "@:@stdlib", "GKSwstype" => "100")
        path = joinpath(_loading_artifacts, name * ".log")
        ok = open(path, "w") do io
            success(pipeline(cmd; stdout=io, stderr=io))
        end
        ok || print(read(path, String))
        @test ok
        @test occursin("LOADING_CASE_OK", read(path, String))
    end
end
const _loading_preamble = "using Test; "
const _loading_finish = "println(\"LOADED_PACKAGES=\", sort!([id.name for id in keys(Base.loaded_modules)])); println(\"LOADING_CASE_OK\")"

@testset "Fresh consumer extension loading" begin
    absence = "using NativeMinuit; " *
        "@test all(Base.get_extension(NativeMinuit, e) === nothing for e in " * repr(_loading_extensions) * "); " *
        "m=Minuit(x->sum(abs2,x), [0.0,0.0]); migrad!(m); @test m.valid; "
    _loading_case("core-only", _loading_core_project, _loading_preamble * absence * _loading_finish)
    _loading_case("installed-not-loaded", _loading_full_project, _loading_preamble * absence * _loading_finish)
    operations = [
        ("ForwardDiff", "NativeMinuitForwardDiffExt", "cf=CostFunctionAD(x->sum(abs2,x)); @test is_valid(migrad(cf,[0.0,0.0],[0.1,0.1]));"),
        ("DataFrames", "NativeMinuitDataFramesExt", "d=Data(DataFrame(x=[1.0],y=[2.0],err=[0.1])); @test d.x==[1.0];"),
        ("Clustering", "NativeMinuitClusteringExt", "m=Minuit(x->sum(abs2,x),[0.0,0.0]); migrad!(m); s=[0.0 0.0; 0.1 0.0; 5.0 5.0; 5.1 5.0]; modes=find_solution_modes(s,m;method=:dbscan,whiten=:errors); @test length(modes)==2;"),
        ("Optim", "NativeMinuitOptimExt", "m=Minuit(x->sum(abs2,x),[1.0,1.0]); optim(m); @test m.valid;")]
    for (trigger, extension, operation) in operations, reverse in (false,true)
        imports = reverse ? "using $trigger; using NativeMinuit; " : "using NativeMinuit; using $trigger; "
        code = _loading_preamble * imports * "@test Base.get_extension(NativeMinuit,:$extension)!==nothing; " * operation * _loading_finish
        _loading_case(lowercase(trigger) * (reverse ? "-trigger-first" : "-parent-first"), _loading_full_project, code)
    end
    # AdvancedHMC transitively loads the two log-density packages. Real missing
    # trigger cases use only subsets whose dependency closure remains absent.
    subsets = [("missing-advancedhmc", "ForwardDiff, LogDensityProblems, LogDensityProblemsAD, TransformVariables", "AdvancedHMC"),
               ("missing-transformvariables", "AdvancedHMC, ForwardDiff", "TransformVariables"),
               ("missing-forwarddiff-and-transformvariables", "AdvancedHMC", "ForwardDiff")]
    for (name, imports, missing) in subsets
        code = _loading_preamble * "using NativeMinuit; using $imports; " *
            "@test !any(id.name==\"$missing\" for id in keys(Base.loaded_modules)); " *
            "@test Base.get_extension(NativeMinuit,:NativeMinuitAdvancedHMCExt)===nothing; " *
            "m=Minuit(x->sum(abs2,x),[0.0]); migrad!(m); @test_throws ArgumentError posterior_sample(m;sampler=:nuts,nsteps=20,burn=10,nchains=1,warn=false); " * _loading_finish
        _loading_case(name, _loading_full_project, code)
    end
    transitive = _loading_preamble * "using NativeMinuit; using AdvancedHMC, TransformVariables; " *
        "@test any(id.name==\"ForwardDiff\" for id in keys(Base.loaded_modules)); " *
        "@test Base.get_extension(NativeMinuit,:NativeMinuitAdvancedHMCExt)!==nothing; " * _loading_finish
    _loading_case("transitive-trigger-completion", _loading_full_project, transitive)
    for reverse in (false,true)
        triggers="AdvancedHMC, LogDensityProblems, LogDensityProblemsAD, TransformVariables, ForwardDiff"
        imports=reverse ? "using $triggers; using NativeMinuit; " : "using NativeMinuit; using $triggers; "
        code=_loading_preamble * imports *
            "@test Base.get_extension(NativeMinuit,:NativeMinuitAdvancedHMCExt)!==nothing; " *
            "@test Base.get_extension(NativeMinuit,:NativeMinuitForwardDiffExt)!==nothing; " *
            "m=Minuit(x->sum(abs2,x),[0.0]); migrad!(m); p=posterior_sample(m;sampler=:nuts,nsteps=20,burn=10,nchains=1,seed=19,warn=false); " *
            "@test p.sampler===:nuts; @test size(p.ensemble.samples)==(10,1); @test all(isfinite,p.ensemble.samples); " * _loading_finish
        _loading_case("advancedhmc" * (reverse ? "-trigger-first" : "-parent-first"), _loading_full_project, code)
    end
end
