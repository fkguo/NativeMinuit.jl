# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Documenter.jl entry point. Builds docs into docs/build/.
# Run from the repo root with:
#     julia --project=docs docs/make.jl

using Documenter
using JuMinuit

DocMeta.setdocmeta!(JuMinuit, :DocTestSetup, :(using JuMinuit); recursive = true)

makedocs(
    sitename = "JuMinuit.jl",
    authors  = "Feng-Kun Guo",
    modules  = [JuMinuit],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://fkguo.github.io/JuMinuit.jl",
        edit_link  = "main",
        repolink   = "https://github.com/fkguo/JuMinuit.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Quickstart" => "tutorials/quickstart.md",
            "Bounded parameters" => "tutorials/bounded.md",
            "MINOS errors & contours" => "tutorials/minos_contours.md",
        ],
        "Cost functions" => "cost_functions.md",
        "Error analysis" => "error_analysis.md",
        "API Reference" => "api.md",
        "Internals" => "internals.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

# CI deployment to gh-pages (skips on local builds)
if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    deploydocs(
        repo = "github.com/fkguo/JuMinuit.jl",
        devbranch = "main",
        push_preview = true,
    )
end
