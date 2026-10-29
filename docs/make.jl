using Documenter
using dACE

makedocs(
    sitename = "dACE.jl",
    modules = [dACE],
    checkdocs = :exports,
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
    pages = [
        "Home" => "index.md",
        "Reproducing the paper" => "reproducing.md",
        "API reference" => "api.md",
    ],
)

if get(ENV, "GITHUB_ACTIONS", "false") == "true" && get(ENV, "GITHUB_REF", "") == "refs/heads/main"
    deploydocs(
        repo = "github.com/TSGut/dACE.jl.git",
        devbranch = "main",
    )
end
