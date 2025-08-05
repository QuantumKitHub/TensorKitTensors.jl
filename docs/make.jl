using TensorKitTensors
using Documenter

DocMeta.setdocmeta!(
    TensorKitTensors, :DocTestSetup, :(using TensorKitTensors);
    recursive = true
)

operatorpages = [
    "spinoperators.md", "bosonoperators.md", "fermionoperators.md",
    "tjoperators.md", "hubbardoperators.md",
]
makedocs(;
    modules = [TensorKitTensors],
    authors = "QuantumKitHub",
    sitename = "TensorKitTensors.jl",
    format = Documenter.HTML(;
        canonical = "https://QuantumKitHub.github.io/TensorKitTensors.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "Operators" => operatorpages],
)

deploydocs(;
    repo = "github.com/QuantumKitHub/TensorKitTensors.jl",
    devbranch = "main",
    push_preview = true
)
