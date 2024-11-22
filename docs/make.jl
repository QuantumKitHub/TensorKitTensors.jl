using TensorKitTensors
using Documenter

DocMeta.setdocmeta!(TensorKitTensors, :DocTestSetup, :(using TensorKitTensors);
                    recursive=true)

makedocs(;
         modules=[TensorKitTensors],
         authors="QuantumKitHub",
         sitename="TensorKitTensors.jl",
         format=Documenter.HTML(;
                                canonical="https://QuantumKitHub.github.io/TensorKitTensors.jl",
                                edit_link="main",
                                assets=String[],),
         pages=["Home" => "index.md"],)

deploydocs(; repo="github.com/QuantumKitHub/TensorKitTensors.jl", devbranch="main")
