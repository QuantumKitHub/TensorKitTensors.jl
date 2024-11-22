using SafeTestsets

@time @safetestset "SpinOperators" begin
    include("spinoperators.jl")
end

@time @safetestset "Aqua" begin
    include("aqua.jl")
end
