using SafeTestsets

@time @safetestset "spin operators" begin
    include("spinoperators.jl")
end

@time @safetestset "boson operators" begin
    include("bosonoperators.jl")
end

@time @safetestset "Hubbard operators" begin
    include("hubbardoperators.jl")
end

@time @safetestset "tj operators" begin
    include("tjoperators.jl")
end

@time @safetestset "spinless fermion operators" begin
    include("fermionoperators.jl")
end

@time @safetestset "Aqua" begin
    include("aqua.jl")
end
