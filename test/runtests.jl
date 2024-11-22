using TensorKitTensors
using Test
using Aqua

@testset "TensorKitTensors.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TensorKitTensors)
    end
    # Write your tests here.
end
