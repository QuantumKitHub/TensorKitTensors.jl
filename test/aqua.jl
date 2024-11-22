using Test
using Aqua
using TensorKitTensors

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(TensorKitTensors)
end
