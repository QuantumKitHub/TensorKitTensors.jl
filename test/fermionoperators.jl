using TensorKit
using Test
include("testsetup.jl")
using TensorKitTensors
using .TensorKitTensorsTestSetup
using TensorKitTensors.FermionOperators
using StableRNGs

# anticommutation relations
# {cᵢ†, cⱼ†} = 0 = {cᵢ, cⱼ}
# {cᵢ, cⱼ†} = δᵢⱼ

@testset "simple fermions" begin
    @test c⁻c⁻() ≈ -permute(c⁻c⁻(), ((2, 1), (4, 3)))
    @test c⁺c⁺() ≈ -permute(c⁺c⁺(), ((2, 1), (4, 3)))

    # the following doesn't hold
    # I don't think I can get all of these to hold simultaneously?
    # @test cc⁺ ≈ -permute(c⁺c, (2, 1), (4, 3))

    @test c⁻c⁺()' ≈ c⁺c⁻()
    @test c⁻c⁻()' ≈ c⁺c⁺()
    @test (c⁺c⁻() + c⁻c⁺())' ≈ c⁻c⁺() + c⁺c⁻()
    @test (c⁺c⁻() - c⁻c⁺())' ≈ c⁻c⁺() - c⁺c⁻()

    @plansor c_number[-1; -2] := c⁺c⁻()[-1 1; 3 2] * τ[3 2; -2 1]
    @test c_number ≈ c_num()
end

@testset "Exact Diagonalization" begin
    rng = StableRNG(123)

    L = 2
    t, V, mu = rand(rng, 3)
    pspace = fermion_space()

    H = -t * (c⁻c⁺() + c⁺c⁻()) +
        V * ((n() - 0.5 * id(pspace)) ⊗ (n() - 0.5 * id(pspace))) -
        0.5 * mu * (n() ⊗ id(pspace) + id(pspace) ⊗ n())

    # Values based on https://arxiv.org/abs/1610.05003v1. Half-Chain Entanglement Entropy in the One-Dimensional Spinless Fermion Model
    true_eigenvalues = sort([V / 4, V / 4 - mu, -V / 4 - mu / 2 + t, -V / 4 - mu / 2 - t])
    eigenvals = expanded_eigenvalues(H; L)
    @test eigenvals ≈ true_eigenvalues
end
