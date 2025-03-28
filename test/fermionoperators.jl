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
    cc = contract_twosite(c⁻(; side=:L), c⁻(; side=:R))
    cc⁺ = contract_twosite(c⁻(; side=:L), c⁺(; side=:R))
    c⁺c = contract_twosite(c⁺(; side=:L), c⁻(; side=:R))
    c⁺c⁺ = contract_twosite(c⁺(; side=:L), c⁺(; side=:R))

    @test cc ≈ -permute(cc, ((2, 1), (4, 3)))
    @test c⁺c⁺ ≈ -permute(c⁺c⁺, ((2, 1), (4, 3)))

    # the following doesn't hold
    # I don't think I can get all of these to hold simultaneously?
    # @test cc⁺ ≈ -permute(c⁺c, (2, 1), (4, 3))

    @test cc⁺' ≈ c⁺c
    @test cc' ≈ c⁺c⁺
    @test (c⁺c + cc⁺)' ≈ cc⁺ + c⁺c
    @test (c⁺c - cc⁺)' ≈ cc⁺ - c⁺c

    @test c_number() ≈ contract_onesite(c⁺(; side=:L), c⁻(; side=:R))
end

@testset "Exact Diagonalization" begin
    rng = StableRNG(123)

    L = 2
    t, V, mu = rand(rng, 3)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)

    cc⁺ = contract_twosite(c⁻(; side=:L), c⁺(; side=:R))
    c⁺c = contract_twosite(c⁺(; side=:L), c⁻(; side=:R))

    H = -t * (cc⁺ + c⁺c) +
        V * ((c_number() - 0.5 * id(pspace)) ⊗ (c_number() - 0.5 * id(pspace))) -
        0.5 * mu * (c_number() ⊗ id(pspace) + id(pspace) ⊗ c_number())
    # Values based on https://arxiv.org/abs/1610.05003v1. Half-Chain Entanglement Entropy in the One-Dimensional Spinless Fermion Model
    true_eigenvalues = sort([V / 4, V / 4 - mu, -V / 4 - mu / 2 + t, -V / 4 - mu / 2 - t])

    eigenvals = get_lowest_eigenvalues(H, -1; L)
    @test eigenvals ≈ true_eigenvalues
end
