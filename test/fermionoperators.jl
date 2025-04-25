using TensorKit
using Test
include("testsetup.jl")
using TensorKitTensors
using .TensorKitTensorsTestSetup
using TensorKitTensors.FermionOperators
using StableRNGs

# anticommutation relations
# {fᵢ†, fⱼ†} = 0 = {fᵢ, fⱼ}
# {fᵢ, fⱼ†} = δᵢⱼ

@testset "simple fermions" begin
    @test f⁻f⁻() ≈ -permute(f⁻f⁻(), ((2, 1), (4, 3)))
    @test f⁺f⁺() ≈ -permute(f⁺f⁺(), ((2, 1), (4, 3)))

    # the following doesn't hold
    # I don't think I can get all of these to hold simultaneously?
    # @test ff⁺ ≈ -permute(f⁺f, (2, 1), (4, 3))

    @test f⁻f⁺()' ≈ -f⁺f⁻()
    @test f⁻f⁻()' ≈ f⁺f⁺()
    @test (f⁺f⁻() - f⁻f⁺())' ≈ f⁺f⁻() - f⁻f⁺()
    @test (f⁺f⁻() + f⁻f⁺())' ≈ -(f⁻f⁺() + f⁺f⁻())

    @plansor f_number[-1; -2] := f⁺f⁻()[-1 1; 3 2] * τ[3 2; -2 1]
    @test f_number ≈ f_num()

    @test f_hop() ≈ f_plus_f_min() - f_min_f_plus()
end

@testset "Exact Diagonalization" begin
    rng = StableRNG(123)

    L = 2
    t, V, mu = rand(rng, 3)
    pspace = fermion_space()

    H = -t * (f⁺f⁻() - f⁻f⁺()) +
        V * ((n() - 0.5 * id(pspace)) ⊗ (n() - 0.5 * id(pspace))) -
        0.5 * mu * (n() ⊗ id(pspace) + id(pspace) ⊗ n())

    # Values based on https://arxiv.org/abs/1610.05003v1. Half-Chain Entanglement Entropy in the One-Dimensional Spinless Fermion Model
    true_eigenvals = sort([V / 4, V / 4 - mu, -V / 4 - mu / 2 + t, -V / 4 - mu / 2 - t])
    eigenvals = expanded_eigenvalues(H; L)
    @test eigenvals ≈ true_eigenvals
end
