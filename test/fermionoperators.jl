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

const symmetries = (Trivial, U1Irrep)

@testset "basis transformations" begin
    for sym in symmetries
        U = basis_transform(sym)
        @test U isa AbstractTensorMap
        @test scalartype(U) === Int
        @test U == one(U)
    end
    # real and wide scalar types are preserved
    @test scalartype(f_num(Float64, U1Irrep)) === Float64
    @test scalartype(f_hopping(Complex{BigFloat}, U1Irrep)) === Complex{BigFloat}
end

@testset "fermion properties" begin
    @test f⁻f⁻() ≈ -swap_2sites(f⁻f⁻())
    @test f⁺f⁺() ≈ -swap_2sites(f⁺f⁺())

    # the following doesn't hold
    # I don't think I can get all of these to hold simultaneously?
    # @test ff⁺ ≈ -swap_2sites(f⁺f)

    @test f⁻f⁻()' ≈ -f⁺f⁺()
    for sym in symmetries
        @test f⁻f⁺(sym)' ≈ -f⁺f⁻(sym)
        @test (f⁺f⁻(sym) - f⁻f⁺(sym))' ≈ f⁺f⁻(sym) - f⁻f⁺(sym)
        @test (f⁺f⁻(sym) + f⁻f⁺(sym))' ≈ -(f⁻f⁺(sym) + f⁺f⁻(sym))

        @plansor f_number[-1; -2] := f⁺f⁻(sym)[-1 1; 3 2] * τ[3 2; -2 1]
        @test f_number ≈ f_num(sym)

        @test f_hop(sym) ≈ f_plus_f_min(sym) - f_min_f_plus(sym)
    end

    @test_throws ArgumentError f⁻f⁻(U1Irrep)
    @test_throws ArgumentError f⁺f⁺(U1Irrep)
end

@testset "Exact Diagonalization" begin
    rng = StableRNG(123)

    L = 2
    t, V, mu = rand(rng, 3)
    # Values based on https://arxiv.org/abs/1610.05003v1. Half-Chain Entanglement Entropy in the One-Dimensional Spinless Fermion Model
    true_eigenvals = sort([V / 4, V / 4 - mu, -V / 4 - mu / 2 + t, -V / 4 - mu / 2 - t])

    for sym in symmetries
        pspace = fermion_space(sym)
        H = -t * (f⁺f⁻(sym) - f⁻f⁺(sym)) +
            V * ((n(sym) - 0.5 * id(pspace)) ⊗ (n(sym) - 0.5 * id(pspace))) -
            0.5 * mu * (n(sym) ⊗ id(pspace) + id(pspace) ⊗ n(sym))
        eigenvals = expanded_eigenvalues(H; L)
        @test eigenvals ≈ true_eigenvals
    end
end
