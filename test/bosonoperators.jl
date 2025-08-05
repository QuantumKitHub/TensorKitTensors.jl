using TensorKit
using LinearAlgebra: tr
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.BosonOperators
using StableRNGs

@testset "Non-symmetric bosonic operators" begin
    cutoff = 4

    # inferrability
    B⁻ = @inferred b⁻(; cutoff)
    B⁺ = @inferred b⁺(; cutoff)
    N = @inferred n(; cutoff)
    B⁻B⁻ = @inferred b⁻b⁻(; cutoff)
    B⁺B⁻ = @inferred b⁺b⁻(; cutoff)
    B⁻B⁺ = @inferred b⁻b⁺(; cutoff)
    B⁺B⁺ = @inferred b⁺b⁺(; cutoff)
    Bhop = @inferred b_hop(; cutoff)
    V = @inferred boson_space(Trivial; cutoff)

    # test adjoints
    @test B⁻' ≈ B⁺
    @test B⁻B⁻' ≈ B⁺B⁺
    @test B⁺B⁻' ≈ B⁻B⁺
    @test N' ≈ N

    # commutation relations are modified because hilbert space has cutoff!
    # [a, a⁺] = 1 except when aplied to `|cutoff>`
    id_modified = id(V)
    id_modified[cutoff + 1, cutoff + 1] = -cutoff
    @test (B⁻ * B⁺ - B⁺ * B⁻) ≈ id_modified

    # definition of N
    @test B⁻' * B⁻ ≈ N

    # definition of Bhop
    @test Bhop ≈ B⁺B⁻ + B⁻B⁺

    # composite operators
    @test B⁻B⁻ ≈ B⁻ ⊗ B⁻
    @test B⁺B⁻ ≈ B⁺ ⊗ B⁻
    @test B⁻B⁺ ≈ B⁻ ⊗ B⁺
    @test B⁺B⁺ ≈ B⁺ ⊗ B⁺
end

@testset "U1-symmetric bosonic operators" begin
    cutoff = 4

    rng = StableRNG(123)
    # inferrability
    N = @inferred n(U1Irrep; cutoff)
    B⁺B⁻ = @inferred b⁺b⁻(U1Irrep; cutoff)
    B⁻B⁺ = @inferred b⁻b⁺(U1Irrep; cutoff)
    V = @inferred boson_space(U1Irrep; cutoff)

    # non-symmetric operators throw error
    @test_throws ArgumentError b⁻(U1Irrep; cutoff)
    @test_throws ArgumentError b⁺(U1Irrep; cutoff)

    @test_throws ArgumentError b_plus_b_plus(U1Irrep; cutoff)
    @test_throws ArgumentError b_min_b_min(U1Irrep; cutoff)

    L = 4
    b_pm, b_mp, b_n = rand(rng, 3)
    O_u1 = (N ⊗ id(V) + id(V) ⊗ N) * b_n + B⁻B⁺ * b_mp + B⁺B⁻ * b_pm

    O_triv = (
        n(; cutoff) ⊗ id(boson_space(Trivial; cutoff)) +
            id(boson_space(Trivial; cutoff)) ⊗ n(; cutoff)
    ) * b_n +
        b⁺b⁻(; cutoff) * b_pm + b⁻b⁺(; cutoff) * b_mp

    test_operator(O_u1, O_triv; L)
end

@testset "Exact Diagonalization" begin
    cutoff = 1
    L = 2
    for symmetry in [Trivial U1Irrep]
        rng = StableRNG(123)
        # inferrability
        N = @inferred n(U1Irrep; cutoff)
        B⁺B⁻ = @inferred b⁺b⁻(U1Irrep; cutoff)
        B⁻B⁺ = @inferred b⁻b⁺(U1Irrep; cutoff)
        V = @inferred boson_space(U1Irrep; cutoff)

        b_pm, b_mp, b_n = rand(rng, 3)
        O = (N ⊗ id(V) + id(V) ⊗ N) * b_n + B⁻B⁺ * b_mp + B⁺B⁻ * b_pm

        true_eigenvals = sort(
            [0, 2 * b_n, b_n + sqrt(b_mp * b_pm), b_n - sqrt(b_mp * b_pm)]
        )
        eigenvals = expanded_eigenvalues(O; L)
        @test eigenvals ≈ true_eigenvals
    end
end
