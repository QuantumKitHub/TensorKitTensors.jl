using TensorKit
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.BosonOperators
using StableRNGs

@testset "basis transformations" begin
    cutoff = 4
    # the symmetry argument defaults to `Trivial`, as in the other modules
    @test boson_space(; cutoff) == boson_space(Trivial; cutoff)
    for symmetry in (Trivial, U1Irrep)
        U = basis_transform(symmetry; cutoff)
        @test U isa AbstractTensorMap{Int}
        @test U == one(U)
    end
    # real and wide scalar types are preserved
    @test scalartype(b_num(Float64, U1Irrep; cutoff)) === Float64
    N_big = b_num(BigFloat, U1Irrep; cutoff)
    @test scalartype(N_big) === BigFloat
    @test all(c -> block(N_big, c)[1] == big(c.charge), sectors(boson_space(U1Irrep; cutoff)))
end

@testset "type inference" begin
    cutoff = 2

    @test (@testinferred b_num(; cutoff)) isa AbstractTensorMap
    @test (@testinferred b_num(Float64; cutoff)) isa AbstractTensorMap
    @test (@testinferred b_num(U1Irrep; cutoff)) isa AbstractTensorMap
    @test (@testinferred b_num(Float64, U1Irrep; cutoff)) isa AbstractTensorMap
    @test (@testinferred b_hopping(U1Irrep; cutoff)) isa AbstractTensorMap
    @test (@testinferred b_hopping(Float64, U1Irrep; cutoff)) isa AbstractTensorMap
end

@testset "Non-symmetric bosonic operators" begin
    cutoff = 4

    # inferrability
    B⁻ = @testinferred b⁻(; cutoff)
    B⁺ = @testinferred b⁺(; cutoff)
    N = @testinferred n(; cutoff)
    B⁻B⁻ = @testinferred b⁻b⁻(; cutoff)
    B⁺B⁻ = @testinferred b⁺b⁻(; cutoff)
    B⁻B⁺ = @testinferred b⁻b⁺(; cutoff)
    B⁺B⁺ = @testinferred b⁺b⁺(; cutoff)
    Bhop = @testinferred b_hop(; cutoff)
    V = @testinferred boson_space(Trivial; cutoff)

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

    # inferrability
    N = @testinferred n(U1Irrep; cutoff)
    B⁺B⁻ = @testinferred b⁺b⁻(U1Irrep; cutoff)
    B⁻B⁺ = @testinferred b⁻b⁺(U1Irrep; cutoff)
    V = @testinferred boson_space(U1Irrep; cutoff)

    # non-symmetric operators throw error
    @test_throws ArgumentError b⁻(U1Irrep; cutoff)
    @test_throws ArgumentError b⁺(U1Irrep; cutoff)

    @test_throws ArgumentError b_plus_b_plus(U1Irrep; cutoff)
    @test_throws ArgumentError b_min_b_min(U1Irrep; cutoff)

    # element-wise comparison against the trivial operators in the dense basis
    U = basis_transform(U1Irrep; cutoff)
    test_operator_dense(N, n(; cutoff), U)
    test_operator_dense(B⁺B⁻, b⁺b⁻(; cutoff), U)
    test_operator_dense(B⁻B⁺, b⁻b⁺(; cutoff), U)
end

@testset "Exact Diagonalization" begin
    cutoff = 1
    for symmetry in (Trivial, U1Irrep)
        rng = StableRNG(123)
        # inferrability
        N = @testinferred n(symmetry; cutoff)
        B⁺B⁻ = @testinferred b⁺b⁻(symmetry; cutoff)
        B⁻B⁺ = @testinferred b⁻b⁺(symmetry; cutoff)
        V = @testinferred boson_space(symmetry; cutoff)

        b_pm, b_mp, b_n = rand(rng, 3)
        O = (N ⊗ id(V) + id(V) ⊗ N) * b_n + B⁻B⁺ * b_mp + B⁺B⁻ * b_pm

        true_eigenvals = sort(
            [0, 2 * b_n, b_n + sqrt(b_mp * b_pm), b_n - sqrt(b_mp * b_pm)]
        )
        eigenvals = expanded_eigenvalues(O)
        @test eigenvals ≈ true_eigenvals
    end
end
