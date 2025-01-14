using TensorKit
using LinearAlgebra: tr
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.BosonOperators

const cutoff = 4

@testset "Non-symmetric bosonic operators" begin
    # inferrability
    A = @inferred a(; cutoff)
    A⁺ = @inferred a⁺(; cutoff)
    N = @inferred n(; cutoff)
    AA = @inferred aa(; cutoff)
    A⁺A = @inferred a⁺a(; cutoff)
    AA⁺ = @inferred aa⁺(; cutoff)
    A⁺A⁺ = @inferred a⁺a⁺(; cutoff)
    V = @inferred boson_space(Trivial; cutoff)

    # test adjoints
    @test A' ≈ A⁺
    @test AA' ≈ A⁺A⁺
    @test A⁺A' ≈ AA⁺
    @test N' ≈ N

    # commutation relations are modified because hilbert space has cutoff!
    # [a, a⁺] = 1 except when aplied to `|cutoff>`
    id_modified = id(V)
    id_modified[cutoff + 1, cutoff + 1] = -cutoff
    @test (A * A⁺ - A⁺ * A) ≈ id_modified

    # definition of N
    @test A' * A ≈ N

    # composite operators
    @test AA ≈ A ⊗ A
    @test A⁺A ≈ A⁺ ⊗ A
    @test AA⁺ ≈ A ⊗ A⁺
    @test A⁺A⁺ ≈ A⁺ ⊗ A⁺
end

@testset "U1-symmetric bosonic operators" begin
    # inferrability
    N = @inferred n(U1Irrep; cutoff)
    A⁺A = @inferred a⁺a(U1Irrep; cutoff)
    AA⁺ = @inferred aa⁺(U1Irrep; cutoff)
    V = @inferred boson_space(U1Irrep; cutoff)

    # non-symmetric operators throw error
    @test_throws ArgumentError a(U1Irrep; cutoff)
    @test_throws ArgumentError a⁺(U1Irrep; cutoff)

    @test_throws ArgumentError a_plusplus(U1Irrep; cutoff)
    @test_throws ArgumentError a_minmin(U1Irrep; cutoff)

    L = 4
    a_pm, a_mp, a_n = rand(3)
    O_u1 = (N ⊗ id(V) + id(V) ⊗ N) * a_n + AA⁺ * a_mp + A⁺A * a_pm

    O_triv = (n(; cutoff) ⊗ id(boson_space(Trivial; cutoff)) +
              id(boson_space(Trivial; cutoff)) ⊗ n(; cutoff)) * a_n +
             a⁺a(; cutoff) * a_pm + aa⁺(; cutoff) * a_mp

    test_operator(O_u1, O_triv; L)
end
