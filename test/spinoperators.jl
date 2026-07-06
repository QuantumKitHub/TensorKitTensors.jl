using TensorKit
using RationalRoots: RationalRoot
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.SpinOperators

@testset "basis transformations" begin
    for spin in (1 // 2):(1 // 2):(5 // 2)
        for symmetry in (Trivial, U1Irrep, SU2Irrep)
            U = basis_transform(symmetry; spin)
            @test U isa AbstractTensorMap
            @test scalartype(U) === Int # exact entries promote without precision loss
            @test U' * U == one(U)
        end
        @test basis_transform(Trivial; spin) == one(basis_transform(Trivial; spin))
    end
    U = basis_transform(Z2Irrep)
    @test U' * U ≈ one(U)
    @test convert(Array, U) ≈ [1 1; 1 -1] / sqrt(2)
    @test_throws ArgumentError basis_transform(Z2Irrep; spin = 1)

    # the Hadamard transformation is exact, and converts to any precision without loss
    @test scalartype(U) === RationalRoot{Int}
    @test abs(BigFloat(U[1, 1]) - 1 / sqrt(big(2))) < eps(BigFloat)
end

@testset "scalar types and precision" begin
    # real scalar types stay real end-to-end
    @test scalartype(S_z(Float64, U1Irrep)) === Float64
    @test scalartype(S_x(Float32, Z2Irrep)) === Float32
    @test scalartype(S_exchange(Float64, SU2Irrep)) === Float64
    @test scalartype(S_y_S_y(Float64)) === Float64
    @test scalartype(S_y_S_y(Float64, Z2Irrep)) === Float64

    # abelian symmetries preserve full precision
    Z = S_z(Complex{BigFloat}, U1Irrep; spin = 1)
    @test all(c -> block(Z, c)[1] == big(c.charge), sectors(spin_space(U1Irrep; spin = 1)))
    X = S_x(Complex{BigFloat}, Z2Irrep)
    @test abs(block(X, Z2Irrep(0))[1] - big(1) / 2) < big(2.0)^-200

    # non-abelian symmetries construct at wide scalar types, with Float64-limited
    # accuracy set by TensorKit's fusion-tensor data
    SS = S_exchange(Complex{BigFloat}, SU2Irrep)
    @test abs(block(SS, SU2Irrep(1))[1] - 1 // 4) < 1.0e-14
end

@testset "Non-symmetric spin $spin operators" for spin in (1 // 2):(1 // 2):(5 // 2)
    # inferrability
    X = @inferred S_x(; spin)
    Y = @inferred S_y(; spin)
    Z = @inferred S_z(; spin)
    S⁺ = @inferred S_plus(; spin)
    S⁻ = @inferred S_min(; spin)
    S⁺S⁻ = @inferred S_plus_S_min(; spin)
    S⁻S⁺ = @inferred S_min_S_plus(; spin)
    XX = @inferred S_x_S_x(; spin)
    YY = @inferred S_y_S_y(; spin)
    ZZ = @inferred S_z_S_z(; spin)
    SS = @inferred S_exchange(; spin)

    # hermiticity, normalization, and su(2) commutation relations
    test_spin_algebra(X, Y, Z; spin)

    # definition of +-
    @test (X + im * Y) ≈ S⁺
    @test (X - im * Y) ≈ S⁻
    @test S⁺' ≈ S⁻

    # composite operators
    @test XX ≈ X ⊗ X
    @test YY ≈ Y ⊗ Y
    @test ZZ ≈ Z ⊗ Z
    @test S⁺S⁻ ≈ S⁺ ⊗ S⁻
    @test S⁻S⁺ ≈ S⁻ ⊗ S⁺
    @test (S⁺S⁻ + S⁻S⁺) / 2 ≈ XX + YY
    @test SS ≈ X ⊗ X + Y ⊗ Y + Z ⊗ Z
    @test SS ≈ Z ⊗ Z + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
end

@testset "Z2-Symmetric spin 1//2 operators" begin
    # inferrability
    X = @inferred S_x(Z2Irrep)
    XX = @inferred S_x_S_x(Z2Irrep)
    YY = @inferred S_y_S_y(Z2Irrep)
    ZZ = @inferred S_z_S_z(Z2Irrep)
    exchange = @inferred S_exchange(Z2Irrep)

    @test_throws ArgumentError S_x(Z2Irrep; spin = 1)
    @test_throws ArgumentError S_x_S_x(Z2Irrep; spin = 1)
    @test_throws ArgumentError S_y_S_y(Z2Irrep; spin = 1)
    @test_throws ArgumentError S_z_S_z(Z2Irrep; spin = 1)
    @test_throws ArgumentError S_exchange(Z2Irrep; spin = 1)

    @test_throws ArgumentError S_plus(Z2Irrep)
    @test_throws ArgumentError S_min(Z2Irrep)
    @test_throws ArgumentError S_plus_S_min(Z2Irrep)
    @test_throws ArgumentError S_min_S_plus(Z2Irrep)
    @test_throws ArgumentError S_y(Z2Irrep)
    @test_throws ArgumentError S_z(Z2Irrep)

    # element-wise comparison against the trivial operators in the dense basis
    U = basis_transform(Z2Irrep)
    test_operator_dense(X, S_x(), U)
    test_operator_dense(XX, S_x_S_x(), U)
    test_operator_dense(YY, S_y_S_y(), U)
    test_operator_dense(ZZ, S_z_S_z(), U)
    test_operator_dense(exchange, S_exchange(), U)
end

@testset "U1-Symmetric spin $spin operators" for spin in (1 // 2):(1 // 2):(5 // 2)
    # inferrability
    Z = @inferred S_z(U1Irrep; spin)
    ZZ = @inferred S_z_S_z(U1Irrep; spin)
    plusmin = @inferred S_plus_S_min(U1Irrep; spin)
    minplus = @inferred S_min_S_plus(U1Irrep; spin)
    exchange = @inferred S_exchange(U1Irrep; spin)

    @test_throws ArgumentError S_x(U1Irrep; spin)
    @test_throws ArgumentError S_y(U1Irrep; spin)
    @test_throws ArgumentError S_plus(U1Irrep; spin)
    @test_throws ArgumentError S_min(U1Irrep; spin)
    @test_throws ArgumentError S_x_S_x(U1Irrep; spin)
    @test_throws ArgumentError S_y_S_y(U1Irrep; spin)

    # element-wise comparison against the trivial operators in the dense basis
    U = basis_transform(U1Irrep; spin)
    for f in (S_z, S_z_S_z, S_plus_S_min, S_min_S_plus, S_exchange)
        test_operator_dense(f(U1Irrep; spin), f(; spin), U)
    end
end

@testset "SU2-Symmetric spin $spin operators" for spin in (1 // 2):(1 // 2):(5 // 2)
    # inferrability
    V = @inferred spin_space(SU2Irrep; spin)
    SS = @inferred S_exchange(SU2Irrep; spin)

    @test_throws ArgumentError S_x(SU2Irrep; spin)
    @test_throws ArgumentError S_y(SU2Irrep; spin)
    @test_throws ArgumentError S_z(SU2Irrep; spin)
    @test_throws ArgumentError S_plus(SU2Irrep; spin)
    @test_throws ArgumentError S_min(SU2Irrep; spin)
    @test_throws ArgumentError S_min_S_plus(SU2Irrep; spin)
    @test_throws ArgumentError S_plus_S_min(SU2Irrep; spin)
    @test_throws ArgumentError S_x_S_x(SU2Irrep; spin)
    @test_throws ArgumentError S_y_S_y(SU2Irrep; spin)
    @test_throws ArgumentError S_z_S_z(SU2Irrep; spin)

    SS_triv = S_exchange(; spin)

    # element-wise comparison against the trivial operator in the dense basis
    test_operator_dense(SS, SS_triv, basis_transform(SU2Irrep; spin))
end

@testset "Exact diagonalisation for $sector symmetry" for sector in [Trivial, U1Irrep]
    spin = 1

    ZZ = @inferred S_z_S_z(sector; spin)
    plusmin = @inferred S_plus_S_min(sector; spin)
    minplus = @inferred S_min_S_plus(sector; spin)
    O = ZZ + 0.5 * (plusmin + minplus)

    true_eigenvals = vcat([-2.0], repeat([-1.0], 3), repeat([1.0], 5))
    eigenvals = expanded_eigenvalues(O; L = 2)
    @test eigenvals ≈ true_eigenvals

    # Value based on https://doi.org/10.1088/0953-8984/2/26/010. Exact diagonalisations of open spin-1 chains
    eigenvals = expanded_eigenvalues(O; L = 4)
    @test eigenvals[2] - eigenvals[1] ≈ 0.50917 atol = 1.0e-6
end
