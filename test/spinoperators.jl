using TensorKit
using LinearAlgebra: tr, I, eigvals
using Test
using TensorKitTensors.SpinOperators

ε = zeros(Int, 3, 3, 3)
for i in 1:3
    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
end

@testset "Non-symmetric spin $spin operators" for spin in (1 // 2):(1 // 2):(5 // 2)
    # inferrability
    X = @inferred S_x(; spin)
    Y = @inferred S_y(; spin)
    Z = @inferred S_z(; spin)
    S⁺ = @inferred S_plus(; spin)
    S⁻ = @inferred S_min(; spin)
    S⁺⁻ = @inferred S_plusmin(; spin)
    S⁻⁺ = @inferred S_minplus(; spin)
    XX = @inferred S_xx(; spin)
    YY = @inferred S_yy(; spin)
    ZZ = @inferred S_zz(; spin)
    SS = @inferred S_exchange(; spin)
    Svec = [X Y Z]

    # operators should be hermitian
    for s in Svec
        @test s' ≈ s
    end

    # operators should be normalized
    @test sum(tr(Svec[i]^2) for i in 1:3) / (2spin + 1) ≈ spin * (spin + 1)

    # commutation relations
    for i in 1:3, j in 1:3
        @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
              sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
    end

    # definition of +-
    @test (X + im * Y) ≈ S⁺
    @test (X - im * Y) ≈ S⁻
    @test S⁺' ≈ S⁻

    # composite operators
    @test XX ≈ X ⊗ X
    @test YY ≈ Y ⊗ Y
    @test ZZ ≈ Z ⊗ Z
    @test S⁺⁻ ≈ S⁺ ⊗ S⁻
    @test S⁻⁺ ≈ S⁻ ⊗ S⁺
    @test (S⁺⁻ + S⁻⁺) / 2 ≈ XX + YY
    @test SS ≈ X ⊗ X + Y ⊗ Y + Z ⊗ Z
    @test SS ≈ Z ⊗ Z + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
end

function operator_sum(O::AbstractTensorMap; L::Int=4)
    I = id(space(O, 1))
    n = numin(O)
    return sum(1:(L - n + 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - n)), i, O))
    end
end

function test_operator(O1::AbstractTensorMap, O2::AbstractTensorMap; L::Int=4, atol=1e-8)
    H1 = operator_sum(O1; L)
    H2 = operator_sum(O2; L)
    eigenvals1 = mapreduce(vcat, eigvals(H1)) do (c, vals)
        return repeat(vals, dim(c))
    end
    eigenvals2 = mapreduce(vcat, eigvals(H2)) do (c, vals)
        return repeat(vals, dim(c))
    end
    @test isapprox(sort!(eigenvals1; by=real), sort!(eigenvals2; by=real); atol)
end

@testset "Z2-Symmetric spin 1//2 operators" begin
    # inferrability
    X = @inferred S_x(Z2Irrep)
    XX = @inferred S_xx(Z2Irrep)
    ZZ = @inferred S_zz(Z2Irrep)

    @test_throws ArgumentError S_x(Z2Irrep; spin=1)
    @test_throws ArgumentError S_xx(Z2Irrep; spin=1)
    @test_throws ArgumentError S_zz(Z2Irrep; spin=1)

    @test_throws ArgumentError S_plus(Z2Irrep)
    @test_throws ArgumentError S_min(Z2Irrep)
    @test_throws ArgumentError S_plusmin(Z2Irrep)
    @test_throws ArgumentError S_minplus(Z2Irrep)
    @test_throws ArgumentError S_y(Z2Irrep)
    @test_throws ArgumentError S_yy(Z2Irrep)
    @test_throws ArgumentError S_z(Z2Irrep)

    @test_broken S_exchange(Z2Irrep)

    L = 4
    a_x, a_xx, a_zz = rand(3)
    O_z2 = (X ⊗ id(domain(X)) + id(domain(X)) ⊗ X) * a_x + XX * a_xx + ZZ * a_zz

    O_triv = (S_x() ⊗ id(domain(S_x())) + id(domain(S_x())) ⊗ S_x()) * a_x +
             S_xx() * a_xx + S_zz() * a_zz

    test_operator(O_z2, O_triv; L)
end

@testset "U1-Symmetric spin $spin operators" for spin in (1 // 2):(1 // 2):(5 // 2)
    # inferrability
    Z = @inferred S_z(U1Irrep; spin)
    ZZ = @inferred S_zz(U1Irrep; spin)
    plusmin = @inferred S_plusmin(U1Irrep; spin)
    minplus = @inferred S_minplus(U1Irrep; spin)
    exchange = @inferred S_exchange(U1Irrep; spin)

    @test_throws ArgumentError S_x(U1Irrep; spin)
    @test_throws ArgumentError S_y(U1Irrep; spin)
    @test_throws ArgumentError S_plus(U1Irrep; spin)
    @test_throws ArgumentError S_min(U1Irrep; spin)
    @test_throws ArgumentError S_xx(U1Irrep; spin)
    @test_throws ArgumentError S_yy(U1Irrep; spin)

    L = 4
    for f in (S_z, S_zz, S_plusmin, S_minplus, S_exchange)
        test_operator(f(U1Irrep; spin), f(; spin); L)
    end

    a_z, a_zz, a_plusmin, a_minplus, a_exchange = rand(5)
    O_u1 = (Z ⊗ id(domain(Z)) + id(domain(Z)) ⊗ Z) * a_z +
           ZZ * a_zz +
           plusmin * a_plusmin +
           minplus * a_minplus +
           exchange * a_exchange
    O_triv = (S_z(; spin) ⊗ id(domain(S_z(; spin))) +
              id(domain(S_z(; spin))) ⊗ S_z(; spin)) * a_z +
             S_zz(; spin) * a_zz +
             S_plusmin(; spin) * a_plusmin +
             S_minplus(; spin) * a_minplus +
             S_exchange(; spin) * a_exchange
    test_operator(O_u1, O_triv; L)
end
