using TensorKit
using Test
using LinearAlgebra: I
using SUNRepresentations
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.SUNOperators

@testset "sun_space" begin
    V2 = @inferred sun_space(SUNIrrep; irrep = (1, 0))
    @test only(sectors(V2)) == SUNIrrep{2}(1, 0)
    @test dim(V2) == 2

    V3 = @inferred sun_space(SUNIrrep; irrep = (1, 0, 0))
    @test only(sectors(V3)) == SUNIrrep{3}(1, 0, 0)
    @test dim(V3) == 3

    Vt = @inferred sun_space(Trivial; irrep = (1, 0))
    @test Vt == ComplexSpace(2)

    Vt3 = @inferred sun_space(Trivial; irrep = (1, 0, 0))
    @test Vt3 == ComplexSpace(3)
end

@testset "exchange SUNIrrep" begin
    # SU(3) fundamental: known block eigenvalues (−2/3 on the 3̄, 1/3 on the 6)
    ex3 = exchange(Float64, SUNIrrep; irrep = (1, 0, 0))
    for (c, b) in blocks(ex3)
        dl = dynkin_label(c)
        val = b[1, 1]
        if dl == (0, 1)
            @test val ≈ -2 / 3
        elseif dl == (2, 0)
            @test val ≈ 1 / 3
        end
    end

    # fundamental swap identity: swap = 2·exchange + (1/N)·id
    sw3 = swap(Float64, SUNIrrep; irrep = (1, 0, 0))
    V3 = sun_space(SUNIrrep; irrep = (1, 0, 0))
    @test sw3 ≈ 2 * ex3 + (1 // 3) * id(V3 ⊗ V3)
end

@testset "exchange multiplicity (SU(3) adjoint)" begin
    # (2,1,0) ⊗ (2,1,0) contains (2,1,0) with outer multiplicity 2; exchange is a scalar on
    # each coupled irrep, so every block is proportional to the identity
    adj = exchange(Float64, SUNIrrep; irrep = (2, 1, 0))
    for (c, b) in blocks(adj)
        n = size(b, 1)
        @test b ≈ b[1, 1] * I(n)
    end
end

@testset "exchange dense vs symmetric" begin
    for irrep in [(1, 0), (1, 0, 0), (2, 1, 0)]
        ex_sun = exchange(Float64, SUNIrrep; irrep)
        ex_triv = exchange(Float64, Trivial; irrep)
        U = basis_transform(SUNIrrep; irrep)
        test_operator_dense(ex_sun, ex_triv, U)
    end

    # asymmetric coupling of two different SU(3) irreps, with per-leg transforms
    irreps = ((1, 0, 0), (1, 1, 0))
    ex_sun = exchange(Float64, SUNIrrep; irreps)
    ex_triv = exchange(Float64, Trivial; irreps)
    U1 = basis_transform(SUNIrrep; irrep = irreps[1])
    U2 = basis_transform(SUNIrrep; irrep = irreps[2])
    @test space(ex_sun, 1) == sun_space(SUNIrrep; irrep = irreps[1])
    @test space(ex_sun, 2) == sun_space(SUNIrrep; irrep = irreps[2])
    test_operator_dense(ex_sun, ex_triv, (U1, U2))
end

@testset "biquadratic" begin
    for irrep in [(1, 0, 0), (2, 1, 0)]
        ex = exchange(Float64, SUNIrrep; irrep)
        bq = biquadratic(Float64, SUNIrrep; irrep)
        @test bq ≈ ex * ex
        # dense vs symmetric round-trip
        bq_triv = biquadratic(Float64, Trivial; irrep)
        U = basis_transform(SUNIrrep; irrep)
        test_operator_dense(bq, bq_triv, U)
    end
end

@testset "swap dense vs symmetric" begin
    # the literal permutation, for a non-fundamental irrep too
    for irrep in [(1, 0, 0), (2, 1, 0)]
        sw_sun = swap(Float64, SUNIrrep; irrep)
        sw_triv = swap(Float64, Trivial; irrep)
        U = basis_transform(SUNIrrep; irrep)
        test_operator_dense(sw_sun, sw_triv, U)
    end
end

@testset "default arguments and error paths" begin
    # default element type (ComplexF64) and default symmetry (Trivial) forwarding
    @test sun_space(; irrep = (1, 0)) == sun_space(SUNIrrep; irrep = (1, 0))
    @test exchange(; irrep = (1, 0)) ≈ exchange(ComplexF64, Trivial; irrep = (1, 0))
    @test exchange(SUNIrrep; irrep = (1, 0)) ≈ exchange(ComplexF64, SUNIrrep; irrep = (1, 0))
    @test swap(; irrep = (1, 0)) ≈ swap(ComplexF64, Trivial; irrep = (1, 0))
    @test biquadratic(Float64; irrep = (1, 0, 0)) isa AbstractTensorMap

    # the Trivial basis transform is the identity
    U = basis_transform(Trivial; irrep = (1, 0, 0))
    @test U ≈ id(domain(U))

    # invalid inputs
    @test_throws ArgumentError exchange(Float64, SUNIrrep)                              # no irrep
    @test_throws ArgumentError exchange(Float64, SUNIrrep; irreps = ((1, 0), (1, 0, 0)))  # mismatched N
end
