using TensorKit
using Test
using LinearAlgebra: I
using SUNRepresentations
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.SUNOperators

@testset "sun_space" begin
    V2 = @inferred sun_space(SUNIrrep; irrep = (1, 0))
    @test V2 == Vect[SUNIrrep{2}](SUNIrrep(1, 0) => 1)
    @test dim(V2) == 2

    V3 = @inferred sun_space(SUNIrrep; irrep = (1, 0, 0))
    @test dim(V3) == 3

    Vt = @inferred sun_space(Trivial; irrep = (1, 0))
    @test Vt == ComplexSpace(2)

    Vt3 = @inferred sun_space(Trivial; irrep = (1, 0, 0))
    @test Vt3 == ComplexSpace(3)
end

@testset "exchange SUNIrrep" begin
    # SU(3) fundamental: check known block eigenvalues
    ex3 = @inferred exchange(Float64, SUNIrrep; irrep = (1, 0, 0))
    for (c, b) in blocks(ex3)
        dl = dynkin_label(c)
        val = b[1, 1]
        if dl == (0, 1)
            @test val ≈ -2 / 3
        elseif dl == (2, 0)
            @test val ≈ 1 / 3
        end
    end

    # SU(3) swap identity: swap = 2*exchange + (1/N)*id
    sw3 = @inferred swap(Float64, SUNIrrep; irrep = (1, 0, 0))
    V3 = sun_space(SUNIrrep; irrep = (1, 0, 0))
    @test sw3 ≈ 2 * ex3 + (1 // 3) * id(V3 ⊗ V3)
end

@testset "exchange multiplicity (SU(3) adjoint)" begin
    # (2,1,0) ⊗ (2,1,0) contains sector (2,1,0) with outer multiplicity 2
    adj = @inferred exchange(Float64, SUNIrrep; irrep = (2, 1, 0))
    for (c, b) in blocks(adj)
        # blocks are proportional to identity within each sector
        n = size(b, 1)
        val = b[1, 1]
        @test b ≈ val * I(n)
    end
end

@testset "twosite_casimir SUNIrrep" begin
    # twosite_casimir and exchange satisfy: exchange = (twosite_casimir(2) - c2_1 - c2_2) / 2
    dl = (1, 0, 0)
    sun_irrep = SUNIrrep(dl...)
    c2 = SUNRepresentations.casimir(2, sun_irrep)
    ex = exchange(Float64, SUNIrrep; irrep = dl)
    tc3 = twosite_casimir(Float64, SUNIrrep; k = 2, irrep = dl)
    V3 = sun_space(SUNIrrep; irrep = dl)
    @test ex ≈ (tc3 - 2 * c2 * id(V3 ⊗ V3)) / 2
end

@testset "exchange Trivial vs SUNIrrep" begin
    # Trivial and SUNIrrep exchange should have the same eigenvalues
    for dl in [(1, 0), (1, 0, 0)]
        ex_sun = exchange(Float64, SUNIrrep; irrep = dl)
        ex_triv = exchange(Float64, Trivial; irrep = dl)
        test_operator(ex_sun, ex_triv)
    end
end

@testset "asymmetric exchange" begin
    # Two different SU(3) irreps
    ex_asym = @inferred exchange(
        Float64, SUNIrrep; irreps = ((1, 0, 0), (1, 1, 0))
    )
    # Check it is well-formed (correct codomain/domain)
    V1 = sun_space(SUNIrrep; irrep = (1, 0, 0))
    V2 = sun_space(SUNIrrep; irrep = (1, 1, 0))
    @test space(ex_asym, 1) == V1
    @test space(ex_asym, 2) == V2
end
