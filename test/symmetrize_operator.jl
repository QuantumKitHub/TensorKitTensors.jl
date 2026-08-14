using TensorKit
using Test
import TensorKitTensors.SpinOperators as SO
import TensorKitTensors.BosonOperators as BO
import TensorKitTensors.FermionOperators as FO
import TensorKitTensors.HubbardOperators as HO
import TensorKitTensors.TJOperators as TJ
import TensorKitTensors.QuantumGates as QG

function reference_operator(A::AbstractArray, V::ElementarySpace)
    sites = ndims(A) ÷ 2
    return TensorMap(A, V^sites ← V^sites)
end

@testset "exports" begin
    for operators in (SO, BO, FO, HO, TJ, QG)
        @test :symmetrize_operator in names(operators)
    end
end

@testset "SpinOperators" begin
    Sz = convert(Array, SO.S_z())
    O = reference_operator(Sz, SO.spin_space(Trivial))
    @test SO.symmetrize_operator(O, U1Irrep) ≈ SO.S_z(U1Irrep)

    Sz_spin1 = SO.S_z(Float64, Trivial; spin = 1)
    @test SO.symmetrize_operator(Sz_spin1, U1Irrep; spin = 1) ≈
        SO.S_z(Float64, U1Irrep; spin = 1)

    # the number of sites and scalar type are inherited from the reference TensorMap
    SS_big = SO.S_exchange(Complex{BigFloat})
    SS_symmetric = SO.symmetrize_operator(SS_big, SU2Irrep)
    @test numout(SS_symmetric) == numin(SS_symmetric) == 2
    @test scalartype(SS_symmetric) === Complex{BigFloat}
    @test SS_symmetric ≈ SO.S_exchange(Complex{BigFloat}, SU2Irrep)

    # tol is used only for symmetry projection
    X_pert = copy(convert(Array, SO.S_x()))
    X_pert[1, 1] += 1.0e-6
    O_pert = reference_operator(X_pert, SO.spin_space(Trivial))
    @test_throws ArgumentError SO.symmetrize_operator(O_pert, Z2Irrep)
    @test SO.symmetrize_operator(O_pert, Z2Irrep; tol = 1.0e-3) ≈
        SO.S_x(Z2Irrep) atol = 1.0e-5

    @test_throws ArgumentError SO.symmetrize_operator(O, FermionParity)
end

@testset "BosonOperators" begin
    cutoff = 3
    N = BO.b_num(Float64, Trivial; cutoff)
    @test BO.symmetrize_operator(N, U1Irrep; cutoff) ≈
        BO.b_num(Float64, U1Irrep; cutoff)

    hopping = BO.b_hopping(Float64, Trivial; cutoff)
    @test BO.symmetrize_operator(hopping, U1Irrep; cutoff) ≈
        BO.b_hopping(Float64, U1Irrep; cutoff)
end

@testset "FermionOperators" begin
    hopping = FO.f_hopping(ComplexF64, Trivial)
    @test FO.symmetrize_operator(hopping, Trivial) ≈ hopping
    @test FO.symmetrize_operator(hopping, U1Irrep) ≈
        FO.f_hopping(ComplexF64, U1Irrep)

    # the reference TensorMap itself enforces the mandatory fermion-parity grading
    parity_odd = ComplexF64[0 1; 0 0]
    V = FO.fermion_space(Trivial)
    @test_throws ArgumentError TensorMap(parity_odd, V ← V)
end

@testset "HubbardOperators" begin
    hopping = HO.e_hopping(ComplexF64, Trivial, Trivial)
    @test HO.symmetrize_operator(hopping, SU2Irrep, SU2Irrep) ≈
        HO.e_hopping(ComplexF64, SU2Irrep, SU2Irrep)

    hopping_real = HO.e_hopping(Float64, Trivial, Trivial)
    @test_throws ArgumentError HO.symmetrize_operator(
        hopping_real, SU2Irrep, SU2Irrep
    )

    # test an operator with more than two sites
    paircor = HO.Δ⁺ij_Δkl(ComplexF64, Trivial, Trivial)
    @test HO.symmetrize_operator(paircor, U1Irrep, SU2Irrep) ≈
        HO.Δ⁺ij_Δkl(ComplexF64, U1Irrep, SU2Irrep)
end

@testset "TJOperators" begin
    hopping = TJ.e_hopping(ComplexF64, Trivial, Trivial)
    hopping_slave = TJ.symmetrize_operator(
        hopping, U1Irrep, SU2Irrep; slave_fermion = true
    )
    generated_slave = TJ.e_hopping(
        ComplexF64, U1Irrep, SU2Irrep; slave_fermion = true
    )
    @test hopping_slave ≈ generated_slave
    @test generated_slave ≈ TJ.transform_slave_fermion(
        TJ.e_hopping(ComplexF64, U1Irrep, SU2Irrep)
    )
end

@testset "QuantumGates" begin
    Z = QG.pauli_z(Trivial)
    @test QG.symmetrize_operator(Z, U1Irrep) ≈ QG.pauli_z(U1Irrep)

    CZ = QG.cz(Trivial)
    @test QG.symmetrize_operator(CZ, U1Irrep) ≈ QG.cz(U1Irrep)

    theta = 0.37
    @test QG.rotation_z(U1Irrep; theta) ≈
        QG.symmetrize_operator(QG.rotation_z(Trivial; theta), U1Irrep)
end
