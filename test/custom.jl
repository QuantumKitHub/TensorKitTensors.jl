using TensorKit
using Test
import TensorKitTensors
using TensorKitTensors: desymmetrize
import TensorKitTensors.SpinOperators as SO
import TensorKitTensors.BosonOperators as BO
import TensorKitTensors.FermionOperators as FO
import TensorKitTensors.HubbardOperators as HO
import TensorKitTensors.TJOperators as TJ
import TensorKitTensors.QuantumGates as QG

@testset "SpinOperators" begin
    Sz = convert(Array, SO.S_z())
    @test SO.custom(Sz, U1Irrep) ≈ SO.S_z(U1Irrep)

    Sz_spin1 = SO.S_z(Float64, Trivial; spin = 1)
    @test SO.custom(convert(Array, Sz_spin1), U1Irrep; spin = 1) ≈
        SO.S_z(Float64, U1Irrep; spin = 1)

    # rank determines the number of sites and the scalar type is preserved
    SS_big = SO.S_exchange(Complex{BigFloat})
    SS_custom = SO.custom(convert(Array, SS_big), SU2Irrep)
    @test numout(SS_custom) == numin(SS_custom) == 2
    @test scalartype(SS_custom) === Complex{BigFloat}
    @test SS_custom ≈ SO.S_exchange(Complex{BigFloat}, SU2Irrep)

    # tol is used only for symmetry projection
    X_pert = copy(convert(Array, SO.S_x()))
    X_pert[1, 1] += 1.0e-6
    @test_throws ArgumentError SO.custom(X_pert, Z2Irrep)
    @test SO.custom(X_pert, Z2Irrep; tol = 1.0e-3) ≈
        SO.S_x(Z2Irrep) atol = 1.0e-5

    # common dense-array validation
    @test_throws ArgumentError SO.custom(fill("x", 2, 2), Trivial)
    @test_throws ArgumentError SO.custom(zeros(2), Trivial)
    @test_throws ArgumentError SO.custom(fill(1.0), Trivial)
    @test_throws ArgumentError SO.custom(zeros(2, 3), Trivial)
    @test_throws ArgumentError SO.custom(zeros(4, 4), Trivial)
    @test_throws ArgumentError SO.custom(Sz, FermionParity)
end

@testset "BosonOperators" begin
    cutoff = 3
    N = BO.b_num(Float64, Trivial; cutoff)
    @test BO.custom(convert(Array, N), U1Irrep; cutoff) ≈
        BO.b_num(Float64, U1Irrep; cutoff)

    hopping = BO.b_hopping(Float64, Trivial; cutoff)
    @test BO.custom(convert(Array, hopping), U1Irrep; cutoff) ≈
        BO.b_hopping(Float64, U1Irrep; cutoff)
end

@testset "FermionOperators" begin
    hopping = FO.f_hopping(ComplexF64, Trivial)
    A = convert(Array, desymmetrize(hopping))
    @test FO.custom(A, Trivial) ≈ hopping
    @test FO.custom(A, U1Irrep) ≈
        FO.f_hopping(ComplexF64, U1Irrep)

    # `Trivial` still enforces the mandatory fermion-parity grading
    parity_odd = ComplexF64[0 1; 0 0]
    @test_throws ArgumentError FO.custom(parity_odd, Trivial)
end

@testset "HubbardOperators" begin
    hopping = HO.e_hopping(ComplexF64, Trivial, Trivial)
    A = convert(Array, desymmetrize(hopping))
    @test HO.custom(A, SU2Irrep, SU2Irrep) ≈
        HO.e_hopping(ComplexF64, SU2Irrep, SU2Irrep)

    hopping_real = HO.e_hopping(Float64, Trivial, Trivial)
    @test_throws ArgumentError HO.custom(
        convert(Array, desymmetrize(hopping_real)), SU2Irrep, SU2Irrep
    )

    # test an operator with more than two-sites
    paircor = HO.Δ⁺ij_Δkl(ComplexF64, Trivial, Trivial)
    A = convert(Array, desymmetrize(paircor))
    @test HO.custom(A, U1Irrep, SU2Irrep) ≈
        HO.Δ⁺ij_Δkl(ComplexF64, U1Irrep, SU2Irrep)
end

@testset "TJOperators" begin
    hopping = TJ.e_hopping(ComplexF64, Trivial, Trivial)
    A = convert(Array, desymmetrize(hopping))
    @test TJ.custom(A, U1Irrep, SU2Irrep; slave_fermion = true) ≈
        TJ.e_hopping(ComplexF64, U1Irrep, SU2Irrep; slave_fermion = true)
end

@testset "QuantumGates" begin
    Z = convert(Array, QG.pauli_z())
    @test QG.custom(Z, U1Irrep) ≈ QG.pauli_z(U1Irrep)

    CZ = convert(Array, QG.cz())
    @test QG.custom(CZ, U1Irrep) ≈ QG.cz(U1Irrep)
end
